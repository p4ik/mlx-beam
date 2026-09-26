"""The verify path: a generation batch that decodes several tokens per model
call when a proposer drafts them.

One cycle for a single row: the proposer drafts k tokens after the token the
target just sampled; the target runs one forward over those k + 1 tokens; the
longest prefix of drafts the target's own argmax reproduces is committed,
plus the target's token after it; everything the forward wrote past that
point is rolled back - attention caches trim, recurrent layers redo their
recurrence over the accepted prefix from what the layer stashed. Every
committed token is the target's own: its argmax over the verify forward for
a greedy row, its own draw for a sampled one. A sampled row draws by
Gumbel-max under a key per position (`SeededSampler`); the proposer drafts
under the same keys, and a draft is accepted when it equals the target's
draw - no residual distribution, no draft probabilities: the noise does not
depend on the logits, so the committed token at every position has exactly
the target's distribution whatever was drafted (measured against
Leviathan/Chen under the production sampler: within 0.005, 2026-09-19).
Against plain decoding, one token per forward, the block verify agrees up
to the rounding of kernels that run at a different width (on Metal the
quantized matmul from two rows on, the multi-query attention; the warm-up
width check in /health says whether this machine rounds the same). The two
exact modes close that gap: `positions` runs one forward per token and is
exact by construction, `kernels` runs the block through per-row kernels and
keeps them only where its own check finds them bit-equal (exact.py).

Rows speculate one at a time (measured: a batch of four gains under 1.2x on
this hardware without a small-M kernel, 2026-09-19). Logits processors run
per verify position over the context that position would have seen in a
plain step; a thinking budget only while its state cannot change within the
cycle (`ThinkingBudget.stable_for`) - while it forces a close the row
decodes plainly through the same class, so no request is refused for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import GenerationBatch, _cache_arrays
from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache, BatchRotatingKVCache
from mlx_beam._vendor.mlx_lm.models.gated_delta import gated_delta_update
from mlx_beam._vendor.mlx_lm.models.ssm import ssm_update
from mlx_beam.engine.exact import exact_forward
from mlx_beam.engine.proposer import Proposer, trunk
from mlx_beam.engine.regulator import Regulator


class RollbackUnsupported(RuntimeError):
    """A cache layer the verify path cannot take back."""


class Speculator:
    """What the engine hands the batch: the proposer, the depth, who may
    speculate, and the counters `/health` reports."""

    def __init__(
        self,
        proposer: Proposer,
        depth: int,
        eligible: Callable[[int, int], bool],
        coupling: Callable[[int], Any] = lambda uid: None,
    ):
        self.proposer = proposer
        # The cap; the regulator picks the depth of each cycle below it.
        self.depth = depth
        self.regulator = Regulator(depth)
        # eligible(uid, tokens_this_cycle): no logits processor whose state
        # could change within the cycle.
        self.eligible = eligible
        # coupling(uid): the row's SeededSampler when it samples - the verify
        # draws every position through it and the proposer drafts through
        # it - or None for a greedy row (argmax on both sides).
        self.coupling = coupling
        # "block": one forward over the k+1 tokens (the fast path);
        # "kernels": the same block through the exact projections and
        # per-query attention (mlx_beam.engine.exact), checked at warm-up
        # against single forwards and dropped to "block" when not bit-equal;
        # "positions": one forward per token, stopping at the first
        # rejected draft - exact by construction, plain decoding's cost, the
        # reference the other two are measured against.
        self.mode = "block"
        # The warm-up's width check: the block forward against single
        # forwards from the same state, logits compared bit for bit.
        self.exact: dict | None = None
        self.cycles = 0
        self.plain_steps = 0
        self.drafted = 0
        self.accepted = 0

    def describe(self) -> dict:
        reg = self.regulator.describe()
        return {
            "proposer": self.proposer.describe(),
            "depth": reg["depth"],
            "max_depth": self.depth,
            "mode": self.mode,
            "exact": self.exact,
            "cycles": self.cycles,
            "plain_steps": self.plain_steps,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "acceptance": (
                round(self.accepted / self.drafted, 3) if self.drafted else None
            ),
            "tokens_per_cycle": (
                round(1 + self.accepted / self.cycles, 2) if self.cycles else None
            ),
            "parked": reg["parked"],
            "reason": reg["reason"],
            "regulator": reg,
        }


def rollback_window(cache, before: tuple, keep: int, total: int) -> None:
    """Put a sliding-window layer back to `keep` of the `total` tokens the
    verify wrote. The verify goes through the concat path, which leaves the
    buffer in temporal order with the cycle's tokens last: the kept ones are
    sliced out, the state before the cycle is restored, and they are written
    again - the same buffers a plain decode of just those tokens would have
    built. A rotated ring cannot simply trim: its stale tail would stay."""
    n = cache.keys.shape[2]
    kept_keys = cache.keys[..., n - total : n - total + keep, :]
    kept_values = cache.values[..., n - total : n - total + keep, :]
    cache.state = before
    if keep:  # a cycle always keeps y; a direct caller may keep nothing
        cache.update_and_fetch(mx.contiguous(kept_keys), mx.contiguous(kept_values))


def rollback_recurrent(cache: ArraysCache, keep: int, total: int) -> None:
    """Put a recurrent layer back to the state after `keep` of the `total`
    tokens the last forward ran, from what the layer stashed: the conv window
    is a slice of the conv input, the delta-rule state is the recurrence over
    the kept prefix. Exact with respect to that forward: the kernel advances
    one token at a time, so the state equals the forward's own state after
    `keep` tokens (bit for bit, measured 2026-09-24). A separate forward of
    just those tokens can differ by its projections' rounding at another
    width; see the module docstring."""
    stash = cache.stash
    if not stash:
        raise RollbackUnsupported(
            f"{type(cache).__name__} recorded nothing to roll back from; the "
            "model's recurrent layer does not support the speculative verify"
        )
    n_keep = stash["n_keep"]
    cache[0] = mx.contiguous(stash["conv_input"][:, keep : keep + n_keep, :])
    mask = stash["mask"]
    kind = stash.get("kind", "gated_delta")
    if kind == "gated_delta":
        _, state = gated_delta_update(
            stash["q"][:, :keep],
            stash["k"][:, :keep],
            stash["v"][:, :keep],
            stash["a"][:, :keep],
            stash["b"][:, :keep],
            stash["A_log"],
            stash["dt_bias"],
            stash["state"],
            None if mask is None else mask[:, :keep],
            use_kernel=stash["use_kernel"],
        )
    elif kind == "mamba2":
        # Mamba-2 (Granite 4, Nemotron-H): the same replay over the
        # selective-scan update the layer ran, on the kept prefix.
        _, state = ssm_update(
            stash["hidden"][:, :keep],
            stash["A_log"],
            stash["B"][:, :keep],
            stash["C"][:, :keep],
            stash["D"],
            stash["dt"][:, :keep],
            stash["dt_bias"],
            stash["state"],
            stash["time_step_limit"],
            None if mask is None else mask[:, :keep],
        )
    else:
        raise RollbackUnsupported(f"recurrent stash of kind {kind!r} has no replay")
    cache[1] = state
    cache.advance(-(total - keep))


def width_check(
    model, policy, prompt: list[int], width: int, exact: bool = False
) -> dict:
    """Does a forward over `width` tokens give the same logits as `width`
    forwards of one token from the same state? On Metal the answer is no
    for the quantized matmul from two rows on (measured 2026-09-25); this
    is the evidence /health carries for the block verify, and what the
    kernel path (`exact=True`) is held to. Fresh caches under the policy,
    the warm-up prompt prefilled, then the last `width` tokens of it again
    as the probe."""
    import copy

    from mlx_beam.engine.kv import make_request_cache

    probe = (prompt * width)[-width:]
    cache_block = make_request_cache(model, policy)
    mx.eval(model(mx.array([prompt]), cache=cache_block))
    cache_single = copy.deepcopy(cache_block)
    if exact:
        with exact_forward():
            block = model(mx.array([probe]), cache=cache_block)
    else:
        block = model(mx.array([probe]), cache=cache_block)
    singles = mx.concatenate(
        [model(mx.array([[t]]), cache=cache_single) for t in probe], axis=1
    )
    mx.eval(block, singles)
    diff = mx.abs(block.astype(mx.float32) - singles.astype(mx.float32)).max()
    return {
        "width": width,
        "path": "kernels" if exact else "block",
        "block_equals_positions": bool(mx.array_equal(block, singles).item()),
        "max_abs_logit_diff": float(diff.item()),
        "argmax_equal": bool(
            mx.array_equal(
                mx.argmax(block, axis=-1), mx.argmax(singles, axis=-1)
            ).item()
        ),
    }


class SpeculativeGenerationBatch(GenerationBatch):
    """GenerationBatch with the verify cycle. `speculator` is bound per
    engine on a subclass (see `bound`); None decodes plainly everywhere."""

    speculator: Speculator | None = None
    # The row's rope delta (GenerationRequest.rope_delta), 0 for text; the
    # decode's positions continue where a prompt with images left them.
    delta_of = staticmethod(lambda uid: 0)

    @classmethod
    def bound(cls, speculator: Speculator | None, delta_of=lambda uid: 0) -> type:
        return type(
            "BoundSpeculativeGenerationBatch",
            (cls,),
            {"speculator": speculator, "delta_of": staticmethod(delta_of)},
        )

    def __init__(self, model, *args, **kwargs):
        self._inner, self._lm_head, _ = trunk(model)
        # The post-norm hidden state of every row at the position that
        # produced `_next_tokens`; what the proposer drafts from.
        self._hidden_slot: mx.array | None = None
        super().__init__(model, *args, **kwargs)

    # -- the plain step, with the hidden state kept ---------------------------

    def _forward(self, inputs: mx.array):
        """(post-norm hidden, logits): the model's two halves called one after
        the other, the head with the model's own logit post-processing
        (`trunk`, held to the model's forward at warm-up by `check_head`)."""
        kwargs = {}
        deltas = [self.delta_of(uid) for uid in self.uids]
        if any(deltas):
            kwargs["rope_offset"] = mx.array(deltas, dtype=mx.int32)
        hidden = self._inner(inputs, cache=self.prompt_cache, **kwargs)
        return hidden, self._lm_head(hidden)

    def _step(self):
        self._current_tokens = self._next_tokens
        self._current_logprobs = self._next_logprobs
        inputs = self._current_tokens
        hidden, logits = self._forward(inputs[:, None])
        logits = logits[:, -1, :]
        token_context = []
        if any(self.logits_processors):
            processed_logits = []
            for e in range(len(self.uids)):
                sample_logits = logits[e : e + 1]
                processors = self.logits_processors[e] or ()
                if processors:
                    context = self._token_context[e].update_and_fetch(inputs[e : e + 1])
                    token_context.append(context)
                    for processor in processors:
                        sample_logits = processor(context, sample_logits)
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        samplers = [
            self.fallback_sampler if s is None else s
            for s in (self.samplers or [None] * len(self.uids))
        ]
        if not samplers or all(s is samplers[0] for s in samplers):
            sampled = (samplers[0] if samplers else self.fallback_sampler)(logprobs)
        else:
            sampled = mx.concatenate(
                [s(logprobs[e : e + 1]) for e, s in enumerate(samplers)], axis=0
            )
        self._next_tokens = sampled
        self._next_logprobs = list(logprobs)
        before = self._hidden_slot
        self._hidden_slot = hidden[:, -1:, :]
        mx.async_eval(
            self._next_tokens, self._next_logprobs, self._hidden_slot, token_context
        )
        mx.eval(inputs, self._current_logprobs)
        mx.eval(*_cache_arrays(self.prompt_cache))
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs, strict=True):
            sti.append(ti)
        if self.speculator is not None:
            self.speculator.plain_steps += 1
            if before is not None:
                # The pair this step went through, so the proposer's
                # history stays in step: the state before it, the token fed.
                for e, uid in enumerate(self.uids):
                    self.speculator.proposer.follow(
                        uid, before[e : e + 1], self._current_tokens[e : e + 1]
                    )
        return inputs, self._current_logprobs

    # -- batch bookkeeping -----------------------------------------------------

    def extend(self, batch):
        mine, theirs = self._hidden_slot, getattr(batch, "_hidden_slot", None)
        had_rows = bool(self.uids)
        super().extend(batch)
        if not had_rows:
            self._hidden_slot = theirs
        elif mine is not None and theirs is not None:
            self._hidden_slot = mx.concatenate([mine, theirs])
        else:
            self._hidden_slot = None  # the next plain step fills it again

    def filter(self, keep):
        if self.speculator is not None and self._hidden_slot is not None:
            # A row that leaves has fed its last token; the hidden state
            # after it is the open end of its history, for the store.
            for e, uid in enumerate(self.uids):
                if e not in keep and self.tokens[e]:
                    self.speculator.proposer.prime(
                        uid,
                        self._hidden_slot[e : e + 1],
                        [self.tokens[e][-1]],
                        len(self.tokens[e]) - 1,
                    )
        super().filter(keep)
        if self._hidden_slot is not None:
            self._hidden_slot = self._hidden_slot[keep] if keep else None

    # -- the cycle -------------------------------------------------------------

    def _alone(self) -> bool:
        spec = self.speculator
        return (
            spec is not None and len(self.uids) == 1 and self._hidden_slot is not None
        )

    def _plain(self, measure: bool):
        """One plain step; alone, its wall time is the regulator's depth 0."""
        tic = time.perf_counter()
        out = super().next()
        if measure:
            self.speculator.regulator.observe_plain(time.perf_counter() - tic)
        return out

    def next(self):
        if not self._alone():
            return super().next()
        spec = self.speculator
        k = spec.regulator.choose()
        if k == 0:
            return self._plain(measure=True)
        if not spec.eligible(self.uids[0], k + 1):
            spec.regulator.skipped()
            return self._plain(measure=True)
        tic = time.perf_counter()
        accepted, drafted = spec.accepted, spec.drafted
        self._compared = True
        responses = self._cycle(k)
        if responses is None:
            spec.regulator.skipped()
            return self._plain(measure=True)
        # What the proposer actually drafted, not the depth asked for.
        spec.regulator.observe_cycle(
            spec.drafted - drafted,
            spec.accepted - accepted,
            time.perf_counter() - tic,
            rejected=self._compared,
        )
        return responses

    def _arm(self) -> list[ArraysCache]:
        """Prepare every cache for a possible rollback: recurrent layers
        stash what the forward computes, sliding windows keep the state
        they had before the cycle (a rotated ring cannot trim - its stale
        tail would stay in the buffer), the rest must trim."""
        recurrent = []
        self._window_states = {}
        for i, c in enumerate(self.prompt_cache):
            if isinstance(c, ArraysCache):
                c.stash = {}
                recurrent.append(c)
            elif isinstance(c, BatchRotatingKVCache):
                # Views, not references: an in-place write copies first.
                self._window_states[i] = tuple(
                    v[:] if isinstance(v, mx.array) else v for v in c.state
                )
            elif not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                raise RollbackUnsupported(
                    f"{type(c).__name__} cannot trim; the speculative verify "
                    "cannot take a rejected draft back from it"
                )
        return recurrent

    def _processed(self, logits: mx.array, fed: list[int]) -> mx.array:
        """The row's logits processors over the cycle's positions, each with
        the context a plain step would have handed it: the tokens before
        the cycle plus `fed` up to and including the token that produced
        the position. Normalised to logprobs. The processors are stateless
        here - a budget in a state that could change is kept out of the
        cycle by `eligible`, a bias or penalty reads only the context."""
        processors = self.logits_processors[0] if self.logits_processors else ()
        if processors:
            base = self._token_context[0].tokens
            rows = []
            n = logits.shape[1]
            for j in range(n):
                # The positions are the last n of `fed`: the block verify
                # hands all of them, the per-position one the latest only.
                upto = len(fed) - n + j + 1
                context = mx.concatenate([base, mx.array(fed[:upto], dtype=mx.int32)])
                row = logits[:, j, :]
                for processor in processors:
                    row = processor(context, row)
                rows.append(row)
            logits = mx.stack(rows, axis=1)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _choose(self, uid: int, logprobs: mx.array, offset: int) -> mx.array:
        """The target's token at verify position `offset` from these (1, V)
        logprobs: argmax for a greedy row, the row's own draw at the
        position it will be for a sampled one."""
        coupling = self.speculator.coupling(uid)
        if coupling is None:
            return mx.argmax(logprobs, axis=-1)
        return coupling.draw(logprobs, coupling.position + offset)

    def _chooser(self, uid: int):
        """What the proposer drafts with: None (argmax) for a greedy row, the
        row's draw under the key of the position each draft is for."""
        coupling = self.speculator.coupling(uid)
        if coupling is None:
            return None

        def choose(logits: mx.array, i: int) -> mx.array:
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            return coupling.draw(logprobs, coupling.position + i)

        return choose

    def _cycle_by_position(self, uid, y, lp_y, drafts, k):
        """The verify one token at a time: feed y, read the model's token;
        feed the draft only while it is that token. Nothing rejected ever
        enters a cache, so nothing rolls back - the same forwards plain
        decoding runs, and the same bytes (the check mode). The row's
        limits are checked on each token before the next is fed: a token
        past the stop or the length limit never enters the caches either,
        so the entry the store gets is exactly what was emitted."""
        spec = self.speculator
        drafts_l = drafts.tolist()
        tokens = [int(y.item()), *drafts_l]
        hiddens, logprob_rows, chosen, responses = [], [], [], []
        m, finish = 0, None
        for j, token in enumerate(tokens):
            hidden, logits = self._forward(mx.array([[token]], dtype=y.dtype))
            logprobs = self._processed(logits[:, -1:, :], tokens[: j + 1])
            nxt = self._choose(uid, logprobs[:, -1, :], j)
            hiddens.append(hidden)
            logprob_rows.append(logprobs[0, -1])
            chosen.append(nxt)
            finish = self._limit(token)
            responses.append((token, lp_y if j == 0 else logprob_rows[j - 1], finish))
            if finish is not None:
                # Ended by the row's limit: draft j was never compared.
                self._compared = j >= k
                break
            if j < k and int(nxt.item()) == drafts_l[j]:
                m += 1
            else:
                self._compared = j < k
                break
        spec.cycles += 1
        spec.drafted += k
        spec.accepted += m
        hidden = mx.concatenate(hiddens, axis=1)  # (1, m + 1, H)
        sampled = mx.concatenate(chosen)[None]
        return self._finish_cycle(
            uid, responses, finish, sampled, logprob_rows, hidden, m, drafts_l
        )

    def _limit(self, token: int) -> str | None:
        """One emitted token against the row's limits: the count and the
        stop matcher advance, the finish reason comes back if it ends here."""
        finish = None
        self._num_tokens[0] += 1
        if self._num_tokens[0] >= self.max_tokens[0]:
            finish = "length"
        if self._matchers[0].advance(token):
            finish = "stop"
        return finish

    def _emit(self, emitted):
        """Cut the cycle's tokens short at the row's own limits."""
        responses, finish = [], None
        for token, lp in emitted:
            finish = self._limit(token)
            responses.append((token, lp, finish))
            if finish is not None:
                break
        return responses, finish

    def _cycle(self, depth: int):
        spec = self.speculator
        uid = self.uids[0]
        y = self._next_tokens  # (1,), sampled, not yet in the caches
        lp_y = self._next_logprobs[0]
        h = self._hidden_slot
        drafts = spec.proposer.propose(uid, h, y, self._chooser(uid), depth)
        k = int(drafts.shape[0]) if drafts.ndim else 0
        if k == 0:
            return None
        inputs = mx.concatenate([y, drafts])[None]  # (1, 1 + k)
        if spec.mode == "positions":
            return self._cycle_by_position(uid, y, lp_y, drafts, k)
        recurrent = self._arm()
        try:
            if spec.mode == "kernels":
                with exact_forward():
                    hidden, logits = self._forward(inputs)
            else:
                hidden, logits = self._forward(inputs)
            drafts_l = drafts.tolist()
            logprobs = self._processed(logits, [int(y.item()), *drafts_l])
            sampled = mx.concatenate(
                [self._choose(uid, logprobs[:, j, :], j) for j in range(1 + k)]
            )[
                None
            ]  # (1, 1 + k)
            mx.eval(sampled)
            sampled_l = sampled[0].tolist()
            m = 0
            while m < k and sampled_l[m] == drafts_l[m]:
                m += 1
            # All k held: no position was refused; else draft m was.
            self._compared = m < k
            spec.cycles += 1
            spec.drafted += k
            spec.accepted += m
            # Tokens the caller sees this cycle: y and the accepted drafts, in
            # order, each cut short by the row's own limits.
            emitted: list[tuple[int, mx.array]] = [(int(y.item()), lp_y)]
            emitted += [(drafts_l[i], logprobs[0, i]) for i in range(m)]
            responses, finish = self._emit(emitted)
            e = len(responses)
            total = 1 + k
            if e < total:
                for i, c in enumerate(self.prompt_cache):
                    if isinstance(c, ArraysCache):
                        rollback_recurrent(c, e, total)
                    elif i in self._window_states:
                        rollback_window(c, self._window_states[i], e, total)
                    else:
                        c.trim(total - e)
        finally:
            for c in recurrent:
                c.stash = None
            self._window_states = {}
        return self._finish_cycle(
            uid,
            responses,
            finish,
            sampled,
            [logprobs[0, i] for i in range(m + 1)],
            hidden,
            m,
            drafts_l,
        )

    def _finish_cycle(
        self, uid, responses, finish, sampled, logprob_rows, hidden, m, drafts_l
    ):
        """Book the cycle: committed tokens into the row, the token after the
        accepted prefix as the next cycle's start, the proposer told what
        held. `logprob_rows[i]` is the row at position i, i in 0..m."""
        spec = self.speculator
        committed = [t for t, _, _ in responses]
        self.tokens[0].extend(committed)
        coupling = spec.coupling(uid)
        if coupling is not None:
            # Positions 0..m were drawn: the accepted drafts' successors and
            # the token after them, the next cycle's start.
            coupling.advance(m + 1)
        if self.logits_processors and self.logits_processors[0]:
            # An inert budget still watches the context for its opener.
            self._token_context[0].update_and_fetch(mx.array(committed, dtype=mx.int32))
        if finish is None:
            # The target's token after the accepted prefix is the next
            # cycle's start; the hidden state that produced it drafts from.
            self._next_tokens = sampled[:, m]
            self._next_logprobs = [logprob_rows[m]]
            self._hidden_slot = hidden[:, m : m + 1, :]
            mx.async_eval(self._next_tokens, self._next_logprobs, self._hidden_slot)
            spec.proposer.commit(uid, hidden[:, :m, :], drafts_l[:m])
        else:
            # The row ends here: its history closes on the drafts that were
            # committed (the cut may have dropped some), the engine takes
            # the snapshot for the store before it drops the row.
            kept = len(committed) - 1
            spec.proposer.commit(uid, hidden[:, :kept, :], drafts_l[:kept])
            self._hidden_slot = hidden[:, kept : kept + 1, :]
        mx.eval(*_cache_arrays(self.prompt_cache))
        out = []
        for token, lp, fin in responses:
            out.append(
                self.Response(
                    uid=uid,
                    token=token,
                    logprobs=lp,
                    finish_reason=fin,
                    prompt_cache=self.extract_cache(0) if fin else None,
                    all_tokens=self.tokens[0] if fin else None,
                )
            )
        if finish is not None:
            self.filter([])
        return out
