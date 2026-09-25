"""The verify path: a generation batch that decodes several tokens per model
call when a proposer drafts them.

One cycle for a single row: the proposer drafts k tokens after the token the
target just sampled; the target runs one forward over those k + 1 tokens; the
longest prefix of drafts the target's own argmax reproduces is committed,
plus the target's token after it; everything the forward wrote past that
point is rolled back - attention caches trim, recurrent layers redo their
recurrence over the accepted prefix from what the layer stashed. Greedy only:
every committed token is the target's own argmax over the verify forward.
Against plain decoding, one token per forward, that agrees up to the
rounding of kernels that run at a different width: in bf16 the two were
bit-identical in every measured case (recurrent layers, Metal, head dims 32
and 128, 2026-09-24), in float32 the projections of a four-token forward
differ from a one-token forward by ~1e-8, so a logit tie can fall the other
way. Nothing here can close that gap: the inputs of every layer already
come from the wider forward below it.

Rows speculate one at a time (measured: a batch of four gains under 1.2x on
this hardware without a small-M kernel, 2026-09-19), greedy, without logits
processors other than a logit bias (applied to every verify position) and
an inert thinking budget; everything else decodes plainly through the same
class, so no request is refused for it.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import GenerationBatch, _cache_arrays
from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache, BatchRotatingKVCache
from mlx_beam._vendor.mlx_lm.models.gated_delta import gated_delta_update
from mlx_beam._vendor.mlx_lm.models.ssm import ssm_update
from mlx_beam._vendor.mlx_lm.sample_utils import greedy_sampler
from mlx_beam.engine.exact import exact_forward
from mlx_beam.engine.proposer import Proposer, trunk


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
        logit_bias: Callable[
            [int], tuple[mx.array, mx.array] | None
        ] = lambda uid: None,
    ):
        self.proposer = proposer
        self.depth = depth
        # eligible(uid, tokens_this_cycle): greedy, no logits processor that
        # could act within the cycle.
        self.eligible = eligible
        # logit_bias(uid): (indices, values) of the row's additive bias, the
        # one processor a verify can apply itself - it has no state and no
        # context, so every position gets the same add as a plain step does.
        self.logit_bias = logit_bias
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
        return {
            "proposer": self.proposer.describe(),
            "depth": self.depth,
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
            "parked": False,
            "reason": None,
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
    if keep:
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

    @classmethod
    def bound(cls, speculator: Speculator) -> type:
        return type(
            "BoundSpeculativeGenerationBatch", (cls,), {"speculator": speculator}
        )

    def __init__(self, model, *args, **kwargs):
        self._inner, self._lm_head, _ = trunk(model)
        # The post-norm hidden state of every row at the position that
        # produced `_next_tokens`; what the proposer drafts from.
        self._hidden_slot: mx.array | None = None
        super().__init__(model, *args, **kwargs)

    # -- the plain step, with the hidden state kept ---------------------------

    def _forward(self, inputs: mx.array):
        """(post-norm hidden, logits): the model's own two halves, called
        one after the other - the same ops as the model's __call__."""
        hidden = self._inner(inputs, cache=self.prompt_cache)
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
        self._hidden_slot = hidden[:, -1:, :]
        mx.async_eval(self._next_tokens, self._next_logprobs, token_context)
        mx.eval(inputs, self._current_logprobs)
        mx.eval(*_cache_arrays(self.prompt_cache))
        inputs = inputs.tolist()
        for sti, ti in zip(self.tokens, inputs, strict=True):
            sti.append(ti)
        if self.speculator is not None:
            self.speculator.plain_steps += 1
            for uid in self.uids:
                self.speculator.proposer.drop(uid)
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
        super().filter(keep)
        if self._hidden_slot is not None:
            self._hidden_slot = self._hidden_slot[keep] if keep else None

    # -- the cycle -------------------------------------------------------------

    def _greedy(self, i: int) -> bool:
        s = self.samplers[i] if self.samplers else None
        s = self.fallback_sampler if s is None else s
        return s is greedy_sampler

    def _can_speculate(self) -> bool:
        spec = self.speculator
        return (
            spec is not None
            and len(self.uids) == 1
            and self._hidden_slot is not None
            and self._greedy(0)
            and spec.eligible(self.uids[0], spec.depth + 1)
        )

    def next(self):
        if not self._can_speculate():
            return super().next()
        responses = self._cycle()
        if responses is None:
            return super().next()
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

    def _logprobs(self, uid: int, logits: mx.array) -> mx.array:
        bias = self.speculator.logit_bias(uid)
        if bias is not None:
            # The same scatter-add mlx-lm's processor does on one row.
            indices, values = bias
            logits = logits.at[:, :, indices].add(values)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def _cycle_by_position(self, uid, y, lp_y, drafts, k):
        """The verify one token at a time: feed y, read the model's token;
        feed the draft only while it is that token. Nothing rejected ever
        enters a cache, so nothing rolls back - the same forwards plain
        decoding runs, and the same bytes (the check mode)."""
        spec = self.speculator
        drafts_l = drafts.tolist()
        tokens = [int(y.item()), *drafts_l]
        hiddens, logprob_rows = [], []
        m = 0
        for j, token in enumerate(tokens):
            hidden, logits = self._forward(mx.array([[token]], dtype=y.dtype))
            logprobs = self._logprobs(uid, logits)
            nxt = int(mx.argmax(logprobs[0, -1]).item())
            hiddens.append(hidden)
            logprob_rows.append(logprobs[0, -1])
            if j < k and nxt == drafts_l[j]:
                m += 1
            else:
                break
        spec.cycles += 1
        spec.drafted += k
        spec.accepted += m
        hidden = mx.concatenate(hiddens, axis=1)  # (1, m + 1, H)
        emitted = [(tokens[0], lp_y)] + [
            (drafts_l[i], logprob_rows[i]) for i in range(m)
        ]
        responses, finish = self._emit(emitted)
        sampled = mx.array([[int(mx.argmax(r).item()) for r in logprob_rows]])
        return self._finish_cycle(
            uid, responses, finish, sampled, logprob_rows, hidden, m, drafts_l
        )

    def _emit(self, emitted):
        """Cut the cycle's tokens short at the row's own limits."""
        responses, finish = [], None
        for token, lp in emitted:
            self._num_tokens[0] += 1
            if self._num_tokens[0] >= self.max_tokens[0]:
                finish = "length"
            if self._matchers[0].advance(token):
                finish = "stop"
            responses.append((token, lp, finish))
            if finish is not None:
                break
        return responses, finish

    def _cycle(self):
        spec = self.speculator
        uid = self.uids[0]
        y = self._next_tokens  # (1,), sampled, not yet in the caches
        lp_y = self._next_logprobs[0]
        h = self._hidden_slot
        drafts = spec.proposer.propose(uid, h, y)
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
            logprobs = self._logprobs(uid, logits)
            sampled = mx.argmax(logprobs, axis=-1)  # (1, 1 + k)
            mx.eval(sampled, drafts)
            sampled_l, drafts_l = sampled[0].tolist(), drafts.tolist()
            m = 0
            while m < k and sampled_l[m] == drafts_l[m]:
                m += 1
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
            spec.proposer.drop(uid)
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
