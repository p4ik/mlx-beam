"""The verify path: a generation batch that decodes several tokens per model
call when a proposer drafts them.

One cycle for a single row: the proposer drafts k tokens after the token the
target just sampled; the target runs one forward over those k + 1 tokens; the
longest prefix of drafts the target's own argmax reproduces is committed,
plus the target's token after it; everything the forward wrote past that
point is rolled back - attention caches trim, recurrent layers redo their
recurrence over the accepted prefix from what the layer stashed. Greedy only:
the committed tokens are exactly what plain decoding would have produced.

Rows speculate one at a time (measured: a batch of four gains under 1.2x on
this hardware without a small-M kernel, 2026-09-19), greedy, without logits
processors other than an inert thinking budget; everything else decodes
plainly through the same class, so no request is refused for it.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import GenerationBatch, _cache_arrays
from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache
from mlx_beam._vendor.mlx_lm.models.gated_delta import gated_delta_update
from mlx_beam._vendor.mlx_lm.sample_utils import greedy_sampler
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
    ):
        self.proposer = proposer
        self.depth = depth
        # eligible(uid, tokens_this_cycle): greedy, no logits processor that
        # could act within the cycle.
        self.eligible = eligible
        self.cycles = 0
        self.plain_steps = 0
        self.drafted = 0
        self.accepted = 0
        self.checked = False

    def describe(self) -> dict:
        return {
            "proposer": self.proposer.describe(),
            "depth": self.depth,
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


def rollback_recurrent(cache: ArraysCache, keep: int, total: int) -> None:
    """Put a recurrent layer back to the state after `keep` of the `total`
    tokens the last forward ran, from what the layer stashed: the conv window
    is a slice of the conv input, the delta-rule state is the recurrence over
    the kept prefix - the same kernel, so the state is what a forward of just
    those tokens would have left."""
    stash = cache.stash
    if not stash:
        raise RollbackUnsupported(
            f"{type(cache).__name__} recorded nothing to roll back from; the "
            "model's recurrent layer does not support the speculative verify"
        )
    n_keep = stash["n_keep"]
    cache[0] = mx.contiguous(stash["conv_input"][:, keep : keep + n_keep, :])
    mask = stash["mask"]
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
    cache[1] = state
    cache.advance(-(total - keep))


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
        recurrent = []
        for c in self.prompt_cache:
            if isinstance(c, ArraysCache):
                c.stash = {}
                recurrent.append(c)
            elif not (hasattr(c, "is_trimmable") and c.is_trimmable()):
                raise RollbackUnsupported(
                    f"{type(c).__name__} cannot trim; the speculative verify "
                    "cannot take a rejected draft back from it"
                )
        return recurrent

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
        recurrent = self._arm()
        try:
            hidden, logits = self._forward(inputs)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
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
            responses, e, finish = [], 0, None
            for token, lp in emitted:
                e += 1
                self._num_tokens[0] += 1
                if self._num_tokens[0] >= self.max_tokens[0]:
                    finish = "length"
                if self._matchers[0].advance(token):
                    finish = "stop"
                responses.append((token, lp, finish))
                if finish is not None:
                    break
            total = 1 + k
            if e < total:
                for c in self.prompt_cache:
                    if isinstance(c, ArraysCache):
                        rollback_recurrent(c, e, total)
                    else:
                        c.trim(total - e)
        finally:
            for c in recurrent:
                c.stash = None
        committed = [t for t, _, _ in responses]
        self.tokens[0].extend(committed)
        if self.logits_processors and self.logits_processors[0]:
            # An inert budget still watches the context for its opener.
            self._token_context[0].update_and_fetch(mx.array(committed, dtype=mx.int32))
        if finish is None:
            # The target's token after the accepted prefix is the next
            # cycle's start; the hidden state that produced it drafts from.
            self._next_tokens = sampled[:, m]
            self._next_logprobs = [logprobs[0, m]]
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
