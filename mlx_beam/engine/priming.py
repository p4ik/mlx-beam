"""The prefill that feeds the draft head along: the trunk runs without the
`lm_head` over each prompt chunk (its logits were computed and discarded
before), the post-norm hidden states go to the proposer as pairs (hidden at
position t, token at t + 1) - the head's history over the prompt, which
`propose` continues from. One head call per chunk and row, no `lm_head`.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import PromptProcessingBatch, _cache_arrays
from mlx_beam.engine.proposer import trunk


class PrimingPromptBatch(PromptProcessingBatch):
    """PromptProcessingBatch whose chunks prime the proposer bound to the
    generation batch class (`SpeculativeGenerationBatch.speculator`)."""

    def __init__(self, model, *args, **kwargs):
        self._inner, _, _ = trunk(model)
        super().__init__(model, *args, **kwargs)

    @property
    def _proposer(self):
        spec = getattr(self.generation_batch, "speculator", None)
        return None if spec is None else spec.proposer

    def prompt(self, tokens):
        proposer = self._proposer
        lengths = {len(t) for t in tokens}
        if proposer is None or not tokens or len(lengths) != 1:
            # Rows of unequal length need padding; the engine never sends
            # those (it cuts every prefill call to the shortest segment).
            return super().prompt(tokens)
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        starts = [len(t) for t in self.tokens]
        for sti, ti in zip(self.tokens, tokens, strict=True):
            sti += ti
        rows = tokens
        tokens = mx.array(tokens)
        done = 0
        while tokens.shape[1] > 0:
            n = min(self.prefill_step_size, tokens.shape[1])
            hidden = self._inner(tokens[:, :n], cache=self.prompt_cache)
            mx.eval(hidden, *_cache_arrays(self.prompt_cache))
            for i, uid in enumerate(self.uids):
                proposer.prime(
                    uid, hidden[i : i + 1], rows[i][done : done + n], starts[i] + done
                )
            mx.clear_cache()
            tokens = tokens[:, n:]
            done += n

    def generate(self, tokens):
        proposer = self._proposer
        if proposer is not None:
            # The last prompt token is fed by the generation batch's first
            # step; its pair with the hidden state before it closes the
            # prompt's history.
            for uid, t in zip(self.uids, tokens, strict=True):
                proposer.follow(uid, None, mx.array([t[-1]]))
        return super().generate(tokens)
