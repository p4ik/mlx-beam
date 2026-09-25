"""The prefill that runs the trunk without the `lm_head` over each prompt
chunk (its logits were computed and discarded before) and uses the hidden
states: the post-norm hidden states go to the proposer as pairs (hidden at
position t, token at t + 1) - the head's history over the prompt, which
`propose` continues from, one head call per chunk and row - and rows with
images get their placeholder positions embedded from the image's features
instead of the vocabulary, with what a frontend adds at those positions
after the first layers (DeepStack) applied through the trunk's layer hook.
"""

from __future__ import annotations

import inspect

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.generate import PromptProcessingBatch, _cache_arrays
from mlx_beam.engine.proposer import trunk


def _inner_takes(inner, name: str) -> bool:
    try:
        return name in inspect.signature(inner.__call__).parameters
    except (TypeError, ValueError):
        return False


class PrimingPromptBatch(PromptProcessingBatch):
    """PromptProcessingBatch whose chunks prime the proposer bound to the
    generation batch class (`SpeculativeGenerationBatch.speculator`) and
    embed image spans. `spans_of(uid)` is bound per engine (see `bound`):
    the request's ImageSpans by absolute prompt position."""

    spans_of = staticmethod(lambda uid: ())

    @classmethod
    def bound(cls, spans_of) -> type:
        return type(
            "BoundPrimingPromptBatch", (cls,), {"spans_of": staticmethod(spans_of)}
        )

    def __init__(self, model, *args, **kwargs):
        self._inner, _, self._embed = trunk(model)
        self._embeds_kw = _inner_takes(self._inner, "input_embeddings")
        self._hook_kw = _inner_takes(self._inner, "layer_hook")
        super().__init__(model, *args, **kwargs)

    @property
    def _proposer(self):
        spec = getattr(self.generation_batch, "speculator", None)
        return None if spec is None else spec.proposer

    def _overlaps(self, starts: list[int], done: int, n: int) -> list[tuple]:
        """(row, chunk offset a, chunk offset b, span, span offset) for every
        image span that reaches into the chunk [done, done + n) of its row."""
        out = []
        for i, uid in enumerate(self.uids):
            lo = starts[i] + done
            hi = lo + n
            for span in self.spans_of(uid):
                a, b = max(span.start, lo), min(span.end, hi)
                if a < b:
                    out.append((i, a - lo, b - lo, span, a - span.start))
        return out

    def _embedded(self, chunk: mx.array, overlaps: list[tuple]) -> mx.array:
        """The chunk's embeddings with every image position taken from its
        span's features."""
        h = self._embed(chunk)
        for i, a, b, span, off in overlaps:
            h[i, a:b, :] = span.features[off : off + (b - a)].astype(h.dtype)
        return h

    def _hook(self, overlaps: list[tuple]):
        """What DeepStack adds after layer i at the image positions: the
        span's i-th extra features, for the layers that have them."""
        depth = max((len(span.deepstack) for _, _, _, span, _ in overlaps), default=0)
        if depth == 0:
            return None

        def hook(i: int, h: mx.array) -> mx.array:
            for row, a, b, span, off in overlaps:
                if i < len(span.deepstack):
                    extra = span.deepstack[i][off : off + (b - a)].astype(h.dtype)
                    h[row, a:b, :] = h[row, a:b, :] + extra
            return h

        return hook

    def prompt(self, tokens):
        proposer = self._proposer
        lengths = {len(t) for t in tokens}
        if not tokens or len(lengths) != 1:
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
            chunk = tokens[:, :n]
            overlaps = self._overlaps(starts, done, n)
            kwargs = {}
            if overlaps:
                if not self._embeds_kw:
                    raise ValueError(
                        f"{type(self.model).__name__} takes no input embeddings; "
                        "it cannot serve images"
                    )
                kwargs["input_embeddings"] = self._embedded(chunk, overlaps)
                hook = self._hook(overlaps)
                if hook is not None:
                    if not self._hook_kw:
                        raise ValueError(
                            f"{type(self.model).__name__} has no layer hook; the "
                            "frontend's per-layer image features cannot be applied"
                        )
                    kwargs["layer_hook"] = hook
            hidden = self._inner(chunk, cache=self.prompt_cache, **kwargs)
            mx.eval(hidden, *_cache_arrays(self.prompt_cache))
            if proposer is not None:
                for i, uid in enumerate(self.uids):
                    proposer.prime(
                        uid,
                        hidden[i : i + 1],
                        rows[i][done : done + n],
                        starts[i] + done,
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
