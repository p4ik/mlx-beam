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

from mlx_beam._vendor.mlx_lm.generate import (
    PromptProcessingBatch,
    _cache_arrays,
    _right_pad_prompts,
)
from mlx_beam.engine.proposer import trunk


def _inner_takes(inner, name: str) -> bool:
    try:
        return name in inspect.signature(inner.__call__).parameters
    except (TypeError, ValueError):
        return False


# What a frontend may need of the trunk -> the keyword the trunk takes for it.
IMAGE_NEEDS = {
    "input_embeddings": "input_embeddings",
    "layer_hook": "layer_hook",
    "position_ids": "position_ids",
    "block_mask": "block_ids",
}


def image_capabilities(model) -> dict[str, bool]:
    """What the model's text trunk takes of what an image frontend hands
    the prefill: `input_embeddings` (features in place of the placeholder
    tokens' embeddings, every frontend), `layer_hook` (per-layer extras
    ahead of certain layers, DeepStack), `position_ids` (the prompt's
    multimodal positions, Qwen's MRoPE) and `block_mask` (an image's
    tokens attending both ways within their block, Gemma 4's encoder-free
    models - the trunk takes `block_ids`). Read from the signature, not
    the config: a text model that lacks one cannot serve that frontend,
    and the engine refuses the request before the worker sees it."""
    try:
        inner, _, _ = trunk(model)
    except ValueError:
        return {name: False for name in IMAGE_NEEDS}
    return {name: _inner_takes(inner, kw) for name, kw in IMAGE_NEEDS.items()}


class PrimingPromptBatch(PromptProcessingBatch):
    """PromptProcessingBatch whose chunks prime the proposer bound to the
    generation batch class (`SpeculativeGenerationBatch.speculator`) and
    embed image spans. `spans_of(uid)` is bound per engine (see `bound`):
    the request's ImageSpans by absolute prompt position."""

    spans_of = staticmethod(lambda uid: ())
    # The request's positions (3, L), or None: see GenerationRequest.positions.
    positions_of = staticmethod(lambda uid: None)

    @classmethod
    def bound(cls, spans_of, positions_of=lambda uid: None) -> type:
        return type(
            "BoundPrimingPromptBatch",
            (cls,),
            {
                "spans_of": staticmethod(spans_of),
                "positions_of": staticmethod(positions_of),
            },
        )

    def __init__(self, model, *args, **kwargs):
        self._inner, _, embed = trunk(model)
        # What the model's input_embeddings stand in for: its own embedding
        # of the ids, normed when the model norms them first (Muse).
        self._embed = getattr(self._inner, "embed_inputs", None) or embed
        self._embeds_kw = _inner_takes(self._inner, "input_embeddings")
        self._hook_kw = _inner_takes(self._inner, "layer_hook")
        self._positions_kw = _inner_takes(self._inner, "position_ids")
        self._blocks_kw = _inner_takes(self._inner, "block_ids")
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

    def _positions(self, starts: list[int], done: int, n: int):
        """The chunk's positions (3, B, n) when a row carries multimodal
        positions, every other row at its indices; None when no row does."""
        given = [self.positions_of(uid) for uid in self.uids]
        if all(g is None for g in given):
            return None
        rows = []
        for i, g in enumerate(given):
            lo = starts[i] + done
            if g is None:
                rows.append(
                    mx.tile(mx.arange(lo, lo + n, dtype=mx.int32)[None], (3, 1))
                )
            else:
                rows.append(g[:, lo : lo + n].astype(mx.int32))
        return mx.stack(rows, axis=1)

    def width_for_blocks(self, width: int) -> int:
        """For the scheduler: the width of its next call, grown so that no
        row's block span is cut by it (the span's end at the latest, from
        where the row stands)."""
        return self._whole_blocks([len(t) for t in self.tokens], 0, width)

    def _whole_blocks(self, starts: list[int], done: int, n: int) -> int:
        """The chunk length that cuts no block span: a span whose tokens
        attend to each other both ways has to be prefilled in one call
        (its keys must all be there for its queries), so a chunk reaching
        into one grows to the span's end."""
        grown = n
        for i, uid in enumerate(self.uids):
            lo = starts[i] + done
            for span in self.spans_of(uid):
                if span.block and span.start < lo + grown < span.end:
                    grown = span.end - lo
        return grown

    def _block_ids(self, overlaps: list[tuple], n: int):
        """(B, n) block ids for the chunk, -1 for text; None without a
        block span in it. A block is numbered by its span's start, which
        no two spans of a row share."""
        if not any(span.block for _, _, _, span, _ in overlaps):
            return None
        ids = mx.full((len(self.uids), n), -1, dtype=mx.int32)
        for i, a, b, span, _ in overlaps:
            if span.block:
                ids[i, a:b] = span.start
        return ids

    def _hook(self, overlaps: list[tuple]):
        """What DeepStack adds at the image positions ahead of a layer: the
        span's extra features for that layer, where it has them."""
        if not any(span.extras for _, _, _, span, _ in overlaps):
            return None

        def hook(i: int, h: mx.array) -> mx.array:
            for row, a, b, span, off in overlaps:
                extra = span.extras.get(i)
                if extra is not None:
                    h[row, a:b, :] = h[row, a:b, :] + extra[off : off + (b - a)].astype(
                        h.dtype
                    )
            return h

        return hook

    def prompt(self, tokens):
        proposer = self._proposer
        if len(self.uids) != len(tokens):
            raise ValueError("The batch length doesn't match the number of inputs")
        starts = [len(t) for t in self.tokens]
        lengths = [len(t) for t in tokens]
        widest = max(lengths, default=0)
        padding = [widest - n for n in lengths]
        if not tokens or (
            max(padding) > 0
            and not self._overlaps(starts, 0, widest)
            and all(self.positions_of(uid) is None for uid in self.uids)
        ):
            # Text only: upstream's padded prefill.
            return super().prompt(tokens)
        for sti, ti in zip(self.tokens, tokens, strict=True):
            sti += ti
        rows = tokens
        padded = max(padding) > 0
        if padded:
            # The scheduler cuts every call to the shortest row, except
            # over a block span: then the shorter rows are right-padded, as
            # upstream pads, and their caches trimmed again at the end.
            tokens = _right_pad_prompts(tokens, max_length=widest)
            for c in self.prompt_cache:
                c.prepare(lengths=lengths, right_padding=padding)
        else:
            tokens = mx.array(tokens)
        done = 0
        while tokens.shape[1] > 0:
            n = min(self.prefill_step_size, tokens.shape[1])
            n = min(self._whole_blocks(starts, done, n), tokens.shape[1])
            chunk = tokens[:, :n]
            overlaps = self._overlaps(starts, done, n)
            kwargs = {}
            ids = chunk
            if overlaps:
                # With embeddings given the ids feed only what a model reads
                # per position beside them (Gemma's per-layer inputs); the
                # image positions read the pad id there, as the reference
                # implementations do.
                ids = mx.array(chunk)
                for i, a, b, _, _ in overlaps:
                    ids[i, a:b] = 0
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
                blocks = self._block_ids(overlaps, n)
                if blocks is not None:
                    if not self._blocks_kw:
                        raise ValueError(
                            f"{type(self.model).__name__} takes no block ids; the "
                            "image's tokens cannot attend to each other"
                        )
                    kwargs["block_ids"] = blocks
            positions = self._positions(starts, done, n)
            if positions is not None:
                if not self._positions_kw:
                    raise ValueError(
                        f"{type(self.model).__name__} takes no position ids; the "
                        "frontend's multimodal positions cannot be applied"
                    )
                kwargs["position_ids"] = positions
            hidden = self._inner(ids, cache=self.prompt_cache, **kwargs)
            mx.eval(hidden, *_cache_arrays(self.prompt_cache))
            if proposer is not None:
                for i, uid in enumerate(self.uids):
                    real = rows[i][done : done + n]
                    if real:
                        proposer.prime(
                            uid,
                            hidden[i : i + 1, : len(real)],
                            real,
                            starts[i] + done,
                        )
            mx.clear_cache()
            tokens = tokens[:, n:]
            done += n
        if padded:
            for c in self.prompt_cache:
                c.finalize()
            mx.eval(*_cache_arrays(self.prompt_cache))
            mx.clear_cache()

    def generate(self, tokens):
        proposer = self._proposer
        if proposer is not None:
            # The last prompt token is fed by the generation batch's first
            # step; its pair with the hidden state before it closes the
            # prompt's history.
            for uid, t in zip(self.uids, tokens, strict=True):
                proposer.follow(uid, None, mx.array([t[-1]]))
        return super().generate(tokens)
