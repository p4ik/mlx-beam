# Copyright © 2026 the mlx-beam contributors. Apache-2.0.
"""Multimodal RoPE for the Qwen vision families (Qwen3-VL, Qwen3.5/3.8):
every token carries three positions - time, height, width - and each
rotary frequency reads one of them. Text tokens carry the same value on
all three axes, so for them this is the plain RoPE the fast kernel
computes; an image's tokens spread over the height and width axes and
the text after it continues from the block's largest position, which is
fewer than its token count. The engine hands the positions in only for
prompts with images (``position_ids``, shape (3, B, L)); decoding after
an image shifts the fast path's offset by the prompt's delta instead.
Reference: transformers' Qwen3VLTextRotaryEmbedding and mlx-vlm's
apply_multimodal_rotary_pos_emb (interleaved layout)."""

from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn


def mrope_section_of(rope_scaling: Optional[dict]) -> Optional[list]:
    """The three section sizes from a config's rope dict, when the layout is
    the interleaved one these families use; None for any other rope."""
    if not isinstance(rope_scaling, dict):
        return None
    section = rope_scaling.get("mrope_section")
    if not section or len(section) != 3:
        return None
    if rope_scaling.get("mrope_interleaved", True) is not True:
        return None
    return [int(s) for s in section]


def interleaved_selector(mrope_section: Sequence[int], freq_dim: int) -> mx.array:
    """Which axis each frequency reads: time by default, height at every
    third frequency from 1, width at every third from 2, each axis for as
    many frequencies as its section says."""
    selector = [0] * freq_dim
    for axis, first in ((1, 1), (2, 2)):
        for idx in range(first, min(mrope_section[axis] * 3, freq_dim), 3):
            selector[idx] = axis
    return mx.array(selector, dtype=mx.int32)


def apply_mrope(
    x: mx.array, position_ids: mx.array, rope: nn.RoPE, selector: mx.array
) -> mx.array:
    """`x` (B, H, L, D) rotated over its first `rope.dims` with the positions
    each frequency selects; the pairing is the fast kernel's (i, i + dims/2),
    so text positions give the fast path's numbers."""
    dims, half = rope.dims, rope.dims // 2
    inv_freq = rope.base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    positions = mx.take(position_ids, selector, axis=0)  # (half, B, L)
    positions = positions.transpose(1, 2, 0).astype(mx.float32) * rope.scale
    angles = positions * inv_freq  # (B, L, half)
    cos = mx.cos(angles)[:, None]
    sin = mx.sin(angles)[:, None]
    rotated = x[..., :dims].astype(mx.float32)
    x1, x2 = rotated[..., :half], rotated[..., half:]
    out = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    out = out.astype(x.dtype)
    if dims == x.shape[-1]:
        return out
    return mx.concatenate([out, x[..., dims:]], axis=-1)


def rotate(rope, x: mx.array, cache, position_ids, rope_offset, selector):
    """The attention's one call: explicit positions when given (and the
    model has the layout for them), else the fast kernel from the cache's
    offset, shifted per row by `rope_offset` when a prompt's images left
    the positions behind its token count."""
    if position_ids is not None:
        if selector is None or not isinstance(rope, nn.RoPE):
            raise ValueError(
                "this model has no interleaved MRoPE layout; explicit "
                "positions cannot be applied"
            )
        return apply_mrope(x, position_ids, rope, selector)
    if cache is None:
        return rope(x)
    offset = cache.offset if rope_offset is None else cache.offset + rope_offset
    return rope(x, offset=offset)
