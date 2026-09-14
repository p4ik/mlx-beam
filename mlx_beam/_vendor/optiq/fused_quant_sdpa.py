# Copyright (c) 2026 Thin Signal (mlx-optiq, MIT). Taken from
# optiq/runtime/fused_quant_sdpa.py 0.5.6 without the monkeypatch installer;
# the vendored mlx-lm calls it directly. See VENDORED.md.
"""Tiled quantized SDPA that never materialises the full scores matrix.

The stock quantized attention computes one ``Q @ K^T`` scores matrix per
call, ``(B, n_kv_heads, n_repeats, L_q, N)`` after GQA expansion: at a 2048
token prefill chunk against 16k cached tokens that is gigabytes of fp16, and
the softmax output is co-resident with it. Quantized KV then peaked above
fp16 KV on a 24 GB machine (16.35 vs 11.51 GB at 32k), defeating its purpose.

This is FlashAttention-2 over the N axis with ``mx.quantized_matmul`` as the
inner kernel: per-tile scores are bounded by ``n_chunk``, the online softmax
keeps a running max and sum. Measured 7.60 GB peak at 32k with 4-bit KV.
Affine quantization only, bits 4 or 8, group size 32/64/128, causal or no
mask; anything else must take the stock path.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map

# Tile width along the cached-token axis; bounds the per-tile scores tensor.
N_CHUNK = 512


def supported(queries, bits: int, group_size: int, mask) -> bool:
    return (
        bits in (4, 8)
        and group_size in (32, 64, 128)
        and queries.dtype in (mx.float16, mx.bfloat16)
        and (mask is None or (isinstance(mask, str) and mask == "causal"))
    )


def fused_quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys,
    q_values,
    scale: float,
    mask,
    group_size: int = 64,
    bits: int = 8,
    n_chunk: int = N_CHUNK,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    N = q_keys[0].shape[-2]

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    mask_arr = None
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            q_indices = mx.arange(N - L, N)
            k_indices = mx.arange(N)
            mask_arr = q_indices[:, None] >= k_indices[None]
        else:
            mask_arr = mask

    o_acc = None
    row_max = None
    row_sum = None

    for n_start in range(0, N, n_chunk):
        n_end = min(n_start + n_chunk, N)
        k_packed, k_scales, k_biases = (x[..., n_start:n_end, :] for x in q_keys)
        v_packed, v_scales, v_biases = (x[..., n_start:n_end, :] for x in q_values)

        scores = mx.quantized_matmul(
            queries,
            k_packed,
            k_scales,
            k_biases,
            transpose=True,
            group_size=group_size,
            bits=bits,
        )

        if mask_arr is not None:
            if mask_arr.ndim == 2:
                mask_chunk = mask_arr[:, n_start:n_end]
            else:
                mask_chunk = mask_arr[..., :, n_start:n_end]
            if mask_chunk.dtype == mx.bool_:
                scores = mx.where(mask_chunk, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask_chunk

        chunk_max = mx.max(scores, axis=-1, keepdims=True)
        if o_acc is None:
            row_max = chunk_max
            exps = mx.exp(scores - row_max)
            row_sum = mx.sum(exps, axis=-1, keepdims=True)
            o_acc = mx.quantized_matmul(
                exps,
                v_packed,
                v_scales,
                v_biases,
                transpose=False,
                group_size=group_size,
                bits=bits,
            )
        else:
            new_max = mx.maximum(row_max, chunk_max)
            factor = mx.exp(row_max - new_max)
            exps = mx.exp(scores - new_max)
            new_sum = factor * row_sum + mx.sum(exps, axis=-1, keepdims=True)
            delta_out = mx.quantized_matmul(
                exps,
                v_packed,
                v_scales,
                v_biases,
                transpose=False,
                group_size=group_size,
                bits=bits,
            )
            o_acc = o_acc * factor + delta_out
            row_max = new_max
            row_sum = new_sum

    out = o_acc / row_sum

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out
