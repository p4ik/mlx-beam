# Copyright © 2023 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map

from ...optiq import fused_quant_sdpa


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if n_repeats > 1 and mask.ndim > 3:
            mask = mx.expand_dims(mask, -3)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def _packed_layout(queries, keys):
    """Bits and group size of packed keys, read off their shapes: a layer
    that shares another layer's cache (Gemma 4) gets the packed triple
    with no cache of its own to ask."""
    head_dim = queries.shape[-1]
    packed, scales = keys[0], keys[1]
    return packed.shape[-1] * 32 // head_dim, head_dim // scales.shape[-1]


# Set by mlx_beam.engine.exact for the duration of an exact forward: a
# block of L queries is attended one query at a time, so the quantized KV
# path sums like plain decoding does (VENDORED.md, exact verify).
EXACT_PER_QUERY = False


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    L = queries.shape[2]
    if EXACT_PER_QUERY and L > 1 and mask is not None:
        S = (keys[0] if isinstance(keys, (tuple, list)) else keys).shape[2]
        rows = []
        for i in range(L):
            if isinstance(mask, mx.array):
                row = mask[..., i : i + 1, :]
            else:  # "causal": query i of the block sees the keys up to its own
                row = (mx.arange(S) <= S - L + i)[None, None, None, :]
            rows.append(
                scaled_dot_product_attention(
                    queries[:, :, i : i + 1], keys, values, cache, scale, row, sinks
                )
            )
        return mx.concatenate(rows, axis=2)
    if hasattr(cache, "bits"):
        bits, group_size = cache.bits, cache.group_size
    elif isinstance(keys, (tuple, list)):
        bits, group_size = _packed_layout(queries, keys)
    else:
        bits = None
    if bits is not None:
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        # The tiled path bounds the prefill transient; the stock one takes
        # the shapes and masks it does not cover (VENDORED.md, tiled SDPA).
        if fused_quant_sdpa.supported(queries, bits, group_size, mask):
            return fused_quant_sdpa.fused_quantized_scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=scale,
                mask=mask,
                group_size=group_size,
                bits=bits,
            )
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=group_size,
            bits=bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
