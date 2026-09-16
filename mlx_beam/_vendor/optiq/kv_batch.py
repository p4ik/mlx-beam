# Copyright (c) 2026 Thin Signal (mlx-optiq, MIT). Ported from
# optiq/runtime/kv/batch.py 0.5.6 to the vendored mlx-lm cache API; the
# installer that swapped these classes into BatchGenerator at run time is
# gone - the engine builds the per-request caches itself. See VENDORED.md.
"""Quantized KV cache for the batch generation path.

``BatchQuantizedKVCache`` is ``QuantizedKVCache``'s storage (a
``(packed_uint32, scales, biases)`` triple per tensor) with ``BatchKVCache``'s
batching semantics (per-sequence ``left_padding`` and ``offset``, plus
``merge`` / ``extend`` / ``filter`` / ``extract`` / ``prepare`` / ``finalize``).
Every batch-shaped operation is a slice or concatenate along the batch or
token axis, applied to the three component arrays independently.

Sliding-window layers are not covered: their KV is capped at the window size
and does not grow with context, and rotating a quantized ring buffer is a
harder problem for a much smaller payoff.
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from ..mlx_lm.models.cache import (
    BatchKVCache,
    QuantizedKVCache,
    create_causal_mask,
    dynamic_roll,
)


def _quant_triple(x, group_size: int, bits: int):
    return mx.quantize(x, group_size=group_size, bits=bits)


class BatchQuantizedKVCache(BatchKVCache):
    """A ``BatchKVCache`` whose keys and values are stored affine-quantized.

    ``keys`` and ``values`` are 3-tuples ``(packed, scales, biases)`` rather
    than single arrays, matching ``QuantizedKVCache`` so the quantized SDPA
    path applies unchanged.
    """

    step = 256

    def __init__(self, left_padding: List[int], group_size: int = 64, bits: int = 8):
        super().__init__(left_padding)
        self.group_size = group_size
        self.bits = bits

    def _alloc(self, B, n_kv_heads, n_steps, dim, dtype):
        el_per_int = 8 * mx.uint32.size // self.bits
        shape = (B, n_kv_heads, n_steps)
        return (
            mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
            mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            mx.zeros((*shape, dim // self.group_size), dtype=dtype),
        )

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            n_steps = (self.step + num_steps - 1) // self.step * self.step
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                def expand(x):
                    pad = mx.zeros((*x.shape[:2], n_steps, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, pad], axis=-2)

                self.keys, self.values = tree_map(expand, (self.keys, self.values))
            else:
                self.keys = self._alloc(B, n_kv_heads, n_steps, k_head_dim, keys.dtype)
                self.values = self._alloc(
                    B, n_kv_heads, n_steps, v_head_dim, values.dtype
                )

        self.offset = self.offset + num_steps
        self._idx += num_steps

        qk = _quant_triple(keys, self.group_size, self.bits)
        qv = _quant_triple(values, self.group_size, self.bits)
        for i in range(3):
            self.keys[i][..., prev : self._idx, :] = qk[i]
            self.values[i][..., prev : self._idx, :] = qv[i]

        # Evaluate the whole state, bookkeeping included: with only the
        # tensors evaluated, offset and left_padding once came back corrupted
        # (0x3F800000, float32 1.0) after the buffer pool was released under
        # them, and the mask built from them had zero width.
        if num_steps > 1:
            mx.eval(self.keys, self.values, self.offset, self.left_padding)

        return self.keys_and_values()

    def keys_and_values(self):
        return tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

    def finalize(self):
        # dynamic_roll along the token axis is per token, so rolling the three
        # component arrays by the same amount keeps a token's packed weights
        # with its own scale and bias.
        if self._right_padding is None:
            return
        padding = self._right_padding
        self.keys = tree_map(
            lambda x: dynamic_roll(x, padding[:, None], axis=2), self.keys
        )
        self.values = tree_map(
            lambda x: dynamic_roll(x, padding[:, None], axis=2), self.values
        )
        self.offset = self.offset - padding
        self.left_padding += padding
        self._right_padding = None

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tree_map(lambda x: x[batch_indices], self.keys)
            self.values = tree_map(lambda x: x[batch_indices], self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = min(self.left_padding.tolist())
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = tree_map(lambda x: x[..., min_left_pad:, :], self.keys)
                self.values = tree_map(lambda x: x[..., min_left_pad:, :], self.values)
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        ref = self if self.keys is not None else other
        H = ref.keys[0].shape[1]
        max_size = max(
            (c.keys[0].shape[2] if c.keys is not None else 0) for c in (self, other)
        )

        def pad(c):
            if c.keys is None:
                Bc = c.offset.shape[0]
                k = tuple(
                    mx.zeros((Bc, H, 0, x.shape[-1]), dtype=x.dtype) for x in ref.keys
                )
                v = tuple(
                    mx.zeros((Bc, H, 0, x.shape[-1]), dtype=x.dtype) for x in ref.values
                )
            else:
                k, v = c.keys, c.values
            left = max_idx - c._idx
            right = max_size - k[0].shape[2] - left
            if right < 0:
                k = tree_map(lambda x: x[..., :right, :], k)
                v = tree_map(lambda x: x[..., :right, :], v)
                right = 0
            if left or right:
                pads = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = tree_map(lambda x: mx.pad(x, pads), k)
                v = tree_map(lambda x: mx.pad(x, pads), v)
            return k, v, c.offset, c.left_padding + left

        a, b = pad(self), pad(other)
        self.keys = tuple(mx.concatenate([x, y]) for x, y in zip(a[0], b[0]))
        self.values = tuple(mx.concatenate([x, y]) for x, y in zip(a[1], b[1]))
        self.offset = mx.concatenate([a[2], b[2]])
        self.left_padding = mx.concatenate([a[3], b[3]])
        self._idx = max_idx

    def extract(self, idx):
        # A sequence that leaves the batch keeps a cache that can join the
        # next one (prompt reuse across turns needs merge on it).
        cache = MergeableQuantizedKVCache(group_size=self.group_size, bits=self.bits)
        mx.eval(self.left_padding)
        padding = self.left_padding.tolist()[idx]
        cache.keys = tuple(
            mx.contiguous(x[idx : idx + 1, :, padding : self._idx]) for x in self.keys
        )
        cache.values = tuple(
            mx.contiguous(x[idx : idx + 1, :, padding : self._idx]) for x in self.values
        )
        cache.offset = cache.keys[0].shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        # Bits and group size come from any cache that carries them, keys or
        # not: fresh prompts hold no keys yet, and taking the defaults here
        # silently turned a 4-bit configuration into 8-bit batches.
        spec = next(
            ((c.bits, c.group_size) for c in caches if hasattr(c, "bits")), (8, 64)
        )
        bits, group_size = spec

        if max_length == 0:
            return cls([0] * len(caches), group_size=group_size, bits=bits)

        first = next(c for c in caches if getattr(c, "keys", None) is not None)
        padding = [max_length - lg for lg in lengths]
        B = len(caches)
        H = first.keys[0].shape[1]

        def zeros_like_component(x):
            return mx.zeros((B, H, max_length, x.shape[-1]), dtype=x.dtype)

        keys = tuple(zeros_like_component(x) for x in first.keys)
        values = tuple(zeros_like_component(x) for x in first.values)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if getattr(c, "keys", None) is None:
                continue
            n = c.size()
            for j in range(3):
                keys[j][i : i + 1, :, p : p + n] = c.keys[j][..., :n, :]
                values[j][i : i + 1, :, p : p + n] = c.values[j][..., :n, :]

        cache = cls(padding, group_size=group_size, bits=bits)
        cache.keys, cache.values = keys, values
        cache.offset = cache.offset + max_length
        cache._idx = max_length
        return cache

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    def is_trimmable(self):
        return True

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)

    @property
    def state(self):
        return (
            self.keys,
            self.values,
            self.offset,
            self.left_padding,
            self._idx,
            self.group_size,
            self.bits,
        )

    @state.setter
    def state(self, v):
        (
            self.keys,
            self.values,
            self.offset,
            self.left_padding,
            self._idx,
            self.group_size,
            self.bits,
        ) = v


class MergeableQuantizedKVCache(QuantizedKVCache):
    """A ``QuantizedKVCache`` that can join a batch.

    The batch generator admits a model when every cache has ``merge``, and
    upstream's ``QuantizedKVCache`` has none - the structural reason quantized
    KV and batching were mutually exclusive. It also inherits ``size()`` as 0.
    Both are one line each. The guards below exist because these caches are
    built empty at request time, while upstream only ever builds them from a
    populated cache.
    """

    def size(self):
        return self.offset

    @classmethod
    def merge(cls, caches):
        return BatchQuantizedKVCache.merge(caches)

    def keys_and_values(self):
        if self.keys is None:
            return None, None
        return super().keys_and_values()

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)
