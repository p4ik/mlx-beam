"""Which KV cache each layer of a request gets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.models.cache import (
    BatchKVCache,
    KVCache,
    make_prompt_cache,
)
from mlx_beam._vendor.optiq.kv_batch import (
    BatchQuantizedKVCache,
    MergeableQuantizedKVCache,
)

VALID_BITS = (4, 8)
VALID_GROUP_SIZES = (32, 64, 128)
PREFILL_MODES = ("exact", "quantized")


@dataclass(frozen=True)
class KVPolicy:
    """Bits per full-attention layer; ``None`` keeps a layer at model precision.

    ``bits`` applies to every full-attention layer, ``layers`` overrides single
    layers (mixed precision, set at conversion time and carried in the
    package). Recurrent state, sliding-window and chunked caches are never
    quantized here: their size is bounded and the batch path has no quantized
    form for them.

    ``prefill`` says when a layer's cache becomes quantized. ``exact`` keeps
    the prompt's keys and values at model precision while it is prefilled
    and quantizes them once, when the request moves to decoding - what
    mlx-lm's own quantized path does, and the semantics every calibration
    of a bit profile was measured under. ``quantized`` writes each token
    quantized as the prefill goes, so the prefill itself attends over
    quantized data: it saves the prompt's full-precision transient (a 64k
    prompt on a 27B is ~2 GB), and on some models it costs accuracy (Gemma 4,
    8 bit: KL 0.022 against 0.0005 the exact way, measured 2026-09-18).
    A package may carry ``quantized`` for a profile it was measured with.
    """

    bits: int | None = None
    group_size: int = 64
    layers: dict[int, int] = field(default_factory=dict)
    prefill: str = "exact"
    # Where ``prefill`` came from: "default", "profile" (the package's
    # kv_config, measured for it) or "flag" (the operator's word).
    prefill_source: str = "default"

    def __post_init__(self):
        for b in (self.bits, *self.layers.values()):
            if b is not None and b not in VALID_BITS:
                raise ValueError(f"kv bits must be one of {VALID_BITS}, got {b}")
        if self.group_size not in VALID_GROUP_SIZES:
            raise ValueError(
                f"kv group size must be one of {VALID_GROUP_SIZES}, got {self.group_size}"
            )
        if self.prefill not in PREFILL_MODES:
            raise ValueError(
                f"kv prefill must be one of {PREFILL_MODES}, got {self.prefill!r}"
            )

    def bits_for(self, layer: int) -> int | None:
        return self.layers.get(layer, self.bits)

    @property
    def quantizes(self) -> bool:
        return self.bits is not None or any(b for b in self.layers.values())

    def describe(self) -> dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "layers": dict(sorted(self.layers.items())),
            "prefill": self.prefill,
            "prefill_source": self.prefill_source,
        }


def _write_packed(dst, src, row: int, start: int):
    """Copy packed (codes, scales, biases) of ``src`` into row ``row`` of
    ``dst`` from token ``start`` on."""
    n = src[0].shape[2]
    for d, s_ in zip(dst, src, strict=True):
        d[row : row + 1, :, start : start + n] = s_


class DeferredQuantizedKVCache(KVCache):
    """A plain cache for the prefill that knows it will be quantized.

    Attribute names avoid ``bits``: that name is what routes attention to
    the quantized kernels, and during the prefill this cache is exact.

    A prefix restored from the store keeps its original codes: they are
    written back at the handover instead of re-quantizing values that were
    dequantized from them (a round trip is not exact, and it would drift
    a little with every turn).
    """

    def __init__(self, bits: int, group_size: int):
        super().__init__()
        self.pending_bits = bits
        self.pending_group_size = group_size
        # (packed keys, packed values) of the restored prefix, or None
        self.packed_prefix = None

    @classmethod
    def merge(cls, caches):
        batch = BatchKVCache.merge(caches)
        batch.__class__ = DeferredBatchKVCache
        spec = {(c.pending_bits, c.pending_group_size) for c in caches}
        if len(spec) > 1:
            raise ValueError(
                f"cannot merge caches quantized differently: {sorted(spec)}"
            )
        batch.pending_bits, batch.pending_group_size = spec.pop()
        batch.packed_prefixes = [c.packed_prefix for c in caches]
        return batch

    def to_quantized(self, group_size: int | None = None, bits: int | None = None):
        cache = MergeableQuantizedKVCache(
            group_size=group_size or self.pending_group_size,
            bits=bits or self.pending_bits,
        )
        cache.offset = self.offset
        if self.keys is not None:
            cache.keys = mx.quantize(
                self.keys, group_size=cache.group_size, bits=cache.bits
            )
            cache.values = mx.quantize(
                self.values, group_size=cache.group_size, bits=cache.bits
            )
            if self.packed_prefix is not None:
                _write_packed(cache.keys, self.packed_prefix[0], 0, 0)
                _write_packed(cache.values, self.packed_prefix[1], 0, 0)
        return cache


class DeferredBatchKVCache(BatchKVCache):
    """The prompt batch of deferred caches; ``quantized()`` is the handover."""

    pending_bits = 8
    pending_group_size = 64
    packed_prefixes: list = []

    def filter(self, batch_indices):
        super().filter(batch_indices)
        self.packed_prefixes = [self.packed_prefixes[i] for i in batch_indices]

    def extend(self, other):
        super().extend(other)
        self.packed_prefixes = self.packed_prefixes + list(other.packed_prefixes)

    def extract(self, idx):
        plain = super().extract(idx)
        cache = DeferredQuantizedKVCache(self.pending_bits, self.pending_group_size)
        cache.keys, cache.values, cache.offset = plain.keys, plain.values, plain.offset
        cache.packed_prefix = self.packed_prefixes[idx]
        return cache

    def quantized(self) -> BatchQuantizedKVCache:
        """The same rows, keys and values quantized: what decoding reads."""
        q = BatchQuantizedKVCache(
            [0], group_size=self.pending_group_size, bits=self.pending_bits
        )
        q.left_padding, q.offset, q._idx = self.left_padding, self.offset, self._idx
        if self.keys is not None:
            q.keys = mx.quantize(self.keys, group_size=q.group_size, bits=q.bits)
            q.values = mx.quantize(self.values, group_size=q.group_size, bits=q.bits)
            mx.eval(self.left_padding)
            for row, (prefix, pad) in enumerate(
                zip(self.packed_prefixes, self.left_padding.tolist(), strict=True)
            ):
                if prefix is not None:
                    _write_packed(q.keys, prefix[0], row, pad)
                    _write_packed(q.values, prefix[1], row, pad)
        return q


def dequantized(cache: MergeableQuantizedKVCache) -> DeferredQuantizedKVCache:
    """A stored, quantized layer back at model precision for an exact prefill;
    its codes come back unchanged when the request is stored."""
    out = DeferredQuantizedKVCache(cache.bits, cache.group_size)
    if cache.keys is not None:
        n = cache.offset
        packed = tuple(
            tuple(mx.contiguous(a[..., :n, :]) for a in part)
            for part in (cache.keys, cache.values)
        )
        out.keys = mx.dequantize(
            *packed[0], group_size=cache.group_size, bits=cache.bits
        )
        out.values = mx.dequantize(
            *packed[1], group_size=cache.group_size, bits=cache.bits
        )
        out.offset = n
        out.packed_prefix = packed
    return out


def for_prefill(caches: list[Any], policy: KVPolicy) -> list[Any]:
    """A restored cache list as this policy's prefill wants it."""
    if policy.prefill != "exact":
        return caches
    return [
        dequantized(c) if isinstance(c, MergeableQuantizedKVCache) else c
        for c in caches
    ]


def for_store(caches: list[Any]) -> list[Any]:
    """A cache list as the store keeps it: every deferred layer quantized."""
    return [
        c.to_quantized() if isinstance(c, DeferredQuantizedKVCache) else c
        for c in caches
    ]


def make_request_cache(model: Any, policy: KVPolicy) -> list[Any]:
    """A fresh per-request cache list, quantized where the policy says so.

    Starts from what the model asks for (hybrids bring their own layout via
    ``make_cache``) and swaps only plain ``KVCache`` entries.
    """
    caches = make_prompt_cache(model)
    quantized = [
        i
        for i, c in enumerate(caches)
        if policy.bits_for(i) is not None and type(c) is KVCache
    ]
    if quantized:
        # mx.quantize wants the head dim divisible by the group; say so in
        # terms of the policy instead of a kernel error at warm-up.
        args = getattr(model, "args", None)
        for name in ("head_dim", "global_head_dim"):
            dim = getattr(args, name, None)
            if isinstance(dim, int) and dim % policy.group_size:
                raise ValueError(
                    f"kv group size {policy.group_size} does not divide the "
                    f"model's {name} {dim}; use one of "
                    f"{[g for g in VALID_GROUP_SIZES if dim % g == 0]}"
                )
    for idx in quantized:
        if policy.prefill == "exact":
            caches[idx] = DeferredQuantizedKVCache(
                policy.bits_for(idx), policy.group_size
            )
        else:
            caches[idx] = MergeableQuantizedKVCache(
                group_size=policy.group_size, bits=policy.bits_for(idx)
            )
    return caches


def describe_caches(caches: list[Any]) -> list[dict]:
    """What a cache list actually is, layer by layer - the evidence for /health."""
    out = []
    for c in caches:
        entry = {"type": type(c).__name__}
        if hasattr(c, "bits"):
            entry["bits"] = c.bits
            entry["group_size"] = c.group_size
            entry["prefill"] = "quantized"
        elif isinstance(c, DeferredQuantizedKVCache):
            entry["bits"] = c.pending_bits
            entry["group_size"] = c.pending_group_size
            entry["prefill"] = "exact"
        out.append(entry)
    return out
