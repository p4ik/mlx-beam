"""Which KV cache each layer of a request gets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_beam._vendor.mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_beam._vendor.optiq.kv_batch import MergeableQuantizedKVCache

VALID_BITS = (4, 8)
VALID_GROUP_SIZES = (32, 64, 128)


@dataclass(frozen=True)
class KVPolicy:
    """Bits per full-attention layer; ``None`` keeps a layer at model precision.

    ``bits`` applies to every full-attention layer, ``layers`` overrides single
    layers (mixed precision, set at conversion time and carried in the
    package). Recurrent state, sliding-window and chunked caches are never
    quantized here: their size is bounded and the batch path has no quantized
    form for them.
    """

    bits: int | None = None
    group_size: int = 64
    layers: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for b in (self.bits, *self.layers.values()):
            if b is not None and b not in VALID_BITS:
                raise ValueError(f"kv bits must be one of {VALID_BITS}, got {b}")
        if self.group_size not in VALID_GROUP_SIZES:
            raise ValueError(
                f"kv group size must be one of {VALID_GROUP_SIZES}, got {self.group_size}"
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
        }


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
        out.append(entry)
    return out
