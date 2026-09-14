"""Which KV cache each layer of a request gets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    bits: Optional[int] = None
    group_size: int = 64
    layers: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for b in (self.bits, *self.layers.values()):
            if b is not None and b not in VALID_BITS:
                raise ValueError(f"kv bits must be one of {VALID_BITS}, got {b}")
        if self.group_size not in VALID_GROUP_SIZES:
            raise ValueError(
                f"kv group size must be one of {VALID_GROUP_SIZES}, got {self.group_size}"
            )

    def bits_for(self, layer: int) -> Optional[int]:
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


def make_request_cache(model: Any, policy: KVPolicy) -> List[Any]:
    """A fresh per-request cache list, quantized where the policy says so.

    Starts from what the model asks for (hybrids bring their own layout via
    ``make_cache``) and swaps only plain ``KVCache`` entries.
    """
    caches = make_prompt_cache(model)
    for idx, c in enumerate(caches):
        bits = policy.bits_for(idx)
        if bits is not None and type(c) is KVCache:
            caches[idx] = MergeableQuantizedKVCache(
                group_size=policy.group_size, bits=bits
            )
    return caches


def describe_caches(caches: List[Any]) -> List[dict]:
    """What a cache list actually is, layer by layer - the evidence for /health."""
    out = []
    for c in caches:
        entry = {"type": type(c).__name__}
        if hasattr(c, "bits"):
            entry["bits"] = c.bits
            entry["group_size"] = c.group_size
        out.append(entry)
    return out
