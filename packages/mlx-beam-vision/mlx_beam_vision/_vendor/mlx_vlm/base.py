# Copyright © 2024 Apple Inc. and the mlx-vlm contributors. MIT License.
#
# The two pieces the vendored towers import from mlx_vlm/models/base.py,
# copied verbatim; the rest of that module (processors, features) stays out.

import inspect
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        if not params:
            return cls()
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def ensure_fused_sdpa(q, k, v, scale, mask=None):
    fused_dims = (64, 80, 128)  # supported by MLX's fused SDPA kernel
    d = q.shape[-1]
    target = next((t for t in fused_dims if d <= t), d)
    if target != d:
        pad = [(0, 0)] * (q.ndim - 1) + [(0, target - d)]
        q, k, v = mx.pad(q, pad), mx.pad(k, pad), mx.pad(v, pad)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)[
        ..., :d
    ]
