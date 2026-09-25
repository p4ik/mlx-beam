"""The exact verify: a block of T rows through the model gives the bytes T
single-row forwards would give.

On Metal the quantized matmul sums in a different order from two rows on,
so the block verify of the speculative path can differ from plain decoding
by a rounding - enough to flip a bf16 tie (measured 2026-09-25). Two ways
around it, both switched on per forward with `exact_forward()`:

* every `QuantizedLinear` of the model gets this module's subclass swapped
  in (the instance keeps its weights; the class is ours, not a patch of
  mlx's); inside an exact forward it projects through the vendored
  mlx-vlm kernel that keeps single-row arithmetic for a block, and where
  that kernel does not apply (shape, dtype, no Metal) through one native
  call per position, which is exact by construction;
* the vendored attention takes one query at a time (`base.py` hook), the
  quantized KV path being width-dependent too.

Whether the result is bit-equal to single forwards on this machine is not
assumed: the engine measures it at warm-up (`speculative.width_check`) and
falls back to the block verify when it is not, saying so in /health.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

from mlx_beam._vendor.mlx_lm.models import base
from mlx_beam._vendor.mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_beam._vendor.mlx_vlm.quantized_verifier import (
    _exact_time_batch,
    exact_quantized_switch_linear,
    optimized_affine_linear,
    supports_quantization,
)

_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "exact_forward", default=False
)


@contextmanager
def exact_forward():
    """Every projection and attention inside runs in its exact form."""
    token = _ACTIVE.set(True)
    base.EXACT_PER_QUERY = True
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        base.EXACT_PER_QUERY = False


def active() -> bool:
    return _ACTIVE.get()


class ExactQuantizedLinear(nn.QuantizedLinear):
    """A QuantizedLinear that, inside `exact_forward()`, projects a block of
    T > 1 rows the way T single rows would be projected."""

    def __call__(self, x: mx.array) -> mx.array:
        if not _ACTIVE.get() or x.ndim != 3 or x.shape[1] <= 1:
            return super().__call__(x)
        # The mlx-vlm kernel: single-row qmv arithmetic for a whole block
        # (affine 4/5/8 bit, K % 512, N % 8, on Metal). None where it does
        # not apply - then one native call per position.
        out = optimized_affine_linear(self, x)
        if out is not None:
            return out
        if (
            supports_quantization(self)
            and x.shape[-1] == self.scales.shape[-1] * self.group_size
        ):
            return _exact_time_batch(self, x)
        return super().__call__(x)


class ExactQuantizedSwitchLinear(QuantizedSwitchLinear):
    """The MoE expert projection: one hidden vector through its selected
    experts exactly, per position."""

    def __call__(self, x, indices, sorted_indices=False):
        if not _ACTIVE.get() or x.ndim != 5 or x.shape[1] <= 1:
            return super().__call__(x, indices, sorted_indices)
        # (B, T, 1, 1, K) rows, (B, T, E) indices - per position through the
        # stock call, which is a single-row call for each.
        outs = [
            super().__call__(x[:, t : t + 1], indices[:, t : t + 1], sorted_indices)
            for t in range(x.shape[1])
        ]
        return mx.concatenate(outs, axis=1)


def install(model) -> dict:
    """Swap the model's quantized projections for the exact subclasses.
    Returns what was swapped, for /health."""
    counts = {"linear": 0, "switch": 0}
    for _, module in model.named_modules():
        if type(module) is nn.QuantizedLinear:
            module.__class__ = ExactQuantizedLinear
            counts["linear"] += 1
        elif type(module) is QuantizedSwitchLinear:
            module.__class__ = ExactQuantizedSwitchLinear
            counts["switch"] += 1
    return counts


def uninstall(model) -> None:
    for _, module in model.named_modules():
        if type(module) is ExactQuantizedLinear:
            module.__class__ = nn.QuantizedLinear
        elif type(module) is ExactQuantizedSwitchLinear:
            module.__class__ = QuantizedSwitchLinear


_ = exact_quantized_switch_linear  # kept importable for the MoE path's callers
