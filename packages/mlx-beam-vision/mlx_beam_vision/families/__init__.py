"""One module per tower family. A family knows which model types it
serves, how to build its tower from the checkpoint's config and tensors,
and how the processor's pixel inputs become the features of one image -
projected to the text model's width, with the per-layer extras a family
adds after the first text layers (DeepStack) when it has them."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class Encoded:
    """One image encoded: features (tokens, hidden), and what is added at
    its positions ahead of certain text layers ({layer: features})."""

    features: mx.array
    deepstack: dict[int, mx.array] = field(default_factory=dict)


FAMILIES: dict[str, str] = {
    # model_type -> family module under mlx_beam_vision.families
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_moe": "qwen3_vl",
    "qwen3_5": "qwen3_vl",
    "qwen3_5_moe": "qwen3_vl",
    "mistral3": "mistral3",
    "gemma4": "gemma4",
    "gemma4_unified": "gemma4",
    "muse_glimmer": "muse_glimmer",
    "granite4_vision": "granite4_vision",
}
