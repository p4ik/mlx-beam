"""Qwen3-VL's tower, which Qwen3.5 and Qwen3.8 carry unchanged: a ViT with
2D rotary embeddings, windowed and full blocks, a patch merger projecting to
the text width, and DeepStack - the outputs of three intermediate blocks,
merged the same way, that the text model adds at the image positions after
its first three layers (`deepstack_visual_indexes` in the vision config).
The processor's pixel inputs (`pixel_values` with `image_grid_thw`) come
from transformers' own Qwen processors.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_beam_vision._vendor.mlx_vlm.qwen3_vl.config import VisionConfig
from mlx_beam_vision._vendor.mlx_vlm.qwen3_vl.vision import VisionModel
from mlx_beam_vision.families import Encoded
from mlx_beam_vision.weights import load_prefixed, strip_prefix

# Where the checkpoints keep the tower: the HF layout of Qwen3-VL and of
# Qwen3.5 (`model.visual`), and mlx-vlm's own conversions (`vision_tower`).
PREFIXES = ("model.visual.", "visual.", "vision_tower.")
NAME = "qwen3_vl"


class Tower:
    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        vc = config.get("vision_config") or {}
        self.config = VisionConfig.from_dict(vc)
        self.image_token_id = int(config.get("image_token_id", 151655))
        self.video_token_id = int(config.get("video_token_id", 151656))
        self.model = VisionModel(self.config)
        self.dtype = dtype
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        raw = load_prefixed(model_path, PREFIXES)
        if not raw:
            raise FileNotFoundError(
                f"{model_path}: no tensors under {PREFIXES}; the checkpoint ships "
                "no vision tower the index knows"
            )
        prefix = next(p for p in PREFIXES if any(k.startswith(p) for k in raw))
        weights = self.model.sanitize(strip_prefix(raw, prefix))
        if any(k.endswith(".scales") for k in weights):
            raise ValueError("a quantized vision tower is not supported yet")
        weights = {
            k: v.astype(self.dtype) if v.dtype != mx.uint32 else v
            for k, v in weights.items()
        }
        self.model.load_weights(list(weights.items()), strict=True)
        mx.eval(self.model.parameters())
        self.loaded_from = sorted({k.split("/")[0] for k in raw})

    @property
    def tokens_per_patch_group(self) -> int:
        return self.config.spatial_merge_size**2

    def encode(self, pixel_values: mx.array, grid_thw: mx.array) -> list[Encoded]:
        """All images of one processor call at once (the tower takes them
        concatenated with their grids), split back per image."""
        features, deepstack = self.model(pixel_values.astype(self.dtype), grid_thw)
        counts = [
            int(t * h * w) // self.tokens_per_patch_group
            for t, h, w in grid_thw.tolist()
        ]
        out = []
        start = 0
        for n in counts:
            # The tower's k-th DeepStack feature goes in after text layer
            # k, i.e. ahead of layer k + 1.
            out.append(
                Encoded(
                    features[start : start + n],
                    {k + 1: d[start : start + n] for k, d in enumerate(deepstack)},
                )
            )
            start += n
        return out

    def describe(self) -> dict:
        return {
            "family": NAME,
            "depth": self.config.depth,
            "hidden_size": self.config.hidden_size,
            "out_hidden_size": self.config.out_hidden_size,
            "patch_size": self.config.patch_size,
            "spatial_merge_size": self.config.spatial_merge_size,
            "deepstack_layers": len(self.config.deepstack_visual_indexes),
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
        }


def build(config: dict, model_path: Path | None, dtype=mx.bfloat16) -> Tower:
    return Tower(config, model_path, dtype)


def pixel_inputs(processed: dict) -> tuple[mx.array, mx.array]:
    """The tower's inputs from what the processor returned."""
    return mx.array(processed["pixel_values"]), mx.array(processed["image_grid_thw"])


__all__ = ["NAME", "PREFIXES", "Tower", "build", "pixel_inputs", "nn"]
