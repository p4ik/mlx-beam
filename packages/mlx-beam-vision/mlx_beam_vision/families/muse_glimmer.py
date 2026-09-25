"""Muse Glimmer's tower (Meta): a ViT with windowed and full blocks over
patches of 2 x 14 x 14, a pixel shuffle merging 2 x 2 patches, then the
adapter (two linears with GELUs), the projection to the text width and a
scaleless RMSNorm. The text model norms its token embeddings with the
same scaleless RMSNorm (`embed_inputs`) and takes given input embeddings
as already normed, so the features go in as the projector leaves them -
the way the reference scatters them. Placeholders are one contiguous run
per image. Tower vendored from mlx-vlm, the adapter and projection after
its `muse_glimmer.py`.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_beam_vision._vendor.mlx_vlm.muse_glimmer.config import VisionConfig
from mlx_beam_vision._vendor.mlx_vlm.muse_glimmer.vision import VisionModel
from mlx_beam_vision.families import (
    Encoded,
    Loaded,
    cast,
    grid_count,
    grid_inputs,
    grid_select,
    split_by_grid,
)

NAME = "muse_glimmer"
PARTS = ("vision_tower", "vision_adapter", "vision_projection")


class Adapter(nn.Module):
    def __init__(self, out_hidden: int, projector_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(out_hidden, projector_hidden, bias=False)
        self.fc2 = nn.Linear(projector_hidden, projector_hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.fc2(nn.gelu(self.fc1(x))))


def per_layer(config: dict) -> bool:
    return False  # everything enters at the embedding


class Tower:
    per_layer = False

    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        self.config = VisionConfig.from_dict(config.get("vision_config") or {})
        text = config.get("text_config") or {}
        self.image_token_id = int(config.get("image_token_id", 200092))
        self.eps = float(text.get("rms_norm_eps", 1e-5))
        self.dtype = dtype
        self.model = VisionModel(self.config)
        self.vision_adapter = Adapter(
            int(config.get("out_hidden_size", 6144)),
            int(config.get("projector_hidden_size", 4096)),
        )
        self.vision_projection = nn.Linear(
            int(config.get("projector_hidden_size", 4096)),
            int(text.get("hidden_size", 4096)),
            bias=False,
        )
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        loaded = Loaded(model_path, PARTS)
        tower = {
            k: v
            for k, v in loaded.part("vision_tower").items()
            if "rotary_emb.inv_freq" not in k
        }
        self.model.load_weights(cast(tower, self.dtype), strict=True)
        self.vision_adapter.load_weights(
            cast(loaded.part("vision_adapter"), self.dtype), strict=True
        )
        self.vision_projection.load_weights(
            cast(loaded.part("vision_projection"), self.dtype), strict=True
        )
        mx.eval(
            self.model.parameters(),
            self.vision_adapter.parameters(),
            self.vision_projection.parameters(),
        )
        self.loaded_from = loaded.shards

    def encode(self, pixel_values: mx.array, grid_thw: mx.array) -> list[Encoded]:
        features = self.model(pixel_values.astype(self.dtype), grid_thw)
        features = self.vision_projection(self.vision_adapter(features))
        features = mx.fast.rms_norm(features, None, self.eps)
        return [
            Encoded(f)
            for f in split_by_grid(features, grid_thw, self.config.merge_size**2)
        ]

    def describe(self) -> dict:
        return {
            "family": NAME,
            "hidden_size": self.config.hidden_size,
            "patch_size": self.config.patch_size,
            "merge_size": self.config.merge_size,
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
        }


pixel_inputs = grid_inputs
select = grid_select
count = grid_count
