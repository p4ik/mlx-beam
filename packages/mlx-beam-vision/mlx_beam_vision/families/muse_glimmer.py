"""Muse Glimmer's tower (Meta): a ViT with windowed and full blocks over
patches of 2 x 14 x 14, a pixel shuffle merging 2 x 2 patches, then the
adapter (two linears with GELUs), the projection to the text width and a
scaleless RMSNorm. The text model norms its input embeddings with the same
scaleless RMSNorm, which is idempotent on features already normed, so the
features go in as the projector leaves them. Placeholders are one
contiguous run per image. Tower vendored from mlx-vlm, the adapter and
projection after its `muse_glimmer.py`.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_beam_vision._vendor.mlx_vlm.muse_glimmer.config import VisionConfig
from mlx_beam_vision._vendor.mlx_vlm.muse_glimmer.vision import VisionModel
from mlx_beam_vision.families import Encoded
from mlx_beam_vision.weights import load_prefixed, strip_prefix

NAME = "muse_glimmer"
PARTS = ("vision_tower", "vision_adapter", "vision_projection")
PREFIX_ROOTS = ("model.", "")


class Adapter(nn.Module):
    def __init__(self, out_hidden: int, projector_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(out_hidden, projector_hidden, bias=False)
        self.fc2 = nn.Linear(projector_hidden, projector_hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.fc2(nn.gelu(self.fc1(x))))


class Tower:
    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        self.config = VisionConfig.from_dict(config.get("vision_config") or {})
        text = config.get("text_config") or {}
        self.image_token_id = int(config.get("image_token_id", 200092))
        self.eps = float(text.get("rms_norm_eps", 1e-5))
        self.dtype = dtype
        self.model = VisionModel(self.config)
        self.vision_adapter = Adapter(
            int(config.get("out_hidden_size", 6144)), int(config.get("projector_hidden_size", 4096))
        )
        self.vision_projection = nn.Linear(
            int(config.get("projector_hidden_size", 4096)), int(text.get("hidden_size", 4096)), bias=False
        )
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        prefixes = tuple(f"{root}{part}." for root in PREFIX_ROOTS for part in PARTS)
        raw = load_prefixed(model_path, prefixes)
        root = next((r for r in PREFIX_ROOTS if any(k.startswith(f"{r}vision_tower.") for k in raw)), None)
        if root is None:
            raise FileNotFoundError(f"{model_path}: no vision tower in the index")
        raw = {k: v for k, v in raw.items() if "rotary_emb.inv_freq" not in k}
        if any(k.endswith(".scales") for k in raw):
            raise ValueError("a quantized vision tower is not supported yet")
        cast = lambda d: [(k, v.astype(self.dtype)) for k, v in d.items()]  # noqa: E731
        self.model.load_weights(cast(strip_prefix(raw, f"{root}vision_tower.")), strict=True)
        self.vision_adapter.load_weights(cast(strip_prefix(raw, f"{root}vision_adapter.")), strict=True)
        self.vision_projection.load_weights(cast(strip_prefix(raw, f"{root}vision_projection.")), strict=True)
        mx.eval(self.model.parameters(), self.vision_adapter.parameters(), self.vision_projection.parameters())
        self.loaded_from = sorted({k.split("/")[0] for k in raw})

    def encode(self, pixel_values: mx.array, grid_thw: mx.array) -> list[Encoded]:
        features = self.model(pixel_values.astype(self.dtype), grid_thw)
        features = self.vision_projection(self.vision_adapter(features))
        features = mx.fast.rms_norm(features, None, self.eps)
        merge = self.config.merge_size**2
        out = []
        at = 0
        for t, h, w in grid_thw.tolist():
            n = int(t * h * w) // merge
            out.append(Encoded(features[at : at + n]))
            at += n
        return out

    def describe(self) -> dict:
        return {
            "family": NAME,
            "hidden_size": self.config.hidden_size,
            "patch_size": self.config.patch_size,
            "merge_size": self.config.merge_size,
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
        }


def build(config: dict, model_path: Path | None, dtype=mx.bfloat16) -> Tower:
    return Tower(config, model_path, dtype)


def pixel_inputs(processed: dict):
    return mx.array(processed["pixel_values"]), mx.array(processed["image_grid_thw"])
