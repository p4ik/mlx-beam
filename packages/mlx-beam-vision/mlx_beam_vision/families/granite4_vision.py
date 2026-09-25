"""Granite Vision 4.1's tower: SigLIP over AnyRes tiles (a base view plus a
grid of tiles), then window Q-Former projectors - one per entry of
`deepstack_layer_map` (a vision layer's output for a text layer) and four
spatial ones sampling the 2 x 2 offsets of the last vision layer for
`spatial_target_layers`. Nothing enters at the embedding: the placeholder
positions are zero, every projector's features are added ahead of its
text layer. The tile grid is unpadded to the image's aspect ratio with an
image-newline feature per row, as the reference packs it. Tower, config,
Q-Former and downsamplers vendored from mlx-vlm; the packing and the
per-layer routing after its `granite4_vision.py`.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_beam_vision._vendor.mlx_vlm.granite4_vision.config import ModelConfig
from mlx_beam_vision._vendor.mlx_vlm.granite4_vision.downsampling import (
    WindowQFormerDownsampler,
)
from mlx_beam_vision._vendor.mlx_vlm.granite4_vision.vision import VisionModel
from mlx_beam_vision.families import Encoded
from mlx_beam_vision.weights import load_prefixed, strip_prefix

NAME = "granite4_vision"
PARTS = ("vision_tower", "layerwise_projectors", "spatial_projectors", "image_newline")


def grid_shape(image_size, pinpoints, patch: int) -> tuple[int, int]:
    from transformers.image_processing_utils import select_best_resolution

    height, width = select_best_resolution(list(image_size), pinpoints)
    return height // patch, width // patch


def unpad(tensor: mx.array, original_size) -> mx.array:
    """(C, H, W) cut back to the image's aspect ratio."""
    oh, ow = original_size
    ch, cw = tensor.shape[1], tensor.shape[2]
    if ow / oh > cw / ch:
        new_h = int(oh * (cw / ow))
        pad = (ch - new_h) // 2
        return tensor[:, pad : ch - pad, :]
    new_w = int(ow * (ch / oh))
    pad = (cw - new_w) // 2
    return tensor[:, :, pad : cw - pad]


class Tower:
    # The extras ahead of certain text layers need the prefill's layer hook.
    per_layer = True

    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        cfg = dict(config)
        cfg.setdefault("model_type", NAME)
        self.config = ModelConfig.from_dict(cfg)
        self.image_token_id = int(
            config.get("image_token_index") or config.get("image_token_id") or 100352
        )
        self.dtype = dtype
        self.model = VisionModel(self.config.vision_config)
        self.layerwise_projectors = [
            WindowQFormerDownsampler(self.config)
            for _ in (self.config.deepstack_layer_map or [])
        ]
        self.spatial_projectors = (
            [WindowQFormerDownsampler(self.config, spatial_offset=i) for i in range(4)]
            if self.config.use_spatial_sampling
            else []
        )
        hidden = self.config.text_config.hidden_size
        self.image_newline = (
            mx.random.normal((hidden,)) / math.sqrt(hidden)
            if self.config.use_image_newline_parameter
            else None
        )
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    # -- weights ---------------------------------------------------------------

    def load(self, model_path: Path) -> None:
        prefixes = tuple(f"{root}{part}" for root in ("model.", "") for part in PARTS)
        raw = load_prefixed(model_path, prefixes)
        if not raw:
            raise FileNotFoundError(f"{model_path}: no vision tower in the index")
        raw = {
            (k[len("model.") :] if k.startswith("model.") else k): v
            for k, v in raw.items()
        }
        raw = {k: v for k, v in raw.items() if "position_ids" not in k}
        if any(k.endswith(".scales") for k in raw):
            raise ValueError("a quantized vision tower is not supported yet")
        cast = lambda d: [(k, v.astype(self.dtype)) for k, v in d.items()]  # noqa: E731
        tower = self.model.sanitize(strip_prefix(raw, "vision_tower."))
        self.model.load_weights(cast(tower), strict=True)
        for i, proj in enumerate(self.layerwise_projectors):
            proj.load_weights(
                cast(strip_prefix(raw, f"layerwise_projectors.{i}.")), strict=True
            )
        for i, proj in enumerate(self.spatial_projectors):
            proj.load_weights(
                cast(strip_prefix(raw, f"spatial_projectors.{i}.")), strict=True
            )
        if self.image_newline is not None and "image_newline" in raw:
            self.image_newline = raw["image_newline"].astype(self.dtype)
        mx.eval(
            self.model.parameters(),
            *(p.parameters() for p in self.layerwise_projectors),
            *(p.parameters() for p in self.spatial_projectors),
        )
        self.loaded_from = sorted({k.split("/")[0] for k in raw})

    # -- one image -------------------------------------------------------------

    def _pack(self, projected: mx.array, image_size) -> mx.array:
        """A projector's output for one image's tiles (tiles, tokens, hidden)
        packed as the reference does: the base view first, the tile grid
        laid out as one picture, unpadded, a newline feature per row."""
        cfg = self.config
        if projected.shape[0] == 1:
            feature = projected[0]
            if self.image_newline is not None:
                feature = mx.concatenate([feature, self.image_newline[None]], axis=0)
            return feature
        base, tiles = projected[0], projected[1:]
        side = int(
            (cfg.vision_config.image_size // cfg.vision_config.patch_size)
            * Fraction(cfg.downsample_rate)
        )
        nph, npw = grid_shape(
            image_size, cfg.image_grid_pinpoints, cfg.vision_config.image_size
        )
        grid = tiles.reshape(nph, npw, side, side, -1)
        grid = mx.transpose(grid, axes=(4, 0, 2, 1, 3))
        channels = grid.shape[0]
        grid = grid.reshape(channels, nph * side, npw * side)
        grid = unpad(grid, image_size)
        if self.image_newline is not None:
            newline = mx.broadcast_to(
                self.image_newline[:, None, None], (channels, grid.shape[1], 1)
            )
            grid = mx.concatenate([grid, newline], axis=-1)
        return mx.concatenate([base, grid.reshape(channels, -1).T], axis=0)

    def encode(self, tiles: list[mx.array], image_sizes: list) -> list[Encoded]:
        """Per image its tiles (n, C, H, W) and its real (height, width):
        zero features at the placeholders, per-layer extras for the text
        layers the projectors target."""
        cfg = self.config
        out = []
        for image, size in zip(tiles, image_sizes, strict=True):
            pv = image.astype(self.dtype).transpose(0, 2, 3, 1)
            *_, states = self.model(pv, output_hidden_states=True)
            extras: dict[int, mx.array] = {}
            for i, (vision_layer, text_layer) in enumerate(
                cfg.deepstack_layer_map or []
            ):
                selected = states[vision_layer]
                if cfg.vision_feature_select_strategy == "default":
                    selected = selected[:, 1:]
                extras[int(text_layer)] = self._pack(
                    self.layerwise_projectors[i](selected), size
                )
            if self.spatial_projectors and cfg.spatial_target_layers:
                spatial = states[cfg.spatial_vision_layer]
                if cfg.vision_feature_select_strategy == "default":
                    spatial = spatial[:, 1:]
                for group, text_layer in enumerate(cfg.spatial_target_layers):
                    extras[int(text_layer)] = self._pack(
                        self.spatial_projectors[group](spatial), size
                    )
            n = next(iter(extras.values())).shape[0]
            out.append(
                Encoded(
                    mx.zeros((n, cfg.text_config.hidden_size), dtype=self.dtype), extras
                )
            )
        return out

    def describe(self) -> dict:
        return {
            "family": NAME,
            "hidden_size": self.config.vision_config.hidden_size,
            "patch_size": self.config.vision_config.patch_size,
            "downsample_rate": self.config.downsample_rate,
            "deepstack_layers": [
                int(t) for _, t in (self.config.deepstack_layer_map or [])
            ],
            "spatial_layers": list(self.config.spatial_target_layers or []),
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
        }


def build(config: dict, model_path: Path | None, dtype=mx.bfloat16) -> Tower:
    return Tower(config, model_path, dtype)


def pixel_inputs(processed: dict):
    """(tiles per image, real sizes) from what a LLaVA-NeXT-style processor
    returned: `pixel_values` (images, tiles, C, H, W) or a list of them, and
    `image_sizes`."""
    pv = processed["pixel_values"]
    images = [mx.array(p) for p in pv]
    sizes = [(int(h), int(w)) for h, w in processed["image_sizes"]]
    return images, sizes


__all__ = ["NAME", "Tower", "build", "pixel_inputs", "nn"]
