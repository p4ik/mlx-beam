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

from fractions import Fraction
from pathlib import Path

import mlx.core as mx
from mlx_beam_vision._vendor.mlx_vlm.granite4_vision.config import ModelConfig
from mlx_beam_vision._vendor.mlx_vlm.granite4_vision.downsampling import (
    WindowQFormerDownsampler,
)
from mlx_beam_vision._vendor.mlx_vlm.granite4_vision.vision import VisionModel
from mlx_beam_vision.families import Encoded, Loaded, cast, list_count, list_select

NAME = "granite4_vision"
PARTS = ("vision_tower", "layerwise_projectors", "spatial_projectors", "image_newline")


def grid_shape(image_size, pinpoints, patch: int) -> tuple[int, int]:
    from transformers.image_processing_utils import select_best_resolution

    height, width = select_best_resolution(list(image_size), pinpoints)
    return height // patch, width // patch


def unpad(tensor: mx.array, original_size) -> mx.array:
    """(C, H, W) cut back to the image's aspect ratio. The scaled edge is
    rounded to seven places before it is truncated, as the reference
    does: `int(3.9999999)` would drop a row the reference keeps."""
    oh, ow = original_size
    ch, cw = tensor.shape[1], tensor.shape[2]
    if ow / oh > cw / ch:
        new_h = int(round(oh * (cw / ow), 7))
        pad = (ch - new_h) // 2
        return tensor[:, pad : ch - pad, :]
    new_w = int(round(ow * (ch / oh), 7))
    pad = (cw - new_w) // 2
    return tensor[:, :, pad : cw - pad]


def per_layer(config: dict) -> bool:
    return True  # every projector's features enter ahead of a text layer


class Tower:
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
            if self.config.use_spatial_sampling and self.config.spatial_target_layers
            else []
        )
        if not self.layerwise_projectors and not self.spatial_projectors:
            raise ValueError(
                "the config names no text layer for the projectors "
                "(deepstack_layer_map, spatial_target_layers); nothing would "
                "carry the image"
            )
        # A learned feature per row of the tile grid; the checkpoint must
        # bring it, a random one would be a silent wrong answer.
        self.image_newline: mx.array | None = None
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    # -- weights ---------------------------------------------------------------

    def load(self, model_path: Path) -> None:
        loaded = Loaded(model_path, PARTS)
        tower = {
            k: v
            for k, v in loaded.part("vision_tower").items()
            if "position_ids" not in k
        }
        self.model.load_weights(
            cast(self.model.sanitize(tower), self.dtype), strict=True
        )
        for i, proj in enumerate(self.layerwise_projectors):
            proj.load_weights(
                cast(loaded.part(f"layerwise_projectors.{i}"), self.dtype), strict=True
            )
        for i, proj in enumerate(self.spatial_projectors):
            proj.load_weights(
                cast(loaded.part(f"spatial_projectors.{i}"), self.dtype), strict=True
            )
        if self.config.use_image_newline_parameter:
            newline = loaded.tensor("image_newline")
            if newline is None:
                raise FileNotFoundError(
                    f"{model_path}: the config asks for image_newline, the "
                    "checkpoint has none"
                )
            self.image_newline = newline.astype(self.dtype)
        mx.eval(
            self.model.parameters(),
            *(p.parameters() for p in self.layerwise_projectors),
            *(p.parameters() for p in self.spatial_projectors),
            *([self.image_newline] if self.image_newline is not None else []),
        )
        self.loaded_from = loaded.shards

    # -- one image -------------------------------------------------------------

    def tiles_of(self, image_size) -> int:
        """How many tiles the processor cut this image into: the base view
        plus the grid of its best resolution."""
        cfg = self.config
        nph, npw = grid_shape(
            image_size, cfg.image_grid_pinpoints, cfg.vision_config.image_size
        )
        return 1 + nph * npw

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
        layers the projectors target. A processor pads every image to
        the largest tile count of the call; only the image's own tiles
        are read."""
        cfg = self.config
        out = []
        for image, size in zip(tiles, image_sizes, strict=True):
            pv = image[: self.tiles_of(size)].astype(self.dtype).transpose(0, 2, 3, 1)
            *_, states = self.model(pv, output_hidden_states=True)
            extras: dict[int, mx.array] = {}
            packed: list[tuple[int, mx.array]] = []
            for i, (vision_layer, text_layer) in enumerate(
                cfg.deepstack_layer_map or []
            ):
                selected = states[vision_layer]
                if cfg.vision_feature_select_strategy == "default":
                    selected = selected[:, 1:]
                packed.append(
                    (
                        int(text_layer),
                        self._pack(self.layerwise_projectors[i](selected), size),
                    )
                )
            if self.spatial_projectors:
                spatial = states[cfg.spatial_vision_layer]
                if cfg.vision_feature_select_strategy == "default":
                    spatial = spatial[:, 1:]
                for group, text_layer in enumerate(cfg.spatial_target_layers):
                    packed.append(
                        (
                            int(text_layer),
                            self._pack(self.spatial_projectors[group](spatial), size),
                        )
                    )
            for text_layer, features in packed:
                # Two projectors aimed at one layer both add there.
                prior = extras.get(text_layer)
                extras[text_layer] = features if prior is None else prior + features
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


def pixel_inputs(processed: dict):
    """(tiles per image, real sizes) from what a LLaVA-NeXT-style processor
    returned: `pixel_values` (images, tiles, C, H, W) or a list of them, and
    `image_sizes`."""
    pv = processed["pixel_values"]
    images = [mx.array(p) for p in pv]
    sizes = [(int(h), int(w)) for h, w in processed["image_sizes"]]
    return images, sizes


select = list_select
count = list_count
