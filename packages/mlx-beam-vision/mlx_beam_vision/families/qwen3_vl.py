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
from mlx_beam_vision._vendor.mlx_vlm.qwen3_vl.config import VisionConfig
from mlx_beam_vision._vendor.mlx_vlm.qwen3_vl.vision import VisionModel
from mlx_beam_vision.families import (
    Encoded,
    Loaded,
    cast,
    grid_count,
    grid_inputs,
    grid_select,
    split_by_grid,
)

NAME = "qwen3_vl"
# What the checkpoints call the tower: the HF layout of Qwen3-VL and of
# Qwen3.5 (`visual`), mlx-vlm's own conversions (`vision_tower`).
PARTS = ("visual", "vision_tower")


def positions(ids: list[int], image_token_id: int, grids, merge: int):
    """The prompt's MRoPE positions (3, len(ids)) and the decode delta, as
    the reference's get_rope_index builds them for images: text tokens at
    one running position on all three axes; an image's tokens at the
    block's start plus (frame, row, column) of its merged grid, the text
    after it continuing from the block's largest position plus one. The
    delta is that continuation minus the token count: what the decode adds
    to its offsets."""
    import numpy as np

    out = np.zeros((3, len(ids)), dtype=np.int32)
    grids = [tuple(int(v) for v in g) for g in grids]
    start = 0  # the next free position
    i = 0
    image = 0
    while i < len(ids):
        if ids[i] != image_token_id:
            out[:, i] = start
            start += 1
            i += 1
            continue
        if image >= len(grids):
            raise ValueError("more image placeholders than images in the prompt")
        t, h, w = grids[image]
        h, w = h // merge, w // merge
        n = t * h * w
        if ids[i : i + n] != [image_token_id] * n:
            raise ValueError(f"image {image} needs {n} contiguous placeholders at {i}")
        frame = np.repeat(np.arange(t), h * w)
        row = np.tile(np.repeat(np.arange(h), w), t)
        col = np.tile(np.arange(w), t * h)
        out[0, i : i + n] = start + frame
        out[1, i : i + n] = start + row
        out[2, i : i + n] = start + col
        start = int(out[:, i : i + n].max()) + 1
        i += n
        image += 1
    if image != len(grids):
        raise ValueError("fewer image placeholders than images in the prompt")
    return out, start - len(ids)


def per_layer(config: dict) -> bool:
    """The extras ahead of certain text layers need the prefill's layer
    hook - when the checkpoint has DeepStack blocks at all."""
    return bool((config.get("vision_config") or {}).get("deepstack_visual_indexes"))


class Tower:
    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        vc = config.get("vision_config") or {}
        self.config = VisionConfig.from_dict(vc)
        self.per_layer = per_layer(config)
        self.image_token_id = int(config.get("image_token_id", 151655))
        self.model = VisionModel(self.config)
        self.dtype = dtype
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        loaded = Loaded(model_path, PARTS)
        weights = self.model.sanitize(loaded.part(*PARTS))
        self.model.load_weights(cast(weights, self.dtype), strict=True)
        mx.eval(self.model.parameters())
        self.loaded_from = loaded.shards

    def encode(self, pixel_values: mx.array, grid_thw: mx.array) -> list[Encoded]:
        """All images of one processor call at once (the tower takes them
        concatenated with their grids), split back per image."""
        features, deepstack = self.model(pixel_values.astype(self.dtype), grid_thw)
        merge = self.config.spatial_merge_size**2
        per_image = split_by_grid(features, grid_thw, merge)
        # The tower's k-th DeepStack feature goes in after text layer k,
        # i.e. ahead of layer k + 1.
        extras = [split_by_grid(d, grid_thw, merge) for d in deepstack]
        return [
            Encoded(f, {k + 1: e[i] for k, e in enumerate(extras)})
            for i, f in enumerate(per_image)
        ]

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


pixel_inputs = grid_inputs
select = grid_select
count = grid_count
