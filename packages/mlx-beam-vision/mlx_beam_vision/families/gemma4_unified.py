"""Gemma 4's encoder-free vision (the 12B and up): no tower. The
checkpoint's processor cuts an image into 48-pixel patches with their
(x, y) positions, the embedder projects each patch into the text width
(LayerNorm, Dense, LayerNorm, plus a factorized position embedding,
LayerNorm, then a scaleless RMSNorm and a Linear - transformers'
Gemma4UnifiedVisionEmbedder), and the text model lets an image's tokens
attend to each other both ways in its sliding layers: every span is a
block, and the engine hands the trunk the block ids.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_beam_vision.families import Encoded, Loaded

NAME = "gemma4_unified"
# The HF layout keeps everything under `embed_vision`; mlx-vlm's
# conversions split the patch embedder off as `vision_embedder`.
PARTS = ("embed_vision", "vision_embedder")
# An image's tokens form one block in the text model's sliding layers.
BLOCKS = True


def per_layer(config: dict) -> bool:
    return False


class Embedder(nn.Module):
    """Patches (N, P, patch_dim) with positions (N, P, 2), -1 for padding,
    to (N, P, mm_dim): the position embedding is factorized, one table per
    axis, summed; a padded patch gets none."""

    def __init__(self, patch_dim: int, mm_dim: int, posemb_size: int):
        super().__init__()
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_dim)
        self.patch_ln2 = nn.LayerNorm(mm_dim)
        self.pos_embedding = mx.zeros((posemb_size, 2, mm_dim))
        self.pos_norm = nn.LayerNorm(mm_dim)

    def __call__(self, patches: mx.array, positions: mx.array) -> mx.array:
        h = self.patch_ln2(self.patch_dense(self.patch_ln1(patches)))
        clamped = mx.maximum(positions, 0)
        valid = (positions != -1).astype(h.dtype)
        pos = (
            self.pos_embedding[clamped[..., 0], 0] * valid[..., 0:1]
            + self.pos_embedding[clamped[..., 1], 1] * valid[..., 1:2]
        )
        return self.pos_norm(h + pos)


class Projection(nn.Module):
    """The multimodal embedder: RMSNorm without a scale, then the Linear
    into the text width."""

    def __init__(self, dims: int, text_hidden: int, eps: float):
        super().__init__()
        self.eps = eps
        self.embedding_projection = nn.Linear(dims, text_hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding_projection(mx.fast.rms_norm(x, None, self.eps))


class Tower:
    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        vc = config.get("vision_config") or {}
        text = config.get("text_config") or {}
        self.patch = int(vc.get("model_patch_size", 48))
        self.mm_dim = int(vc.get("mm_embed_dim", 3840))
        self.image_token_id = int(config.get("image_token_id", 262145))
        self.per_layer = False
        self.embedder = Embedder(
            self.patch * self.patch * 3,
            self.mm_dim,
            int(vc.get("mm_posemb_size", 1120)),
        )
        self.projection = Projection(
            int(vc.get("output_proj_dims", self.mm_dim)),
            int(text.get("hidden_size", 3840)),
            float(vc.get("rms_norm_eps", 1e-6)),
        )
        self.dtype = dtype
        self.loaded_from: list[str] = []
        self.quantized = False
        self.checkpoint = config
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        loaded = Loaded(model_path, PARTS, self.checkpoint)
        # The projection sits under embed_vision in both layouts (HF one
        # level deeper, as multimodal_embedder); the patch embedder under
        # vision_embedder in mlx-vlm's, beside the projection in HF's.
        embed_vision = loaded.part("embed_vision")
        projection = {
            k.removeprefix("multimodal_embedder."): v
            for k, v in embed_vision.items()
            if "embedding_projection" in k
        }
        try:
            embedder = loaded.part("vision_embedder")
        except FileNotFoundError:
            embedder = {k: v for k, v in embed_vision.items() if k not in projection}
            embedder = {
                k: v for k, v in embedder.items() if not k.startswith("multimodal_")
            }
        loaded.fit(self.embedder, embedder, self.dtype)
        loaded.fit(self.projection, projection, self.dtype)
        mx.eval(self.embedder.parameters(), self.projection.parameters())
        self.loaded_from = loaded.shards
        self.quantized = loaded.quantized

    def encode(self, pixel_values: mx.array, positions: mx.array) -> list[Encoded]:
        """All images of one processor call: patches padded to the
        processor's budget, the real ones where the position is not -1."""
        features = self.projection(
            self.embedder(pixel_values.astype(self.dtype), positions)
        )
        out = []
        for i in range(features.shape[0]):
            keep = positions[i, :, 0] != -1
            n = int(keep.sum().item())
            # The processor pads at the end, so the real patches lead.
            out.append(Encoded(features[i, :n]))
        return out

    def describe(self) -> dict:
        return {
            "family": NAME,
            "encoder": "none",
            "patch": self.patch,
            "mm_embed_dim": self.mm_dim,
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
            "quantized": self.quantized,
        }


def pixel_inputs(processed: dict):
    """(patches (N, P, patch_dim), positions (N, P, 2)) as the processor
    returns them."""
    return mx.array(processed["pixel_values"]), mx.array(
        processed["image_position_ids"]
    )


def select(inputs, keep: list[int]):
    pv, pos = inputs
    idx = mx.array(keep)
    return pv[idx], pos[idx]


def count(inputs) -> int:
    return int(inputs[0].shape[0])
