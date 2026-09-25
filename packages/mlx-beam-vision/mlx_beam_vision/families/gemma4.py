"""Gemma 4's tower (12B and E4B): a ViT over an aspect-ratio-preserving
resize with patch positions, bidirectional attention, a 3 x 3 pooler down
to the soft tokens, then `embed_vision` - an RMSNorm without scale and a
projection to the text width. Every image is one contiguous run of
`<image_soft_token>` placeholders. mlx-lm's text model scales its input
embeddings by sqrt(hidden) itself, so the projected features are handed
in divided by that scale and come out as the tower produced them.
Tower vendored from mlx-vlm, the embedder after its `gemma4.py`.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_beam_vision._vendor.mlx_vlm.gemma4.config import VisionConfig
from mlx_beam_vision._vendor.mlx_vlm.gemma4.vision import VisionModel
from mlx_beam_vision.families import Encoded
from mlx_beam_vision.weights import load_prefixed, strip_prefix

NAME = "gemma4"
TOWER_PREFIXES = ("model.vision_tower.", "vision_tower.")
EMBED_PREFIXES = ("model.embed_vision.", "embed_vision.")
# Clipping bounds the tower's checkpoints carry for its clipped linears;
# without use_clipped_linears the modules have no such parameters.
CLIP_KEYS = ("input_max", "input_min", "output_max", "output_min")


class Embedder(nn.Module):
    """RMSNorm without a learned scale, then the projection."""

    def __init__(self, dim: int, text_hidden: int, eps: float):
        super().__init__()
        self.eps = eps
        self.embedding_projection = nn.Linear(dim, text_hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding_projection(mx.fast.rms_norm(x, None, self.eps))


class Tower:
    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        self.config = VisionConfig.from_dict(config.get("vision_config") or {})
        text = config.get("text_config") or {}
        self.image_token_id = int(config.get("image_token_id", 258880))
        text_hidden = int(text.get("hidden_size", 2560))
        self.embed_scale = text_hidden**0.5
        self.dtype = dtype
        self.model = VisionModel(self.config)
        self.embedder = Embedder(self.config.hidden_size, text_hidden, self.config.rms_norm_eps)
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        raw = load_prefixed(model_path, TOWER_PREFIXES + EMBED_PREFIXES)
        tower = next((p for p in TOWER_PREFIXES if any(k.startswith(p) for k in raw)), None)
        embed = next((p for p in EMBED_PREFIXES if any(k.startswith(p) for k in raw)), None)
        if tower is None or embed is None:
            raise FileNotFoundError(f"{model_path}: no vision tower and embedder in the index")
        weights = strip_prefix(raw, tower)
        if not self.config.use_clipped_linears:
            weights = {k: v for k, v in weights.items() if not any(c in k for c in CLIP_KEYS)}
        weights = {k: v for k, v in weights.items() if "rotary_emb" not in k}
        for w in (weights, strip_prefix(raw, embed)):
            if any(k.endswith(".scales") for k in w):
                raise ValueError("a quantized vision tower is not supported yet")
        cast = lambda d: {k: v.astype(self.dtype) for k, v in d.items()}  # noqa: E731
        self.model.load_weights(list(cast(self.model.sanitize(weights)).items()), strict=True)
        self.embedder.load_weights(list(cast(strip_prefix(raw, embed)).items()), strict=True)
        mx.eval(self.model.parameters(), self.embedder.parameters())
        self.loaded_from = sorted({k.split("/")[0] for k in raw})

    def encode(self, images: list[mx.array], positions: list | None) -> list[Encoded]:
        """One tower pass per image (their sizes differ); `positions` are
        the processor's patch positions when it gives them, else the tower
        lays the patches out itself from the image's size."""
        out = []
        for i, image in enumerate(images):
            pos = None if positions is None else positions[i]
            pv = image.astype(self.dtype)
            if pos is None and pv.ndim == 3:
                pv = pv[None]
            hidden = self.model(pv, pos)[0]  # (soft tokens, hidden)
            out.append(Encoded(self.embedder(hidden) / self.embed_scale))
        return out

    def describe(self) -> dict:
        return {
            "family": NAME,
            "hidden_size": self.config.hidden_size,
            "patch_size": self.config.patch_size,
            "pooling_kernel_size": self.config.pooling_kernel_size,
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
        }


def build(config: dict, model_path: Path | None, dtype=mx.bfloat16) -> Tower:
    return Tower(config, model_path, dtype)


def pixel_inputs(processed: dict):
    """(images, patch positions or None) from what the processor returned:
    a stacked (N, C, H, W) array or a list of images of their own sizes,
    with `pixel_position_ids` when the processor patchified."""
    pv = processed["pixel_values"]
    images = [mx.array(p) for p in pv] if isinstance(pv, list) else [mx.array(p) for p in pv]
    pos = processed.get("pixel_position_ids")
    positions = None if pos is None else [mx.array(p) for p in pos]
    return images, positions
