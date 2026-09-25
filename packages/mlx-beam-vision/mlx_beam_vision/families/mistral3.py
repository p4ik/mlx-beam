"""Mistral 3's tower: Pixtral (a ViT over the whole image at its own
aspect ratio, 2D rotary embeddings, one block-diagonal attention over the
images of a call), then Mistral 3's projector - an RMSNorm, a patch merger
that folds 2 x 2 patches into one feature, two linears with a GELU
between. The placeholders are `[IMG]` tokens laid out in rows with an
`[IMG_BREAK]` after each row, so one image spans several runs; the
frontend deals the features out along the placeholder positions.
Tower vendored from mlx-vlm, projector after its `mistral3.py` (the
merger's unfold written for the token grid directly).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_beam_vision._vendor.mlx_vlm.pixtral.config import VisionConfig
from mlx_beam_vision._vendor.mlx_vlm.pixtral.vision import VisionModel
from mlx_beam_vision.families import Encoded
from mlx_beam_vision.weights import load_prefixed, strip_prefix

NAME = "mistral3"
TOWER_PREFIXES = ("vision_tower.", "model.vision_tower.")
PROJECTOR_PREFIXES = ("multi_modal_projector.", "model.multi_modal_projector.")


class PatchMerger(nn.Module):
    """2 x 2 patches of the token grid folded into one feature of four
    times the width, then projected back; the order of the four is the
    row-major order of the block, as unfold gives it."""

    def __init__(self, hidden: int, merge: int, patch: int):
        super().__init__()
        self.merge = merge
        self.patch = patch
        self.merging_layer = nn.Linear(hidden * merge**2, hidden, bias=False)

    def __call__(self, features: mx.array, sizes: list[tuple[int, int]]) -> mx.array:
        d = features.shape[-1]
        m = self.merge
        out = []
        at = 0
        for height, width in sizes:
            h, w = height // self.patch, width // self.patch
            grid = features[at : at + h * w].reshape(h, w, d)
            at += h * w
            hm, wm = h // m, w // m
            blocks = grid[: hm * m, : wm * m].reshape(hm, m, wm, m, d)
            # unfold's order: channel first, then the block's rows and columns.
            blocks = blocks.transpose(0, 2, 4, 1, 3).reshape(hm * wm, d * m * m)
            out.append(blocks)
        return self.merging_layer(mx.concatenate(out, axis=0))


class Projector(nn.Module):
    def __init__(
        self,
        vision_hidden: int,
        text_hidden: int,
        eps: float,
        merge: int,
        patch: int,
        bias: bool,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(vision_hidden, eps=eps)
        self.patch_merger = PatchMerger(vision_hidden, merge, patch)
        self.linear_1 = nn.Linear(vision_hidden, text_hidden, bias=bias)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden, text_hidden, bias=bias)

    def __call__(self, x: mx.array, sizes: list[tuple[int, int]]) -> mx.array:
        x = self.patch_merger(self.norm(x), sizes)
        return self.linear_2(self.gelu(self.linear_1(x)))


class Tower:
    def __init__(self, config: dict, model_path: Path | None, dtype=mx.bfloat16):
        vc = dict(config.get("vision_config") or {})
        vc.setdefault("model_type", "pixtral")
        self.config = VisionConfig.from_dict(vc)
        text = config.get("text_config") or {}
        self.image_token_id = int(
            config.get("image_token_id") or config.get("image_token_index") or 10
        )
        self.merge = int(config.get("spatial_merge_size", 2))
        self.feature_layer = int(config.get("vision_feature_layer", -1))
        self.dtype = dtype
        self.model = VisionModel(self.config)
        self.projector = Projector(
            self.config.hidden_size,
            int(text.get("hidden_size", 5120)),
            float(text.get("rms_norm_eps", 1e-5)),
            self.merge,
            self.config.patch_size,
            bool(config.get("multimodal_projector_bias", False)),
        )
        self.loaded_from: list[str] = []
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        raw = load_prefixed(model_path, TOWER_PREFIXES + PROJECTOR_PREFIXES)
        tower = next(
            (p for p in TOWER_PREFIXES if any(k.startswith(p) for k in raw)), None
        )
        proj = next(
            (p for p in PROJECTOR_PREFIXES if any(k.startswith(p) for k in raw)), None
        )
        if tower is None or proj is None:
            raise FileNotFoundError(
                f"{model_path}: no vision tower and projector in the index"
            )
        weights = self.model.sanitize(strip_prefix(raw, tower))
        for w in (weights, strip_prefix(raw, proj)):
            if any(k.endswith(".scales") for k in w):
                raise ValueError("a quantized vision tower is not supported yet")
        cast = lambda d: {k: v.astype(self.dtype) for k, v in d.items()}  # noqa: E731
        self.model.load_weights(list(cast(weights).items()), strict=True)
        self.projector.load_weights(
            list(cast(strip_prefix(raw, proj)).items()), strict=True
        )
        mx.eval(self.model.parameters(), self.projector.parameters())
        self.loaded_from = sorted({k.split("/")[0] for k in raw})

    def encode(
        self, pixel_values: mx.array, image_sizes: list[tuple[int, int]]
    ) -> list[Encoded]:
        """`pixel_values` (N, H, W, C) padded to one size, `image_sizes` the
        real (height, width) of each; per image its merged features."""
        _, states = self.model(
            pixel_values.astype(self.dtype),
            output_hidden_states=True,
            image_sizes=image_sizes,
        )
        selected = states[self.feature_layer][0]
        merged = self.projector(selected, image_sizes)
        out = []
        at = 0
        for height, width in image_sizes:
            n = (height // self.config.patch_size // self.merge) * (
                width // self.config.patch_size // self.merge
            )
            out.append(Encoded(merged[at : at + n]))
            at += n
        return out

    def describe(self) -> dict:
        return {
            "family": NAME,
            "hidden_size": self.config.hidden_size,
            "patch_size": self.config.patch_size,
            "spatial_merge_size": self.merge,
            "dtype": str(self.dtype),
            "loaded_from": self.loaded_from,
        }


def build(config: dict, model_path: Path | None, dtype=mx.bfloat16) -> Tower:
    return Tower(config, model_path, dtype)


def pixel_inputs(processed: dict):
    """(pixel values as (N, H, W, C), the real sizes) from what the
    processor returned: transformers gives (N, C, H, W) and `image_sizes`."""
    pv = mx.array(processed["pixel_values"])
    if pv.ndim == 3:
        pv = pv[None]
    pv = pv.transpose(0, 2, 3, 1)
    sizes = [(int(h), int(w)) for h, w in processed["image_sizes"]]
    return pv, sizes
