"""One module per tower family. A family's `Tower` knows how to build
itself from the checkpoint's config and tensors, and how the processor's
pixel inputs (`pixel_inputs`) become the features of one image -
projected to the text model's width, with the per-layer extras a family
adds ahead of certain text layers (DeepStack) when it has them. What the
five share - reading the tower's shards, the prefix a layout uses, the
refusal of quantized towers, the cast, the split of a stacked output per
image - lives here."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import mlx.core as mx
from mlx_beam_vision.weights import load_prefixed


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
    "muse_glimmer": "muse_glimmer",
    "granite4_vision": "granite4_vision",
}

# Model types with images this package knows and does not serve, and why;
# the reason reaches /health.vision.refused instead of a wrong tower.
UNSERVED: dict[str, str] = {
    "gemma4_unified": (
        "encoder-free vision (vision_embedder) whose image tokens attend "
        "bidirectionally within their block in the text model's sliding "
        "layers; the text trunk applies no such mask yet"
    ),
}


def module_for(family: str) -> ModuleType:
    return importlib.import_module(f"mlx_beam_vision.families.{family}")


def needs(family: str, config: dict) -> tuple[str, ...]:
    """What the family's tower asks of the prefill for this checkpoint,
    read from the config alone: the features at the placeholder positions
    always, the per-layer extras when it adds them (DeepStack)."""
    module = module_for(family)
    out = ["input_embeddings"]
    if module.per_layer(config):
        out.append("layer_hook")
    if hasattr(module, "positions"):
        # The family's text model rotates with several position axes.
        out.append("position_ids")
    return tuple(out)


ROOTS = ("model.", "")
"""Where checkpoints keep a part: HF nests everything under `model.`,
mlx-vlm's conversions do not."""


class Loaded:
    """The tensors of the named parts (`vision_tower`, `embed_vision`, ...)
    under either root, the root stripped, and the shards they came from
    (`/health.vision.tower.loaded_from`). A quantized tower is refused."""

    def __init__(self, model_path: Path, parts: Iterable[str]):
        parts = tuple(parts)
        prefixes = tuple(f"{root}{part}" for root in ROOTS for part in parts)
        raw, self.shards = load_prefixed(model_path, prefixes)
        if not raw:
            raise FileNotFoundError(
                f"{model_path}: no tensors under {parts}; the checkpoint ships "
                "no vision tower the index knows"
            )
        if any(k.endswith(".scales") for k in raw):
            raise ValueError("a quantized vision tower is not supported yet")
        self.weights = {k.removeprefix("model."): v for k, v in raw.items()}

    def part(self, *names: str) -> dict[str, mx.array]:
        """The tensors under the first of `names` that has any, that name
        stripped; a part a layout may call two ways names both."""
        for name in names:
            found = {
                k[len(name) + 1 :]: v
                for k, v in self.weights.items()
                if k.startswith(name + ".")
            }
            if found:
                return found
        raise FileNotFoundError(f"no tensors under {names} in the checkpoint")

    def tensor(self, name: str) -> mx.array | None:
        return self.weights.get(name)


def cast(weights: dict[str, mx.array], dtype) -> list[tuple[str, mx.array]]:
    return [(k, v.astype(dtype)) for k, v in weights.items()]


def split_by_grid(features: mx.array, grid_thw: mx.array, merge: int) -> list[mx.array]:
    """A stacked (tokens, hidden) output cut per image: one image is
    t * h * w patches, merged `merge` at a time."""
    out = []
    at = 0
    for t, h, w in grid_thw.tolist():
        n = int(t * h * w) // merge
        out.append(features[at : at + n])
        at += n
    return out


def grid_inputs(processed: dict) -> tuple[mx.array, mx.array]:
    """(pixel_values, image_grid_thw) as the Qwen-style processors return
    them: the patches of every image concatenated, with their grids."""
    return mx.array(processed["pixel_values"]), mx.array(processed["image_grid_thw"])


def grid_select(inputs: tuple[mx.array, mx.array], keep: list[int]):
    """The images `keep` (indices) out of grid inputs: their patches and
    their grids, the rest left out of the tower pass."""
    pv, grid = inputs
    sizes = [int(t * h * w) for t, h, w in grid.tolist()]
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]
    return (
        mx.concatenate([pv[offsets[i] : offsets[i] + sizes[i]] for i in keep]),
        mx.stack([grid[i] for i in keep]),
    )


def grid_count(inputs: tuple[mx.array, mx.array]) -> int:
    return int(inputs[1].shape[0])


def list_select(inputs: tuple, keep: list[int]):
    """The images `keep` out of inputs given per image as lists (a None
    stays None)."""
    return tuple(None if part is None else [part[i] for i in keep] for part in inputs)


def list_count(inputs: tuple) -> int:
    return len(inputs[0])
