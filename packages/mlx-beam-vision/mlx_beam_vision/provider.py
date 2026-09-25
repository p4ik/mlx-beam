"""The entry point mlx-beam finds (group `mlx_beam.modalities`): serves a
checkpoint whose model type has a tower family here and whose config
carries a vision_config; a type the package knows and cannot serve is
claimed and refused with its reason, so the health says why."""

from __future__ import annotations

import logging
from pathlib import Path

from mlx_beam_vision import families
from mlx_beam_vision.families import FAMILIES, UNSERVED
from mlx_beam_vision.frontend import VisionFrontend, load_processor

logger = logging.getLogger("beam.vision")


def family_for(config: dict) -> str | None:
    if not config.get("vision_config"):
        return None
    return FAMILIES.get(config.get("model_type", ""))


def supports(config: dict) -> bool:
    return family_for(config) is not None or config.get("model_type") in UNSERVED


def needs(config: dict) -> tuple[str, ...]:
    """What the family's tower asks of the prefill, known before any
    weight is read - so a trunk that cannot serve it is refused first."""
    family = family_for(config)
    if family is None:
        return ("input_embeddings",)
    return families.needs(family, config)


def load(model, model_path, config: dict, tokenizer, *, trust_remote_code=False):
    kind = config.get("model_type")
    if kind in UNSERVED:
        raise ValueError(f"{kind}: {UNSERVED[kind]}")
    family = family_for(config)
    if family is None:
        raise ValueError(f"no tower family for model type {kind!r}")
    tower = families.module_for(family).Tower(config, Path(model_path))
    processor = load_processor(Path(model_path), trust_remote_code)
    return VisionFrontend(processor, tower, family)
