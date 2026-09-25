"""The entry point mlx-beam finds (group `mlx_beam.modalities`): serves a
checkpoint whose model type has a tower family here and whose config
carries a vision_config."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from mlx_beam_vision.families import FAMILIES
from mlx_beam_vision.frontend import VisionFrontend, load_processor

logger = logging.getLogger("beam.vision")


def family_for(config: dict) -> str | None:
    if not config.get("vision_config"):
        return None
    return FAMILIES.get(config.get("model_type", ""))


def supports(config: dict) -> bool:
    return family_for(config) is not None


def load(model, model_path, config: dict, tokenizer) -> VisionFrontend:
    family = family_for(config)
    if family is None:
        raise ValueError(f"no tower family for model type {config.get('model_type')!r}")
    module = importlib.import_module(f"mlx_beam_vision.families.{family}")
    tower = module.build(config, Path(model_path))
    processor = load_processor(Path(model_path))
    return VisionFrontend(processor, tower, family)
