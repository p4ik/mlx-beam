# Copyright © 2025 Apple Inc.
# The text-only view of a Granite Vision 4.1 checkpoint, after mlx-lm's
# qwen3_vl.py and mistral3.py: the language model one level down under
# `language_model`, the tower and the projectors dropped at load. The
# tower lives in the vision package; VENDORED.md, granite4_vision.

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import granite, granitemoehybrid
from .base import BaseModelArgs
from .cache import KVCache

# What the vision package loads itself and the text view drops.
TOWER_PARTS = (
    "vision_tower",
    "layerwise_projectors",
    "spatial_projectors",
    "image_newline",
    "multi_modal_projector",
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # Granite 4 comes dense (granite) or as the Mamba-2 hybrid
        # (granitemoehybrid); the text config says which.
        text_type = args.text_config.get("model_type", "granitemoehybrid")
        module = granite if text_type == "granite" else granitemoehybrid
        self.language_model = module.Model(module.ModelArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        layer_hook=None,
    ):
        return self.language_model(
            inputs, cache=cache, input_embeddings=input_embeddings, layer_hook=layer_hook
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        scopes = [weights]
        if isinstance(weights.get("model"), dict):
            scopes.append(weights["model"])
        for scope in scopes:
            for part in TOWER_PARTS:
                scope.pop(part, None)
        # HF checkpoints nest the language model under a top-level "model".
        if isinstance(weights.get("model"), dict):
            if language_model := weights["model"].get("language_model"):
                weights["model"] = language_model
        inner = {}
        for key, value in tree_flatten(weights):
            if key.startswith("language_model."):
                key = key[len("language_model.") :]
            inner[key] = value
        if hasattr(self.language_model, "sanitize"):
            inner = self.language_model.sanitize(inner)
        return {"language_model." + k: v for k, v in inner.items()}

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        # The hybrid needs its Mamba caches; the dense one the default.
        if hasattr(self.language_model, "make_cache"):
            return self.language_model.make_cache()
        return [KVCache() for _ in self.layers]
