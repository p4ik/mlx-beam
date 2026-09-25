"""Modalities beyond text come from separate packages; the core knows them
only through the entry-point group `mlx_beam.modalities` and this contract.

A provider module exposes `supports(config) -> bool` (the checkpoint's
config.json, plus what it ships) and `load(model, model_path, config,
tokenizer) -> Frontend`. A frontend turns a request's messages and images
into the prompt's token ids and the image spans the prefill embeds
(`build`), and says what it is (`describe`). Nothing here imports a
frontend: the core without the package answers image input with a 400
that names the extra, never with a guess.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol

from mlx_beam.engine.request import ImageSpan

logger = logging.getLogger("beam.modalities")

ENTRY_POINT_GROUP = "mlx_beam.modalities"


@dataclass(frozen=True)
class Image:
    """An image as the request carried it: the bytes and their digest, the
    prefix cache's key for the span it becomes."""

    data: bytes
    media_type: str = "image/png"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class Built:
    """A frontend's result: the prompt's token ids with the placeholders
    expanded, and the image spans by position."""

    tokens: list[int]
    spans: list[ImageSpan] = field(default_factory=list)


class Frontend(Protocol):
    name: str
    # What the prefill must take of the model for this frontend's spans:
    # "input_embeddings" always, "layer_hook" when a family adds per-layer
    # extras (DeepStack). A frontend without the attribute needs the first.
    needs: tuple[str, ...]

    def build(
        self, messages: list[dict], images: Sequence[Image], template_kwargs: dict
    ) -> Built:
        """The prompt for `messages` (content lists with `{"type": "image"}`
        parts in the order of `images`), rendered by the model's own
        template and processor, the images encoded by the tower."""

    def describe(self) -> dict: ...


# Providers registered in-process (tests, embedding use); entry points
# come after them.
_registered: list[Any] = []


def register(provider: Any) -> None:
    _registered.append(provider)


def providers() -> list[Any]:
    found = list(_registered)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            found.append(ep.load())
        except Exception as e:  # noqa: BLE001 - a broken package is reported, not fatal
            logger.warning("modality provider %s failed to load: %s", ep.name, e)
    return found


def unmet_needs(model, frontend: Any) -> list[str]:
    """What this frontend needs of the prefill that the model's text trunk
    does not take - read from the model, not promised by the family."""
    from mlx_beam.engine.priming import image_capabilities

    can = image_capabilities(model)
    needs = getattr(frontend, "needs", ("input_embeddings",))
    return [name for name in needs if not can.get(name, False)]


def load_frontend(
    model, model_path, config: dict, tokenizer
) -> tuple[Frontend | None, str | None]:
    """(frontend, reason): the first provider that supports this checkpoint
    and whose needs the model's trunk meets, loaded; else None and, when a
    provider was refused, why - the health carries it. No provider at all:
    (None, None), the core stays text-only and says so."""
    reason = None
    for provider in providers():
        try:
            if not provider.supports(config):
                continue
            frontend = provider.load(model, model_path, config, tokenizer)
        except Exception as e:  # noqa: BLE001 - one provider's failure is its own
            reason = f"{getattr(provider, '__name__', provider)}: {e}"
            logger.warning("modality provider %s", reason)
            continue
        missing = unmet_needs(model, frontend)
        if missing:
            reason = (
                f"{frontend.name} needs {', '.join(missing)} of the text model, "
                f"which {type(model).__name__} does not take"
            )
            logger.warning("modality provider refused: %s", reason)
            continue
        logger.info("modality: %s", frontend.describe())
        return frontend, None
    return None, reason
