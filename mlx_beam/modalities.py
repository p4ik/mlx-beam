"""Modalities beyond text come from separate packages; the core knows them
only through the entry-point group `mlx_beam.modalities` and this contract.

A provider module exposes `supports(config) -> bool` (the checkpoint's
config.json, plus what it ships), `load(model, model_path, config,
tokenizer, *, trust_remote_code=False) -> Frontend`, and may expose
`needs(config) -> tuple[str, ...]` - what its frontend will ask of the
prefill, known before any weight is read. A frontend turns a request's
messages and images into the prompt's token ids and the image spans the
prefill embeds (`build`), and says what it is (`describe`). Nothing here
imports a frontend: the core without the package answers image input
with a 400 that names the extra, never with a guess.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
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

    @cached_property
    def digest(self) -> str:
        # Hashed once: a span per placeholder run reads it, Mistral has dozens.
        return hashlib.sha256(self.data).hexdigest()


@dataclass
class Built:
    """A frontend's result: the prompt's token ids with the placeholders
    expanded, the image spans by position, and where the assistant's own
    turn begins in the tokens (0 when the frontend cannot tell: the whole
    prompt is then searched for an open think block)."""

    tokens: list[int]
    spans: list[ImageSpan] = field(default_factory=list)
    assistant_start: int = 0
    # For a text model rotating with several position axes (Qwen's MRoPE):
    # the prompt's positions (3, len(tokens)) and the shift the decode
    # continues with (GenerationRequest.positions, rope_delta); None and 0
    # for a family whose text model reads one position per token.
    positions: Any = None
    rope_delta: int = 0


class Frontend(Protocol):
    name: str
    # What the prefill must take of the model for this frontend's spans:
    # "input_embeddings" always, "layer_hook" when a family adds per-layer
    # extras (DeepStack). A frontend without the attribute needs the first.
    needs: tuple[str, ...]
    # A chat template the server sets (its --chat-template flag) in place
    # of the processor's own; None keeps the checkpoint's.
    chat_template: str | None

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


def unmet_needs(model, needs: Sequence[str]) -> list[str]:
    """What of `needs` the model's text trunk does not take - read from
    the model, not promised by the family."""
    from mlx_beam.engine.priming import image_capabilities

    can = image_capabilities(model)
    return [name for name in needs if not can.get(name, False)]


def load_frontend(
    model, model_path, config: dict, tokenizer, *, trust_remote_code: bool = False
) -> tuple[Frontend | None, str | None]:
    """(frontend, reason): the first provider that supports this checkpoint
    and whose needs the model's trunk meets, loaded; else None and, when a
    provider was refused, why - the health carries it. No provider at all:
    (None, None), the core stays text-only and says so. A provider that
    states its needs up front is refused before its tower is read."""
    reason = None
    for provider in providers():
        name = getattr(provider, "__name__", str(provider))
        try:
            if not provider.supports(config):
                continue
            stated = getattr(provider, "needs", None)
            missing = unmet_needs(model, stated(config)) if stated else []
            if not missing:
                frontend = provider.load(
                    model,
                    model_path,
                    config,
                    tokenizer,
                    trust_remote_code=trust_remote_code,
                )
                missing = unmet_needs(
                    model, getattr(frontend, "needs", ("input_embeddings",))
                )
                name = frontend.name
        except Exception as e:  # noqa: BLE001 - one provider's failure is its own
            reason = f"{name}: {e}"
            logger.warning("modality provider %s", reason)
            continue
        if missing:
            reason = (
                f"{name} needs {', '.join(missing)} of the text model, "
                f"which {type(model).__name__} does not take"
            )
            logger.warning("modality provider refused: %s", reason)
            continue
        logger.info("modality: %s", frontend.describe())
        return frontend, None
    return None, reason
