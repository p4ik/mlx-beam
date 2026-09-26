"""The frontend mlx-beam calls for a request with images: the checkpoint's
own processor renders the chat template and expands every image part into
its placeholder tokens; the tower encodes the images (once per digest,
then from the cache); the placeholder runs become the image spans the
engine's prefill embeds. Nothing here touches the language model."""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
from mlx_beam_vision import families
from mlx_beam_vision.cache import FeatureCache

from mlx_beam.engine.request import ImageSpan
from mlx_beam.modalities import Built, Image

logger = logging.getLogger("beam.vision")

# The largest image decoded, in pixels. The tower runs in the request's
# thread, outside the engine's prefill valve, and PIL's own guard starts
# at ~89 MP; Qwen's processors take up to ~16.7 MP. Decided 2026-09-25;
# a larger image is a 400, not a stall of the whole server.
MAX_IMAGE_PIXELS = 32_000_000


def _spans_from(
    ids: list[int], token: int, counts: list[int]
) -> list[list[tuple[int, int]]]:
    """Per image, the runs [start, end) of its placeholders: the positions
    holding `token`, in order, dealt out by the images' token counts. A
    layout that breaks an image's row with another token (Mistral's
    `[IMG_BREAK]`) gives an image several runs."""
    positions = [i for i, t in enumerate(ids) if t == token]
    if len(positions) != sum(counts):
        raise ValueError(
            f"the template placed {len(positions)} image placeholders for "
            f"{sum(counts)} image tokens ({len(counts)} images)"
        )
    spans = []
    at = 0
    for n in counts:
        chunk = positions[at : at + n]
        at += n
        runs = []
        for pos in chunk:
            if runs and runs[-1][1] == pos:
                runs[-1][1] = pos + 1
            else:
                runs.append([pos, pos + 1])
        spans.append([tuple(r) for r in runs])
    return spans


def open_images(images: Sequence[Image]):
    """PIL images from the request's bytes: size checked before a pixel
    is decoded, EXIF orientation applied (a phone's photo is stored as
    the sensor saw it), RGB."""
    from PIL import Image as PILImage
    from PIL import ImageOps

    out = []
    for img in images:
        try:
            pil = PILImage.open(io.BytesIO(img.data))
            w, h = pil.size
            if w * h > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"{w} x {h} pixels is above the {MAX_IMAGE_PIXELS} this "
                    "server decodes; scale the image down"
                )
            pil = ImageOps.exif_transpose(pil)
            pil.load()
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 - the bytes are the client's
            raise ValueError(f"not a readable image ({e})") from None
        out.append(pil.convert("RGB"))
    return out


class VisionFrontend:
    """`processor` is the checkpoint's transformers processor (its
    `apply_chat_template` and `__call__`), `tower` a family's Tower.
    `chat_template` set by the server overrides the processor's own
    (the `--chat-template` flag reaches image requests too)."""

    name = "mlx-beam-vision"

    def __init__(self, processor, tower, family: str, cache_bytes: int = 512 << 20):
        self.processor = processor
        self.tower = tower
        self.family = family
        self.cache = FeatureCache(cache_bytes)
        self.chat_template: str | None = None
        self.images_encoded = 0
        self._count = threading.Lock()

    @property
    def needs(self) -> tuple[str, ...]:
        """What the prefill must take of the text model for this tower's
        spans: the features at the placeholder positions always, the
        per-layer extras when the family adds them (DeepStack)."""
        if getattr(self.tower, "per_layer", False):
            return ("input_embeddings", "layer_hook")
        return ("input_embeddings",)

    def _text_kwargs(self, text: str) -> dict:
        # The rendered template already carries the BOS the tokenizer would
        # add; adding it again would start the prompt with two.
        tok = getattr(self.processor, "tokenizer", None)
        bos = getattr(tok, "bos_token", None)
        if bos and isinstance(text, str) and text.startswith(bos):
            return {"add_special_tokens": False}
        return {}

    def build(self, messages, images: Sequence[Image], template_kwargs: dict) -> Built:
        kwargs = dict(template_kwargs)
        if self.chat_template is not None:
            kwargs["chat_template"] = self.chat_template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        pil = open_images(images)
        processed = self.processor(
            text=[text],
            images=pil or None,
            return_tensors="np",
            **self._text_kwargs(text),
        )
        ids = [int(t) for t in processed["input_ids"][0]]
        encoded = self._encode(images, processed)
        counts = [int(enc.features.shape[0]) for enc in encoded]
        spans = []
        for runs, image, enc in zip(
            _spans_from(ids, self.tower.image_token_id, counts),
            images,
            encoded,
            strict=True,
        ):
            off = 0
            for start, end in runs:
                n = end - start
                spans.append(
                    ImageSpan(
                        start,
                        end,
                        enc.features[off : off + n],
                        image.digest,
                        {k: d[off : off + n] for k, d in enc.deepstack.items()},
                    )
                )
                off += n
        return Built(ids, spans, self._assistant_start(messages, kwargs, text, ids))

    def _assistant_start(self, messages, kwargs: dict, text, ids: list[int]) -> int:
        """Where the generation prompt begins in `ids`: the template
        rendered without it is a prefix of `text`, and the frame that
        follows tokenizes on its own (it opens with a marker or a line
        break). No processor call, no image: the frame's ids are matched
        against the prompt's tail. A template that renders the same either
        way puts the start at the end (nothing of the prompt is the
        assistant's, as the text path reads it); 0 - the whole prompt - is
        the answer only when the template will not render or the frame's
        ids do not match."""
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None or not isinstance(text, str):
            return 0
        try:
            without = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            )
        except Exception:  # noqa: BLE001 - the template's business
            return 0
        if not isinstance(without, str) or not text.startswith(without):
            return 0
        suffix = text[len(without) :]
        if not suffix:
            return len(ids)
        frame = tok.encode(suffix, add_special_tokens=False)
        n = len(frame)
        if not n or n > len(ids) or ids[-n:] != [int(t) for t in frame]:
            return 0
        return len(ids) - n

    def _encode(self, images: Sequence[Image], processed: dict) -> list:
        """Every image's features: from the cache by digest, else one tower
        pass over the ones missing (the family picks them out of the
        processor's batch), evaluated and cached as arrays of their own."""
        family = families.module_for(self.family)
        keys = [(img.digest, self.family) for img in images]
        found = [self.cache.get(k) for k in keys]
        missing = [i for i, f in enumerate(found) if f is None]
        if missing:
            inputs = family.pixel_inputs(processed)
            if family.count(inputs) != len(images):
                raise ValueError("the processor's grids do not match the images")
            if len(missing) < len(images):
                inputs = family.select(inputs, missing)
            fresh = self.tower.encode(*inputs)
            for i, enc in zip(missing, fresh, strict=True):
                # A slice of the call's batch would keep the whole batch
                # alive in the cache and count only its own bytes.
                enc.features = mx.contiguous(enc.features)
                enc.deepstack = {k: mx.contiguous(d) for k, d in enc.deepstack.items()}
                mx.eval(enc.features, *enc.deepstack.values())
                self.cache.put(keys[i], enc, [enc.features, *enc.deepstack.values()])
                found[i] = enc
            with self._count:
                self.images_encoded += len(missing)
        return found

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "family": self.family,
            "tower": self.tower.describe(),
            "processor": type(self.processor).__name__,
            "chat_template": (
                "server" if self.chat_template is not None else "processor"
            ),
            "feature_cache": self.cache.describe(),
            "images_encoded": self.images_encoded,
        }


def load_processor(model_path: Path, trust_remote_code: bool = False):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=trust_remote_code
    )
    # A checkpoint without preprocessor_config.json gets a bare tokenizer
    # back from AutoProcessor: that is no image path, and saying so here
    # beats a KeyError on the first image.
    if getattr(processor, "image_processor", None) is None:
        raise ValueError(
            f"{type(processor).__name__} has no image processor; the checkpoint "
            "ships no preprocessor_config.json, so images cannot be prepared"
        )
    return processor
