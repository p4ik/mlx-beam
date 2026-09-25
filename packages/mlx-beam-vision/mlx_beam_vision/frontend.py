"""The frontend mlx-beam calls for a request with images: the checkpoint's
own processor renders the chat template and expands every image part into
its placeholder tokens; the tower encodes the images (once per digest,
then from the cache); the placeholder runs become the image spans the
engine's prefill embeds. Nothing here touches the language model."""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
from mlx_beam_vision.cache import FeatureCache

from mlx_beam.engine.request import ImageSpan
from mlx_beam.modalities import Built, Image

logger = logging.getLogger("beam.vision")


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
    from PIL import Image as PILImage

    out = []
    for img in images:
        try:
            pil = PILImage.open(io.BytesIO(img.data))
            pil.load()
        except Exception as e:  # noqa: BLE001 - the bytes are the client's
            raise ValueError(f"not a readable image ({e})") from None
        out.append(pil.convert("RGB"))
    return out


class VisionFrontend:
    """`processor` is the checkpoint's transformers processor (its
    `apply_chat_template` and `__call__`), `tower` a family's Tower."""

    name = "mlx-beam-vision"

    def __init__(self, processor, tower, family: str, cache_bytes: int = 512 << 20):
        self.processor = processor
        self.tower = tower
        self.family = family
        self.cache = FeatureCache(cache_bytes)
        self.images_encoded = 0

    def _placeholder(self) -> int:
        return self.tower.image_token_id

    def build(self, messages, images: Sequence[Image], template_kwargs: dict) -> Built:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        pil = open_images(images)
        processed = self.processor(text=[text], images=pil or None, return_tensors="np")
        ids = [int(t) for t in processed["input_ids"][0]]
        encoded = self._encode(images, processed)
        counts = [int(enc.features.shape[0]) for enc in encoded]
        spans = []
        for runs, image, enc in zip(
            _spans_from(ids, self._placeholder(), counts), images, encoded, strict=True
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
                        tuple(d[off : off + n] for d in enc.deepstack),
                    )
                )
                off += n
        return Built(ids, spans)

    def _encode(self, images: Sequence[Image], processed: dict) -> list:
        """Every image's features: from the cache by digest, else one tower
        pass over the ones missing (all at once), evaluated and cached."""
        family = self._family_module()
        keys = [(img.digest, self.family) for img in images]
        found = [self.cache.get(k) for k in keys]
        missing = [i for i, f in enumerate(found) if f is None]
        if missing:
            fresh = self.tower.encode(*family.pixel_inputs(processed))
            if len(fresh) != len(images):
                raise ValueError("the processor's grids do not match the images")
            for i in missing:
                enc = fresh[i]
                mx.eval(enc.features, *enc.deepstack)
                self.cache.put(keys[i], enc, [enc.features, *enc.deepstack])
                found[i] = enc
                self.images_encoded += 1
        return found

    def _family_module(self):
        import importlib

        return importlib.import_module(f"mlx_beam_vision.families.{self.family}")

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "family": self.family,
            "tower": self.tower.describe(),
            "processor": type(self.processor).__name__,
            "feature_cache": self.cache.describe(),
            "images_encoded": self.images_encoded,
        }


def load_processor(model_path: Path, trust_remote_code: bool = False):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(model_path), trust_remote_code=trust_remote_code
    )
