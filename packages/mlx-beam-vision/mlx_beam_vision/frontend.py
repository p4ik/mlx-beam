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

    def __init__(
        self,
        processor,
        tower,
        family: str,
        cache_bytes: int = 512 << 20,
        processor_info: dict | None = None,
    ):
        self.processor = processor
        self.tower = tower
        self.family = family
        self.processor_info = processor_info or {
            "source": "checkpoint",
            "class": type(processor).__name__,
        }
        self.cache = FeatureCache(cache_bytes)
        self.chat_template: str | None = None
        self.images_encoded = 0
        self._count = threading.Lock()

    @property
    def needs(self) -> tuple[str, ...]:
        """What the prefill must take of the text model for this tower's
        spans: the features at the placeholder positions always, the
        per-layer extras when the family adds them (DeepStack)."""
        out = ["input_embeddings"]
        if getattr(self.tower, "per_layer", False):
            out.append("layer_hook")
        if hasattr(families.module_for(self.family), "positions"):
            out.append("position_ids")
        return tuple(out)

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
        family = families.module_for(self.family)
        positions, delta = None, 0
        if hasattr(family, "positions"):
            positions, delta = family.positions(
                ids,
                self.tower.image_token_id,
                processed.get("image_grid_thw", []),
                self.tower.config.spatial_merge_size,
            )
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
        return Built(
            ids,
            spans,
            self._assistant_start(messages, kwargs, text, ids),
            positions=positions,
            rope_delta=delta,
        )

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
            # The checkpoint's own, or built from the model when it ships no
            # preprocessor_config.json - then with where each setting came from.
            "processor": self.processor_info,
            "chat_template": (
                "server" if self.chat_template is not None else "processor"
            ),
            # How the text model places image tokens: three axes (Qwen's
            # MRoPE, the frontend hands the positions in) or one per token.
            "positions": "mrope" if "position_ids" in self.needs else "sequential",
            "feature_cache": self.cache.describe(),
            "images_encoded": self.images_encoded,
        }


def load_processor(
    model_path: Path, config: dict | None = None, trust_remote_code: bool = False
) -> tuple[object, dict]:
    """The checkpoint's processor and where it came from. A checkpoint that
    ships no preprocessor_config.json (mlx-community's conversions leave it
    out) gets one built from what it does carry - see `build_processor`;
    without a config to build from, that checkpoint is refused here rather
    than with a KeyError on the first image."""
    from transformers import AutoProcessor

    shipped = any(
        (model_path / name).is_file()
        for name in ("preprocessor_config.json", "processor_config.json")
    )
    try:
        processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )
    except OSError:
        # "Can't load image processor": AutoProcessor knows the model type
        # and looked for the file; a missing directory would have failed
        # when the text model loaded.
        if shipped:
            raise
        processor = None
    if processor is not None and getattr(processor, "image_processor", None):
        return processor, {"source": "checkpoint", "class": type(processor).__name__}
    if config is None or shipped:
        what = "AutoProcessor" if processor is None else type(processor).__name__
        raise ValueError(
            f"{what} has no image processor; the checkpoint ships no "
            "preprocessor_config.json, so images cannot be prepared"
        )
    return build_processor(model_path, config, trust_remote_code)


def classes_for(model_type: str):
    """transformers' own (processor, image processor, video processor or
    None) for a model type; the image backend is torchvision when it is
    installed, PIL otherwise, as AutoImageProcessor picks it."""
    from transformers.models.auto.image_processing_auto import (
        IMAGE_PROCESSOR_MAPPING_NAMES,
        get_image_processor_class_from_name,
    )
    from transformers.models.auto.processing_auto import (
        PROCESSOR_MAPPING_NAMES,
        processor_class_from_name,
    )
    from transformers.models.auto.video_processing_auto import (
        VIDEO_PROCESSOR_MAPPING_NAMES,
        video_processor_class_from_name,
    )
    from transformers.utils import is_torchvision_available

    proc = PROCESSOR_MAPPING_NAMES.get(model_type)
    names = IMAGE_PROCESSOR_MAPPING_NAMES.get(model_type)
    if not proc or not names:
        raise ValueError(
            f"the checkpoint ships no preprocessor_config.json and transformers "
            f"has no processor for model type {model_type!r}, so images cannot "
            "be prepared"
        )
    if isinstance(names, dict):
        backend = "torchvision" if is_torchvision_available() else "pil"
        image = names.get(backend) or next(n for n in names.values() if n)
    else:
        image = names if isinstance(names, str) else next(n for n in names if n)
    video = VIDEO_PROCESSOR_MAPPING_NAMES.get(model_type)
    return (
        processor_class_from_name(proc),
        get_image_processor_class_from_name(image),
        video_processor_class_from_name(video) if video else None,
    )


class ModelValues:
    """What a checkpoint says about its own preprocessing, looked up by the
    names transformers' classes use: a key is read from vision_config, then
    config.json's top level, under its own name or the one the config uses
    for it (`merge_size` is `spatial_merge_size` there, `size` is
    `image_size`). Every hit is recorded with its origin."""

    ALIASES = {"merge_size": "spatial_merge_size", "size": "image_size"}

    def __init__(self, config: dict):
        self.config = config
        self.vision = config.get("vision_config") or {}
        self.origin: dict[str, str] = {}

    def find(self, key: str):
        for name in (key, self.ALIASES.get(key)):
            if name is None:
                continue
            for where, table in (
                ("vision_config", self.vision),
                ("config.json", self.config),
            ):
                if name in table and not isinstance(table[name], (dict, list)):
                    self.origin[key] = f"{where}.{name}"
                    return table[name]
        return None

    def fill(self, cls, keys, defaults: dict) -> dict:
        """The kwargs for `cls`: the model's value for every key it has one
        for, shaped like the class's default when that is a size dict
        (`{"longest_edge": 1024}` takes the model's image_size; a
        `{"height", "width"}` pair takes its patch size); the rest stays the
        class's default and is listed as such."""
        out = {}
        for key in keys:
            value = self.find(key)
            default = defaults.get(key)
            if value is None:
                continue
            if isinstance(default, dict) and not isinstance(value, dict):
                if len(default) == 1 or set(default) == {"height", "width"}:
                    value = {k: value for k in default}
                else:
                    self.origin.pop(key, None)
                    continue
            out[key] = value
        return out


def _defaults(cls) -> dict:
    """The class's own defaults for the keys its kwargs TypedDict names."""
    kwargs_type = getattr(cls, "valid_kwargs", None)
    keys = list(getattr(kwargs_type, "__annotations__", {}))
    return {k: getattr(cls, k) for k in keys if hasattr(cls, k)}


def build_processor(model_path: Path, config: dict, trust_remote_code: bool = False):
    """A processor for a checkpoint without preprocessor_config.json, from
    the model first and the class's defaults only where the model says
    nothing: the classes are transformers' own for the model type, the
    tokenizer and chat template the checkpoint's, sizes and merges from
    config.json, the image token the one config.json's id names (a token
    default the tokenizer does not know is refused). The second value says
    where each setting came from; it reaches /health.vision.processor."""
    import inspect

    from transformers import AutoTokenizer

    model_type = str(config.get("model_type", ""))
    processor_cls, image_cls, video_cls = classes_for(model_type)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=trust_remote_code
    )
    values = ModelValues(config)
    parts = {}
    defaults_used: dict[str, list[str]] = {}
    for name, cls in (("image_processor", image_cls), ("video_processor", video_cls)):
        if cls is None:
            continue
        defaults = _defaults(cls)
        kwargs = values.fill(cls, defaults, defaults)
        parts[name] = cls(**kwargs)
        defaults_used[cls.__name__] = sorted(k for k in defaults if k not in kwargs)
    attributes = set(processor_cls.get_attributes())
    parts = {k: v for k, v in parts.items() if k in attributes}
    extra = {}
    for name, param in inspect.signature(processor_cls.__init__).parameters.items():
        if name in attributes or name in ("self", "chat_template", "args", "kwargs"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = param.default if param.default is not param.empty else None
        if name.endswith("_token") and isinstance(default, str):
            token_id = (
                config.get(f"{name}_id") or config.get(f"{name}_index")
                if name == "image_token"
                else None
            )
            if token_id is not None:
                extra[name] = tokenizer.convert_ids_to_tokens(int(token_id))
                values.origin[name] = "config.json.image_token_id"
            elif tokenizer.convert_tokens_to_ids(default) in (
                None,
                tokenizer.unk_token_id,
            ):
                raise ValueError(
                    f"{processor_cls.__name__} expects the token {default!r}, "
                    "which the checkpoint's tokenizer does not know"
                )
            continue
        value = values.fill(processor_cls, [name], {name: default}).get(name)
        if value is not None:
            extra[name] = value
    processor = processor_cls(
        **parts, tokenizer=tokenizer, chat_template=tokenizer.chat_template, **extra
    )
    provenance = {
        "source": "built",
        "class": processor_cls.__name__,
        "from_model": dict(sorted(values.origin.items())),
        "defaults": defaults_used,
    }
    logger.info(
        "no preprocessor_config.json: %s built from the model (%s); %s",
        processor_cls.__name__,
        ", ".join(f"{k} from {v}" for k, v in sorted(values.origin.items())),
        "; ".join(
            f"{cls} defaults for {', '.join(keys)}"
            for cls, keys in defaults_used.items()
        ),
    )
    return processor, provenance
