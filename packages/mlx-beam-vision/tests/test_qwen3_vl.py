"""The Qwen3-VL family on a tiny tower: pixel inputs to features per image
with DeepStack extras, the frontend's spans from the processor's
placeholder runs, the feature cache by digest, the provider's dispatch,
and the whole path through mlx-beam's engine against a direct forward."""

import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx_beam_vision import frontend, provider
from mlx_beam_vision.families import qwen3_vl
from mlx_beam_vision.frontend import VisionFrontend

from mlx_beam import modalities
from mlx_beam.api import chat
from mlx_beam.engine import Engine
from mlx_beam.modalities import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # mlx-beam's own tests: the tiny text model
import importlib  # noqa: E402

_core_tests = importlib.import_module("tests.stub_tokenizer")
StubTokenizer = _core_tests.StubTokenizer
tiny_qwen35 = importlib.import_module("tests.test_speculative").tiny_qwen35
reference = importlib.import_module("tests.test_vision_core").reference

PATCH, MERGE, T = 14, 2, 2
TOWER_CONFIG = {
    "model_type": "qwen3_5",
    "image_token_id": 62,
    "video_token_id": 61,
    "vision_config": {
        "depth": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "out_hidden_size": 32,
        "num_heads": 2,
        "patch_size": PATCH,
        "spatial_patch_size": PATCH,
        "spatial_merge_size": MERGE,
        "temporal_patch_size": T,
        "num_position_embeddings": 64,
        "fullatt_block_indexes": [1],
        "deepstack_visual_indexes": [0],
        "in_channels": 3,
    },
}


def tower():
    mx.random.seed(0)
    t = qwen3_vl.Tower(TOWER_CONFIG, None, dtype=mx.float32)
    mx.eval(t.model.parameters())
    return t


def pixels(grid):
    """Random pixel inputs for one image of grid (t, h, w) patches."""
    n = grid[0] * grid[1] * grid[2]
    return np.random.RandomState(n).randn(n, 3 * T * PATCH * PATCH).astype(np.float32)


IMAGE_TOKEN = "<|image_pad|>"


def image_parts(messages):
    return [
        p
        for m in messages
        if isinstance(m["content"], list)
        for p in m["content"]
        if p["type"] == "image"
    ]


class Rendered(str):
    """What the fake template renders: a str for the frontend's BOS check
    and assistant-frame probe (a real processor renders text), carrying
    the messages the fake tokenizer reads, one placeholder per image part
    (`placeholders` overrides the count). The generation prompt is the
    frame `<gen>`, one token (7) in the fake's ids."""

    def __new__(cls, messages, with_bos, generation_prompt=True, placeholders=None):
        base = "<bos>..." if with_bos else "..."
        if placeholders is None:
            placeholders = len(image_parts(messages))
        base += IMAGE_TOKEN * placeholders
        self = super().__new__(cls, base + ("<gen>" if generation_prompt else ""))
        self.messages = messages
        return self


class FakeProcessor:
    """Stands in for transformers' processor: the stub tokenizer's text,
    each image part `image_tokens` placeholders, random pixels."""

    image_token = IMAGE_TOKEN

    class Tok:
        bos_token = "<bos>"

        def encode(self, text, add_special_tokens=True):
            return [7] if text == "<gen>" else [9] * len(text)

    tokenizer = Tok()

    def __init__(self, grids, with_bos=False, no_generation_suffix=False):
        self.grids = list(grids)
        self.tok = StubTokenizer()
        self.with_bos = with_bos
        # A template that ignores add_generation_prompt renders the same.
        self.no_generation_suffix = no_generation_suffix
        self.calls: list[dict] = []
        self.template_kwargs: list[dict] = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kw
    ):
        self.template_kwargs.append(kw)
        return Rendered(
            messages, self.with_bos, add_generation_prompt or self.no_generation_suffix
        )

    def __call__(self, text, images=None, return_tensors="np", **kw):
        self.calls.append(kw)
        messages = text[0].messages
        ids = [2]
        grids = iter(self.grids)
        used = []
        for m in messages:
            ids.append({"system": 5, "user": 6, "assistant": 7}[m["role"]])
            content = (
                m["content"]
                if isinstance(m["content"], list)
                else [{"type": "text", "text": m["content"]}]
            )
            for part in content:
                if part["type"] == "text":
                    ids += self.tok.encode(part["text"])
                else:
                    g = next(grids)
                    used.append(g)
                    ids += [TOWER_CONFIG["image_token_id"]] * (
                        g[0] * g[1] * g[2] // MERGE**2
                    )
        ids.append(7)
        out = {"input_ids": np.array([ids])}
        if used:
            out["pixel_values"] = np.concatenate([pixels(g) for g in used])
            out["image_grid_thw"] = np.array(used)
        return out


def _count(encode):
    """`encode` wrapped to record the grids each tower pass saw."""

    def wrapped(pixel_values, grid_thw):
        wrapped.grids.append(grid_thw.tolist())
        return encode(pixel_values, grid_thw)

    wrapped.grids = []
    return wrapped


def png(seed):
    from PIL import Image as PILImage

    buf = io.BytesIO()
    arr = np.random.RandomState(seed).randint(0, 255, (8, 8, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(buf, format="PNG")
    return Image(buf.getvalue(), "image/png")


def test_tower_encodes_per_image_with_deepstack():
    t = tower()
    grids = [(1, 4, 4), (1, 2, 6)]
    pv = mx.array(np.concatenate([pixels(g) for g in grids]))
    encoded = t.encode(pv, mx.array(grids))
    assert [e.features.shape for e in encoded] == [(4, 32), (3, 32)]
    assert all(
        list(e.deepstack) == [1] and e.deepstack[1].shape == e.features.shape
        for e in encoded
    )
    d = t.describe()
    assert d["family"] == "qwen3_vl" and d["deepstack_layers"] == 1


def test_frontend_builds_the_mrope_positions_and_the_decode_delta():
    """Text before an image at one running position; the image's tokens at
    the block's start plus (frame, row, column) of the merged grid; the
    text after it from the block's largest position plus one - and the
    delta the decode continues with is that minus the token count."""
    front = VisionFrontend(FakeProcessor([(1, 4, 4)]), tower(), "qwen3_vl")
    assert front.needs == ("input_embeddings", "layer_hook", "position_ids")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "w1"},
                {"type": "image"},
                {"type": "text", "text": "w3"},
            ],
        }
    ]
    built = front.build(messages, [png(1)], {})
    n = len(built.tokens)
    start, end = built.spans[0].start, built.spans[0].end
    assert (end - start) == 4  # a 4 x 4 grid merged by 2
    pos = built.positions
    assert pos.shape == (3, n)
    # Text before the image: its index on every axis.
    assert pos[:, :start].tolist() == [list(range(start))] * 3
    # The image: frame 0, rows 0 and 1, columns 0 and 1, from `start`.
    assert pos[0, start:end].tolist() == [start] * 4
    assert pos[1, start:end].tolist() == [start, start, start + 1, start + 1]
    assert pos[2, start:end].tolist() == [start, start + 1, start, start + 1]
    # After it: from the block's largest position plus one.
    after = list(range(start + 2, start + 2 + (n - end)))
    assert pos[:, end:].tolist() == [after] * 3
    assert built.rope_delta == after[-1] + 1 - n == -2
    # A text-only prompt: every token at its index, no delta.
    plain = front.build([{"role": "user", "content": "w1"}], [], {})
    assert plain.positions.tolist() == [list(range(len(plain.tokens)))] * 3
    assert plain.rope_delta == 0


def test_the_positions_match_the_transformers_reference():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from mlx_beam_vision.families.qwen3_vl import positions
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    class Reference:
        config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = Qwen3VLModel.get_vision_position_ids
        get_rope_index = Qwen3VLModel.get_rope_index

    image = TOWER_CONFIG["image_token_id"]
    # A 4 x 4 grid merged by 2 is 2 x 2 = 4 tokens, a 2 x 6 grid 1 x 3 = 3.
    ids = [3, 4] + [image] * 4 + [5, 6] + [image] * 3 + [7]
    grids = [(1, 4, 4), (1, 2, 6)]
    token_types = [1 if t == image else 0 for t in ids]
    expected, delta = Reference().get_rope_index(
        torch.tensor([ids]),
        torch.tensor([token_types]),
        image_grid_thw=torch.tensor(grids),
    )
    ours, our_delta = positions(ids, image, grids, 2)
    assert ours.tolist() == expected[:, 0].tolist()
    assert our_delta == int(delta.item())


def test_frontend_builds_spans_and_caches_by_digest():
    front = VisionFrontend(FakeProcessor([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "w1"},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]
    a, b = png(1), png(2)
    built = front.build(messages, [a, b], {})
    assert [(s.start, s.end) for s in built.spans] == [(3, 7), (7, 10)]
    assert built.tokens[3:10] == [TOWER_CONFIG["image_token_id"]] * 7
    # The assistant's frame is the last token; the search for an open
    # think block starts there, not in the user's text.
    assert built.assistant_start == len(built.tokens) - 1
    assert built.spans[0].digest == a.digest and built.spans[1].deepstack[1].shape == (
        3,
        32,
    )
    assert front.images_encoded == 2
    again = front.build(messages, [a, b], {})
    assert front.images_encoded == 2 and front.cache.describe()["hits"] == 2
    assert mx.array_equal(again.spans[0].features, built.spans[0].features)
    # A cached entry is its own array, not a slice keeping the call's batch.
    assert again.spans[0].features.nbytes == 4 * 4 * 32
    d = front.describe()
    assert d["provider"] == "mlx-beam-vision" and d["feature_cache"]["entries"] == 2
    # One image known, one new: only the new one goes through the tower,
    # picked out of the processor's batch (its patches and its grid).
    front.tower.encode = _count(front.tower.encode)
    third = front.build(messages, [a, png(3)], {})
    assert front.images_encoded == 3 and front.tower.encode.grids == [[[1, 2, 6]]]
    assert mx.array_equal(third.spans[0].features, built.spans[0].features)
    # More images than image parts is the engine's mistake (it makes both
    # in one step), caught before the template renders - cold or warm.
    with pytest.raises(Exception, match="3 images for 2 image parts") as exc:
        front.build(messages, [a, b, png(9)], {})
    assert exc.type is RuntimeError
    with pytest.raises(Exception, match="3 images for 2 image parts") as exc:
        front.build(messages, [a, b, png(3)], {})
    assert exc.type is RuntimeError
    with pytest.raises(ValueError, match="not a readable image"):
        front.build(messages, [Image(b"nope"), b], {})


def test_frontend_guards_the_prompt_and_the_pixels():
    """The rendered template already carries the BOS: the processor must
    not add a second. A server-set chat template reaches the processor.
    An image above the pixel cap is refused before a pixel is decoded;
    EXIF orientation is applied."""
    from mlx_beam_vision import frontend as fe
    from PIL import Image as PILImage

    proc = FakeProcessor([(1, 4, 4)], with_bos=True)
    front = VisionFrontend(proc, tower(), "qwen3_vl")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "w1"}, {"type": "image"}]}
    ]
    front.build(messages, [png(1)], {"enable_thinking": False})
    assert proc.calls == [{"add_special_tokens": False}]
    # Rendered twice per build: with the generation prompt and, for the
    # assistant-frame probe, without - the same kwargs both times.
    assert proc.template_kwargs == [{"enable_thinking": False}] * 2
    front.chat_template = "{{ messages }}"
    front.build(messages, [png(1)], {})
    assert proc.template_kwargs[-1] == {"chat_template": "{{ messages }}"}
    assert front.describe()["chat_template"] == "server"
    plain = FakeProcessor([(1, 4, 4)])
    VisionFrontend(plain, tower(), "qwen3_vl").build(messages, [png(1)], {})
    assert plain.calls == [{}]
    # A template that renders the assistant's prefix either way: the frame
    # begins at the end, not at 0 - a `<think>` in the user's text is text.
    same = FakeProcessor([(1, 4, 4)], no_generation_suffix=True)
    built = VisionFrontend(same, tower(), "qwen3_vl").build(messages, [png(1)], {})
    assert built.assistant_start == len(built.tokens)
    # The cap reads the header, not the pixels: a 8000 x 8000 PNG of one
    # colour is a few kilobytes and would decode to 192 MB.
    buf = io.BytesIO()
    PILImage.new("RGB", (8000, 8000)).save(buf, format="PNG")
    with pytest.raises(ValueError, match="pixels is above"):
        fe.open_images([Image(buf.getvalue())])
    # EXIF says rotated: a 8 x 4 photo stored sideways comes out 4 x 8.
    buf = io.BytesIO()
    exif = PILImage.Exif()
    exif[0x0112] = 6
    PILImage.new("RGB", (8, 4)).save(buf, format="JPEG", exif=exif)
    (opened,) = fe.open_images([Image(buf.getvalue(), "image/jpeg")])
    assert opened.size == (4, 8)


def test_provider_dispatches_by_model_type():
    assert (
        provider.supports(TOWER_CONFIG)
        and provider.family_for(TOWER_CONFIG) == "qwen3_vl"
    )
    assert not provider.supports(
        {"model_type": "qwen3_5"}
    )  # no vision_config: text only
    assert not provider.supports({"model_type": "llama", "vision_config": {}})
    # A type the package knows and cannot serve is claimed and refused
    # with the reason, so the health says why instead of a wrong tower.
    from mlx_beam_vision import families

    families.UNSERVED["fake_vlm"] = "no tower for it here"
    try:
        unserved = {"model_type": "fake_vlm", "vision_config": {"x": 1}}
        assert provider.supports(unserved) and provider.family_for(unserved) is None
        with pytest.raises(ValueError, match="no tower for it"):
            provider.load(None, ".", unserved, None)
    finally:
        del families.UNSERVED["fake_vlm"]
    # Gemma 4's encoder-free models: a family of their own, whose spans
    # are blocks the trunk must mask.
    unified = {"model_type": "gemma4_unified", "vision_config": {"x": 1}}
    assert provider.family_for(unified) == "gemma4_unified"
    assert provider.needs(unified) == ("input_embeddings", "block_mask")
    # What the tower will ask of the prefill, from the config alone.
    assert provider.needs(TOWER_CONFIG) == (
        "input_embeddings",
        "layer_hook",
        "position_ids",
    )
    no_deepstack = {
        **TOWER_CONFIG,
        "vision_config": {
            **TOWER_CONFIG["vision_config"],
            "deepstack_visual_indexes": [],
        },
    }
    assert provider.needs(no_deepstack) == ("input_embeddings", "position_ids")
    assert provider.needs({"model_type": "gemma4", "vision_config": {}}) == (
        "input_embeddings",
    )
    assert any(
        ep.name == "vision"
        for ep in modalities.entry_points(group=modalities.ENTRY_POINT_GROUP)
    )


def test_a_checkpoint_without_an_image_processor_is_refused_at_load(
    tmp_path, monkeypatch
):
    """AutoProcessor hands a bare tokenizer back for a directory without
    preprocessor_config.json; that is refused with the reason at load, not
    with a KeyError on the first image."""
    import transformers
    from mlx_beam_vision import frontend

    class Bare:  # a tokenizer: no image_processor
        pass

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda path, trust_remote_code=False: Bare()),
    )
    with pytest.raises(ValueError, match="no image processor"):
        frontend.load_processor(tmp_path)


def test_engine_path_matches_a_direct_forward():
    model = tiny_qwen35()
    front = VisionFrontend(FakeProcessor([(1, 4, 4)]), tower(), "qwen3_vl")
    tok = StubTokenizer()
    data = __import__("base64").b64encode(png(3).data).decode()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "w3 w5"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{data}"},
                    },
                ],
            }
        ],
        "max_tokens": 6,
    }
    req = chat.parse_chat_request(body, "m", vision=True)
    gen = chat.to_generation_request(tok, req, frontend=front)
    assert len(gen.spans) == 1 and gen.spans[0].deepstack
    with Engine(model) as engine:
        out = [e.token for e in engine.submit(gen)]
        h = engine.health()
    # 4 placeholders of a 2 x 2 block take 2 positions: the delta is -2.
    assert gen.positions is not None and gen.rope_delta == -2
    assert out == reference(model, gen.tokens, list(gen.spans), 6, gen.positions)
    assert h["alive"]


def test_mistral3_family_merges_and_deals_features_over_broken_runs():
    """Pixtral tower on a tiny config: the projector merges 2 x 2 patches
    (the merger's block order equals unfold's), one image's tokens sit in
    rows broken by [IMG_BREAK], so the frontend gives it several spans."""
    from mlx_beam_vision.families import mistral3

    config = {
        "model_type": "mistral3",
        "image_token_id": 62,
        "spatial_merge_size": 2,
        "vision_feature_layer": -1,
        "vision_config": {
            "model_type": "pixtral",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "head_dim": 16,
            "image_size": 64,
            "patch_size": 16,
            "num_channels": 3,
        },
        "text_config": {"hidden_size": 32, "rms_norm_eps": 1e-5},
    }
    mx.random.seed(1)
    t = mistral3.Tower(config, None, dtype=mx.float32)
    mx.eval(t.model.parameters(), t.projector.parameters())
    # Two images: 64x32 (4x2 patches -> 2x1 merged) and 32x32 (2x2 -> 1x1).
    pv = mx.array(np.random.RandomState(0).randn(2, 64, 64, 3).astype(np.float32))
    encoded = t.encode(pv, [(64, 32), (32, 32)])
    assert [e.features.shape for e in encoded] == [(2, 32), (1, 32)]
    # The merger against unfold's order, on one 2 x 2 block of known values.
    grid = mx.arange(4 * 32, dtype=mx.float32).reshape(4, 32)  # 2 x 2 patches
    merger = t.projector.patch_merger
    merger.merging_layer.weight = mx.eye(32 * 4)[
        :32, :
    ]  # reads channel 0 of patch 0 ... etc
    out = merger(grid, [(32, 32)])
    # unfold's order: feature index c*4 + (di*2 + dj), the patch index row-major.
    expected = mx.array([[grid[j % 4, j // 4] for j in range(32)]])
    assert mx.allclose(out, expected)

    class Proc:
        image_token = "[IMG]"

        def apply_chat_template(self, messages, **kw):
            return "[IMG]" * len(image_parts(messages))

        def __call__(self, text, images=None, return_tensors="np"):
            # 2x1 merged tokens as one row of two [IMG] then [IMG_BREAK] (63),
            # then the 1x1 image: one [IMG] + [IMG_END] (60).
            ids = [2, 6, 3, 62, 62, 63, 60, 62, 60, 7]
            return {
                "input_ids": np.array([ids]),
                "pixel_values": np.random.RandomState(0)
                .randn(2, 3, 64, 64)
                .astype(np.float32),
                "image_sizes": np.array([[64, 32], [32, 32]]),
            }

    front = VisionFrontend(Proc(), t, "mistral3")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "w3"},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]
    built = front.build(messages, [png(5), png(6)], {})
    assert [(s.start, s.end) for s in built.spans] == [(3, 5), (7, 8)]
    assert built.spans[0].features.shape == (2, 32) and built.spans[
        1
    ].features.shape == (1, 32)
    assert provider.family_for(config) == "mistral3"

    # The same tower behind mistral-common: the backend renders the ids
    # and the pixel values itself, the images reach it as data URLs, and
    # images of two sizes come back one array each - the frontend pads
    # them to one batch with the real sizes beside, as the tower takes
    # them. No generation prompt in Mistral's format: the assistant
    # starts where the prompt ends.
    class MistralCommonBackend:
        chat_template = None

        def __init__(self):
            self.seen = []

        def apply_chat_template(self, messages, tokenize=True, return_dict=True, **kw):
            self.seen.append(messages)
            pixels = np.random.RandomState(0).randn(2, 3, 64, 64).astype(np.float32)
            return {
                "input_ids": [2, 6, 3, 62, 62, 63, 60, 62, 60, 7],
                "pixel_values": [pixels[0], pixels[1][:, :32, :32]],
                "image_sizes": [[64, 32], [32, 32]],
            }

    class Wrapper:  # mlx-lm's wrapper around the backend, as the engine hands it in
        def __init__(self, inner):
            self._tokenizer = inner

    backend = MistralCommonBackend()
    assert frontend.mistral_common_backend(Wrapper(backend)) is backend
    assert frontend.mistral_common_backend(Wrapper(Proc())) is None
    native = VisionFrontend(None, t, "mistral3", renderer=backend)
    built_native = native.build(messages, [png(5), png(6)], {})
    assert [(s.start, s.end) for s in built_native.spans] == [(3, 5), (7, 8)]
    assert built_native.assistant_start == len(built_native.tokens) == 10
    parts = backend.seen[0][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    for span, want in zip(built_native.spans, built.spans, strict=True):
        assert mx.allclose(span.features, want.features, atol=1e-5)
    assert native.describe()["processor"] == {
        "source": "mistral-common",
        "class": "MistralCommonBackend",
    }
    # A tower whose image_token_id is not the processor's placeholder: the
    # template placed its tokens, the spans find none - the engine's error.
    t.image_token_id = 61
    with pytest.raises(Exception, match="image tokens") as exc:
        front.build(messages, [png(5), png(6)], {})
    assert exc.type is RuntimeError


class FakeTokenizer:
    """What the builder asks a tokenizer: the chat template the checkpoint
    carries and the tokens it knows."""

    chat_template = "<template>"
    unk_token_id = 0
    vocab = {"[IMG]": 10, "[IMG_BREAK]": 11, "[IMG_END]": 12, "<|image_pad|>": 62}

    def convert_ids_to_tokens(self, i):
        return {v: k for k, v in self.vocab.items()}[i]

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


class _Kwargs(dict):
    """Stands in for a transformers kwargs TypedDict: `__annotations__`
    names the keys a class accepts."""


def _image_processor_class(name, **defaults):
    kw = type(f"{name}Kwargs", (), {"__annotations__": {k: object for k in defaults}})

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    return type(name, (), {**defaults, "valid_kwargs": kw, "__init__": __init__})


class FakePixtralProcessor:
    """PixtralProcessor's signature: two attributes, the rest settings."""

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=16,
        spatial_merge_size=1,
        chat_template=None,
        image_token="[IMG]",
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.settings = dict(
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            image_token=image_token,
            image_break_token=image_break_token,
            image_end_token=image_end_token,
        )

    @classmethod
    def get_attributes(cls):
        return ["image_processor", "tokenizer"]


class FakeQwenProcessor:
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
    ):
        self.image_processor = image_processor
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @classmethod
    def get_attributes(cls):
        return ["image_processor", "tokenizer", "video_processor"]


PixtralImages = _image_processor_class(
    "PixtralImageProcessor",
    size={"longest_edge": 1024},
    patch_size={"height": 16, "width": 16},
    image_mean=[0.48, 0.46, 0.41],
    image_std=[0.27, 0.26, 0.28],
    do_resize=True,
)
QwenImages = _image_processor_class(
    "Qwen2VLImageProcessor",
    size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280},
    patch_size=14,
    temporal_patch_size=2,
    merge_size=2,
    image_mean=[0.48, 0.46, 0.41],
)
QwenVideos = _image_processor_class(
    "Qwen3VLVideoProcessor",
    patch_size=16,
    temporal_patch_size=2,
    merge_size=2,
    max_frames=768,
)


def _built_from(monkeypatch, classes, config, tmp_path):
    import transformers
    from mlx_beam_vision import frontend

    monkeypatch.setattr(frontend, "classes_for", lambda model_type: classes)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, trust_remote_code=False: FakeTokenizer()),
    )
    return frontend.build_processor(tmp_path, config)


def test_a_processor_is_built_from_the_model_before_the_class_defaults(
    tmp_path, monkeypatch
):
    """mlx-community's Devstral ships no preprocessor_config.json. The
    builder takes transformers' classes for the model type and fills them
    from config.json: the image size and patch size shaped like the
    class's size dicts, the merge size, the image token the config's id
    names; mean, std and the break/end tokens stay the class's defaults.
    The provenance says which is which."""
    config = {
        "model_type": "mistral3",
        "image_token_index": 10,
        "spatial_merge_size": 2,
        "vision_config": {"image_size": 1540, "patch_size": 14, "hidden_size": 1024},
    }
    processor, info = _built_from(
        monkeypatch, (FakePixtralProcessor, PixtralImages, None, None), config, tmp_path
    )
    assert processor.image_processor.kwargs == {
        "size": {"longest_edge": 1540},
        "patch_size": {"height": 14, "width": 14},
    }
    assert processor.settings == dict(
        patch_size=14,
        spatial_merge_size=2,
        image_token="[IMG]",
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
    )
    assert processor.chat_template == "<template>"
    assert info["source"] == "built" and info["class"] == "FakePixtralProcessor"
    assert info["from_model"] == {
        "image_token": "config.json.image_token_id",
        "patch_size": "vision_config.patch_size",
        "size": "vision_config.image_size",
        "spatial_merge_size": "config.json.spatial_merge_size",
    }
    assert info["defaults"] == {
        "PixtralImageProcessor": ["do_resize", "image_mean", "image_std"]
    }


def test_the_builder_keeps_a_size_rule_the_model_has_no_value_for(
    tmp_path, monkeypatch
):
    """Qwen's processors size by pixel bounds (two keys, nothing in the
    model to take) and need a video processor beside the image one; both
    take the patch, temporal patch and merge sizes from vision_config."""
    config = {
        "model_type": "qwen3_5",
        "image_token_id": 62,
        "vision_config": {
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    }
    processor, info = _built_from(
        monkeypatch, (FakeQwenProcessor, QwenImages, QwenVideos, None), config, tmp_path
    )
    want = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}
    assert processor.image_processor.kwargs == want
    assert processor.video_processor.kwargs == want
    assert "size" not in info["from_model"]
    assert info["defaults"]["Qwen2VLImageProcessor"] == ["image_mean", "size"]
    assert info["defaults"]["Qwen3VLVideoProcessor"] == ["max_frames"]


def test_the_builder_refuses_a_token_default_the_tokenizer_lacks(tmp_path, monkeypatch):
    class OtherTokens(FakePixtralProcessor):
        def __init__(
            self,
            image_processor=None,
            tokenizer=None,
            chat_template=None,
            image_break_token="[ROW]",
        ):
            pass

    config = {"model_type": "mistral3", "vision_config": {"patch_size": 14}}
    with pytest.raises(ValueError, match="does not know"):
        _built_from(
            monkeypatch, (OtherTokens, PixtralImages, None, None), config, tmp_path
        )


def test_load_processor_builds_only_when_the_checkpoint_ships_no_file(
    tmp_path, monkeypatch
):
    """AutoProcessor fails two ways on such a checkpoint - a bare tokenizer
    back (Devstral) or "Can't load image processor" (Qwen3.5) - and both
    go to the builder; the same failures with the file present are
    errors, and without a config the old refusal stands."""
    import transformers
    from mlx_beam_vision import frontend

    class Bare:
        pass

    config = {
        "model_type": "mistral3",
        "image_token_index": 10,
        "vision_config": {"patch_size": 14},
    }
    monkeypatch.setattr(
        frontend,
        "classes_for",
        lambda t: (FakePixtralProcessor, PixtralImages, None, None),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, trust_remote_code=False: FakeTokenizer()),
    )
    for failure in (
        lambda: Bare(),
        lambda: (_ for _ in ()).throw(OSError("Can't load image processor")),
    ):
        monkeypatch.setattr(
            transformers.AutoProcessor,
            "from_pretrained",
            staticmethod(lambda path, trust_remote_code=False, f=failure: f()),
        )
        processor, info = frontend.load_processor(tmp_path, config)
        assert isinstance(processor, FakePixtralProcessor) and info["source"] == "built"
    with pytest.raises(ValueError, match="no image processor"):
        frontend.load_processor(tmp_path, None)
    (tmp_path / "preprocessor_config.json").write_text("{}")
    with pytest.raises(OSError):
        frontend.load_processor(tmp_path, config)


def test_classes_for_reads_a_video_mapping_per_backend(monkeypatch):
    """transformers names the video processor per backend, as it names the
    image processor: a dict, not a string. Under a model type of its own
    the mappings get such an entry, and the class comes back by name."""
    from transformers.models.auto import (
        image_processing_auto,
        processing_auto,
        video_processing_auto,
    )

    monkeypatch.setitem(
        processing_auto.PROCESSOR_MAPPING_NAMES, "beam_test_vl", "Qwen3VLProcessor"
    )
    monkeypatch.setitem(
        image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES,
        "beam_test_vl",
        image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES["qwen3_vl"],
    )
    monkeypatch.setitem(
        video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES,
        "beam_test_vl",
        {"torchvision": "Qwen3VLVideoProcessor"},
    )
    assert frontend.classes_for("beam_test_vl")[2].__name__ == "Qwen3VLVideoProcessor"


def test_classes_for_resolves_the_installed_mappings():
    """Against the installed transformers, without a mock: the families with
    a video processor get its class, the ones without get None."""
    expected = {
        "qwen3_vl": "Qwen3VLVideoProcessor",
        "qwen3_5": "Qwen3VLVideoProcessor",
        "gemma4": "Gemma4VideoProcessor",
        "gemma4_unified": "Gemma4UnifiedVideoProcessor",
        "muse_glimmer": "MuseGlimmerVideoProcessor",
    }
    for model_type, name in expected.items():
        classes = frontend.classes_for(model_type)
        assert all(isinstance(c, type) for c in classes[:3]), model_type
        assert classes[2].__name__ == name
    for model_type in ("mistral3", "granite4_vision"):
        assert frontend.classes_for(model_type)[2] is None


def test_a_mapping_value_becomes_one_class_name():
    """The helper picks one class name out of a transformers mapping value:
    a string as it is; a dict by backend, the first name it has when the
    backend's is missing, None when it has none; a tuple by its first name;
    None for nothing."""
    assert hasattr(frontend, "_class_name")
    pick = frontend._class_name
    assert pick("PixtralImageProcessor", "torchvision") == "PixtralImageProcessor"
    assert pick({"torchvision": "A", "pil": "B"}, "torchvision") == "A"
    assert pick({"pil": "B"}, "torchvision") == "B"
    assert pick({"torchvision": None}, "torchvision") is None
    assert pick(("A", "B"), "torchvision") == "A"
    assert pick((None, "B"), "pil") == "B"
    assert pick(None, "torchvision") is None


def test_a_quantized_tower_loads_with_the_bits_its_tensors_imply(tmp_path):
    """An 8-bit conversion packs the tower's linears too (weight as uint32
    words, scales and biases beside it) and may leave some of them in
    floating point; the loader swaps exactly the packed ones for their
    quantized form, with bits and group size read off the tensors, and
    the features match the quantized source. The health says so."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    mx.random.seed(2)
    source = tower()
    nn.quantize(
        source.model,
        group_size=32,
        bits=8,
        # A mixed conversion: the last merger linear stays floating point.
        class_predicate=lambda p, m: isinstance(m, nn.Linear)
        and p != "merger.linear_fc2",
    )
    mx.eval(source.model.parameters())
    weights = dict(tree_flatten(source.model.parameters()))
    assert "blocks.0.attn.qkv.scales" in weights
    assert "merger.linear_fc2.scales" not in weights
    stored = {
        f"model.visual.{k}": (
            v.transpose(0, 4, 1, 2, 3) if k == "patch_embed.proj.weight" else v
        )
        for k, v in weights.items()
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), stored)
    config = {**TOWER_CONFIG, "quantization": {"group_size": 32, "bits": 8}}
    t = qwen3_vl.Tower(config, tmp_path, dtype=mx.float32)
    assert t.quantized and t.describe()["quantized"]
    qkv = t.model.blocks[0].attn.qkv
    assert isinstance(qkv, nn.QuantizedLinear) and (qkv.bits, qkv.group_size) == (8, 32)
    assert isinstance(t.model.merger.linear_fc2, nn.Linear)
    assert not isinstance(t.model.merger.linear_fc2, nn.QuantizedLinear)
    pv, grid = mx.array(pixels((1, 4, 4))), mx.array([(1, 4, 4)])
    want = source.encode(pv, grid)[0].features
    assert mx.allclose(t.encode(pv, grid)[0].features, want, atol=1e-5)
    # The same checkpoint in floating point is close, not equal: the test
    # would pass trivially if the packed tensors were silently cast.
    plain = tower()
    assert not mx.allclose(plain.encode(pv, grid)[0].features, want, atol=1e-5)


def test_mistral3_loads_the_hf_layout_and_mlx_vlms(tmp_path):
    """HF keeps Pixtral's blocks right under `vision_tower` (and everything
    under `model.`); mlx-vlm's conversions add a `vision_model` level. Both
    load into the same tower, strictly."""
    from mlx.utils import tree_flatten
    from mlx_beam_vision.families import mistral3

    config = {
        "model_type": "mistral3",
        "vision_config": {
            "model_type": "pixtral",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "head_dim": 16,
            "image_size": 64,
            "patch_size": 16,
            "num_channels": 3,
        },
        "text_config": {"hidden_size": 32, "rms_norm_eps": 1e-5},
    }
    mx.random.seed(1)
    source = mistral3.Tower(config, None, dtype=mx.float32)
    mx.eval(source.model.parameters(), source.projector.parameters())
    tower = dict(tree_flatten(source.model.parameters()))
    proj = dict(tree_flatten(source.projector.parameters()))
    layouts = {
        "hf": {
            **{
                f"model.vision_tower.{k.removeprefix('vision_model.')}": v
                for k, v in tower.items()
            },
            **{f"model.multi_modal_projector.{k}": v for k, v in proj.items()},
        },
        "mlx-vlm": {
            **{f"vision_tower.{k}": v for k, v in tower.items()},
            **{f"multi_modal_projector.{k}": v for k, v in proj.items()},
        },
    }
    pv = mx.array(np.random.RandomState(0).randn(1, 64, 64, 3).astype(np.float32))
    want = source.encode(pv, [(64, 64)])[0].features
    for name, weights in layouts.items():
        path = tmp_path / name
        path.mkdir()
        # Conv weights in PyTorch's layout, as a checkpoint stores them.
        stored = {
            k: (v.transpose(0, 3, 1, 2) if "patch_conv.weight" in k else v)
            for k, v in weights.items()
        }
        mx.save_safetensors(str(path / "model.safetensors"), stored)
        t = mistral3.Tower(config, path, dtype=mx.float32)
        assert t.loaded_from == ["model.safetensors"]
        assert mx.allclose(t.encode(pv, [(64, 64)])[0].features, want, atol=1e-5)


def test_a_tower_loaded_through_the_registry_encodes_in_another_thread(tmp_path):
    """The Pixtral tower builds its rotary table while it loads, an array
    outside parameters(); the tower then encodes in request threads. Loaded
    the way the server loads it, that table must be evaluated already: the
    worker builds the pixels and evaluates the features itself."""
    from mlx.utils import tree_flatten
    from mlx_beam_vision.families import mistral3

    config = {
        "model_type": "mistral3",
        "image_token_id": 62,
        "vision_config": {
            "model_type": "pixtral",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "head_dim": 16,
            "image_size": 64,
            "patch_size": 16,
            "num_channels": 3,
        },
        "text_config": {"hidden_size": 32, "rms_norm_eps": 1e-5},
    }
    mx.random.seed(1)
    source = mistral3.Tower(config, None, dtype=mx.float32)
    mx.eval(source.model.parameters(), source.projector.parameters())
    weights = {
        **{f"vision_tower.{k}": v for k, v in tree_flatten(source.model.parameters())},
        **{
            f"multi_modal_projector.{k}": v
            for k, v in tree_flatten(source.projector.parameters())
        },
    }
    stored = {
        k: (v.transpose(0, 3, 1, 2) if "patch_conv.weight" in k else v)
        for k, v in weights.items()
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), stored)

    class MistralCommonBackend:  # the tekken path: no processor is read
        chat_template = None

    class Wrapper:
        def __init__(self, inner):
            self._tokenizer = inner

    modalities.register(provider)
    try:
        loaded, reason = modalities.load_frontend(
            tiny_qwen35(), tmp_path, config, Wrapper(MistralCommonBackend())
        )
    finally:
        modalities._registered.remove(provider)
    assert reason is None and loaded.tower.loaded_from == ["model.safetensors"]

    def encode():
        pv = mx.array(np.random.RandomState(0).randn(1, 64, 64, 3).astype(np.float32))
        features = [e.features for e in loaded.tower.encode(pv, [(64, 64)])]
        mx.eval(*features)
        return [f.shape for f in features]

    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(encode).result() == [(4, 32)]


def test_gemma4_unified_embeds_patches_and_marks_its_spans_as_blocks(tmp_path):
    """The encoder-free family on a tiny config: the embedder against a
    numpy reading of transformers' Gemma4UnifiedVisionEmbedder (patch
    norms, the factorized position embedding summed per axis, the
    scaleless RMSNorm before the projection), padded patches dropped per
    image; both checkpoint layouts load, mlx-vlm's with the projection
    quantized as the 8-bit conversions ship it; the frontend's spans are
    blocks and the family asks the trunk for the block mask."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_beam_vision.families import gemma4_unified

    patch, mm, pos_size, text_hidden = 4, 32, 8, 12
    config = {
        "model_type": "gemma4_unified",
        "image_token_id": 62,
        "vision_config": {
            "model_patch_size": patch,
            "mm_embed_dim": mm,
            "mm_posemb_size": pos_size,
            "output_proj_dims": mm,
            "rms_norm_eps": 1e-6,
        },
        "text_config": {"hidden_size": text_hidden},
    }
    mx.random.seed(3)
    t = gemma4_unified.Tower(config, None, dtype=mx.float32)
    # Random parameters everywhere, so the norms and tables matter.
    for module in (t.embedder, t.projection):
        module.load_weights(
            [
                (k, mx.random.normal(v.shape) * 0.5)
                for k, v in tree_flatten(module.parameters())
            ]
        )
    mx.eval(t.embedder.parameters(), t.projection.parameters())
    rs = np.random.RandomState(0)
    # Two images: 6 real patches and 3, padded to the budget of 6.
    pv = rs.randn(2, 6, patch * patch * 3).astype(np.float32)
    pos = np.array(
        [
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            [[0, 0], [1, 0], [2, 0], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    encoded = t.encode(mx.array(pv), mx.array(pos))
    assert [e.features.shape for e in encoded] == [(6, text_hidden), (3, text_hidden)]

    def ln(x, w, b, eps=1e-5):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps) * w + b

    e = {k: np.asarray(v) for k, v in tree_flatten(t.embedder.parameters())}
    proj = np.asarray(t.projection.embedding_projection.weight)
    for i, enc in enumerate(encoded):
        n = enc.features.shape[0]
        h = ln(pv[i, :n], e["patch_ln1.weight"], e["patch_ln1.bias"])
        h = h @ e["patch_dense.weight"].T + e["patch_dense.bias"]
        h = ln(h, e["patch_ln2.weight"], e["patch_ln2.bias"])
        h = (
            h
            + e["pos_embedding"][pos[i, :n, 0], 0]
            + e["pos_embedding"][pos[i, :n, 1], 1]
        )
        h = ln(h, e["pos_norm.weight"], e["pos_norm.bias"])
        h = h / np.sqrt((h**2).mean(-1, keepdims=True) + 1e-6)
        assert np.allclose(np.asarray(enc.features), h @ proj.T, atol=1e-4)

    # Both layouts; the mlx-vlm one with the projection quantized.
    embedder = dict(tree_flatten(t.embedder.parameters()))
    projection = dict(tree_flatten(t.projection.parameters()))
    quantized = gemma4_unified.Projection(mm, text_hidden, 1e-6)
    quantized.load_weights(list(projection.items()))
    nn.quantize(quantized, group_size=32, bits=8)
    mx.eval(quantized.parameters())
    layouts = {
        "hf": {
            **{f"model.embed_vision.{k}": v for k, v in embedder.items()},
            **{
                f"model.embed_vision.multimodal_embedder.{k}": v
                for k, v in projection.items()
            },
        },
        "mlx-vlm": {
            **{f"vision_embedder.{k}": v for k, v in embedder.items()},
            **{f"embed_vision.{k}": v for k, v in tree_flatten(quantized.parameters())},
        },
    }
    want = [np.asarray(e.features) for e in encoded]
    for name, weights in layouts.items():
        path = tmp_path / name
        path.mkdir()
        mx.save_safetensors(str(path / "model.safetensors"), weights)
        loaded = gemma4_unified.Tower(
            {**config, "quantization": {"group_size": 32, "bits": 8}},
            path,
            dtype=mx.float32,
        )
        assert loaded.quantized == (name == "mlx-vlm")
        got = loaded.encode(mx.array(pv), mx.array(pos))
        tol = 1e-5 if name == "hf" else 0.3  # the 8-bit projection
        for a, b in zip(got, want, strict=True):
            assert np.allclose(np.asarray(a.features), b, atol=tol)

    # The frontend: soft-token runs become block spans.
    class Proc:
        image_token = "<|image|>"

        def apply_chat_template(self, messages, **kw):
            return "<|image|>" * len(image_parts(messages))

        def __call__(self, text, images=None, return_tensors="np"):
            ids = [2, 6, 3, 61, 62, 62, 62, 62, 62, 62, 60, 61, 62, 62, 62, 60, 7]
            return {
                "input_ids": np.array([ids]),
                "pixel_values": pv,
                "image_position_ids": pos,
            }

    front = VisionFrontend(Proc(), t, "gemma4_unified")
    assert front.needs == ("input_embeddings", "block_mask")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "w3"},
                {"type": "image"},
                {"type": "image"},
            ],
        }
    ]
    built = front.build(messages, [png(8), png(9)], {})
    assert [(s.start, s.end, s.block) for s in built.spans] == [
        (4, 10, True),
        (12, 15, True),
    ]
    assert provider.needs(config) == ("input_embeddings", "block_mask")


def test_gemma4_family_pools_and_projects_per_image():
    from mlx_beam_vision.families import gemma4

    config = {
        "model_type": "gemma4",
        "image_token_id": 62,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "global_head_dim": 16,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 64,
            "layer_types": ["full_attention"],
        },
        "text_config": {"hidden_size": 32},
    }
    mx.random.seed(2)
    t = gemma4.Tower(config, None, dtype=mx.float32)
    mx.eval(t.model.parameters(), t.embedder.parameters())
    # 96 x 96 -> 6 x 6 patches -> 2 x 2 pooled = 4 soft tokens; 48 x 96 -> 3x6 -> 1x2 = 2.
    big = mx.array(np.random.RandomState(0).randn(3, 96, 96).astype(np.float32))
    small = mx.array(np.random.RandomState(1).randn(3, 48, 96).astype(np.float32))
    encoded = t.encode([big, small], None)
    assert [e.features.shape for e in encoded] == [(4, 32), (2, 32)]
    assert provider.family_for(config) == "gemma4"
    # As transformers' Gemma 4 processor hands them: patchified and padded
    # to one length, with `image_position_ids` (-1 on the padding rows).
    processed = _gemma_processed([big, small], patch=16, max_patches=36)
    images, positions = gemma4.pixel_inputs(processed)
    assert [p.shape for p in images] == [(36, 768), (36, 768)]
    assert positions[1].shape == (36, 2) and positions[1][-1].tolist() == [-1, -1]
    from_processor = t.encode(images, positions)
    assert [e.features.shape for e in from_processor] == [(4, 32), (2, 32)]
    assert mx.allclose(from_processor[0].features, encoded[0].features, atol=1e-4)
    assert mx.allclose(from_processor[1].features, encoded[1].features, atol=1e-4)


def _gemma_processed(images, patch, max_patches):
    """Patches (N, max_patches, patch pixels) and positions (N, max_patches,
    2) the way image_processing_gemma4 lays them out: row-major patches,
    (x, y) positions, padding rows zero with position -1."""
    pixel_values, position_ids = [], []
    for image in images:
        c, h, w = image.shape
        ph, pw = h // patch, w // patch
        # convert_image_to_patches: (pH, pW, p, p, C), then row-major.
        patches = (
            image.reshape(c, ph, patch, pw, patch)
            .transpose(1, 3, 2, 4, 0)
            .reshape(ph * pw, c * patch * patch)
        )
        xs, ys = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
        pos = np.stack([xs.flatten(), ys.flatten()], axis=-1)
        pad = max_patches - ph * pw
        patches = np.pad(np.array(patches), [(0, pad), (0, 0)])
        pos = np.pad(pos, [(0, pad), (0, 0)], constant_values=-1)
        pixel_values.append(patches)
        position_ids.append(pos)
    return {
        "pixel_values": np.stack(pixel_values),
        "image_position_ids": np.stack(position_ids),
    }


def test_muse_glimmer_family_shuffles_adapts_and_projects():
    from mlx_beam_vision.families import muse_glimmer

    config = {
        "model_type": "muse_glimmer",
        "image_token_id": 62,
        "out_hidden_size": 128,
        "projector_hidden_size": 48,
        "vision_config": {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "patch_size": 14,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
        },
        "text_config": {"hidden_size": 32, "rms_norm_eps": 1e-5},
    }
    mx.random.seed(3)
    t = muse_glimmer.Tower(config, None, dtype=mx.float32)
    mx.eval(
        t.model.parameters(),
        t.vision_adapter.parameters(),
        t.vision_projection.parameters(),
    )
    grids = [(1, 4, 4), (1, 2, 4)]
    n = sum(g[0] * g[1] * g[2] for g in grids)
    pv = mx.array(np.random.RandomState(0).randn(n, 3 * 2 * 14 * 14).astype(np.float32))
    encoded = t.encode(pv, mx.array(grids))
    assert [e.features.shape for e in encoded] == [(4, 32), (2, 32)]
    # Normed by the projector's scaleless norm: what the text model's
    # embed_inputs produces for text, so the features go in as they are.
    f = encoded[0].features
    assert mx.allclose(f.square().mean(-1), mx.ones(f.shape[0]), atol=1e-2)
    assert provider.family_for(config) == "muse_glimmer"
    # The engine's prefill builds the text positions the way the model
    # itself does (embed_inputs), and the model takes the whole thing.
    import importlib

    muse = importlib.import_module("mlx_beam._vendor.mlx_lm.models.muse_glimmer")
    assert hasattr(muse.MuseGlimmerModel, "embed_inputs")


GRANITE_CONFIG = {
    "model_type": "granite4_vision",
    "image_token_index": 62,
    "image_grid_pinpoints": [[32, 64], [64, 32]],
    "downsample_rate": "1/2",
    "deepstack_layer_map": [[0, 0]],
    "use_spatial_sampling": True,
    "spatial_vision_layer": -1,
    "spatial_target_layers": [1],
    "vision_feature_select_strategy": "full",
    "vision_config": {
        "model_type": "siglip_vision_model",
        "num_hidden_layers": 1,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "image_size": 32,
        "patch_size": 16,
        "num_channels": 3,
    },
    "text_config": {
        "model_type": "granite",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-5,
        "vocab_size": 64,
        "logits_scaling": 1.0,
        "attention_multiplier": 0.25,
        "embedding_multiplier": 2.0,
        "residual_multiplier": 0.5,
        "max_position_embeddings": 512,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": 10000.0,
    },
}


def tiny_granite_text():
    from mlx_beam._vendor.mlx_lm.models import granite

    mx.random.seed(4)
    model = granite.Model(granite.ModelArgs.from_dict(GRANITE_CONFIG["text_config"]))
    mx.eval(model.parameters())
    return model


def granite_tower(config=GRANITE_CONFIG, seed=5):
    from mlx_beam_vision.families import granite4_vision

    mx.random.seed(seed)
    t = granite4_vision.Tower(config, None, dtype=mx.float32)
    # The checkpoint brings the newline feature; here a stand-in.
    t.image_newline = mx.random.normal((32,))
    return t


def test_granite_vision_family_projects_ahead_of_its_target_layers():
    """Granite adds nothing at the embedding: zero features at the
    placeholders, one projected set per target layer - the base view, the
    unpadded tile grid with a newline per row."""
    t = granite_tower()
    # One tile (32 x 32): 1 token from a 2x2 patch grid at rate 1/2, plus the newline.
    one = mx.array(np.random.RandomState(0).randn(1, 3, 32, 32).astype(np.float32))
    # A 32 x 64 image: base view plus a 1 x 2 tile grid -> 1 + (1 row x 2 + newline) = 4.
    two = mx.array(np.random.RandomState(1).randn(3, 3, 32, 32).astype(np.float32))
    encoded = t.encode([one, two], [(32, 32), (32, 64)])
    assert [e.features.shape for e in encoded] == [(2, 32), (4, 32)]
    assert all(mx.all(e.features == 0).item() for e in encoded)
    assert [sorted(e.deepstack) for e in encoded] == [[0, 1], [0, 1]]
    assert encoded[1].deepstack[0].shape == (4, 32) and encoded[1].deepstack[
        1
    ].shape == (4, 32)
    assert provider.family_for(GRANITE_CONFIG) == "granite4_vision"
    d = t.describe()
    assert d["deepstack_layers"] == [0] and d["spatial_layers"] == [1]
    # A LLaVA-NeXT processor pads every image to the call's largest tile
    # count: beside a 64 x 64 image (base + 2 x 2 tiles, 5) the 32 x 64
    # one arrives with two zero tiles and must read the same as alone.
    wide = {**GRANITE_CONFIG, "image_grid_pinpoints": [[32, 64], [64, 32], [64, 64]]}
    t = granite_tower(wide)
    assert t.tiles_of((32, 64)) == 3 and t.tiles_of((64, 64)) == 5
    alone = t.encode([two], [(32, 64)])[0]
    padded = mx.concatenate([two, mx.zeros((2, 3, 32, 32))])
    four = mx.array(np.random.RandomState(2).randn(5, 3, 32, 32).astype(np.float32))
    beside = t.encode([padded, four], [(32, 64), (64, 64)])
    assert mx.array_equal(beside[0].deepstack[0], alone.deepstack[0])
    assert beside[1].deepstack[0].shape[0] == 1 + 2 * (2 + 1)


def test_granite_unpad_rounds_before_it_truncates():
    """303 x 303 into a 12 x 10 grid: the scaled edge is 9.999999999999998,
    which int() would cut to 9 and the reference keeps at 10."""
    from mlx_beam_vision.families.granite4_vision import unpad

    grid = mx.zeros((4, 12, 10))
    assert unpad(grid, (303, 303)).shape == (4, 10, 10)


def test_granite_projectors_aimed_at_one_layer_both_add_there():
    from mlx_beam_vision.families import granite4_vision

    both = {**GRANITE_CONFIG, "spatial_target_layers": [0]}
    t = granite_tower(both)
    one = mx.array(np.random.RandomState(0).randn(1, 3, 32, 32).astype(np.float32))
    (enc,) = t.encode([one], [(32, 32)])
    assert list(enc.deepstack) == [0]
    # The same sum as two projectors kept apart.
    apart = granite_tower()
    (split,) = apart.encode([one], [(32, 32)])
    assert mx.allclose(
        enc.deepstack[0], split.deepstack[0] + split.deepstack[1], atol=1e-5
    )
    none = {**GRANITE_CONFIG, "deepstack_layer_map": [], "use_spatial_sampling": False}
    with pytest.raises(ValueError, match="no text layer"):
        granite4_vision.Tower(none, None, dtype=mx.float32)


def test_granite_engine_path_matches_a_direct_forward():
    from mlx_beam.engine import GenerationRequest

    model = tiny_granite_text()
    t = granite_tower(seed=6)
    tiles = mx.array(np.random.RandomState(2).randn(1, 3, 32, 32).astype(np.float32))
    (enc,) = t.encode([tiles], [(32, 32)])
    from mlx_beam.engine import ImageSpan

    tokens = [3, 7, 62, 62, 5, 9]
    span = ImageSpan(2, 4, enc.features, "cd" * 32, enc.deepstack)
    with Engine(model) as engine:
        out = [
            e.token
            for e in engine.submit(
                GenerationRequest(tokens, max_tokens=5, spans=(span,))
            )
        ]
    assert out == reference(model, tokens, [span], 5)


TWO_PARTS = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "w1"},
            {"type": "image"},
            {"type": "image"},
        ],
    }
]


def _warnings(caplog):
    return [
        r
        for r in caplog.records
        if r.name == "beam.vision" and r.levelname == "WARNING"
    ]


def test_a_processor_failure_is_the_client_s_refusal_with_its_traceback_logged(
    caplog,
):
    """Whatever the processor raises is the client's 400 - transformers
    reports a literal placeholder in the text with a bare StopIteration -
    but its traceback goes to the log, in case the engine was at fault."""
    import logging

    class Choking(FakeProcessor):
        def __call__(self, text, images=None, return_tensors="np", **kw):
            raise StopIteration()

    front = VisionFrontend(Choking([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    with caplog.at_level(logging.WARNING, logger="beam.vision"):
        with pytest.raises(
            Exception, match=r"the processor refused the request \(StopIteration"
        ) as exc:
            front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is ValueError
    (record,) = _warnings(caplog)
    assert record.exc_info[0] is StopIteration
    assert front.images_encoded == 0


def test_a_renderer_failure_is_the_client_s_refusal_too():
    """mistral-common refuses a message with its own exception type."""

    class TokenizerException(Exception):
        pass

    class Backend:
        def apply_chat_template(self, messages, tokenize=True, return_dict=True, **kw):
            raise TokenizerException("Invalid assistant message")

    # The tower is never reached: the renderer refuses first.
    front = VisionFrontend(None, object(), "mistral3", renderer=Backend())
    with pytest.raises(
        Exception,
        match=r"the renderer refused the request \(TokenizerException: Invalid",
    ) as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is ValueError


def test_more_images_than_image_parts_are_refused_before_the_renderer():
    """The engine makes the image part and the image in one step; a count
    that differs is its own mistake, not the client's - and it is caught
    before mistral-common renders."""

    class Backend:
        def __init__(self):
            self.seen = []

        def apply_chat_template(self, messages, tokenize=True, return_dict=True, **kw):
            self.seen.append(messages)
            return {"input_ids": [2, 6, 3, 62, 60, 7], "pixel_values": []}

    backend = Backend()
    front = VisionFrontend(None, object(), "mistral3", renderer=backend)
    one_part = [{"role": "user", "content": [{"type": "image"}]}]
    with pytest.raises(Exception, match="2 images for 1 image parts") as exc:
        front.build(one_part, [png(1), png(2)], {})
    assert exc.type is RuntimeError
    assert backend.seen == []


@pytest.mark.parametrize(
    "raised",
    [ValueError("shapes do not broadcast"), StopIteration()],
    ids=["ValueError", "StopIteration"],
)
def test_a_tower_failure_is_the_server_s_error_whatever_its_type(raised):
    """MLX reports a shape error as ValueError, the client's type; a
    contextmanager generator would hand a StopIteration on as itself.
    Both come out of the tower as RuntimeError with the original as cause."""
    front = VisionFrontend(FakeProcessor([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")

    def failing(pixel_values, grid_thw):
        raise raised

    front.tower.encode = failing
    with pytest.raises(Exception, match="the vision tower failed") as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is RuntimeError and exc.value.__cause__ is raised


def test_an_image_part_the_template_drops_is_refused_before_the_processor():
    """A template that renders no placeholder for an image part (Qwen3.5's
    drops the image of an assistant turn before the last user message when
    the image comes before a `</think>` in that turn's text): the processor
    would still take the image and carry its grid, and the mismatch would
    surface in the engine. It is the client's 400, found in the text."""

    class Swallowing(FakeProcessor):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kw
        ):
            return Rendered(messages, self.with_bos, add_generation_prompt, 1)

        def __call__(self, text, images=None, return_tensors="np", **kw):
            out = super().__call__(text, images, return_tensors, **kw)
            ids = list(out["input_ids"][0])
            t, h, w = self.grids[1]
            image = TOWER_CONFIG["image_token_id"]
            last = max(i for i, v in enumerate(ids) if v == image)
            del ids[last + 1 - t * h * w // MERGE**2 : last + 1]
            out["input_ids"] = np.array([ids])
            return out

    proc = Swallowing([(1, 4, 4), (1, 2, 6)])
    front = VisionFrontend(proc, tower(), "qwen3_vl")
    with pytest.raises(
        ValueError, match="the chat template placed 1 image placeholders for 2 images"
    ):
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert proc.calls == [] and front.images_encoded == 0


def test_a_grid_too_many_from_the_processor_is_the_engine_s_error():
    """The processor's grids for images whose count the text already
    confirmed: one more is the processor's fault, not the client's."""

    class Padding(FakeProcessor):
        def __call__(self, text, images=None, return_tensors="np", **kw):
            out = super().__call__(text, images, return_tensors, **kw)
            extra = (1, 2, 2)
            out["pixel_values"] = np.concatenate([out["pixel_values"], pixels(extra)])
            out["image_grid_thw"] = np.concatenate([out["image_grid_thw"], [extra]])
            return out

    front = VisionFrontend(Padding([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    with pytest.raises(Exception, match="do not match the images") as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is RuntimeError


def test_a_tower_token_id_that_is_not_the_placeholder_is_the_engine_s_error():
    """The template placed its placeholders (counted), the processor
    expanded them; a tower looking for another id finds none - the
    positions report it, as the engine's error."""
    front = VisionFrontend(FakeProcessor([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    front.tower.image_token_id = 63
    with pytest.raises(Exception, match="the image positions failed") as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is RuntimeError
    assert isinstance(exc.value.__cause__, ValueError)
    assert "fewer image placeholders" in str(exc.value.__cause__)


def test_the_template_s_own_verdict_passes_unwrapped_and_unlogged(caplog):
    import logging

    from jinja2.exceptions import TemplateError

    class Strict(FakeProcessor):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kw
        ):
            raise TemplateError("roles must alternate")

    front = VisionFrontend(Strict([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    with caplog.at_level(logging.WARNING, logger="beam.vision"):
        with pytest.raises(TemplateError, match="roles must alternate") as exc:
            front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is TemplateError and _warnings(caplog) == []


def test_a_renderer_s_mismatched_pixel_arrays_are_the_server_s_error():
    """mistral-common's images come back one array each and are padded to
    one size for the tower; arrays that do not fit together (a channel
    count that differs) are the renderer's fault, not the client's."""

    class Backend:
        def apply_chat_template(self, messages, tokenize=True, return_dict=True, **kw):
            return {
                "input_ids": [2, 6, 62, 62, 7],
                "pixel_values": [np.zeros((3, 4, 4)), np.zeros((2, 4, 4))],
                "image_sizes": [[4, 4], [4, 4]],
            }

    front = VisionFrontend(None, object(), "mistral3", renderer=Backend())
    with pytest.raises(Exception, match="the renderer's output failed") as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is RuntimeError
    assert isinstance(exc.value.__cause__, ValueError)


def test_a_generation_prompt_that_cannot_be_located_is_logged(caplog):
    """`assistant_start` 0 is the whole prompt: the frame is not located,
    so the reasoning check and the parser read the whole prompt, and no
    boundary can be claimed at the generation prompt. The log says why -
    the template would not render without the generation prompt, what it
    rendered is no prefix of the prompt, or the frame's ids are not the
    prompt's tail."""
    import logging

    class Refusing(FakeProcessor):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kw
        ):
            if not add_generation_prompt:
                raise ValueError("add_generation_prompt is required")
            return super().apply_chat_template(
                messages, tokenize, add_generation_prompt, **kw
            )

    class Elsewhere(FakeProcessor):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kw
        ):
            rendered = super().apply_chat_template(
                messages, tokenize, add_generation_prompt, **kw
            )
            return rendered if add_generation_prompt else "<other>" + rendered

    class Mismatched(FakeProcessor):
        class Tok(FakeProcessor.Tok):
            def encode(self, text, add_special_tokens=True):
                return (
                    [8] if text == "<gen>" else super().encode(text, add_special_tokens)
                )

        tokenizer = Tok()

    cases = (
        (Refusing, "will not render without the generation prompt (ValueError"),
        (Elsewhere, "does not start"),
        (Mismatched, "not the prompt's tail"),
    )
    for cls, why in cases:
        front = VisionFrontend(cls([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="beam.vision"):
            built = front.build(TWO_PARTS, [png(1), png(2)], {})
        assert built.assistant_start == 0
        logged = [r.getMessage() for r in _warnings(caplog)]
        assert len(logged) == 1 and "assistant_start 0" in logged[0], logged
        assert why in logged[0], logged


def test_a_template_failure_is_the_client_s_refusal_naming_the_template():
    """A template that fails with another type than its own verdict is
    still the client's 400, and the message names the template, not the
    processor."""

    class Broken(FakeProcessor):
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True, **kw
        ):
            raise KeyError("bos_token")

    front = VisionFrontend(Broken([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    with pytest.raises(
        Exception, match=r"the template refused the request \(KeyError"
    ) as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is ValueError and exc.value.__cause__.__class__ is KeyError


def test_a_processor_output_without_integer_ids_is_the_server_s_error():
    """The processor took the request; ids that are no integers are its
    output's fault, the server's 500 - named as the output, not the tower."""

    class Garbled(FakeProcessor):
        def __call__(self, text, images=None, return_tensors="np", **kw):
            out = super().__call__(text, images, return_tensors, **kw)
            out["input_ids"] = np.full_like(out["input_ids"], np.nan, dtype=float)
            return out

    front = VisionFrontend(Garbled([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    with pytest.raises(Exception, match="the processor's output failed") as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is RuntimeError and isinstance(exc.value.__cause__, ValueError)


def test_a_processor_output_without_the_pixel_arrays_is_the_output_s_error():
    """The family reads the pixel arrays out of the processor's output on
    the way to the tower; a key missing there is the output's fault, the
    server's 500 - named as the output, not the tower."""

    class Gridless(FakeProcessor):
        def __call__(self, text, images=None, return_tensors="np", **kw):
            out = super().__call__(text, images, return_tensors, **kw)
            del out["image_grid_thw"]
            return out

    front = VisionFrontend(Gridless([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    with pytest.raises(
        Exception, match=r"the processor's output failed \(KeyError"
    ) as exc:
        front.build(TWO_PARTS, [png(1), png(2)], {})
    assert exc.type is RuntimeError and exc.value.__cause__.__class__ is KeyError


THINK_START, THINK_END = _core_tests.THINK_START, _core_tests.THINK_END


class ThinkOffRendered(str):
    """`Rendered` with the empty think block behind the frame."""

    def __new__(cls, rendered):
        self = super().__new__(cls, rendered + "<think></think>")
        self.messages = rendered.messages
        return self


class ThinkOffFakeProcessor(FakeProcessor):
    """A template like Qwen3's with thinking off: the generation prompt
    carries an empty think block behind the assistant header (three
    tokens), the finished turn in the history does not - so the prompt end
    is no prefix of the next round."""

    class Tok(FakeProcessor.Tok):
        def encode(self, text, add_special_tokens=True):
            if text == "<gen><think></think>":
                return [7, THINK_START, THINK_END]
            return super().encode(text, add_special_tokens)

    tokenizer = Tok()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kw
    ):
        rendered = super().apply_chat_template(
            messages, tokenize, add_generation_prompt, **kw
        )
        if add_generation_prompt and kw.get("enable_thinking") is False:
            return ThinkOffRendered(rendered)
        return rendered

    def __call__(self, text, images=None, return_tensors="np", **kw):
        out = super().__call__(text, images, return_tensors, **kw)
        if isinstance(text[0], ThinkOffRendered):
            ids = out["input_ids"][0].tolist() + [THINK_START, THINK_END]
            out["input_ids"] = np.array([ids])
        return out


def test_an_image_turn_resumes_from_the_generation_prompt():
    """Round two renders the first answer without the think block the
    prompt had asked for, so the entry's checkpoint at the prompt end lies
    behind the point the rounds agree on; the recurrent layers resume from
    the boundary where the generation prompt begins - with a cold run's
    tokens."""
    import base64

    model = tiny_qwen35()
    front = VisionFrontend(ThinkOffFakeProcessor([(1, 4, 4)]), tower(), "qwen3_vl")
    tok = StubTokenizer()
    data = base64.b64encode(png(3).data).decode()
    asked = {
        "role": "user",
        "content": [
            {"type": "text", "text": "w3 w5"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{data}"},
            },
        ],
    }

    def generation(messages):
        req = chat.parse_chat_request(
            {"messages": messages, "max_tokens": 4, "enable_thinking": False},
            "m",
            vision=True,
        )
        return chat.to_generation_request(tok, req, frontend=front), req

    gen1, req1 = generation([asked])
    start = req1.assistant_start
    assert gen1.tokens[start:] == [7, THINK_START, THINK_END]
    # The history carries the answer as the client sends it back - a fixed
    # one, so the rounds part exactly where the frame's think block begins.
    turn2 = [
        asked,
        {"role": "assistant", "content": "w4"},
        {"role": "user", "content": "w9"},
    ]
    with Engine(model) as engine:
        [e.token for e in engine.submit(gen1)]
        second = engine.submit(generation(turn2)[0])
        out2 = [e.token for e in second]
    assert second.prompt_cached == start
    with Engine(model) as cold:
        assert out2 == [e.token for e in cold.submit(generation(turn2)[0])]


def _serializable_part(name, base, **defaults):
    """A part fake transformers can serialize: `from_args_and_dict` logs
    the processor's repr, which runs every part's `to_dict`; `valid_kwargs`
    names the keys the builder may fill from the model."""
    kw = type(f"{name}Kwargs", (), {"__annotations__": {k: object for k in defaults}})

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    return type(name, (base,), {**defaults, "valid_kwargs": kw, "__init__": __init__})


@pytest.mark.parametrize("parts", ["three", "four", "audio"])
def test_a_missing_video_processor_does_not_block_images(
    tmp_path, monkeypatch, caplog, parts
):
    """A processor_config.json that carries the image processor (and the
    audio feature extractor) but no video block makes AutoProcessor fail
    on the video processor it cannot find. The loader then puts the
    processor together itself: every part with a block in the file the
    way transformers builds it, only the missing one from the model and
    the class defaults, the top-level values and the chat template from
    the file (the feature extractor's block under either name transformers
    reads it from); the health says what was built. A part that cannot be
    built is the original refusal with the cause behind it, both in the
    health, the traceback in the log."""
    import json
    import logging

    import transformers
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.image_processing_base import ImageProcessingMixin
    from transformers.processing_utils import ProcessorMixin

    Images = _serializable_part(
        "FakeImageProcessor", ImageProcessingMixin, size={"height": 896}, patch_size=14
    )
    Videos = _serializable_part(
        "FakeVideoProcessor", ImageProcessingMixin, patch_size=16, max_frames=768
    )
    Audio = _serializable_part(
        "FakeFeatureExtractor", FeatureExtractionMixin, sampling_rate=16000
    )

    class ThreeParts(ProcessorMixin):
        def __init__(
            self,
            image_processor=None,
            tokenizer=None,
            video_processor=None,
            chat_template=None,
        ):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.video_processor = video_processor
            self.chat_template = chat_template

    class FourParts(ProcessorMixin):
        def __init__(
            self,
            feature_extractor=None,
            image_processor=None,
            tokenizer=None,
            video_processor=None,
            chat_template=None,
            audio_ms_per_token=40,
        ):
            self.feature_extractor = feature_extractor
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.video_processor = video_processor
            self.chat_template = chat_template
            self.audio_ms_per_token = audio_ms_per_token

    processor_cls = ThreeParts if parts == "three" else FourParts
    blocks = {"four": "feature_extractor", "audio": "audio_processor"}
    image_block = {
        "image_processor_type": "FakeImageProcessor",
        "size": {"height": 448},
        "patch_size": 14,
    }
    audio_block = {
        "feature_extractor_type": "FakeFeatureExtractor",
        "sampling_rate": 24000,
    }
    file = {"processor_class": processor_cls.__name__, "image_processor": image_block}
    if parts != "three":
        file.update({blocks[parts]: audio_block, "audio_ms_per_token": 80})
    (tmp_path / "processor_config.json").write_text(json.dumps(file))
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    config = {"model_type": "gemma4_unified", "vision_config": {"patch_size": 16}}

    def from_file(name, cls):
        def from_pretrained(path, trust_remote_code=False):
            file = json.loads((Path(path) / "processor_config.json").read_text())
            return cls(**(file.get(name) or file["audio_processor"]))

        return staticmethod(from_pretrained)

    def refused(path, trust_remote_code=False):
        raise OSError("Can't load video processor")

    monkeypatch.setattr(
        frontend, "classes_for", lambda t: (processor_cls, Images, Videos, Audio)
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, trust_remote_code=False: FakeTokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", staticmethod(refused)
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        from_file("image_processor", Images),
    )
    monkeypatch.setattr(
        transformers.AutoFeatureExtractor,
        "from_pretrained",
        from_file("feature_extractor", Audio),
    )
    caplog.set_level(logging.WARNING, logger="beam.vision")
    processor, info = frontend.load_processor(tmp_path, config)
    assert isinstance(processor, processor_cls)
    # The log keeps AutoProcessor's own failure beside what was assembled.
    assembled = [
        r.getMessage() for r in caplog.records if "assembled from" in r.getMessage()
    ]
    assert len(assembled) == 1 and "Can't load video processor" in assembled[0]
    assert processor.image_processor.kwargs == image_block
    assert isinstance(processor.video_processor, Videos)
    assert processor.video_processor.kwargs == {"patch_size": 16}
    assert processor.chat_template == "{{ messages }}"
    assert info["source"] == "checkpoint" and info["class"] == processor_cls.__name__
    assert info["built"] == ["video_processor"]
    if parts != "three":
        assert processor.feature_extractor.kwargs == audio_block
        assert processor.audio_ms_per_token == 80

    def no_backend(path, trust_remote_code=False):
        raise ValueError("no backend for this image processor")

    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", staticmethod(no_backend)
    )
    caplog.set_level(logging.WARNING, logger="beam.vision")
    with pytest.raises(OSError, match="video processor") as exc:
        frontend.load_processor(tmp_path, config)
    assert isinstance(exc.value.__cause__, ValueError)
    assert "no backend" in str(exc.value.__cause__)
    assert str(exc.value).startswith("assembling: ValueError: no backend")
    assert "AutoProcessor: Can't load video processor" in str(exc.value)
    record = next(r for r in caplog.records if r.name == "beam.vision" and r.exc_info)
    assert record.exc_info[0] is ValueError


MUSE_CONFIG = {
    "model_type": "muse_glimmer",
    "image_token_id": 62,
    "out_hidden_size": 128,
    # 64, not 48: the projection's input must divide by a group size.
    "projector_hidden_size": 64,
    "vision_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "patch_size": 14,
        "patch_temporal": 2,
        "merge_size": 2,
        "pos_emb_height": 4,
        "pos_emb_width": 4,
    },
    "text_config": {"hidden_size": 32, "rms_norm_eps": 1e-5},
}


def test_a_quantized_root_module_loads(tmp_path):
    """An 8-bit conversion packs Muse's projection, a bare linear that is a
    part of its own: there is no child to swap for a quantized layer, the
    module itself is the layer. `fit` swaps it and hands the quantized
    module back; the family keeps what it gets, and the features match
    the quantized source."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_beam_vision.families import Loaded, muse_glimmer

    mx.random.seed(5)
    packed = nn.Linear(64, 32, bias=False).to_quantized(group_size=32, bits=8)
    mx.eval(packed.parameters())
    stored = {f"vision_projection.{k}": v for k, v in tree_flatten(packed.parameters())}
    bare = tmp_path / "bare"
    bare.mkdir()
    mx.save_safetensors(str(bare / "model.safetensors"), stored)
    quantization = {"quantization": {"group_size": 32, "bits": 8}}
    loaded = Loaded(bare, ("vision_projection",), quantization)
    fitted = loaded.fit(
        nn.Linear(64, 32, bias=False), loaded.part("vision_projection"), mx.float32
    )
    assert isinstance(fitted, nn.QuantizedLinear)
    assert (fitted.bits, fitted.group_size) == (8, 32)
    x = mx.random.normal((3, 64))
    assert mx.allclose(fitted(x), packed(x), atol=1e-5)

    mx.random.seed(3)
    source = muse_glimmer.Tower(MUSE_CONFIG, None, dtype=mx.float32)
    nn.quantize(source.vision_adapter, group_size=32, bits=8)
    source.vision_projection = source.vision_projection.to_quantized(
        group_size=32, bits=8
    )
    mx.eval(
        source.model.parameters(),
        source.vision_adapter.parameters(),
        source.vision_projection.parameters(),
    )
    stored = {
        **{f"vision_tower.{k}": v for k, v in tree_flatten(source.model.parameters())},
        **{
            f"vision_adapter.{k}": v
            for k, v in tree_flatten(source.vision_adapter.parameters())
        },
        **{
            f"vision_projection.{k}": v
            for k, v in tree_flatten(source.vision_projection.parameters())
        },
    }
    muse = tmp_path / "muse"
    muse.mkdir()
    mx.save_safetensors(str(muse / "model.safetensors"), stored)
    t = muse_glimmer.Tower({**MUSE_CONFIG, **quantization}, muse, dtype=mx.float32)
    assert t.quantized and isinstance(t.vision_projection, nn.QuantizedLinear)
    assert isinstance(t.vision_adapter.fc1, nn.QuantizedLinear)
    grids = [(1, 4, 4), (1, 2, 4)]
    n = sum(g[0] * g[1] * g[2] for g in grids)
    pv = mx.array(np.random.RandomState(0).randn(n, 3 * 2 * 14 * 14).astype(np.float32))
    want = source.encode(pv, mx.array(grids))
    got = t.encode(pv, mx.array(grids))
    for a, b in zip(got, want, strict=True):
        assert mx.allclose(a.features, b.features, atol=1e-5)


MISTRAL3_CONFIG = {
    "model_type": "mistral3",
    "vision_config": {
        "model_type": "pixtral",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "head_dim": 16,
        "image_size": 64,
        "patch_size": 16,
        "num_channels": 3,
    },
    "text_config": {"hidden_size": 32, "rms_norm_eps": 1e-5},
    "quantization": {"group_size": 32, "bits": 8},
}


def _mistral3_mini():
    """A tiny Mistral 3 tower with its projector's three linears packed
    (norm and tower in floating point), and its tensors in mlx-vlm's
    layout as a conversion stores them: (tower, projector) by name."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_beam_vision.families import mistral3

    mx.random.seed(1)
    source = mistral3.Tower(MISTRAL3_CONFIG, None, dtype=mx.float32)
    nn.quantize(source.projector, group_size=32, bits=8)
    mx.eval(source.model.parameters(), source.projector.parameters())
    tower = {
        f"vision_tower.{k}": (
            v.transpose(0, 3, 1, 2) if "patch_conv.weight" in k else v
        )
        for k, v in tree_flatten(source.model.parameters())
    }
    projector = {
        f"multi_modal_projector.{k}": v
        for k, v in tree_flatten(source.projector.parameters())
    }
    return source, tower, projector


def _mistral3_features(t):
    pv = mx.array(np.random.RandomState(0).randn(1, 64, 64, 3).astype(np.float32))
    return t.encode(pv, [(64, 64)])[0].features


def _write_index(path, weight_map):
    import json

    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )


def _spy_on_load(monkeypatch):
    seen = []
    real = mx.load

    def load(path, *args, **kwargs):
        seen.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(mx, "load", load)
    return seen


def test_mistral3_projector_loads_partly_quantized(tmp_path):
    """A conversion that packs the projector's linears and leaves the
    tower and the projector's norm in floating point loads as such, with
    the features of the source."""
    import mlx.nn as nn
    from mlx_beam_vision.families import mistral3

    source, tower, projector = _mistral3_mini()
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {**tower, **projector})
    t = mistral3.Tower(MISTRAL3_CONFIG, tmp_path, dtype=mx.float32)
    assert t.quantized and isinstance(t.projector.linear_2, nn.QuantizedLinear)
    assert isinstance(t.projector.norm, nn.RMSNorm)
    assert mx.allclose(_mistral3_features(t), _mistral3_features(source), atol=1e-5)


SHARD_1, SHARD_2, SHARD_3 = (f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3))


@pytest.mark.parametrize("index", ["missing_shard", "no_names", "no_hits"])
def test_a_stale_index_is_dropped_for_the_shards_present(tmp_path, monkeypatch, index):
    """A checkpoint whose index belongs to another conversion - it names
    shards that are not there, or none of the names it has for the tower
    is in the shard it points at - is loaded from the files that are
    there: the headers of `model*.safetensors` say which hold tower
    tensors, only those are read, the health says the index was stale.
    What the files hold must still be complete: strict loading refuses a
    packed layer without its biases."""
    from mlx_beam_vision.families import mistral3

    source, tower, projector = _mistral3_mini()
    text = {"language_model.embed_tokens.weight": mx.zeros((4, 8))}
    # Index names as another conversion writes them: the tower without
    # the `vision_model` level, `.weight` only.
    other_names = [
        k.replace("vision_tower.vision_model.", "vision_tower.")
        for k in (*tower, *projector)
        if k.endswith(".weight")
    ]
    weight_map = {
        "missing_shard": {k: "model-00001-of-00006.safetensors" for k in other_names},
        "no_names": {
            f"language_model.layers.{i}.weight": s
            for i, s in enumerate((SHARD_1, SHARD_2, SHARD_3))
        },
        "no_hits": {k: SHARD_1 for k in other_names},
    }[index]

    def checkpoint(name, shard_3):
        path = tmp_path / name
        path.mkdir()
        mx.save_safetensors(str(path / SHARD_1), tower)
        mx.save_safetensors(str(path / SHARD_2), text)
        mx.save_safetensors(str(path / SHARD_3), shard_3)
        _write_index(path, weight_map)
        return path

    seen = _spy_on_load(monkeypatch)
    t = mistral3.Tower(
        MISTRAL3_CONFIG, checkpoint("whole", projector), dtype=mx.float32
    )
    assert mx.allclose(_mistral3_features(t), _mistral3_features(source), atol=1e-5)
    assert t.loaded_from == [SHARD_1, SHARD_3]
    assert t.describe().get("index") == "stale"
    assert SHARD_2 not in seen
    short = {k: v for k, v in projector.items() if not k.endswith("linear_2.biases")}
    with pytest.raises(ValueError, match="biases"):
        mistral3.Tower(MISTRAL3_CONFIG, checkpoint("short", short), dtype=mx.float32)


def test_a_valid_index_is_followed(tmp_path, monkeypatch):
    """An index that matches the files is followed as before: the tower
    comes from the file the index names (a package's own, in a
    subfolder), the text shard is never opened, the health says so; a
    single file without an index is read as before, and the health says
    that too. An index without the tower, and no file holding one either,
    is the plain refusal - nothing about a stale index."""
    from mlx_beam_vision.families import mistral3

    source, tower, projector = _mistral3_mini()
    text = {"language_model.embed_tokens.weight": mx.zeros((4, 8))}
    text_shard = "model-00001-of-00001.safetensors"
    packed = tmp_path / "packed"
    (packed / "optiq").mkdir(parents=True)
    mx.save_safetensors(
        str(packed / "optiq" / "optiq_vision.safetensors"), {**tower, **projector}
    )
    mx.save_safetensors(str(packed / text_shard), text)
    _write_index(
        packed,
        {
            **{k: "optiq/optiq_vision.safetensors" for k in (*tower, *projector)},
            **{k: text_shard for k in text},
        },
    )
    seen = _spy_on_load(monkeypatch)
    t = mistral3.Tower(MISTRAL3_CONFIG, packed, dtype=mx.float32)
    assert mx.allclose(_mistral3_features(t), _mistral3_features(source), atol=1e-5)
    assert t.loaded_from == ["optiq/optiq_vision.safetensors"]
    assert t.describe().get("index") == "followed"
    assert text_shard not in seen

    single = tmp_path / "single"
    single.mkdir()
    mx.save_safetensors(str(single / "model.safetensors"), {**tower, **projector})
    t = mistral3.Tower(MISTRAL3_CONFIG, single, dtype=mx.float32)
    assert t.loaded_from == ["model.safetensors"]
    assert t.describe().get("index") == "none"

    bare = tmp_path / "bare"
    bare.mkdir()
    mx.save_safetensors(str(bare / text_shard), text)
    _write_index(bare, {k: text_shard for k in text})
    with pytest.raises(FileNotFoundError, match="no tensors under") as exc:
        mistral3.Tower(MISTRAL3_CONFIG, bare, dtype=mx.float32)
    assert "stale" not in str(exc.value) and "dropped" not in str(exc.value)


def _auto_processor_refuses(monkeypatch, what="Can't load video processor"):
    import transformers

    def refused(path, trust_remote_code=False):
        raise OSError(what)

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", staticmethod(refused)
    )


def test_a_file_naming_another_processor_class_is_refused_with_the_reason(
    tmp_path, monkeypatch
):
    """processor_config.json names a processor class other than the one
    transformers has for the model type: nothing is assembled, and the
    refusal - what reaches /health.vision.refused - carries AutoProcessor's
    text and the reason."""
    import json

    from transformers.image_processing_base import ImageProcessingMixin
    from transformers.processing_utils import ProcessorMixin

    class Expected(ProcessorMixin):
        def __init__(self, image_processor=None, tokenizer=None, chat_template=None):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.chat_template = chat_template

    Images = _serializable_part("FakeImageProcessor", ImageProcessingMixin, size=1)
    file = {
        "processor_class": "AnotherProcessor",
        "image_processor": {"image_processor_type": "FakeImageProcessor"},
    }
    (tmp_path / "processor_config.json").write_text(json.dumps(file))
    monkeypatch.setattr(
        frontend, "classes_for", lambda t: (Expected, Images, None, None)
    )
    _auto_processor_refuses(monkeypatch)
    with pytest.raises(OSError, match="video processor") as exc:
        frontend.load_processor(tmp_path, {"model_type": "x", "vision_config": {}})
    assert "assembling: ValueError: processor_config.json names AnotherProcessor" in (
        str(exc.value)
    )
    assert isinstance(exc.value.__cause__, ValueError)


def test_a_file_with_a_block_for_every_part_is_not_assembled(tmp_path, monkeypatch):
    """AutoProcessor fails on a file that has a block for every part the
    class takes: the failure is not a missing part, so no other processor
    is built out of the file, and the refusal says so."""
    import json

    import transformers
    from transformers.image_processing_base import ImageProcessingMixin
    from transformers.processing_utils import ProcessorMixin

    Images = _serializable_part("FakeImageProcessor", ImageProcessingMixin, size=1)
    Videos = _serializable_part("FakeVideoProcessor", ImageProcessingMixin, size=2)

    class TwoParts(ProcessorMixin):
        def __init__(
            self,
            image_processor=None,
            tokenizer=None,
            video_processor=None,
            chat_template=None,
        ):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.video_processor = video_processor
            self.chat_template = chat_template

    file = {
        "processor_class": "TwoParts",
        "image_processor": {"image_processor_type": "FakeImageProcessor"},
        "video_processor": {"video_processor_type": "FakeVideoProcessor"},
    }
    (tmp_path / "processor_config.json").write_text(json.dumps(file))
    monkeypatch.setattr(
        frontend, "classes_for", lambda t: (TwoParts, Images, Videos, None)
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda path, trust_remote_code=False: FakeTokenizer()),
    )
    for auto, cls in (
        (transformers.AutoImageProcessor, Images),
        (transformers.AutoVideoProcessor, Videos),
    ):
        monkeypatch.setattr(
            auto,
            "from_pretrained",
            staticmethod(lambda path, trust_remote_code=False, cls=cls: cls()),
        )
    _auto_processor_refuses(monkeypatch, "Can't instantiate the processor")
    with pytest.raises(OSError, match="instantiate the processor") as exc:
        frontend.load_processor(tmp_path, {"model_type": "x", "vision_config": {}})
    assert "assembling: ValueError: every part the class takes has a block" in str(
        exc.value
    )


def test_an_unreadable_shard_header_stops_the_scan_with_its_name(tmp_path):
    """Once the index is dropped, the headers of every `model*.safetensors`
    file are read; one that cannot be read is refused by name, not skipped
    - the files behind it might have held the tower."""
    from mlx_beam_vision.families import mistral3

    _, tower, projector = _mistral3_mini()
    mx.save_safetensors(str(tmp_path / SHARD_1), {**tower, **projector})
    (tmp_path / SHARD_2).write_bytes(b"\xff" * 16)
    _write_index(tmp_path, {k: "model-00001-of-00006.safetensors" for k in tower})
    with pytest.raises(Exception, match="not a readable safetensors header") as exc:
        mistral3.Tower(MISTRAL3_CONFIG, tmp_path, dtype=mx.float32)
    assert exc.type is ValueError and SHARD_2 in str(exc.value)
