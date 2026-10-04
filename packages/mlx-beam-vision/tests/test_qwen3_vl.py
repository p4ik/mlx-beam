"""The Qwen3-VL family on a tiny tower: pixel inputs to features per image
with DeepStack extras, the frontend's spans from the processor's
placeholder runs, the feature cache by digest, the provider's dispatch,
and the whole path through mlx-beam's engine against a direct forward."""

import io
import sys
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


class Rendered(str):
    """What the fake template renders: a str for the frontend's BOS check
    and assistant-frame probe (a real processor renders text), carrying
    the messages the fake tokenizer reads. The generation prompt is the
    frame `<gen>`, one token (7) in the fake's ids."""

    def __new__(cls, messages, with_bos, generation_prompt=True):
        base = "<bos>..." if with_bos else "..."
        self = super().__new__(cls, base + ("<gen>" if generation_prompt else ""))
        self.messages = messages
        return self


class FakeProcessor:
    """Stands in for transformers' processor: the stub tokenizer's text,
    each image part `image_tokens` placeholders, random pixels."""

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
    with pytest.raises(ValueError, match="do not match the images"):
        front.build(messages, [a, b, png(9)], {})
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
        def apply_chat_template(self, messages, **kw):
            return messages

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
        monkeypatch, (FakePixtralProcessor, PixtralImages, None), config, tmp_path
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
        monkeypatch, (FakeQwenProcessor, QwenImages, QwenVideos), config, tmp_path
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
        _built_from(monkeypatch, (OtherTokens, PixtralImages, None), config, tmp_path)


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
        frontend, "classes_for", lambda t: (FakePixtralProcessor, PixtralImages, None)
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
        def apply_chat_template(self, messages, **kw):
            return messages

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
