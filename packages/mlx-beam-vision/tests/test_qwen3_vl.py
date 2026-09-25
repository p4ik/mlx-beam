"""The Qwen3-VL family on a tiny tower: pixel inputs to features per image
with DeepStack extras, the frontend's spans from the processor's
placeholder runs, the feature cache by digest, the provider's dispatch,
and the whole path through mlx-beam's engine against a direct forward."""

import hashlib
import io
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_beam import modalities
from mlx_beam.api import chat
from mlx_beam.engine import Engine
from mlx_beam.modalities import Image
from mlx_beam_vision import provider
from mlx_beam_vision.families import qwen3_vl
from mlx_beam_vision.frontend import VisionFrontend

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
    t = qwen3_vl.build(TOWER_CONFIG, None, dtype=mx.float32)
    mx.eval(t.model.parameters())
    return t


def pixels(grid):
    """Random pixel inputs for one image of grid (t, h, w) patches."""
    n = grid[0] * grid[1] * grid[2]
    return np.random.RandomState(n).randn(n, 3 * T * PATCH * PATCH).astype(np.float32)


class FakeProcessor:
    """Stands in for transformers' processor: the stub tokenizer's text,
    each image part `image_tokens` placeholders, random pixels."""

    def __init__(self, grids):
        self.grids = list(grids)
        self.tok = StubTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return messages

    def __call__(self, text, images=None, return_tensors="np"):
        messages = text[0]
        ids = [2]
        grids = iter(self.grids)
        used = []
        for m in messages:
            ids.append({"system": 5, "user": 6, "assistant": 7}[m["role"]])
            content = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]
            for part in content:
                if part["type"] == "text":
                    ids += self.tok.encode(part["text"])
                else:
                    g = next(grids)
                    used.append(g)
                    ids += [TOWER_CONFIG["image_token_id"]] * (g[0] * g[1] * g[2] // MERGE**2)
        ids.append(7)
        out = {"input_ids": np.array([ids])}
        if used:
            out["pixel_values"] = np.concatenate([pixels(g) for g in used])
            out["image_grid_thw"] = np.array(used)
        return out


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
    assert all(len(e.deepstack) == 1 and e.deepstack[0].shape == e.features.shape for e in encoded)
    d = t.describe()
    assert d["family"] == "qwen3_vl" and d["deepstack_layers"] == 1


def test_frontend_builds_spans_and_caches_by_digest():
    front = VisionFrontend(FakeProcessor([(1, 4, 4), (1, 2, 6)]), tower(), "qwen3_vl")
    messages = [{"role": "user", "content": [{"type": "text", "text": "w1"}, {"type": "image"}, {"type": "image"}]}]
    a, b = png(1), png(2)
    built = front.build(messages, [a, b], {})
    assert [(s.start, s.end) for s in built.spans] == [(3, 7), (7, 10)]
    assert built.tokens[3:10] == [TOWER_CONFIG["image_token_id"]] * 7
    assert built.spans[0].digest == a.digest and built.spans[1].deepstack[0].shape == (3, 32)
    assert front.images_encoded == 2
    again = front.build(messages, [a, b], {})
    assert front.images_encoded == 2 and front.cache.describe()["hits"] == 2
    assert mx.array_equal(again.spans[0].features, built.spans[0].features)
    d = front.describe()
    assert d["provider"] == "mlx-beam-vision" and d["feature_cache"]["entries"] == 2
    with pytest.raises(ValueError, match="do not match the images"):
        front.build(messages, [a, b, png(9)], {})
    with pytest.raises(ValueError, match="not a readable image"):
        front.build(messages, [Image(b"nope"), b], {})


def test_provider_dispatches_by_model_type():
    assert provider.supports(TOWER_CONFIG) and provider.family_for(TOWER_CONFIG) == "qwen3_vl"
    assert not provider.supports({"model_type": "qwen3_5"})  # no vision_config: text only
    assert not provider.supports({"model_type": "llama", "vision_config": {}})
    assert any(ep.name == "vision" for ep in modalities.entry_points(group=modalities.ENTRY_POINT_GROUP))


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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}},
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
    assert out == reference(model, gen.tokens, list(gen.spans), 6)
    assert h["alive"]
