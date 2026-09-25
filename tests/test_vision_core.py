"""The core's side of images, without any tower: a fake frontend hands the
engine image spans (features for the placeholder positions, per-layer
extras for DeepStack), and the prefill embeds them - the same forward a
direct call with input embeddings and the layer hook makes; the prefix
cache keys on the image's digest; the API decodes data URLs and refuses
the rest; without a frontend an image is the 400 naming the extra."""

import base64
import hashlib

import mlx.core as mx
import pytest

from mlx_beam import modalities
from mlx_beam.api import chat
from mlx_beam.api.errors import ApiError
from mlx_beam.engine import Engine, GenerationRequest, ImageSpan
from mlx_beam.engine.kv import KVPolicy, make_request_cache
from mlx_beam.engine.proposer import trunk
from mlx_beam.modalities import Built, Image
from mlx_beam.server import Served
from tests.stub_tokenizer import StubTokenizer
from tests.test_speculative import tiny_qwen35

PAD = 62  # the placeholder id a fake processor expands an image into
K = 3  # placeholders per image
H = 32


def features_for(image: Image, layers: int = 2):
    """Deterministic 'tower': the digest seeds the features."""
    seed = int(image.digest[:8], 16)
    mx.random.seed(seed)
    feats = mx.random.normal((K, H))
    # Ahead of layers 1 and 2 (what a Qwen tower adds after layers 0 and 1).
    extras = {i + 1: mx.random.normal((K, H)) * 0.1 for i in range(layers)}
    mx.eval(feats, *extras.values())
    return feats, extras


class FakeFrontend:
    """Text tokens through the stub tokenizer, every image part K pads."""

    name = "fake"

    def __init__(self, tokenizer, deepstack=True):
        self._tok = tokenizer
        self.deepstack = deepstack
        self.builds = 0

    def build(self, messages, images, template_kwargs):
        self.builds += 1
        tokens, spans = [2], []
        it = iter(images)
        for m in messages:
            tokens.append(
                {"system": 5, "user": 6, "assistant": 7, "tool": 8}[m["role"]]
            )
            content = m.get("content") or ""
            parts = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": content}]
            )
            for part in parts:
                if part["type"] == "text":
                    tokens += self._tok.encode(part["text"])
                else:
                    image = next(it)
                    feats, extras = features_for(image)
                    spans.append(
                        ImageSpan(
                            len(tokens),
                            len(tokens) + K,
                            feats,
                            image.digest,
                            extras if self.deepstack else {},
                        )
                    )
                    tokens += [PAD] * K
        tokens.append(7)
        return Built(tokens, spans)

    def describe(self):
        return {"provider": "fake", "tower": "none"}


def reference(model, tokens, spans, n):
    """Greedy tokens from a direct forward: embeddings with the spans'
    features scattered in, the extras added after the first layers."""
    inner, lm_head, embed = trunk(model)
    cache = make_request_cache(model, KVPolicy())
    ids = mx.array([tokens])
    h = embed(ids)
    for s in spans:
        h[0, s.start : s.end, :] = s.features.astype(h.dtype)

    def hook(i, x):
        for s in spans:
            extra = s.extras.get(i)
            if extra is not None:
                x[0, s.start : s.end, :] = x[0, s.start : s.end, :] + extra.astype(
                    x.dtype
                )
        return x

    out = lm_head(inner(ids, cache=cache, input_embeddings=h, layer_hook=hook))
    result = []
    y = mx.argmax(out[0, -1])
    for _ in range(n):
        result.append(int(y.item()))
        out = lm_head(inner(y.reshape(1, 1), cache=cache))
        y = mx.argmax(out[0, -1])
    return result


def png(seed: int) -> Image:
    return Image(hashlib.sha256(str(seed).encode()).digest() * 4, "image/png")


def request(front, images, text="w1", n=6):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": text}]
            + [{"type": "image"}] * len(images),
        }
    ]
    built = front.build(messages, images, {})
    return (
        GenerationRequest(built.tokens, max_tokens=n, spans=tuple(built.spans)),
        built,
    )


def test_prefill_embeds_the_spans_like_a_direct_forward():
    model = tiny_qwen35()
    front = FakeFrontend(StubTokenizer())
    img = png(1)
    req, built = request(front, [img])
    with Engine(model) as engine:
        out = [e.token for e in engine.submit(req)]
        plain = [
            e.token
            for e in engine.submit(GenerationRequest(built.tokens, max_tokens=6))
        ]
    assert out == reference(model, built.tokens, built.spans, 6)
    assert out != plain  # the placeholders' own embeddings give another text


def test_prefill_in_slices_and_beside_a_text_row():
    """The span straddles prefill chunks and the row shares the batch with
    a text-only row: still the direct forward's tokens."""
    model = tiny_qwen35()
    front = FakeFrontend(StubTokenizer())
    img = png(2)
    req, built = request(front, [img], text="w3 w5 w7 w9 w11 w13")
    assert built.spans[0].start > 4
    with Engine(model, prefill_step_size=4, prefill_slice=4) as engine:
        a = engine.submit(req)
        b = engine.submit(GenerationRequest([3, 7, 11, 13, 5, 9], max_tokens=6))
        out_a = [e.token for e in a]
        out_b = [e.token for e in b]
        solo_b = [
            e.token
            for e in engine.submit(
                GenerationRequest([3, 7, 11, 13, 5, 9], max_tokens=6)
            )
        ]
    assert out_a == reference(model, built.tokens, built.spans, 6)
    assert out_b == solo_b


def test_the_store_keys_on_the_image_digest():
    model = tiny_qwen35()
    front = FakeFrontend(StubTokenizer())
    with Engine(model) as engine:
        req1, built1 = request(front, [png(3)])
        req2, built2 = request(front, [png(4)])
        assert built1.tokens == built2.tokens  # same placeholders
        assert req1.cache_key != req2.cache_key
        assert all(
            k < 0 for k in req1.cache_key[built1.spans[0].start : built1.spans[0].end]
        )
        first = engine.submit(req1)
        [e.token for e in first]
        assert first.prompt_cached == 0
        other = engine.submit(req2)
        [e.token for e in other]
        assert other.prompt_cached == 0  # another image: no hit on the placeholders
        again = engine.submit(request(front, [png(3)])[0])
        [e.token for e in again]
        assert again.prompt_cached >= built1.spans[0].end


def test_api_decodes_data_urls_and_refuses_the_rest():
    data = base64.b64encode(b"\x89PNG fake").decode()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "w1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{data}"},
                    },
                ],
            }
        ]
    }
    with pytest.raises(ApiError) as exc:
        chat.parse_chat_request(body, "m")
    assert exc.value.code == "extra_not_installed"
    req = chat.parse_chat_request(body, "m", vision=True)
    assert len(req.images) == 1 and req.images[0].media_type == "image/png"
    assert req.messages[0]["content"] == [
        {"type": "text", "text": "w1"},
        {"type": "image"},
    ]
    remote = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                ],
            }
        ]
    }
    with pytest.raises(ApiError, match="only data: URLs"):
        chat.parse_chat_request(remote, "m", vision=True)
    tok = StubTokenizer()
    front = FakeFrontend(tok)
    with pytest.raises(ApiError) as exc:
        chat.build_prompt(tok, req)
    assert exc.value.code == "extra_not_installed"
    gen = chat.to_generation_request(tok, req, frontend=front)
    assert len(gen.spans) == 1 and gen.boundaries == [] and gen.system_end is None
    assert gen.tokens[gen.spans[0].start : gen.spans[0].end] == [PAD] * K


def test_served_reports_the_frontend_and_the_registry_finds_a_provider():
    model = tiny_qwen35()
    tok = StubTokenizer()
    front = FakeFrontend(tok)
    with Engine(model) as engine:
        served = Served(engine, tok, "tiny", frontend=front)
        h = served.health()
        assert h["capabilities"]["vision"] is True and h["vision"]["provider"] == "fake"
        bare = Served(engine, tok, "tiny")
        assert bare.health()["vision"] is None and not bare.capabilities()["vision"]

    class Provider:
        __name__ = "fake-provider"

        @staticmethod
        def supports(config):
            return config.get("model_type") == "qwen3_5"

        @staticmethod
        def load(model, model_path, config, tokenizer):
            return FakeFrontend(tokenizer)

    modalities.register(Provider)
    try:
        assert (
            modalities.load_frontend(model, None, {"model_type": "qwen3_5"}, tok).name
            == "fake"
        )
        assert (
            modalities.load_frontend(model, None, {"model_type": "llama"}, tok) is None
        )
    finally:
        modalities._registered.remove(Provider)


def test_a_model_that_norms_its_embeddings_takes_the_features_as_they_are():
    """Muse norms token embeddings before the first layer (embed_inputs);
    the prefill builds the text positions the same way and the image
    features go in untouched - one direct forward says the same."""
    from tests.test_model_classes import tiny

    model = tiny("muse_glimmer")
    inner, lm_head, _ = trunk(model)
    assert hasattr(inner, "embed_inputs")
    tokens = [3, 7, 11, PAD, PAD, PAD, 5, 9]
    mx.random.seed(11)
    feats = mx.random.normal((K, 64))
    feats = mx.fast.rms_norm(feats, None, 1e-5)
    span = ImageSpan(3, 6, feats, "ab" * 32)
    with Engine(model) as engine:
        out = [
            e.token
            for e in engine.submit(
                GenerationRequest(tokens, max_tokens=5, spans=(span,))
            )
        ]
    cache = make_request_cache(model, KVPolicy())
    ids = mx.array([tokens])
    h = inner.embed_inputs(ids)
    h[0, 3:6, :] = feats
    logits = lm_head(inner(ids, cache=cache, input_embeddings=h))
    want = []
    y = mx.argmax(logits[0, -1])
    for _ in range(5):
        want.append(int(y.item()))
        y = mx.argmax(lm_head(inner(y.reshape(1, 1), cache=cache))[0, -1])
    assert out == want
