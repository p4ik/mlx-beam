"""The core's side of images, without any tower: a fake frontend hands the
engine image spans (features for the placeholder positions, per-layer
extras for DeepStack), and the prefill embeds them - the same forward a
direct call with input embeddings and the layer hook makes; the prefix
cache keys on the image's digest; the API decodes data URLs and refuses
the rest; without a frontend an image is the 400 naming the extra."""

import base64
import hashlib
import itertools

import mlx.core as mx
import pytest

from mlx_beam import modalities
from mlx_beam._vendor.mlx_lm.generate import BatchGenerator, _cache_arrays
from mlx_beam._vendor.mlx_lm.models.cache import BatchKVCache
from mlx_beam.api import chat
from mlx_beam.api.errors import ApiError
from mlx_beam.engine import ContextTooLong, Engine, GenerationRequest, ImageSpan
from mlx_beam.engine.core import STALL_ROUNDS
from mlx_beam.engine.kv import KVPolicy, for_prefill, for_store, make_request_cache
from mlx_beam.engine.priming import PrimingPromptBatch
from mlx_beam.engine.proposer import trunk
from mlx_beam.engine.request import PromptProgress, ResultStream
from mlx_beam.engine.valve import PrefillValve
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

    def __init__(self, tokenizer, deepstack=True, block=False):
        self._tok = tokenizer
        self.deepstack = deepstack
        self.block = block
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
                            block=self.block,
                        )
                    )
                    tokens += [PAD] * K
        tokens.append(7)
        # The assistant's frame is the last token: the reasoning state is
        # searched from there, not from the user's text.
        return Built(tokens, spans, assistant_start=len(tokens) - 1)

    def describe(self):
        return {"provider": "fake", "tower": "none"}


def reference(model, tokens, spans, n, positions=None):
    """Greedy tokens from a direct forward: embeddings with the spans'
    features scattered in, the extras added after the first layers; with
    `positions` (3, L), the prompt rotated by them and the decode
    continuing from their largest value plus one."""
    inner, lm_head, embed = trunk(model)
    cache = make_request_cache(model, KVPolicy())
    ids = mx.array([tokens])
    kwargs = {}
    next_pos = None
    if positions is not None:
        kwargs["position_ids"] = mx.array(positions)[:, None, :]
        next_pos = int(mx.array(positions).max().item()) + 1
    h = embed(ids)
    for s in spans:
        h[0, s.start : s.end, :] = s.features.astype(h.dtype)
    if any(s.block for s in spans):
        blocks = mx.full((1, len(tokens)), -1, dtype=mx.int32)
        for s in spans:
            if s.block:
                blocks[0, s.start : s.end] = s.start
        kwargs["block_ids"] = blocks

    def hook(i, x):
        for s in spans:
            extra = s.extras.get(i)
            if extra is not None:
                x[0, s.start : s.end, :] = x[0, s.start : s.end, :] + extra.astype(
                    x.dtype
                )
        return x

    import inspect

    if "layer_hook" in inspect.signature(inner.__call__).parameters:
        kwargs["layer_hook"] = hook
    out = lm_head(inner(ids, cache=cache, input_embeddings=h, **kwargs))
    result = []
    y = mx.argmax(out[0, -1])
    for _ in range(n):
        result.append(int(y.item()))
        step = {}
        if next_pos is not None:
            step["position_ids"] = mx.full((3, 1, 1), next_pos, dtype=mx.int32)
            next_pos += 1
        out = lm_head(inner(y.reshape(1, 1), cache=cache, **step))
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


def test_a_block_span_is_prefilled_whole_with_the_model_s_block_ids():
    """Gemma 4's encoder-free models: an image's tokens attend to each
    other both ways in the sliding layers. The frontend marks the span,
    the prefill hands the trunk the block ids and never cuts the span
    across chunks (a chunk that would reach into it stops at its start,
    the next takes it whole) - the
    direct forward's tokens either way, and other tokens than the model
    gives without the mask. A trunk without `block_ids` refuses such a
    span before the worker sees it."""
    from mlx_beam._vendor.mlx_lm.models import gemma4_text
    from mlx_beam.engine import InvalidRequest
    from tests.test_engine import GEMMA4_SHARED, _tiny

    model = _tiny(
        gemma4_text,
        **dict(GEMMA4_SHARED, hidden_size=H, use_bidirectional_attention="vision"),
    )
    front = FakeFrontend(StubTokenizer(), deepstack=False, block=True)
    req, built = request(front, [png(4)], text="w3 w5 w7 w9 w11")
    assert built.spans[0].block and built.spans[0].start > 4
    want = reference(model, built.tokens, built.spans, 6)
    calls = []
    original = gemma4_text.Gemma4TextModel.__call__

    def counting(self, inputs, *args, **kwargs):
        calls.append((inputs.shape[1], kwargs.get("block_ids") is not None))
        return original(self, inputs, *args, **kwargs)

    with Engine(model, prefill_step_size=4, prefill_slice=4) as engine:
        assert engine.image_capabilities["block_mask"]
        gemma4_text.Gemma4TextModel.__call__ = counting
        try:
            assert [e.token for e in engine.submit(req)] == want
        finally:
            gemma4_text.Gemma4TextModel.__call__ = original
    # Chunks of four, but the one that would reach into the span at 7..10
    # stops at its start: 4, 3, then the span whole (with block ids).
    prefill = [c for c in calls if c[0] > 1]
    assert prefill[:3] == [(4, False), (3, False), (3, True)]
    with Engine(model) as engine:
        assert [e.token for e in engine.submit(req)] == want
    # The mask's effect, on the hidden states: the block's first token
    # sees the block's last one (both ways), the text before it is as
    # without the mask.
    inner, _, embed = trunk(model)
    span = built.spans[0]
    ids = mx.array([built.tokens])
    h = embed(ids)
    h[0, span.start : span.end, :] = span.features.astype(h.dtype)
    blocks = mx.full((1, len(built.tokens)), -1, dtype=mx.int32)
    blocks[0, span.start : span.end] = span.start
    masked = inner(
        ids,
        cache=make_request_cache(model, KVPolicy()),
        input_embeddings=h,
        block_ids=blocks,
    )
    causal = inner(ids, cache=make_request_cache(model, KVPolicy()), input_embeddings=h)
    assert not mx.allclose(masked[0, span.start], causal[0, span.start], atol=1e-5)
    assert mx.allclose(masked[0, span.start - 1], causal[0, span.start - 1], atol=1e-5)
    with Engine(tiny_qwen35()) as engine:
        with pytest.raises(InvalidRequest, match="block ids"):
            engine.submit(req)


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
    assert req.assistant_start == len(gen.tokens) - 1


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
        def load(model, model_path, config, tokenizer, *, trust_remote_code=False):
            return FakeFrontend(tokenizer)

    modalities.register(Provider)
    try:
        found, why = modalities.load_frontend(
            model, None, {"model_type": "qwen3_5"}, tok
        )
        assert found.name == "fake" and why is None
        assert modalities.load_frontend(model, None, {"model_type": "llama"}, tok) == (
            None,
            None,
        )
    finally:
        modalities._registered.remove(Provider)


def test_a_frontend_the_text_model_cannot_serve_is_refused_at_load():
    """The family may claim the checkpoint; whether the prefill can apply
    its spans is read from the text model. A frontend that adds per-layer
    extras is refused on a trunk without the layer hook, the reason goes
    to health, and the request path says the same before the worker."""
    from mlx_beam._vendor.mlx_lm.models import llama
    from mlx_beam.engine import InvalidRequest

    tok = StubTokenizer()
    plain = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            num_key_value_heads=1,
        )
    )
    mx.eval(plain.parameters())

    class Needy(FakeFrontend):
        needs = ("input_embeddings", "layer_hook")

    class Provider:
        __name__ = "needy-provider"

        @staticmethod
        def supports(config):
            return True

        @staticmethod
        def load(model, model_path, config, tokenizer, *, trust_remote_code=False):
            return Needy(tokenizer)

    modalities.register(Provider)
    try:
        found, why = modalities.load_frontend(plain, None, {"model_type": "llama"}, tok)
        assert found is None and "layer_hook" in why
        assert (
            modalities.load_frontend(
                tiny_qwen35(), None, {"model_type": "qwen3_5"}, tok
            )[1]
            is None
        )
    finally:
        modalities._registered.remove(Provider)
    with Engine(plain) as engine:
        served = Served(engine, tok, "tiny", vision_refused=why)
        h = served.health()
        assert h["capabilities"]["vision"] is False and h["vision"] == {"refused": why}
        assert engine.image_capabilities == {
            "input_embeddings": True,
            "layer_hook": False,
            "position_ids": False,
            "block_mask": False,
        }
        span = ImageSpan(1, 3, mx.ones((2, 32)), "a" * 64, {1: mx.ones((2, 32))})
        with pytest.raises(InvalidRequest, match="layer hook"):
            engine.submit(GenerationRequest([1, 62, 62, 2, 3], spans=[span]))
        # Without extras the same model serves the span.
        out = list(
            engine.submit(
                GenerationRequest(
                    [1, 62, 62, 2, 3],
                    spans=[ImageSpan(1, 3, mx.ones((2, 32)), "a" * 64)],
                    max_tokens=2,
                )
            )
        )
        assert len(out) == 2 and engine.health()["alive"]


def test_qwen3_vl_wrapper_takes_deepstack_extras():
    """Qwen3-VL's text model is `qwen3`, not Qwen3.5's: the hook is vendored
    there too, and a span with per-layer extras runs through generation."""
    from mlx_beam._vendor.mlx_lm.models import qwen3_vl

    config = dict(
        model_type="qwen3",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rope_theta=10000,
        head_dim=16,
        tie_word_embeddings=False,
    )
    model = qwen3_vl.Model(qwen3_vl.ModelArgs("qwen3_vl", config))
    mx.eval(model.parameters())
    with Engine(model) as engine:
        assert engine.image_capabilities == {
            "input_embeddings": True,
            "layer_hook": True,
            "position_ids": True,
            "block_mask": False,
        }
        span = ImageSpan(1, 3, mx.ones((2, 32)), "a" * 64, {1: mx.ones((2, 32)) * 0.1})
        out = list(
            engine.submit(
                GenerationRequest([1, 62, 62, 2, 3], spans=[span], max_tokens=2)
            )
        )
        assert len(out) == 2 and engine.health()["alive"]


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


def test_the_key_carries_the_whole_digest_not_a_slice_of_it():
    # Two digests that agree in their low 31 bits and differ higher up: a
    # key built from one slice of the digest would be the same for both.
    digests = [f"{n:016x}" + "b" * 48 for n in (1, 1 + 2**31)]
    keys = [
        GenerationRequest(
            [1, 62, 62, 2], spans=[ImageSpan(1, 3, mx.ones((2, 32)) * i, d)]
        ).cache_key
        for i, d in enumerate(digests)
    ]
    assert keys[0] != keys[1]
    assert all(k < 0 for key in keys for k in key[1:3])
    # The same image twice: the same key, and each position its own id.
    same = [
        GenerationRequest(
            [1] + [62] * 12 + [2],
            spans=[ImageSpan(1, 13, mx.ones((12, 32)), "ab" * 32)],
        ).cache_key
        for _ in range(2)
    ]
    assert same[0] == same[1] and len(set(same[0][1:13])) == 12


@pytest.mark.parametrize(
    "field,value",
    [("max_tokens", 0), ("top_logprobs", -1), ("min_response_tokens", -1)],
)
def test_the_request_still_rejects_bad_limits(field, value):
    with pytest.raises(ValueError):
        GenerationRequest([1, 2, 3], **{field: value})


def test_an_image_span_cannot_end_the_prompt():
    """The first generation step feeds the last prompt token without
    image features; a span reaching it would read the placeholder's
    embedding. Refused before admission, past the end as before."""
    span = ImageSpan(1, 3, mx.ones((2, 32)), "a" * 64)
    GenerationRequest([1, 62, 62, 2], spans=[span])
    with pytest.raises(ValueError, match="last prompt token"):
        GenerationRequest([1, 62, 62], spans=[span])
    with pytest.raises(ValueError, match="past the prompt"):
        GenerationRequest([1, 62], spans=[span])


GRANITE_TEXT = dict(
    model_type="granite",
    hidden_size=32,
    num_hidden_layers=2,
    intermediate_size=64,
    num_attention_heads=2,
    num_key_value_heads=1,
    rms_norm_eps=1e-5,
    vocab_size=64,
    logits_scaling=1.0,
    attention_multiplier=0.25,
    embedding_multiplier=2.0,
    residual_multiplier=0.5,
    max_position_embeddings=512,
    attention_bias=False,
    mlp_bias=False,
    rope_theta=10000.0,
)


@pytest.mark.parametrize("text_type", ["granite", "granitemoehybrid"])
def test_a_granite_vision_checkpoint_loads_through_the_model_loader(
    tmp_path, text_type
):
    """`model_type: granite4_vision` has a text-only class in the vendored
    loader: the language model by the text config's type, the tower's
    weights dropped, the nested HF layout lifted - the path the CLI takes
    before a vision provider is asked."""
    import json

    from mlx.utils import tree_flatten

    from mlx_beam._vendor.mlx_lm.utils import load_model
    from mlx_beam.engine.priming import image_capabilities

    if text_type == "granite":
        text = GRANITE_TEXT
    else:
        from tests.test_model_classes import CLASSES

        text = CLASSES["granitemoehybrid"]["cfg"]
    from mlx_beam._vendor.mlx_lm.models import granite, granitemoehybrid

    module = granite if text_type == "granite" else granitemoehybrid
    mx.random.seed(11)
    inner = module.Model(module.ModelArgs.from_dict(dict(text)))
    mx.eval(inner.parameters())
    config = {
        "model_type": "granite4_vision",
        "image_token_index": 62,
        "text_config": dict(text),
        "vision_config": {"model_type": "siglip_vision_model", "hidden_size": 8},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    # The HF layout: the language model under model.language_model, the
    # tower and the projectors beside it, all of which the text view drops.
    weights = {
        "model.language_model." + k[len("model.") :] if k.startswith("model.") else k: v
        for k, v in tree_flatten(inner.parameters())
    }
    weights["model.vision_tower.embeddings.weight"] = mx.zeros((4, 8))
    weights["model.layerwise_projectors.0.weight"] = mx.zeros((4, 8))
    weights["model.image_newline"] = mx.zeros((32,))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    loaded, cfg = load_model(tmp_path)
    assert type(loaded).__name__ == "Model" and loaded.model_type == "granite4_vision"
    assert type(loaded.language_model) is module.Model
    assert image_capabilities(loaded) == {
        "input_embeddings": True,
        "layer_hook": True,
        "position_ids": False,
        "block_mask": False,
    }
    x = mx.array([[3, 7, 11, 13]])
    assert mx.array_equal(loaded(x), inner(x))
    with Engine(loaded) as engine:
        span = ImageSpan(
            1,
            3,
            mx.zeros((2, text["hidden_size"])),
            "c" * 64,
            {1: mx.ones((2, text["hidden_size"]))},
        )
        out = list(
            engine.submit(
                GenerationRequest([1, 62, 62, 2, 3], spans=[span], max_tokens=2)
            )
        )
        assert len(out) == 2 and engine.health()["alive"]


def test_responses_take_images_through_the_same_frontend():
    """`input_image` parts reach the chat path as image parts: decoded
    with a frontend, refused without one - the same as chat. A base64
    payload wrapped in lines decodes; `detail` is ignored."""
    from mlx_beam.api import responses

    flat = __import__("base64").b64encode(png(4).data).decode()
    lines = "\n".join(flat[i : i + 8] for i in range(0, len(flat), 8))
    body = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "w1"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{lines}",
                        "detail": "high",
                    },
                ],
            }
        ]
    }
    with pytest.raises(ApiError) as exc:
        responses.parse_responses_request(body, "m")
    assert (
        exc.value.code == "extra_not_installed"
        and "/health.vision" in exc.value.message
    )
    req = responses.parse_responses_request(body, "m", vision=True)
    assert len(req.chat.images) == 1 and req.chat.images[0].digest == png(4).digest
    assert req.chat.messages[0]["content"] == [
        {"type": "text", "text": "w1"},
        {"type": "image"},
    ]
    tok = StubTokenizer()
    gen = responses.to_generation_request(tok, req, frontend=FakeFrontend(tok))
    assert len(gen.spans) == 1


def test_images_in_tool_results_reach_the_frontend():
    """A tool that returns an image (a screenshot) hands it back inside
    the tool's result: Responses as parts in `function_call_output.output`,
    Messages as an image block in `tool_result.content`. Both go through
    the same image path as a user's message - not serialized to text, not
    refused - and the tool turn keeps its id."""
    from mlx_beam.api import messages, responses

    data = __import__("base64").b64encode(png(4).data).decode()
    url = f"data:image/png;base64,{data}"
    body = {
        "input": [
            {"role": "user", "content": "Inspect the screenshot."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "shot",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "taken"},
                    {"type": "input_image", "image_url": url},
                ],
            },
        ]
    }
    req = responses.parse_responses_request(body, "m", vision=True)
    assert len(req.chat.images) == 1 and req.chat.images[0].digest == png(4).digest
    tool_turn = req.chat.messages[-1]
    assert tool_turn["role"] == "tool" and tool_turn["tool_call_id"] == "call_1"
    assert tool_turn["content"] == [
        {"type": "text", "text": "taken"},
        {"type": "image"},
    ]
    # Text-only results still fold to a string; other values stay JSON.
    body["input"][2]["output"] = [{"type": "input_text", "text": "plain"}]
    assert (
        responses.parse_responses_request(body, "m").chat.messages[-1]["content"]
        == "plain"
    )
    body["input"][2]["output"] = {"ok": True}
    assert (
        responses.parse_responses_request(body, "m").chat.messages[-1]["content"]
        == '{"ok": true}'
    )

    body = {
        "max_tokens": 4,
        "messages": [
            {"role": "user", "content": "Inspect the screenshot."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "shot", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "text", "text": "taken"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": data,
                                },
                            },
                        ],
                    }
                ],
            },
        ],
    }
    req = messages.parse_messages_request(body, "m", vision=True)
    assert len(req.chat.images) == 1 and req.chat.images[0].digest == png(4).digest
    tool_turn = req.chat.messages[-1]
    assert tool_turn["role"] == "tool" and tool_turn["tool_call_id"] == "t1"
    assert tool_turn["content"] == [
        {"type": "text", "text": "taken"},
        {"type": "image"},
    ]
    # Without the vision extra the image is refused as everywhere else.
    with pytest.raises(ApiError) as exc:
        messages.parse_messages_request(body, "m")
    assert exc.value.code == "extra_not_installed"


def test_an_image_path_climbs_the_effort_ladder_and_reports_the_processor():
    """With images the prompt goes through the frontend, and through the
    same ladder: an effort word the template rejects is replaced by the
    rung it takes; a processor error is the client's 400, not a 500."""
    tok = StubTokenizer()

    class Picky(FakeFrontend):
        def build(self, messages, images, template_kwargs):
            if template_kwargs.get("reasoning_effort") not in (
                None,
                "low",
                "medium",
                "high",
            ):
                raise ValueError("reasoning_effort must be one of low, medium, high")
            if template_kwargs.get("boom"):
                raise RuntimeError("the processor choked")
            return super().build(messages, images, template_kwargs)

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "w1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
        # As a template kwarg: goes through as written, whatever the
        # template's measured effort capability says.
        "chat_template_kwargs": {"reasoning_effort": "xhigh"},
    }
    req = chat.parse_chat_request(body, "m", vision=True)
    tokens = chat.build_prompt(tok, req, Picky(tok))
    assert req.template_kwargs["reasoning_effort"] == "high" and len(req.spans) == 1
    assert tokens[req.spans[0].start] == PAD
    req.template_kwargs["boom"] = True
    with pytest.raises(ApiError, match="image input was refused: the processor choked"):
        chat.build_prompt(tok, req, Picky(tok))
    with pytest.raises(ApiError) as exc:
        chat.build_prompt(tok, req, None)
    assert exc.value.code == "extra_not_installed"


def test_a_provider_stating_its_needs_is_refused_before_its_tower_loads():
    from mlx_beam._vendor.mlx_lm.models import llama

    tok = StubTokenizer()
    plain = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            num_key_value_heads=1,
        )
    )
    mx.eval(plain.parameters())
    loads = []

    class Provider:
        __name__ = "stating-provider"

        @staticmethod
        def supports(config):
            return True

        @staticmethod
        def needs(config):
            return ("input_embeddings", "layer_hook")

        @staticmethod
        def load(model, model_path, config, tokenizer, *, trust_remote_code=False):
            loads.append(trust_remote_code)
            return FakeFrontend(tokenizer)

    modalities.register(Provider)
    try:
        found, why = modalities.load_frontend(plain, None, {"model_type": "llama"}, tok)
        assert found is None and "layer_hook" in why and loads == []
        found, why = modalities.load_frontend(
            tiny_qwen35(), None, {"model_type": "qwen3_5"}, tok, trust_remote_code=True
        )
        assert found is not None and why is None and loads == [True]
    finally:
        modalities._registered.remove(Provider)


def test_unequal_prefill_rows_with_spans_are_right_padded():
    """The engine cuts every prefill call to the shortest row, except over
    a block span; then a shorter row is right-padded for the call, as
    upstream pads text rows, and its cache trimmed again. The padded
    row's keys equal its own solo prefill's, the image row's too."""
    from mlx_beam.engine.priming import PrimingPromptBatch

    model = tiny_qwen35()
    front = FakeFrontend(StubTokenizer())
    _, built = request(front, [png(7)])
    short = [3, 7, 11, 13, 5]
    cls = PrimingPromptBatch.bound(lambda uid: built.spans if uid == 1 else ())

    def state_after(uids, rows):
        caches = [make_request_cache(model, KVPolicy()) for _ in uids]
        batch = cls(model, uids, caches, prefill_step_size=64)
        batch.prompt(rows)
        out = [[] for _ in rows]
        for c in batch.prompt_cache:
            keys = getattr(c, "keys", None)
            if keys is not None:
                # A right-padded row ends up left-padded after finalize:
                # its real keys are the last len(row) positions either way.
                n = keys.shape[2]
                for i, r in enumerate(rows):
                    out[i].append(keys[i, :, n - len(r) :, :])
            else:  # the hybrid's recurrent state, one array per row
                for i in range(len(rows)):
                    out[i].extend(a[i] for a in c.cache if a is not None)
        return out

    both = state_after([1, 2], [built.tokens, short])
    solo_image = state_after([1], [built.tokens])[0]
    solo_text = state_after([2], [short])[0]
    assert len(both[0]) == len(solo_image) > 1
    for a, b in zip(both[0], solo_image, strict=True):
        assert mx.allclose(a, b, atol=1e-5)
    for a, b in zip(both[1], solo_text, strict=True):
        assert mx.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("stream", [False, True])
def test_a_think_marker_in_the_user_text_beside_an_image_is_text(stream):
    """The frontend says where the assistant's frame begins, and the
    reasoning state is seeded from there only: a `<think>` the user wrote
    next to an image does not open a block, so the answer's tokens reach
    the content, streamed or not."""
    from mlx_beam.engine.request import TokenEvent
    from tests.stub_tokenizer import EOS

    tok = StubTokenizer()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<think> w1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            }
        ],
        "stream": stream,
    }
    req = chat.parse_chat_request(body, "m", vision=True)
    gen = chat.to_generation_request(tok, req, frontend=FakeFrontend(tok))
    assert not gen.reasoning.seeded and req.assistant_start == len(gen.tokens) - 1
    responder = chat.ChatResponder(tok, req, gen.tokens)
    events = [TokenEvent(3, -0.1), TokenEvent(EOS, -0.2, "stop")]
    if stream:
        chunks = list(responder.stream(iter(events), 0))
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    else:
        content = responder.complete(iter(events), 0)["choices"][0]["message"][
            "content"
        ]
    assert content == "w3 "


# -- image blocks beside other prompts: progress, width, the valve -----------


def gemma_with_blocks():
    """The tiny Gemma 4 whose sliding layers take block ids."""
    from mlx_beam._vendor.mlx_lm.models import gemma4_text
    from tests.test_engine import GEMMA4_SHARED, _tiny

    return _tiny(
        gemma4_text, **dict(GEMMA4_SHARED, use_bidirectional_attention="vision")
    )


def block(start, end, seed=0):
    """An image block span of random features at the tiny model's width."""
    mx.random.seed(seed)
    feats = mx.random.normal((end - start, 64))
    mx.eval(feats)
    return ImageSpan(start, end, feats, f"{seed:02x}" * 32, block=True)


def prompt_of(n, shift=0):
    return [1 + (i + shift) % 62 for i in range(n)]


class Room:
    """A valve over synthetic memory: `tokens` is the room in tokens (one
    byte a token; the peak never moves, so nothing is learned), a number
    or a callable read at every decision. Under a `recommended` ceiling
    (none unless given) the room is `recommended` less what `held` reads
    (0 unless given), when that is smaller. Every width(), peek() and
    observe() the scheduler makes once the engine is ready is logged with
    the labels of the rows in the prompt batch, so a test sees who ran,
    who stalled, and in which order."""

    def __init__(self, tokens, held=lambda: 0, recommended=0):
        self.tokens = tokens
        self.engine = None
        self.log = []
        ticks = itertools.count(0, 2)  # every reading is past the free-memory TTL
        self.valve = PrefillValve(
            active=held,
            peak=lambda: 0,
            free=lambda: int(self.tokens() if callable(self.tokens) else self.tokens),
            recommended=recommended,
            clock=lambda: float(next(ticks)),
        )
        self.valve.bytes_per_token = 1.0

    def rows(self):
        gen = None if self.engine is None else self.engine._gen
        if gen is None:
            return None
        live = self.engine._live
        return [getattr(live.get(u), "label", u) for u in gen._prompt_batch.uids]

    def _note(self, kind, tokens, result):
        if self.engine is None or self.engine._ready.is_set():
            self.log.append((kind, tokens, result, self.rows()))

    def width(self, wanted):
        out = self.valve.width(wanted)
        self._note("width", wanted, out)
        return out

    def peek(self, wanted):
        out = self.valve.peek(wanted)
        self._note("peek", wanted, out)
        return out

    def observe(self, tokens):
        self.valve.observe(tokens)
        self._note("observe", tokens, None)

    def describe(self):
        return self.valve.describe()

    def round(self):
        return self.valve.round()

    def calls(self):
        """(tokens, row labels) of every prefill call that ran."""
        return [(t, rows) for kind, t, _, rows in self.log if kind == "observe"]


def queue_before_start(engine, requests, labels):
    """Streams queued before the worker starts, so that its first round
    admits them together, in this order: `submit` needs a running worker,
    and two submits race for the same round."""
    streams = []
    for req, label in zip(requests, labels, strict=True):
        engine._validate(req)
        for span in req.spans:
            mx.eval(span.features, *span.extras.values())
        stream = ResultStream(req, engine._request_cancel)
        stream.completion_cap, stream.reasoning_cap = engine.admit(req)
        stream.label = label
        engine._inbox.put(stream)
        streams.append(stream)
    return streams


def outcome(stream):
    """What the worker put on the stream: (progress reports, token events,
    the error or None)."""
    progress, tokens = [], []
    while True:
        item = stream._queue.get(timeout=120)
        if item is None:
            return progress, tokens, None
        if isinstance(item, PromptProgress):
            progress.append(item)
        elif isinstance(item, BaseException):
            return progress, tokens, item
        else:
            tokens.append(item)
            if item.finish_reason is not None:
                return progress, tokens, None


def record_rounds(monkeypatch, room):
    """Per gen.next() once the engine is ready: the labels of the rows
    whose prompt ended in it, how many prefill calls ran in it, and the
    stall state and the decode after it."""
    rounds = []
    real = BatchGenerator.next

    def recording(self):
        before = len(room.calls())
        prompt_responses, generation_responses = real(self)
        if room.engine is not None and room.engine._ready.is_set():
            live = room.engine._live
            rounds.append(
                dict(
                    ended=[
                        getattr(live.get(r.uid), "label", r.uid)
                        for r in prompt_responses
                        if r.end_of_prompt
                    ],
                    calls=len(room.calls()) - before,
                    streak=self.stall_streak,
                    block_stall=getattr(self, "block_stall", None),
                    decoding=len(self._generation_batch) > 0,
                )
            )
        return prompt_responses, generation_responses

    monkeypatch.setattr(BatchGenerator, "next", recording)
    return rounds


def engine_with(model, room, **kwargs):
    engine = Engine(model, **kwargs)
    engine.prefill_valve = room
    room.engine = engine
    return engine


def test_a_padded_row_files_its_checkpoint_at_its_own_position():
    """A text row beside an image row whose block makes the call wider
    than the text row's rest: the text row is right-padded for that call
    and reports its own tokens, not the call's width. The engine files
    its checkpoint under that position, and a later prefix hit restores
    the states of one position."""
    from mlx_beam.engine.prefix import window_layers

    model = gemma_with_blocks()
    image = GenerationRequest(prompt_of(12), max_tokens=12, spans=[block(3, 10)])
    text = GenerationRequest(prompt_of(5, shift=20), max_tokens=12)
    room = Room(1 << 20)
    engine = engine_with(model, room, prefill_step_size=4, prefill_slice=4)
    a, b = queue_before_start(engine, [image, text], ["image", "text"])
    with engine:
        _, out_a, err_a = outcome(a)
        progress, out_b, err_b = outcome(b)
        assert err_a is None and err_b is None and len(out_a) == len(out_b) == 12
        # Calls of four, cut at the block's start: 3 for both rows, then
        # the block whole (7) with the text row padded. The text row goes
        # 3 -> 4, the end of its first segment (its last token stands
        # alone), the image row 3 -> 10.
        assert room.calls()[:2] == [(6, ["image", "text"]), (14, ["image", "text"])]
        assert [p.processed for p in progress] == [0, 3, 4, 5]
        generated = [e.token for e in out_b]
        query = text.tokens + generated[:5] + [63]
        hit = engine.prefix_store.fetch(engine.model_key, query)
    assert hit is not None and hit.covered == 4
    offsets = [int(c.offset) for c in hit.cache if c is not None]
    assert offsets and set(offsets) == {4}
    # The window layers hold what a solo prefill of the first four tokens holds.
    solo = make_request_cache(model, KVPolicy())
    model(mx.array([text.tokens[:4]]), cache=solo)
    windows = window_layers(hit.cache)
    assert windows
    for i in windows:
        assert mx.allclose(hit.cache[i].keys, solo[i].keys, atol=1e-5)


def test_block_width_is_a_fixpoint_in_any_row_order():
    """The width that cuts no block is grown over every block of every
    row until it stands still: a row visited early can have a block the
    growth a later row caused reaches into."""
    model = gemma_with_blocks()

    def batch_for(spans, step=4, tokens=None):
        cls = PrimingPromptBatch.bound(lambda uid: spans.get(uid, ()))
        uids = list(spans)
        return cls(
            model,
            uids,
            [make_request_cache(model, KVPolicy()) for _ in uids],
            tokens=tokens,
            prefill_step_size=step,
        )

    first, second = block(4, 9, seed=1), block(2, 6, seed=2)
    for spans in ({0: [first], 1: [second]}, {0: [second], 1: [first]}):
        assert batch_for(spans)._whole_blocks([0, 0], 0, 4) == 9
    # A chain over three rows, the row whose block grows last visited first.
    chain = {
        0: [block(12, 20, seed=3)],
        1: [block(8, 13, seed=4)],
        2: [block(4, 9, seed=5)],
    }
    assert batch_for(chain)._whole_blocks([0, 0, 0], 0, 5) == 20
    # The scheduler's view: the cut at the first block start reached into,
    # the growth from there, and the need of every row at a block start.
    at = batch_for({0: [first], 1: [second]}, tokens=[[7] * 4, [7] * 2])
    assert at.block_widths(8) == (8, 8, {0: 5, 1: 4})
    assert at.block_widths(3) == (3, 5, {0: 5, 1: 4})
    # Both rows at 0: the cut stops at the first block start reached into,
    # 2, and nothing grows from there.
    assert batch_for({0: [first], 1: [second]}).block_widths(8) == (2, 2, {})
    assert batch_for({0: [first], 1: [second]}).block_widths(3) == (2, 2, {})
    # Cut, then grown: a block at 0 is its row's need; the cut at the other
    # row's block start reaches into it, and its end reaches into the other.
    grows = batch_for({0: [block(0, 5, seed=6)], 1: [block(3, 9, seed=7)]})
    assert grows.block_widths(8) == (3, 9, {0: 5})
    # Numerically, in the order that cut a block: the keys of the shared
    # prefill in chunks are the keys of the rows' solo prefills.
    rows = {0: prompt_of(12), 1: prompt_of(12, shift=30)}
    spans = {0: [first], 1: [second]}

    def keys_after(spans, rows, step):
        batch = batch_for({u: spans[u] for u in rows}, step)
        batch.prompt([rows[u] for u in batch.uids])
        return [
            [
                c.keys[i, :, :12]
                for c in batch.prompt_cache
                if isinstance(c, BatchKVCache)
            ]
            for i in range(len(batch.uids))
        ]

    shared = keys_after(spans, rows, 4)
    for i, uid in enumerate(rows):
        solo = keys_after({uid: spans[uid]}, {uid: rows[uid]}, 64)[0]
        assert len(solo) > 0
        for a, b in zip(shared[i], solo, strict=True):
            assert mx.allclose(a, b, atol=1e-5)


def test_a_block_that_does_not_fit_stalls_instead_of_passing_the_valve():
    """The valve has room for 64 tokens, the row's block [32, 256) needs
    224 in one call: the first call stops at the block's start, then every
    round is a block stall, counted like a valve stall, and the engine
    refuses the row after STALL_ROUNDS, naming the block's need."""
    model = gemma_with_blocks()
    spans = {0: [block(32, 256)]}
    room = Room(64)
    gen = BatchGenerator(
        model,
        prompt_batch=PrimingPromptBatch.bound(lambda uid: spans.get(uid, ())),
        prefill_step_size=128,
        prefill_slice=128,
        stop_tokens=[],
        prefill_valve=room,
    )
    gen.insert([prompt_of(300)], max_tokens=[1])
    gen.next()
    assert len(gen._prompt_batch.tokens[0]) == 32 and room.calls() == [(32, None)]
    for n in range(1, STALL_ROUNDS + 2):
        prompt_responses, generated = gen.next()
        assert not prompt_responses and not generated
        assert gen.stall_streak == n and gen.block_stalls == n
        assert gen.block_stall.need == 224 and gen.block_stall.counts
    assert len(gen._prompt_batch.tokens[0]) == 32 and len(room.calls()) == 1
    gen.close()

    room = Room(64)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    (stream,) = queue_before_start(
        engine,
        [GenerationRequest(prompt_of(300), max_tokens=2, spans=[block(32, 256)])],
        ["block"],
    )
    with engine:
        _, _, error = outcome(stream)
        h = engine.health()
    assert isinstance(error, ContextTooLong)
    assert "32 of it prefilled" in str(error) and "needs 224 tokens" in str(error)
    assert h["alive"] and h["prefill_valve"]["refused"] == 1
    assert h["prefill_valve"]["block_stalled_rounds"] == STALL_ROUNDS
    assert room.calls() == [(32, ["block"])]


def test_the_block_row_is_refused_not_the_text_row_with_the_longer_prefix():
    """Beside a text row the block row parks and the text row runs on;
    the text row ends, and only then is the block row refused, standing
    alone - the rule 'the row with the longest known prefix' cannot pick
    the text row, which has prefilled far more."""
    model = gemma_with_blocks()
    room = Room(64)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(300), max_tokens=2, spans=[block(0, 224)]),
            GenerationRequest(prompt_of(300, shift=7), max_tokens=3),
        ],
        ["block", "text"],
    )
    with engine:
        _, out_b, err_b = outcome(b)
        _, _, err_a = outcome(a)
        h = engine.health()
    assert err_b is None and len(out_b) == 3
    assert isinstance(err_a, ContextTooLong)
    assert "0 of it prefilled" in str(err_a) and "needs 224 tokens" in str(err_a)
    assert h["prefill_valve"]["refused"] == 1
    assert h["prefill_valve"]["parked_rows"] == 1
    assert room.calls() and all(rows == ["text"] for _, rows in room.calls())


def test_a_block_that_fits_alone_waits_for_the_text_row_and_runs_after_it(
    monkeypatch,
):
    """Two rows at 0, the block [0, 80), room for 128: beside the text row
    the block does not fit (80 for two rows), alone it does. No stall, no
    refusal: the text row runs, the block row waits and runs alone after
    it, and the round that hands the text row over to generation runs
    no prefill call."""
    model = gemma_with_blocks()
    room = Room(128)
    rounds = record_rounds(monkeypatch, room)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(100), max_tokens=2, spans=[block(0, 80)]),
            GenerationRequest(prompt_of(100, shift=5), max_tokens=2),
        ],
        ["block", "text"],
    )
    with engine:
        _, out_a, err_a = outcome(a)
        _, out_b, err_b = outcome(b)
        h = engine.health()
    assert err_a is None and err_b is None and len(out_a) == len(out_b) == 2
    calls = room.calls()
    assert calls[0] == (64, ["text"])
    assert all(rows == ["text"] for _, rows in calls[:-1])
    assert calls[-1] == (99, ["block"])
    v = h["prefill_valve"]
    assert v["refused"] == 0 and v["block_stalled_rounds"] == 0
    assert v["stalled_rounds"] == 0 and v["parked_rows"] == 1 and v["parked"] == 0
    handover = [r for r in rounds if "text" in r["ended"]]
    assert len(handover) == 1 and handover[0]["calls"] == 0


@pytest.mark.parametrize("a_first", [True, False])
def test_only_the_row_at_a_block_start_forces_the_width(a_first):
    """A's block [0, 80) and B's [64, 160), both rows at 0, room for 128:
    the fixpoint width is 160, but only A stands at a block start and
    forces its own need, 80. B is no forcing row: it runs 64 in the first
    round and takes its block whole in a later call. Both end, whichever
    row comes first."""
    model = gemma_with_blocks()
    room = Room(128)
    reqs = {
        "A": GenerationRequest(
            prompt_of(200), max_tokens=2, spans=[block(0, 80, seed=1)]
        ),
        "B": GenerationRequest(
            prompt_of(200, shift=3), max_tokens=2, spans=[block(64, 160, seed=2)]
        ),
    }
    order = ["A", "B"] if a_first else ["B", "A"]
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    streams = queue_before_start(engine, [reqs[k] for k in order], order)
    with engine:
        results = {k: outcome(s) for k, s in zip(order, streams, strict=True)}
        h = engine.health()
    for k in order:
        assert results[k][2] is None and len(results[k][1]) == 2
    calls = room.calls()
    assert calls[0] == (64, ["B"])
    assert all(rows == ["B"] for _, rows in calls[:3])
    assert calls[3] == (128, ["A"])
    v = h["prefill_valve"]
    assert v["refused"] == 0 and v["block_stalled_rounds"] == 0
    assert v["parked_rows"] == 1


def test_when_every_row_forces_the_first_that_fits_runs_alone():
    """Two rows at 0 whose blocks [0, 80) and [0, 96) both exceed the
    width the valve allows for two rows (128 for both): the first in
    batch order that fits alone runs alone, the second parks, and no
    round stalls."""
    model = gemma_with_blocks()
    room = Room(128)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(200), max_tokens=2, spans=[block(0, 80, 1)]),
            GenerationRequest(
                prompt_of(200, shift=3), max_tokens=2, spans=[block(0, 96, 2)]
            ),
        ],
        ["A", "B"],
    )
    with engine:
        _, out_a, err_a = outcome(a)
        _, out_b, err_b = outcome(b)
        h = engine.health()
    assert err_a is None and err_b is None and len(out_a) == len(out_b) == 2
    calls = room.calls()
    assert calls[0] == (80, ["A"])
    assert all(rows == ["A"] for _, rows in calls[:2])
    assert calls[2] == (128, ["B"])
    v = h["prefill_valve"]
    assert v["refused"] == 0 and v["block_stalled_rounds"] == 0
    assert v["stalled_rounds"] == 0 and v["parked_rows"] == 1


def test_a_parked_row_is_not_refused_while_the_text_row_decodes(monkeypatch):
    """Room for 64 tokens while the text row lives, for 128 once it has
    ended. The block row (need 96) parks without a check of its own, the
    text row prefills and decodes; the block row comes back alone and
    stands in a block stall for more than STALL_ROUNDS rounds beside the
    decode, while the store gives up an entry - and is not refused, since
    those rounds do not count. Once the text row has ended, it runs."""
    model = gemma_with_blocks()
    engine = Engine(
        model, prefill_step_size=128, prefill_slice=128, prompt_concurrency=3
    )
    s, a, b = queue_before_start(
        engine,
        [
            # One token: finished at once, an entry for the store to give up.
            GenerationRequest([5], max_tokens=1),
            GenerationRequest(prompt_of(100), max_tokens=2, spans=[block(0, 96)]),
            GenerationRequest(prompt_of(40, shift=9), max_tokens=30),
        ],
        ["seed", "block", "text"],
    )
    room = Room(lambda: 64 if b in engine._live.values() else 128)
    engine.prefill_valve = room
    room.engine = engine
    rounds = record_rounds(monkeypatch, room)
    with engine:
        outcome(s)
        _, out_b, err_b = outcome(b)
        _, out_a, err_a = outcome(a)
        h = engine.health()
        evicted = engine.prefix_store.describe()["evicted_for_memory"]
    assert err_b is None and len(out_b) == 30
    assert err_a is None and len(out_a) == 2
    calls = room.calls()
    assert all(rows == ["text"] for _, rows in calls[:-1])
    assert calls[-1] == (99, ["block"])
    v = h["prefill_valve"]
    assert v["refused"] == 0 and v["parked_rows"] == 1
    assert v["block_stalled_rounds"] > STALL_ROUNDS and evicted == 1
    beside = [r for r in rounds if r["decoding"] and r["block_stall"] is not None]
    assert len(beside) > STALL_ROUNDS


def test_a_block_stall_is_refused_only_after_the_last_decode_with_its_need(
    monkeypatch,
):
    """Room for 64 only, with or without the text row. The text row runs
    and ends, and nothing stalls while it prefills; the block row is
    refused only after the text row's last token, standing alone with
    nothing decoding, and the refusal names its need of 96."""
    model = gemma_with_blocks()
    room = Room(64)
    rounds = record_rounds(monkeypatch, room)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(100), max_tokens=2, spans=[block(0, 96)]),
            GenerationRequest(prompt_of(40, shift=9), max_tokens=30),
        ],
        ["block", "text"],
    )
    with engine:
        _, out_b, err_b = outcome(b)
        _, _, err_a = outcome(a)
        h = engine.health()
    assert err_b is None and len(out_b) == 30
    assert isinstance(err_a, ContextTooLong)
    assert "needs 96 tokens" in str(err_a) and "0 of it prefilled" in str(err_a)
    assert h["prefill_valve"]["refused"] == 1
    ended = next(i for i, r in enumerate(rounds) if "text" in r["ended"])
    assert all(r["streak"] == 0 and r["block_stall"] is None for r in rounds[:ended])
    # Beside the decode the streak never grows; the refusal follows rounds
    # with nothing decoding.
    assert max((r["streak"] for r in rounds if r["decoding"]), default=0) <= 1
    last_decode = max((i for i, r in enumerate(rounds) if r["decoding"]), default=-1)
    assert last_decode >= 0
    # After it, the stalled rounds: exactly STALL_ROUNDS counted rounds
    # precede the refusal, and no round before them reached the count.
    streaks = [r["streak"] for r in rounds[last_decode + 1 :] if r["streak"]]
    assert streaks[-STALL_ROUNDS:] == list(range(1, STALL_ROUNDS + 1))
    assert all(s < STALL_ROUNDS for s in streaks[:-STALL_ROUNDS])


@pytest.mark.parametrize("variant", ["blocked", "room_after_b", "room_after_text"])
def test_waiting_rows_run_in_order_and_pass_only_a_blocked_front(variant, monkeypatch):
    """Three rows at 0 in this order - a text row, A with the block
    [0, 192), B with [0, 96) - step and slice 128, room for 128 (42 a
    row for three): A and B park in this order and the text row runs.
    The lot is a queue: A heads it, and B passes only while A is blocked.
    Blocked for good, B runs and A is refused after both others ended;
    room for A once B is prefilled, A runs after B; room for A as soon as
    the text row is prefilled, A runs before B."""
    model = gemma_with_blocks()
    engine = Engine(
        model, prefill_step_size=128, prefill_slice=128, prompt_concurrency=3
    )
    t, a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(260), max_tokens=2),
            GenerationRequest(
                prompt_of(260, shift=3), max_tokens=2, spans=[block(0, 192, 1)]
            ),
            GenerationRequest(
                prompt_of(260, shift=6), max_tokens=2, spans=[block(0, 96, 2)]
            ),
        ],
        ["text", "A", "B"],
    )

    def prefilled(stream):
        """The row has left the prompt batch and the lot: decoding, or
        finished (a finished row may have left the generation batch
        before the engine drops its stream)."""
        gen = engine._gen
        uid = next((u for u, s in engine._live.items() if s is stream), None)
        if uid is None:
            return True
        lot = getattr(gen, "_parked", None)
        return uid not in gen._prompt_batch.uids and (
            lot is None or uid not in lot.uids
        )

    if variant == "blocked":
        room = Room(128)
    elif variant == "room_after_b":
        room = Room(lambda: 256 if prefilled(b) else 128)
    else:
        room = Room(lambda: 256 if prefilled(t) else 128)
    engine.prefill_valve = room
    room.engine = engine
    rounds = record_rounds(monkeypatch, room)
    with engine:
        results = {k: outcome(s) for k, s in (("text", t), ("B", b), ("A", a))}
        h = engine.health()
    v = h["prefill_valve"]
    calls = room.calls()
    assert calls[0] == (42, ["text"]) and v["parked_rows"] == 2
    for k in ("text", "B"):
        assert results[k][2] is None and len(results[k][1]) == 2
    first_block = next(rows[0] for _, rows in calls if rows != ["text"])
    error = results["A"][2]
    if variant == "blocked":
        assert first_block == "B"
        assert isinstance(error, ContextTooLong) and "needs 192 tokens" in str(error)
        assert v["refused"] == 1 and not any(rows == ["A"] for _, rows in calls)
        # Refused only after the last token of the text row and of B: no
        # count grew beside their decodes, and exactly STALL_ROUNDS counted
        # rounds follow the last one.
        last_decode = max(
            (i for i, r in enumerate(rounds) if r["decoding"]), default=-1
        )
        assert last_decode >= 0
        assert all(r["streak"] <= 1 for r in rounds[: last_decode + 1])
        streaks = [r["streak"] for r in rounds[last_decode + 1 :] if r["streak"]]
        assert streaks[-STALL_ROUNDS:] == list(range(1, STALL_ROUNDS + 1))
        assert all(s < STALL_ROUNDS for s in streaks[:-STALL_ROUNDS])
        return
    assert error is None and len(results["A"][1]) == 2 and v["refused"] == 0
    a_runs = calls.index((192, ["A"]))
    b_runs = next(i for i, (_, rows) in enumerate(calls) if rows == ["B"])
    if variant == "room_after_b":
        assert first_block == "B" and b_runs < a_runs
    else:
        assert first_block == "A" and a_runs < b_runs


@pytest.mark.parametrize(
    "policy",
    [KVPolicy(), KVPolicy(bits=8), KVPolicy(bits=8, prefill="quantized")],
    ids=["plain", "exact", "quantized"],
)
def test_a_parked_row_comes_back_intact(policy):
    """Of four text rows two park after the first call (split); one of
    them is removed while parked, the other waits while the rest prefill
    for more rounds than the sliding window holds (8 // 2) with no
    one-wide call, and comes back (extend) behind them. Its first decode
    step gives a solo run's logprobs, and its caches hold what the solo
    run's hold - for every batch cache class the policy builds: the
    plain KV and the rotating window, the deferred cache of an exact
    quantized prefill (with a restored, packed prefix), and the quantized
    cache of a quantized prefill."""
    model = gemma_with_blocks()
    cls = PrimingPromptBatch.bound(lambda uid: ())
    prompts = [
        prompt_of(17),
        prompt_of(11, shift=5),
        prompt_of(9, shift=10),
        prompt_of(19, shift=15),
    ]
    prefix, rest = prompts[1][:4], prompts[1][4:]

    def generator():
        return BatchGenerator(
            model,
            prompt_batch=cls,
            prefill_step_size=2,
            prefill_slice=2,
            stop_tokens=[],
        )

    def restored():
        """A cache over the prefix as the store hands it back under the
        policy: for an exact prefill dequantized, its codes packed beside."""
        caches = make_request_cache(model, policy)
        model(mx.array([prefix]), cache=caches)
        mx.eval(*_cache_arrays(caches))
        return for_prefill(for_store(caches), policy)

    def first_step(gen, uid):
        """(logprobs, extracted caches) of the row's first decode step."""
        for _ in range(80):
            _, generated = gen.next()
            for r in generated:
                if r.uid == uid:
                    batch = gen._generation_batch
                    return r, batch.extract_cache(batch.uids.index(uid))
        raise AssertionError("the row never decoded")

    def state(caches):
        """Keys and values at model precision: a window layer's last
        window, every other layer's whole row."""
        out = []
        for c in caches:
            if c is None:
                continue
            keys, values = c.keys, c.values
            if isinstance(keys, tuple):
                keys = mx.dequantize(*keys, group_size=c.group_size, bits=c.bits)
                values = mx.dequantize(*values, group_size=c.group_size, bits=c.bits)
            n = getattr(c, "max_size", None)
            if n is None:
                keys, values = keys[..., : c.offset, :], values[..., : c.offset, :]
            else:
                keys, values = keys[..., -n:, :], values[..., -n:, :]
            out.append((int(c.offset), keys, values))
        return out

    # The batch appends to the lists it is handed: copies of the prefix.
    solo = generator()
    (solo_uid,) = solo.insert(
        [rest], max_tokens=[2], caches=[restored()], all_tokens=[list(prefix)]
    )
    want, want_caches = first_step(solo, solo_uid)

    gen = generator()
    caches = [make_request_cache(model, policy) for _ in prompts]
    caches[1] = restored()
    uids = gen.insert(
        [prompts[0], rest, prompts[2], prompts[3]],
        max_tokens=[2] * 4,
        caches=caches,
        all_tokens=[[], list(prefix), [], []],
    )
    gen.next()
    assert [len(t) for t in gen._prompt_batch.tokens] == [2, 6, 2, 2]
    without = gen.prompt_cache_nbytes
    assert hasattr(gen, "_park")  # the lot is the fix
    gen._park([1, 2])
    assert gen._parked.uids == [uids[1], uids[2]]
    assert gen._parked_processing[0][0][0] == rest[2:-1]
    assert gen._find_uids(uids) == {
        uids[0]: (1, 0),
        uids[3]: (1, 1),
        uids[1]: (3, 0),
        uids[2]: (3, 1),
    }
    # The lot's caches count. The split trims the padding row 1's prefix
    # had put on the others, so the sum is below `without`.
    lot = sum(c.nbytes for c in gen._parked.prompt_cache)
    assert lot > 0
    assert gen.prompt_cache_nbytes == lot + sum(
        c.nbytes for c in gen._prompt_batch.prompt_cache
    )
    gen.remove([uids[2]])
    assert gen._parked.uids == [uids[1]] and uids[2] not in gen._find_uids(uids)
    assert 0 < gen.prompt_cache_nbytes < without
    for _ in range(5):
        prompt_responses, _ = gen.next()
        assert all(r.progress[0] % 2 == 0 for r in prompt_responses)
    assert [len(t) for t in gen._prompt_batch.tokens] == [12, 12]
    gen._take(0)
    assert gen._parked is None and gen._prompt_batch.uids[-1] == uids[1]
    assert len(gen._prompt_batch.tokens[-1]) == 6
    got, got_caches = first_step(gen, uids[1])
    assert got.token == want.token
    assert mx.allclose(got.logprobs, want.logprobs, atol=1e-2)
    for (n, k, v), (m, k2, v2) in zip(
        state(got_caches), state(want_caches), strict=True
    ):
        assert n == m and k.shape == k2.shape
        assert mx.allclose(k, k2, atol=5e-2) and mx.allclose(v, v2, atol=5e-2)
    gen.close()
    solo.close()


def test_a_row_at_a_block_start_is_not_cut_by_the_valve():
    """A row at 32 with the block [32, 112) and a valve with room for 64:
    the call stalls instead of being cut at 64 inside the block - or
    grown past the valve to 80."""
    model = gemma_with_blocks()
    spans = {0: [block(32, 112)]}
    room = Room(64)
    gen = BatchGenerator(
        model,
        prompt_batch=PrimingPromptBatch.bound(lambda uid: spans.get(uid, ())),
        prefill_step_size=128,
        prefill_slice=128,
        stop_tokens=[],
        prefill_valve=room,
    )
    gen.insert([prompt_of(200)], max_tokens=[1])
    gen.next()
    assert len(gen._prompt_batch.tokens[0]) == 32  # cut at the block's start
    gen.next()
    assert len(gen._prompt_batch.tokens[0]) == 32 and gen.block_stall.need == 80
    assert gen.stall_streak == 1 and room.calls() == [(32, None)]
    gen.close()


def test_the_query_for_a_waiting_row_asks_what_the_round_asks():
    """The lot's head needs 48 tokens for its block and the valve has room
    for 50 - under its narrowest call (64), so the round's width() for the
    row's call answers None. The query that brings a row back must answer
    the same: else the head comes back by query every round, each round
    is a block stall that does not count, and the row behind it, which
    fits (49 tokens left, its block among them), is never asked. The
    head's block itself fits (48 under 50): the stall is its call's, not
    its block's, and says so."""
    model = gemma_with_blocks()
    spans = {
        0: [block(64, 104, seed=1)],
        1: [block(64, 112, seed=2)],
        2: [block(64, 100, seed=3)],
    }
    room = Room(1 << 20)
    gen = BatchGenerator(
        model,
        prompt_batch=PrimingPromptBatch.bound(lambda uid: spans.get(uid, ())),
        prefill_step_size=128,
        prefill_slice=128,
        stop_tokens=[],
        prefill_valve=room,
    )
    uids = gen.insert(
        [prompt_of(300), prompt_of(300, shift=3), prompt_of(114, shift=6)],
        max_tokens=[1, 1, 1],
    )
    gen.next()
    assert [len(t) for t in gen._prompt_batch.tokens] == [64, 64, 64]
    # Room for 80: every row forces its block, the first fits alone (40)
    # and runs, the other two park in order.
    room.tokens = 80
    gen.next()
    assert gen._parked.uids == [uids[1], uids[2]] and room.calls()[-1] == (40, None)
    for _ in range(20):
        if uids[0] not in gen._prompt_batch.uids:
            break
        gen.next()
    assert gen._prompt_batch.uids == [] and gen._parked.uids == [uids[1], uids[2]]
    room.tokens = 50
    gen.next()
    # The head's own call (128 wide) gets no width; the row behind it, with
    # 49 tokens left, does - and runs, its block whole.
    assert gen.block_stall is None and gen._parked.uids == [uids[1]]
    assert gen._prompt_batch.uids == [uids[2]] and room.calls()[-1] == (49, None)
    assert len(gen._prompt_batch.tokens[0]) == 113
    gen.next()  # the finished prompt is handed over
    gen.next()
    # Now the head stands alone: a block stall that counts.
    stall = gen.block_stall
    assert stall.uid == uids[1] and stall.need == 48 and stall.counts
    assert not stall.at_block and gen.block_stalls == 0
    assert gen._parked is None and gen.stall_streak == 1
    gen.next()
    assert gen.block_stall.counts and gen.stall_streak == 2
    gen.close()


def test_a_stall_at_a_block_that_fits_is_refused_for_the_call():
    """A row at a block of 48 tokens with room for 50: the block fits, the
    row's call does not (the valve's narrowest is 64). The stall is the
    valve's, not the block's - the refusal says so, and the health counts
    no block stall."""
    model = gemma_with_blocks()
    room = Room(lambda: 50 if room.calls() else 1 << 20)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    (stream,) = queue_before_start(
        engine,
        [GenerationRequest(prompt_of(300), max_tokens=2, spans=[block(64, 112)])],
        ["row"],
    )
    with engine:
        _, _, error = outcome(stream)
        h = engine.health()
    assert room.calls() == [(64, ["row"])]
    assert isinstance(error, ContextTooLong)
    assert "64 of it prefilled" in str(error)
    assert "found no room for the narrowest call" in str(error)
    assert "image block" not in str(error)
    assert h["prefill_valve"]["refused"] == 1
    assert h["prefill_valve"]["block_stalled_rounds"] == 0


def test_a_row_without_a_block_stalling_beside_the_lot_is_refused_for_the_call():
    """B's block [0, 96) runs alone while A's [0, 192) waits; past its block,
    with the room under the narrowest call, B stalls on the valve with no
    block ahead: held like a block stall (nobody is admitted) but not
    counted, as B does not head the queue, and refused with the valve's
    message, not as a block stall. A, at the head, is refused first, for
    its block; then B, for its call."""
    model = gemma_with_blocks()
    engine = Engine(model, prefill_step_size=128, prefill_slice=128)
    a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(260), max_tokens=2, spans=[block(0, 192, 1)]),
            GenerationRequest(
                prompt_of(260, shift=3), max_tokens=2, spans=[block(0, 96, 2)]
            ),
        ],
        ["A", "B"],
    )

    past = []  # B has prefilled its block: the room stays small from then on

    def room_now():
        gen = engine._gen
        if gen is not None and any(len(t) >= 96 for t in gen._prompt_batch.tokens):
            past.append(True)
        return 50 if past else 128

    room = Room(room_now)
    engine.prefill_valve = room
    room.engine = engine
    with engine:
        _, _, err_b = outcome(b)
        _, _, err_a = outcome(a)
        h = engine.health()
    assert room.calls() == [(96, ["B"])]
    assert isinstance(err_b, ContextTooLong) and "96 of it prefilled" in str(err_b)
    assert "found no room for the narrowest call" in str(err_b)
    assert isinstance(err_a, ContextTooLong) and "needs 192 tokens" in str(err_a)
    assert h["prefill_valve"]["refused"] == 2
    assert h["prefill_valve"]["block_stalled_rounds"] == STALL_ROUNDS


def test_the_round_decides_from_the_reading_its_queries_answered_from(monkeypatch):
    """A's block [0, 192) waits in the lot while the text row runs; room
    for 256 under the recommended ceiling. The query that brings A back
    sees room, and what is held rises by 206 before the round asks its
    width (another thread allocating in between), for one reading. The
    round still runs A's block whole, from the reading the query answered
    from, and no round stalls: read anew, the width would find room for
    50 and stall A, uncounted since a query put it there - every round,
    were the rise to follow the queries."""
    model = gemma_with_blocks()
    rise = []  # what is held, read once more after A's query saw room

    class Rising(Room):
        rose = False

        def peek(self, wanted):
            out = super().peek(wanted)
            if out == 192 and not self.rose:
                self.rose = True
                rise.append(206)
            return out

    room = Rising(1 << 20, held=lambda: rise.pop() if rise else 0, recommended=256)
    rounds = record_rounds(monkeypatch, room)
    engine = engine_with(model, room, prefill_step_size=128, prefill_slice=128)
    a, b = queue_before_start(
        engine,
        [
            GenerationRequest(prompt_of(260), max_tokens=2, spans=[block(0, 192, 1)]),
            GenerationRequest(prompt_of(100, shift=5), max_tokens=2),
        ],
        ["A", "text"],
    )
    with engine:
        _, out_b, err_b = outcome(b)
        _, out_a, err_a = outcome(a)
        h = engine.health()
    assert room.rose and not rise  # the rise came, and was read
    assert all(r["streak"] == 0 and r["block_stall"] is None for r in rounds)
    assert err_a is None and err_b is None and len(out_a) == len(out_b) == 2
    calls = room.calls()
    assert calls[0] == (99, ["text"]) and calls[1:] == [(192, ["A"]), (67, ["A"])]
    v = h["prefill_valve"]
    assert v["stalled_rounds"] == 0 and v["block_stalled_rounds"] == 0
    assert v["refused"] == 0 and v["parked_rows"] == 1


def test_a_boundary_inside_an_image_block_is_refused():
    tokens = prompt_of(12)
    span = block(3, 10)
    for b in (3, 10):
        GenerationRequest(tokens, spans=[span], boundaries=[b])
    with pytest.raises(ValueError, match="inside an image block"):
        GenerationRequest(tokens, spans=[span], boundaries=[6])
    # A span that is no block takes any boundary.
    plain = ImageSpan(3, 10, span.features, span.digest)
    GenerationRequest(tokens, spans=[plain], boundaries=[6])


def test_prompt_refuses_a_call_that_ends_inside_a_block():
    """A call whose real tokens for a row reach into a block and stop
    before its end is refused, not clamped; a padded row whose block
    begins behind its real tokens is fine."""
    model = gemma_with_blocks()
    spans = {0: [block(3, 10)], 1: [block(6, 9, seed=1)]}
    cls = PrimingPromptBatch.bound(lambda uid: spans.get(uid, ()))

    def batch(uids):
        return cls(
            model,
            list(uids),
            [make_request_cache(model, KVPolicy()) for _ in uids],
            prefill_step_size=64,
        )

    tokens = prompt_of(12)
    with pytest.raises(ValueError, match="ends at 6, inside the image block 3..10"):
        batch([0]).prompt([tokens[:6]])
    both = batch([0, 1])
    both.prompt([tokens[:10], tokens[:5]])
    assert [len(t) for t in both.tokens] == [10, 5]
