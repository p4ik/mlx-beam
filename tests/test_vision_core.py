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
        # The assistant's frame is the last token: the reasoning state is
        # searched from there, not from the user's text.
        return Built(tokens, spans, assistant_start=len(tokens) - 1)

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
    assert image_capabilities(loaded) == {"input_embeddings": True, "layer_hook": True}
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


def test_unequal_prefill_rows_with_spans_are_a_loud_error():
    """The engine cuts every prefill call to the shortest segment; a call
    with rows of unequal length and an image span in them is a bug, and
    raises instead of dropping the images."""
    from mlx_beam.engine.priming import PrimingPromptBatch

    model = tiny_qwen35()
    front = FakeFrontend(StubTokenizer())
    _, built = request(front, [png(7)])
    cls = PrimingPromptBatch.bound(lambda uid: built.spans if uid == 1 else ())
    caches = [make_request_cache(model, KVPolicy()) for _ in range(2)]
    batch = cls(model, [1, 2], caches, prefill_step_size=64)
    with pytest.raises(ValueError, match="unequal length"):
        batch.prompt([built.tokens, built.tokens[:-2]])


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
