"""Request aliases, effort levels and the mirroring of earlier reasoning."""

import pytest

from mlx_beam.api import chat, responses
from mlx_beam.api.errors import ApiError
from mlx_beam.api.reasoning import (
    EffortCapability,
    effort_capability,
    map_effort,
    mirror_reasoning,
    probe_effort,
    read_aliases,
    renderer_reasoning_keys,
    translate_effort,
)
from tests.stub_tokenizer import StubTokenizer

MSGS = [{"role": "user", "content": "w1"}]


def test_aliases_resolve_and_the_explicit_name_wins():
    assert read_aliases({}) == {}
    assert read_aliases({"thinking_token_budget": 100}) == {"max_reasoning_tokens": 100}
    assert read_aliases({"reasoning": {"effort": "high", "max_tokens": 50}}) == {
        "reasoning_effort": "high",
        "max_reasoning_tokens": 50,
    }
    both = {
        "reasoning_effort": "low",
        "max_reasoning_tokens": 1,
        "thinking_token_budget": 2,
        "reasoning": {"effort": "high", "max_tokens": 3},
        "enable_thinking": 0,
    }
    assert read_aliases(both) == {
        "reasoning_effort": "low",
        "max_reasoning_tokens": 1,
        "enable_thinking": False,
    }


def test_effort_words():
    assert map_effort(None) is None
    assert map_effort("none") is None and map_effort("off") is None
    # The client's word stays its word; what the template makes of it is
    # decided against the template (translate_effort).
    assert map_effort("minimal") == "minimal" and map_effort("LOW") == "low"
    assert map_effort("ultra") == "ultra" and map_effort(True) == "xhigh"
    with pytest.raises(ApiError) as exc:
        map_effort("turbo")
    assert exc.value.param == "reasoning_effort"


def test_translation_follows_what_the_template_accepts():
    checks = EffortCapability(
        "template", "reasoning_effort", True, ("low", "medium", "xhigh", "high")
    )
    assert translate_effort("high", checks) == "high"
    assert translate_effort("max", checks) == "xhigh"
    assert translate_effort("ultra", checks) == "xhigh"
    assert translate_effort("minimal", checks) == "low"
    passes = EffortCapability("template", "reasoning_strength", False)
    assert translate_effort("max", passes) == "max"
    assert translate_effort("minimal", passes) == "minimal"
    assert translate_effort("high", EffortCapability("none")) is None
    only_high = EffortCapability("template", "reasoning_effort", True, ("high",))
    assert translate_effort("minimal", only_high) == "high"
    assert translate_effort("max", only_high) == "high"


def test_probe_reads_the_kwarg_and_measures_the_set():
    tok = StubTokenizer()
    tok.chat_template = "{{ message.content }}"
    assert probe_effort(tok).describe() == {"source": "none"}

    class Passes(StubTokenizer):
        chat_template = "Reasoning strength: {{ reasoning_strength }}"

        def apply_chat_template(self, messages, reasoning_strength=None, **kw):
            return super().apply_chat_template(messages, **kw)

    cap = probe_effort(Passes())
    assert cap.describe() == {
        "source": "template",
        "kwarg": "reasoning_strength",
        "validates": False,
    }

    class Budget(StubTokenizer):
        chat_template = "{% if thinking_budget %}{{ thinking_budget }}{% endif %}"

    cap = probe_effort(Budget())
    assert cap.kwarg == "thinking_budget" and not cap.validates and not cap.levels


def test_parsed_request_carries_level_switch_and_limits():
    req = chat.parse_chat_request({"messages": MSGS, "reasoning_effort": "none"}, "m")
    assert req.reasoning_effort is None
    assert req.template_kwargs == {"enable_thinking": False}
    req = chat.parse_chat_request(
        {"messages": MSGS, "reasoning": {"effort": "high", "max_tokens": 40}}, "m"
    )
    assert req.reasoning_effort == "high" and req.max_reasoning_tokens == 40
    # The template's kwarg is set when the prompt is built, against what
    # the template takes; nothing is decided at parse time.
    assert req.template_kwargs == {}
    req = chat.parse_chat_request(
        {
            "messages": MSGS,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": True},
            "min_response_tokens": 12,
            "max_prompt_tokens": 500,
        },
        "m",
    )
    # An explicit chat_template_kwargs entry beats the top-level alias.
    assert req.template_kwargs == {"enable_thinking": True}
    assert (req.min_response_tokens, req.max_prompt_tokens) == (12, 500)
    with pytest.raises(ApiError):
        chat.parse_chat_request({"messages": MSGS, "max_prompt_tokens": 0}, "m")
    r = responses.parse_responses_request(
        {"input": "w1", "reasoning": {"effort": "minimal", "max_tokens": 9}}, "m"
    )
    assert r.chat.reasoning_effort == "minimal" and r.chat.max_reasoning_tokens == 9


class LevelTokenizer(StubTokenizer):
    """A template that knows two levels and rejects the rest, Qwen-style."""

    accepted = ("medium", "xhigh")
    chat_template = "{{ reasoning_effort }} {{ message.reasoning_content }}"

    def __init__(self):
        super().__init__()
        self.seen = []

    def apply_chat_template(self, messages, reasoning_effort=None, **kw):
        self.seen.append(reasoning_effort)
        if reasoning_effort is not None and reasoning_effort not in self.accepted:
            raise ValueError(f"Unexpected reasoning effort: {reasoning_effort}")
        return super().apply_chat_template(messages, **kw)


def test_a_checking_template_gets_the_nearest_rung_it_accepts():
    tok = LevelTokenizer()
    cap = effort_capability(tok)
    assert cap.validates and cap.levels == ("medium", "xhigh")
    tok.seen.clear()
    req = chat.parse_chat_request({"messages": MSGS, "reasoning_effort": "low"}, "m")
    chat.build_prompt(tok, req)
    # No ladder: the probe knew the set, `low` went up to `medium` at once.
    assert tok.seen == ["medium"]
    assert req.template_kwargs["reasoning_effort"] == "medium"
    tok.seen.clear()
    chat.to_generation_request(tok, req)
    assert set(tok.seen) == {"medium"}
    req = chat.parse_chat_request({"messages": MSGS, "reasoning_effort": "max"}, "m")
    tok.seen.clear()
    chat.build_prompt(tok, req)
    assert tok.seen == ["xhigh"]


def test_effort_ladder_climbs_only_on_an_effort_rejection():
    # The template rejects at request time what it took at load (tools in
    # the context, say): the ladder climbs from the translated rung.
    tok = LevelTokenizer()
    effort_capability(tok)
    tok.accepted = ("xhigh",)
    tok.seen.clear()
    req = chat.parse_chat_request({"messages": MSGS, "reasoning_effort": "low"}, "m")
    chat.build_prompt(tok, req)
    assert tok.seen[0] == "medium" and tok.seen[-1] == "xhigh"
    assert req.template_kwargs["reasoning_effort"] == "xhigh"
    tok = LevelTokenizer()
    tok.accepted = ()
    with pytest.raises(ApiError) as exc:
        chat.build_prompt(
            tok,
            chat.parse_chat_request(
                {"messages": MSGS, "reasoning_effort": "high"}, "m"
            ),
        )
    assert exc.value.param == "reasoning_effort" and "high, xhigh, medium" in str(
        exc.value
    )

    class Broken(StubTokenizer):
        def apply_chat_template(self, messages, **kw):
            raise ValueError("unknown role")

    with pytest.raises(ApiError) as exc:
        chat.build_prompt(
            Broken(),
            chat.parse_chat_request({"messages": MSGS, "reasoning_effort": "low"}, "m"),
        )
    assert exc.value.param == "messages"


@pytest.mark.parametrize(
    "template, keys",
    [
        ("{{ message.reasoning_content }}", ("reasoning_content",)),
        ("{{ m['thinking'] }} {{ enable_thinking }}", ("thinking",)),
        ('{% if message["reasoning"] %}{{ reasoning_effort }}', ("reasoning",)),
        (
            "{{ message.reasoning_content }}{{ message.thinking }}",
            ("reasoning_content", "thinking"),
        ),
        ("{{ message.content }}", ()),
    ],
)
def test_renderer_keys_come_from_the_template_text(template, keys):
    tok = StubTokenizer()
    tok.chat_template = template
    assert renderer_reasoning_keys(tok) == keys


def test_renderer_keys_come_from_a_python_renderer_too():
    from mlx_beam._vendor.mlx_lm.chat_templates import deepseek_v32

    tok = StubTokenizer()
    tok._chat_template = deepseek_v32.apply_chat_template
    assert renderer_reasoning_keys(tok) == ("reasoning_content",)


def test_mirroring_fills_only_the_keys_the_renderer_reads_and_the_client_left():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning": "r1"},
        {"role": "assistant", "content": "b", "thinking": "t", "reasoning": "r2"},
        {
            "role": "assistant",
            "content": "c",
            "reasoning_content": "",
            "reasoning": "r3",
        },
        {"role": "assistant", "content": "d"},
    ]
    mirror_reasoning(msgs, ("reasoning_content",))
    assert msgs[1]["reasoning_content"] == "r1" and msgs[1]["reasoning"] == "r1"
    # thinking outranks reasoning as the source.
    assert msgs[2]["reasoning_content"] == "t"
    assert msgs[3]["reasoning_content"] == "r3"
    assert "reasoning_content" not in msgs[4] and "reasoning" not in msgs[0]
    # A key the client filled is never overwritten.
    msgs = [
        {"role": "assistant", "content": "a", "reasoning_content": "x", "thinking": "y"}
    ]
    mirror_reasoning(msgs, ("reasoning_content", "thinking"))
    assert msgs[0] == {
        "role": "assistant",
        "content": "a",
        "reasoning_content": "x",
        "thinking": "y",
    }
    # A renderer that reads none of them leaves the message alone.
    msgs = [{"role": "assistant", "content": "a", "reasoning": "r"}]
    mirror_reasoning(msgs, ())
    assert msgs[0] == {"role": "assistant", "content": "a", "reasoning": "r"}
    # Through build_prompt: the stub template reads reasoning_content.
    req = chat.parse_chat_request(
        {
            "messages": [
                {"role": "user", "content": "w1"},
                {"role": "assistant", "content": "w2", "reasoning": "w3"},
                {"role": "user", "content": "w4"},
            ]
        },
        "m",
    )
    chat.build_prompt(StubTokenizer(), req)
    assert req.messages[1]["reasoning_content"] == "w3"


@pytest.mark.parametrize(
    "server, body, expected",
    [
        (
            {"enable_thinking": False},
            {"enable_thinking": True},
            {"enable_thinking": True},
        ),
        (
            {"enable_thinking": True},
            {"reasoning_effort": "none"},
            {"enable_thinking": False},
        ),
        # The request's effort replaces the server's; it reaches the template
        # in build_prompt, translated - nothing sits in the kwargs yet.
        (
            {"reasoning_effort": "low"},
            {"reasoning_effort": "high"},
            {},
        ),
        ({"lang": "de"}, {}, {"lang": "de"}),
    ],
)
def test_request_aliases_override_the_server_template_args(server, body, expected):
    from mlx_beam.api.defaults import RequestDefaults

    d = RequestDefaults.resolve(flags={"chat_template_args": server})
    req = chat.parse_chat_request({"messages": MSGS, **body}, "m", d)
    assert req.template_kwargs == expected
    # Explicit chat_template_kwargs still sit on top of both.
    req = chat.parse_chat_request(
        {"messages": MSGS, **body, "chat_template_kwargs": {"enable_thinking": None}},
        "m",
        d,
    )
    assert req.template_kwargs["enable_thinking"] is None


def test_responses_take_the_prompt_cap_and_the_reserve():
    from mlx_beam.api.defaults import RequestDefaults

    d = RequestDefaults.resolve(flags={"min_response_tokens": 12})
    r = responses.parse_responses_request(
        {"input": "w1", "max_prompt_tokens": 2, "min_response_tokens": 0}, "m", d
    )
    gen = responses.to_generation_request(StubTokenizer(), r, d)
    assert gen.max_prompt_tokens == 2 and gen.min_response_tokens == 0
    r = responses.parse_responses_request({"input": "w1"}, "m", d)
    assert (
        responses.to_generation_request(StubTokenizer(), r, d).min_response_tokens == 12
    )
    with pytest.raises(ApiError):
        responses.parse_responses_request({"input": "w1", "max_prompt_tokens": 0}, "m")
