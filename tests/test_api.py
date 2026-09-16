"""The API layer without a model: parsing, prompt building, text assembly."""

import json

import pytest

from mlx_beam.api import chat, completions, responses
from mlx_beam.api.errors import ApiError
from mlx_beam.api.text import TextAssembler
from mlx_beam.engine.request import TokenEvent
from tests.stub_tokenizer import (
    EOS,
    THINK_END,
    THINK_START,
    TOOL_END,
    TOOL_START,
    StubTokenizer,
)


def events(ids, finish="stop"):
    out = [TokenEvent(t, -0.1) for t in ids]
    if finish == "stop":
        out.append(TokenEvent(EOS, -0.1, "stop"))
    else:
        out[-1] = TokenEvent(ids[-1], -0.1, "length")
    return out


def test_chat_request_defaults_and_limits():
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    assert req.max_tokens == chat.DEFAULTS.max_completion_tokens and req.model == "m"
    # mlx-lm's defaults: greedy unless the client or a flag says otherwise.
    assert req.sampling.temperature == 0.0 and req.sampling.top_p == 1.0
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "max_completion_tokens": 7,
            "temperature": 0,
            "top_k": -1,
            "stop": "w9",
            "presence_penalty": 0.5,
        },
        "m",
    )
    assert req.max_tokens == 7 and req.sampling.top_k == 0 and req.stop == ["w9"]
    assert req.sampling.presence_penalty == 0.5


@pytest.mark.parametrize(
    "body, param",
    [
        ({"messages": []}, "messages"),
        ({"messages": [{"content": "x"}]}, "messages"),
        (
            {"messages": [{"role": "user", "content": "x"}], "temperature": 3},
            "temperature",
        ),
        ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
        (
            {"messages": [{"role": "user", "content": "x"}], "top_logprobs": 2},
            "top_logprobs",
        ),
        (
            {"messages": [{"role": "user", "content": "x"}], "xtc_threshold": 0.7},
            "xtc_threshold",
        ),
        ({"messages": [{"role": "user", "content": "x"}], "stop": 3}, "stop"),
    ],
)
def test_chat_request_rejections(body, param):
    with pytest.raises(ApiError) as exc:
        chat.parse_chat_request(body, "m")
    assert exc.value.param == param and exc.value.status == 400


def test_image_parts_name_the_missing_extra():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "w1"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
    }
    with pytest.raises(ApiError) as exc:
        chat.parse_chat_request(body, "m")
    assert exc.value.code == "extra_not_installed" and "vision" in exc.value.message


def test_chat_prompt_goes_through_the_template():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [
                {"role": "system", "content": "w1"},
                {"role": "user", "content": "w2 w3"},
            ]
        },
        "m",
    )
    gen = chat.to_generation_request(tok, req)
    assert gen.tokens == [2, 5, 1, 6, 2, 3, 7]
    assert (EOS,) in gen.stop_sequences


def test_assembler_routes_reasoning_content_and_tool_calls():
    tok = StubTokenizer()
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], tools=[{"type": "function"}])
    ids = [THINK_START, 10, 11, THINK_END, 12, TOOL_START, 20, 21, TOOL_END]
    content = reasoning = ""
    tool_calls = []
    finish = None
    for ev in events(ids):
        d = asm.feed(ev)
        content += d.content
        reasoning += d.reasoning
        tool_calls += d.tool_calls
        finish = d.finish_reason or finish
    assert reasoning == "w10 w11 "
    assert content == "w12 "
    assert tool_calls and tool_calls[0]["function"]["name"] == "w20"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"words": "w21"}
    assert finish == "tool_calls"


def test_assembler_length_flushes_and_stop_hides_the_eos():
    tok = StubTokenizer()
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7])
    deltas = [asm.feed(e) for e in events([10, 11], finish="length")]
    assert "".join(d.content for d in deltas) == "w10 w11 "
    assert deltas[-1].finish_reason == "length"
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7])
    deltas = [asm.feed(e) for e in events([10])]
    assert (
        "".join(d.content for d in deltas) == "w10 "
        and deltas[-1].finish_reason == "stop"
    )


def test_chat_responder_shapes():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "stream_options": {"include_usage": True},
        },
        "m",
    )
    gen = chat.to_generation_request(tok, req)
    out = chat.ChatResponder(tok, req, gen.tokens).complete(events([10, 11]), cached=2)
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "w10 w11 "}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["completion_tokens"] == 3
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    req.stream = True
    chunks = list(
        chat.ChatResponder(tok, req, gen.tokens).stream(events([10, 11]), cached=0)
    )
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "w10 "}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and "usage" in chunks[-1]


def test_completions_accept_token_ids_and_echo():
    tok = StubTokenizer()
    req = completions.parse_completion_request(
        {"prompt": [1, 2, 3], "echo": True, "max_tokens": 2}, "m"
    )
    gen = completions.to_generation_request(tok, req)
    assert gen.tokens == [1, 2, 3]
    out = completions.CompletionResponder(tok, req, gen.tokens).complete(
        events([10, 11], "length"), 0
    )
    assert out["choices"][0]["text"] == "w1 w2 w3 w10 w11 "
    assert out["choices"][0]["finish_reason"] == "length"
    with pytest.raises(ApiError):
        completions.parse_completion_request({"prompt": [1, "x"]}, "m")


def test_responses_items_become_messages_and_back():
    tok = StubTokenizer()
    body = {
        "instructions": "w1",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "w2"}],
            },
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "w20",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "c1", "output": "w3"},
        ],
        "tools": [{"type": "function", "name": "w20", "parameters": {}}],
        "max_output_tokens": 5,
    }
    req = responses.parse_responses_request(body, "m")
    roles = [m["role"] for m in req.chat.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert req.chat.tools == [
        {"type": "function", "function": {"name": "w20", "parameters": {}}}
    ]
    gen = responses.to_generation_request(tok, req)
    out = responses.ResponsesResponder(tok, req, gen.tokens).complete(
        events([THINK_START, 10, THINK_END, 11, TOOL_START, 20, 21, TOOL_END]), 0
    )
    kinds = [item["type"] for item in out["output"]]
    assert kinds == ["reasoning", "message", "function_call"]
    assert out["output"][1]["content"][0]["text"] == "w11 "
    assert out["output"][2]["name"] == "w20" and out["status"] == "completed"
    assert out["usage"]["input_tokens"] == len(gen.tokens)


def test_responses_refuse_state_and_hosted_tools():
    with pytest.raises(ApiError) as exc:
        responses.parse_responses_request(
            {"input": "w1", "previous_response_id": "resp_x"}, "m"
        )
    assert "full input" in exc.value.message
    with pytest.raises(ApiError) as exc:
        responses.parse_responses_request(
            {"input": "w1", "tools": [{"type": "web_search"}]}, "m"
        )
    assert exc.value.code == "unsupported"


def test_responses_stream_events():
    tok = StubTokenizer()
    req = responses.parse_responses_request({"input": "w1", "stream": True}, "m")
    gen = responses.to_generation_request(tok, req)
    evs = list(
        responses.ResponsesResponder(tok, req, gen.tokens).stream(events([10, 11]), 0)
    )
    types = [e["type"] for e in evs]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    assert types.count("response.output_text.delta") == 2
    assert evs[-1]["response"]["output"][0]["content"][0]["text"] == "w10 w11 "


def test_text_level_stop_word_ends_the_stream():
    # The engine matches stop words on token ids; the text match catches a
    # word whose tokenization differs and ends the stream instead of hiding it.
    tok = StubTokenizer()
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], stop_words=["w11"])
    deltas = [asm.feed(TokenEvent(t, -0.1)) for t in (10, 11, 12)]
    assert deltas[0].content == "w10 "
    assert deltas[1].finish_reason == "stop" and deltas[1].content == ""
    assert asm.stopped and deltas[2].empty() and deltas[2].finish_reason is None


def test_non_finite_numbers_are_rejected():
    with pytest.raises(ApiError) as exc:
        chat.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "x"}],
                "temperature": float("nan"),
            },
            "m",
        )
    assert exc.value.param == "temperature"


def test_responses_stream_has_the_item_and_part_events():
    tok = StubTokenizer()
    req = responses.parse_responses_request({"input": "w1", "stream": True}, "m")
    gen = responses.to_generation_request(tok, req)
    evs = list(
        responses.ResponsesResponder(tok, req, gen.tokens).stream(
            events([THINK_START, 10, THINK_END, 11, TOOL_START, 20, 21, TOOL_END]), 0
        )
    )
    types = [e["type"] for e in evs]
    for needed in (
        "response.in_progress",
        "response.reasoning_text.delta",
        "response.content_part.added",
        "response.output_text.delta",
        "response.content_part.done",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ):
        assert needed in types, needed
    final = evs[-1]["response"]
    assert [i["type"] for i in final["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert [e["sequence_number"] for e in evs] == list(range(1, len(evs) + 1))


def test_reasoning_tokens_are_counted():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    out = chat.ChatResponder(tok, req, gen.tokens).complete(
        events([THINK_START, 10, 11, THINK_END, 12]), cached=0
    )
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3
    assert out["choices"][0]["message"]["reasoning"] == "w10 w11 "


def test_stop_word_inside_the_last_token_keeps_nothing_behind_it():
    # finish_reason "length" flushes the buffer - but not the tail behind a
    # stop word that already matched.
    tok = StubTokenizer()
    tok._words[1] = "hello STOPtail"
    asm = TextAssembler(tok, prompt_tokens=[2], stop_words=["STOP"])
    d = asm.feed(TokenEvent(1, -0.1, "length"))
    assert d.content == "hello " and d.finish_reason == "stop"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("kind", ["chat", "completions", "responses"])
def test_responders_stop_reading_after_a_text_level_stop(kind, streaming):
    tok = StubTokenizer()
    tok._words[1] = "hello STOPtail"
    seen = []

    def engine_events():
        for i in range(4):
            seen.append(i)
            yield TokenEvent(1 if i == 0 else 3, -0.1, "length" if i == 3 else None)

    if kind == "chat":
        req = chat.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "w2"}],
                "stop": "STOP",
                "stream": streaming,
            },
            "m",
        )
        responder = chat.ChatResponder(tok, req, [2])
    elif kind == "completions":
        req = completions.parse_completion_request(
            {"prompt": [2], "stop": "STOP", "stream": streaming}, "m"
        )
        responder = completions.CompletionResponder(tok, req, [2])
    else:
        req = responses.parse_responses_request(
            {"input": "w2", "stream": streaming}, "m"
        )
        responder = responses.ResponsesResponder(tok, req, [2])
        # Responses has no stop words; end the stream with the assembler's own
        # stop instead so the loop behaviour is what is tested.
        responder.assembler = TextAssembler(tok, prompt_tokens=[2], stop_words=["STOP"])
    if streaming:
        list(responder.stream(engine_events(), 0))
    else:
        responder.complete(engine_events(), 0)
    assert seen == [0], seen


@pytest.mark.parametrize(
    "field, keys",
    [
        ("reasoning", ["reasoning"]),
        ("reasoning_content", ["reasoning_content"]),
        ("both", ["reasoning", "reasoning_content"]),
    ],
)
def test_reasoning_field_names_the_thinking(field, keys):
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    out = chat.ChatResponder(tok, req, gen.tokens, reasoning_field=field).complete(
        events([THINK_START, 10, THINK_END, 11]), cached=0
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "w11 "
    assert {k: v for k, v in msg.items() if k.startswith("reasoning")} == {
        k: "w10 " for k in keys
    }
    req.stream = True
    chunks = list(
        chat.ChatResponder(tok, req, gen.tokens, reasoning_field=field).stream(
            events([THINK_START, 10, THINK_END, 11]), cached=0
        )
    )
    first = chunks[0]["choices"][0]["delta"]
    assert all(first[k] == "w10 " for k in keys) and "content" not in first


def test_reasoning_field_none_leaves_the_markers_in_the_content():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    out = chat.ChatResponder(tok, req, gen.tokens, reasoning_field="none").complete(
        events([THINK_START, 10, THINK_END, 11]), cached=0
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "<think>w10 </think>w11 "
    assert not any(k.startswith("reasoning") for k in msg)
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    with pytest.raises(ValueError):
        chat.ChatResponder(tok, req, gen.tokens, reasoning_field="thoughts")


def test_request_defaults_precedence(tmp_path):
    from mlx_beam.api.defaults import RequestDefaults

    (tmp_path / "generation_config.json").write_text(
        '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "do_sample": true}'
    )
    d = RequestDefaults.resolve(tmp_path, flags={"top_p": 0.95, "temp": None})
    assert (d.temperature, d.top_p, d.top_k, d.min_p) == (0.7, 0.95, 20, 0.0)
    assert d.sources["temperature"] == "generation_config.json"
    assert d.sources["top_p"] == "flag" and d.sources["min_p"] == "mlx-lm"
    # A request still wins over every server-side default.
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}], "temperature": 0.1},
        "m",
        d,
    )
    assert req.sampling.temperature == 0.1 and req.sampling.top_p == 0.95
    # do_sample false is the author saying greedy, whatever temperature says.
    (tmp_path / "generation_config.json").write_text(
        '{"temperature": 0.7, "do_sample": false}'
    )
    assert RequestDefaults.resolve(tmp_path).temperature == 0.0
    # No file, no flags: mlx-lm's numbers.
    assert RequestDefaults.resolve(tmp_path / "missing").describe()["top_k"] == {
        "value": 0,
        "source": "mlx-lm",
    }


def test_seed_and_xtc_reach_the_engine_request():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "seed": 42,
            "xtc_probability": 0.5,
            "presence_penalty": 0.5,
            "presence_context_size": 7,
        },
        "m",
    )
    gen = chat.to_generation_request(tok, req)
    assert gen.sampling.seed == 42 and gen.sampling.presence_context_size == 7
    # XTC may not cut the eos or the newline; the tokenizer says which ids.
    assert EOS in gen.sampling.xtc_special_tokens
    plain = chat.to_generation_request(
        tok, chat.parse_chat_request({"messages": req.messages}, "m")
    )
    assert plain.sampling.xtc_special_tokens == () and plain.sampling.seed is None


def test_chat_logprobs_carry_text_bytes_and_alternatives():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "logprobs": True,
            "top_logprobs": 2,
        },
        "m",
    )
    gen = chat.to_generation_request(tok, req)
    assert gen.top_logprobs == 2
    evs = [
        TokenEvent(10, -0.1, None, ((10, -0.1), (11, -2.5))),
        TokenEvent(11, -0.3, None, ((11, -0.3), (10, -1.0))),
        TokenEvent(EOS, -0.1, "stop", ((EOS, -0.1), (10, -3.0))),
    ]
    out = chat.ChatResponder(tok, req, gen.tokens).complete(iter(evs), cached=0)
    content = out["choices"][0]["logprobs"]["content"]
    assert [c["token"] for c in content] == ["w10 ", "w11 ", tok.decode([EOS])]
    assert content[0]["bytes"] == list(b"w10 ")
    assert [t["token"] for t in content[0]["top_logprobs"]] == ["w10 ", "w11 "]
    req.stream = True
    chunks = list(chat.ChatResponder(tok, req, gen.tokens).stream(iter(evs), cached=0))
    streamed = [c for ch in chunks for c in ch["choices"][0]["logprobs"]["content"]]
    assert [c["token"] for c in streamed] == [c["token"] for c in content]
    # Without the flag the field stays away entirely.
    req.logprobs = False
    out = chat.ChatResponder(tok, req, gen.tokens).complete(iter(evs), cached=0)
    assert "logprobs" not in out["choices"][0]


def test_completions_logprobs_use_the_legacy_columns():
    tok = StubTokenizer()
    req = completions.parse_completion_request(
        {"prompt": "w1", "logprobs": 1, "max_tokens": 2}, "m"
    )
    gen = completions.to_generation_request(tok, req)
    assert gen.top_logprobs == 1
    evs = [
        TokenEvent(10, -0.1, None, ((10, -0.1),)),
        TokenEvent(11, -0.3, "length", ((12, -0.2),)),
    ]
    out = completions.CompletionResponder(tok, req, gen.tokens).complete(iter(evs), 0)
    lp = out["choices"][0]["logprobs"]
    assert lp["tokens"] == ["w10 ", "w11 "] and lp["token_logprobs"] == [-0.1, -0.3]
    assert lp["top_logprobs"] == [{"w10 ": -0.1}, {"w12 ": -0.2}]
    assert lp["text_offset"] == [0, 4]
    with pytest.raises(ApiError):
        completions.parse_completion_request({"prompt": "w1", "logprobs": 21}, "m")


def test_server_template_args_are_overridden_by_the_request():
    from mlx_beam.api.defaults import RequestDefaults

    d = RequestDefaults.resolve(
        flags={"chat_template_args": {"enable_thinking": False, "lang": "de"}}
    )
    msgs = [{"role": "user", "content": "w1"}]
    req = chat.parse_chat_request({"messages": msgs}, "m", d)
    assert req.template_kwargs == {"enable_thinking": False, "lang": "de"}
    req = chat.parse_chat_request(
        {"messages": msgs, "chat_template_kwargs": {"enable_thinking": True}}, "m", d
    )
    assert req.template_kwargs == {"enable_thinking": True, "lang": "de"}
    # A plain server keeps sending nothing extra.
    assert chat.parse_chat_request({"messages": msgs}, "m").template_kwargs == {}
