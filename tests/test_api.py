"""The API layer without a model: parsing, prompt building, text assembly."""

import json

import pytest

from mlx_beam.api import chat, completions, responses
from mlx_beam.api.errors import ApiError
from mlx_beam.api.text import TextAssembler, initial_state
from mlx_beam.engine.request import TokenEvent
from tests.stub_tokenizer import (
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    EOS,
    LABEL,
    LABEL_SPACED,
    NEWLINE,
    THINK_END,
    THINK_START,
    TOOL_END,
    TOOL_START,
    ChannelStubTokenizer,
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


def test_tool_call_arguments_reach_the_template_as_a_mapping():
    """The wire carries arguments as a JSON string and content as null; the
    Qwen3.8 template raises on a string, others trim the content."""

    def call(args):
        return {
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": args},
        }

    body = {
        "messages": [
            {"role": "user", "content": "w1"},
            {"role": "assistant", "content": None, "tool_calls": [call('{"a": 1}')]},
            {"role": "tool", "tool_call_id": "call_1", "content": "w2"},
            {"role": "assistant", "content": None, "tool_calls": [call("")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "w3"},
            {"role": "assistant", "content": None, "tool_calls": [call({"b": 2})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "w4"},
        ]
    }
    req = chat.parse_chat_request(body, "m")
    turns = [m for m in req.messages if m.get("tool_calls")]
    assert [t["content"] for t in turns] == ["", "", ""]
    assert [t["tool_calls"][0]["function"]["arguments"] for t in turns] == [
        {"a": 1},
        {},
        {"b": 2},
    ]
    # The request body is left as the client sent it.
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    for bad in ("{not json", "[1, 2]", 3):
        body["messages"][1]["tool_calls"][0]["function"]["arguments"] = bad
        with pytest.raises(ApiError) as exc:
            chat.parse_chat_request(body, "m")
        assert exc.value.param == "messages" and "arguments" in exc.value.message


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
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "w9"}],
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
    # The reasoning the client handed back rides on the turn it preceded.
    assert req.chat.messages[2]["reasoning_content"] == "w9"
    assert req.instructions == "w1" and req.tool_choice == "auto"
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
    assert out["instructions"] == "w1"


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
    req = responses.parse_responses_request(
        {
            "input": "w1",
            "stream": True,
            "tools": [{"type": "function", "name": "w20", "parameters": {}}],
        },
        "m",
    )
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


def assembled(asm, ids, finish="stop"):
    deltas = [asm.feed(e) for e in events(ids, finish)]
    return "".join(d.reasoning for d in deltas), "".join(d.content for d in deltas)


@pytest.mark.parametrize("label", [LABEL, LABEL_SPACED])
def test_a_channel_is_thinking_whatever_token_its_label_comes_as(label):
    # Gemma 4's template writes `<|channel>thought`; after a tool response
    # the model opens the channel itself and writes the label as another
    # token (` thought`). The marker is the channel, the label up to the
    # line end belongs to it - neither is text.
    tok = ChannelStubTokenizer()
    ids = [CHANNEL_OPEN, label, NEWLINE, 10, 11, CHANNEL_CLOSE, 12]
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7])
    assert assembled(asm, ids) == ("w10 w11 ", "w12 ")
    # Marker, label and line end count like any opener's tokens.
    assert asm.reasoning_tokens == 5
    # Routing off: the client gets the block as the model wrote it.
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], route_thinking=False)
    reasoning, content = assembled(asm, ids)
    assert reasoning == ""
    assert content == "<|channel>" + tok.decode([label]) + "\nw10 w11 <channel|>w12 "


def test_a_channel_closed_on_its_label_line_is_an_empty_block():
    tok = ChannelStubTokenizer()
    ids = [CHANNEL_OPEN, LABEL, CHANNEL_CLOSE, 12]
    assert assembled(TextAssembler(tok, prompt_tokens=[6, 1, 7]), ids) == ("", "w12 ")
    asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], route_thinking=False)
    assert assembled(asm, ids) == ("", "<|channel>thought<channel|>w12 ")


@pytest.mark.parametrize("route", [True, False])
def test_a_label_and_its_line_end_in_one_segment_are_still_the_opener(route):
    # The automaton reports one text per segment and the state after it;
    # a detokenizer may hand the label and the line end over together.
    tok = ChannelStubTokenizer()
    tok._words[40] = "thought\n"
    tok._words[41] = "thought\nw10 "
    for merged, rest in ((40, "w10 "), (41, "")):
        ids = [CHANNEL_OPEN, merged] + ([10] if rest else []) + [CHANNEL_CLOSE, 11]
        asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], route_thinking=route)
        got = assembled(asm, ids)
        if route:
            assert got == ("w10 ", "w11 ")
        else:
            assert got == ("", "<|channel>thought\nw10 <channel|>w11 ")


def test_a_stop_word_inside_the_label_leaves_no_label_behind():
    tok = ChannelStubTokenizer()
    for stop, ids in (
        ("ught", [CHANNEL_OPEN, LABEL]),
        ("w1", [CHANNEL_OPEN, LABEL, 10]),
    ):
        asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], stop_words=[stop])
        assert assembled(asm, ids) == ("", "")


@pytest.mark.parametrize("route", [True, False])
def test_a_stop_word_after_the_label_end_in_one_segment_keeps_the_text_before_it(
    route,
):
    # Label, line end, reasoning and the stop word in one segment: the label
    # goes, the reasoning before the stop stays.
    tok = ChannelStubTokenizer()
    tok._words[40] = "thought\nhello STOP"
    asm = TextAssembler(
        tok, prompt_tokens=[6, 1, 7], stop_words=["STOP"], route_thinking=route
    )
    deltas = [asm.feed(TokenEvent(t, -0.1)) for t in (CHANNEL_OPEN, 40)]
    got = "".join(d.reasoning for d in deltas), "".join(d.content for d in deltas)
    assert got == (("hello ", "") if route else ("", "<|channel>thought\nhello "))
    assert asm.stopped and deltas[-1].finish_reason == "stop"


@pytest.mark.parametrize(
    "tail, generated",
    [
        ([CHANNEL_OPEN], [LABEL, NEWLINE, 10, CHANNEL_CLOSE, 11]),
        ([CHANNEL_OPEN, LABEL], [NEWLINE, 10, CHANNEL_CLOSE, 11]),
    ],
)
def test_a_prompt_that_ends_inside_the_label_starts_in_the_label(tail, generated):
    tok = ChannelStubTokenizer()
    prompt = [6, 1, 7] + tail
    assert initial_state(tok, prompt)[0] == "label"
    assert chat.reasoning_limits(tok, prompt, 100).seeded
    assert assembled(TextAssembler(tok, prompt_tokens=prompt), generated) == (
        "w10 ",
        "w11 ",
    )
    asm = TextAssembler(tok, prompt_tokens=prompt, route_thinking=False)
    assert assembled(asm, generated) == ("", "<|channel>thought\nw10 <channel|>w11 ")


def test_a_channel_the_prompt_opened_continues_as_thinking():
    tok = ChannelStubTokenizer()
    prompt = [6, 1, 7, CHANNEL_OPEN, LABEL, NEWLINE]
    ids = [10, CHANNEL_CLOSE, 11]
    assert assembled(TextAssembler(tok, prompt_tokens=prompt), ids) == ("w10 ", "w11 ")
    asm = TextAssembler(tok, prompt_tokens=prompt, route_thinking=False)
    assert assembled(asm, ids) == ("", "<|channel>thought\nw10 <channel|>w11 ")


def test_the_channel_family_is_inferred_on_the_marker_alone():
    from mlx_beam._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper

    class Gemmaish:
        """What the wrapper reads: a vocabulary with the channel markers."""

        eos_token_id = 1
        chat_template = "{{ messages }}"
        _vocab = {
            "<eos>": 1,
            "<|channel>": 5,
            "<channel|>": 6,
            "thought": 7,
            "\n": 8,
            "a": 9,
            ",": 10,
            "b": 11,
        }

        def get_vocab(self):
            return self._vocab

        def encode(self, text, add_special_tokens=False):
            if text.strip(" ") == "\n\n":
                return [8, 8]
            return [self._vocab[t] for t in text.replace(" ", "").split(",") if t]

        def decode(self, ids, **kw):
            names = {v: k for k, v in self._vocab.items()}
            return "".join(names.get(i, "") for i in ids)

        def apply_chat_template(self, messages, **kw):
            return [9]

    tok = TokenizerWrapper(Gemmaish())
    assert tok.think_start == "<|channel>" and tok.think_start_tokens == (5,)
    assert tok.think_end == "<channel|>" and tok.think_end_tokens == (6,)
    assert tok.think_label_end == "\n"
    # The engine's budget matches the opener on that one token too.
    limits = chat.reasoning_limits(tok, [9], 100)
    assert limits.start == (5,) and limits.end == (6,)

    class Qwenish(Gemmaish):
        _vocab = {
            "<eos>": 1,
            "<think>": 5,
            "</think>": 6,
            "\n": 8,
            "a": 9,
            ",": 10,
            "b": 11,
        }

    tok = TokenizerWrapper(Qwenish())
    assert tok.think_start == "<think>" and tok.think_label_end is None


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
    # The automaton runs in every mode; only the routing follows the switch.
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
    with pytest.raises(ValueError):
        chat.ChatResponder(tok, req, gen.tokens, reasoning_field="thoughts")


def test_reasoning_field_none_shows_a_think_block_the_prompt_opened():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    # Qwen-style: the template ends the prompt with the open marker, the
    # model never emits it, so a client parsing the text would miss it.
    prompt = chat.to_generation_request(tok, req).tokens + [THINK_START]
    out = chat.ChatResponder(tok, req, prompt, reasoning_field="none").complete(
        events([10, THINK_END, 11]), cached=0
    )
    assert out["choices"][0]["message"]["content"] == "<think>w10 </think>w11 "
    assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1
    req.stream = True
    chunks = list(
        chat.ChatResponder(tok, req, prompt, reasoning_field="none").stream(
            events([10, THINK_END, 11]), cached=0
        )
    )
    assert chunks[0]["choices"][0]["delta"]["content"] == "<think>w10 "
    # The block closes on the very first token: the opener still goes first.
    out = chat.ChatResponder(tok, req, prompt, reasoning_field="none").complete(
        events([THINK_END, 11]), cached=0
    )
    assert out["choices"][0]["message"]["content"] == "<think></think>w11 "
    # Routed, the same prompt yields the reasoning field, no marker anywhere.
    out = chat.ChatResponder(tok, req, prompt).complete(
        events([10, THINK_END, 11]), cached=0
    )
    msg = out["choices"][0]["message"]
    assert msg["reasoning"] == "w10 " and msg["content"] == "w11 "


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
    # A default a request could not ask for is refused at start, not later
    # on every request that leaves the field to the server.
    (tmp_path / "generation_config.json").write_text('{"temperature": 3.0}')
    with pytest.raises(ValueError, match="temperature from generation_config"):
        RequestDefaults.resolve(tmp_path)
    with pytest.raises(ValueError, match="top_p from flag"):
        RequestDefaults.resolve(tmp_path / "missing", flags={"top_p": 1.5})


def test_logit_bias_values_are_bounded_numbers():
    for bad in ({"1": "nan"}, {"1": "inf"}, {"1": 1e6}, {"1": True}):
        with pytest.raises(ApiError) as exc:
            chat.parse_chat_request(
                {"messages": [{"role": "user", "content": "w1"}], "logit_bias": bad},
                "m",
            )
        assert exc.value.param == "logit_bias", bad
    with pytest.raises(ApiError) as exc:
        chat.parse_chat_request(
            {"messages": [{"role": "user", "content": "w1"}], "n": True}, "m"
        )
    assert exc.value.param == "n"


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
    assert [c["token"] for c in content] == ["w10 ", "w11 "]  # no stop token
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


def test_forced_close_keeps_a_replacement_char_without_byte_evidence():
    # Only a detokenizer that shows its held bytes (BPE, SPM - see the
    # ByteTokenizer test) can prove a U+FFFD was a cut character. The stub
    # cannot, so its U+FFFD stays: a spare one is cosmetic, a deleted real
    # character is not.
    tok = StubTokenizer()
    tok._words[40] = "�\n"
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    evs = [
        TokenEvent(THINK_START, -0.1),
        TokenEvent(10, -0.1),
        TokenEvent(40, 0.0, forced=True),
        TokenEvent(THINK_END, 0.0, forced=True),
        TokenEvent(11, -0.1),
        TokenEvent(EOS, -0.1, "stop"),
    ]
    out = chat.ChatResponder(tok, req, gen.tokens).complete(iter(evs), cached=0)
    msg = out["choices"][0]["message"]
    assert msg["reasoning"] == "w10 \ufffd\n" and msg["content"] == "w11 "
    req.stream = True
    chunks = list(chat.ChatResponder(tok, req, gen.tokens).stream(iter(evs), cached=0))
    joined = "".join(c["choices"][0]["delta"].get("reasoning", "") for c in chunks)
    assert joined == "w10 \ufffd\n"
    # In none mode the same text sits in the content, markers and all.
    out = chat.ChatResponder(tok, req, gen.tokens, reasoning_field="none").complete(
        iter(evs), cached=0
    )
    assert out["choices"][0]["message"]["content"] == "<think>w10 \ufffd\n</think>w11 "
    # The model's own U+FFFD, not forced, is left alone.
    evs[2] = TokenEvent(40, -0.1)
    out = chat.ChatResponder(tok, req, gen.tokens).complete(iter(evs), cached=0)
    assert out["choices"][0]["message"]["reasoning"] == "w10 �\n"


class PickyTokenizer(StubTokenizer):
    """A tool parser that rejects calls whose name starts with 'bad'."""

    @staticmethod
    def tool_parser(text: str, tools):
        name, _, rest = text.strip().partition(" ")
        if name.startswith("bad") or not name:
            raise ValueError(f"no such tool: {name!r}")
        return {"name": name, "arguments": {"words": rest.strip()}}


def _tool_events(*blocks, finish="stop"):
    """Token events for tool blocks; each block is a list of word ids."""
    out = []
    for words in blocks:
        out += [TokenEvent(TOOL_START, -0.1)]
        out += [TokenEvent(w, -0.1) for w in words]
        out += [TokenEvent(TOOL_END, -0.1)]
    if finish == "stop":
        out.append(TokenEvent(EOS, -0.1, "stop"))
    else:
        out[-1] = TokenEvent(out[-1].token, -0.1, "length")
    return out


def _chat(tok, stream=False):
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "tools": [{"type": "function", "function": {"name": "w10"}}],
            "stream": stream,
        },
        "m",
    )
    return req, chat.to_generation_request(tok, req).tokens


def test_a_tool_call_that_does_not_parse_comes_back_as_text():
    tok = PickyTokenizer()
    tok._words[20] = "bad "
    req, prompt = _chat(tok)
    out = chat.ChatResponder(tok, req, prompt).complete(_tool_events([20, 11]), 0)
    choice = out["choices"][0]
    # No call was made, so no call is reported; the client sees the text.
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "<tool_call>bad w11 </tool_call>"


def test_valid_calls_survive_next_to_an_invalid_one():
    tok = PickyTokenizer()
    tok._words[20] = "bad "
    req, prompt = _chat(tok)
    out = chat.ChatResponder(tok, req, prompt).complete(
        _tool_events([10, 11], [20, 12], [10, 13]), 0
    )
    choice = out["choices"][0]
    names = [c["function"]["name"] for c in choice["message"]["tool_calls"]]
    assert names == ["w10", "w10"] and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "<tool_call>bad w12 </tool_call>"


def test_a_call_cut_by_the_length_limit_keeps_its_text_and_reason():
    tok = PickyTokenizer()
    tok._words[20] = "bad "
    req, prompt = _chat(tok)
    evs = _tool_events([20, 11], finish="length")[:-1]
    evs[-1] = TokenEvent(11, -0.1, "length")
    out = chat.ChatResponder(tok, req, prompt).complete(iter(evs), 0)
    choice = out["choices"][0]
    assert choice["finish_reason"] == "length" and "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "<tool_call>bad w11 "
    # Streamed: the same text arrives as a content chunk, no tool-call chunk.
    req.stream = True
    chunks = list(chat.ChatResponder(tok, req, prompt).stream(iter(evs), 0))
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert (
        content == "<tool_call>bad w11 "
        and chunks[-1]["choices"][0]["finish_reason"] == "length"
    )
    assert not any("tool_calls" in c["choices"][0]["delta"] for c in chunks)


class MistralLike(StubTokenizer):
    """Mistral's shape: a start marker, no end marker, several calls in a row."""

    tool_call_start = "[TOOL_CALLS]"
    tool_call_end = ""

    def __init__(self):
        super().__init__()
        self._words[TOOL_START] = "[TOOL_CALLS]"
        self._words[30] = 'good[ARGS]{"x": 1}'
        self._words[31] = 'bad[ARGS]{"x":'
        self._words[32] = 'other[ARGS]{"y": 2}'

    @staticmethod
    def tool_parser(text, tools):
        from mlx_beam._vendor.mlx_lm.tool_parsers import mistral

        return mistral.parse_tool_call(text, tools)


def test_mistral_calls_before_a_cut_are_kept_and_the_cut_one_is_text():
    tok = MistralLike()
    req, prompt = _chat(tok)
    # good, then a second call cut by the length limit.
    evs = [
        TokenEvent(TOOL_START, -0.1),
        TokenEvent(30, -0.1),
        TokenEvent(TOOL_START, -0.1),
        TokenEvent(31, -0.1, "length"),
    ]
    out = chat.ChatResponder(tok, req, prompt).complete(iter(evs), 0)
    choice = out["choices"][0]
    assert [c["function"]["name"] for c in choice["message"]["tool_calls"]] == ["good"]
    assert choice["message"]["content"] == '[TOOL_CALLS]bad[ARGS]{"x":'
    assert choice["finish_reason"] == "length"
    # Two complete calls, ended by the eos: both parsed, a tool-call turn.
    evs = [
        TokenEvent(TOOL_START, -0.1),
        TokenEvent(30, -0.1),
        TokenEvent(TOOL_START, -0.1),
        TokenEvent(32, -0.1),
        TokenEvent(EOS, -0.1, "stop"),
    ]
    out = chat.ChatResponder(tok, req, prompt).complete(iter(evs), 0)
    choice = out["choices"][0]
    assert [c["function"]["name"] for c in choice["message"]["tool_calls"]] == [
        "good",
        "other",
    ]
    assert (
        choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    )


def test_mistral_marker_inside_a_json_string_is_not_a_call():
    """A quoted "[TOOL_CALLS]…" in an argument must never become a call of
    its own, and the complete call that holds it must survive a later cut."""
    tok = MistralLike()
    quoted = "Example: [TOOL_CALLS]example_action[ARGS]{}"
    tok._words[33] = "write[ARGS]" + json.dumps({"content": quoted})
    tools = [
        {"type": "function", "function": {"name": n}}
        for n in ("write", "example_action", "bad")
    ]
    for stream in (False, True):
        req, prompt = _chat(tok, stream=stream)
        req.tools = tools
        evs = [
            TokenEvent(TOOL_START, -0.1),
            TokenEvent(33, -0.1),
            TokenEvent(TOOL_START, -0.1),
            TokenEvent(31, -0.1, "length"),
        ]
        responder = chat.ChatResponder(tok, req, prompt)
        if stream:
            chunks = list(responder.stream(iter(evs), 0))
            calls = [
                c
                for ch in chunks
                for c in ch["choices"][0]["delta"].get("tool_calls", [])
            ]
            text = "".join(
                ch["choices"][0]["delta"].get("content", "") for ch in chunks
            )
            finish = chunks[-1]["choices"][0]["finish_reason"]
        else:
            msg = responder.complete(iter(evs), 0)["choices"][0]
            calls, text, finish = (
                msg["message"]["tool_calls"],
                msg["message"]["content"],
                msg["finish_reason"],
            )
        assert [c["function"]["name"] for c in calls] == ["write"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"content": quoted}
        assert text == '[TOOL_CALLS]bad[ARGS]{"x":' and finish == "length"


def test_a_parser_result_json_cannot_carry_comes_back_as_text():
    """The Qwen parser evaluates literals; a set or bytes is a Python value
    JSON has no form for. That is an unreadable call, not a 500."""

    class SetTokenizer(StubTokenizer):
        @staticmethod
        def tool_parser(text, tools):
            return {"name": "f", "arguments": {"x": {1, 2}}}

    tok = SetTokenizer()
    req, prompt = _chat(tok)
    out = chat.ChatResponder(tok, req, prompt).complete(_tool_events([10]), 0)
    choice = out["choices"][0]
    assert "tool_calls" not in choice["message"] and choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "<tool_call>w10 </tool_call>"


class ByteTokenizer(StubTokenizer):
    """The stub vocabulary as raw bytes, detokenized by mlx-lm's real BPE or
    SPM detokenizer: token 40 is the first byte of a cut character, token 10
    a complete U+FFFD, token 41 a newline."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.raw = [w.encode() for w in self._words]
        self.raw[10] = "�".encode()
        self.raw[40] = b"\xe2"
        self.raw[41] = b"\n"
        self.raw[42] = b" <"

    def __len__(self):
        return len(self.raw)

    def convert_ids_to_tokens(self, ids):
        from mlx_beam._vendor.mlx_lm.tokenizer_utils import _byte_decoder

        if self.kind == "bpe":
            to_char = {v: k for k, v in _byte_decoder().items()}
            return ["".join(to_char[b] for b in self.raw[t]) for t in ids]
        return [
            f"<0x{self.raw[t][0]:02X}>" if t == 40 else self.raw[t].decode()
            for t in ids
        ]

    @property
    def detokenizer(self):
        from mlx_beam._vendor.mlx_lm import tokenizer_utils as tu

        cls = {"bpe": tu.BPEStreamingDetokenizer, "spm": tu.SPMStreamingDetokenizer}
        return cls[self.kind](self)


@pytest.mark.parametrize("kind", ["bpe", "spm"])
@pytest.mark.parametrize("held", ["fragment", "real"])
def test_forced_close_drops_a_byte_fragment_but_keeps_a_real_replacement_char(
    kind, held
):
    tok = ByteTokenizer(kind)
    token = 40 if held == "fragment" else 10
    req, prompt = _chat(tok)
    evs = [
        TokenEvent(THINK_START, -0.1),
        TokenEvent(token, -0.1),
        TokenEvent(41, 0.0, forced=True),
        TokenEvent(THINK_END, 0.0, forced=True),
        TokenEvent(11, -0.1),
        TokenEvent(EOS, -0.1, "stop"),
    ]
    out = chat.ChatResponder(tok, req, prompt).complete(iter(evs), 0)
    reasoning = out["choices"][0]["message"]["reasoning"]
    assert reasoning == ("\n" if held == "fragment" else "�\n")


def test_text_held_back_at_the_eos_is_not_lost():
    """A ' <' the automaton keeps as a possible marker start, and a lone
    space the detokenizer holds, are the model's text when the eos comes."""
    tok = ByteTokenizer("bpe")
    req, prompt = _chat(tok)
    evs = [TokenEvent(11, -0.1), TokenEvent(42, -0.1), TokenEvent(EOS, -0.1, "stop")]
    out = chat.ChatResponder(tok, req, prompt).complete(iter(evs), 0)
    assert out["choices"][0]["message"]["content"] == "w11  <"


def test_completions_return_the_raw_text_markers_and_tool_blocks_included():
    tok = StubTokenizer()
    req = completions.parse_completion_request({"prompt": [1, 2]}, "m")
    ids = [THINK_START, 10, THINK_END, 11, TOOL_START, 20, 21, TOOL_END, 12]
    out = completions.CompletionResponder(tok, req, [1, 2]).complete(events(ids), 0)
    choice = out["choices"][0]
    assert (
        choice["text"] == "<think>w10 </think>w11 <tool_call>w20 w21 </tool_call>w12 "
    )
    assert choice["finish_reason"] == "stop"


def test_chat_without_tools_on_offer_keeps_a_tool_block_as_text():
    tok = StubTokenizer()
    req, prompt = _chat(tok)
    req.tools = None
    out = chat.ChatResponder(tok, req, prompt).complete(_tool_events([20, 21]), 0)
    choice = out["choices"][0]
    assert "tool_calls" not in choice["message"] and choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "<tool_call>w20 w21 </tool_call>"


def test_responses_items_that_interleave_get_their_own_ids_and_text():
    """Text, a call, then text again is three items with three ids; the
    second message holds only its own text, and complete() agrees with the
    final stream event."""
    tok = PickyTokenizer()
    tok._words[20] = "bad "
    body = {
        "input": "w1",
        "stream": True,
        "tools": [{"type": "function", "name": "w10", "parameters": {}}],
    }
    req = responses.parse_responses_request(body, "m")
    gen = responses.to_generation_request(tok, req)

    def evs():
        return iter(
            [TokenEvent(12, -0.1)] + _tool_events([10, 11], [20, 13], finish="stop")
        )

    stream = list(responses.ResponsesResponder(tok, req, gen.tokens).stream(evs(), 0))
    output = stream[-1]["response"]["output"]
    assert [i["type"] for i in output] == ["message", "function_call", "message"]
    assert output[0]["id"] != output[2]["id"]
    assert output[0]["content"][0]["text"] == "w12 "
    assert output[2]["content"][0]["text"] == "<tool_call>bad w13 </tool_call>"
    done = [e for e in stream if e["type"] == "response.output_text.done"]
    assert [e["text"] for e in done] == ["w12 ", "<tool_call>bad w13 </tool_call>"]
    assert all("logprobs" in e for e in done)
    args_done = [
        e for e in stream if e["type"] == "response.function_call_arguments.done"
    ]
    assert args_done[0]["name"] == "w10"
    req.chat.stream = False
    whole = responses.ResponsesResponder(tok, req, gen.tokens).complete(evs(), 0)

    def strip(items):
        return [
            {k: v for k, v in i.items() if k not in ("id", "call_id")} for i in items
        ]

    assert strip(whole["output"]) == strip(output)


@pytest.mark.parametrize(
    "body, param",
    [
        (
            {"messages": [{"role": "user", "content": "x"}], "response_format": "json"},
            "response_format",
        ),
        (
            {"messages": [{"role": "user", "content": "x"}], "stream_options": [1]},
            "stream_options",
        ),
        (
            {
                "messages": [{"role": "user", "content": "x"}],
                "chat_template_kwargs": "a",
            },
            "chat_template_kwargs",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "text", "text": 5}]}]},
            "messages",
        ),
    ],
)
def test_wrongly_shaped_nested_fields_are_400s_not_500s(body, param):
    with pytest.raises(ApiError) as exc:
        chat.parse_chat_request(body, "m")
    assert exc.value.status == 400 and exc.value.param == param


def test_responses_and_completions_reject_wrong_shapes_and_suffix():
    for body, param in (
        ({"input": "w1", "metadata": "x"}, "metadata"),
        ({"input": "w1", "text": "abc"}, "text"),
        ({"input": "w1", "text": {"format": "json"}}, "text"),
        (
            {
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": 1}]}
                ]
            },
            "input",
        ),
    ):
        with pytest.raises(ApiError) as exc:
            responses.parse_responses_request(body, "m")
        assert exc.value.status == 400 and exc.value.param == param, body
    with pytest.raises(ApiError) as exc:
        completions.parse_completion_request({"prompt": "w1", "suffix": "w2"}, "m")
    assert exc.value.param == "suffix"
    with pytest.raises(ApiError) as exc:
        completions.parse_completion_request({"prompt": "w1", "stream_options": 3}, "m")
    assert exc.value.param == "stream_options"


def test_completions_stream_usage_rides_on_a_choiceless_last_chunk():
    tok = StubTokenizer()
    req = completions.parse_completion_request(
        {"prompt": [1, 2], "stream": True, "stream_options": {"include_usage": True}},
        "m",
    )
    chunks = list(
        completions.CompletionResponder(tok, req, [1, 2]).stream(events([10, 11]), 0)
    )
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 3
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in c for c in chunks[:-1])


def test_a_cut_mistral_json_list_comes_back_as_text_not_as_a_quoted_call():
    tok = MistralLike()
    quoted = "Example: example_action[ARGS]{}"
    tok._words[34] = (
        '[{"name": "write", "arguments": {"content": "' + quoted + '"}}, '
        '{"name": "bad", "arguments":'
    )
    tools = [
        {"type": "function", "function": {"name": n}}
        for n in ("write", "example_action", "bad")
    ]
    for stream in (False, True):
        req, prompt = _chat(tok, stream=stream)
        req.tools = tools
        evs = [TokenEvent(TOOL_START, -0.1), TokenEvent(34, -0.1, "length")]
        responder = chat.ChatResponder(tok, req, prompt)
        if stream:
            chunks = list(responder.stream(iter(evs), 0))
            calls = [
                c
                for ch in chunks
                for c in ch["choices"][0]["delta"].get("tool_calls", [])
            ]
            text = "".join(
                ch["choices"][0]["delta"].get("content", "") for ch in chunks
            )
        else:
            msg = responder.complete(iter(evs), 0)["choices"][0]["message"]
            calls, text = msg.get("tool_calls", []), msg["content"]
        assert not calls and text.startswith("[TOOL_CALLS][{")


def test_responses_cut_message_item_is_incomplete_on_the_wire_already():
    """The status an item is sent with is the status it keeps; a client
    that stores items as their done events arrive must not see completed
    for a message the final response calls incomplete."""
    tok = StubTokenizer()
    req = responses.parse_responses_request({"input": "w1", "stream": True}, "m")
    gen = responses.to_generation_request(tok, req)
    wire = [
        json.loads(json.dumps(e))  # as the handler serialises, event by event
        for e in responses.ResponsesResponder(tok, req, gen.tokens).stream(
            iter([TokenEvent(10, -0.1, "length")]), 0
        )
    ]
    done = next(e["item"] for e in wire if e["type"] == "response.output_item.done")
    final = wire[-1]["response"]
    assert done["status"] == final["output"][0]["status"] == "incomplete"
    assert wire[-1]["type"] == "response.incomplete"


@pytest.mark.parametrize("content", [42, [{"type": "reasoning_text", "text": 42}]])
def test_a_badly_shaped_reasoning_input_item_is_a_400(content):
    with pytest.raises(ApiError) as exc:
        responses.parse_responses_request(
            {
                "input": [
                    {"type": "reasoning", "content": content},
                    {"role": "assistant", "content": "w2"},
                ]
            },
            "m",
        )
    assert exc.value.status == 400 and exc.value.param == "input"


def test_responses_cut_tail_next_to_a_recovered_call_is_the_incomplete_item():
    """A block with a complete call and a cut-off second one yields the call
    as completed and the tail as a message that is incomplete - on the wire
    and in the final response, in that order."""
    tok = MistralLike()
    tok._words[35] = 'good[ARGS]{"x": 1}[TOOL_CALLS]bad[ARGS]{"x":'
    body = {
        "input": "w1",
        "stream": True,
        "tools": [{"type": "function", "name": n} for n in ("good", "bad")],
    }
    req = responses.parse_responses_request(body, "m")
    gen = responses.to_generation_request(tok, req)
    evs = [TokenEvent(TOOL_START, -0.1), TokenEvent(35, -0.1, "length")]
    wire = [
        json.loads(json.dumps(e))
        for e in responses.ResponsesResponder(tok, req, gen.tokens).stream(iter(evs), 0)
    ]
    items = [e["item"] for e in wire if e["type"] == "response.output_item.done"]
    assert [i["type"] for i in items] == ["function_call", "message"]
    assert items[0]["name"] == "good" and items[0]["status"] == "completed"
    assert items[1]["status"] == "incomplete"
    assert items[1]["content"][0]["text"] == '[TOOL_CALLS]bad[ARGS]{"x":'
    final = wire[-1]["response"]
    assert final["status"] == "incomplete"
    assert [i["status"] for i in final["output"]] == ["completed", "incomplete"]
    req.chat.stream = False
    whole = responses.ResponsesResponder(tok, req, gen.tokens).complete(iter(evs), 0)
    assert [i["status"] for i in whole["output"]] == ["completed", "incomplete"]


def test_a_multi_token_stop_sequence_leaves_no_prefix_behind():
    # The engine matches the sequence on token ids and ends on its last
    # token; the automaton held the start as a possible prefix. The last
    # token goes through the detokenizer so the word completes and is cut
    # - only an eos token is dropped unseen.
    tok = StubTokenizer()
    for stream in (False, True):
        req = chat.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "w1"}],
                "stop": "w20 w34",
                "stream": stream,
            },
            "m",
        )
        gen = chat.to_generation_request(tok, req)
        assert (20, 34) in gen.stop_sequences
        ev = [TokenEvent(20, -0.1), TokenEvent(34, -0.1, "stop")]
        responder = chat.ChatResponder(tok, req, gen.tokens)
        if stream:
            chunks = list(responder.stream(iter(ev), 0))
            text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        else:
            text = responder.complete(iter(ev), 0)["choices"][0]["message"]["content"]
        assert not text
    # A prefix before an eos is text the model said; so is one cut by length.
    for ev in (
        [TokenEvent(20, -0.1), TokenEvent(EOS, -0.1, "stop")],
        [TokenEvent(20, -0.1, "length")],
    ):
        asm = TextAssembler(tok, prompt_tokens=[6, 1, 7], stop_words=["w20 w34"])
        assert "".join(asm.feed(e).content for e in ev) == "w20 "


class _ByteTokenizer(StubTokenizer):
    """Three tokens carry the bytes of one character (E2 82 AC = the euro
    sign); the real byte-level detokenizer holds them until it completes."""

    has_thinking = False
    has_tool_calling = False

    def __init__(self):
        super().__init__()
        self.raw = [w.encode() for w in self._words]
        self.raw[40], self.raw[41], self.raw[42] = b"\xe2", b"\x82", b"\xac"

    def __len__(self):
        return len(self.raw)

    def convert_ids_to_tokens(self, ids):
        from mlx_beam._vendor.mlx_lm.tokenizer_utils import _byte_decoder

        to_char = {v: k for k, v in _byte_decoder().items()}
        return ["".join(to_char[b] for b in self.raw[t]) for t in ids]

    def decode(self, ids):
        return b"".join(self.raw[t] for t in ids).decode("utf-8", "replace")

    @property
    def detokenizer(self):
        from mlx_beam._vendor.mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

        return BPEStreamingDetokenizer(self)


@pytest.mark.parametrize("stream", [False, True])
def test_logprobs_keep_the_byte_tokens_of_a_character(stream):
    tok = _ByteTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "logprobs": True,
            "stream": stream,
        },
        "m",
    )
    ev = [
        TokenEvent(40, -0.1),
        TokenEvent(41, -0.2),
        TokenEvent(42, -0.3),
        TokenEvent(EOS, -0.4, "stop"),
    ]
    responder = chat.ChatResponder(tok, req, [1])
    if stream:
        chunks = list(responder.stream(iter(ev), 0))
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        entries = [e for c in chunks for e in c["choices"][0]["logprobs"]["content"]]
    else:
        choice = responder.complete(iter(ev), 0)["choices"][0]
        content = choice["message"]["content"]
        entries = choice["logprobs"]["content"]
    assert content == "€"
    assert [e["logprob"] for e in entries] == [-0.1, -0.2, -0.3]


def test_responses_take_a_null_tool_choice_as_auto():
    # Clients send the default explicitly as null; that is not a choice.
    req = responses.parse_responses_request({"input": "w1", "tool_choice": None}, "m")
    assert req.tool_choice == "auto"


def test_logprobs_book_a_held_prefix_with_the_token_that_releases_it():
    """A token the automaton holds back as the possible start of a stop
    word is content once the word does not come; the eos token itself is
    never content, even when its arrival flushes held text."""
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {
            "messages": [{"role": "user", "content": "w1"}],
            "logprobs": True,
            "stop": "w10 w11",
        },
        "m",
    )
    responder = chat.ChatResponder(tok, req, [1])
    # w10 is held (prefix of the stop word), w12 releases both.
    ev = [TokenEvent(10, -0.1), TokenEvent(12, -0.2), TokenEvent(EOS, -0.3, "stop")]
    choice = responder.complete(iter(ev), 0)["choices"][0]
    assert choice["message"]["content"] == "w10 w12 "
    assert [e["logprob"] for e in choice["logprobs"]["content"]] == [-0.1, -0.2]
    # Held at the end: the eos flushes w10 as content, but is none itself.
    responder = chat.ChatResponder(tok, req, [1])
    ev = [TokenEvent(10, -0.1), TokenEvent(EOS, -0.3, "stop")]
    choice = responder.complete(iter(ev), 0)["choices"][0]
    assert choice["message"]["content"] == "w10 "
    assert [e["logprob"] for e in choice["logprobs"]["content"]] == [-0.1]


def test_completions_do_not_repeat_the_opener_the_prompt_ends_with():
    # The raw prompt already carries `<think>`; the completion is what the
    # model wrote after it, not the opener again.
    tok = StubTokenizer()
    req = completions.parse_completion_request(
        {"prompt": [6, 1, 7, THINK_START], "max_tokens": 4}, "m"
    )
    responder = completions.CompletionResponder(tok, req, req.prompt)
    out = responder.complete(events([10, THINK_END, 11]), 0)
    assert out["choices"][0]["text"] == "w10 </think>w11 "


def test_a_marker_inside_a_user_message_does_not_open_the_assistant_state():
    """A `<think>` the user wrote (code, a question about the token) is
    text; only the assistant's own frame seeds the reasoning state."""
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w2 <think> w3"}]}, "m"
    )
    gen = chat.to_generation_request(tok, req)
    assert THINK_START in gen.tokens and req.assistant_start == len(gen.tokens) - 1
    assert gen.reasoning is not None and not gen.reasoning.seeded
    responder = chat.ChatResponder(tok, req, gen.tokens)
    out = responder.complete(events([10, 11]), 0)["choices"][0]["message"]
    assert out["content"] == "w10 w11 " and not out.get("reasoning")


def test_logprob_bytes_of_a_byte_level_token_are_its_own():
    tok = _ByteTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}], "logprobs": True}, "m"
    )
    ev = [
        TokenEvent(40, -0.1),
        TokenEvent(41, -0.2),
        TokenEvent(42, -0.3),
        TokenEvent(EOS, -0.4, "stop"),
    ]
    choice = chat.ChatResponder(tok, req, [1]).complete(iter(ev), 0)["choices"][0]
    assert [e["bytes"] for e in choice["logprobs"]["content"]] == [
        [0xE2],
        [0x82],
        [0xAC],
    ]
