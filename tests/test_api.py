"""The API layer without a model: parsing, prompt building, text assembly."""

import json

import pytest

from mlx_beam.api import chat, completions, responses
from mlx_beam.api.errors import ApiError
from mlx_beam.api.text import TextAssembler
from mlx_beam.engine.request import TokenEvent
from tests.stub_tokenizer import EOS, THINK_END, THINK_START, TOOL_END, TOOL_START
from tests.stub_tokenizer import StubTokenizer


def events(ids, finish="stop"):
    out = [TokenEvent(t, -0.1) for t in ids]
    if finish == "stop":
        out.append(TokenEvent(EOS, -0.1, "stop"))
    else:
        out[-1] = TokenEvent(ids[-1], -0.1, "length")
    return out


def test_chat_request_defaults_and_limits():
    req = chat.parse_chat_request({"messages": [{"role": "user", "content": "w1"}]}, "m")
    assert req.max_tokens == chat.DEFAULT_MAX_TOKENS and req.model == "m"
    assert req.sampling.temperature == 1.0 and req.sampling.top_p == 1.0
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
        ({"messages": [{"role": "user", "content": "x"}], "temperature": 3}, "temperature"),
        ({"messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
        ({"messages": [{"role": "user", "content": "x"}], "seed": 1}, "seed"),
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
        {"messages": [{"role": "system", "content": "w1"}, {"role": "user", "content": "w2 w3"}]},
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
    assert "".join(d.content for d in deltas) == "w10 " and deltas[-1].finish_reason == "stop"


def test_chat_responder_shapes():
    tok = StubTokenizer()
    req = chat.parse_chat_request(
        {"messages": [{"role": "user", "content": "w1"}], "stream_options": {"include_usage": True}},
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
    chunks = list(chat.ChatResponder(tok, req, gen.tokens).stream(events([10, 11]), cached=0))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "w10 "}
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and "usage" in chunks[-1]


def test_completions_accept_token_ids_and_echo():
    tok = StubTokenizer()
    req = completions.parse_completion_request({"prompt": [1, 2, 3], "echo": True, "max_tokens": 2}, "m")
    gen = completions.to_generation_request(tok, req)
    assert gen.tokens == [1, 2, 3]
    out = completions.CompletionResponder(tok, req, gen.tokens).complete(events([10, 11], "length"), 0)
    assert out["choices"][0]["text"] == "w1 w2 w3 w10 w11 "
    assert out["choices"][0]["finish_reason"] == "length"
    with pytest.raises(ApiError):
        completions.parse_completion_request({"prompt": [1, "x"]}, "m")


def test_responses_items_become_messages_and_back():
    tok = StubTokenizer()
    body = {
        "instructions": "w1",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "w2"}]},
            {"type": "function_call", "call_id": "c1", "name": "w20", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "w3"},
        ],
        "tools": [{"type": "function", "name": "w20", "parameters": {}}],
        "max_output_tokens": 5,
    }
    req = responses.parse_responses_request(body, "m")
    roles = [m["role"] for m in req.chat.messages]
    assert roles == ["system", "user", "assistant", "tool"]
    assert req.chat.tools == [{"type": "function", "function": {"name": "w20", "parameters": {}}}]
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
        responses.parse_responses_request({"input": "w1", "previous_response_id": "resp_x"}, "m")
    assert "full input" in exc.value.message
    with pytest.raises(ApiError) as exc:
        responses.parse_responses_request({"input": "w1", "tools": [{"type": "web_search"}]}, "m")
    assert exc.value.code == "unsupported"


def test_responses_stream_events():
    tok = StubTokenizer()
    req = responses.parse_responses_request({"input": "w1", "stream": True}, "m")
    gen = responses.to_generation_request(tok, req)
    evs = list(responses.ResponsesResponder(tok, req, gen.tokens).stream(events([10, 11]), 0))
    types = [e["type"] for e in evs]
    assert types[0] == "response.created" and types[-1] == "response.completed"
    assert types.count("response.output_text.delta") == 2
    assert evs[-1]["response"]["output"][0]["content"][0]["text"] == "w10 w11 "
