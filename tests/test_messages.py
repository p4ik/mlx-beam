"""Anthropic's Messages dialect on the stub tokenizer: request blocks to
messages, tools both ways, blocks in order, the stream's events, the
error envelope, count_tokens, cache markers at boundaries only."""

import base64
import json

import pytest

from mlx_beam.api import messages
from mlx_beam.api.errors import ApiError
from mlx_beam.engine.request import TokenEvent
from tests.stub_tokenizer import (
    THINK_END,
    THINK_START,
    TOOL_END,
    TOOL_START,
    StubTokenizer,
)
from tests.test_api import events
from tests.test_server import call, server  # noqa: F401 - fixture

TOOLS = [
    {
        "name": "lookup",
        "description": "look something up",
        "input_schema": {"type": "object", "properties": {"words": {"type": "string"}}},
    },
    # What the stub parser names a call after its first word.
    {"name": "w20", "input_schema": {"type": "object"}},
]


def test_request_blocks_become_messages_and_tools_map():
    req = messages.parse_messages_request(
        {
            "model": "m",
            "max_tokens": 5,
            "system": [
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [
                {"role": "user", "content": "w1"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "x"},
                        {"type": "text", "text": "w2"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"words": "w3"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "w4"}],
                        },
                        {
                            "type": "text",
                            "text": "w5",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
            ],
            "tools": TOOLS,
            "tool_choice": {"type": "auto"},
            "stop_sequences": ["w9"],
            "temperature": 0.5,
            "top_k": 5,
            "thinking": {"type": "enabled", "budget_tokens": 32},
            "metadata": {"user_id": "u"},
        },
        "served",
    )
    c = req.chat
    assert c.model == "m" and c.max_tokens == 5 and c.stop == ["w9"]
    assert c.sampling.temperature == 0.5 and c.sampling.top_k == 5
    assert c.max_reasoning_tokens == 32 and c.template_kwargs["enable_thinking"] is True
    assert [m["role"] for m in c.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert c.messages[0]["content"] == "sys"
    assert (
        c.messages[2]["reasoning_content"] == "hmm" and c.messages[2]["content"] == "w2"
    )
    assert c.messages[2]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": {"words": "w3"},
    }
    assert c.messages[3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "w4"}
    assert c.messages[4]["content"] == "w5"
    assert c.tools[0]["function"]["parameters"] == TOOLS[0]["input_schema"]
    # cache_control and metadata are accepted and ignored: the store
    # caches at every boundary by itself, nothing is keyed on a client's id.


def test_refusals_are_explicit():
    base = {"max_tokens": 4, "messages": [{"role": "user", "content": "w1"}]}
    # cache_control on any block is fine (Claude Code marks blocks inside
    # a message): accepted, not a refusal.
    req = messages.parse_messages_request(
        {
            **base,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "a",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
        },
        "m",
    )
    assert req.chat.messages[-1]["content"] == "ab"
    with pytest.raises(ApiError) as exc:
        messages.parse_messages_request({**base, "tool_choice": {"type": "any"}}, "m")
    assert exc.value.code == "unsupported"
    with pytest.raises(ApiError) as exc:
        messages.parse_messages_request({**base, "messages": [IMAGE_TURN]}, "m")
    assert exc.value.code == "extra_not_installed"
    with pytest.raises(ApiError, match="source.type"):
        messages.parse_messages_request(
            {
                **base,
                "messages": [
                    {"role": "user", "content": [{"type": "image", "source": {}}]}
                ],
            },
            "m",
            vision=True,
        )
    with pytest.raises(ApiError):
        messages.parse_messages_request(
            {**base, "messages": [{"role": "system", "content": "x"}]}, "m"
        )
    with pytest.raises(ApiError):
        messages.parse_messages_request(
            {**base, "thinking": {"type": "sometimes"}}, "m"
        )
    req = messages.parse_messages_request(
        {**base, "thinking": {"type": "disabled"}}, "m"
    )
    assert req.chat.template_kwargs["enable_thinking"] is False
    req = messages.parse_messages_request(
        {**base, "tools": TOOLS, "tool_choice": {"type": "none"}}, "m"
    )
    assert req.chat.tools is None


IMAGE_TURN = {
    "role": "user",
    "content": [
        {"type": "text", "text": "w1"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(b"\x89PNG" * 8).decode(),
            },
        },
        {"type": "text", "text": "w3"},
    ],
}


def test_image_blocks_reach_the_frontend_like_chat_parts():
    """An Anthropic image block is the same image the chat path takes: a
    base64 source decoded with a frontend, the turn's parts kept in block
    order; a URL source is refused since nothing is fetched. The token
    count sees the expanded prompt."""
    from tests.test_vision_core import FakeFrontend

    body = {"max_tokens": 4, "messages": [IMAGE_TURN]}
    req = messages.parse_messages_request(body, "m", vision=True)
    assert len(req.chat.images) == 1
    assert req.chat.images[0].data == b"\x89PNG" * 8
    assert req.chat.images[0].media_type == "image/png"
    assert req.chat.messages[-1]["content"] == [
        {"type": "text", "text": "w1"},
        {"type": "image"},
        {"type": "text", "text": "w3"},
    ]
    tok = StubTokenizer()
    front = FakeFrontend(tok)
    gen = messages.to_generation_request(tok, req, frontend=front)
    assert len(gen.spans) == 1
    counted = messages.count_tokens(tok, req, front)
    assert counted["input_tokens"] == len(gen.tokens)
    remote = {
        "role": "user",
        "content": [{"type": "image", "source": {"type": "url", "url": "https://x/y"}}],
    }
    with pytest.raises(ApiError, match="only base64"):
        messages.parse_messages_request(
            {"max_tokens": 4, "messages": [remote]}, "m", vision=True
        )


def run(ids, finish="stop", tools=None):
    tok = StubTokenizer()
    body = {"max_tokens": 16, "messages": [{"role": "user", "content": "w1"}]}
    if tools:
        body["tools"] = tools
    req = messages.parse_messages_request(body, "m")
    gen = messages.to_generation_request(tok, req)
    responder = messages.MessagesResponder(tok, req, gen.tokens)
    evs = list(responder.stream(events(ids, finish), cached=3))
    return evs, responder.final


def test_blocks_in_order_and_the_stream_events():
    ids = [THINK_START, 10, 11, THINK_END, 12, TOOL_START, 20, 21, TOOL_END]
    evs, final = run(ids, tools=TOOLS)
    types = [e["type"] for e in evs]
    assert types[:2] == ["message_start", "ping"] and types[-2:] == [
        "message_delta",
        "message_stop",
    ]
    starts = [
        e["content_block"]["type"] for e in evs if e["type"] == "content_block_start"
    ]
    assert starts == ["thinking", "text", "tool_use"]
    assert [e["index"] for e in evs if e["type"] == "content_block_stop"] == [0, 1, 2]
    deltas = [e["delta"] for e in evs if e["type"] == "content_block_delta"]
    assert deltas[0] == {"type": "thinking_delta", "thinking": "w10 "}
    assert deltas[2] == {"type": "text_delta", "text": "w12 "}
    assert deltas[-1]["type"] == "input_json_delta"
    assert json.loads(deltas[-1]["partial_json"]) == {"words": "w21"}
    assert evs[-2]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert evs[-2]["usage"] == {"output_tokens": 10}
    assert final["type"] == "message" and final["role"] == "assistant"
    assert [b["type"] for b in final["content"]] == ["thinking", "text", "tool_use"]
    assert final["content"][0]["thinking"] == "w10 w11 "
    assert final["content"][2]["name"] == "w20" and final["content"][2]["input"] == {
        "words": "w21"
    }
    assert final["stop_reason"] == "tool_use"
    prompt = messages.to_generation_request(
        StubTokenizer(),
        messages.parse_messages_request(
            {
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "w1"}],
                "tools": TOOLS,
            },
            "m",
        ),
    ).tokens
    # Anthropic's input_tokens exclude what the cache served.
    assert final["usage"] == {
        "input_tokens": len(prompt) - 3,
        "output_tokens": 10,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    }
    evs, final = run([10, 11], finish="length")
    assert final["stop_reason"] == "max_tokens" and final["content"] == [
        {"type": "text", "text": "w10 w11 "}
    ]
    evs, final = run([10])
    assert final["stop_reason"] == "end_turn"


def test_endpoints_and_the_error_envelope(server):  # noqa: F811
    body = {"max_tokens": 3, "messages": [{"role": "user", "content": "w1"}]}
    status, _, raw = call(server, "POST", "/v1/messages", body)
    out = json.loads(raw)
    assert status == 200 and out["type"] == "message" and out["content"]
    assert out["stop_reason"] in ("end_turn", "max_tokens")
    status, ctype, raw = call(server, "POST", "/v1/messages", {**body, "stream": True})
    assert status == 200 and ctype.startswith("text/event-stream")
    text = raw.decode()
    assert "event: message_start" in text and "event: message_stop" in text
    assert '"message"' not in text.split("event: message_stop")[1]
    status, _, raw = call(server, "POST", "/v1/messages/count_tokens", body)
    counted = json.loads(raw)
    assert status == 200 and counted == {"input_tokens": out["usage"]["input_tokens"]}
    status, _, raw = call(
        server, "POST", "/v1/messages", {"max_tokens": 3, "messages": []}
    )
    err = json.loads(raw)
    assert status == 400 and err == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": err["error"]["message"]},
    }
    status, _, raw = call(server, "POST", "/v1/messages", {**body, "model": "other"})
    assert status == 404 and json.loads(raw)["error"]["type"] == "not_found_error"


def test_stop_sequences_end_the_text_and_name_the_sequence():
    tok = StubTokenizer()
    req = messages.parse_messages_request(
        {
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "w1"}],
            "stop_sequences": ["w11"],
        },
        "m",
    )
    assert req.chat.stop == ["w11"]
    responder = messages.MessagesResponder(tok, req, [1])
    ev = [TokenEvent(10, -0.1), TokenEvent(11, -0.1), TokenEvent(12, -0.1)]
    final = responder.complete(iter(ev), 0)
    assert final["stop_reason"] == "stop_sequence" and final["stop_sequence"] == "w11"
    assert final["content"] == [{"type": "text", "text": "w10 "}]
