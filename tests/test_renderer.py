"""The renderer decision for a checkpoint, and the engine on a renderer
without template text: mistral-common (transformers' MistralCommonBackend)
hands ids back, names no template and takes the tool-call parser from the
vocabulary. A stand-in with that shape runs the real tokenizer wrapper and
the real prompt path. The real backend runs here too, over the tekken
vocabulary mistral-common ships, for what only it can show: the
detokenizer the loader picks for it, the bytes of its tokens, and the
tool-call ids its validator takes."""

import hashlib
import importlib.resources
import json
import re
import shutil
import string

import pytest
from transformers.tokenization_mistral_common import (
    MistralCommonBackend as RealMistralCommonBackend,
)

from mlx_beam import renderer
from mlx_beam._vendor.mlx_lm import tokenizer_utils
from mlx_beam._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_beam._vendor.mlx_lm.tool_parsers import mistral
from mlx_beam.api import chat, messages, responses, roles
from mlx_beam.api.defaults import RequestDefaults
from mlx_beam.api.errors import ApiError
from mlx_beam.api.text import TextAssembler, holds_incomplete_bytes, logprob_entry
from mlx_beam.engine import Engine
from mlx_beam.engine.request import TokenEvent
from mlx_beam.server import Served
from tests.test_vendor_optiq_kv import tiny_llama

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    }
]


class MistralCommonBackend:
    """What the engine reads of transformers' MistralCommonBackend: no
    template text, a vocabulary with the control tokens, and
    `apply_chat_template` that renders ids (a dict with the pixel values
    when the messages carry images), taking any keyword without complaint.
    The class name is what the vision package detects."""

    chat_template = None
    eos_token_id = 2
    unk_token_id = 0
    SPECIAL = {"<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4}
    ROLES = {"system": 17, "user": 3, "assistant": 5, "tool": 7}

    def __init__(self, tools_trained=True):
        self.vocab = dict(self.SPECIAL)
        if tools_trained:
            self.vocab["[TOOL_CALLS]"] = 9
        self.init_kwargs = {}
        self.calls = []

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        # Control tokens written as text are plain words to mistral-common.
        return [100 + (ord(c) % 50) for c in text]

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        return_dict=True,
        **kwargs,
    ):
        self.calls.append({"tools": tools, "kwargs": kwargs})
        ids = [1]
        if tools:
            ids.append(11)
        for m in conversation:
            if m["role"] not in self.ROLES:
                raise ValueError(f"unknown role {m['role']}")
            ids.append(self.ROLES[m["role"]])
            content = m.get("content")
            if isinstance(content, str):
                ids += self.encode(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        ids += self.encode(part["text"])
                    elif part.get("type") == "image_url":
                        ids += [10, 10, 12, 10, 10, 13]
            for call in m.get("tool_calls") or ():
                ids += [9] + self.encode(call["function"]["name"])
        if not tokenize:
            return self.decode(ids)
        if not return_dict:
            return ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def wrapped(tools_trained=True):
    return TokenizerWrapper(MistralCommonBackend(tools_trained))


def test_choose_takes_mistral_common_for_a_tekken_checkpoint(tmp_path):
    """Without tekken.json the template path; with it mistral-common -
    also when a conversion added a template beside it (the note says the
    template is left aside). A template on the command line keeps the
    tokenizer on the template path whatever the checkpoint ships."""
    assert renderer.choose(tmp_path).name == "jinja"
    (tmp_path / "tekken.json").write_text("{}")
    chosen = renderer.choose(tmp_path)
    assert chosen.name == "mistral-common"
    assert chosen.tokenizer_kwargs == {"mistral_format": True} and chosen.note is None
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    assert "left aside" in renderer.choose(tmp_path).note
    assert renderer.choose(tmp_path, "{{ flag }}").tokenizer_kwargs == {
        "mistral_format": False
    }


def test_the_tool_parser_comes_from_the_vocabulary():
    """No template names the parser, so the `[TOOL_CALLS]` token does:
    its id straight from the vocabulary (encoded as text it would be
    words), mlx-lm's mistral parser behind it; a vocabulary without the
    token gets none."""
    tok = wrapped()
    assert not tok.has_tool_calling
    assert renderer.install_tool_parser(tok) == "mistral"
    assert tok.has_tool_calling and tok.tool_parser_type == "mistral"
    assert tok.tool_call_start == "[TOOL_CALLS]" and tok.tool_call_start_tokens == (9,)
    assert tok.tool_parser('calc[ARGS]{"expr": "2+2"}', TOOLS) == {
        "name": "calc",
        "arguments": {"expr": "2+2"},
    }
    assert renderer.install_tool_parser(tok) == "mistral"  # idempotent
    assert renderer.install_tool_parser(wrapped(tools_trained=False)) is None


def test_the_prompt_path_runs_on_ids_alone():
    """Through the real wrapper and the real chat path: the wrapper's
    template probes find no text and settle (no thinking, no effort
    kwarg, developer rendered as system), the prompt comes back as ids
    with the tools passed through, and a tool-call turn renders."""
    tok = wrapped()
    renderer.install_tool_parser(tok)
    assert tok.has_chat_template and not tok.has_thinking
    assert roles.describe(tok) == {"developer": "as system"}
    defaults = RequestDefaults.resolve(None, flags={})
    body = {
        "model": "m",
        "messages": [
            {"role": "developer", "content": "brief"},
            {"role": "user", "content": "2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "abcdefghi",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "abcdefghi", "content": "4"},
        ],
        "tools": TOOLS,
        "reasoning_effort": "low",
    }
    req = chat.parse_chat_request(body, "m", defaults)
    ids = chat.build_prompt(tok, req)
    backend = tok._tokenizer
    assert isinstance(ids, list) and ids[:2] == [1, 11] and 17 in ids and 9 in ids
    assert backend.calls[-1]["tools"] == TOOLS


# What mistral-common's validators take as a tool-call id (v3 to v11).
VALID_ID = re.compile(r"^[a-zA-Z0-9]{9}$")
BASE62 = string.digits + string.ascii_uppercase + string.ascii_lowercase
TEXT = "Grün heißt Küstenökosystem, Čeština, ę, ā 🌊"
CALL = 'calc[ARGS]{"expr": "2+2"}'


@pytest.fixture(scope="module")
def tekken(tmp_path_factory):
    """The real backend over the v3 tekken vocabulary mistral-common ships,
    loaded the way the engine loads a checkpoint: beside a tokenizer.json
    with a ByteLevel decoder, as the published conversions carry it (what
    the loader read the detokenizer off)."""
    root = tmp_path_factory.mktemp("tekken")
    data = importlib.resources.files("mistral_common") / "data" / "tekken_240911.json"
    with importlib.resources.as_file(data) as src:
        shutil.copy(src, root / "tekken.json")
    (root / "tokenizer.json").write_text(json.dumps({"decoder": {"type": "ByteLevel"}}))
    tok = tokenizer_utils.load(root, {"mistral_format": True})
    renderer.install_tool_parser(tok)
    return tok


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def mapped_id(original):
    """The substitute for an id the validator refuses: base62 of its
    SHA-256, most significant digit first, cut to nine."""
    n = int.from_bytes(hashlib.sha256(original.encode()).digest(), "big")
    digits = ""
    while n:
        n, r = divmod(n, 62)
        digits = BASE62[r] + digits
    return digits[:9]


def tool_turn(*ids, results=None):
    """An assistant turn calling `calc` once per id, and a result per id
    (the result ids default to the call ids)."""
    calls = [
        {
            "id": i,
            "type": "function",
            "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
        }
        for i in ids
    ]
    turn = [{"role": "assistant", "content": None, "tool_calls": calls}]
    for i in ids if results is None else results:
        turn.append({"role": "tool", "tool_call_id": i, "content": "4"})
    return turn


def assemble(tok, ids):
    """The ids through the assembler as a stream that ends on eos: the
    text and the tool calls it released."""
    asm = TextAssembler(tok, prompt_tokens=[], tools=TOOLS)
    deltas = [asm.feed(TokenEvent(t, -0.1)) for t in ids]
    deltas.append(asm.feed(TokenEvent(tok.eos_token_id, -0.1, "stop")))
    text = "".join(d.content for d in deltas)
    calls = [c for d in deltas for c in d.tool_calls]
    return text, calls, deltas[-1].finish_reason


def test_the_loader_picks_the_byte_detokenizer_for_tekken(tekken):
    """A tekken checkpoint's text comes from the backend's own bytes, not
    from the tokenizer.json beside it (whose ByteLevel decoder would pick
    the BPE detokenizer, built for another tokenizer's alphabet)."""
    assert isinstance(tekken._tokenizer, RealMistralCommonBackend)
    # By name: a loader without the class picks BPE, and the name says so.
    assert type(tekken._detokenizer).__name__ == "BytePieceStreamingDetokenizer"


def test_tekken_text_round_trips(tekken):
    """Token by token, as the assembler reads it (`last_segment` after
    every token and after `finalize`): the segments and `text` are the
    original - no U+FFFD, no control bytes, no trimmed space."""
    ids = encode(tekken, TEXT)
    detok = tekken.detokenizer
    segments = []
    for t in ids:
        detok.add_token(t)
        segments.append(detok.last_segment)
    detok.finalize()
    segments.append(detok.last_segment)
    assert "".join(segments) == TEXT
    assert detok.text == TEXT and detok.tokens == ids


def test_finalize_never_raises(tekken):
    """A stream cut inside a character (the budget, a stop) is flushed as
    U+FFFD, not as an exception on the way out."""
    ids = encode(tekken, "🌊")
    assert len(ids) >= 2  # the emoji is spread over tokens
    detok = tekken.detokenizer
    detok.add_token(ids[0])
    detok.finalize()
    assert detok.text == "�"


def test_a_cut_sequence_is_held_and_an_invalid_byte_is_not(tekken):
    """A character still forming is held back (the fragment check reads
    the bytes); a byte no sequence can start comes out as U+FFFD at once,
    not after the next token."""
    ids = encode(tekken, "🌊")
    assert len(ids) >= 2
    detok = tekken.detokenizer
    detok.add_token(ids[0])
    assert holds_incomplete_bytes(detok)
    assert "�" not in detok.text
    for t in ids[1:]:
        detok.add_token(t)
    assert detok.last_segment == "🌊" and not holds_incomplete_bytes(detok)
    byte_piece = tekken._detokenizer.byte_piece
    invalid = next(i for i in range(len(tekken)) if byte_piece(i) == b"\xff")
    detok.add_token(invalid)
    assert detok.last_segment == "�" and not holds_incomplete_bytes(detok)
    (a,) = encode(tekken, "a")
    detok.add_token(a)
    assert detok.text == "🌊�a"


def test_tool_calls_marker_reaches_the_parser(tekken):
    """`[TOOL_CALLS]` is a special token: it comes out as its text, so the
    automaton opens the tool block, and the whole call is parsed."""
    marker = tekken.convert_tokens_to_ids("[TOOL_CALLS]")
    detok = tekken.detokenizer
    detok.add_token(marker)
    assert detok.last_segment == "[TOOL_CALLS]"
    text, calls, finish = assemble(tekken, [marker, *encode(tekken, CALL)])
    assert text == "" and finish == "tool_calls"
    assert [c["function"]["name"] for c in calls] == ["calc"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"expr": "2+2"}


def test_mistral_tool_ids_render_on_the_next_turn(tekken):
    """A call without an id of its own gets one the validator takes; an
    id a client sends in another form is mapped when the prompt renders,
    the same for the call and its result - on every entrance - and the
    boundary of the turn after it is found."""
    marker = tekken.convert_tokens_to_ids("[TOOL_CALLS]")
    _, calls, _ = assemble(tekken, [marker, *encode(tekken, CALL)])
    fresh = calls[0]["id"]
    assert VALID_ID.match(fresh)
    # Sent back as the call's and its result's id, it renders as it is.
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "2+2?"}, *tool_turn(fresh)],
        "tools": TOOLS,
    }
    req = chat.parse_chat_request(body, "m")
    assert chat.build_prompt(tekken, req)
    assert req.messages[1]["tool_calls"][0]["id"] == fresh
    assert req.messages[2]["tool_call_id"] == fresh

    foreign = "call_abc123def456abc123def456"
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "2+2?"}, *tool_turn(foreign)],
        "tools": TOOLS,
    }
    req = chat.parse_chat_request(body, "m")
    assert chat.build_prompt(tekken, req)
    call_id = req.messages[1]["tool_calls"][0]["id"]
    assert VALID_ID.match(call_id) and call_id == mapped_id(foreign)
    assert req.messages[2]["tool_call_id"] == call_id

    resp = responses.parse_responses_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "2+2?"},
                {
                    "type": "function_call",
                    "call_id": foreign,
                    "name": "calc",
                    "arguments": '{"expr": "2+2"}',
                },
                {"type": "function_call_output", "call_id": foreign, "output": "4"},
            ],
            "tools": TOOLS,
        },
        "m",
    )
    assert chat.build_prompt(tekken, resp.chat)
    assert resp.chat.messages[1]["tool_calls"][0]["id"] == call_id
    assert resp.chat.messages[2]["tool_call_id"] == call_id

    msg = messages.parse_messages_request(
        {
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "2+2?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "calc",
                            "input": {"expr": "2+2"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "4",
                        }
                    ],
                },
            ],
            "tools": [
                {"name": "calc", "input_schema": TOOLS[0]["function"]["parameters"]}
            ],
        },
        "m",
    )
    assert chat.build_prompt(tekken, msg.chat)
    assert msg.chat.messages[1]["tool_calls"][0]["id"] == mapped_id("toolu_1")
    assert msg.chat.messages[2]["tool_call_id"] == mapped_id("toolu_1")

    # The turn after the tool turn is re-rendered as a prefix for its
    # boundary; that prefix holds the tool turn, so it renders only with
    # the mapped ids on the request itself. This tokenizer version puts
    # the system prompt and the tools at the last user message, so the
    # prefix and the prompt part right after the tool results.
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "brief"},
            {"role": "user", "content": "2+2?"},
            *tool_turn(foreign),
            {"role": "user", "content": "and 3+3?"},
            {"role": "assistant", "content": "6"},
            {"role": "user", "content": "thanks"},
        ],
        "tools": TOOLS,
    }
    gen = chat.to_generation_request(tekken, chat.parse_chat_request(body, "m"))
    end = tekken.convert_tokens_to_ids("[/TOOL_RESULTS]")
    assert gen.tokens.index(end) + 1 in gen.boundaries


def test_mapped_tool_ids_stay_distinct_within_a_turn(tekken):
    """Within a turn no two ids fall together: parallel calls keep apart,
    a valid id keeps its place and a foreign id that would map onto it
    takes the next substitute, a later turn changes nothing earlier, a
    result whose id maps onto a call it never answered stays unpaired, and
    a result answering no call of the turn is mapped whatever its form."""

    def render(turns):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "2+2?"}, *turns],
            "tools": TOOLS,
        }
        req = chat.parse_chat_request(body, "m")
        assert chat.build_prompt(tekken, req)
        return req.messages

    out = render(tool_turn("call_a", "call_b"))
    ids = [c["id"] for c in out[1]["tool_calls"]]
    assert len(set(ids)) == 2 and all(VALID_ID.match(i) for i in ids)
    assert [m["tool_call_id"] for m in out[2:4]] == ids

    valid = mapped_id("call_abc")
    assert VALID_ID.match(valid)
    out = render(tool_turn(valid, "call_abc"))
    ids = [c["id"] for c in out[1]["tool_calls"]]
    assert ids == [valid, mapped_id("1:call_abc")]
    assert [m["tool_call_id"] for m in out[2:4]] == ids

    longer = render(
        [
            *tool_turn(valid, "call_abc"),
            {"role": "user", "content": "again"},
            *tool_turn("call_abc"),
        ]
    )
    assert [c["id"] for c in longer[1]["tool_calls"]] == ids
    assert [m["tool_call_id"] for m in longer[2:4]] == ids
    # The next turn has its own table: the first substitute is free there.
    assert longer[5]["tool_calls"][0]["id"] == valid
    assert longer[6]["tool_call_id"] == valid

    out = render(tool_turn(valid, results=["call_abc"]))
    assert out[1]["tool_calls"][0]["id"] == valid
    assert out[2]["tool_call_id"] == mapped_id("1:call_abc")

    # A valid id on a result that answers no call of the turn: as sent, it
    # could name the substitute of a call it never answered (the validator's
    # error messages spell the mapped ids out), so it is mapped too.
    out = render(tool_turn("call_abc", results=[valid]))
    assert out[1]["tool_calls"][0]["id"] == valid
    assert out[2]["tool_call_id"] == mapped_id(valid) != valid
    out = render(tool_turn(valid, results=["abcdefghi"]))
    assert out[1]["tool_calls"][0]["id"] == valid
    assert out[2]["tool_call_id"] == mapped_id("abcdefghi")

    resp = responses.parse_responses_request(
        {
            "input": [
                {"type": "message", "role": "user", "content": "2+2?"},
                {
                    "type": "function_call",
                    "call_id": valid,
                    "name": "calc",
                    "arguments": '{"expr": "2+2"}',
                },
                {"type": "function_call_output", "call_id": "call_abc", "output": "4"},
            ],
            "tools": TOOLS,
        },
        "m",
    )
    assert chat.build_prompt(tekken, resp.chat)
    assert resp.chat.messages[1]["tool_calls"][0]["id"] == valid
    assert resp.chat.messages[2]["tool_call_id"] == mapped_id("1:call_abc")


def test_the_mistral_parser_reads_a_call_id():
    """The header may carry the id the model wrote; it is taken as is,
    an empty one is none, and a header without one parses as before."""
    assert mistral.parse_tool_call('calc[CALL_ID]abcdefghi[ARGS]{"expr": "1"}') == {
        "name": "calc",
        "arguments": {"expr": "1"},
        "id": "abcdefghi",
    }
    assert (
        mistral.parse_tool_call("calc[CALL_ID] call_x-1 [ARGS]{}")["id"] == "call_x-1"
    )
    assert "id" not in mistral.parse_tool_call("calc[CALL_ID][ARGS]{}")
    assert mistral.parse_tool_call("calc[ARGS]{}") == {"name": "calc", "arguments": {}}


def test_logprob_bytes_are_the_tekken_bytes(tekken):
    """A token that is part of a character reports the token's own bytes
    (not those of the U+FFFD its text decodes to), as a list the response
    can serialize - in `top_logprobs` too."""
    ids = encode(tekken, "🌊")
    assert len(ids) >= 2
    entries = [
        logprob_entry(
            tekken, TokenEvent(t, -0.1, top_logprobs=((t, -0.1), (ids[0], -2.0)))
        )
        for t in ids
    ]
    for entry in entries:
        assert all(isinstance(b, int) for b in entry["bytes"])
        assert entry["top_logprobs"][1]["bytes"] == entries[0]["bytes"]
        json.dumps(entry)
    assert b"".join(bytes(e["bytes"]) for e in entries) == "🌊".encode()
    assert entries[0]["bytes"] == list(tekken._detokenizer.byte_piece(ids[0]))


def test_health_names_the_detokenizer_in_use(tekken):
    """`/health.template.detokenizer` is the evidence that a tekken
    checkpoint decodes from its own byte pieces. The engine is not started:
    the health reads the loaded tokenizer, not the worker."""
    health = Served(Engine(tiny_llama()), tekken, "tiny").health()
    assert health["template"].get("detokenizer") == "BytePieceStreamingDetokenizer"


def test_a_call_id_that_is_no_string_is_the_validator_s_400(tekken):
    """A guard: an id sent as a list or an object is not mapped but left to
    mistral-common, whose validator refuses it - the client's 400, not a
    TypeError on the way there."""
    for bad in ([1, 2], {"x": 1}):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "2+2?"}, *tool_turn(bad)],
            "tools": TOOLS,
        }
        req = chat.parse_chat_request(body, "m")
        with pytest.raises(ApiError) as exc:
            chat.build_prompt(tekken, req)
        assert exc.value.status == 400
