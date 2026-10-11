"""Which renderer turns a request's messages into the prompt: the
checkpoint's Jinja chat template through the tokenizer, or mistral-common
for a checkpoint that ships `tekken.json` instead of a template (every
official Mistral release). Decided from the files before the model
loads, since the choice is a tokenizer argument."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

JINJA = "jinja"
MISTRAL_COMMON = "mistral-common"

# What mistral-common's tool-call marker is called in every tekken
# vocabulary; its presence is what makes the checkpoint tool-capable.
TOOL_CALL_START = "[TOOL_CALLS]"

# The tool-call id mistral-common's validators take, in every tokenizer
# version that checks one (v3 to v11; v13 on want a non-empty id only).
TOOL_CALL_ID = re.compile(r"^[a-zA-Z0-9]{9}$")
_BASE62 = string.digits + string.ascii_uppercase + string.ascii_lowercase


@dataclass
class Renderer:
    name: str
    # Handed to the tokenizer at load (transformers' `mistral_format`).
    tokenizer_kwargs: dict = field(default_factory=dict)
    note: str | None = None


def ships_tekken(model_path: Path) -> bool:
    return (model_path / "tekken.json").is_file()


def ships_template(model_path: Path) -> bool:
    """A Jinja template the tokenizer would pick up: the file, or the
    field in tokenizer_config.json."""
    if (model_path / "chat_template.jinja").is_file():
        return True
    for name in ("chat_template.json", "tokenizer_config.json"):
        file = model_path / name
        if not file.is_file():
            continue
        try:
            if json.loads(file.read_text()).get("chat_template"):
                return True
        except (OSError, ValueError):
            continue
    return False


def choose(model_path: Path, template_flag: str | None = None) -> Renderer:
    """The renderer for this checkpoint. A template given on the command
    line always renders (and keeps the tokenizer on the Jinja path, since
    mistral-common takes no template). Otherwise a tekken checkpoint
    renders through mistral-common - also when a conversion added a Jinja
    template beside it, since the template is a reconstruction and the
    tokenizer the original."""
    if template_flag is not None or not ships_tekken(model_path):
        return Renderer(JINJA, {"mistral_format": False})
    note = None
    if ships_template(model_path):
        note = (
            "tekken.json and a chat template both present: rendering "
            "through mistral-common, the template is left aside"
        )
    return Renderer(MISTRAL_COMMON, {"mistral_format": True}, note)


def install_tool_parser(tokenizer) -> str | None:
    """mistral-common renders tool calls and the model answers with
    `[TOOL_CALLS]`; the parser for that is mlx-lm's `mistral` one, chosen
    from the vocabulary because there is no template to read it from. The
    marker's id comes from the vocabulary too: mistral-common encodes the
    marker's text as plain words. Returns the parser's name, or None when
    the vocabulary has no marker (the model was not trained for tools)."""
    if tokenizer.tool_parser_type is not None:
        return tokenizer.tool_parser_type
    marker = tokenizer.convert_tokens_to_ids(TOOL_CALL_START)
    if marker is None or marker == tokenizer.unk_token_id:
        return None
    module = importlib.import_module("mlx_beam._vendor.mlx_lm.tool_parsers.mistral")
    # The same fields mlx-lm's loader fills when a template names the parser.
    tokenizer._tool_parser = module.parse_tool_call
    tokenizer._tool_call_start = module.tool_call_start
    tokenizer._tool_call_end = module.tool_call_end
    tokenizer._tool_call_start_tokens = (int(marker),)
    tokenizer._tool_call_end_tokens = ()
    tokenizer._tool_parser_type = "mistral"
    return "mistral"


def is_mistral_common(tokenizer) -> bool:
    """Whether the prompt renders through mistral-common: the object behind
    the engine's tokenizer is transformers' MistralCommonBackend (the mark
    the vision package reads too). Its validator sets the id format."""
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    return type(inner).__name__ == "MistralCommonBackend"


def mistral_tool_call_id() -> str:
    """A fresh tool-call id in the form mistral-common's validators take."""
    return "".join(random.choices(_BASE62, k=9))


def mistral_tool_call_ids(messages: list[dict]) -> None:
    """Rewrite, in place, the tool-call ids mistral-common's validator would
    refuse into the nine-character form. Per turn - an assistant message's
    calls and the tool results up to the next assistant message - a call's
    valid id stays and every other non-empty id gets a substitute: a call's
    in another form, and a result's that answers no call of the turn,
    whatever its form (as sent, it could name the substitute of a call it
    never answered); the same one for a call and its result. None and ""
    stay for mistral-common to report. Per turn, since that is where the
    validator checks for duplicates, and so a later turn never changes an
    earlier one. Run once per request, when the prompt renders: a second
    run would give a result that answers no call a new substitute."""
    turn: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant" and turn:
            _map_turn(turn)
            turn = []
        turn.append(m)
    _map_turn(turn)


def _needs_mapping(value) -> bool:
    return isinstance(value, str) and bool(value) and not TOOL_CALL_ID.match(value)


def _digest_id(original: str) -> str:
    # Base62 of SHA-256, most significant digit first, cut to nine.
    n = int.from_bytes(hashlib.sha256(original.encode()).digest(), "big")
    digits = ""
    while n:
        n, r = divmod(n, 62)
        digits = _BASE62[r] + digits
    return digits[:9]


def _map_turn(turn: list[dict]) -> None:
    calls = [
        c
        for m in turn
        if m.get("role") == "assistant"
        for c in m.get("tool_calls") or ()
    ]
    results = [m for m in turn if m.get("role") == "tool"]
    # String ids only: anything else is left for the validator to refuse.
    sent = {c["id"] for c in calls if isinstance(c.get("id"), str)}
    taken = {
        c["id"]
        for c in calls
        if isinstance(c.get("id"), str) and TOOL_CALL_ID.match(c["id"])
    }
    mapped: dict[str, str] = {}

    def substitute(original: str) -> str:
        if original not in mapped:
            candidate, n = _digest_id(original), 0
            # An id another call of the turn holds would pair a result with
            # a call it never answered (the client's pairing is what counts).
            while candidate in taken:
                n += 1
                candidate = _digest_id(f"{n}:{original}")
            taken.add(candidate)
            mapped[original] = candidate
        return mapped[original]

    for c in calls:
        if _needs_mapping(c.get("id")):
            c["id"] = substitute(c["id"])
    for m in results:
        rid = m.get("tool_call_id")
        # A result answering no call of the turn is mapped whatever its form.
        if _needs_mapping(rid) or (isinstance(rid, str) and rid and rid not in sent):
            m["tool_call_id"] = substitute(rid)
