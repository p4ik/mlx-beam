"""Which renderer turns a request's messages into the prompt: the
checkpoint's Jinja chat template through the tokenizer, or mistral-common
for a checkpoint that ships `tekken.json` instead of a template (every
official Mistral release). Decided from the files before the model
loads, since the choice is a tokenizer argument."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

JINJA = "jinja"
MISTRAL_COMMON = "mistral-common"

# What mistral-common's tool-call marker is called in every tekken
# vocabulary; its presence is what makes the checkpoint tool-capable.
TOOL_CALL_START = "[TOOL_CALLS]"


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
