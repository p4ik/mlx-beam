"""What a request says about thinking, and what the chat template can take.

Three things live here: the aliases clients use for the same limits, the
effort levels mapped onto what a template accepts, and the mirroring of a
conversation's earlier reasoning into the message keys the renderer reads.
The engine never invents reasoning; it only copies what the client sent
into a key the client left empty.
"""

from __future__ import annotations

import inspect
import re
from functools import lru_cache

from mlx_beam.api.errors import ApiError

# reasoning_effort, as the OpenAI API and its imitators send it, mapped onto
# the three levels the templates we serve distinguish. "none" is thinking off.
EFFORT_TABLE = {
    "none": None,
    "false": None,
    "off": None,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
    "ultra": "xhigh",
}

# When a template rejects a level, the next rung is tried; a template that
# knows no levels ignores the kwarg and never gets here.
EFFORT_LADDER = {
    "low": ("low", "minimal", "medium"),
    "medium": ("medium", "moderate", "high", "low"),
    "xhigh": ("xhigh", "max", "high", "medium"),
}

# The keys earlier assistant turns carry their thinking in, most specific
# first: that order decides which value fills a key the client left empty.
REASONING_KEYS = ("reasoning_content", "thinking", "reasoning")

# message.thinking, message["thinking"], msg.get("thinking"): an access, not
# the word in prose (an ellipsis before it is prose).
_KEY_READ = re.compile(
    r"""(?:(?<!\.)\.|\[\s*['"]|get\(\s*['"])(reasoning_content|thinking|reasoning)(?![\w])"""
)


def map_effort(effort) -> str | None:
    """The template level for a request's reasoning_effort; None for off."""
    if effort is None:
        return None
    if isinstance(effort, bool):
        return "xhigh" if effort else None
    key = str(effort).lower()
    if key not in EFFORT_TABLE:
        raise ApiError(
            f"unknown reasoning_effort {effort!r}; one of {', '.join(EFFORT_TABLE)}",
            param="reasoning_effort",
        )
    return EFFORT_TABLE[key]


def is_effort_rejection(error: BaseException) -> bool:
    text = str(error).lower()
    return "reasoning effort" in text or "reasoning_effort" in text


def effort_candidates(level: str) -> tuple[str, ...]:
    return EFFORT_LADDER.get(level, (level,))


def read_aliases(body: dict) -> dict:
    """The four request aliases, resolved to our names. An explicit name wins
    over its alias, so a client sending both gets what it wrote out."""
    out: dict = {}
    reasoning = body.get("reasoning")
    reasoning = reasoning if isinstance(reasoning, dict) else {}
    if body.get("reasoning_effort") is not None:
        out["reasoning_effort"] = body["reasoning_effort"]
    elif reasoning.get("effort") is not None:
        out["reasoning_effort"] = reasoning["effort"]
    for key in ("max_reasoning_tokens", "thinking_token_budget"):
        if body.get(key) is not None:
            out["max_reasoning_tokens"] = body[key]
            break
    else:
        if reasoning.get("max_tokens") is not None:
            out["max_reasoning_tokens"] = reasoning["max_tokens"]
    if body.get("enable_thinking") is not None:
        out["enable_thinking"] = bool(body["enable_thinking"])
    return out


@lru_cache(maxsize=16)
def _keys_in_source(source: str) -> tuple[str, ...]:
    found = set(_KEY_READ.findall(source))
    return tuple(k for k in REASONING_KEYS if k in found)


def renderer_reasoning_keys(tokenizer) -> tuple[str, ...]:
    """Which of the reasoning keys the chat template reads from a message:
    looked up in the Jinja text, or in the source of a Python renderer."""
    renderer = getattr(tokenizer, "_chat_template", None)
    if callable(renderer):
        try:
            source = inspect.getsource(inspect.getmodule(renderer))
        except (OSError, TypeError):
            source = ""
        return _keys_in_source(source)
    template = getattr(tokenizer, "chat_template", None)
    return _keys_in_source(template) if isinstance(template, str) else ()


def mirror_reasoning(messages: list[dict], keys_read: tuple[str, ...]) -> None:
    """Copy an assistant turn's reasoning into the keys the renderer reads
    and the client left empty; keys the client filled stay as sent."""
    if not keys_read:
        return
    for m in messages:
        if m.get("role") != "assistant":
            continue
        source = next(
            (m[k] for k in REASONING_KEYS if isinstance(m.get(k), str) and m[k]),
            None,
        )
        if source is None:
            continue
        for key in keys_read:
            if not m.get(key):
                m[key] = source
