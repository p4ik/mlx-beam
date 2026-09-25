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
from dataclasses import dataclass
from functools import lru_cache

from mlx_beam.api.errors import ApiError

# The effort words a request may send, in order of strength; `ultra` sits
# with `max`. What a template makes of them is measured at load, not
# assumed (probe_effort).
ORDINAL = ("minimal", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {"ultra": "max"}
# Words that mean thinking off.
EFFORT_OFF = ("none", "false", "off")
# The keys earlier assistant turns carry their thinking in, most specific
# first: that order decides which value fills a key the client left empty.
REASONING_KEYS = ("reasoning_content", "thinking", "reasoning")
# Kwarg names a template may take the effort under, searched in its source.
EFFORT_KWARGS = ("reasoning_effort", "reasoning_strength", "thinking_budget")
# Words the probe renders with, the client's vocabulary plus what some
# templates call the rungs; a template that validates says which it takes.
PROBE_WORDS = ORDINAL + ("ultra", "moderate")

# When a validating template rejects a level at request time after all
# (tools in the context, say), the next rung is tried: up first, then down.
EFFORT_LADDER = {
    "minimal": ("minimal", "low", "medium"),
    "low": ("low", "minimal", "medium"),
    "medium": ("medium", "moderate", "high", "low"),
    "high": ("high", "xhigh", "medium"),
    "xhigh": ("xhigh", "max", "high", "medium"),
    "max": ("max", "ultra", "xhigh", "high"),
    "ultra": ("ultra", "max", "xhigh", "high"),
}


@dataclass(frozen=True)
class EffortCapability:
    """What the loaded model's template does with an effort: nothing (no
    kwarg in its source), takes it under `kwarg` without checking, or
    checks it against a set (`levels` - the words it rendered, `rejected`
    the ones it refused)."""

    source: str  # "template" or "none"
    kwarg: str | None = None
    validates: bool = False
    levels: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()

    def describe(self) -> dict:
        if self.source == "none":
            return {"source": "none"}
        out = {"source": self.source, "kwarg": self.kwarg, "validates": self.validates}
        if self.validates:
            out["levels"] = list(self.levels)
        return out


def _template_source(tokenizer) -> str:
    renderer = getattr(tokenizer, "_chat_template", None)
    if callable(renderer):
        try:
            return inspect.getsource(inspect.getmodule(renderer))
        except (OSError, TypeError):
            return ""
    template = getattr(tokenizer, "chat_template", None)
    return template if isinstance(template, str) else ""


def probe_effort(tokenizer) -> EffortCapability:
    """Measured once per model: the kwarg the template reads the effort
    from (found in its source), then a one-turn conversation rendered with
    every probe word - what renders is accepted, what raises is rejected.
    A template that rejects nothing does not check; its accepted set says
    nothing about the model and is not reported as levels."""
    source = _template_source(tokenizer)
    kwarg = next((k for k in EFFORT_KWARGS if k in source), None)
    if kwarg is None:
        # Another reasoning_* name read as a variable, not a message key
        # (`message.reasoning_content` is the turn's own field).
        for m in re.finditer(r"""(?<![\.\w'"\[])(reasoning_[a-z_]+)\b""", source):
            if m.group(1) not in REASONING_KEYS:
                kwarg = m.group(1)
                break
    if kwarg is None:
        return EffortCapability("none")
    if not kwarg.startswith("reasoning_"):
        # A budget in tokens, not a word: reported, never probed with words.
        return EffortCapability("template", kwarg, False)
    accepted, rejected = [], []
    for word in PROBE_WORDS:
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                add_generation_prompt=True,
                tokenize=False,
                **{kwarg: word},
            )
            accepted.append(word)
        except Exception:  # noqa: BLE001 - the template's verdict is the data
            rejected.append(word)
    return EffortCapability(
        "template", kwarg, bool(rejected), tuple(accepted), tuple(rejected)
    )


def effort_capability(tokenizer) -> EffortCapability:
    """The probe's result, kept on the tokenizer after the first call."""
    cached = getattr(tokenizer, "_effort_capability", None)
    if cached is None:
        cached = probe_effort(tokenizer)
        try:
            tokenizer._effort_capability = cached
        except AttributeError:
            pass
    return cached


def normalise_effort(effort) -> str | None:
    """The request's effort word, checked against the vocabulary; None for
    thinking off. Unknown words are refused with the list that applies."""
    if effort is None:
        return None
    if isinstance(effort, bool):
        return "xhigh" if effort else None
    word = str(effort).lower()
    if word in EFFORT_OFF:
        return None
    if word not in ORDINAL and word not in EFFORT_ALIASES:
        raise ApiError(
            f"unknown reasoning_effort {effort!r}; one of "
            f"{', '.join(EFFORT_OFF + ORDINAL + tuple(EFFORT_ALIASES))}",
            param="reasoning_effort",
        )
    return word


def _rank(word: str) -> int:
    return ORDINAL.index(EFFORT_ALIASES.get(word, word))


def translate_effort(word: str, cap: EffortCapability) -> str | None:
    """The value the template gets for the client's word: the word itself
    when the template takes it unchecked (OpenAI's vocabulary, which such
    models were trained on) or accepts it; else the nearest accepted rung
    on the ordinal ladder, up first, then down. None: the template knows
    no effort, the kwarg stays out."""
    if cap.source == "none":
        return None
    if not cap.validates or word in cap.levels:
        return word
    accepted = [w for w in cap.levels if w in ORDINAL or w in EFFORT_ALIASES]
    if not accepted:
        return word  # it rejected everything we know: leave it to the ladder
    rank = _rank(word)
    up = sorted((w for w in accepted if _rank(w) >= rank), key=_rank)
    down = sorted((w for w in accepted if _rank(w) < rank), key=_rank, reverse=True)
    return (up or down)[0]


def map_effort(effort) -> str | None:
    """The request's word for the template, or None for off; kept for the
    callers that only need the switch."""
    return normalise_effort(effort)


# message.thinking, message["thinking"], msg.get("thinking"): an access, not
# the word in prose (an ellipsis before it is prose).
_KEY_READ = re.compile(
    r"""(?:(?<!\.)\.|\[\s*['"]|get\(\s*['"])(reasoning_content|thinking|reasoning)(?![\w])"""
)


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
        value = body["enable_thinking"]
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"  # some clients send the word
        out["enable_thinking"] = bool(value)
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
