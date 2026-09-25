"""The repair ladder for tool calls: strict, then repaired, then coerced -
each rung validated against the tool's declared schema, nothing invented.

A call the dialect's parser reads and the schema accepts passes as it is.
One the parser refuses is parsed again after the usual defects of model
JSON are mended (a trailing comma, single quotes, Python's literals, a
code fence, braces left open). Arguments the schema rejects by type are
coerced where the value plainly is the declared type written another way
("42" for an integer, "true" for a boolean, a JSON text for an object);
anything else fails the rung. What each rung did is the call's
`repair_actions`, and a call no rung can make valid comes back as text.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Counters since start, for /health.tools.repairs.
STATS = {"strict": 0, "repaired": 0, "coerced": 0, "failed": 0}

# The JSON types whose check is a plain isinstance; integer and number
# exclude bool by hand (bool is an int in Python).
_TYPES = {
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _spec(props: Any, key: str) -> dict:
    """A property's schema as a dict: JSON Schema allows `true`/`false` as
    a schema too, and a schema that is not a dict says nothing here."""
    spec = props.get(key) if isinstance(props, dict) else None
    return spec if isinstance(spec, dict) else {}


def declared_names(tools: Any) -> set[str]:
    """The names the request declared; empty when it named none (then any
    call the model makes is passed on, there is nothing to hold it to)."""
    names = set()
    for tool in tools or ():
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def schema_for(tools: Any, name: str) -> dict | None:
    """The declared parameters of the tool called `name`, or None when the
    request declared no such tool (nothing to validate against)."""
    for tool in tools or ():
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", tool)
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters") or fn.get("input_schema")
            return params if isinstance(params, dict) else {}
    return None


def _matches(value: Any, kind: str) -> bool:
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _TYPES.get(kind)
    return expected is None or isinstance(value, expected)


def validate(args: Any, schema: dict) -> list[str]:
    """What the schema objects to: not an object, a required key missing,
    a value of the wrong type, a key the schema forbids."""
    if not isinstance(args, dict):
        return ["arguments are not an object"]
    problems = []
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    for key in schema.get("required") or ():
        if key not in args:
            problems.append(f"missing required {key!r}")
    for key, value in args.items():
        if key not in props:
            if schema.get("additionalProperties") is False:
                problems.append(f"unknown key {key!r}")
            continue
        spec = _spec(props, key)
        kinds = spec.get("type")
        if kinds is None:
            continue
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(_matches(value, k) for k in kinds):
            problems.append(f"{key!r} is not {' or '.join(kinds)}")
    return problems


def coerce(args: dict, schema: dict) -> tuple[dict, list[str]]:
    """Values that are the declared type written another way, converted;
    the actions taken, one per key. A value that is not plainly the type
    stays as it is (and fails validation afterwards)."""
    props = schema.get("properties") or {}
    out, actions = dict(args), []
    for key, value in args.items():
        spec = _spec(props, key)
        kinds = spec.get("type")
        if kinds is None:
            continue
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if any(_matches(value, k) for k in kinds):
            continue
        new = _coerce_one(value, kinds)
        if new is not value and any(_matches(new, k) for k in kinds):
            out[key] = new
            actions.append(f"{key}: {type(value).__name__} -> {kinds[0]}")
    return out, actions


def _coerce_one(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        if isinstance(value, str):
            text = value.strip()
            if kind == "integer" and re.fullmatch(r"[+-]?\d+", text):
                return int(text)
            if kind == "number" and re.fullmatch(
                r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text
            ):
                return float(text)
            if kind == "boolean" and text.lower() in ("true", "false"):
                return text.lower() == "true"
            if kind == "null" and text.lower() in ("null", "none"):
                return None
            if kind in ("object", "array"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = _loads_repaired(text)
                if parsed is not None and _matches(parsed, kind):
                    return parsed
        elif kind == "string" and isinstance(value, (int, float, bool)):
            return json.dumps(value)
        elif kind == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        elif (
            kind == "array"
            and value is not None
            and not isinstance(value, (list, dict))
        ):
            # A lone scalar for a list is the list of it; null is no list.
            return [value]
    return value


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_PY_LITERAL = re.compile(r"(?<![\w\"'])(True|False|None)(?![\w\"'])")


def _segments(text: str) -> list[tuple[str, bool]]:
    """The text cut into (segment, inside a string) pieces by JSON's own
    string rules - quotes and backslash escapes - so a repair can touch the
    syntax and leave every string's content as it came."""
    out: list[tuple[str, bool]] = []
    start = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                out.append((text[start : i + 1], True))
                start = i + 1
                in_string = False
        elif ch == '"':
            if i > start:
                out.append((text[start:i], False))
            start = i
            in_string = True
    if start < len(text):
        out.append((text[start:], in_string))
    return out


def repair_json_text(text: str) -> str:
    """The usual defects of model JSON mended, in the syntax only - the
    dialect's markers stay where they are and a string's content is never
    touched: a code fence, Python's literals, single-quoted strings and
    keys, a trailing comma, braces or brackets left open at the end."""
    out = _FENCE.sub("", text)
    if "'" in out and '"' not in out:
        out = out.replace("'", '"')
    pieces = _segments(out)
    mended = []
    for segment, in_string in pieces:
        if not in_string:
            segment = _PY_LITERAL.sub(
                lambda m: {"True": "true", "False": "false", "None": "null"}[
                    m.group(1)
                ],
                segment,
            )
            segment = _TRAILING_COMMA.sub(r"\1", segment)
        mended.append(segment)
    out = "".join(mended)
    stack = []
    for segment, in_string in pieces:
        if in_string:
            continue
        for ch in segment:
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack and stack[-1] == ch:
                stack.pop()
    if pieces and pieces[-1][1]:
        last = pieces[-1][0]
        if len(last) == 1 or not last.endswith('"'):
            out += '"'  # a string left open at the end
    # A comma left dangling before the closing we add.
    out = re.sub(r",\s*$", "", out) if stack else out
    out += "".join(reversed(stack))
    return out


def _loads_repaired(text: str) -> Any:
    try:
        return json.loads(repair_json_text(text))
    except json.JSONDecodeError:
        return None
