"""ATEM (Muse): tool calls as an XML-like block the template documents as
"not expected to be valid XML and parsed with regular expressions".

    <atem:function_calls>
    <atem:invoke name="NAME">
    <atem:parameter name="k">v</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

String and scalar values come as they are, lists and objects as JSON; the
tool's own schema (when handed in) says which parameters are numbers or
booleans, so "42" for an integer parameter becomes 42 and stays "42" for a
string one.
"""

import json
import re
from typing import Any, Optional

_INVOKE = re.compile(
    r'<atem:invoke\s+name="([^"]+)"\s*>(.*?)</atem:invoke>', re.DOTALL
)
_PARAM = re.compile(
    r'<atem:parameter\s+name="([^"]+)"\s*>(.*?)</atem:parameter>', re.DOTALL
)


def _schema(tools, name):
    for tool in tools or ():
        fn = tool.get("function", tool) if isinstance(tool, dict) else None
        if fn and fn.get("name") == name:
            return (fn.get("parameters") or {}).get("properties") or {}
    return {}


def _value(raw: str, kind):
    if kind in ("object", "array"):
        return json.loads(raw)
    if kind == "boolean":
        if raw.strip().lower() in ("true", "false"):
            return raw.strip().lower() == "true"
        raise ValueError(f"not a boolean: {raw!r}")
    if kind == "integer":
        return int(raw.strip())
    if kind == "number":
        return float(raw.strip())
    if kind == "string" or kind is not None:
        return raw
    # No schema: JSON for what looks like JSON, the literals, else the text.
    stripped = raw.strip()
    if stripped in ("true", "false", "null"):
        return json.loads(stripped)
    if stripped[:1] in ("{", "["):
        return json.loads(stripped)
    return raw


def parse_tool_call(text: str, tools: Optional[Any] = None):
    matches = list(_INVOKE.finditer(text))
    if not matches:
        raise ValueError("no <atem:invoke> block")
    calls = []
    for m in matches:
        name = m.group(1)
        schema = _schema(tools, name)
        arguments = {}
        for pm in _PARAM.finditer(m.group(2)):
            key = pm.group(1)
            kind = (schema.get(key) or {}).get("type") if schema else None
            arguments[key] = _value(pm.group(2), kind)
        calls.append(dict(name=name, arguments=arguments))
    return calls[0] if len(calls) == 1 else calls


# The recipient after `<|start|>assistant` opens the call as a label (the
# text assembler routes a label that is neither `self` nor `user` here).
tool_call_start = " to="
tool_call_end = "</atem:function_calls>"
