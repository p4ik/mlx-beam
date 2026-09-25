"""The tool-call repair ladder: strict, repaired, coerced - each rung
validated against the declared schema, nothing invented, every action
reported; what no rung makes valid comes back as text."""

import json

from mlx_beam._vendor.mlx_lm.tool_parsers import json_tools
from mlx_beam.api import repair
from mlx_beam.api.text import ToolCallParser

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
        "units": {"type": "string"},
        "opts": {"type": "object"},
        "flag": {"type": "boolean"},
    },
    "required": ["city"],
    "additionalProperties": False,
}
TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": SCHEMA}}]


def parse(text, tools=TOOLS):
    p = ToolCallParser(
        json_tools.parse_tool_call,
        tools,
        streaming=False,
        start="<tool_call>",
        end="</tool_call>",
    )
    calls, unparsed = p([(text, True)])
    return calls, unparsed


def test_strict_call_passes_untouched():
    calls, unparsed = parse(
        '{"name": "weather", "arguments": {"city": "Oslo", "days": 2}}'
    )
    assert not unparsed and "repair_actions" not in calls[0]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Oslo", "days": 2}


def test_mended_json_is_read_and_reported():
    before = dict(repair.STATS)
    text = (
        "```json\n{'name': 'weather', 'arguments': {'city': 'Oslo', 'flag': True,}\n```"
    )
    calls, unparsed = parse(text)
    assert not unparsed and calls[0]["repair_actions"] == ["json repaired"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "city": "Oslo",
        "flag": True,
    }
    assert repair.STATS["repaired"] == before["repaired"] + 1
    # Braces left open at the end are closed.
    calls, unparsed = parse('{"name": "weather", "arguments": {"city": "Oslo"')
    assert not unparsed and json.loads(calls[0]["function"]["arguments"]) == {
        "city": "Oslo"
    }


def test_values_are_coerced_to_the_declared_type_and_reported():
    text = '{"name": "weather", "arguments": {"city": 7, "days": "3", "flag": "false", "opts": "{\\"a\\": 1}"}}'
    calls, unparsed = parse(text)
    assert not unparsed
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"city": "7", "days": 3, "flag": False, "opts": {"a": 1}}
    assert calls[0]["repair_actions"] == [
        "city: int -> string",
        "days: str -> integer",
        "flag: str -> boolean",
        "opts: str -> object",
    ]


def test_what_the_schema_still_rejects_comes_back_as_text():
    before = repair.STATS["failed"]
    # A required key is missing: nothing is invented.
    calls, unparsed = parse('{"name": "weather", "arguments": {"days": 2}}')
    assert not calls and unparsed[0].startswith("<tool_call>")
    # A value that is not plainly the type.
    calls, unparsed = parse(
        '{"name": "weather", "arguments": {"city": "Oslo", "days": "soon"}}'
    )
    assert not calls and unparsed
    # A key the schema forbids.
    calls, unparsed = parse(
        '{"name": "weather", "arguments": {"city": "Oslo", "x": 1}}'
    )
    assert not calls and unparsed
    assert repair.STATS["failed"] == before + 3
    # A tool the request did not declare: the parser's reading, unvalidated.
    calls, unparsed = parse('{"name": "other", "arguments": {"z": 1}}')
    assert calls and not unparsed


def test_validate_and_coerce_directly():
    assert repair.validate({"city": "x"}, SCHEMA) == []
    assert repair.validate({"city": 1}, SCHEMA) == ["'city' is not string"]
    assert repair.validate([], SCHEMA) == ["arguments are not an object"]
    args, actions = repair.coerce({"days": 2.0, "flag": "True"}, SCHEMA)
    assert args == {"days": 2, "flag": True} and len(actions) == 2
    assert repair.repair_json_text('{"a": [1, 2,') == '{"a": [1, 2]}'
    assert repair.repair_json_text('{"a": "unterminated') == '{"a": "unterminated"}'
