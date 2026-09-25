"""The tool-call repair ladder: strict, repaired, coerced - each rung
validated against the declared schema, nothing invented, every action
reported; what no rung makes valid comes back as text."""

import json

import pytest

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


def parse(text, tools=TOOLS, closed=True, dialect="json_tools"):
    p = ToolCallParser(
        json_tools.parse_tool_call,
        tools,
        streaming=False,
        start="<tool_call>",
        end="</tool_call>",
        dialect=dialect,
    )
    calls, unparsed = p([(text, closed)])
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
    # A tool the request did not declare is text, not a call the client
    # could run - unless the request declared no names at all (then the
    # parser's reading stands, unvalidated).
    calls, unparsed = parse('{"name": "other", "arguments": {"z": 1}}')
    assert not calls and unparsed
    calls, unparsed = parse('{"name": "other", "arguments": {"z": 1}}', tools=[])
    assert calls and not unparsed


def test_validate_and_coerce_directly():
    assert repair.validate({"city": "x"}, SCHEMA) == []
    assert repair.validate({"city": 1}, SCHEMA) == ["'city' is not string"]
    assert repair.validate([], SCHEMA) == ["arguments are not an object"]
    args, actions = repair.coerce({"days": 2.0, "flag": "True"}, SCHEMA)
    assert args == {"days": 2, "flag": True} and len(actions) == 2
    assert repair.repair_json_text('{"a": [1, 2,') == '{"a": [1, 2]}'
    assert repair.repair_json_text('{"a": "unterminated') == '{"a": "unterminated"}'


@pytest.mark.parametrize(
    "value", ["say True now", "a,}b", "None of it", 'quoted \\" inside, }', "x' y"]
)
def test_a_repair_leaves_string_contents_alone(value):
    """The mending is syntax only: a Python literal or a comma before a
    closing bracket inside a string is content, and a tool that writes
    files gets it as the model wrote it."""
    raw = json.dumps({"name": "weather", "arguments": {"city": value}})
    calls, unparsed = parse(raw[:-1] + ",}")  # a trailing comma to repair
    assert calls and not unparsed
    assert calls[0]["repair_actions"] == ["json repaired"]
    assert json.loads(calls[0]["function"]["arguments"])["city"] == value
    assert repair.repair_json_text('{"a": "unterminated') == '{"a": "unterminated"}'
    assert repair.repair_json_text('{"a": "') == '{"a": ""}'


def test_a_call_the_stream_cut_off_is_not_mended_into_a_valid_one():
    """max_tokens ended the block inside a string: closing it would hand
    the tool a truncated value as if it were what the model meant. The
    block comes back as text, with no end marker."""
    cut = '{"name": "weather", "arguments": {"city": "Os'
    calls, unparsed = parse(cut, closed=False)
    assert not calls and unparsed == ["<tool_call>" + cut]
    # The same text closed by the model is mended as before.
    calls, unparsed = parse(cut + '"}', closed=True)
    assert calls and json.loads(calls[0]["function"]["arguments"]) == {"city": "Os"}


def test_the_text_repair_runs_for_json_dialects_only():
    # An XML-like dialect keeps its values outside JSON quotes; the
    # literal and comma rewrites would change them.
    calls, unparsed = parse(
        '{"name": "weather", "arguments": {"city": "Oslo"},}', dialect="qwen3_coder"
    )
    assert not calls and unparsed


def test_boolean_subschemas_and_null_for_an_array():
    """JSON Schema lets a property's schema be `true`; that is not a 500.
    And null is not turned into `[null]` for an array."""
    schema = {"type": "object", "properties": {"x": True, "y": {"type": "array"}}}
    assert repair.validate({"x": 1, "y": []}, schema) == []
    assert repair.validate({"y": None}, schema) == ["'y' is not array"]
    args, actions = repair.coerce({"y": None, "x": 2}, schema)
    assert args["y"] is None and not actions
    args, actions = repair.coerce({"y": 5}, schema)
    assert args["y"] == [5] and actions


def test_a_call_to_an_undeclared_tool_is_text():
    calls, unparsed = parse('{"name": "rm_rf", "arguments": {"path": "/"}}')
    assert not calls and unparsed and "rm_rf" in unparsed[0]


def test_single_quoted_strings_keep_their_apostrophes():
    """Python-style quoting is read with its escapes: `it\\'s` is an
    apostrophe, not a quote that ends the string; a backslash stays one; a
    string left open is not mended at all (the text comes back as text)."""
    calls, unparsed = parse(r"{'name': 'weather', 'arguments': {'city': 'it\'s'}}")
    assert not unparsed and json.loads(calls[0]["function"]["arguments"]) == {
        "city": "it's"
    }
    calls, _ = parse(r"{'name': 'weather', 'arguments': {'city': 'a\\b \"c\"'}}")
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": 'a\\b "c"'}
    calls, unparsed = parse("{'name': 'weather', 'arguments': {'city': 'open}}")
    assert not calls and unparsed
