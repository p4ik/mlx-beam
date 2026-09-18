"""The vendored tool-call parsers on the shapes the models actually emit."""

import copy

import pytest

from mlx_beam._vendor.mlx_lm.tool_parsers import mistral, qwen3_coder

WRITE = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "lines": {"type": "integer"},
                },
            },
        },
    }
]


def test_qwen_parameter_without_closing_bracket():
    # The model sometimes drops the ">" after the parameter name.
    text = "<function=write><parameter=path\n/tmp/a</parameter></function>"
    assert qwen3_coder.parse_tool_call(text, WRITE) == {
        "name": "write",
        "arguments": {"path": "/tmp/a"},
    }
    text = "<function=write><parameter=lines>\n42\n</parameter></function>"
    assert qwen3_coder.parse_tool_call(text, WRITE)["arguments"] == {"lines": 42}


def test_mistral_json_list_and_cut_call():
    calls = mistral.parse_tool_call(
        '[{"name": "write", "arguments": {"path": "a"}, "id": "abc"},'
        ' {"name": "read", "arguments": "{\\"path\\": \\"b\\"}"}]'
    )
    assert calls == [
        {"name": "write", "arguments": {"path": "a"}, "id": "abc"},
        {"name": "read", "arguments": {"path": "b"}},
    ]
    assert mistral.parse_tool_call('write[ARGS]{"path": "a"}') == {
        "name": "write",
        "arguments": {"path": "a"},
    }
    # A call cut short is an error now, not a silently shorter list.
    with pytest.raises(ValueError):
        mistral.parse_tool_call('write[ARGS]{"path": "a"}[TOOL_CALLS]read[ARGS]{"pa')


def test_qwen_literal_end_tag_in_a_value():
    text = (
        "<function=write><parameter=path>a.xml</parameter>"
        "<parameter=content>before </parameter> after</parameter>"
        "<parameter=lines>2</parameter></function>"
    )
    assert qwen3_coder.parse_tool_call(text, WRITE)["arguments"] == {
        "path": "a.xml",
        "content": "before </parameter> after",
        "lines": 2,
    }
    # The tag at the very end of a value, and as the whole value.
    text = "<function=write><parameter=content>x</parameter></parameter></function>"
    assert qwen3_coder.parse_tool_call(text, WRITE)["arguments"] == {
        "content": "x</parameter>"
    }
    # A parameter cut short is an error, not a missing argument.
    with pytest.raises(ValueError):
        qwen3_coder.parse_tool_call(
            "<function=write><parameter=path>a</parameter><parameter=content>bef</function>",
            WRITE,
        )


def test_qwen_parameter_tag_inside_a_value_is_refused_not_silently_split():
    """A literal <parameter=…> in a value cannot be told from a second
    parameter; a name seen twice makes the call an error the caller shows,
    never a call with one argument overwritten by the other."""
    text = (
        "<function=write><parameter=path>intended.txt</parameter>"
        "<parameter=content>Use </parameter> before <parameter=path>other.txt"
        "</parameter> after</parameter></function>"
    )
    with pytest.raises(ValueError, match="twice|outside"):
        qwen3_coder.parse_tool_call(text, WRITE)
    # A name outside a closed schema is refused the same way; an open
    # schema (no additionalProperties: false) still lets it through.
    text = (
        "<function=write><parameter=path>a</parameter>"
        "<parameter=extra>b</parameter></function>"
    )
    assert qwen3_coder.parse_tool_call(text, WRITE)["arguments"] == {
        "path": "a",
        "extra": "b",
    }
    closed = copy.deepcopy(WRITE)
    closed[0]["function"]["parameters"]["additionalProperties"] = False
    with pytest.raises(ValueError, match="schema"):
        qwen3_coder.parse_tool_call(text, closed)
    with pytest.raises(ValueError, match="No function"):
        qwen3_coder.parse_tool_call("plain text", WRITE)


def test_mistral_headers_are_anchored_never_read_out_of_a_json_string():
    """A cut-off JSON list is not a call; a search for "name[ARGS]" would
    find one inside a string argument. Headers stand at the start or right
    after the previous call, with the marker the model repeats in between."""
    cut = (
        '[{"name":"write","arguments":{"content":"Example: example_action[ARGS]{}"}},'
        ' {"name":"bad","arguments":'
    )
    with pytest.raises(ValueError):
        mistral.parse_tool_call(cut, None)
    calls = mistral.parse_tool_call(
        'write[ARGS]{"a": 1}[TOOL_CALLS]other[ARGS]{"b": 2}', None
    )
    assert [c["name"] for c in calls] == ["write", "other"]
    with pytest.raises(ValueError):
        mistral.parse_tool_call('write[ARGS]{"a": 1} stray text', None)


def test_qwen_text_between_parameters_is_an_error_not_dropped():
    """A literal <parameter=…> in a value that names an unused, allowed
    parameter would split the value and drop the words after the literal
    end tag; that text belongs to nobody, so the call is refused."""
    text = (
        "<function=write><parameter=content>Use </parameter> before "
        "<parameter=path>other.txt</parameter> after</parameter></function>"
    )
    closed = copy.deepcopy(WRITE)
    closed[0]["function"]["parameters"]["additionalProperties"] = False
    with pytest.raises(ValueError, match="outside"):
        qwen3_coder.parse_tool_call(text, closed)
    # Whitespace and newlines between parameters are the format itself.
    ok = "<function=write>\n<parameter=path>\na\n</parameter>\n<parameter=content>\nx\n</parameter>\n</function>"
    assert qwen3_coder.parse_tool_call(ok, closed)["arguments"] == {
        "path": "a",
        "content": "x",
    }
