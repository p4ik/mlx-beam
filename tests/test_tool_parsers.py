"""The vendored tool-call parsers on the shapes the models actually emit."""

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
