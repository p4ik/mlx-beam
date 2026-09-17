# Copyright © 2025 Apple Inc.

"""
Modified from:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
"""

import ast
import json
from typing import Any, Optional

import regex as re

_function_regex = re.compile(r"<function=(.*?)</function>$", re.DOTALL)
_parameter_start = re.compile(r"<parameter=")
_parameter_end = "</parameter>"
_name_regex = re.compile(r"\s*([^\s<>]+)>?")

_string_types = {"string", "str", "text", "varchar", "char", "enum"}
_bool_types = {"boolean", "bool", "binary"}
_obj_types = {"object", "array", "arr"}


def _get_arguments_config(func_name: str, tools: Optional[Any]) -> dict:
    """Extract argument configuration for a function."""
    if tools is None:
        return {}
    for tool in tools:
        if not (function := tool.get("function", False)):
            continue
        if function["name"] == func_name:
            if not (params := function.get("parameters", False)):
                return {}
            return params.get("properties", {})
    return {}


def _closed_parameters(func_name: str, tools: Optional[Any]) -> Optional[set]:
    """The parameter names a schema with ``additionalProperties: false``
    allows, or None when the schema leaves the names open."""
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool, dict) else None
        if not function or function.get("name") != func_name:
            continue
        params = function.get("parameters")
        if isinstance(params, dict) and params.get("additionalProperties") is False:
            return set(params.get("properties") or {})
        return None
    return None


def _convert_param_value(param_value: str, param_name: str, param_config: dict) -> Any:
    """Convert parameter value based on its type in the schema."""
    if param_value.lower() == "null":
        return None

    if not (param := param_config.get(param_name, False)):
        return param_value

    if "type" in param:
        param_type = str(param["type"]).strip().lower()
    else:
        param_type = "string"
    if param_type in _string_types:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        if float_param_value - int_param_value != 0:
            raise ValueError(f"Invalid integer literal {param_value!r}")
        return int_param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        return (
            float_param_value
            if (float_param_value - int_param_value) != 0
            else int_param_value
        )
    elif param_type in _bool_types:
        return param_value.lower() == "true"
    else:
        if (
            param_type in _obj_types
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value, strict=False)
            except json.JSONDecodeError:
                pass

        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            return param_value


def _parameter_bodies(parameters: str) -> list:
    """Each parameter's ``name>value``. A parameter ends at the last
    ``</parameter>`` before the next ``<parameter=`` (or the end), so a value
    that contains the end tag literally - HTML, XML, a file with this very
    markup - is kept whole instead of cut at its first occurrence. A
    parameter without an end tag means the call was cut short."""
    starts = [m.end() for m in _parameter_start.finditer(parameters)]
    bodies = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] - len("<parameter=") if i + 1 < len(starts) else None
        chunk = parameters[start:stop]
        close = chunk.rfind(_parameter_end)
        if close < 0:
            raise ValueError("Parameter without a closing tag.")
        bodies.append(chunk[:close])
    return bodies


def _parse_xml_function_call(function_call_str: str, tools: Optional[Any]):
    name_match = _name_regex.match(function_call_str)
    if name_match is None:
        raise ValueError("No function name provided.")
    function_name = name_match.group(1)
    param_config = _get_arguments_config(function_name, tools)
    closed = _closed_parameters(function_name, tools)
    parameters = function_call_str[name_match.end() :]
    param_dict = {}
    for match_text in _parameter_bodies(parameters):
        param_match = _name_regex.match(match_text)
        if param_match is None:
            continue
        param_name = param_match.group(1)
        # A parameter tag inside a value looks like a second parameter; a
        # name seen twice, or one the schema rules out, is that case or a
        # broken call - either way not a call to make in silence.
        if param_name in param_dict:
            raise ValueError(f"Parameter {param_name!r} given twice.")
        if closed is not None and param_name not in closed:
            raise ValueError(f"Parameter {param_name!r} is not in the schema.")
        param_value = str(match_text[param_match.end() :])
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]

        param_dict[param_name] = _convert_param_value(
            param_value, param_name, param_config
        )
    return dict(name=function_name, arguments=param_dict)


tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(
    model_output: str,
    tools: Optional[Any] = None,
):
    match = _function_regex.findall(model_output)
    if not match:
        raise ValueError("No function provided.")
    return _parse_xml_function_call(match[0], tools)
