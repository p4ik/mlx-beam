"""Harmony (gpt-oss): a tool call is a commentary message with a recipient.

The text handed in is the message header from the recipient on plus the
body: `commentary to=functions.NAME <|constrain|>json<|message|>{...}` when
the recipient follows the channel, `functions.NAME<|channel|>commentary
json<|message|>{...}` when it precedes it (how the chat template renders
history). The name is the recipient without its `functions.` namespace, the
arguments are the JSON body.
"""

import json
import re
from typing import Any, Optional

_RECIPIENT = re.compile(r"(?:^|\s|to=)functions\.([\w.\-]+)")


def parse_tool_call(text: str, _: Optional[Any] = None):
    header, sep, body = text.partition("<|message|>")
    if not sep:
        raise ValueError("no <|message|> after the tool call's header")
    m = _RECIPIENT.search(header)
    if m is None:
        raise ValueError("the header names no functions.* recipient")
    body = body.strip()
    for stop in ("<|call|>", "<|end|>"):
        if body.endswith(stop):
            body = body[: -len(stop)].rstrip()
    arguments = json.loads(body) if body else {}
    if not isinstance(arguments, dict):
        raise ValueError("the tool call's body is not a JSON object")
    return dict(name=m.group(1), arguments=arguments)


# The recipient opens the call as a label of the message header (the text
# assembler routes a label naming a recipient here); <|call|> ends it.
tool_call_start = " to="
tool_call_end = "<|call|>"
