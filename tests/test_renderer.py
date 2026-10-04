"""The renderer decision for a checkpoint, and the engine on a renderer
without template text: mistral-common (transformers' MistralCommonBackend)
hands ids back, names no template and takes the tool-call parser from the
vocabulary. A stand-in with that shape runs the real tokenizer wrapper and
the real prompt path; the backend itself is exercised in the acceptance
runs on a tekken checkpoint."""

import json

from mlx_beam import renderer
from mlx_beam._vendor.mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_beam.api import chat, roles
from mlx_beam.api.defaults import RequestDefaults

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        },
    }
]


class MistralCommonBackend:
    """What the engine reads of transformers' MistralCommonBackend: no
    template text, a vocabulary with the control tokens, and
    `apply_chat_template` that renders ids (a dict with the pixel values
    when the messages carry images), taking any keyword without complaint.
    The class name is what the vision package detects."""

    chat_template = None
    eos_token_id = 2
    unk_token_id = 0
    SPECIAL = {"<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "[/INST]": 4}
    ROLES = {"system": 17, "user": 3, "assistant": 5, "tool": 7}

    def __init__(self, tools_trained=True):
        self.vocab = dict(self.SPECIAL)
        if tools_trained:
            self.vocab["[TOOL_CALLS]"] = 9
        self.init_kwargs = {}
        self.calls = []

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        # Control tokens written as text are plain words to mistral-common.
        return [100 + (ord(c) % 50) for c in text]

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        return_dict=True,
        **kwargs,
    ):
        self.calls.append({"tools": tools, "kwargs": kwargs})
        ids = [1]
        if tools:
            ids.append(11)
        for m in conversation:
            if m["role"] not in self.ROLES:
                raise ValueError(f"unknown role {m['role']}")
            ids.append(self.ROLES[m["role"]])
            content = m.get("content")
            if isinstance(content, str):
                ids += self.encode(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        ids += self.encode(part["text"])
                    elif part.get("type") == "image_url":
                        ids += [10, 10, 12, 10, 10, 13]
            for call in m.get("tool_calls") or ():
                ids += [9] + self.encode(call["function"]["name"])
        if not tokenize:
            return self.decode(ids)
        if not return_dict:
            return ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def wrapped(tools_trained=True):
    return TokenizerWrapper(MistralCommonBackend(tools_trained))


def test_choose_takes_mistral_common_for_a_tekken_checkpoint(tmp_path):
    """Without tekken.json the template path; with it mistral-common -
    also when a conversion added a template beside it (the note says the
    template is left aside). A template on the command line keeps the
    tokenizer on the template path whatever the checkpoint ships."""
    assert renderer.choose(tmp_path).name == "jinja"
    (tmp_path / "tekken.json").write_text("{}")
    chosen = renderer.choose(tmp_path)
    assert chosen.name == "mistral-common"
    assert chosen.tokenizer_kwargs == {"mistral_format": True} and chosen.note is None
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    assert "left aside" in renderer.choose(tmp_path).note
    assert renderer.choose(tmp_path, "{{ flag }}").tokenizer_kwargs == {
        "mistral_format": False
    }


def test_the_tool_parser_comes_from_the_vocabulary():
    """No template names the parser, so the `[TOOL_CALLS]` token does:
    its id straight from the vocabulary (encoded as text it would be
    words), mlx-lm's mistral parser behind it; a vocabulary without the
    token gets none."""
    tok = wrapped()
    assert not tok.has_tool_calling
    assert renderer.install_tool_parser(tok) == "mistral"
    assert tok.has_tool_calling and tok.tool_parser_type == "mistral"
    assert tok.tool_call_start == "[TOOL_CALLS]" and tok.tool_call_start_tokens == (9,)
    assert tok.tool_parser('calc[ARGS]{"expr": "2+2"}', TOOLS) == {
        "name": "calc",
        "arguments": {"expr": "2+2"},
    }
    assert renderer.install_tool_parser(tok) == "mistral"  # idempotent
    assert renderer.install_tool_parser(wrapped(tools_trained=False)) is None


def test_the_prompt_path_runs_on_ids_alone():
    """Through the real wrapper and the real chat path: the wrapper's
    template probes find no text and settle (no thinking, no effort
    kwarg, developer rendered as system), the prompt comes back as ids
    with the tools passed through, and a tool-call turn renders."""
    tok = wrapped()
    renderer.install_tool_parser(tok)
    assert tok.has_chat_template and not tok.has_thinking
    assert roles.describe(tok) == {"developer": "as system"}
    defaults = RequestDefaults.resolve(None, flags={})
    body = {
        "model": "m",
        "messages": [
            {"role": "developer", "content": "brief"},
            {"role": "user", "content": "2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "abcdefghi",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "abcdefghi", "content": "4"},
        ],
        "tools": TOOLS,
        "reasoning_effort": "low",
    }
    req = chat.parse_chat_request(body, "m", defaults)
    ids = chat.build_prompt(tok, req)
    backend = tok._tokenizer
    assert isinstance(ids, list) and ids[:2] == [1, 11] and 17 in ids and 9 in ids
    assert backend.calls[-1]["tools"] == TOOLS
