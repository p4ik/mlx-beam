"""A tokenizer with a 64-word vocabulary for tests that need no real model."""

from __future__ import annotations

EOS = 63
THINK_START, THINK_END = 60, 61
TOOL_START, TOOL_END = 58, 59


class StubDetokenizer:
    """mlx-lm's contract: ``last_segment`` is the text since it was last
    read; ``finalize`` flushes what was held back (nothing here)."""

    def __init__(self, words):
        self._words = words
        self.text = ""
        self._offset = 0

    def add_token(self, token: int) -> None:
        self.text += self._words[token]

    def finalize(self) -> None:
        pass

    @property
    def last_segment(self) -> str:
        segment = self.text[self._offset :]
        self._offset = len(self.text)
        return segment


class StubTokenizer:
    """Token t is the word ``w<t>``; a few ids are markers."""

    has_chat_template = True
    has_thinking = True
    has_tool_calling = True
    think_start = "<think>"
    think_end = "</think>"
    think_start_tokens = (THINK_START,)
    think_end_tokens = (THINK_END,)
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"
    structural_markers = ()
    # What the key detection reads: this "template" takes reasoning_content.
    chat_template = (
        "{% for message in messages %}{{ message.reasoning_content }}{% endfor %}"
    )

    def __init__(self):
        self._words = [f"w{t} " for t in range(64)]
        self._words[THINK_START] = "<think>"
        self._words[THINK_END] = "</think>"
        self._words[TOOL_START] = "<tool_call>"
        self._words[TOOL_END] = "</tool_call>"
        self._words[EOS] = "<eos>"
        self._ids = {w.strip(): t for t, w in enumerate(self._words)}
        self.eos_token_ids = {EOS}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._ids[w] for w in text.split() if w in self._ids]

    def decode(self, ids) -> str:
        return "".join(self._words[t] for t in ids)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True, tools=None, **kw
    ):
        ids = [2] if not tools else [2, 4]
        for m in messages:
            ids += [{"system": 5, "user": 6, "assistant": 7, "tool": 8}[m["role"]]]
            ids += self.encode(m.get("content") or "")
        if add_generation_prompt:
            ids.append(7)
        return ids if tokenize else self.decode(ids)

    @property
    def detokenizer(self):
        return StubDetokenizer(self._words)

    @staticmethod
    def tool_parser(text: str, tools):
        name, _, rest = text.strip().partition(" ")
        return {"name": name, "arguments": {"words": rest.strip()}}

    def rfind_think_start(self, tokens, start=None, end=None):
        return max((i for i, t in enumerate(tokens) if t == THINK_START), default=-1)

    def rfind_think_end(self, tokens, start=None, end=None):
        return max((i for i, t in enumerate(tokens) if t == THINK_END), default=-1)
