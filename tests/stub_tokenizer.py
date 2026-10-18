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


CHANNEL_OPEN, CHANNEL_CLOSE = 56, 57
LABEL, LABEL_SPACED, NEWLINE = 55, 54, 53


class ChannelStubTokenizer(StubTokenizer):
    """Gemma 4's family: the marker opens a channel, a label up to the line
    end names it, and the label may come as either of two tokens."""

    think_family = "channel"
    think_start = "<|channel>"
    think_end = "<channel|>"
    think_label_end = "\n"
    think_start_tokens = (CHANNEL_OPEN,)
    think_end_tokens = (CHANNEL_CLOSE,)

    def __init__(self):
        super().__init__()
        self._words[CHANNEL_OPEN] = "<|channel>"
        self._words[CHANNEL_CLOSE] = "<channel|>"
        self._words[LABEL] = "thought"
        self._words[LABEL_SPACED] = " thought"
        self._words[NEWLINE] = "\n"
        self._ids = {w.strip(): t for t, w in enumerate(self._words) if w.strip()}
        self._ids["\n"] = NEWLINE

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "\n":
            return [NEWLINE]
        return super().encode(text, add_special_tokens)

    def rfind_think_start(self, tokens, start=None, end=None):
        return max((i for i, t in enumerate(tokens) if t == CHANNEL_OPEN), default=-1)

    def rfind_think_end(self, tokens, start=None, end=None):
        return max((i for i, t in enumerate(tokens) if t == CHANNEL_CLOSE), default=-1)


FS, ASST, MSG, EOM = 36, 37, 38, 39
TO, SELF, EQ, SP, SELF2, SELFSP, S, ELF, USER, CHECK = (
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
)
# The close the family was trained to write: end marker, frame, message.
MUSE_CLOSE = [EOM, FS, ASST, MSG]


class MuseStubTokenizer(StubTokenizer):
    """Muse's family: the recipient ` to=` right after the frame
    `<|start|>assistant` says what follows - `self` the reasoning, `user`
    the answer, any other name a tool. The vocabulary merges `=self` into
    one token and also writes it as `=` and `self`, with or without a
    space, or as `=`, `s`, `elf`."""

    think_family = "muse"
    think_start = " to="
    think_end = "<|eom|>"
    think_label_end = "<|message|>"
    think_start_tokens = (TO, SELF)
    think_end_tokens = (EOM,)
    think_label_end_tokens = (MSG,)
    reasoning_label_tokens = ()
    think_close_tokens = tuple(MUSE_CLOSE)
    think_openers = (" to=",)
    frame_start = "<|start|>assistant"
    structural_markers = ("<|start|>assistant", "<|message|>", "<|eom|>")
    answer_opener_tokens = None
    tool_call_via_label = True
    tool_call_start = "<atem:function_calls>"
    tool_call_end = "</atem:function_calls>"
    tool_parser_type = "atem"

    _TEXTS = {
        " to=": [TO, EQ],
        " to=self": [TO, SELF],
        "<|start|>assistant": [FS, ASST],
        "<|message|>": [MSG],
        "<|eom|>": [EOM],
        "\n": [],
        "\n\n": [],
    }
    _WORDS = {
        FS: "<|start|>",
        ASST: "assistant",
        MSG: "<|message|>",
        EOM: "<|eom|>",
        TO: " to",
        SELF: "=self",
        EQ: "=",
        SP: " ",
        SELF2: "self",
        SELFSP: " self",
        S: "s",
        ELF: "elf",
        USER: "=user",
        CHECK: "_check",
    }

    def __init__(self):
        super().__init__()
        for t, w in self._WORDS.items():
            self._words[t] = w

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self._TEXTS:
            return list(self._TEXTS[text])
        return super().encode(text, add_special_tokens)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True, tools=None, **kw
    ):
        ids = super().apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True, tools=tools, **kw
        )
        if add_generation_prompt:
            # The generation prompt ends with the frame, as the template's does.
            ids += [FS, ASST]
        return ids if tokenize else self.decode(ids)

    def rfind_think_start(self, tokens, start=None, end=None):
        pairs = (i for i in range(len(tokens) - 1) if tokens[i : i + 2] == [TO, SELF])
        return max(pairs, default=-1)

    def rfind_think_end(self, tokens, start=None, end=None):
        return max((i for i, t in enumerate(tokens) if t == EOM), default=-1)
