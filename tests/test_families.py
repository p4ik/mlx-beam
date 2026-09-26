"""Marker families with a message frame: Harmony (gpt-oss) and Muse (ATEM).

The reasoning, the answer and a tool call all sit behind a label - the
channel or the recipient - and the text assembler routes by what the label
says. The chat templates are the models' own (tests/fixtures/templates)."""

import re
from pathlib import Path

import pytest

from mlx_beam._vendor.mlx_lm import tokenizer_utils as tu
from mlx_beam._vendor.mlx_lm.tool_parsers import atem, harmony
from mlx_beam.api import chat
from mlx_beam.api.reasoning import effort_capability
from mlx_beam.api.text import TextAssembler, initial_state
from mlx_beam.engine.request import TokenEvent
from mlx_beam.engine.thinking import ReasoningLimits, ThinkingBudget
from tests.stub_tokenizer import StubDetokenizer
from tests.test_thinking import Steps

TEMPLATES = Path(__file__).parent / "fixtures" / "templates"

HARMONY_SPECIALS = (
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|constrain|>",
    "<|return|>",
    "<|call|>",
)
MUSE_SPECIALS = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|begin_of_text|>")


class FakeHF:
    """Enough of a Hugging Face tokenizer for the wrapper's inference: a
    vocabulary of special tokens and words, a chat template, encode by
    whitespace with the specials as their own tokens."""

    def __init__(self, specials, template: str):
        self._specials = list(specials)
        self.chat_template = template
        self.eos_token_id = 0
        self._vocab = {t: i + 1 for i, t in enumerate(self._specials)}

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, ids, **kw):
        back = {i: t for t, i in self._vocab.items()}
        return " ".join(back.get(i, "?") for i in ids)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True, **kw
    ):
        # The template rendered by jinja2 as transformers would render it.
        from jinja2 import Environment

        env = Environment()
        env.filters["tojson"] = lambda v: __import__("json").dumps(v)
        text = env.from_string(self.chat_template).render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token="<|begin_of_text|>",
            raise_exception=lambda m: (_ for _ in ()).throw(ValueError(m)),
            strftime_now=lambda fmt: "2026-09-26",
            **kw,
        )
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in _split_specials(text, self._specials):
            if piece in self._vocab:
                ids.append(self._vocab[piece])
            else:
                for word in piece.split(" "):
                    ids.append(self._vocab.setdefault(word, len(self._vocab) + 1))
        return ids


def _split_specials(text, specials):
    out, buf = [], ""
    i = 0
    while i < len(text):
        for sp in specials:
            if text.startswith(sp, i):
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(sp)
                i += len(sp)
                break
        else:
            buf += text[i]
            i += 1
    if buf:
        out.append(buf)
    return out


def wrapper(specials, template_name):
    template = (TEMPLATES / template_name).read_text()
    hf = FakeHF(specials, template)
    parser = tu._infer_tool_parser(hf)
    module = harmony if parser == "harmony" else atem
    w = tu.TokenizerWrapper(
        hf,
        tool_parser=module.parse_tool_call,
        tool_call_start=module.tool_call_start,
        tool_call_end=module.tool_call_end,
    )
    w._tool_parser_type = parser
    return w


def test_harmony_is_inferred_from_vocab_and_template():
    t = wrapper(HARMONY_SPECIALS, "gpt-oss-20b.jinja")
    assert t.think_family == "harmony" and t.tool_parser_type == "harmony"
    assert t.think_start == "<|channel|>" and t.think_end == "<|end|>"
    assert t.think_openers == ("<|channel|>", " to=")
    assert t.think_label_end == "<|message|>"
    assert t.tool_call_via_label and t.tool_call_end == "<|call|>"
    assert t.structural_markers == ("<|start|>assistant", "<|end|>")
    close = t._tokenizer.encode("<|end|><|start|>assistant<|channel|>final<|message|>")
    assert t.think_close_tokens == tuple(close)
    assert t.answer_opener_tokens == tuple(
        t._tokenizer.encode("<|channel|>final<|message|>")
    )
    assert t.frame_start == "<|start|>assistant"
    assert t.reasoning_label_tokens == tuple(t._tokenizer.encode("analysis"))
    assert t.think_label_end_tokens == tuple(t._tokenizer.encode("<|message|>"))


def test_muse_is_inferred_from_vocab_and_template():
    t = wrapper(MUSE_SPECIALS, "muse-glimmer-30b.jinja")
    assert t.think_family == "muse" and t.tool_parser_type == "atem"
    assert t.think_start == " to=" and t.think_end == "<|eom|>"
    assert t.think_label_end == "<|message|>" and t.think_openers == (" to=",)
    assert t.tool_call_via_label and t.tool_call_end == "</atem:function_calls>"
    assert t.answer_opener_tokens == tuple(t._tokenizer.encode("<|message|>"))
    assert t.frame_start == "<|start|>assistant"
    # This vocabulary keeps `to=self` in one token: the opener's token form
    # is the whole pair, no label follows (see MergingHF below).
    assert t.think_start_tokens == tuple(t._tokenizer.encode(" to=self"))
    assert t.reasoning_label_tokens == ()


class MergingHF(FakeHF):
    """A vocabulary that merges the opener's tail into the label, as Muse's
    does (` to=self` is ` to`, `=self`; ` to=` alone is ` to`, `=`)."""

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in _split_specials(text, self._specials):
            if piece in self._vocab:
                ids.append(self._vocab[piece])
                continue
            for word in re.findall(r" to(?==)|=\w+|=|\S+|\s+", piece):
                ids.append(self._vocab.setdefault(word, len(self._vocab) + 1))
        return ids


def test_muse_opener_tokens_follow_the_vocabulary():
    """The budget matches tokens: with the label merged into the opener's
    last token, the opener's token form is the pair and no label follows."""
    hf = MergingHF(MUSE_SPECIALS, (TEMPLATES / "muse-glimmer-30b.jinja").read_text())
    assert hf.encode(" to=self") != hf.encode(" to=") + hf.encode("self")
    t = tu.TokenizerWrapper(hf, tool_parser=atem.parse_tool_call,
                            tool_call_start=atem.tool_call_start,
                            tool_call_end=atem.tool_call_end)  # fmt: skip
    assert t.think_start == " to=" and t.think_start_tokens == tuple(
        hf.encode(" to=self")
    )
    assert t.reasoning_label_tokens == ()
    # An answer's recipient is not the reasoning's opener.
    assert (
        t.rfind_think_start(hf.encode("<|start|>assistant to=user<|message|>x")) == -1
    )
    prompt = hf.encode(
        "<|start|>assistant to=self<|message|>x<|eom|><|start|>assistant"
    )
    assert initial_state(t, prompt) == ("frame", "")


# -- the stream ---------------------------------------------------------------


class ScriptTokenizer:
    """A tokenizer whose token i decodes to piece i of the script: the text
    the model would produce, cut where the test wants token boundaries."""

    has_chat_template = True
    has_thinking = True
    has_tool_calling = True
    structural_markers = ()
    chat_template = ""

    def __init__(self, wrapped, pieces):
        self._pieces = [*pieces, ""]  # the last id is the stop token
        for name in (
            "think_family",
            "think_start",
            "think_end",
            "think_openers",
            "think_label_end",
            "tool_call_start",
            "tool_call_end",
            "tool_call_via_label",
            "tool_parser",
            "structural_markers",
            "frame_start",
            "answer_opener_tokens",
        ):
            setattr(self, name, getattr(wrapped, name))
        self.eos_token_ids = set()

    @property
    def detokenizer(self):
        return StubDetokenizer(self._pieces)

    def decode(self, ids):
        return "".join(self._pieces[i] for i in ids)

    def rfind_think_start(self, tokens, start=None, end=None):
        return -1

    def rfind_think_end(self, tokens, start=None, end=None):
        return -1


def run(tok, pieces, tools=None, tools_enabled=True):
    asm = TextAssembler(
        tok, prompt_tokens=[], tools=tools, streaming=False, tools_enabled=tools_enabled
    )
    content = reasoning = ""
    calls = []
    # The stop token (the last piece, an EOS) is never text.
    for i in range(len(pieces) + 1):
        d = asm.feed(
            TokenEvent(i, -0.1, finish_reason="stop" if i == len(pieces) else None)
        )
        content += d.content
        reasoning += d.reasoning
        calls += d.tool_calls
    return content, reasoning, calls, asm


HARMONY = wrapper(HARMONY_SPECIALS, "gpt-oss-20b.jinja")
MUSE = wrapper(MUSE_SPECIALS, "muse-glimmer-30b.jinja")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]


def test_harmony_stream_routes_analysis_final_and_tool_calls():
    pieces = [
        "<|channel|>", "analysis", "<|message|>", "The user ", "asks.", "<|end|>",
        "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "Hello", " there",
    ]  # fmt: skip
    content, reasoning, calls, asm = run(ScriptTokenizer(HARMONY, pieces), pieces)
    assert reasoning == "The user asks." and content == "Hello there" and not calls
    # The channel's label counts (as Gemma's does), the answer's does not.
    assert asm.reasoning_tokens == 5

    pieces = [
        "<|channel|>", "analysis", "<|message|>", "call it", "<|end|>", "<|start|>",
        "assistant", "<|channel|>", "commentary", " to=", "functions.get_weather",
        " ", "<|constrain|>", "json", "<|message|>", '{"city":', ' "Berlin"}', "<|call|>",
    ]  # fmt: skip
    content, reasoning, calls, _ = run(ScriptTokenizer(HARMONY, pieces), pieces, TOOLS)
    assert reasoning == "call it" and content == ""
    assert [c["function"]["name"] for c in calls] == ["get_weather"]
    assert calls[0]["function"]["arguments"] == '{"city": "Berlin"}'

    # The recipient before the channel, as the template renders history.
    pieces = [
        "<|start|>", "assistant", " to=", "functions.get_weather", "<|channel|>",
        "commentary", " json", "<|message|>", '{"city": "Oslo"}', "<|call|>",
    ]  # fmt: skip
    content, reasoning, calls, _ = run(ScriptTokenizer(HARMONY, pieces), pieces, TOOLS)
    assert calls[0]["function"]["arguments"] == '{"city": "Oslo"}' and content == ""

    # A commentary without a recipient is a visible preamble.
    pieces = ["<|channel|>", "commentary", "<|message|>", "Let me look.", "<|end|>"]
    content, reasoning, calls, _ = run(ScriptTokenizer(HARMONY, pieces), pieces, TOOLS)
    assert content == "Let me look." and reasoning == "" and not calls


def test_muse_stream_routes_self_user_and_a_tool_recipient():
    pieces = [
        " to=", "self", "<|message|>", "plan ", "first", "<|eom|>", "<|start|>",
        "assistant", "<|message|>", "Done.",
    ]  # fmt: skip
    content, reasoning, calls, asm = run(ScriptTokenizer(MUSE, pieces), pieces)
    assert reasoning == "plan first" and content == "Done." and not calls
    assert asm.reasoning_tokens == 5

    block = (
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Berlin</atem:parameter>\n'
        '<atem:parameter name="days">3</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    pieces = [
        " to=",
        "get_weather",
        "<|message|>",
        block[:20],
        block[20:60],
        block[60:],
    ]
    content, reasoning, calls, _ = run(ScriptTokenizer(MUSE, pieces), pieces, TOOLS)
    assert content == "" and reasoning == ""
    assert calls[0]["function"]["name"] == "get_weather"
    assert calls[0]["function"]["arguments"] == '{"city": "Berlin", "days": 3}'

    # Tools off: the recipient's text is the model's, shown as it is.
    content, _, calls, _ = run(
        ScriptTokenizer(MUSE, pieces), pieces, TOOLS, tools_enabled=False
    )
    assert not calls and content.startswith("<atem:function_calls>")


def test_parsers_refuse_what_is_not_a_call():
    with pytest.raises(ValueError):
        harmony.parse_tool_call("commentary<|message|>just text")
    with pytest.raises(ValueError):
        harmony.parse_tool_call("commentary to=functions.x<|message|>[1, 2]")
    with pytest.raises(ValueError):
        atem.parse_tool_call("no block here")
    out = atem.parse_tool_call(
        '<atem:invoke name="f">\n<atem:parameter name="v">{"a": 1}</atem:parameter>\n'
        '<atem:parameter name="t">true</atem:parameter>\n</atem:invoke>'
    )
    assert out == {"name": "f", "arguments": {"v": {"a": 1}, "t": True}}


def test_thinking_off_opens_the_answer_in_the_prompt():
    for tok, opener in (
        (HARMONY, "<|channel|>final<|message|>"),
        (MUSE, "<|message|>"),
    ):
        req = chat.parse_chat_request(
            {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False},
            "m",
        )
        tokens = chat.build_prompt(tok, req)
        tail = tok._tokenizer.encode(opener)
        assert tokens[-len(tail) :] == tail
        assert initial_state(tok, tokens) == ("normal", "")
        on = chat.parse_chat_request(
            {"messages": [{"role": "user", "content": "hi"}]}, "m"
        )
        assert chat.build_prompt(tok, on)[-len(tail) :] != tail
        # With tools on offer the opener stays out: a call needs the channel
        # or recipient it would skip past, so the frame is left to the model.
        with_tools = chat.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "enable_thinking": False,
                "tools": TOOLS,
            },
            "m",
        )
        tokens = chat.build_prompt(tok, with_tools)
        assert tokens[-len(tail) :] != tail
        assert initial_state(tok, tokens) == ("frame", "")


def test_thinking_off_with_tools_is_a_zero_budget():
    """Without the opener in the prompt the block is kept out by the
    budget: zero masks the reasoning label behind the shared opener
    (Harmony's `analysis`, Muse's `=self`), the answer's and the tools'
    channels stay open. Without tools the opener does the job and no
    budget is set."""
    for tok in (HARMONY, MUSE):
        req = chat.parse_chat_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "enable_thinking": False,
                "tools": TOOLS,
            },
            "m",
        )
        gen = chat.to_generation_request(tok, req)
        assert gen.reasoning is not None and gen.reasoning.max_tokens == 0
        tracker = ThinkingBudget(gen.reasoning)
        # Harmony: the label after the shared marker; Muse: the opener's own
        # last token (` to` + `=self`), so ` to=<tool>` stays free.
        label = tuple(getattr(tok, "reasoning_label_tokens", None) or ())[:1]
        assert tracker._block and tracker._cut == tuple(tok.think_start_tokens) + label
        without = chat.parse_chat_request(
            {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False},
            "m",
        )
        assert chat.to_generation_request(tok, without).reasoning.max_tokens is None


def test_developer_is_rendered_as_the_template_takes_it():
    """gpt-oss's template names the role and renders it like system: native;
    a template that writes any role into the frame never saw the word,
    so the message goes in as system."""
    from jinja2 import Environment

    from mlx_beam.api import roles

    assert roles.developer_rendering(HARMONY) == "native"

    class Generic:
        chat_template = (
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
        )

        def apply_chat_template(
            self, messages, add_generation_prompt=True, tokenize=True, **kw
        ):
            text = (
                Environment().from_string(self.chat_template).render(messages=messages)
            )
            return text.split("<|")[1:] if tokenize else text

    generic = Generic()
    assert roles.developer_rendering(generic) == "as system"
    dev = [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]
    assert roles.for_template(generic, dev)[0]["role"] == "system"
    assert roles.for_template(HARMONY, dev)[0]["role"] == "developer"


def test_budget_forces_the_whole_close_sequence():
    """Harmony closes the analysis channel by opening the final one: six
    tokens the queue forces after the free token, all flagged."""
    END, START, ASSISTANT, CHANNEL, FINAL, MESSAGE = 40, 41, 42, 43, 44, 45
    limits = ReasoningLimits(
        start=(CHANNEL, 46, MESSAGE),
        end=(END,),
        close=(END, START, ASSISTANT, CHANNEL, FINAL, MESSAGE),
        seeded=True,
        max_tokens=4,
    )
    tracker = ThinkingBudget(limits)
    steps = Steps(tracker, list(range(10, 30)))
    out = steps.run(13)
    # Four counted tokens (the end marker leaves the block, the rest of the
    # close is the answer's opener), then the forced six.
    assert out[:10] == [10, 11, 12, 13, END, START, ASSISTANT, CHANNEL, FINAL, MESSAGE]
    assert tracker.reasoning_tokens == 4 and tracker.thinking_truncated
    assert steps.forced[:10] == [False] * 4 + [True] * 6
    assert not any(steps.forced[10:]) and not tracker.in_reasoning


def test_effort_is_passed_through_by_templates_that_do_not_check():
    """gpt-oss writes `Reasoning: <word>` into the system prompt, Muse
    `Reasoning strength: <word>` from another kwarg; neither refuses a
    word, so the client's word goes in unchanged and no set is claimed."""
    cap = effort_capability(HARMONY)
    assert cap.describe() == {
        "source": "template",
        "kwarg": "reasoning_effort",
        "validates": False,
    }
    cap = effort_capability(MUSE)
    assert cap.kwarg == "reasoning_strength" and not cap.validates
    messages = [{"role": "user", "content": "hi"}]
    for tok, kwarg, line in (
        (HARMONY, "reasoning_effort", "Reasoning: max\n"),
        (MUSE, "reasoning_strength", "Reasoning strength: max."),
    ):
        req = chat.parse_chat_request(
            {"messages": messages, "reasoning_effort": "max"}, "m"
        )
        chat.build_prompt(tok, req)
        assert req.template_kwargs[kwarg] == "max"
        text = tok.apply_chat_template(messages, tokenize=False, **req.template_kwargs)
        assert line in text


def test_a_recipient_marker_inside_the_answer_is_text():
    """` to=` opens a label only at the message's frame - right after
    `<|start|>assistant` - where the recipient goes. In the answer it is
    what the model wrote (`send(msg, to=addr)`), not a label that swallows
    the rest of the reply."""
    pieces = [
        "<|channel|>", "final", "<|message|>", "Use send(msg,", " to=", "addr)",
        " and go.",
    ]  # fmt: skip
    content, reasoning, calls, _ = run(ScriptTokenizer(HARMONY, pieces), pieces)
    assert content == "Use send(msg, to=addr) and go." and reasoning == "" and not calls
    pieces = [
        " to=", "user", "<|message|>", "Call send(x,", " to=", "y).",
    ]  # fmt: skip
    content, reasoning, calls, _ = run(ScriptTokenizer(MUSE, pieces), pieces)
    assert content == "Call send(x, to=y)." and reasoning == "" and not calls
    # At the frame the same text is the recipient, as before.
    pieces = [
        " to=", "self", "<|message|>", "think", "<|eom|>", "<|start|>", "assistant",
        "<|message|>", "answer", " to=", "x",
    ]  # fmt: skip
    content, reasoning, calls, _ = run(ScriptTokenizer(MUSE, pieces), pieces)
    assert reasoning == "think" and content == "answer to=x"


def test_muse_opener_survives_the_trimmed_leading_space():
    """The detokenizer drops the space a sequence starts with, so the first
    recipient arrives as `to=self`; it is the frame's opener all the same,
    and the reasoning goes where it belongs instead of into the answer."""
    pieces = [
        "to=", "self", "<|message|>", "plan ", "first", "<|eom|>", "<|start|>",
        "assistant", " to=", "user", "<|message|>", "Done.",
    ]  # fmt: skip
    content, reasoning, calls, asm = run(ScriptTokenizer(MUSE, pieces), pieces)
    assert reasoning == "plan first" and content == "Done." and not calls
    # The same for a tool named at the frame without the space.
    block = (
        '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Kiel</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    pieces = ["to=", "get_weather", "<|message|>", block[:30], block[30:]]
    content, reasoning, calls, _ = run(ScriptTokenizer(MUSE, pieces), pieces, TOOLS)
    assert reasoning == "" and content == ""
    assert calls[0]["function"]["arguments"] == '{"city": "Kiel"}'
