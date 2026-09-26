"""Message roles: which ones the wire formats admit, and how a template
renders `developer` - measured on the template, not assumed."""

from __future__ import annotations

from collections.abc import Sequence

from mlx_beam.api.errors import ApiError

# OpenAI's chat roles; `developer` is the newer name for `system`.
CHAT_ROLES = ("system", "developer", "user", "assistant", "tool")

_PROBE = [{"role": "developer", "content": "probe"}, {"role": "user", "content": "hi"}]


def check_role(role, where: str, allowed: Sequence[str] = CHAT_ROLES) -> str:
    """The role, or a 400 naming the ones the format admits: a role the
    format does not know is the client's error, whatever a template would
    make of it."""
    if role not in allowed:
        raise ApiError(
            f"{where}.role must be one of {', '.join(allowed)}, not {role!r}",
            param="messages",
        )
    return role


def probe_developer(tokenizer) -> str:
    """How this template takes a `developer` message: "native" when it
    renders it as its own thing (gpt-oss does), "as system" when it
    refuses the role or only renders it the way it renders any unknown
    role (a Qwen template writes the word into the frame; the model never
    saw it). The verdict is the template's source and its rendering, not a
    table."""
    render = getattr(tokenizer, "apply_chat_template", None)
    if render is None:
        return "as system"
    try:
        as_developer = render(_PROBE, add_generation_prompt=True, tokenize=False)
    except Exception:  # noqa: BLE001 - the template's refusal is the verdict
        return "as system"
    as_system = render(
        [{"role": "system", "content": "probe"}, _PROBE[1]],
        add_generation_prompt=True,
        tokenize=False,
    )
    if as_developer == as_system:
        return "native"
    from mlx_beam.api.reasoning import _template_source

    return "native" if "developer" in _template_source(tokenizer) else "as system"


def developer_rendering(tokenizer) -> str:
    """The probe's result, kept on the tokenizer after the first call."""
    cached = getattr(tokenizer, "_developer_rendering", None)
    if cached is None:
        cached = probe_developer(tokenizer)
        try:
            tokenizer._developer_rendering = cached
        except AttributeError:
            pass
    return cached


def for_template(tokenizer, messages: list[dict]) -> list[dict]:
    """The messages as the template should see them: `developer` stays
    where the template knows it, becomes `system` where it does not."""
    if developer_rendering(tokenizer) == "native":
        return messages
    return [
        {**m, "role": "system"} if m.get("role") == "developer" else m for m in messages
    ]


def describe(tokenizer) -> dict:
    return {"developer": developer_rendering(tokenizer)}
