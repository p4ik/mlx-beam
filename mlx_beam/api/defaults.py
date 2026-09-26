"""Server-side defaults for what a request may leave out.

Precedence, highest first: the request, a `beam serve` flag, the model's
``generation_config.json``, mlx-lm's own defaults. ``sources`` records
which of the four each value came from, for the start banner and /health.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

# mlx-lm's defaults (sample_utils.make_sampler, server.py at the pin).
MLX_LM_DEFAULTS = {
    "max_completion_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": None,
    "presence_penalty": None,
    "frequency_penalty": None,
    "chat_template_args": {},
}

# Ours, with no mlx-lm counterpart: no reasoning budget, no reserve.
BEAM_DEFAULTS = {"max_reasoning_tokens": None, "min_response_tokens": 0}

# What a request may ask for; a default outside this is refused at start,
# not on every request that leaves the field empty.
RANGES = {
    "temperature": (0.0, 2.0),
    "top_p": (0.0, 1.0),
    "min_p": (0.0, 1.0),
    "top_k": (-1, None),
    "repetition_penalty": (0.0, None),
    "presence_penalty": (-2.0, 2.0),
    "frequency_penalty": (-2.0, 2.0),
    "max_completion_tokens": (1, None),
    "max_reasoning_tokens": (0, None),
    "min_response_tokens": (0, None),
}
# Bounded below without the bound itself (a request refuses a penalty of 0).
POSITIVE = frozenset({"repetition_penalty"})

# The generation_config.json keys we read, and what they map to.
GENERATION_CONFIG_KEYS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


@dataclass
class RequestDefaults:
    max_completion_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Handed to every chat template render; a request's chat_template_kwargs
    # override it key by key.
    chat_template_args: dict = field(default_factory=dict)
    # Reasoning tokens a request may spend when it names no budget (None:
    # unbounded) and the room the answer keeps after the think block.
    max_reasoning_tokens: int | None = None
    min_response_tokens: int = 0
    sources: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # A bare instance carries the built-in numbers, so say so.
        for f in fields(self):
            if f.name != "sources":
                self.sources.setdefault(
                    f.name, "mlx-beam" if f.name in BEAM_DEFAULTS else "mlx-lm"
                )

    @classmethod
    def resolve(
        cls, model_path: str | Path | None = None, flags: dict | None = None
    ) -> RequestDefaults:
        """Flags beat the model's generation_config.json, which beats mlx-lm."""
        values = {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in MLX_LM_DEFAULTS.items()
        }
        sources = {k: "mlx-lm" for k in values}
        values.update(BEAM_DEFAULTS)
        sources.update({k: "mlx-beam" for k in BEAM_DEFAULTS})
        cfg = read_generation_config(model_path) if model_path else {}
        for key, ours in GENERATION_CONFIG_KEYS.items():
            if key in cfg and cfg[key] is not None:
                values[ours] = cfg[key]
                sources[ours] = "generation_config.json"
        if cfg.get("do_sample") is False:
            # Greedy by the model author's word; temperature 0 is greedy here.
            values["temperature"] = 0.0
            sources["temperature"] = "generation_config.json"
        for key, value in (flags or {}).items():
            if value is not None:
                values[key] = value
                sources[key] = "flag"
        for key, (lo, hi) in RANGES.items():
            value = values.get(key)
            if value is None:
                continue
            # The request parser's rules (chat.py): a finite number in the
            # range, and a penalty above zero - a default a request could
            # not ask for would fail every request that leaves the field.
            bad = (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            )
            if not bad and (
                (lo is not None and value < lo)
                or (hi is not None and value > hi)
                or (key in POSITIVE and value <= lo)
            ):
                bad = True
            if bad:
                raise ValueError(
                    f"{key} from {sources[key]} is {value!r}; a request may use "
                    f"{'above ' if key in POSITIVE else ''}{lo} to "
                    f"{hi if hi is not None else 'any'}"
                )
        out = cls(**{k: values[k] for k in values})
        out.sources = sources
        return out

    def describe(self) -> dict:
        return {
            f.name: {"value": getattr(self, f.name), "source": self.sources.get(f.name)}
            for f in fields(self)
            if f.name != "sources"
        }


def read_generation_config(model_path: str | Path) -> dict:
    path = Path(model_path) / "generation_config.json"
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
