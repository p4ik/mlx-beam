"""The tower's tensors from the checkpoint, the way any loader that follows
the index finds them: `model.safetensors.index.json` names the shard of
every tensor, subfolders included (a package keeps the tower in its own
file). Only the tensors under the tower's prefix are read; the family's
own `sanitize` puts convolutions into MLX's layout."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx


def shards_for(model_path: Path, prefixes: Iterable[str]) -> dict[str, list[str]]:
    """shard file -> the tensor names under `prefixes` it holds."""
    index = model_path / "model.safetensors.index.json"
    prefixes = tuple(prefixes)
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        out: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            if name.startswith(prefixes):
                out.setdefault(shard, []).append(name)
        return out
    single = model_path / "model.safetensors"
    if single.is_file():
        return {single.name: []}  # every tensor under the prefix, read below
    raise FileNotFoundError(
        f"{model_path} has no safetensors index and no model.safetensors"
    )


def load_prefixed(model_path: Path, prefixes: Iterable[str]) -> dict[str, mx.array]:
    """The tensors whose names start with one of `prefixes`, names as in
    the checkpoint."""
    prefixes = tuple(prefixes)
    out: dict[str, mx.array] = {}
    for shard, names in shards_for(model_path, prefixes).items():
        raw = mx.load(str(model_path / shard))
        wanted = set(names) if names else {k for k in raw if k.startswith(prefixes)}
        out.update({k: v for k, v in raw.items() if k in wanted})
    return out


def strip_prefix(weights: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
