"""The tower's tensors from the checkpoint, the way any loader that follows
the index finds them: `model.safetensors.index.json` names the shard of
every tensor, subfolders included (a package keeps the tower in its own
file). Only the tensors under the tower's prefix are read; the family's
own `sanitize` puts convolutions into MLX's layout. An index that belongs
to another conversion of the model is dropped, and the shards present are
read by the tensor names their headers list."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger("beam.vision")

INDEX = "model.safetensors.index.json"
SHARDS = "model*.safetensors"
"""The files scanned when the index is dropped: the checkpoint's root
only, as the text loader reads it."""


def shards_for(model_path: Path, prefixes: Iterable[str]) -> dict[str, list[str]]:
    """From the index: shard file -> the tensor names under `prefixes` it
    holds, or the single `model.safetensors` with no names (every tensor
    under the prefixes is read from it)."""
    index = model_path / INDEX
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


def header_names(path: Path) -> list[str]:
    """The tensor names a safetensors file lists in its header: eight bytes
    of header length, then JSON with one key per tensor (and `__metadata__`).
    No tensor is read."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(min(n, size)))
    except (OSError, ValueError) as e:
        raise ValueError(f"{path}: not a readable safetensors header ({e})") from e
    return [k for k in header if k != "__metadata__"]


def scanned_shards(model_path: Path, prefixes: tuple[str, ...]) -> dict[str, list[str]]:
    """shard file -> the tensor names under `prefixes` its header lists,
    over the `model*.safetensors` in the checkpoint's root; files with
    none are left out."""
    out: dict[str, list[str]] = {}
    for path in sorted(model_path.glob(SHARDS)):
        names = [k for k in header_names(path) if k.startswith(prefixes)]
        if names:
            out[path.name] = names
    return out


def _read(
    model_path: Path, shards: dict[str, list[str]], prefixes: tuple[str, ...]
) -> tuple[dict[str, mx.array], list[str]]:
    out: dict[str, mx.array] = {}
    read: list[str] = []
    for shard, names in shards.items():
        raw = mx.load(str(model_path / shard))
        wanted = set(names) if names else {k for k in raw if k.startswith(prefixes)}
        found = {k: v for k, v in raw.items() if k in wanted}
        if found:
            out.update(found)
            read.append(shard)
    return out, sorted(read)


def load_prefixed(
    model_path: Path, prefixes: Iterable[str]
) -> tuple[dict[str, mx.array], list[str], str]:
    """The tensors whose names start with one of `prefixes`, names as in
    the checkpoint, the shard files they were read from, and how those
    were chosen: `followed` (the index), `none` (no index, the single
    file) or `stale` - the index names a shard that is not there, or no
    tensor it names under the prefixes is in the shard it points at while
    a header of `model*.safetensors` lists some; then the index is dropped
    whole and the files whose headers hold such tensors are read. What
    the tensors read must add up to is the module's business (strict
    loading); an index that is only partly right is followed as it is."""
    prefixes = tuple(prefixes)
    by_index = shards_for(model_path, prefixes)
    if not (model_path / INDEX).is_file():
        out, shards = _read(model_path, by_index, prefixes)
        return out, shards, "none"
    missing = [s for s in by_index if not (model_path / s).is_file()]
    if not missing:
        out, shards = _read(model_path, by_index, prefixes)
        if out:
            return out, shards, "followed"
    scanned = scanned_shards(model_path, prefixes)
    if missing:
        why = f"names {missing[0]}, which is not there"
    elif not scanned:
        # Not stale: a valid index for a checkpoint that has no tower.
        return {}, [], "followed"
    else:
        why = "names no tensor under the prefixes that is in the shard it points at"
    if not scanned:
        raise FileNotFoundError(
            f"{model_path}: the index {why}, so it was dropped, and no {SHARDS} "
            f"file there lists a tensor under {prefixes} in its header"
        )
    logger.warning(
        "%s: %s %s; dropped, the tensors under %s read from %s",
        model_path,
        INDEX,
        why,
        prefixes,
        ", ".join(scanned),
    )
    out, shards = _read(model_path, scanned, prefixes)
    return out, shards, "stale"
