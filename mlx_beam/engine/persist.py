"""The prefix store on disk: written on a clean stop, read back after start."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from time import monotonic, sleep  # noqa: F401 - module names, so tests feed a clock
from typing import Any

import mlx.core as mx

from mlx_beam.engine.kv import KVPolicy
from mlx_beam.engine.prefix import Entry, PrefixStore

FORMAT = 1
# How long a start waits for a namespace held by another process, and the
# poll interval (placeholder: not measured; an evictor holds a lock for one
# directory listing and a few unlinks).
LOCK_WAIT_S = 5.0
LOCK_POLL_S = 0.1


class IncompatibleEntry(Exception):
    """The entry was written for another format, layout or vision identity."""


class PartialEntry(Exception):
    """The entry's file is missing or was never published."""


class CorruptEntry(Exception):
    """The entry's file is truncated or unreadable."""


class NamespaceLocked(RuntimeError):
    """Another process holds the namespace."""


def namespace_parts(identity: dict, policy: KVPolicy, applied: list | None) -> dict:
    kv = policy.describe()
    kv.pop("prefill_source", None)
    return {
        "format": FORMAT,
        "model": identity["model"],
        "mlx": identity["mlx"],
        "engine": identity["engine"],
        "kv": {"policy": kv, "applied": applied},
    }


def namespace(parts: dict) -> str:
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def key_sha(key: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(key)}q", *key)).hexdigest()


@dataclass
class SweepReport:
    waited_ms: int = 0
    skipped_deadline: bool = False
    evicted: int = 0
    evicted_bytes: int = 0
    total_before: int = 0
    total_after: int = 0
    own_bytes: int = 0
    other_namespaces: int = 0


@dataclass
class SaveReport:
    entries: int = 0
    bytes: int = 0
    ms: int = 0
    skipped_entries: int = 0
    skipped_no_space: int = 0
    skipped_over_cap: int = 0
    skipped_unsupported: int = 0
    failures: int = 0
    disk_bytes: int = 0
    sweep: SweepReport | None = None


@dataclass
class LoadReport:
    entries: int = 0
    bytes: int = 0
    ms: int = 0
    incompatible: int = 0
    partial: int = 0
    corrupt: int = 0
    over_budget: int = 0
    sidecars_dropped: int = 0
    failures: int = 0
    disk_bytes: int = 0
    sweep: SweepReport | None = None


def entry_to_tensors(
    key: list[int], entry: Entry, head_digest: str | None = None, vision: Any = None
) -> tuple[dict[str, mx.array], dict[str, str]]:
    return {}, {}


def entry_from_tensors(
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
    head_digest: str | None = None,
) -> tuple[list[int], Entry]:
    return None, None


def write_entry(
    path: str, key: list[int], entry: Entry, head_digest: str | None = None, vision=None
) -> int:
    return 0


def read_entry(path: str, expected: dict) -> tuple[list[int], Entry]:
    return None, None


def write_manifest(ns_dir: str, manifest: dict) -> None:
    return None


def read_manifest(ns_dir: str) -> dict | None:
    return None


def lock_namespace(dir: str, ns: str) -> int:
    return -1


def unlock_namespace(fd: int) -> None:
    return None


def sweep(dir: str, own_ns: str, cap_bytes: int, deadline: float, log) -> SweepReport:
    return SweepReport()


def save_store(
    store: PrefixStore,
    dir: str,
    ns: str,
    parts: dict,
    budget_s: float,
    cap_bytes: int,
    log,
    *,
    head_digest: str | None = None,
    vision: Any = None,
) -> SaveReport:
    return SaveReport()


def load_store(
    store: PrefixStore,
    dir: str,
    ns: str,
    parts: dict,
    model_key: Any,
    head_digest: str | None,
    log,
    *,
    vision: Any = None,
) -> LoadReport:
    return LoadReport()


__all__ = [
    "FORMAT",
    "LOCK_WAIT_S",
    "CorruptEntry",
    "IncompatibleEntry",
    "LoadReport",
    "NamespaceLocked",
    "PartialEntry",
    "SaveReport",
    "SweepReport",
    "entry_from_tensors",
    "entry_to_tensors",
    "key_sha",
    "load_store",
    "lock_namespace",
    "namespace",
    "namespace_parts",
    "read_entry",
    "read_manifest",
    "save_store",
    "sweep",
    "unlock_namespace",
    "write_entry",
    "write_manifest",
]
# Unused in the hull; the fields exist so a report can be built by name.
_ = field
