"""The prefix store on disk: written on a clean stop, read back after start.

One folder per namespace - the model's files, the KV layout, the format
and the mlx and mlx-beam versions, hashed - holds a manifest and one
safetensors file per entry. A published file is never changed: a key that
comes back with other content gets a new generation in its name, the
manifest turns last, only then do the old files go. Anything the manifest
does not name, or that is cut short or of another shape, is a miss and is
removed. A process holds its namespace with a lock for its lifetime; the
folder as a whole is capped, and whole foreign namespaces go first, the
least recently saved first, never one a process holds.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass
from time import monotonic, sleep  # module names, so a test can feed a clock
from typing import Any

import mlx.core as mx

from mlx_beam._vendor.mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx_beam._vendor.optiq.kv_batch import MergeableQuantizedKVCache
from mlx_beam.engine.kv import KVPolicy
from mlx_beam.engine.prefix import HEAD, Entry, PrefixStore

FORMAT = 1
# How long a start waits for a namespace or the budget lock held by another
# process, polled every LOCK_POLL_S (placeholder: not measured; an evictor
# holds a lock for one directory listing and a few unlinks).
LOCK_WAIT_S = 5.0
LOCK_POLL_S = 0.1
MANIFEST = "manifest.json"
LOCK = "lock"
BUDGET_LOCK = "budget.lock"
# Written and read back as they are; another class is skipped at the save
# and incompatible at the load.
CLASSES = {
    c.__name__: c
    for c in (KVCache, MergeableQuantizedKVCache, ArraysCache, RotatingKVCache)
}
# The fields of each class's ``state`` tuple, in its order.
FIELDS = {
    "KVCache": ("keys", "values", "offset"),
    "MergeableQuantizedKVCache": ("keys", "values", "offset", "group_size", "bits"),
    "ArraysCache": ("cache", "left_padding", "lengths"),
    "RotatingKVCache": ("keys", "values", "offset", "keep", "max_size", "_idx"),
}


class IncompatibleEntry(Exception):
    """The entry was written for another format, layout or vision identity."""


class PartialEntry(Exception):
    """The entry's file is missing or was never published."""


class CorruptEntry(Exception):
    """The entry's file is truncated or unreadable."""


class NamespaceLocked(RuntimeError):
    """Another process holds the namespace."""


def namespace_parts(identity: dict, policy: KVPolicy, applied: list | None) -> dict:
    """What a namespace is hashed from; the manifest holds it readable."""
    kv = policy.describe()
    # Where the prefill mode came from is no layout.
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


# -- one entry as tensors and metadata ---------------------------------------


def _put(arrays: dict, name: str, value: Any):
    """Describe `value` for the metadata; arrays go into `arrays` under
    `name`, a tuple or list as its members under `name.k`."""
    if value is None:
        return None
    if isinstance(value, mx.array):
        arrays[name] = value
        return "a"
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (tuple, list)):
        kind = "t" if isinstance(value, tuple) else "l"
        return {kind: [_put(arrays, f"{name}.{k}", v) for k, v in enumerate(value)]}
    raise TypeError(f"{name}: a {type(value).__name__} cannot be stored")


def _get(arrays: dict, name: str, spec: Any):
    if spec is None:
        return None
    if spec == "a":
        return arrays[name]
    if isinstance(spec, dict):
        ((kind, items),) = spec.items()
        values = [_get(arrays, f"{name}.{k}", s) for k, s in enumerate(items)]
        return tuple(values) if kind == "t" else values
    return spec


def _supported(cache: list) -> list[str]:
    """Class names in `cache` the format does not carry."""
    return sorted(
        {
            type(c).__name__
            for c in cache
            if CLASSES.get(type(c).__name__) is not type(c)
        }
    )


def _window(classes: list[str], layer: int) -> str:
    return ".win" if classes[layer] == "RotatingKVCache" else ""


def entry_to_tensors(
    key: list[int], entry: Entry, head_digest: str | None = None, vision: Any = None
) -> tuple[dict[str, mx.array], dict[str, str]]:
    """The entry as safetensors arrays and string metadata. The head's
    sidecar goes only with a digest to check it against; the vision
    identity only with an image id in the key."""
    unsupported = _supported(entry.cache)
    if unsupported:
        raise TypeError(f"cache classes not stored: {', '.join(unsupported)}")
    arrays: dict[str, mx.array] = {"key": mx.array(key, dtype=mx.int64)}
    classes = [type(c).__name__ for c in entry.cache]
    scalars = {}
    for i, c in enumerate(entry.cache):
        fields = dict(zip(FIELDS[classes[i]], c.state, strict=True))
        scalars[str(i)] = {
            f: _put(arrays, f"layer.{i}.{f}", v) for f, v in fields.items()
        }
    checkpoints = {}
    with_head = False
    for pos, snap in entry.checkpoints.items():
        layers = {}
        for layer, value in snap.items():
            if layer == HEAD:
                if head_digest is None:
                    continue
                layers["head"] = _put(arrays, f"cp.{pos}.head", value)
                with_head = True
            else:
                name = f"cp.{pos}.{layer}{_window(classes, layer)}"
                layers[str(layer)] = _put(arrays, name, value)
        checkpoints[str(pos)] = layers
    metadata = {
        "format": str(FORMAT),
        "classes": json.dumps(classes),
        "scalars": json.dumps(scalars),
        "length": str(entry.length),
        "cache_type": entry.cache_type,
        "checkpoints": json.dumps(checkpoints),
    }
    if with_head:
        metadata["head_digest"] = head_digest
    if any(k < 0 for k in key):
        metadata["vision"] = json.dumps(vision, sort_keys=True)
    return arrays, metadata


def entry_from_tensors(
    arrays: dict[str, mx.array],
    metadata: dict[str, str],
    head_digest: str | None = None,
) -> tuple[list[int], Entry]:
    """The entry back; a head sidecar written under another digest, or
    read without one, is left out."""
    if metadata.get("format") != str(FORMAT):
        raise IncompatibleEntry(f"format {metadata.get('format')!r}, this is {FORMAT}")
    key = [int(k) for k in arrays["key"].tolist()]
    classes = json.loads(metadata["classes"])
    scalars = json.loads(metadata["scalars"])
    cache = []
    for i, name in enumerate(classes):
        cls = CLASSES.get(name)
        if cls is None:
            raise IncompatibleEntry(f"layer {i}: cache class {name} is not carried")
        spec = scalars[str(i)]
        state = tuple(_get(arrays, f"layer.{i}.{f}", spec[f]) for f in FIELDS[name])
        cache.append(cls.from_state(state))
    foreign = head_digest is None or metadata.get("head_digest") != head_digest
    checkpoints: dict[int, dict] = {}
    for pos, layers in json.loads(metadata["checkpoints"]).items():
        snap = {}
        for layer, spec in layers.items():
            if layer == "head":
                if not foreign:
                    snap[HEAD] = _get(arrays, f"cp.{pos}.head", spec)
                continue
            i = int(layer)
            snap[i] = _get(arrays, f"cp.{pos}.{i}{_window(classes, i)}", spec)
        checkpoints[int(pos)] = snap
    entry = Entry(cache, int(metadata["length"]), checkpoints, metadata["cache_type"])
    return key, entry


def _dtype(array: mx.array) -> str:
    return str(array.dtype).rsplit(".", 1)[-1]


def tensor_specs(arrays: dict[str, mx.array]) -> dict:
    return {n: {"shape": list(a.shape), "dtype": _dtype(a)} for n, a in arrays.items()}


def _write(path: str, arrays: dict, metadata: dict) -> int:
    """Through a temporary name with the pid, fsync, then into place."""
    directory, name = os.path.split(path)
    stem = name[: -len(".safetensors")] if name.endswith(".safetensors") else name
    tmp = os.path.join(directory, f"{stem}.{os.getpid()}.new.safetensors")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            mx.save_safetensors(f, arrays, metadata)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _unlink(tmp)
        raise
    os.replace(tmp, path)
    return os.stat(path).st_size


def write_entry(
    path: str,
    key: list[int],
    entry: Entry,
    head_digest: str | None = None,
    vision: Any = None,
) -> int:
    """Write the entry to `path` (never a name a manifest holds); the size."""
    arrays, metadata = entry_to_tensors(key, entry, head_digest, vision)
    return _write(path, arrays, metadata)


def read_entry(path: str, expected: dict) -> tuple[list[int], Entry]:
    """The entry in `path`, checked against `expected`: its `bytes` and
    `tensors` from the manifest, the `format`, for an image entry the
    `vision` identity, and the `head_digest` its sidecar must match. Every
    array is read here, so a bad file fails here and not in a request."""
    try:
        size = os.stat(path).st_size
    except FileNotFoundError:
        raise PartialEntry(f"{path}: missing") from None
    want = expected.get("bytes")
    if want is not None and size != want:
        raise CorruptEntry(f"{path}: {size} bytes, the manifest says {want}")
    try:
        arrays, metadata = mx.load(path, return_metadata=True)
    except Exception as e:  # noqa: BLE001 - whatever mlx says, the file is bad
        raise CorruptEntry(f"{path}: {e}") from None
    if metadata.get("format") != str(expected.get("format", FORMAT)):
        raise IncompatibleEntry(f"{path}: format {metadata.get('format')!r}")
    tensors = expected.get("tensors")
    if tensors is not None and tensor_specs(arrays) != tensors:
        raise IncompatibleEntry(f"{path}: tensors differ from the manifest")
    try:
        key = [int(k) for k in arrays["key"].tolist()]
    except Exception as e:  # noqa: BLE001
        raise CorruptEntry(f"{path}: no key ({e})") from None
    if any(k < 0 for k in key):
        written = json.loads(metadata["vision"]) if "vision" in metadata else None
        if written != expected.get("vision"):
            raise IncompatibleEntry(f"{path}: another vision identity")
    try:
        mx.eval(*arrays.values())
    except Exception as e:  # noqa: BLE001
        raise CorruptEntry(f"{path}: {e}") from None
    try:
        key, entry = entry_from_tensors(arrays, metadata, expected.get("head_digest"))
    except IncompatibleEntry:
        raise
    except (KeyError, ValueError, TypeError) as e:
        raise CorruptEntry(f"{path}: {e!r}") from None
    entry.file = os.path.basename(path)
    return key, entry


# -- the manifest and the folder ---------------------------------------------


def read_manifest(ns_dir: str) -> dict | None:
    """The manifest, None when there is none; unreadable is an error."""
    path = os.path.join(ns_dir, MANIFEST)
    try:
        with open(path, "rb") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise ValueError(f"{path} is not a manifest")
    return manifest


def write_manifest(ns_dir: str, manifest: dict) -> None:
    path = os.path.join(ns_dir, MANIFEST)
    tmp = path + ".new"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _unlink(path: str) -> int:
    """Remove the file; its size, or 0 when it was not there."""
    try:
        size = os.stat(path).st_size
        os.unlink(path)
    except FileNotFoundError:
        return 0
    return size


def _file_size(path: str) -> int | None:
    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return None


def _files(ns_dir: str) -> list:
    with os.scandir(ns_dir) as it:
        return [e for e in it if e.is_file(follow_symlinks=False)]


def _namespace_bytes(ns_dir: str) -> int:
    try:
        return sum(e.stat(follow_symlinks=False).st_size for e in _files(ns_dir))
    except FileNotFoundError:
        return 0


def _saved_at(ns_dir: str) -> float | None:
    try:
        with open(os.path.join(ns_dir, MANIFEST), "rb") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    value = manifest.get("saved_at") if isinstance(manifest, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _lock_until(fd: int, deadline: float) -> bool:
    """Try, then look at the clock, then sleep: a free lock is taken even
    when the deadline has passed."""
    while True:
        if _try_lock(fd):
            return True
        if monotonic() >= deadline:
            return False
        sleep(LOCK_POLL_S)


def lock_namespace(dir: str, ns: str) -> int:
    """Create the namespace folder and hold its lock; the fd keeps it."""
    ns_dir = os.path.join(dir, ns)
    os.makedirs(ns_dir, mode=0o700, exist_ok=True)
    fd = os.open(os.path.join(ns_dir, LOCK), os.O_RDWR | os.O_CREAT, 0o600)
    if _lock_until(fd, monotonic() + LOCK_WAIT_S):
        return fd
    os.close(fd)
    raise NamespaceLocked(f"another process holds the prompt cache namespace {ns_dir}")


def unlock_namespace(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _clear_namespace(ns_dir: str) -> int:
    """Data and manifest go; the lock file and the folder stay, so a process
    starting on the namespace meanwhile locks the same file."""
    freed = 0
    for e in _files(ns_dir):
        if e.name != LOCK:
            freed += _unlink(e.path)
    return freed


def _mb(n: int) -> float:
    return n / 2**20


def sweep(dir: str, own_ns: str, cap_bytes: int, deadline: float, log) -> SweepReport:
    """Bring the folder under the cap: real bytes of every namespace folder,
    then whole foreign namespaces go, the least recently saved first, never
    the own one or one a process holds. Under the folder's budget lock,
    waited for until `deadline`; past it the run is skipped and says so."""
    r = SweepReport()
    fd = os.open(os.path.join(dir, BUDGET_LOCK), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        t0 = monotonic()
        locked = _lock_until(fd, deadline)
        r.waited_ms = int(round((monotonic() - t0) * 1000))
        found = {}
        with os.scandir(dir) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    has_manifest = os.path.isfile(os.path.join(e.path, MANIFEST))
                    found[e.name] = (
                        _namespace_bytes(e.path),
                        _saved_at(e.path) if has_manifest else None,
                        has_manifest,
                    )
        r.own_bytes = found.get(own_ns, (0, None, False))[0]
        # A folder with only its lock is no namespace yet.
        foreign = {
            ns: v for ns, v in found.items() if ns != own_ns and (v[0] > 0 or v[2])
        }
        r.total_before = sum(v[0] for v in found.values())
        r.other_namespaces = len(foreign)
        total = r.total_before
        if not locked:
            r.skipped_deadline = True
            r.total_after = total
            log.warning(
                "prompt cache: sweep skipped, the budget lock was not free before the "
                "deadline (%.1f MB on disk, cap %.1f MB)",
                _mb(total),
                _mb(cap_bytes),
            )
            return r
        # Without a manifest a namespace counts as the oldest.
        oldest_first = sorted(foreign, key=lambda ns: foreign[ns][1] or 0.0)
        for ns in oldest_first:
            if total <= cap_bytes:
                break
            ns_dir = os.path.join(dir, ns)
            lock_fd = os.open(os.path.join(ns_dir, LOCK), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if not _try_lock(lock_fd):
                    continue  # a running process owns it
                freed = _clear_namespace(ns_dir)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
            total -= freed
            r.evicted += 1
            r.evicted_bytes += freed
            r.other_namespaces -= 1
            saved_at = foreign[ns][1]
            age = (
                "unknown"
                if saved_at is None
                else f"{(time.time() - saved_at) / 3600:.1f} h"
            )
            log.info(
                "prompt cache: namespace %s evicted for the disk cap (%.1f MB, last saved %s ago)",
                ns[:12],
                _mb(freed),
                age,
            )
        r.total_after = total
        if total > cap_bytes:
            log.warning(
                "prompt cache: %.1f MB on disk stay over the cap of %.1f MB; only the own "
                "namespace and ones other processes hold are left",
                _mb(total),
                _mb(cap_bytes),
            )
    finally:
        os.close(fd)
    return r


# -- the store as a whole ----------------------------------------------------


def _generation(ns_dir: str, old: dict | None) -> int:
    """One past the manifest's count and past every generation a file
    name carries - an unreadable manifest must not reuse a name."""
    gen = int(old.get("generation", 0)) if old else 0
    for e in _files(ns_dir):
        parts = e.name.split(".")
        if len(parts) >= 3 and parts[-1] == "safetensors" and parts[1].isdigit():
            gen = max(gen, int(parts[1]))
    return gen + 1


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
    """The store into its namespace folder within `budget_s`: system
    entries first, then conversations newest first; an entry read from a
    file that is still whole is not written again. Then the manifest
    turns, the files it no longer names go, and the sweep runs under the
    same deadline. A failure raises; the caller counts it."""
    r = SaveReport()
    started = monotonic()
    deadline = started + budget_s
    ns_dir = os.path.join(dir, ns)
    os.makedirs(ns_dir, mode=0o700, exist_ok=True)
    entries = list(store.entries())
    nbytes = sum(e.nbytes for _, e in entries)
    if nbytes > cap_bytes:
        r.skipped_over_cap = 1
        r.disk_bytes = _namespace_bytes(ns_dir)
        log.warning(
            "prompt cache: not saved, the store alone (%.1f MB) is over the disk cap (%.1f MB)",
            _mb(nbytes),
            _mb(cap_bytes),
        )
        return r
    try:
        old = read_manifest(ns_dir)
    except (OSError, ValueError) as e:
        log.warning(
            "prompt cache: the old manifest is unreadable (%s); it is replaced", e
        )
        old = None
    old_records = {
        rec["file"]: rec
        for rec in (old or {}).get("entries", [])
        if isinstance(rec, dict) and "file" in rec
    }
    records: dict[int, dict] = {}
    pending = []
    for key, entry in entries:
        rec = old_records.get(entry.file) if entry.file else None
        if rec is not None and _file_size(os.path.join(ns_dir, entry.file)) == rec.get(
            "bytes"
        ):
            records[id(entry)] = rec
        else:
            pending.append((key, entry))
    need = sum(e.nbytes for _, e in pending)
    st = os.statvfs(ns_dir)
    free = st.f_bavail * st.f_frsize
    # Old and new files lie side by side until the manifest turns.
    if free < need * 1.1:
        r.skipped_no_space = 1
        r.disk_bytes = _namespace_bytes(ns_dir)
        log.warning(
            "prompt cache: not saved, %.1f MB free under %s, %.1f MB needed",
            _mb(free),
            dir,
            _mb(need * 1.1),
        )
        return r
    gen = _generation(ns_dir, old)
    unsupported: set[str] = set()
    for key, entry in pending:
        classes = _supported(entry.cache)
        if classes:
            unsupported.update(classes)
            r.skipped_unsupported += 1
            continue
        if monotonic() >= deadline:
            r.skipped_entries += 1
            continue
        name = f"{key_sha(key)}.{gen}.safetensors"
        arrays, metadata = entry_to_tensors(key, entry, head_digest, vision)
        size = write_entry(os.path.join(ns_dir, name), key, entry, head_digest, vision)
        entry.file = name
        records[id(entry)] = {
            "file": name,
            "bytes": size,
            "cache_type": entry.cache_type,
            "length": entry.length,
            "nbytes": entry.nbytes,
            "image": any(k < 0 for k in key),
            "head": "head_digest" in metadata,
            "head_digest": metadata.get("head_digest"),
            "tensors": tensor_specs(arrays),
        }
        r.entries += 1
        r.bytes += size
    if unsupported:
        log.warning(
            "prompt cache: %d entries not saved, their cache classes are not carried: %s",
            r.skipped_unsupported,
            ", ".join(sorted(unsupported)),
        )
    # The manifest in the store's own order: system first, oldest first.
    ordered = []
    for kind in ("system", "assistant"):
        of_kind = [(k, e) for k, e in entries if e.cache_type == kind]
        for _, entry in reversed(of_kind):
            if id(entry) in records:
                ordered.append(records[id(entry)])
    manifest = {
        "format": FORMAT,
        "namespace": ns,
        "parts": parts,
        "generation": gen,
        "saved_at": time.time(),
        "bytes": sum(rec["bytes"] for rec in ordered),
        "entries": ordered,
    }
    write_manifest(ns_dir, manifest)
    named = {rec["file"] for rec in ordered}
    for e in _files(ns_dir):
        if e.name.endswith(".safetensors") and e.name not in named:
            _unlink(e.path)
    r.disk_bytes = _namespace_bytes(ns_dir)
    r.sweep = sweep(dir, ns, cap_bytes, deadline, log)
    r.ms = int(round((monotonic() - started) * 1000))
    return r


def _within_caps(records: list[dict], store: PrefixStore) -> list[dict]:
    """The records the RAM store would keep, dropped as its eviction would:
    surplus system entries first, then assistant before system, oldest
    first, until the counts and bytes fit."""
    system = [rec for rec in records if rec.get("cache_type") == "system"]
    assistant = [rec for rec in records if rec.get("cache_type") != "system"]
    system = system[max(0, len(system) - store.system_cap) :]

    def total() -> int:
        return sum(int(rec.get("nbytes", 0)) for rec in system + assistant)

    while len(system) + len(assistant) > store.max_entries or total() > store.max_bytes:
        if assistant:
            assistant.pop(0)
        elif system:
            system.pop(0)
        else:
            break
    return system + assistant


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
    """The namespace folder into the store: what the manifest names and the
    caps would keep, each file read whole and checked; a miss is counted,
    logged and removed, never raised. Files the manifest does not name are
    leftovers of a save that never published them."""
    r = LoadReport()
    started = monotonic()
    ns_dir = os.path.join(dir, ns)
    os.makedirs(ns_dir, mode=0o700, exist_ok=True)
    try:
        manifest = read_manifest(ns_dir)
    except (OSError, ValueError) as e:
        r.failures += 1
        r.disk_bytes = _namespace_bytes(ns_dir)
        log.error("prompt cache: %s; nothing loaded, nothing removed", e)
        return r
    records = (
        []
        if manifest is None
        else [
            rec
            for rec in manifest["entries"]
            if isinstance(rec, dict) and "file" in rec
        ]
    )
    if manifest is not None and (
        manifest.get("format") != FORMAT or manifest.get("namespace") != ns
    ):
        log.warning(
            "prompt cache: the manifest in %s is of format %r for namespace %s; "
            "its %d entries are a miss",
            ns_dir,
            manifest.get("format"),
            str(manifest.get("namespace"))[:12],
            len(records),
        )
        for rec in records:
            _unlink(os.path.join(ns_dir, rec["file"]))
            r.incompatible += 1
        records = []
    named = {rec["file"] for rec in records}
    wanted = _within_caps(records, store)
    r.over_budget = len(records) - len(wanted)
    for rec in wanted:
        path = os.path.join(ns_dir, rec["file"])
        expected = {
            **rec,
            "format": FORMAT,
            "vision": vision,
            "head_digest": head_digest,
        }
        try:
            key, entry = read_entry(path, expected)
        except IncompatibleEntry as e:
            r.incompatible += 1
            log.warning("prompt cache: entry is a miss (incompatible): %s", e)
            _unlink(path)
            continue
        except PartialEntry as e:
            r.partial += 1
            log.warning("prompt cache: entry is a miss (partial): %s", e)
            continue
        except CorruptEntry as e:
            r.corrupt += 1
            log.warning("prompt cache: entry is a miss (corrupt): %s", e)
            _unlink(path)
            continue
        if rec.get("head") and (
            head_digest is None or rec.get("head_digest") != head_digest
        ):
            r.sidecars_dropped += 1
        try:
            store.insert(
                model_key,
                key,
                entry.cache,
                entry.checkpoints,
                entry.cache_type,
                file=entry.file,
            )
        except Exception:  # noqa: BLE001 - one entry, not the start
            r.failures += 1
            log.exception("prompt cache: %s could not be inserted", rec["file"])
            continue
        r.entries += 1
        r.bytes += entry.nbytes
    for e in _files(ns_dir):
        if e.name.endswith(".safetensors") and e.name not in named:
            _unlink(e.path)
            r.partial += 1
    r.disk_bytes = _namespace_bytes(ns_dir)
    r.ms = int(round((monotonic() - started) * 1000))
    return r
