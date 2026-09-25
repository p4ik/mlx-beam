"""The package layout as the engine reads it: config.json names the manifest
(`extras.manifest`), the manifest names the parts (`parts.<name>` with a
file, a SHA-256 and what the part carries). A checkpoint without the key is
a plain MLX checkpoint and every reader falls back to its own defaults.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_manifest(model_path: Path) -> dict[str, Any] | None:
    """The manifest config.json points at, or None for a plain checkpoint.
    A key that points nowhere is an error: the package claims a layout it
    does not have."""
    config_file = model_path / "config.json"
    if not config_file.is_file():
        return None
    extras = json.loads(config_file.read_text()).get("extras")
    if not isinstance(extras, dict) or not extras.get("manifest"):
        return None
    file = model_path / extras["manifest"]
    if not file.is_file():
        raise FileNotFoundError(
            f"config.json names extras.manifest {extras['manifest']}, not found"
        )
    manifest = json.loads(file.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("parts"), dict):
        raise ValueError(f"{file} is not a package manifest (no parts object)")
    return manifest


def part(manifest: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    entry = (manifest or {}).get("parts", {}).get(name)
    return entry if isinstance(entry, dict) else None


def sha256_of(file: Path) -> str:
    digest = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_part_file(model_path: Path, name: str, entry: dict[str, Any]) -> Path:
    """The part's file, checked against the manifest's SHA-256 when it has
    one. A mismatch is an error, not a warning: the bytes are not what the
    package was measured with."""
    file = model_path / entry["file"]
    if not file.is_file():
        raise FileNotFoundError(
            f"manifest names parts.{name}.file {entry['file']}, not found"
        )
    expected = entry.get("sha256")
    if expected:
        actual = sha256_of(file)
        if actual != expected:
            raise ValueError(
                f"parts.{name}: {entry['file']} has sha256 {actual[:12]}…, the "
                f"manifest says {expected[:12]}…"
            )
    return file
