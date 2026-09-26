#!/usr/bin/env python3
"""Compare the vendored parts under mlx_beam/_vendor with their upstream.

    tools/vendor_diff.py                 diff every part against its pin
    tools/vendor_diff.py --part mlx-lm   one part
    tools/vendor_diff.py --clock         also report how far a git upstream moved
    tools/vendor_diff.py --clock --max-days 30 --max-commits 20
                                         exit 2 when a pin is older than that, or
                                         upstream changed the vendored files that
                                         often since (the weekly clock job)

A part is pinned to a git commit (repo + commit, with an include list or a
dest-to-upstream file map) or to a PyPI wheel (wheel + sha256 + a file map).
Exit status 1 when a file differs that tools/vendor.toml does not list as
modified, when a listed file is missing or identical, or when the vendored
tree carries files that are neither upstream's nor listed as local.
"""

import argparse
import difflib
import hashlib
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "tools" / "vendor.toml"


def git(*args, cwd=None):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def checkout(repo, commit, into):
    git("init", "-q", into)
    git("remote", "add", "origin", repo, cwd=into)
    git("fetch", "-q", "--depth", "1", "origin", commit, cwd=into)
    git("checkout", "-q", "FETCH_HEAD", cwd=into)


def unpack_wheel(url, sha256, into):
    data = urllib.request.urlopen(url, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{url}: sha256 {digest} != pinned {sha256}")
    whl = Path(into) / "wheel.whl"
    whl.write_bytes(data)
    zipfile.ZipFile(whl).extractall(into)


def upstream_files(src, include):
    for entry in include:
        p = src / entry
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    yield f.relative_to(src)
        elif p.is_file():
            yield p.relative_to(src)
        else:
            raise SystemExit(f"{entry}: not in upstream at the pin")


def fetch_upstream(part, tmp):
    """Materialise the upstream at its pin; returns {dest rel path: upstream file}."""
    if "repo" in part:
        checkout(part["repo"], part["commit"], tmp)
        src = Path(tmp) / part["upstream_subdir"]
        if "files" in part:
            return {Path(d): src / u for d, u in part["files"].items()}
        return {rel: src / rel for rel in upstream_files(src, part["include"])}
    unpack_wheel(part["wheel"], part["sha256"], tmp)
    src = Path(tmp) / part["upstream_subdir"]
    return {Path(d): src / u for d, u in part["files"].items()}


def pin_label(part):
    return part["commit"][:12] if "repo" in part else f"wheel {part['version']}"


def clock(name, part, upstream, tmp):
    """How far a git upstream moved since the pin: days since the pinned
    commit, and commits since it that touched the vendored files (the
    ones that would land in a re-vendoring) - not every commit upstream."""
    head = git("ls-remote", part["repo"], "HEAD").split()[0]
    git("fetch", "-q", "--depth", "200", "origin", head, cwd=tmp)
    pinned_at = int(git("log", "-1", "--format=%ct", part["commit"], cwd=tmp).strip())
    days = (time.time() - pinned_at) / 86400
    paths = [str(f.relative_to(tmp)) for f in upstream.values()]
    touched = git(
        "rev-list", "--count", f"{part['commit']}..{head}", "--", *paths, cwd=tmp
    ).strip()
    print(
        f"{name}: pin {days:.0f} days old; upstream HEAD {head[:12]}, "
        f"{touched} commits since the pin touched the vendored files "
        "(history capped at 200)"
    )
    return days, int(touched)


def diff_part(name, part, tick, limits):
    dest = ROOT / part["dest"]
    with tempfile.TemporaryDirectory() as tmp:
        upstream = fetch_upstream(part, tmp)
        expected = set(upstream)
        listed = set(map(Path, part["modified"]))
        ours = set(map(Path, part["local"]))
        present = {
            f.relative_to(dest)
            for f in dest.rglob("*")
            if f.is_file() and "__pycache__" not in f.parts
        }
        errors = []
        changed = []
        for rel in sorted(expected):
            here = dest / rel
            if not here.exists():
                errors.append(f"missing: {rel}")
                continue
            if upstream[rel].read_bytes() != here.read_bytes():
                changed.append(rel)
                if rel not in listed:
                    errors.append(f"changed but not listed as modified: {rel}")
        for rel in sorted(listed - set(changed)):
            errors.append(f"listed as modified but identical to upstream: {rel}")
        for rel in sorted(present - expected - ours):
            errors.append(f"not upstream's and not listed as local: {rel}")
        print(
            f"{name} @ {pin_label(part)}: {len(expected)} upstream files, "
            f"{len(changed)} modified, {len(ours)} local"
        )
        for rel in changed:
            a = upstream[rel].read_text().splitlines(keepends=True)
            b = (dest / rel).read_text().splitlines(keepends=True)
            sys.stdout.writelines(
                difflib.unified_diff(a, b, f"upstream/{rel}", f"vendored/{rel}", n=2)
            )
        stale = False
        if tick and "repo" in part:
            days, touched = clock(name, part, upstream, tmp)
            max_days, max_commits = limits
            if max_days is not None and days > max_days:
                print(f"STALE {name}: pin older than {max_days} days")
                stale = True
            if max_commits is not None and touched > max_commits:
                print(
                    f"STALE {name}: more than {max_commits} upstream commits on the vendored files"
                )
                stale = True
        for e in errors:
            print(f"ERROR {name}: {e}")
        return not errors, stale


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part")
    ap.add_argument("--clock", action="store_true")
    ap.add_argument("--max-days", type=float, help="with --clock: exit 2 past this age")
    ap.add_argument(
        "--max-commits", type=int, help="with --clock: exit 2 past this many changes"
    )
    args = ap.parse_args()
    parts = tomllib.loads(CONFIG.read_text())["parts"]
    if args.part:
        parts = {args.part: parts[args.part]}
    limits = (args.max_days, args.max_commits)
    results = [diff_part(n, p, args.clock, limits) for n, p in parts.items()]
    if not all(ok for ok, _ in results):
        sys.exit(1)
    sys.exit(2 if any(stale for _, stale in results) else 0)


if __name__ == "__main__":
    main()
