#!/usr/bin/env python3
"""Compare the vendored parts under mlx_beam/_vendor with their upstream.

    tools/vendor_diff.py                 diff every part against its pinned commit
    tools/vendor_diff.py --part mlx-lm   one part
    tools/vendor_diff.py --clock         also report how far upstream moved

Exit status 1 when a file differs that VENDORED.md does not list as modified,
when a listed file is missing, or when the vendored tree carries files that
are neither upstream's nor ours.
"""

import argparse
import difflib
import subprocess
import sys
import tempfile
import tomllib
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
            raise SystemExit(f"{entry}: not in upstream at the pinned commit")


def diff_part(name, part, clock):
    dest = ROOT / part["dest"]
    with tempfile.TemporaryDirectory() as tmp:
        checkout(part["repo"], part["commit"], tmp)
        src = Path(tmp) / part["upstream_subdir"]
        expected = set(upstream_files(src, part["include"]))
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
            a = (src / rel).read_bytes()
            b = here.read_bytes()
            if a != b:
                changed.append(rel)
                if rel not in listed:
                    errors.append(f"changed but not listed as modified: {rel}")
        for rel in sorted(listed - set(changed)):
            errors.append(f"listed as modified but identical to upstream: {rel}")
        for rel in sorted(present - expected - ours):
            errors.append(f"not upstream's and not listed as local: {rel}")
        print(f"{name} @ {part['commit'][:12]}: {len(expected)} upstream files, "
              f"{len(changed)} modified, {len(ours)} local")
        for rel in changed:
            a = (src / rel).read_text().splitlines(keepends=True)
            b = (dest / rel).read_text().splitlines(keepends=True)
            sys.stdout.writelines(
                difflib.unified_diff(a, b, f"upstream/{rel}", f"vendored/{rel}", n=2)
            )
        if clock:
            head = git("ls-remote", part["repo"], "HEAD").split()[0]
            git("fetch", "-q", "--depth", "200", "origin", head, cwd=tmp)
            behind = git("rev-list", "--count", f"{part['commit']}..{head}", cwd=tmp)
            print(f"{name}: upstream HEAD {head[:12]}, {behind.strip()} commits "
                  "after the pin (capped at 200)")
        for e in errors:
            print(f"ERROR {name}: {e}")
        return not errors


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part")
    ap.add_argument("--clock", action="store_true")
    args = ap.parse_args()
    parts = tomllib.loads(CONFIG.read_text())["parts"]
    if args.part:
        parts = {args.part: parts[args.part]}
    ok = all([diff_part(n, p, args.clock) for n, p in parts.items()])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
