"""Nothing on the private denylist reaches the repository.

The denylist is a set of case-insensitive regular expressions that must
not appear in any file, path, commit message or commit identity - and the
list itself is never part of the repository: it comes from the environment
(`LEAK_PATTERNS`, one pattern per line, the CI secret) or from a file
outside the checkout (`MLX_BEAM_DENYLIST`, default
`~/.config/mlx-beam/denylist`). A match is reported by place and count,
never by content.

Modes: files (the pre-commit stage: the staged files by name), a commit
message file plus the identities of the commit being made (the commit-msg
stage), or `--range base..head` (CI: every commit's identities, message and
added lines, plus the whole tree at head).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_FILE = Path.home() / ".config" / "mlx-beam" / "denylist"


def load_patterns() -> list[re.Pattern[str]] | None:
    """The denylist, or None when no source is configured. An empty source is
    a configured list with nothing on it."""
    raw = os.environ.get("LEAK_PATTERNS")
    if raw is None:
        path = Path(os.environ.get("MLX_BEAM_DENYLIST") or DEFAULT_FILE)
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
    patterns = []
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.I))
    return patterns


def hits(text: str, patterns: list[re.Pattern[str]]) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def ident(var: str) -> str:
    return git("var", var).strip()


def scan_files(paths: list[str], patterns) -> list[tuple[str, int]]:
    found = []
    for p in paths:
        n = hits(p, patterns)
        try:
            n += hits(Path(p).read_text(encoding="utf-8", errors="ignore"), patterns)
        except OSError:
            pass
        if n:
            found.append((p if not hits(p, patterns) else "a path", n))
    return found


def scan_tree(rev: str, patterns) -> list[tuple[str, int]]:
    found = []
    for p in git("ls-tree", "-r", "--name-only", rev).splitlines():
        n = hits(p, patterns)
        blob = subprocess.run(
            ["git", "show", f"{rev}:{p}"], capture_output=True, check=True
        ).stdout
        n += hits(blob.decode("utf-8", errors="ignore"), patterns)
        if n:
            found.append((p if not hits(p, patterns) else "a path", n))
    return found


def scan_range(spec: str, patterns) -> list[tuple[str, int]]:
    """Each commit: identities, message, the lines it adds; then the tree at
    the head. A file added and removed again inside the range would still
    sit in the pull request's refs, so every commit's additions count."""
    found = []
    base, _, head_rev = spec.partition("..")
    head_rev = head_rev or "HEAD"
    # A first push has no base (all zeros): everything the head reaches is
    # new to the remote, so the whole history is the range.
    selector = [head_rev] if re.fullmatch(r"0+", base) else [spec]
    log = git(
        "log",
        "--format=%x00%h%n%an <%ae>%n%cn <%ce>%n%B",
        "-p",
        "--no-color",
        *selector,
    )
    for entry in log.split("\0"):
        if not entry.strip():
            continue
        sha, _, rest = entry.partition("\n")
        head, _, patch = rest.partition("\ndiff --git ")
        n = hits(head, patterns)
        n += hits(
            "\n".join(
                line[1:]
                for line in patch.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ),
            patterns,
        )
        if n:
            found.append((f"commit {sha}", n))
    found += scan_tree(head_rev, patterns)
    return found


def main(argv: list[str]) -> int:
    patterns = load_patterns()
    if patterns is None:
        required = os.environ.get("LEAK_PATTERNS_REQUIRED", "1") != "0"
        print(
            "content check: no denylist configured - set LEAK_PATTERNS or put "
            f"one pattern per line into {DEFAULT_FILE} (an empty file is fine)",
            file=sys.stderr,
        )
        return 1 if required else 0
    if len(argv) >= 2 and argv[1] == "--range":
        if len(argv) != 3:
            print(
                "usage: check_content.py --range <base>..<head> | --msg <file> | <files>"
            )
            return 2
        found = scan_range(argv[2], patterns)
    elif len(argv) >= 2 and argv[1] == "--msg":
        text = Path(argv[2]).read_text(encoding="utf-8") if len(argv) > 2 else ""
        n = hits(text, patterns) + hits(
            ident("GIT_AUTHOR_IDENT") + "\n" + ident("GIT_COMMITTER_IDENT"), patterns
        )
        found = [("commit message or identity", n)] if n else []
    else:
        found = scan_files(argv[1:], patterns)
    if found:
        # Where, and how often - never what: the match is what must not
        # appear anywhere, logs included.
        for place, n in found:
            print(f"content check: {n} match(es) in {place}", file=sys.stderr)
        print(
            f"content check: {sum(n for _, n in found)} match(es) of the denylist",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
