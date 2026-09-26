"""Every commit in a range carries only GitHub noreply identities.

Author, committer and every Co-authored-by trailer must be a
`@users.noreply.github.com` address (or GitHub's own noreply): nothing
else is a valid identity for this repository. Run as a commit-msg hook
(the commit being made) or over a range in CI (the pull request).
"""

import re
import subprocess
import sys

ALLOWED = re.compile(
    r"^[^<]*<(?:[^@<>]+@users\.noreply\.github\.com|noreply@github\.com)>$"
)
TRAILER = re.compile(r"^Co-authored-by:\s*(.+)$", re.M | re.I)


def ident(var: str) -> str:
    out = subprocess.run(["git", "var", var], capture_output=True, text=True).stdout
    return re.sub(r"\s+\d+\s+[+-]\d{4}\s*$", "", out.strip())


def check(identities: list[str]) -> list[str]:
    return [i for i in identities if not ALLOWED.match(i)]


def main(argv: list[str]) -> int:
    bad: list[str] = []
    if len(argv) == 2 and argv[1] == "--range":
        print("usage: check_identity.py --range <base>..<head> | <commit-msg-file>")
        return 2
    if len(argv) == 3 and argv[1] == "--range":
        log = subprocess.run(
            ["git", "log", "--format=%an <%ae>%n%cn <%ce>%n%B%x00", argv[2]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for entry in log.split("\0"):
            entry = entry.strip()
            if not entry:
                continue
            lines = entry.split("\n")
            bad += check(lines[:2] + TRAILER.findall("\n".join(lines[2:])))
    else:
        message = open(argv[1], encoding="utf-8").read() if len(argv) > 1 else ""
        bad += check(
            [ident("GIT_AUTHOR_IDENT"), ident("GIT_COMMITTER_IDENT")]
            + TRAILER.findall(message)
        )
    if bad:
        # The offending identity is not printed: it may be what must not
        # appear anywhere, logs included.
        print(
            f"identity check: {len(bad)} identity(ies) outside "
            "@users.noreply.github.com - set user.name/user.email to your "
            "GitHub noreply address and amend",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
