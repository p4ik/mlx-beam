#!/usr/bin/env python3
"""Render the `beam serve` flags into the site's configuration page.

The parser in `mlx_beam/cli.py` is the one source: its groups, help texts,
choices and defaults become the tables between the two markers in
`docs/site/src/pages/configuration.md`; the prose around them is written by
hand and left alone.

    tools/gen_config_reference.py          rewrite the generated region
    tools/gen_config_reference.py --check  exit 1 when the page is stale (CI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "site" / "src" / "pages" / "configuration.md"
START, END = "<!-- generated: beam serve flags -->", "<!-- /generated -->"
# argparse names its default group "options"; on the page it is the server.
TITLES = {"options": "Server"}


def serve_parser() -> argparse.ArgumentParser:
    sys.path.insert(0, str(ROOT))
    from mlx_beam.cli import add_serve_arguments

    parser = argparse.ArgumentParser(prog="beam serve", add_help=False)
    add_serve_arguments(parser)
    return parser


def cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def value_column(action: argparse.Action) -> str:
    if action.nargs == 0:
        return "switch"
    if action.choices:
        return " / ".join(f"`{c}`" for c in action.choices)
    if action.metavar:
        name = action.metavar if isinstance(action.metavar, str) else action.metavar[0]
        return f"`{name}`" + (" (one or more)" if action.nargs == "+" else "")
    kind = getattr(action.type, "__name__", None)
    return {"int": "integer", "float": "number", "str": "text", None: "text"}.get(
        kind, "text"
    )


def default_column(action: argparse.Action) -> str:
    d = action.default
    if d is None or d == argparse.SUPPRESS or d is False or d == {} or d == []:
        return "—"
    if isinstance(d, list):
        return ", ".join(f"`{x}`" for x in d)
    return f"`{d}`"


def render(parser: argparse.ArgumentParser) -> str:
    out = [START, ""]
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if a.option_strings]
        if not actions:
            continue
        title = TITLES.get(group.title, group.title)
        out.append(f"### {title[0].upper() + title[1:]}")
        out.append("")
        if group.description:
            out.append(cell(group.description[0].upper() + group.description[1:]) + ".")
            out.append("")
        out.append("| Flag | Value | Default | What it does |")
        out.append("|---|---|---|---|")
        for a in actions:
            flags = ", ".join(f"`{f}`" for f in a.option_strings)
            out.append(
                f"| {flags} | {value_column(a)} | {default_column(a)} | {cell(a.help or '')} |"
            )
        out.append("")
    out.append(END)
    return "\n".join(out)


def splice(page: str, generated: str) -> str:
    head, rest = page.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + generated + tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare, do not write")
    args = ap.parse_args(argv)
    page = PAGE.read_text(encoding="utf-8")
    if START not in page or END not in page:
        print(f"{PAGE}: markers {START!r} … {END!r} missing", file=sys.stderr)
        return 2
    fresh = splice(page, render(serve_parser()))
    if fresh == page:
        print(f"{PAGE.relative_to(ROOT)}: up to date")
        return 0
    if args.check:
        print(
            f"{PAGE.relative_to(ROOT)} is stale against mlx_beam/cli.py; run "
            "tools/gen_config_reference.py and commit the page",
            file=sys.stderr,
        )
        return 1
    PAGE.write_text(fresh, encoding="utf-8")
    print(f"{PAGE.relative_to(ROOT)}: rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
