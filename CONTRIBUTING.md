# Contributing to mlx-beam

## AI usage policy

AI-generated code is allowed. What is not allowed is submitting code you do not
understand. You are responsible for every line, however it was produced, and
you disclose how AI was used. An agent may draft a pull request description or
an issue; you read, edit and approve it before it goes out, and you keep it
short. Replies to people are written by people.

## Setup

```bash
git clone git@github.com:p4ik/mlx-beam.git
cd mlx-beam
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

MLX only runs on Apple silicon. On Linux, `uv sync --extra dev` installs the
CPU build so the tests run there too; `beam doctor` reports the device.

## Pull requests

- Branch from `main` (`feat/…`, `fix/…`, `docs/…`, `vendor/…`); everything is
  squash-merged, so keep the PR to one change.
- New code is covered by tests; confirm the new tests fail on `main`.
- If performance may be affected, run the benchmark on `main` and on the branch
  and include the scripts and the numbers.
- Format with `uv run pre-commit run --all` before you push.
- Fill in the pull request template; CI must be green before merge.

## Vendored code

`mlx_beam/_vendor/` holds copies of upstream code (see `VENDORED.md`). Do not
monkeypatch it from elsewhere; change the file and record the change. Updating
a vendored part is its own pull request: move the pin in `tools/vendor.toml`,
replace the files, re-apply the local changes, describe them in `VENDORED.md`,
and run `tools/vendor_diff.py` - it fails on any difference the file does not
list. CI runs the same check.

## Issues

Use GitHub issues for bugs. Include the machine (chip, memory, macOS), the
`beam doctor --json` output, the model, and the steps to reproduce.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see LICENSE).
