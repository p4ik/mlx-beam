# Contributing to mlx-beam

## AI usage policy

AI-generated code is allowed. What is not allowed is submitting code you do not
understand. You are responsible for every line, however it was produced, and
you disclose how AI was used. Do not use AI to write issues, pull request
descriptions, discussions or replies to people.

## Setup

```bash
git clone git@github.com:p4ik/mlx-beam.git
cd mlx-beam
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

MLX only runs on Apple silicon. On other machines the tests that need it are
skipped and `beam doctor` reports MLX as missing.

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
a vendored part is its own pull request.

## Issues

Use GitHub issues for bugs. Include the machine (chip, memory, macOS), the
`beam doctor --json` output, the model, and the steps to reproduce.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see LICENSE).
