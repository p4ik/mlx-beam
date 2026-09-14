## What

<!-- One change per PR. Say what changes and why, for a reader without context. -->

## Checklist

- [ ] Tests cover the change (and fail on `main` without it)
- [ ] `uv run pre-commit run --all` is clean
- [ ] CI is green
- [ ] `CHANGELOG.md` has a line under Unreleased
- [ ] `VENDORED.md` updated (only if `mlx_beam/_vendor/` changed)
- [ ] Benchmark on `main` and on this branch attached (only if performance may change)
