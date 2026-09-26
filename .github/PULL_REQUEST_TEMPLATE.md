## What

<!-- One change per PR. List what changed, in plain words. -->

## Checklist

- [ ] Tests cover the change (and fail on `main` without it)
- [ ] `uv run pre-commit run --all` is clean
- [ ] CI is green
- [ ] `CHANGELOG.md` has a line under Unreleased, in the order Added,
      Changed, Fixed - and every `Fixed` line names behaviour of the last
      tag (`git describe --tags --abbrev=0 main`); a correction to something
      added since that tag edits its `Added` line instead
- [ ] `VENDORED.md` updated (only if `mlx_beam/_vendor/` changed)
- [ ] Benchmark on `main` and on this branch attached (only if performance may change)
