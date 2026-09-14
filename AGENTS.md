# Instructions for mlx-beam

## Code standards

- Keep comments concise (usually 1-2 lines). Say why, not what.
- Use ASD-STE100 Simplified Technical English.
- A numeric decision (threshold, budget, chunk size) carries its measurement in
  one line: date, machine, number. Unmeasured values are marked `placeholder`.
- Do not monkeypatch vendored code. Edit the file under `mlx_beam/_vendor/` and
  record the change in `VENDORED.md`.
- Health reports what is active, with evidence - not what is configured.
- Never fall back silently to another inference path. Fail the request.
- English only: code, comments, docstrings, commits, docs, issues.
- Format with `uvx pre-commit run --all` (black, isort, ruff).

### Examples

```python
# Good (explains reason)

# The schema requires "content" to be present, even when empty.
choice["message"]["content"] = text if text else None

# Bad (restates the code)

# Set content to text or None.
choice["message"]["content"] = text if text else None
```

## Tests

- Every bug fix ships the test that fails on `main` without it.
- Exactness tests (marker `exactness`) run on the Apple-silicon CI runner with
  tiny models; large-model runs stay in the lab and are reported with scripts.

## Branches, commits and releases

- Everything lands on `main` through a pull request, squash-merged: one commit
  per change, `<area>: <what>` in English, imperative. Branch commits are
  free-form. No direct pushes to `main`.
- Branch names: `feat/…`, `fix/…`, `docs/…`, `vendor/…`. Delete after merge.
- Versions come from git tags (hatch-vcs); never edit a version string.
  Between tags the tree is `0.y.0.devN+g<hash>`, N = commits since the tag.
- Pre-releases are PEP 440 (`0.2.0.dev14`, `0.2.0a1`, `0.2.0rc1`) and are
  published when a test needs them - `.dev` only from `main`; installers skip
  them unless asked. A burnt release number is never reused.
- Vendored parts are updated in their own commit (`vendor: <part> <old> -> <new>`).

## AI usage

- AI-generated code is allowed. You are responsible for every line and disclose
  how AI was used. Do not let AI write issues, PR descriptions or replies to
  people.
