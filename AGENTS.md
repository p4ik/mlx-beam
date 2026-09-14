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

### Do not

AI-written code tends to over-comment. None of these belong in this repo:

- Narrative comments that tell the story of the code, the alternatives that
  were considered, or what the previous version did. The commit message and
  the PR hold that; the code holds the reason in one line.
- Comments that restate the line below them.
- "What I did here" notes, `NOTE:` / `IMPORTANT:` banners, and TODO essays.
  A TODO is one line with an issue number or it is not written.
- Docstrings that repeat the signature ("Returns the result of ...").
  One line that says what the caller gets; more only when the contract
  needs it (units, invariants, failure modes).
- Section-header comments (`# --- helpers ---`) in files under 300 lines.

### Examples

```python
# Good (explains reason, one line)

# The schema requires "content" to be present, even when empty.
choice["message"]["content"] = text if text else None

# Bad (restates the code)

# Set content to text or None.
choice["message"]["content"] = text if text else None

# Bad (narrative - this is a PR description, not a comment)

# `content` stays present and nullable, the way the schema has it. A model
# that stops while still inside a reasoning block leaves `text` empty, and
# dropping the key makes a client raise KeyError instead of reading an empty
# answer. Streaming deltas are left alone: omitting fields between chunks is
# normal there, so we only touch the final message here.
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
  `release-0.y` is the long-lived branch for patch releases of an older
  minor (created from the tag `v0.y.0` when needed; fixes arrive by PR).
- Versions come from git tags (hatch-vcs); never edit a version string.
  On `main` the tree counts towards the next minor: after `v0.1.0` it is
  `0.2.0.devN`, N = commits since the tag (before any tag: `0.1.0.devN`). A
  fix for a released minor lives on `release-0.y`, where the tree counts
  towards the next patch (`0.1.1.devN`, after `v0.1.1` then `0.1.2.devN`).
  No local `+g<hash>` label: PyPI refuses local versions, and `.dev` builds
  are published.
- Pre-releases are PEP 440 (`0.2.0.dev14`, `0.2.0a1`, `0.2.0rc1`) and are
  published when a test needs them - `.dev` only from `main` or a
  `release-*` branch; installers skip them unless asked. A burnt release
  number is never reused.
- Vendored parts are updated in their own commit (`vendor: <part> <old> -> <new>`).

## AI usage

- AI-generated code is allowed. You are responsible for every line and disclose
  how AI was used. Do not let AI write issues, PR descriptions or replies to
  people.
