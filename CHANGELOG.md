# Changelog

All notable changes to mlx-beam. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
PEP 440 with SemVer meaning (`0.y` may break, `0.y.z` fixes).

## [Unreleased]

### Added
- Engine: one worker thread, continuous batching, KV policy per layer,
  `/health` with the cache layout that was actually built.
- OpenAI-compatible server: `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses` (stateless, function tools), `/v1/models`; `beam serve`.
- Prefix store with recurrent-state checkpoints: hybrid models resume from
  the last system or user boundary instead of re-prefilling everything.
- Quantized KV cache for the batch path and a tiled quantized attention,
  ported from mlx-optiq; the rotating-cache merge guard (see `VENDORED.md`).
- Vendored mlx-lm (inference subset, pinned commit) with four local changes:
  package-relative imports, prefill mask for hybrids, per-step cache eval,
  and a scheduler that neither pads nor stalls (see `VENDORED.md`).
- Project skeleton: `beam --version`, `beam doctor`, packaging with hatch-vcs,
  pre-commit (black, isort, ruff), CI on Apple silicon, contribution rules.
