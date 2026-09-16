# Changelog

All notable changes to mlx-beam. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
PEP 440 with SemVer meaning (`0.y` may break, `0.y.z` fixes).

## [Unreleased]

### Added
- Quantized KV cache for the batch path and a tiled quantized attention,
  ported from mlx-optiq; the rotating-cache merge guard (see `VENDORED.md`).
- Vendored mlx-lm (inference subset, pinned commit) with four local changes:
  package-relative imports, prefill mask for hybrids, per-step cache eval,
  and a scheduler that neither pads nor stalls (see `VENDORED.md`).
- Project skeleton: `beam --version`, `beam doctor`, packaging with hatch-vcs,
  pre-commit (black, isort, ruff), CI on Apple silicon, contribution rules.
