# Vendored code

Third-party code lives under `mlx_beam/_vendor/<part>/` with its own license
file. Each part is pinned to one upstream commit; `tools/vendor.toml` holds the
coordinates and `tools/vendor_diff.py` checks the tree against them (every
difference must be listed here). Updating a part is its own commit
(`vendor: <part> <old> -> <new>`) and a changelog line.

## mlx-lm

| | |
|---|---|
| Upstream | https://github.com/ml-explore/mlx-lm |
| Commit | `dcbcf786c0cf56f9a12fabe9468c887781431ae2` (2026-09-12, "Refactor/detokenizers (#1879)"; 108 commits after v0.31.3) |
| Upstream version string | 0.32.0 (unreleased at the time) |
| License | MIT, `mlx_beam/_vendor/mlx_lm/LICENSE` |
| Taken | `generate.py`, `generate_utils.py`, `sample_utils.py`, `tokenizer_utils.py`, `utils.py`, `py.typed`, `models/`, `tool_parsers/`, `chat_templates/` |
| Left out | the CLIs (`chat`, `generate`, `server`, `convert`, `manage`, ...), training (`lora`, `tuner/`, `fuse`), `evaluate`, `perplexity`, `benchmark`, `quant/`, `gguf`, `cache_prompt`, examples and docs |
| Next check | monthly against `main`; earlier when one of mlx-lm #1788, #990, #1872, #1821 merges |

Why a commit from `main` and not the 0.31.3 release: the release is five
months old and structurally behind (stop-sequence matcher, counters object,
cache state with offsets, `make_prompt_cache` as the one constructor,
CVE-2026-5843 fix). The engine starts from the state the next release will
have.

### Local changes

`import paths` - `utils.py`, `tokenizer_utils.py`, `models/plamo2.py`: three
dynamic imports and one absolute import used the top-level name `mlx_lm`;
they now use the vendored package path. Nothing else about them changed.

`mask` - `models/cache.py`, `ArraysCache.make_mask`: `lengths` decides before
`left_padding`. A right-padded prefill sets `lengths`, but `merge()` of fresh
caches leaves `left_padding = [0] * B` behind, and upstream checks that first:
the recurrent layers of a hybrid then get an all-true mask and run the padding
through the delta rule, so the shorter prompt of a batch decodes from a
corrupted state. Test: `test_arrays_cache_mask_prefers_lengths`.

`cache eval` - `generate.py`, `GenerationBatch._step`: the cache state is
evaluated after every step (`_cache_arrays`). The per-step cache assignments
are lazy; each step's array references the previous one, and a live graph
keeps one Metal buffer per chaining layer per token. Metal counts buffers,
not bytes: a 48-layer recurrent model died after ~10k generated tokens
(mlx-lm #1332, #1845, #1871). Synchronous on purpose - measured equal to the
async variant at 125 ms/step (2026-09-02, M4 Pro, 6.23 vs 6.03 tok/s per
request); on a machine with much shorter steps the sync may cost 15-25 % and
the async variant should be measured again.

`scheduler` - `generate.py`, `BatchGenerator._next` and two constructor
arguments (`prefill_slice=512`, `decode_share=0.5`). Upstream right-pads every
prompt of a prefill batch to the longest slice, runs one uninterruptible call
of `prefill_step_size` width, and gives decoding sequences one token per such
call. Measured on a 27B GDN hybrid (2026-09-13, M4 Pro 64 GB): a 281-token
request next to a 16 860-token prefill waited longer than its 30 s timeout for
the first token. Changes: (1) the width of a prefill call is the shortest
current segment, so nobody is padded; (2) every call is at most
`prefill_slice` tokens wide, so a newcomer waits at most one such call (512 ran
at 125 tok/s against 114 for 2048); (3) while a prefill shares the worker,
decode runs for `decode_share` times the last prefill call's wall time and
hands its tokens back before the next prefill call. Alone, a prompt sees
upstream's chunking at slice width; greedy output is byte-identical
(`test_prefill_slice_keeps_output`, and the greedy replay against an unpatched
worker on the 27B). Tests: `test_batch_matches_solo_on_hybrid`,
`test_scheduler_alternates_under_sharing`.

Fixed upstream since 0.31.3 and therefore not carried: the float32 promotion
in `BatchKVCache.extend` when a fresh prompt joins a batch (mlx-lm #1491).
