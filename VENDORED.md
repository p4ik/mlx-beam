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

`tiled SDPA` - `models/base.py`, `scaled_dot_product_attention`: with a
quantized cache, the tiled attention from the optiq part below runs whenever
it supports the shape (4/8 bit, group 32/64/128, fp16/bf16 queries, causal or
no mask); the stock path stays for everything else.

`rotating merge` - `models/cache.py`, `BatchRotatingKVCache.merge`: a cache
trimmed back to length 0 keeps its buffer, and the source slice `[..., -0:, :]`
is the whole buffer, so the assignment into a zero-width destination raised a
broadcast error and took the generation thread down. Reached by any
sliding-window model reusing a cached prefix across turns. Guarded with
`or l == 0`. Test: `test_rotating_merge_survives_a_zero_length_cache`.

Fixed upstream since 0.31.3 and therefore not carried: the float32 promotion
in `BatchKVCache.extend` when a fresh prompt joins a batch (mlx-lm #1491).

## optiq

| | |
|---|---|
| Upstream | `mlx-optiq` on PyPI (https://mlx-optiq.com); no public source repository, the wheel is the upstream |
| Version | 0.5.6 (wheel sha256 in `tools/vendor.toml`); 0.5.7 checked 2026-09-15, both files unchanged |
| License | MIT, `mlx_beam/_vendor/optiq/LICENSE` |
| Taken | `runtime/kv/batch.py` as `kv_batch.py`, `runtime/fused_quant_sdpa.py` as `fused_quant_sdpa.py` |
| Left out | everything else: the package is a server with its own glue, and the only parts the engine needs are the two below |

Both files are ports, not copies: they were written against mlx-lm 0.31.3
and installed themselves by monkeypatching at run time. Here they target the
vendored mlx-lm commit and are wired in explicitly - `models/base.py` calls
the tiled attention, and the engine builds the quantized per-request caches
itself and hands them to `BatchGenerator.insert`.

### Local changes

`kv_batch.py` - `BatchQuantizedKVCache`, `MergeableQuantizedKVCache`: imports
from the vendored package; `state` carries `_idx`, `group_size` and `bits`
(the old `meta_state` no longer exists upstream); `keys_and_values()` instead
of returning slices from `update_and_fetch`; `merge` takes bits and group size
from any cache that carries them - fresh prompts hold no keys, and taking the
defaults from the first populated cache silently turned a 4-bit configuration
into 8-bit batches (measured: greedy output under 4, 8 and mixed bits
byte-identical); `MergeableQuantizedKVCache.keys_and_values` tolerates an
empty cache; `merge` refuses caches quantized differently instead of
writing packed rows of one width into another. Dropped: the per-layer
`mx.eval` in `update_and_fetch` (the generator evaluates every cache's
full state after each prefill call, before it clears the buffer pool - the
corruption that eval guarded against cannot arise there),
`quantize_batch_cache_layer`, `install_batch_kv_quant` (the monkeypatch
installer). Tests: `test_vendor_optiq_kv.py`.

`fused_quant_sdpa.py`: the tiled attention as a plain function with the
support check next to it; `install`/`uninstall` and the module-level
original dropped. The chunk width is an argument (`n_chunk`, default 512).
Three changes to the kernel itself: it accepts the bool masks the batch
caches build (per-row left padding; the original only knew "causal", so it
never ran on the batch path), it is used for prefill only (one query token
is one matmul on the stock path), and the running max, sum and output
accumulate in float32 across tiles - fp16 stops resolving the sum past a
few thousand tokens. Tests: `test_tiled_sdpa_matches_stock`,
`test_tiled_sdpa_matches_stock_with_a_batch_mask`,
`test_prefill_takes_the_tiled_path`.
