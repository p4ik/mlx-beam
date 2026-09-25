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
hands its tokens back before the next prefill call; a burst also ends when
a row finishes, because its extracted cache is a lazy slice of the batch
buffers until the caller evaluates it, and every further step would copy
the whole batch KV instead of writing in place. (4) The width rule has a
starvation guard: when a row with a whole slice to go was held under a
quarter slice by newcomers in two prefill calls in a row, the next call
admits nobody, so that row gets its full width once; measured on a
600-token prompt under one 8-token newcomer per call, 91 calls to the first
token instead of 295, and a newcomer waits at most one call more
(`test_a_trickle_of_short_prompts_cannot_starve_a_long_prefill`). The
count of such calls is `starved_calls`, reported by the engine's health as
`prefill_starved_calls`. Alone, a prompt sees
upstream's chunking at slice width; greedy output is byte-identical
(`test_prefill_slice_keeps_output`, and the greedy replay against an unpatched
worker on the 27B). Tests: `test_batch_matches_solo_on_hybrid`,
`test_scheduler_alternates_under_sharing`.

`tiled SDPA` - `models/base.py`, `scaled_dot_product_attention`: with a
quantized cache, the tiled attention from the optiq part below runs whenever
it supports the shape (4/8 bit, group 32/64/128, fp16/bf16 queries, causal or
no mask, a prefill's worth of query tokens - the threshold is that part's);
the stock path stays for everything else.

`packed K/V without a cache` - `models/base.py`, `scaled_dot_product_attention`
and `_packed_layout`: a layer that shares another layer's cache (Gemma 4's
KV-shared layers) receives that cache's packed `(packed, scales, biases)`
triple with no cache of its own, and the dispatch keyed on `cache.bits` sent
it down the stock path, which cannot take a triple. Bits and group size are
now read off the packed shapes when the cache does not carry them. And
`models/plamo2.py` calls `base.scaled_dot_product_attention` instead of
`mx.fast.scaled_dot_product_attention` directly, so its attention layers
take a quantized cache like every other model's. Tests:
`test_quantized_kv_on_kv_sharing_and_direct_sdpa_models` (Gemma 4 with and
without shared layers, PLaMo-2; 8-bit within 0.1 logprob of the plain run).

`handover` - `generate.py`, `PromptProcessingBatch.generate`: before the
generation batch is built, a cache that offers `quantized()` is replaced by
what it returns. That is how the engine keeps a quantized layer at model
precision through the prefill and quantizes it once, at the move to
decoding. `generate_step` quantizes with `maybe_quantize_kv_cache` after
each prefill chunk once the cache offset reaches `quantized_kv_start`
(default 5000); with that start at the prompt's end it computes exactly
what the engine's exact mode computes, and the batch path had no place
for any of it. The cache classes live in `mlx_beam/engine/kv.py`; a
prefix restored from the prompt store keeps its original codes across the
round trip. Tests: `tests/test_kv_prefill.py` (exact mode matches
`generate_step` with `quantized_kv_start = len(prompt) - 1` to the bit at
8 and 4 bit, for one and for several prefill chunks; a cancelled prefill
is stored quantized).

`layer hook` - `models/qwen3_5.py`, `Qwen3_5TextModel.__call__`;
`models/qwen3.py`, `Qwen3Model.__call__`; `models/qwen3_moe.py`,
`Qwen3MoeModel.__call__`: one keyword argument, `layer_hook`, a callable
applied to the hidden states before every decoder layer, index first. The
engine's prefill hands in what a vision frontend adds at the image
positions ahead of certain layers (DeepStack: the Qwen3-VL tower's
intermediate features ahead of layers 1 to 3 - on Qwen3-VL's own text
models `qwen3`/`qwen3_moe` as much as on Qwen3.5's - and Granite Vision's
projected features ahead of its target layers); with the argument left out
the loop is upstream's. The text models already took `input_embeddings`
upstream; that is how the image features enter at the placeholder
positions. Tests: `tests/test_vision_core.py` (Qwen3.5 and Qwen3-VL's
wrapper over `qwen3`).

`input embeddings` - `models/muse_glimmer.py`, `MuseGlimmerModel`: one
keyword argument, `input_embeddings`, standing in for `embed_inputs(ids)`
(the token embeddings after the model's scaleless RMSNorm; that method is
new too, so the engine's prefill can build the text positions the same
way); `models/granite.py`, `GraniteModel` and `Model`: `input_embeddings`
(standing in for `embed_tokens(ids)`, the multiplier still applies) and
`layer_hook` as on Qwen3.5, for Granite Vision; `models/granitemoehybrid.py`,
`GraniteMoeHybridModel` and `Model`: the same two, the Mamba-2 hybrid
being the text model Granite Vision 4.1 ships with. Upstream's other text
models took the argument already; Qwen3.5, Mistral 3 and Gemma 4 need
nothing here. Tests:
`packages/mlx-beam-vision/tests`.

`granite4_vision` - `models/granite4_vision.py`, ours (no upstream file):
the text-only view of a Granite Vision 4.1 checkpoint, after upstream's
`qwen3_vl.py` - `language_model` is `granite` or `granitemoehybrid` by the
text config, `sanitize` drops the tower and the projectors (the vision
package loads them itself) and moves the nested `model.language_model`
up, `make_cache` is the hybrid's. Without it the loader has no class for
`model_type: granite4_vision` and the CLI cannot load the checkpoint at
all. Tests: `tests/test_vision_core.py`.

`generation batch` - `generate.py`, `BatchGenerator`, `PromptProcessingBatch`:
two constructor arguments. `generation_batch` is the class built at the move
to decoding (`PromptProcessingBatch.generate`) and for the empty batch;
upstream names `GenerationBatch` there, the engine hands in
`mlx_beam.engine.speculative.SpeculativeGenerationBatch`, which decodes
several tokens per call when a proposer drafts them and plainly otherwise.
`prompt_batch` (on `BatchGenerator` only) is the class that prefills, built
for the empty batch and in `_make_batch`; upstream names
`PromptProcessingBatch`, the engine hands in
`mlx_beam.engine.priming.PrimingPromptBatch` when a proposer is configured,
which runs the trunk without the `lm_head` over the prompt chunks and feeds
the hidden states to the draft head. Nothing about the plain path changed:
with the arguments left out the defaults are upstream's classes. Tests:
`tests/test_speculative.py`.

`exact verify` - `models/base.py`, `scaled_dot_product_attention`: with
`EXACT_PER_QUERY` set (by `mlx_beam.engine.exact.exact_forward`) a block of
L queries is attended one query at a time with its own mask row, so the
quantized KV path sums the way plain decoding does; off, the function is
upstream's.

`recurrent stash` - `models/qwen3_5.py`, `models/qwen3_next.py`,
`GatedDeltaNet.__call__`: when the layer's cache carries a `stash` attribute
that is not None, the layer stores what a partial rollback needs - the conv
input, the state before the call, the projected and normalised `q`, `k`,
`v`, `a`, `b`, the mask and the kernel flag. The speculative verify arms
the stash before its forward over the drafts and, when the target rejects
some of them, redoes the recurrence over the accepted prefix from the stash
(`speculative.rollback_recurrent`) instead of the whole model; attention
caches trim. The kernel advances one token at a time, so the replayed
state is exactly the verify's own state after the accepted prefix (bit for
bit, measured 2026-09-24); a separate forward of just those tokens may
differ by the rounding of its projections at another width - the contract
in `speculative.py`. Without the attribute the layer runs as upstream.
Tests: `tests/test_speculative.py` (`test_committed_tokens_equal_plain_greedy`
with nothing, everything and a ragged mix accepted;
`test_rollback_on_metal_matches_a_shorter_forward` on the Metal kernels).
The stash names its `kind` (`gated_delta`); `models/granitemoehybrid.py`,
`GraniteMoeHybridMamba2Mixer`, keeps the same kind of stash for Mamba-2
(`kind` `mamba2`: the padded conv input, the SSM inputs and the state
before the call), and `_conv` keeps the padded input on the module for it.
`speculative.rollback_recurrent` replays either kind. Tests:
`test_mamba2_rollback_matches_a_shorter_forward` (bit for bit, the scan
runs the same ops on both sides) and the Granite class in
`tests/test_model_classes.py`.

`rotating merge` - `models/cache.py`, `BatchRotatingKVCache.merge`: a cache
trimmed back to length 0 keeps its buffer, and the source slice `[..., -0:, :]`
is the whole buffer, so the assignment into a zero-width destination raised a
broadcast error and took the generation thread down. Reached by any
sliding-window model reusing a cached prefix across turns. Guarded with
`or l == 0`. Test: `test_rotating_merge_survives_a_zero_length_cache`.

`labelled channel` - `tokenizer_utils.py`, `_infer_thinking` and
`TokenizerWrapper.think_label_end`: upstream takes `<|channel>thought` as
the opener of Gemma 4's think block, two tokens with the label fixed.
After a tool response the template leaves the turn open and the model
opens the channel itself, writing the label as ` thought` (a different
token): upstream's opener never matches and `<|channel> thought\n
<channel|>` lands in the text. The opener is now `<|channel>` alone, and
`think_label_end` (`"\n"`) says that a label follows it up to the line
end; the engine's text automaton reads the label at runtime and strips it
whatever token it comes as, which is also what the template's own
`strip_thinking` does. The budget then counts the marker token like any
single-token opener. Tests: `test_a_channel_is_thinking_whatever_token_
its_label_comes_as`, `test_the_channel_family_is_inferred_on_the_marker_
alone`.

`tool parsers, picked from upstream` - `tool_parsers/qwen3_coder.py` at
mlx-lm `e99e3df` (2026-09-14, #1881): the parameter name survives a missing
`>` (`_name_regex`) instead of raising `ValueError: substring not found`.
`tool_parsers/mistral.py` at mlx-lm `5681834` (2026-09-14, #1394): the JSON
list form `[{"name": ..., "arguments": ...}]` is parsed, and a header-form
call cut short raises instead of being skipped in silence. Both files are
the upstream files at those commits, nothing else; they postdate the pin and
go away with the next pin bump. Tests: `test_qwen_parameter_without_closing_bracket`,
`test_mistral_json_list_and_cut_call`.

`anchored headers` - `tool_parsers/mistral.py`, `_parse_header_calls`:
upstream searches the text for the next `name[ARGS]` header, so a cut-off
JSON list (`[{"name": …, "arguments": {"content": "see other[ARGS]{}"}}, {`)
that fails `json.loads` falls through to the header parser, which reads
`other[ARGS]{}` out of the string argument and returns it as a call. A
header now has to stand at the start of the text or right after the
previous call's JSON, with only whitespace and the model's repeated
`[TOOL_CALLS]` marker in between; anything else is an error, and the engine
returns the block as text. Test:
`test_mistral_headers_are_anchored_never_read_out_of_a_json_string`.

`parameter end` - `tool_parsers/qwen3_coder.py`, `_parameter_bodies` and
`_closed_parameters`: upstream cuts every parameter at the first
`</parameter>` (`<parameter=(.*?)</parameter>`), so a value that contains the
tag literally - a file with this markup, HTML - comes back shortened, with no
error. A parameter now ends at the last `</parameter>` before the next
`<parameter=` or the end of the call, and a parameter without an end tag is
an error (the call was cut short) instead of a silently missing argument. A
literal `<parameter=name>` inside a value cannot be told from a second
parameter, so a name seen twice, one a schema with `additionalProperties:
false` rules out, or text left between a parameter's end tag and the next
tag (a literal tag split the value) raises instead of overwriting or
dropping part of an argument; the engine then returns the call as text.
Unescaped markup stays ambiguous, so the promise is "never silently", not
"always parsed". Stays after the pin bump unless upstream fixes it.
Tests: `test_qwen_literal_end_tag_in_a_value`,
`test_qwen_parameter_tag_inside_a_value_is_refused_not_silently_split`,
`test_qwen_text_between_parameters_is_an_error_not_dropped`.

Fixed upstream since 0.31.3 and therefore not carried: the float32 promotion
in `BatchKVCache.extend` when a fresh prompt joins a batch (mlx-lm #1491).

## optiq

| | |
|---|---|
| Upstream | `mlx-optiq` on PyPI (https://mlx-optiq.com); no public source repository, the wheel is the upstream |
| Version | 0.5.6 (wheel sha256 in `tools/vendor.toml`); 0.5.7 checked 2026-09-15 and 0.5.8-0.5.10 checked 2026-09-17: both files unchanged, nothing taken. New in 0.5.9, noted for later: a GQA decode/verify attention kernel for head dim 256 (`ops/gqa_decode_attention.py`, MLX's own kernel with the template constants for dim 256, up to 16 query positions, enabled per tested model only) and a chunked verify attention for 5-15 positions (`ops/chunked_verify_attention.py`) - both for unquantized KV, so they do not touch the tiled path taken here; candidates for the speculative verify pass once it exists, to be measured on our hardware first |
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
never ran on the batch path), it runs from `MIN_TILED_QUERIES` (64) query
tokens on rather than from two - the original switched at one, and a
speculative verify of four tokens paid +40 ms per step at 32k context on
the tiles against the stock path (2026-09-19, M4 Pro, Qwen3.8-27B, 8-bit
KV) - and the running max, sum and output accumulate in float32 across
tiles - fp16 stops resolving the sum past a few thousand tokens. Tests:
`test_tiled_sdpa_matches_stock`,
`test_tiled_sdpa_matches_stock_with_a_batch_mask`,
`test_prefill_takes_the_tiled_path`.

## mlx-vlm

Upstream: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), MIT, the PyPI wheel
0.7.1 (`tools/vendor.toml`, part `mlx-vlm`). Two files only,
`mlx_beam/_vendor/mlx_vlm/`:

- `quantized_verifier.py` (`models/quantized_verifier.py` upstream): Metal
  kernels for quantized projections that give, for a block of T rows, the
  bytes T single-row calls give (affine 4/5/8 bit, group 32/64/128, plus
  fixed-format and MoE variants), and the native per-position fallback
  `_exact_time_batch`.
- `exact_speculative_verify.py`: the dense GEMV block kernel, kept with its
  neighbour; not called yet.

### Local changes

`switch import` - `quantized_verifier.py` imports `QuantizedSwitchLinear`
from the vendored mlx-lm instead of mlx-vlm's own module; the class is the
same one mlx-lm serves. Nothing else changed.

### Where it is used

`mlx_beam/engine/exact.py`: inside `exact_forward()` every
`nn.QuantizedLinear` of the model (its class swapped to a subclass of ours at
start, the instance and its weights untouched) projects through
`optimized_affine_linear`, falling back to `_exact_time_batch`; the vendored
`base.scaled_dot_product_attention` attends one query at a time under the
same switch (`EXACT_PER_QUERY`, the `exact verify` change to `base.py`).
The engine measures at warm-up whether that is bit-equal to single forwards
(`speculative.width_check`) and keeps the block verify otherwise, saying so
in `/health.speculative.exact`. Tests: `tests/test_speculative.py`
(`test_exact_install_…`, `test_width_check_…`), the `kernels` mode of
`tests/test_model_classes.py`. The Metal kernels themselves run on Apple
silicon only; the CPU suite exercises the fallback path.
