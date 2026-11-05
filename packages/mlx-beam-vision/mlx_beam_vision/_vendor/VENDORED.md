# Vendored in mlx-beam-vision

The vision towers come from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
(MIT; `mlx_vlm/LICENSE`), wheel 0.7.1 on PyPI. Each tower is the model
family's `vision.py` and `config.py` under `mlx_vlm/models/<family>/`,
copied and served from `mlx_vlm/<family>/` here, unchanged but for the
local changes listed below; `base.py` holds the two pieces they import
(`BaseModelConfig`, `ensure_fused_sdpa`), copied from
`mlx_vlm/models/base.py`. What mlx-vlm does around a tower - the
processor, the merge of image features into the text embeddings, the
language model - is not vendored: the processor is the checkpoint's own
through transformers, the merge is mlx-beam's prefill (image spans), the
language model is mlx-beam's.

| Family here | Upstream directory | Serves |
|---|---|---|
| `qwen3_vl` | `mlx_vlm/models/qwen3_vl/` (`vision.py`, `config.py`) | Qwen3-VL, Qwen3.5, Qwen3.8 (the same tower; DeepStack features returned alongside) |
| `pixtral` | `mlx_vlm/models/pixtral/` (`vision.py`, `config.py`) | Mistral 3 (the projector is ours, after upstream's `mistral3.py`) |
| `gemma4` | `mlx_vlm/models/gemma4/` (`vision.py`, `config.py`) | Gemma 4 E-series (the embedder is ours, after upstream's `gemma4.py`); nothing of upstream's `gemma4_unified/` is taken - the encoder-free 12B's patch embedder is ours, after transformers' `Gemma4UnifiedVisionEmbedder` (`families/gemma4_unified.py`) |
| `muse_glimmer` | `mlx_vlm/models/muse_glimmer/` (`vision.py`, `config.py`) | Muse Glimmer (adapter and projection ours, after upstream's `muse_glimmer.py`) |
| `granite4_vision` | `mlx_vlm/models/granite4_vision/` (`vision.py`, `config.py`, `downsampling.py`, `qformer.py`) | Granite Vision 4.1 (the packing and per-layer routing ours, after upstream's `granite4_vision.py`) |

`tools/vendor_diff.py` in the repository root compares these files against
the wheel (`tools/vendor.toml`, part `mlx-vlm-vision`).

## Local changes

`moe type` - `qwen3_vl/vision.py`: the tower's type check takes
`qwen3_vl_moe` and `qwen3_vl_moe_vision` (Qwen3-VL-MoE's family name and
transformers' name for its vision config) and `qwen3_vl_vision`
(transformers' name for the dense vision config) beside the dense and
Qwen3.5 names. `families.FAMILIES` routes the types to this tower; without
the entry a checkpoint whose vision_config names its type was refused
before its processor was built. Tests: `tests/test_qwen3_vl.py`
(`test_the_tower_takes_every_vision_type_name_of_its_family`).
