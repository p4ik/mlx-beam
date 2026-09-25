# Vendored in mlx-beam-vision

The vision towers come from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
(MIT; `mlx_vlm/LICENSE`), wheel 0.7.1 on PyPI. Each tower is the model
family's `vision.py` and `config.py` under `mlx_vlm/models/<family>/`,
copied unchanged and served from `mlx_vlm/<family>/` here; `base.py` holds
the two pieces they import (`BaseModelConfig`, `ensure_fused_sdpa`), copied
from `mlx_vlm/models/base.py`. What mlx-vlm does around a tower - the
processor, the merge of image features into the text embeddings, the
language model - is not vendored: the processor is the checkpoint's own
through transformers, the merge is mlx-beam's prefill (image spans), the
language model is mlx-beam's.

| Family here | Upstream directory | Serves |
|---|---|---|
| `qwen3_vl` | `mlx_vlm/models/qwen3_vl/` (`vision.py`, `config.py`) | Qwen3-VL, Qwen3.5, Qwen3.8 (the same tower; DeepStack features returned alongside) |
| `pixtral` | `mlx_vlm/models/pixtral/` (`vision.py`, `config.py`) | Mistral 3 (the projector is ours, after upstream's `mistral3.py`) |
| `gemma4` | `mlx_vlm/models/gemma4/` (`vision.py`, `config.py`) | Gemma 4 E-series (the embedder is ours, after upstream's `gemma4.py`); upstream's `gemma4_unified/` (the encoder-free 12B) is not taken - its image tokens need a bidirectional mask in the text model |
| `muse_glimmer` | `mlx_vlm/models/muse_glimmer/` (`vision.py`, `config.py`) | Muse Glimmer (adapter and projection ours, after upstream's `muse_glimmer.py`) |
| `granite4_vision` | `mlx_vlm/models/granite4_vision/` (`vision.py`, `config.py`, `downsampling.py`, `qformer.py`) | Granite Vision 4.1 (the packing and per-layer routing ours, after upstream's `granite4_vision.py`) |

`tools/vendor_diff.py` in the repository root compares these files against
the wheel (`tools/vendor.toml`, part `mlx-vlm-vision`).
