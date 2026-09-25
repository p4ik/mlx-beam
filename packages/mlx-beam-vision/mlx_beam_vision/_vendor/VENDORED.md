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

`tools/vendor_diff.py` in the repository root compares these files against
the wheel (`tools/vendor.toml`, part `mlx-vlm-vision`).
