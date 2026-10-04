# mlx-beam-vision

The towers that let [mlx-beam](https://github.com/p4ik/mlx-beam) see. Install
it next to the engine (`pip install mlx-beam[vision]`, or this package by
name) and a checkpoint that ships a vision tower serves images: an
`image_url` part with a `data:` URL in a chat request, the checkpoint's own
processor renders the template, the tower encodes the image, and the
engine's prefill takes the image's features in place of the placeholder
tokens - beside text-only requests in the same batch, with the prefix cache
keyed on the image's digest. Without this package the engine answers image
input with a 400 that names the `vision` extra.

Towers vendored from [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (MIT,
see `mlx_beam_vision/_vendor/VENDORED.md`):

| Family | Models | Notes |
|---|---|---|
| `qwen3_vl` | Qwen3-VL, Qwen3.5, Qwen3.8 | DeepStack applied: the tower's intermediate features are added after the text model's first layers; the text model's multimodal positions (MRoPE: time, height, width per image token) are built from the processor's grids and applied through prefill, cache and decoding |
| `mistral3` | Mistral Small 3.x (Pixtral tower) | the image's rows are separate spans (`[IMG_BREAK]` between them) |
| `gemma4` | Gemma 4 E2B, E4B | image only; audio is not served yet |
| `gemma4_unified` | Gemma 4 12B and up | encoder-free: the checkpoint's own patch embedder (after transformers' `Gemma4UnifiedVisionEmbedder`, no mlx-vlm part) in place of a tower; an image's tokens attend to each other both ways in the text model's sliding layers, which the engine's prefill masks and keeps in one call |
| `muse_glimmer` | Muse Glimmer | the projector's norm matches the text model's input norm |
| `granite4_vision` | Granite Vision 4.1 | AnyRes tiles, window Q-Former projectors; nothing at the embedding, every projector's features added ahead of its text layer |

Not yet: audio and video input. A quantized tower, projector or adapter
loads as such. An image above 32 megapixels is refused before it is decoded.

The package installs torch and torchvision: the processors transformers
ships for these families (the template rendering and the image-to-patches
step) are written on them. The towers themselves run on MLX; torch does no
inference here.

Every family is exercised in the tests on a tiny random tower against a
direct forward; the real towers and processors are validated on Apple
silicon with the checkpoints themselves.

`/health.vision` reports the family, the tower, the processor and the
feature cache (encoder outputs by image digest, bounded in bytes).
