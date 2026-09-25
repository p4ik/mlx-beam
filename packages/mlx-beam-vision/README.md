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
| `qwen3_vl` | Qwen3-VL, Qwen3.5, Qwen3.8 | DeepStack applied: the tower's intermediate features are added after the text model's first layers |
| `mistral3` | Mistral Small 3.x (Pixtral tower) | the image's rows are separate spans (`[IMG_BREAK]` between them) |
| `gemma4` | Gemma 4 12B, E4B | image only; audio is not served yet |
| `muse_glimmer` | Muse Glimmer | the projector's norm matches the text model's input norm |
| `granite4_vision` | Granite Vision 4.1 | AnyRes tiles, window Q-Former projectors; nothing at the embedding, every projector's features added ahead of its text layer |

Not yet: audio and video input, quantized towers (the towers load at model
precision; a package that ships a quantized tower is refused with a
message).

Every family is exercised in the tests on a tiny random tower against a
direct forward; the real towers and processors are validated on Apple
silicon with the checkpoints themselves.

`/health.vision` reports the family, the tower, the processor and the
feature cache (encoder outputs by image digest, bounded in bytes).
