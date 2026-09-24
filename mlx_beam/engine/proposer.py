"""Proposers: what drafts the tokens a speculative cycle verifies.

A proposer is asked for up to `depth` draft tokens after the token the target
just sampled, given the hidden state that produced it; after the verify it is
told which drafts held, so a proposer with its own history (the MTP head's KV
cache) stays in step with the target. Nothing here samples: the target's own
logits decide every token, the proposer only saves forward passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn

from mlx_beam._vendor.mlx_lm.models.base import create_attention_mask
from mlx_beam._vendor.mlx_lm.models.cache import KVCache


class Proposer(Protocol):
    kind: str
    depth: int

    def propose(self, uid: int, hidden: mx.array, token: mx.array) -> mx.array:
        """Up to `depth` draft ids (lazy, shape (n,)) following `token` (shape
        (1,)), given the target's hidden state at that position ((1, 1, H)).
        An empty array means: nothing to draft, decode this step plainly."""

    def commit(self, uid: int, hidden: mx.array, tokens: list[int]) -> None:
        """The verify's outcome: `tokens` are the drafts that held (in order),
        `hidden` ((1, m, H)) the target's hidden state at each of them."""

    def drop(self, uid: int) -> None:
        """The row left the batch, or decoded a step without this proposer."""

    def describe(self) -> dict: ...


# -- the bundled MTP head -------------------------------------------------

NORM_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


def trunk(model: Any) -> tuple[Any, Any, Any]:
    """(inner model returning the post-norm hidden state, lm_head callable,
    embedding) of a checkpoint: the multimodal wrapper of Qwen3.5/3.8 keeps
    them one level down under `language_model`."""
    text = getattr(model, "language_model", model)
    inner = getattr(text, "model", None)
    args = getattr(text, "args", None)
    if inner is None or args is None or not hasattr(inner, "embed_tokens"):
        raise ValueError(
            f"{type(model).__name__} does not expose a text model with "
            "embed_tokens and a final norm; the speculative path cannot read "
            "its hidden state"
        )
    embed = inner.embed_tokens
    if getattr(args, "tie_word_embeddings", False):
        head = embed.as_linear
    elif hasattr(text, "lm_head"):
        head = text.lm_head
    else:
        raise ValueError(f"{type(model).__name__} has no lm_head")
    return inner, head, embed


class MTPHead(nn.Module):
    """Qwen3.5/3.8's multi-token-prediction head: fuse (the embedding of the
    token just placed, the trunk's hidden state that placed it) through `fc`,
    one full-attention decoder layer of the trunk's kind with its own KV
    cache, a final norm; the trunk's lm_head reads the result."""

    def __init__(self, args, layer_class):
        super().__init__()
        self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        # The layer index that makes the trunk's DecoderLayer pick attention.
        self.layers = [layer_class(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, hidden, token_ids, embed, cache):
        e = self.pre_fc_norm_embedding(embed(token_ids))
        h = self.pre_fc_norm_hidden(hidden)
        x = self.fc(mx.concatenate([e, h], axis=-1))
        mask = create_attention_mask(x, cache[0])
        x = self.layers[0](x, mask=mask, cache=cache[0])
        return self.norm(x)


def _quant_predicate(bits: int, group_size: int):
    # `fc` and the norms stay at model precision, the layer's linears quantize:
    # the pack's own layout (mtplx "cyankiwi"), and int4 measured equal to
    # bf16 in acceptance (2026-09-19, Qwen3.8-27B).
    def predicate(path: str, module):
        if path == "fc" or path.startswith("pre_fc_norm") or path == "norm":
            return False
        if path.startswith("layers.") and hasattr(module, "to_quantized"):
            return {"group_size": group_size, "bits": bits}
        return False

    return predicate


def shift_norms(weights: dict) -> dict:
    """HF's Qwen3-Next RMSNorm scales by (1 + w), mlx's by w: every norm of
    the head gets +1. All seven, unconditionally - a per-tensor guess by the
    mean leaves the ones near 1 unshifted and costs ~14 points of acceptance."""
    out = dict(weights)
    for k, v in weights.items():
        if v.ndim == 1 and any(k.endswith(s) for s in NORM_KEYS):
            out[k] = v + 1.0
    return out


def _strip(k: str) -> str:
    return k.split("mtp.", 1)[1] if "mtp." in k else k


def bundled_head_files(model_path: Path) -> tuple[list[Path], dict]:
    """Where a checkpoint keeps its MTP head: the file `mtp_file` names in
    config.json (our layout `mtp/weights.safetensors`, optiq's
    `optiq/mtp.safetensors`), else the shards that hold `mtp.*` tensors
    (a base checkpoint). Returns (files, config)."""
    config = json.loads((model_path / "config.json").read_text())
    named = config.get("mtp_file")
    if named:
        file = model_path / named
        if not file.is_file():
            raise FileNotFoundError(f"config.json names mtp_file {named}, not found")
        return [file], config
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards = sorted({v for k, v in weight_map.items() if k.startswith("mtp.")})
        if shards:
            return [model_path / s for s in shards], config
    raise FileNotFoundError(
        f"{model_path} bundles no draft head: config.json names no mtp_file and "
        "no shard holds mtp.* tensors"
    )


def load_bundled_head(model: Any, model_path: Path) -> tuple[MTPHead, dict]:
    """The head from the checkpoint's own tensors, quantized as the pack says
    (`mtplx_mtp_quantization`, or the layout's `beam.mtp`), else int4/64;
    norms shifted unless the layout's manifest says they already are."""
    text = getattr(model, "language_model", model)
    args = text.args
    files, config = bundled_head_files(model_path)
    weights: dict = {}
    for file in files:
        raw = mx.load(str(file))
        weights.update({_strip(k): v for k, v in raw.items() if "mtp." in k})
    if not weights:
        raise ValueError(f"{files[0]} holds no mtp.* tensors")
    layer_class = type(text.model.layers[args.full_attention_interval - 1])
    head = MTPHead(args, layer_class)
    manifest = (config.get("beam") or {}).get("mtp") or {}
    quant = config.get("mtplx_mtp_quantization") or manifest.get("quantization") or {}
    packed = "layers.0.self_attn.k_proj.scales" in weights
    if packed:
        # The packed width says the bits: K / (32 / bits) uint32 per row.
        w = weights["layers.0.self_attn.k_proj.weight"]
        s = weights["layers.0.self_attn.k_proj.scales"]
        bits = 32 * w.shape[1] // args.hidden_size
        group_size = args.hidden_size // s.shape[1]
        nn.quantize(head, class_predicate=_quant_predicate(bits, group_size))
    else:
        # int4 / 64 measured equal to bf16 in acceptance (2026-09-19, 27B);
        # a width 64 does not divide takes the largest group that does.
        default_group = next(g for g in (64, 32, 16, 8) if args.hidden_size % g == 0)
        bits = int(quant.get("bits", 4))
        group_size = int(quant.get("group_size", default_group))
    if manifest.get("norm_convention", "hf") != "mlx":
        weights = shift_norms(weights)
    head.load_weights(list(weights.items()), strict=True)
    if not packed:
        nn.quantize(head, class_predicate=_quant_predicate(bits, group_size))
    mx.eval(head.parameters())
    return head, {
        "kind": "mtp",
        "source": "bundled",
        "files": [str(f.relative_to(model_path)) for f in files],
        "bits": bits,
        "group_size": group_size,
        "prequantized": packed,
        "norms_shifted": manifest.get("norm_convention", "hf") != "mlx",
    }


class BundledHeadProposer:
    """The checkpoint's MTP head as proposer. Per row it keeps the head's KV
    cache over the pairs (hidden, next token) the target confirmed - the
    head's history, worth a third of its acceptance - and the pairs the last
    verify confirmed but did not feed yet; those go in with the next draft's
    first call, so a cycle costs `depth` head calls, no more."""

    kind = "mtp"

    def __init__(self, model: Any, head: MTPHead, depth: int, info: dict):
        self._inner, self._lm_head, self._embed = trunk(model)
        self.head = head
        self.depth = depth
        self.info = info
        self._cache: dict[int, list] = {}
        self._pending: dict[int, tuple[list[mx.array], list[int]]] = {}
        self._chain: dict[int, int] = {}

    def propose(self, uid: int, hidden: mx.array, token: mx.array) -> mx.array:
        cache = self._cache.get(uid)
        if cache is None:
            cache = self._cache[uid] = [KVCache()]
        hs, ts = self._pending.pop(uid, ([], []))
        # Confirmed pairs first, the new one last: one call, its last row drafts.
        h_in = mx.concatenate([*hs, hidden], axis=1) if hs else hidden
        t_in = (
            mx.concatenate([mx.array(ts, dtype=token.dtype), token])[None]
            if ts
            else token[None]
        )
        drafts = []
        out = self.head(h_in, t_in, self._embed, cache)[:, -1:]
        for _ in range(self.depth):
            draft = mx.argmax(self._lm_head(out)[:, -1, :], axis=-1)  # (1,)
            drafts.append(draft)
            if len(drafts) == self.depth:
                break
            out = self.head(out, draft[None], self._embed, cache)
        self._chain[uid] = len(drafts) - 1
        return mx.concatenate(drafts)

    def commit(self, uid: int, hidden: mx.array, tokens: list[int]) -> None:
        cache = self._cache.get(uid)
        if cache is None:
            return
        # The chained draft calls wrote entries the target did not confirm.
        chained = self._chain.pop(uid, 0)
        if chained:
            cache[0].trim(chained)
        if tokens:
            self._pending[uid] = ([hidden], list(tokens))

    def drop(self, uid: int) -> None:
        self._cache.pop(uid, None)
        self._pending.pop(uid, None)
        self._chain.pop(uid, None)

    def describe(self) -> dict:
        return {**self.info, "depth": self.depth, "rows_with_history": len(self._cache)}
