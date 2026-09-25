"""Proposers: what drafts the tokens a speculative cycle verifies.

A proposer is asked for up to `depth` draft tokens after the token the target
just sampled, given the hidden state that produced it; after the verify it is
told which drafts held, so a proposer with its own history (the MTP head's KV
cache) stays in step with the target. The target's own logits decide every
committed token, the proposer only saves forward passes: for a greedy row it
drafts its argmax, for a sampled row it draws through the row's `choose`,
the same filtered Gumbel-max under the same position key the verify uses -
the draft then agrees with the target exactly when both argmaxes agree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn

from mlx_beam._vendor.mlx_lm.models.base import create_attention_mask
from mlx_beam._vendor.mlx_lm.models.cache import KVCache
from mlx_beam.package import part, read_manifest, verify_part_file


class Proposer(Protocol):
    kind: str
    depth: int

    def propose(
        self,
        uid: int,
        hidden: mx.array,
        token: mx.array,
        choose: Callable[[mx.array, int], mx.array] | None = None,
        depth: int | None = None,
    ) -> mx.array:
        """Up to `depth` draft ids (lazy, shape (n,); None: the proposer's
        own depth) following `token` (shape (1,)), given the target's hidden
        state at that position ((1, 1, H)). `choose(logits, i)` turns the
        (1, V) logits of draft i into its token ((1,)); None means argmax.
        An empty array means: nothing to draft, decode this step plainly."""

    def commit(self, uid: int, hidden: mx.array, tokens: list[int]) -> None:
        """The verify's outcome: `tokens` are the drafts that held (in order),
        `hidden` ((1, m, H)) the target's hidden state at each of them."""

    def follow(self, uid: int, hidden: mx.array | None, token: mx.array) -> None:
        """The row moved one token without this proposer: `token` ((1,))
        follows the position whose hidden state is `hidden` ((1, 1, H)) -
        the pair a `propose` at that position would have carried. `hidden`
        None names the last primed position, whose pair the prefill left
        open (the prompt's last token, fed by the first decode step)."""

    def begin(self, uid: int, prompt_len: int, snapshot: list | None) -> None:
        """A row starts: its prompt length (for a proposer that bounds its
        history) and the history to resume from, if the prefix store had one
        at the position the prefill starts."""

    def prime(self, uid: int, hidden: mx.array, tokens: list[int], start: int) -> None:
        """A prefill chunk: `hidden` ((1, L, H)) the target's hidden state at
        prompt positions `start` .. `start + L - 1`, `tokens` the L tokens at
        those positions - the pair of position t is (hidden[t], tokens[t+1]),
        the last one waits for the token that follows the chunk."""

    def snapshot(self, uid: int) -> list | None:
        """The row's history as arrays and scalars for the prefix store, or
        None when there is none; `begin` takes it back."""

    def drop(self, uid: int) -> None:
        """The row left the batch."""

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


class NextNHead(nn.Module):
    """GLM-4.7 / DeepSeek-V3's NextN head: the trunk's hidden state and the
    embedding of the token just placed, each RMS-normed, fused by `eh_proj`
    (2H -> H), one full decoder layer of the trunk's kind - MoE included -
    with its own KV cache, `shared_head.norm`; `shared_head.head` reads the
    result when the checkpoint ships it, else the trunk's lm_head. The
    fuse puts the embedding first (vLLM `glm4_moe_mtp.py`,
    `deepseek_mtp.py`; the paper writes the pair the other way round) -
    to be held against acceptance on the real package."""

    def __init__(self, args, layer_class, own_embed: bool, own_head: bool):
        super().__init__()
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        # Past the trunk's layers, so the layer picks the MoE form.
        self.layers = [layer_class(args, layer_idx=args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if own_embed:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        if own_head:
            self.head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, hidden, token_ids, embed, cache):
        own = getattr(self, "embed_tokens", None)
        e = self.enorm((embed if own is None else own)(token_ids))
        h = self.hnorm(hidden)
        x = self.eh_proj(mx.concatenate([e, h], axis=-1))
        # An array, as the trunk builds it: an MLA layer masks its scores
        # with mx.where and cannot take the "causal" keyword.
        mask = create_attention_mask(x, cache[0], return_array=True)
        x = self.layers[0](x, mask=mask, cache=cache[0])
        return self.norm(x)


def _quant_predicate(bits: int, group_size: int):
    # The fuse (`fc` / `eh_proj`) and the norms stay at model precision, the
    # layer's linears quantize - the pack's own layout (mtplx "cyankiwi"),
    # and int4 measured equal to bf16 in acceptance (2026-09-19,
    # Qwen3.8-27B); a NextN head's own embedding and output head quantize
    # like the layer.
    def predicate(path: str, module):
        if path in ("fc", "eh_proj", "norm") or path.startswith("pre_fc_norm"):
            return False
        if path in ("enorm", "hnorm"):
            return False
        if (path.startswith("layers.") or path in ("embed_tokens", "head")) and hasattr(
            module, "to_quantized"
        ):
            weight = getattr(module, "weight", None)
            if weight is None or weight.shape[-1] % group_size:
                return False  # a width the group does not divide stays as is
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


_LAYER_PREFIX = re.compile(r"^(?:model\.)?layers\.\d+\.")


def _strip(k: str) -> str:
    """The head's own key: Qwen's `mtp.` prefix or a NextN layer's
    `model.layers.N.` prefix, whichever the checkpoint uses."""
    if "mtp." in k:
        return k.split("mtp.", 1)[1]
    return _LAYER_PREFIX.sub("", k)


def _to_head_form(weights: dict) -> dict:
    """NextN keys into the head module's names: the decoder layer's keys
    under `layers.0.`, `shared_head.norm` as `norm`, `shared_head.head` as
    `head`; the fuse and the embedding keep their names."""
    out = {}
    for k, v in weights.items():
        if k.startswith("shared_head.norm."):
            out["norm." + k[len("shared_head.norm.") :]] = v
        elif k.startswith("shared_head.head."):
            out["head." + k[len("shared_head.head.") :]] = v
        elif k.split(".")[0] in ("enorm", "hnorm", "eh_proj", "embed_tokens"):
            out[k] = v
        else:
            out["layers.0." + k] = v
    return out


def head_form(weights: dict) -> str:
    """What the tensors say the head is: Qwen's `fc`, or NextN's `eh_proj`."""
    if "fc.weight" in weights:
        return "qwen_mtp"
    if "eh_proj.weight" in weights:
        return "glm_nextn"
    raise ValueError(
        "the draft head's tensors match neither form: no fc.weight (Qwen MTP) "
        "and no eh_proj.weight (GLM/DeepSeek NextN)"
    )


def bundled_head_files(model_path: Path) -> tuple[list[Path], dict]:
    """Where a checkpoint keeps its MTP head: the package manifest's
    `parts.mtp.file` (checked against its SHA-256), else the file `mtp_file`
    names in config.json (`mtp/weights.safetensors`, optiq's
    `optiq/mtp.safetensors`), else the shards that hold `mtp.*` tensors
    (a base checkpoint). Returns (files, config)."""
    config = json.loads((model_path / "config.json").read_text())
    mtp = part(read_manifest(model_path), "mtp")
    if mtp:
        # A part that names no file cannot be checked; falling back to
        # mtp_file here would load bytes the manifest never vouched for.
        if not mtp.get("file"):
            raise ValueError("manifest parts.mtp names no file")
        return [verify_part_file(model_path, "mtp", mtp)], config
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
        # A NextN head is the layer past the trunk's last (GLM-4.7,
        # DeepSeek-V3: `num_nextn_predict_layers` of them).
        n = config.get("num_hidden_layers")
        if n is not None and config.get("num_nextn_predict_layers"):
            prefix = f"model.layers.{n}."
            shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
            if shards:
                return [model_path / s for s in shards], config
    raise FileNotFoundError(
        f"{model_path} bundles no draft head: config.json names no mtp_file and "
        "no shard holds mtp.* tensors or a NextN layer"
    )


def _nextn_keys(config: dict, k: str) -> bool:
    n = config.get("num_hidden_layers")
    return n is not None and k.startswith(f"model.layers.{n}.")


def load_bundled_head(model: Any, model_path: Path) -> tuple[nn.Module, dict]:
    """The head from the checkpoint's own tensors, in the form the tensors
    (or the manifest's `parts.mtp.form`) say: Qwen's `mtp.*` head or a
    NextN layer. Quantized as the pack says (the manifest's `parts.mtp`,
    or `mtplx_mtp_quantization`), else int4/64; a Qwen head's norms
    shifted unless the manifest says they already are (`norm_convention`
    `mlx`), a NextN head's never (its RMSNorm is the plain kind)."""
    text = getattr(model, "language_model", model)
    args = text.args
    files, config = bundled_head_files(model_path)
    manifest = part(read_manifest(model_path), "mtp") or {}
    weights: dict = {}
    for file in files:
        raw = mx.load(str(file))
        mine = {k: v for k, v in raw.items() if "mtp." in k or _nextn_keys(config, k)}
        if mine and not any("mtp." in k for k in mine):
            # NextN tensors in the checkpoint's own names: the trunk's
            # sanitize stacks the experts and splits the latent projection
            # as it does for its own layers, once they sit at a layer index
            # it keeps.
            renamed = {"model.layers.0." + _strip(k): v for k, v in mine.items()}
            mine = text.sanitize(renamed) if hasattr(text, "sanitize") else renamed
        weights.update({_strip(k): v for k, v in mine.items()})
    if not weights:
        raise ValueError(f"{files[0]} holds no draft-head tensors")
    form = head_form(weights)
    if manifest.get("form") not in (None, form):
        raise ValueError(
            f"manifest parts.mtp.form says {manifest['form']!r}, the tensors are "
            f"a {form} head"
        )
    if form == "qwen_mtp":
        layer_class = type(text.model.layers[args.full_attention_interval - 1])
        head = MTPHead(args, layer_class)
        probe = "layers.0.self_attn.k_proj"
    else:
        weights = _to_head_form(weights)
        layer_class = type(text.model.layers[-1])
        head = NextNHead(
            args,
            layer_class,
            own_embed="embed_tokens.weight" in weights,
            own_head="head.weight" in weights,
        )
        probe = next(
            f"layers.0.self_attn.{n}"
            for n in ("q_a_proj", "kv_a_proj_with_mqa", "q_proj")
            if f"layers.0.self_attn.{n}.weight" in weights
        )
    quant = (
        {k: manifest[k] for k in ("bits", "group_size") if k in manifest}
        or config.get("mtplx_mtp_quantization")
        or {}
    )
    packed = f"{probe}.scales" in weights
    if packed:
        # The packed width says the bits: K / (32 / bits) uint32 per row of
        # a projection that reads the hidden state.
        w = weights[f"{probe}.weight"]
        sc = weights[f"{probe}.scales"]
        bits = 32 * w.shape[1] // args.hidden_size
        group_size = args.hidden_size // sc.shape[1]
        nn.quantize(head, class_predicate=_quant_predicate(bits, group_size))
    else:
        # int4 / 64 measured equal to bf16 in acceptance (2026-09-19, 27B);
        # a width 64 does not divide takes the largest group that does.
        default_group = next(g for g in (64, 32, 16, 8) if args.hidden_size % g == 0)
        bits = int(quant.get("bits", 4))
        group_size = int(quant.get("group_size", default_group))
    shifted = form == "qwen_mtp" and manifest.get("norm_convention", "hf") != "mlx"
    if shifted:
        weights = shift_norms(weights)
    head.load_weights(list(weights.items()), strict=True)
    if not packed:
        nn.quantize(head, class_predicate=_quant_predicate(bits, group_size))
    mx.eval(head.parameters())
    return head, {
        "kind": "mtp",
        "form": form,
        "source": "bundled",
        "files": [str(f.relative_to(model_path)) for f in files],
        "bits": bits,
        "group_size": group_size,
        "prequantized": packed,
        "norms_shifted": shifted,
        "own_embedding": form == "glm_nextn" and hasattr(head, "embed_tokens"),
        "own_head": form == "glm_nextn" and hasattr(head, "head"),
        # Where the bits and norm convention came from: the package's
        # manifest, or the loader's own defaults. "verified" says the head
        # file was hashed against the manifest (bundled_head_files refuses a
        # part without a file, so a hash there was always checked).
        "manifest": bool(manifest),
        "verified": bool(manifest.get("file")) and bool(manifest.get("sha256")),
    }


# Pairs of prompt history the head keeps at most: a prompt longer than this
# is primed from its last HEAD_HISTORY positions on, as a fresh head that
# saw only those (MTPLX keeps the last 8192 from 16k on; the head's
# attention is one layer, its KV 4 KB per pair on the 27B head).
HEAD_HISTORY = 8192
# Pairs queued from plain steps before the head is fed again in one call;
# past this the history restarts from the queue alone.
PENDING_CAP = 512


class BundledHeadProposer:
    """The checkpoint's MTP head as proposer. Per row it keeps the head's KV
    cache over the pairs (hidden, next token) the target went through -
    primed over the prompt in the prefill, continued by every plain step
    and every verify - the head's history, worth a third of its acceptance
    (measured 2026-09-19: 2.39 against 2.23 tokens per cycle with and
    without the prompt primed). Pairs the head has not seen yet wait in a
    queue and go in with the next draft's first call, so a cycle costs
    `depth` head calls, no more."""

    kind = "mtp"

    def __init__(self, model: Any, head: nn.Module, depth: int, info: dict):
        self._inner, self._lm_head, self._embed = trunk(model)
        self.head = head
        own = getattr(head, "head", None)
        if own is not None:
            self._lm_head = own
        self.depth = depth
        self.info = info
        self._cache: dict[int, list] = {}
        self._pending: dict[int, tuple[list[mx.array], list[int]]] = {}
        self._chain: dict[int, int] = {}
        # uid -> the hidden state of the last prefilled position, whose
        # partner token comes with the next chunk or the first decode step.
        self._tail: dict[int, mx.array] = {}
        self._prompt_len: dict[int, int] = {}
        self.primed_pairs = 0
        self.restored = 0

    # -- history --------------------------------------------------------------

    def _queue(self, uid: int, hs: list[mx.array], ts: list[int]) -> None:
        old_h, old_t = self._pending.get(uid, ([], []))
        hs, ts = old_h + hs, old_t + ts
        if len(ts) > PENDING_CAP:
            # Long plain runs (a row beside others): the queue goes into the
            # head now, in one call, so the history stays whole - throwing
            # the cache away would cost a third of the acceptance later.
            self._pending[uid] = (hs, ts)
            self._flush(uid)
            return
        self._pending[uid] = (hs, ts)

    def _flush(self, uid: int):
        """The queued pairs into the head in one call; nothing pending after."""
        hs, ts = self._pending.pop(uid, ([], []))
        if not ts:
            return None
        cache = self._cache.get(uid)
        if cache is None:
            cache = self._cache[uid] = [KVCache()]
        out = self.head(
            mx.concatenate(hs, axis=1),
            mx.array([ts], dtype=mx.int32),
            self._embed,
            cache,
        )
        mx.eval(out, *cache[0].state)
        return out

    def _feed(self, uid: int, hidden: mx.array, tokens: mx.array):
        """Queued pairs first, then these: one head call, its output the
        head's state at the last of them."""
        cache = self._cache.get(uid)
        if cache is None:
            cache = self._cache[uid] = [KVCache()]
        hs, ts = self._pending.pop(uid, ([], []))
        h_in = mx.concatenate([*hs, hidden], axis=1) if hs else hidden
        t_in = (
            mx.concatenate([mx.array([ts], dtype=tokens.dtype), tokens], axis=1)
            if ts
            else tokens
        )
        return self.head(h_in, t_in, self._embed, cache), cache

    def begin(self, uid, prompt_len, snapshot):
        self.drop(uid)
        self._prompt_len[uid] = prompt_len
        if snapshot is not None:
            keys, values, tail = snapshot
            if keys.shape[2] > HEAD_HISTORY:
                # The window applies to a restored history as to a primed
                # one: the last HEAD_HISTORY pairs (their positions are in
                # the keys already, the head attends over what is there).
                keys = keys[:, :, -HEAD_HISTORY:, :]
                values = values[:, :, -HEAD_HISTORY:, :]
            cache = KVCache()
            # Distinct array objects: the store's copy must not see the
            # in-place writes the cache makes from here on.
            cache.state = (keys[:], values[:], keys.shape[2])
            self._cache[uid] = [cache]
            if tail is not None:
                self._tail[uid] = tail[:]
            self.restored += 1

    def prime(self, uid, hidden, tokens, start):
        total = self._prompt_len.get(uid)
        window = 0 if total is None else max(0, total - HEAD_HISTORY)
        n = len(tokens)
        # Pair (tail, tokens[0]) sits at position start - 1; the chunk's own
        # pairs at start .. start + n - 2; the last hidden waits.
        tail = self._tail.pop(uid, None)
        hs, ts = [], []
        if tail is not None and start - 1 >= window:
            hs.append(tail)
            ts.append(int(tokens[0]))
        lo = max(0, window - start)
        if lo < n - 1:
            hs.append(hidden[:, lo : n - 1, :])
            ts.extend(int(t) for t in tokens[lo + 1 :])
        if hs:
            out, cache = self._feed(
                uid, mx.concatenate(hs, axis=1), mx.array([ts], dtype=mx.int32)
            )
            mx.eval(out, *cache[0].state)
            self.primed_pairs += len(ts)
        self._tail[uid] = hidden[:, n - 1 : n, :]

    def follow(self, uid, hidden, token):
        tail = self._tail.pop(uid, None)
        if hidden is None:
            hidden = tail
            if hidden is None:
                return  # nothing primed (or the window left it out)
        self._queue(uid, [hidden], [int(token.item())])

    def snapshot(self, uid):
        if uid not in self._cache and uid not in self._pending:
            return None
        cache = self._cache.get(uid)
        hs, ts = self._pending.pop(uid, ([], []))
        if ts:
            # Feed the queue so the snapshot is one cache, not a queue too.
            _, cache = self._feed(
                uid, mx.concatenate(hs, axis=1), mx.array([ts], dtype=mx.int32)
            )
        keys, values = cache[0].keys_and_values()
        tail = self._tail.get(uid)
        # Contiguous copies: the tail is a slice of a whole prefill chunk's
        # hidden state, which would otherwise stay alive in the store.
        arrays = [
            mx.contiguous(keys),
            mx.contiguous(values),
            None if tail is None else mx.contiguous(tail),
        ]
        mx.eval(*[a for a in arrays if a is not None])
        return arrays

    # -- the cycle ------------------------------------------------------------

    def propose(self, uid, hidden, token, choose=None, depth=None):
        depth = self.depth if depth is None else depth
        self._tail.pop(uid, None)
        drafts = []
        out, cache = self._feed(uid, hidden, token[None])
        out = out[:, -1:]
        for i in range(depth):
            logits = self._lm_head(out)[:, -1, :]
            draft = mx.argmax(logits, axis=-1) if choose is None else choose(logits, i)
            drafts.append(draft)
            if len(drafts) == depth:
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
            self._queue(uid, [hidden], list(tokens))

    def drop(self, uid: int) -> None:
        self._cache.pop(uid, None)
        self._pending.pop(uid, None)
        self._chain.pop(uid, None)
        self._tail.pop(uid, None)
        self._prompt_len.pop(uid, None)

    def describe(self) -> dict:
        return {
            **self.info,
            "depth": self.depth,
            "rows_with_history": len(self._cache),
            "primed_pairs": self.primed_pairs,
            "histories_restored": self.restored,
            "history_window": HEAD_HISTORY,
        }
