#!/usr/bin/env python3
"""Greedy output under different KV layouts - the check that the bits apply.

    tools/kv_raster.py --model <path or repo> [--tokens 64] [--prompt "..."]

Runs the same prompt greedily with fp16 KV, 8 bit, 4 bit and a mixed layout
(4 bit, every fourth full-attention layer 8 bit) and prints the token ids per
layout plus which layouts agree. Identical output under 4 and 8 bit means the
configured bits did not reach the cache. Also prints what /health would
report as the applied layout and the peak memory per run.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402

from mlx_beam._vendor.mlx_lm.models.cache import (  # noqa: E402
    KVCache,
    make_prompt_cache,
)
from mlx_beam._vendor.mlx_lm.utils import load  # noqa: E402
from mlx_beam.engine import Engine, GenerationRequest, KVPolicy  # noqa: E402


def run(model, tokens, policy, n):
    mx.reset_peak_memory()
    with Engine(model, kv_policy=policy) as engine:
        t0 = time.perf_counter()
        out = [e.token for e in engine.submit(GenerationRequest(tokens, max_tokens=n))]
        dt = time.perf_counter() - t0
        applied = engine.health()["kv"]["applied"]
    return out, dt, mx.get_peak_memory() / 2**30, applied


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Write two sentences about the sea.")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )
    attention = [
        i for i, c in enumerate(make_prompt_cache(model)) if type(c) is KVCache
    ]
    layouts = {
        "fp16": KVPolicy(),
        "8bit": KVPolicy(bits=8),
        "4bit": KVPolicy(bits=4),
        "mixed": KVPolicy(bits=4, layers={i: 8 for i in attention[::4]}),
    }
    results = {}
    for name, policy in layouts.items():
        out, dt, peak, applied = run(model, list(prompt), policy, args.tokens)
        results[name] = out
        quantized = [f"{e['bits']}" for e in applied if "bits" in e]
        print(
            f"{name:>6}: {len(out)} tokens in {dt:.1f}s, peak {peak:.2f} GB, "
            f"quantized layers {len(quantized)}/{len(applied)}"
            + (f" ({'/'.join(sorted(set(quantized)))} bit)" if quantized else "")
        )
        print("        ", tokenizer.decode(out)[:160].replace("\n", " "))
    names = list(results)
    print()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            same = results[a] == results[b]
            print(f"{a} vs {b}: {'IDENTICAL' if same else 'differ'}")
    if results["4bit"] == results["8bit"] == results["fp16"]:
        print(
            "\nall layouts agree - either the bits did not apply or the prompt is too easy"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
