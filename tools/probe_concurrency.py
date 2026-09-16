#!/usr/bin/env python3
"""A short request beside a long prefill - the stall check, on a real model.

    tools/probe_concurrency.py --model <path or repo> [--long 16000] [--slice 512]

Measures the time to first token of a short request three ways: alone,
while a long prompt is being prefilled with the default slice, and with the
slice the caller asks for. Prints the long prompt's own time as well, so the
cost of the slicing is visible next to its benefit.
"""

import argparse
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_beam._vendor.mlx_lm.utils import load  # noqa: E402
from mlx_beam.engine import Engine, GenerationRequest, KVPolicy  # noqa: E402


def ttft(engine, tokens, max_tokens=8):
    t0 = time.perf_counter()
    stream = engine.submit(GenerationRequest(tokens, max_tokens=max_tokens))
    it = iter(stream)
    next(it)
    first = time.perf_counter() - t0
    for _ in it:
        pass
    return first, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--long", type=int, default=16000, help="long prompt, tokens")
    ap.add_argument("--slice", type=int, default=512)
    ap.add_argument("--kv-bits", type=int, choices=(4, 8))
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)
    vocab = len(tokenizer)
    rng = random.Random(1)
    short = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Give this chat a three-word title."}],
        add_generation_prompt=True,
    )

    def long_prompt():
        return [rng.randrange(1, min(vocab, 30000)) for _ in range(args.long)]

    policy = KVPolicy(bits=args.kv_bits)
    for slice_w in (2048, args.slice):
        with Engine(
            model, kv_policy=policy, prefill_slice=slice_w, prompt_cache_size=0
        ) as engine:
            solo_first, solo_total = ttft(engine, list(short))
            long_alone, _ = ttft(engine, long_prompt(), max_tokens=1)
            result = {}

            def run_long(eng, tokens, out):
                out["long"] = ttft(eng, tokens, max_tokens=1)

            t = threading.Thread(target=run_long, args=(engine, long_prompt(), result))
            t.start()
            time.sleep(min(2.0, long_alone / 10))
            shared_first, shared_total = ttft(engine, list(short))
            t.join()
        print(
            f"slice {slice_w:>5}: short alone {solo_first:.2f}s (total {solo_total:.2f}s) | "
            f"long alone {long_alone:.1f}s | beside the long prefill: short first token "
            f"{shared_first:.2f}s, total {shared_total:.2f}s; long {result['long'][0]:.1f}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
