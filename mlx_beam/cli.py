"""The `beam` command line."""

import argparse
import json
import platform
import sys

from mlx_beam import __version__


def doctor(as_json: bool = False) -> int:
    """Report what this machine offers: Python, MLX, device, memory."""
    report = {
        "mlx_beam": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "mlx": None,
        "device": None,
        "memory_gb": None,
        "ok": False,
        "error": None,
    }
    try:
        import mlx.core as mx
    except ModuleNotFoundError:
        report["error"] = "mlx is not installed (it runs on Apple silicon only)"
    except ImportError as e:
        # Installed but the extension did not load (Metal unavailable, wrong wheel).
        report["error"] = f"mlx is installed but failed to load: {e!r}"
    else:
        report["mlx"] = getattr(mx, "__version__", "?")
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else {}
            report["device"] = info.get("device_name") or str(mx.default_device())
            mem = info.get("memory_size")
            if mem:
                report["memory_gb"] = round(mem / 2**30, 1)
            report["ok"] = True
        except Exception as e:  # noqa: BLE001 - the error text is the finding
            report["error"] = f"mlx loaded but the device query failed: {e!r}"
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        width = max(len(k) for k in report)
        for key, value in report.items():
            print(f"{key:<{width}}  {value}")
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="beam", description="B.E.A.M. - Batched Engine for Apple Metal"
    )
    parser.add_argument(
        "--version", action="version", version=f"mlx-beam {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    p_doctor = sub.add_parser("doctor", help="show Python, MLX, device and memory")
    p_doctor.add_argument("--json", action="store_true", help="machine-readable output")
    p_serve = sub.add_parser("serve", help="serve a model over the OpenAI API")
    add_serve_arguments(p_serve)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(as_json=args.json)
    if args.command == "serve":
        return serve(args)
    parser.print_help()
    return 0


def add_serve_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="local path or Hugging Face repo id")
    p.add_argument("--model-alias", help="model id shown to clients (default: --model)")
    p.add_argument(
        "--reasoning-field",
        default="reasoning",
        choices=("reasoning", "reasoning_content", "both", "none"),
        help="where a chat completion carries the model's thinking: the field "
        "name(s), or none to leave the think markers in the content",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    limits = p.add_argument_group(
        "token limits", "every value counts tokens; each one counts a different set"
    )
    limits.add_argument(
        "--max-context",
        type=int,
        help="prompt plus generated tokens, a hard cap: a request over it is a "
        "400 (default: the model's own context length)",
    )
    limits.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="generated tokens when the client sends no max_tokens / "
        "max_completion_tokens / max_output_tokens; the request overrides "
        "(mlx-lm: --max-tokens, default 512)",
    )
    sampling = p.add_argument_group(
        "sampling defaults",
        "used when the client sends nothing; a flag beats the model's "
        "generation_config.json, which beats mlx-lm's defaults",
    )
    sampling.add_argument("--temp", type=float, help="temperature")
    sampling.add_argument("--top-p", type=float, help="nucleus sampling")
    sampling.add_argument("--top-k", type=int, help="top-k sampling (0 = off)")
    sampling.add_argument("--min-p", type=float, help="min-p sampling (0 = off)")
    kv = p.add_argument_group("KV cache")
    kv.add_argument(
        "--kv-bits",
        type=int,
        choices=(4, 8),
        help="quantize the full-attention KV cache to this many bits",
    )
    kv.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        choices=(32, 64, 128),
        help="group size of the KV quantization",
    )
    kv.add_argument(
        "--kv-config",
        help='JSON file, bits per layer: {"bits": 4, "group_size": 64, "layers": {"3": 8}}',
    )
    batching = p.add_argument_group("batching")
    batching.add_argument(
        "--decode-concurrency",
        type=int,
        default=8,
        help="sequences decoded in one batch",
    )
    batching.add_argument(
        "--prompt-concurrency",
        type=int,
        default=2,
        help="prompts prefilled in one batch",
    )
    batching.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="prompt tokens per model call while prefilling",
    )
    batching.add_argument(
        "--prefill-slice",
        type=int,
        default=512,
        help="prompt tokens a prefill runs before decode gets a turn",
    )
    batching.add_argument(
        "--decode-share",
        type=float,
        default=0.5,
        help="share of the worker's time decode keeps while a prefill runs (0-1)",
    )
    cache = p.add_argument_group("prompt cache")
    cache.add_argument(
        "--prompt-cache-size",
        type=int,
        default=16,
        help="stored prefixes: a number of entries, not a size in bytes",
    )
    cache.add_argument(
        "--prompt-cache-bytes",
        type=int,
        help="RAM budget in bytes for the stored prefixes (default: unlimited); "
        "stored prefixes yield to the caches of running requests",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="run a model_file shipped inside the checkpoint",
    )
    p.add_argument("--log-level", default="INFO")


def kv_policy_from_args(args):
    from mlx_beam.engine import KVPolicy

    bits, group_size, layers = args.kv_bits, args.kv_group_size, {}
    if args.kv_config:
        with open(args.kv_config) as f:
            cfg = json.load(f)
        bits = cfg.get("bits", bits)
        group_size = cfg.get("group_size", group_size)
        layers = {int(k): v for k, v in (cfg.get("layers") or {}).items()}
    return KVPolicy(bits=bits, group_size=group_size, layers=layers)


def serve(args) -> int:
    import logging
    from pathlib import Path

    from mlx_beam._vendor.mlx_lm.utils import hf_repo_to_path, load
    from mlx_beam.api.defaults import RequestDefaults
    from mlx_beam.engine import Engine, EngineDead
    from mlx_beam.server import Served
    from mlx_beam.server import serve as run_server

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("beam")
    log.info("loading %s", args.model)
    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)
    model_path = (
        Path(args.model) if Path(args.model).exists() else hf_repo_to_path(args.model)
    )
    defaults = RequestDefaults.resolve(
        model_path,
        flags={
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        },
    )
    policy = kv_policy_from_args(args)
    engine = Engine(
        model,
        model_key=args.model,
        kv_policy=policy,
        decode_concurrency=args.decode_concurrency,
        prompt_concurrency=args.prompt_concurrency,
        prefill_step_size=args.prefill_step_size,
        prefill_slice=args.prefill_slice,
        decode_share=args.decode_share,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=args.prompt_cache_bytes or None,
        max_context=args.max_context,
    )
    try:
        engine.start()
    except EngineDead as e:
        log.error("%s", e)
        return 3
    served = Served(
        engine,
        tokenizer,
        args.model_alias or args.model,
        reasoning_field=args.reasoning_field,
        defaults=defaults,
    )
    health = served.health()
    applied = health["kv"]["applied"] or []
    quantized = sum(1 for c in applied if "bits" in c)
    log.info(
        "ready: %d layers, %d with quantized KV (%s), batching %s, capabilities %s, "
        "reasoning field %s, wired limit %s",
        len(applied),
        quantized,
        policy.describe(),
        health["batching"],
        health["capabilities"],
        args.reasoning_field,
        health["wired_limit"],
    )
    log.info(
        "defaults: %s",
        ", ".join(
            f"{k}={v['value']} ({v['source']})" for k, v in defaults.describe().items()
        ),
    )
    try:
        run_server(served, args.host, args.port)
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
