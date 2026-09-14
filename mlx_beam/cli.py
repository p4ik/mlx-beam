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
    p.add_argument("--served-name", help="model id shown to clients (default: --model)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--kv-bits", type=int, choices=(4, 8), help="quantize full-attention KV"
    )
    p.add_argument("--kv-group-size", type=int, default=64, choices=(32, 64, 128))
    p.add_argument(
        "--kv-config",
        help='JSON file: {"bits": 4, "group_size": 64, "layers": {"3": 8}}',
    )
    p.add_argument("--decode-concurrency", type=int, default=8)
    p.add_argument("--prompt-concurrency", type=int, default=2)
    p.add_argument("--prefill-step-size", type=int, default=2048)
    p.add_argument("--prefill-slice", type=int, default=512)
    p.add_argument("--decode-share", type=float, default=0.5)
    p.add_argument("--prompt-cache-size", type=int, default=16)
    p.add_argument(
        "--prompt-cache-gb", type=float, help="RAM budget for stored prefixes"
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

    from mlx_beam._vendor.mlx_lm.utils import load
    from mlx_beam.engine import Engine, EngineDead
    from mlx_beam.server import Served
    from mlx_beam.server import serve as run_server

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("beam")
    log.info("loading %s", args.model)
    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)
    policy = kv_policy_from_args(args)
    engine = Engine(
        model,
        model_key=args.model,
        kv_policy=policy,
        completion_batch_size=args.decode_concurrency,
        prefill_batch_size=args.prompt_concurrency,
        prefill_step_size=args.prefill_step_size,
        prefill_slice=args.prefill_slice,
        decode_share=args.decode_share,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_bytes=(
            int(args.prompt_cache_gb * 2**30) if args.prompt_cache_gb else None
        ),
    )
    try:
        engine.start()
    except EngineDead as e:
        log.error("%s", e)
        return 3
    served = Served(engine, tokenizer, args.served_name or args.model)
    health = served.health()
    applied = health["kv"]["applied"] or []
    quantized = sum(1 for c in applied if "bits" in c)
    log.info(
        "ready: %d layers, %d with quantized KV (%s), batching %s, capabilities %s",
        len(applied),
        quantized,
        policy.describe(),
        health["batching"],
        health["capabilities"],
    )
    try:
        run_server(served, args.host, args.port)
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
