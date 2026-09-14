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
    }
    try:
        import mlx.core as mx

        report["mlx"] = mx.__version__
        info = mx.device_info() if hasattr(mx, "device_info") else {}
        report["device"] = info.get("device_name") or str(mx.default_device())
        mem = info.get("memory_size")
        if mem:
            report["memory_gb"] = round(mem / 2**30, 1)
    except ImportError:
        report["mlx"] = "not installed (mlx runs on Apple silicon only)"
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        width = max(len(k) for k in report)
        for key, value in report.items():
            print(f"{key:<{width}}  {value}")
    return 0 if report["mlx"] and not str(report["mlx"]).startswith("not ") else 1


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
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor(as_json=args.json)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
