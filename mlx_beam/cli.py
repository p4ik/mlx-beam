"""The `beam` command line."""

import argparse
import hashlib
import json
import os
import platform
import stat
import sys
from pathlib import Path

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
    p.add_argument("--host", default="127.0.0.1", help="address to listen on")
    p.add_argument(
        "--port", type=_positive_int, default=8000, help="TCP port to listen on"
    )
    p.add_argument(
        "--allowed-origins",
        nargs="+",
        default=[],
        metavar="ORIGIN",
        help="pages a browser may call the server from and read the answers of; "
        "any other page's Origin gets 403, requests without an Origin header "
        "(CLI, SDKs, curl) are not affected; null and empty entries are refused; "
        "by default none",
    )
    access = p.add_argument_group("access")
    access.add_argument(
        "--api-key",
        metavar="KEY",
        help="the key every request must carry (Authorization: Bearer or "
        "x-api-key); required off loopback unless --skip-api-key",
    )
    access.add_argument(
        "--skip-api-key",
        action="store_true",
        help="serve without a key on a non-loopback --host, on purpose",
    )
    access.add_argument(
        "--allowed-hosts",
        nargs="+",
        default=[],
        metavar="HOST",
        help="Host header values accepted next to localhost and --host, e.g. "
        "the machine's name or LAN address behind a wildcard bind; other "
        "hosts get 403",
    )
    template = p.add_argument_group("chat template")
    template.add_argument(
        "--chat-template",
        help="Jinja text, or the path of a .jinja file, used instead of the "
        "model's own template",
    )
    template.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="give a model that ships no chat template a plain ChatML one",
    )
    template.add_argument(
        "--chat-template-args",
        type=_json_object,
        default={},
        metavar="JSON",
        help="handed to every template render, e.g. '{\"enable_thinking\": false}'; "
        "a request's chat_template_kwargs override it",
    )
    limits = p.add_argument_group(
        "token limits", "every value counts tokens; each one counts a different set"
    )
    limits.add_argument(
        "--max-context",
        type=_positive_int,
        help="prompt plus generated tokens, a hard cap: a prompt whose reserve "
        "does not fit is a 400, a larger max_tokens is served capped at what "
        "the context holds (default: the model's own context length)",
    )
    limits.add_argument(
        "--max-prompt-tokens",
        type=_positive_int,
        help="prompt tokens, a hard cap below the context: a longer prompt is "
        "a 400; a request's max_prompt_tokens may only lower it",
    )
    limits.add_argument(
        "--max-completion-tokens",
        type=_positive_int,
        default=None,
        help="generated tokens when the client sends no max_tokens / "
        "max_completion_tokens / max_output_tokens; the request overrides "
        "(mlx-lm: --max-tokens, default 512)",
    )
    limits.add_argument(
        "--max-reasoning-tokens",
        type=_count,
        help="reasoning tokens (what usage.reasoning_tokens counts) when the "
        "client sends no max_reasoning_tokens; the think block is closed by "
        "force at the budget (default: unbounded); 0 switches reasoning off "
        "for requests that do not ask for it - a request that switches "
        "thinking on (enable_thinking, an effort word, thinking.type enabled) "
        "is bound by its room alone",
    )
    limits.add_argument(
        "--min-response-tokens",
        type=_count,
        default=0,
        help="tokens kept for the answer after the think block when the client "
        "sends no min_response_tokens; the reasoning budget is cut to leave "
        "them, and a request whose context cannot hold them is a 400",
    )
    sampling = p.add_argument_group(
        "sampling defaults",
        "used when the client sends nothing; a flag beats the model's "
        "generation_config.json, which beats mlx-lm's defaults",
    )
    sampling.add_argument("--temp", type=float, help="temperature")
    sampling.add_argument("--top-p", type=float, help="nucleus sampling")
    sampling.add_argument("--top-k", type=_count, help="top-k sampling (0 = off)")
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
        type=_positive_int,
        choices=(32, 64, 128),
        help="group size of the KV quantization (default: 64, or what the "
        "checkpoint's kv_config carries; the flag beats the file)",
    )
    kv.add_argument(
        "--kv-config",
        help='JSON file, bits per layer: {"bits": 4, "group_size": 64, "layers": '
        '{"3": 8}}, or the list a quantized package ships '
        '([{"layer_idx": 3, "bits": 4, "group_size": 64}, ...]): listed layers '
        "take their bits, unlisted ones follow --kv-bits (optiq leaves them "
        "at full precision and ignores --kv-bits)",
    )
    kv.add_argument(
        "--kv-prefill",
        choices=("exact", "quantized"),
        help="when a quantized layer becomes quantized: 'exact' (default) keeps "
        "the prompt at model precision while it is prefilled and quantizes at "
        "the handover to decoding (mlx-lm's generate_step with "
        "quantized_kv_start at the prompt's end); 'quantized' writes it "
        "quantized from the first token, which saves the prompt's full-"
        "precision transient (~2 GB for a 64k prompt on a 27B) and on some "
        "models costs accuracy - use it for a profile that was measured with "
        'it (a kv_config object may carry "prefill": "quantized")',
    )
    p.add_argument(
        "--max-queued",
        type=_positive_int,
        help="requests allowed to wait for a batch slot; one more is a 503 with "
        "Retry-After (default: unlimited)",
    )
    batching = p.add_argument_group("batching")
    batching.add_argument(
        "--decode-concurrency",
        type=_positive_int,
        default=8,
        help="sequences decoded in one batch",
    )
    batching.add_argument(
        "--prompt-concurrency",
        type=_positive_int,
        default=2,
        help="prompts prefilled in one batch",
    )
    batching.add_argument(
        "--prefill-step-size",
        type=_positive_int,
        default=2048,
        help="prompt tokens per model call while prefilling",
    )
    batching.add_argument(
        "--prefill-slice",
        type=_positive_int,
        default=512,
        help="prompt tokens a prefill runs before decode gets a turn",
    )
    batching.add_argument(
        "--decode-share",
        type=lambda v: _unit_interval(v, "--decode-share"),
        metavar="SHARE",
        default=0.5,
        help="share of the worker's time decode keeps while a prefill runs (0-1)",
    )
    spec = p.add_argument_group(
        "speculative decoding",
        "off unless asked; a checkpoint that bundles a draft head says so at start",
    )
    spec.add_argument(
        "--draft-model",
        help="the proposer that drafts tokens for the verify pass: 'bundled' takes "
        "the draft head the checkpoint ships (the package manifest's parts.mtp, "
        "config.json mtp_file, or mtp.* tensors in the shards); a repo or path "
        "for an external drafter is not supported yet",
    )
    spec.add_argument(
        "--exact-verify",
        choices=("off", "kernels", "positions"),
        default="off",
        help="how the verify runs: 'off' checks the k+1 drafts in one forward "
        "(the fast path; /health.speculative.exact says whether that forward "
        "gives the same logits as one-token forwards on this machine); "
        "'kernels' runs that forward through projections and attention that "
        "keep single-row arithmetic for a block (vendored from mlx-vlm) and "
        "keeps them only if the warm-up finds them bit-equal, else falls back "
        "to 'off' and says so; 'positions' feeds one token per forward and "
        "stops at the first rejected draft - exact by construction at plain "
        "decoding's cost, the reference for the other two",
    )
    spec.add_argument(
        "--max-draft-tokens",
        type=_positive_int,
        default=3,
        help="cap on the drafts verified per cycle; the regulator picks each "
        "cycle's depth below it from the acceptance and the cycle cost it "
        "measures (default: 3, the depth with the best gain measured on a 27B)",
    )
    cache = p.add_argument_group("prompt cache")
    cache.add_argument(
        "--prompt-cache-size",
        type=_positive_int,
        default=16,
        help="stored prefixes: a number of entries, not a size in bytes",
    )
    cache.add_argument(
        "--prompt-cache-bytes",
        type=_positive_int,
        help="RAM budget in bytes for the stored prefixes (default: unlimited); "
        "the store's own limit, separate from the caches of running requests",
    )
    cache.add_argument(
        "--prompt-cache-dir",
        metavar="DIR",
        help="directory the store is written to on a clean stop and read back "
        "from after start, one folder per model revision and KV layout "
        "(default: none; the store lives in RAM only)",
    )
    cache.add_argument(
        "--prompt-cache-save-seconds",
        type=_non_negative_float,
        metavar="SECONDS",
        default=8.0,
        help="time budget for writing the store on a stop; system entries go "
        "first, then conversations newest first, the rest is dropped",
    )
    cache.add_argument(
        "--prompt-cache-disk-bytes",
        type=_positive_int,
        default=100 * 1024**3,
        help="cap on the whole --prompt-cache-dir across model revisions and "
        "layouts: when a save would exceed it, other folders go, the least "
        "recently saved first, never one another process holds",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="run a model_file shipped inside the checkpoint",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="how much the server log says",
    )


def _json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not JSON: {e.msg}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return value


# transformers' former default: ChatML, for models that ship no template.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def _unit_interval(value: str, flag: str) -> float:
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"{flag} must be between 0 and 1")
    return x


def _positive_int(value: str) -> int:
    """A count that must be at least 1: a prefill of width 0 never advances
    and a concurrency of 0 admits nobody - argparse refuses them here
    instead of the engine hanging on them."""
    try:
        x = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if x < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return x


def _non_negative_float(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
    if x < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return x


def _count(value: str) -> int:
    """A count that may be 0 (off, none reserved)."""
    try:
        x = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if x < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return x


def chat_template_from_args(args) -> str | None:
    """The --chat-template text: read from the file when the value names one.
    A value that looks like a file name but is none (a typo in the path)
    is refused - rendering the path's text as the template would be a
    silent fallback."""
    if not args.chat_template:
        return None
    value = args.chat_template
    path = Path(value)
    try:
        is_file = path.is_file()
    except OSError:
        # A template's text is longer than a file name may be.
        is_file = False
    if is_file:
        return path.read_text(encoding="utf-8")
    looks_like_path = (
        "{{" not in value
        and "{%" not in value
        and (value.endswith((".jinja", ".j2", ".txt")) or "/" in value)
    )
    if looks_like_path:
        raise StartupError(f"--chat-template {value}: no such file")
    return value


class StartupError(ValueError):
    """A flag, file or environment the server cannot start with; `serve`
    logs it and exits with 3, whether it surfaces before or after the
    load. argparse's own refusals keep exit code 2."""


def kv_policy_from_args(args):
    from mlx_beam.engine import KVPolicy

    bits, group_size, layers = args.kv_bits, args.kv_group_size, {}
    prefill, source = "exact", "default"
    if args.kv_config:
        try:
            with open(args.kv_config) as f:
                cfg = json.load(f)
            if isinstance(cfg, list):
                # The list a quantized package ships: only the layers it
                # names, each with its own bits; the rest follow --kv-bits.
                # A missing or null bits must not pass as "keep the layer
                # unquantized": that would silently undo --kv-bits for it.
                bad = [
                    e
                    for e in cfg
                    if not isinstance(e, dict)
                    or "layer_idx" not in e
                    or not isinstance(e.get("bits"), int)
                ]
                if bad:
                    raise ValueError(
                        f"every list entry needs layer_idx and integer bits, got {bad[0]!r}"
                    )
                layers = {int(e["layer_idx"]): e["bits"] for e in cfg}
                sizes = {e["group_size"] for e in cfg if "group_size" in e}
                if len(sizes) > 1:
                    raise ValueError(
                        f"one group size per policy, the list has {sorted(sizes)}"
                    )
                # A flag beats the file (the help text says so): only an
                # unset --kv-group-size takes the list's.
                if sizes and args.kv_group_size is None:
                    group_size = sizes.pop()
            elif not isinstance(cfg, dict) or not isinstance(
                cfg.get("layers", {}), dict
            ):
                raise ValueError("expected an object with bits, group_size, layers")
            else:
                if bits is None:
                    bits = cfg.get("bits")
                if args.kv_group_size is None:
                    group_size = cfg.get("group_size", group_size)
                layers = {int(k): v for k, v in (cfg.get("layers") or {}).items()}
                if "prefill" in cfg:
                    prefill, source = cfg["prefill"], "profile"
        except (OSError, TypeError, ValueError) as e:
            raise StartupError(f"--kv-config {args.kv_config}: {e}") from None
    if group_size is None:
        group_size = 64
    if getattr(args, "kv_prefill", None):
        prefill, source = args.kv_prefill, "flag"
    try:
        return KVPolicy(
            bits=bits,
            group_size=group_size,
            layers=layers,
            prefill=prefill,
            prefill_source=source,
        )
    except ValueError as e:
        raise StartupError(f"kv policy: {e}") from None


def policy_with_package_prefill(policy, args, model_path: Path, log):
    """The prefill mode a package measured for its own kv_config
    (`parts.kv_config.prefill.mode` in the manifest) applies when that is the
    profile in use - the file --kv-config names has the manifest's SHA-256,
    or, for a manifest without one, is the manifest's file by path - and
    neither the file nor a flag said otherwise. A profile edited in place
    is not the measured one; the hash is what tells."""
    from dataclasses import replace

    from mlx_beam.package import part, read_manifest, sha256_of

    if policy.prefill_source != "default" or not args.kv_config:
        return policy
    entry = part(read_manifest(model_path), "kv_config") or {}
    prefill = entry.get("prefill")
    mode = prefill.get("mode") if isinstance(prefill, dict) else None
    if not mode or not entry.get("file"):
        return policy
    ours = Path(args.kv_config)
    if entry.get("sha256"):
        same = sha256_of(ours).lower() == str(entry["sha256"]).lower()
    else:
        same = ours.resolve() == (model_path / entry["file"]).resolve()
    if not same:
        log.info(
            "kv prefill stays %r: %s is not the profile the package measured "
            "(%s); pass --kv-prefill to choose",
            policy.prefill,
            ours,
            entry["file"],
        )
        return policy
    log.info(
        "kv prefill %r from the package manifest (measured for %s)", mode, entry["file"]
    )
    return replace(policy, prefill=mode, prefill_source="manifest")


def prepare_prompt_cache_dir(path: str) -> Path:
    """The directory --prompt-cache-dir names, created with mode 0700; one
    that exists must be ours and private, or the server does not start:
    the cache is the conversations in another form."""
    directory = Path(path)
    if not directory.exists():
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)
        return directory
    if not directory.is_dir():
        raise StartupError(f"--prompt-cache-dir {path}: not a directory")
    st = directory.stat()
    if st.st_uid != os.getuid():
        raise StartupError(
            f"--prompt-cache-dir {path}: owned by uid {st.st_uid}, the server runs "
            f"as uid {os.getuid()}"
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        raise StartupError(
            f"--prompt-cache-dir {path}: mode {mode:04o} lets others in; make it 0700"
        )
    return directory


# The packages that compute an image's features; each may change what the
# same picture becomes without the frontend package changing.
VISION_LIBRARIES = ("transformers", "torch", "torchvision", "pillow", "mistral-common")


def _named_version(package: str, version) -> str:
    if not version or version == "0.0.0+unknown":
        raise StartupError(
            f"{package} names no version ({version!r}); a build without one cannot "
            "own a prompt cache namespace"
        )
    return version


def _installed(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _settings_digest(frontend) -> str:
    """SHA-256 of the processor's effective image settings: the image
    processor's `to_dict()`, or mistral-common's image config."""
    import dataclasses

    renderer = getattr(frontend, "renderer", None)
    if renderer is not None:
        try:
            config = renderer.tokenizer.instruct_tokenizer.image_encoder.image_config
        except AttributeError as e:
            raise StartupError(
                f"the renderer {type(renderer).__name__} names no image settings: {e}"
            ) from None
        settings = (
            dataclasses.asdict(config)
            if dataclasses.is_dataclass(config)
            else vars(config)
        )
    else:
        processor = getattr(
            getattr(frontend, "processor", None), "image_processor", None
        )
        if processor is None or not hasattr(processor, "to_dict"):
            raise StartupError(
                "the frontend's processor names no image settings "
                "(no image_processor.to_dict)"
            )
        settings = processor.to_dict()
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def cache_identity(model_path: Path, frontend, proposer) -> dict:
    """What a stored entry must match beyond its tokens: the model's files
    (hashed config and index, every weight file by path, size and mtime -
    no reading), the mlx and mlx-beam versions, the vision frontend's
    preprocessing and the draft head's digest."""
    import importlib

    import mlx.core as mx

    from mlx_beam.package import sha256_of

    def digest(name: str) -> str | None:
        file = model_path / name
        return sha256_of(file) if file.is_file() else None

    weights = []
    for file in sorted(model_path.rglob("*.safetensors")):
        st = file.stat()
        weights.append(
            [file.relative_to(model_path).as_posix(), st.st_size, st.st_mtime_ns]
        )
    # A Hugging Face snapshot carries its commit in the path; a local path
    # has none, its files tell.
    snapshot = model_path.name if model_path.parent.name == "snapshots" else None
    identity = {
        "model": {
            "snapshot": snapshot,
            "config": digest("config.json"),
            "index": digest("model.safetensors.index.json"),
            "kv_config": digest("kv_config.json"),
            "weights": weights,
        },
        "mlx": mx.__version__,
        "engine": _named_version("mlx-beam", __version__),
        "vision": None,
        "head_digest": None,
    }
    if frontend is not None:
        described = frontend.describe()
        top = type(frontend).__module__.partition(".")[0]
        package = getattr(importlib.import_module(top), "__version__", None)
        identity["vision"] = {
            "provider": described.get("provider"),
            "family": described.get("family"),
            "processor": described.get("processor"),
            "preprocessor_config": digest("preprocessor_config.json"),
            "processor_config": digest("processor_config.json"),
            "package": _named_version(top, package),
            "libraries": {name: _installed(name) for name in VISION_LIBRARIES},
            "settings": _settings_digest(frontend),
        }
    if proposer is not None:
        identity["head_digest"] = _head_digest(model_path, proposer.describe())
    return identity


def _head_digest(model_path: Path, info: dict) -> str | None:
    """What a draft head's sidecar must match: the head's files by path,
    size and mtime - the weights' measure, no reading - the package
    manifest's SHA-256 when it names one, and the quantization the loader
    gave them. None for a proposer without files of its own."""
    from mlx_beam.package import part, read_manifest

    files = info.get("files")
    if not files:
        return None
    stats = []
    for name in files:
        st = (model_path / name).stat()
        stats.append([name, st.st_size, st.st_mtime_ns])
    mtp = part(read_manifest(model_path), "mtp") or {}
    parts = {
        "files": stats,
        "sha256": mtp.get("sha256"),
        "form": info.get("form"),
        "bits": info.get("bits"),
        "group_size": info.get("group_size"),
        "norms_shifted": info.get("norms_shifted"),
    }
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def proposer_from_args(args, model, model_path: Path, log):
    """The proposer `--draft-model` names, or None. A checkpoint that ships
    a head but was started without the flag gets a hint, not a default."""
    from mlx_beam.engine.proposer import (
        BundledHeadProposer,
        bundled_head_files,
        load_bundled_head,
    )

    if not args.draft_model:
        try:
            files, _ = bundled_head_files(model_path, verify=False)
            log.info(
                "%s bundles a draft head (%s); pass --draft-model bundled to use it",
                args.model,
                ", ".join(str(f.relative_to(model_path)) for f in files),
            )
        except (OSError, ValueError):
            pass
        if args.max_draft_tokens != 3:
            log.info("--max-draft-tokens has no effect without --draft-model")
        return None
    check_draft_flags(args)
    head, info = load_bundled_head(model, model_path)
    return BundledHeadProposer(model, head, args.max_draft_tokens, info)


def check_draft_flags(args) -> None:
    """What --draft-model may name, checked before the load."""
    if args.draft_model and args.draft_model != "bundled":
        raise StartupError(
            "only --draft-model bundled (the checkpoint's own draft head) is "
            "supported at this stage; an external drafter comes with a later "
            "release"
        )


def default_flags(args) -> dict:
    """The request defaults the flags set (None: not set)."""
    return {
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "chat_template_args": args.chat_template_args or None,
        "max_reasoning_tokens": args.max_reasoning_tokens,
        "min_response_tokens": args.min_response_tokens or None,
    }


def probe_port(host: str, port: int) -> None:
    """Bind and release the listening socket once, before the load: an
    occupied port or a bad host fails in a second, not after the weights."""
    import socket

    try:
        family, kind, _, _, address = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )[0]
    except (OSError, IndexError) as e:
        raise StartupError(f"cannot listen on {host}:{port}: {e}") from None
    with socket.socket(family, kind) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(address)
        except OSError as e:
            raise StartupError(f"cannot listen on {host}:{port}: {e}") from None


def serve(args) -> int:
    import logging

    from mlx_beam import modalities, renderer
    from mlx_beam._vendor.mlx_lm.utils import _download, hf_repo_to_path, load
    from mlx_beam.api.access import Access, check_allowed_origins
    from mlx_beam.api.defaults import RequestDefaults
    from mlx_beam.engine import Engine, EngineDead
    from mlx_beam.server import Served
    from mlx_beam.server import serve as run_server

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("beam")
    # Flags are checked before minutes of loading are spent on a typo: the
    # KV policy, the template file, the draft flag, the request defaults
    # the flags set, the access policy, the origin list, and the port.
    try:
        policy = kv_policy_from_args(args)
        template = chat_template_from_args(args)
        check_draft_flags(args)
        RequestDefaults.resolve(None, flags=default_flags(args))
        access = Access.resolve(
            args.host, args.api_key, args.skip_api_key, args.allowed_hosts
        )
        check_allowed_origins(args.allowed_origins)
        probe_port(args.host, args.port)
        if args.prompt_cache_dir:
            prepare_prompt_cache_dir(args.prompt_cache_dir)
    except (StartupError, ValueError, OSError) as e:
        log.error("%s", e)
        return 3
    if access.mode != "loopback":
        log.warning(
            "listening on %s, reachable from other machines: access %s, hosts %s",
            args.host,
            "by key" if access.mode == "key" else "open (--skip-api-key)",
            ", ".join(access.hosts),
        )
        if access.wildcard and not args.allowed_hosts:
            log.warning(
                "a wildcard bind answers only to localhost and the bind address "
                "in the Host header; name the machine with --allowed-hosts"
            )
    log.info("loading %s", args.model)
    # The files come first (a download when the model is a repo id): the
    # renderer is a tokenizer argument, so it is decided before the load.
    rendering = renderer.choose(_download(args.model), template)
    if rendering.note:
        log.info("template: %s", rendering.note)
    # Handed to the tokenizer at load: the tool markers are inferred from
    # the template that will actually render (the think markers come from
    # the vocabulary). Custom tokenizer code needs the same consent the
    # model's does.
    tokenizer_config = {"chat_template": template} if template else {}
    tokenizer_config.update(rendering.tokenizer_kwargs)
    if args.trust_remote_code:
        tokenizer_config["trust_remote_code"] = True
    model, tokenizer = load(
        args.model,
        tokenizer_config=tokenizer_config,
        trust_remote_code=args.trust_remote_code,
    )
    template_source = "flag" if template else "model"
    if rendering.name == renderer.MISTRAL_COMMON:
        template_source = renderer.MISTRAL_COMMON
        parser = renderer.install_tool_parser(tokenizer)
        log.info(
            "template: mistral-common renders the prompt; tool calls %s",
            f"parsed by {parser}" if parser else "not in the vocabulary",
        )
    if args.use_default_chat_template and not tokenizer.has_chat_template:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        tokenizer.has_chat_template = True
        template_source = "default"
    model_path = (
        Path(args.model) if Path(args.model).exists() else hf_repo_to_path(args.model)
    )
    try:
        policy = policy_with_package_prefill(policy, args, model_path, log)
    except (OSError, ValueError) as e:
        log.error("package manifest: %s", e)
        return 3
    try:
        proposer = proposer_from_args(args, model, model_path, log)
    except (OSError, KeyError, ValueError) as e:
        log.error("--draft-model %s: %s", args.draft_model, e)
        return 3
    try:
        defaults = RequestDefaults.resolve(model_path, flags=default_flags(args))
    except ValueError as e:
        # The model's generation_config.json holds a value out of range.
        log.error("request defaults: %s", e)
        return 3
    # A separate package may serve this checkpoint's images (entry point
    # group mlx_beam.modalities); without one the core is text only. Loaded
    # before the engine: the store's identity names the frontend.
    frontend, vision_refused = modalities.load_frontend(
        model,
        model_path,
        json.loads((model_path / "config.json").read_text()),
        tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if frontend is not None and template_source in ("flag", "default"):
        # The flag's template renders image requests too, not only text.
        frontend.chat_template = tokenizer.chat_template
    identity = None
    if args.prompt_cache_dir:
        try:
            identity = cache_identity(model_path, frontend, proposer)
        except (StartupError, OSError) as e:
            log.error("--prompt-cache-dir %s: %s", args.prompt_cache_dir, e)
            return 3
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
        max_prompt_tokens=args.max_prompt_tokens,
        max_queued=args.max_queued,
        proposer=proposer,
        exact_verify=args.exact_verify,
        max_draft_tokens=args.max_draft_tokens,
        prompt_cache_dir=args.prompt_cache_dir or None,
        prompt_cache_save_seconds=args.prompt_cache_save_seconds,
        prompt_cache_disk_bytes=args.prompt_cache_disk_bytes,
        cache_identity=identity,
    )
    try:
        engine.start()
    except EngineDead as e:
        log.error("%s", e)
        return 3
    if engine.speculator is not None:
        log.info("speculative: %s", engine.speculator.describe()["proposer"])
    served = Served(
        engine,
        tokenizer,
        args.model_alias or args.model,
        reasoning_field=args.reasoning_field,
        defaults=defaults,
        allowed_origins=args.allowed_origins,
        access=access,
        chat_template_source=template_source,
        frontend=frontend,
        vision_refused=vision_refused,
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
