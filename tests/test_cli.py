import json
import re

import pytest

from mlx_beam import __version__
from mlx_beam.cli import doctor, main

PEP440 = re.compile(
    r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?(\+[0-9A-Za-z.]+)?$"
)


def test_version_is_pep440():
    assert PEP440.match(__version__), __version__


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_doctor_json_has_the_fields(capsys):
    doctor(as_json=True)
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {
        "mlx_beam",
        "python",
        "platform",
        "mlx",
        "device",
        "memory_gb",
        "ok",
        "error",
    }


def test_doctor_exit_code_follows_mlx(capsys):
    """Exit 0 iff mlx imports; on Apple silicon it must."""
    import platform

    try:
        import mlx.core  # noqa: F401

        has_mlx = True
    except ImportError:
        has_mlx = False
    rc = doctor(as_json=True)
    capsys.readouterr()
    assert rc == (0 if has_mlx else 1)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert has_mlx, "mlx must be installed on Apple silicon"


def test_doctor_keeps_the_load_error(monkeypatch, capsys):
    """An mlx that is installed but fails to load is reported with its error."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("dlopen: Metal not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    rc = doctor(as_json=True)
    report = json.loads(capsys.readouterr().out)
    assert rc == 1 and report["ok"] is False
    assert (
        "failed to load" in report["error"] and "Metal not available" in report["error"]
    )


def test_doctor_reports_missing_package(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise ModuleNotFoundError("No module named 'mlx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    rc = doctor(as_json=True)
    report = json.loads(capsys.readouterr().out)
    assert rc == 1 and "not installed" in report["error"]


def test_serve_kv_policy_from_arguments(tmp_path):
    from mlx_beam.cli import kv_policy_from_args, main

    cfg = tmp_path / "kv.json"
    cfg.write_text('{"bits": 4, "group_size": 32, "layers": {"3": 8}}')
    import argparse

    args = argparse.Namespace(kv_bits=None, kv_group_size=64, kv_config=str(cfg))
    policy = kv_policy_from_args(args)
    assert (policy.bits, policy.group_size, policy.layers) == (4, 32, {3: 8})
    assert policy.bits_for(3) == 8 and policy.bits_for(0) == 4
    with pytest.raises(SystemExit):
        main(["serve", "--kv-bits", "3", "--model", "x"])
    # A package's list names the layers it quantizes; the rest follow --kv-bits.
    listed = tmp_path / "kv_config.json"
    listed.write_text(
        '[{"layer_idx": 3, "bits": 4, "group_size": 32}, {"layer_idx": 7, "bits": 4, "group_size": 32}]'
    )
    policy = kv_policy_from_args(
        argparse.Namespace(kv_bits=8, kv_group_size=64, kv_config=str(listed))
    )
    assert (policy.bits, policy.group_size, policy.layers) == (8, 32, {3: 4, 7: 4})
    assert policy.bits_for(7) == 4 and policy.bits_for(11) == 8
    policy = kv_policy_from_args(
        argparse.Namespace(kv_bits=None, kv_group_size=64, kv_config=str(listed))
    )
    assert policy.bits is None and policy.bits_for(11) is None
    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        '[{"layer_idx": 3, "bits": 4, "group_size": 32}, {"layer_idx": 7, "bits": 4, "group_size": 64}]'
    )
    with pytest.raises(SystemExit, match="one group size"):
        kv_policy_from_args(
            argparse.Namespace(kv_bits=8, kv_group_size=64, kv_config=str(mixed))
        )
    nameless = tmp_path / "nameless.json"
    nameless.write_text('[{"bits": 4}]')
    with pytest.raises(SystemExit, match="layer_idx"):
        kv_policy_from_args(
            argparse.Namespace(kv_bits=8, kv_group_size=64, kv_config=str(nameless))
        )
    # Without bits an entry must be an error, not "this layer stays at model
    # precision" behind the caller's back (nor with an explicit null).
    for text in ('[{"layer_idx": 3}]', '[{"layer_idx": 3, "bits": null}]'):
        bitless = tmp_path / "bitless.json"
        bitless.write_text(text)
        with pytest.raises(SystemExit, match="bits"):
            kv_policy_from_args(
                argparse.Namespace(kv_bits=8, kv_group_size=64, kv_config=str(bitless))
            )


def serve_args(*argv):
    import argparse

    from mlx_beam.cli import add_serve_arguments

    p = argparse.ArgumentParser()
    add_serve_arguments(p)
    return p.parse_args(["--model", "m", *argv])


def test_serve_flags_carry_mlx_lm_names_and_units():
    args = serve_args(
        "--model-alias",
        "alias",
        "--prompt-cache-bytes",
        "1073741824",
        "--max-completion-tokens",
        "64",
        "--decode-concurrency",
        "4",
        "--allowed-origins",
        "http://a",
        "http://b",
        "--chat-template-args",
        '{"enable_thinking": false}',
    )
    assert args.model_alias == "alias" and args.prompt_cache_bytes == 2**30
    assert args.max_completion_tokens == 64 and args.decode_concurrency == 4
    assert args.allowed_origins == ["http://a", "http://b"]
    assert args.chat_template_args == {"enable_thinking": False}
    assert serve_args().allowed_origins == ["*"] and serve_args().temp is None
    with pytest.raises(SystemExit):
        serve_args("--chat-template-args", "[1]")
    with pytest.raises(SystemExit):
        serve_args("--served-name", "x")


def test_chat_template_flag_takes_text_or_a_file(tmp_path):
    from mlx_beam.cli import chat_template_from_args

    path = tmp_path / "t.jinja"
    path.write_text("{{ messages }}")
    assert chat_template_from_args(serve_args("--chat-template", str(path))) == (
        "{{ messages }}"
    )
    assert (
        chat_template_from_args(serve_args("--chat-template", "{{ x }}")) == "{{ x }}"
    )
    assert chat_template_from_args(serve_args()) is None
    # A real template is longer than a file name may be; still text.
    long_template = "{% for m in messages %}{{ m.content }}{% endfor %}" * 12
    assert (
        chat_template_from_args(serve_args("--chat-template", long_template))
        == long_template
    )


def test_bad_flags_fail_before_the_model_loads(tmp_path):
    import argparse

    from mlx_beam.cli import kv_policy_from_args, main

    bad = tmp_path / "kv.json"
    bad.write_text('{"bits": 4, "layers": [1, 2]}')
    with pytest.raises(SystemExit, match="kv-config"):
        kv_policy_from_args(
            argparse.Namespace(kv_bits=None, kv_group_size=64, kv_config=str(bad))
        )
    with pytest.raises(SystemExit, match="kv-config"):
        kv_policy_from_args(
            argparse.Namespace(kv_bits=None, kv_group_size=64, kv_config="/nowhere")
        )
    with pytest.raises(SystemExit):
        main(["serve", "--model", "x", "--log-level", "LOUD"])
    with pytest.raises(SystemExit):
        main(["serve", "--model", "x", "--decode-share", "1.5"])
    assert serve_args("--log-level", "debug").log_level == "DEBUG"
    assert serve_args().max_completion_tokens is None


def test_kv_prefill_comes_from_the_flag_the_profile_or_the_default(tmp_path):
    from mlx_beam.cli import kv_policy_from_args

    policy = kv_policy_from_args(serve_args("--kv-bits", "8"))
    assert (policy.prefill, policy.prefill_source) == ("exact", "default")
    # A package profile measured with the thrifty mode says so in its object form.
    profile = tmp_path / "kv.json"
    profile.write_text(
        '{"bits": 8, "group_size": 64, "layers": {}, "prefill": "quantized"}'
    )
    policy = kv_policy_from_args(serve_args("--kv-config", str(profile)))
    assert (policy.prefill, policy.prefill_source) == ("quantized", "profile")
    # The flag is the operator's word and beats the profile - visibly.
    policy = kv_policy_from_args(
        serve_args("--kv-config", str(profile), "--kv-prefill", "exact")
    )
    assert (policy.prefill, policy.prefill_source) == ("exact", "flag")
    # The package list carries no such field; the flag still applies.
    listed = tmp_path / "kv_config.json"
    listed.write_text('[{"layer_idx": 3, "bits": 4, "group_size": 64}]')
    policy = kv_policy_from_args(
        serve_args(
            "--kv-bits", "8", "--kv-config", str(listed), "--kv-prefill", "quantized"
        )
    )
    assert (policy.prefill, policy.prefill_source) == ("quantized", "flag")
    bad = tmp_path / "bad.json"
    bad.write_text('{"bits": 8, "prefill": "later"}')
    with pytest.raises(SystemExit, match="prefill"):
        kv_policy_from_args(serve_args("--kv-config", str(bad)))
