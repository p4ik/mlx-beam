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
