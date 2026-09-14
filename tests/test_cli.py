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
