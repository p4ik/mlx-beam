"""What the package must bring along for the processors to load."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _requirement_names(requirements):
    return {r.split(";")[0].split(">")[0].split("=")[0].strip() for r in requirements}


def test_the_processors_runtime_is_a_dependency():
    # transformers' processors for the served families are written on
    # torch and torchvision; without them AutoProcessor refuses every
    # checkpoint and the package serves no image at all.
    with PYPROJECT.open("rb") as f:
        deps = tomllib.load(f)["project"]["dependencies"]
    assert {"torch", "torchvision"} <= _requirement_names(deps)
