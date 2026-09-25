"""mlx-beam-vision: the towers that let mlx-beam see."""

try:
    from mlx_beam_vision._version import __version__
except ImportError:  # pragma: no cover - source tree without a build
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
