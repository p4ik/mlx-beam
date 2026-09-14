"""B.E.A.M. - Batched Engine for Apple Metal."""

try:
    from mlx_beam._version import __version__
except ImportError:  # editable install without a build step
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
