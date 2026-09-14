# Copyright © 2023 Apple Inc.
# Vendored from ml-explore/mlx-lm (MIT); see ../../../VENDORED.md.

import os

# Upstream version string at the vendored commit (setup.py / _version.py).
__version__ = "0.32.0"
UPSTREAM_COMMIT = "dcbcf786c0cf56f9a12fabe9468c887781431ae2"

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

from .generate import batch_generate, generate, stream_generate
from .utils import load

__all__ = [
    "__version__",
    "UPSTREAM_COMMIT",
    "batch_generate",
    "generate",
    "stream_generate",
    "load",
]
