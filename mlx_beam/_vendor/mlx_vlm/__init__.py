# Files taken from the mlx-vlm package (MIT); see ../../../VENDORED.md.
# The exact-verify kernels: quantized projections that give, for a block of
# T rows, the bytes T single-row calls would give. The engine uses them for
# its `kernels` verify mode; the import of the switch layer points at the
# vendored mlx-lm, otherwise the files are upstream's.

UPSTREAM_VERSION = "0.7.1"
