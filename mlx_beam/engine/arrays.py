"""Arrays an object graph holds, evaluated before another thread uses them.

A lazy array carries the stream of the thread that built it, and no other
thread can evaluate it. The core loads in one thread and computes in others
(the engine's worker, the request handlers), so what a load leaves lazy -
rope tables, a tower's rotary table, a head's constants - is evaluated here
first, wherever in the object graph it sits.
"""

from __future__ import annotations

import types

import mlx.core as mx

# Libraries whose objects hold no mx.array: a processor's object graph is
# large, and a provider's own package need not be named here.
_FOREIGN = frozenset(
    {
        "PIL",
        "mistral_common",
        "numpy",
        "tokenizers",
        "torch",
        "torchvision",
        "transformers",
        "transformers_modules",
    }
)

# Not entered: a module would take the walk through its whole namespace.
_OPAQUE = (types.ModuleType, type, types.FunctionType, types.MethodType)


def _foreign(obj) -> bool:
    # The class's module name, read from the type: no property of the
    # object runs, and a class may have none.
    module = str(getattr(type(obj), "__module__", None))
    return module.split(".")[0] in _FOREIGN


def materialize(*roots) -> int:
    """Evaluate every mx.array reachable from `roots`; the count evaluated.

    Reachable: the entries of a dict (an nn.Module is one) and its instance
    attributes, a plain object's instance attributes, the elements of lists
    and tuples. Not seen: `__slots__`, an array made after this call, and an
    array only a function holds (a closure cell, a default argument, a
    functools.partial)."""
    # Visited objects are held until the walk ends: a freed object's id can
    # come back on another one, which would then count as visited.
    seen: dict[int, object] = {}
    arrays: list[mx.array] = []
    stack = list(roots)
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        if isinstance(obj, mx.array):
            arrays.append(obj)
            continue
        if isinstance(obj, _OPAQUE) or _foreign(obj):
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
        try:
            stack.extend(vars(obj).values())
        except TypeError:
            pass  # no __dict__: a lock, a generator, a builtin value
    if arrays:
        mx.eval(*arrays)
    return len(arrays)
