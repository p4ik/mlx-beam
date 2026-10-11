"""Every array an object graph holds is evaluated by `materialize`, so a
thread that did not build the graph can compute with it: arrays in a plain
object hung on a module, nested in lists and dicts, behind a cycle - and
none inside an object of a library that holds no arrays."""

import mlx.nn as nn
import pytest

from tests.lazy_arrays import Table, computed_in_another_thread, lazy


def test_the_probe_sees_a_lazy_array():
    """The probe rests on MLX refusing, in another thread, an array this
    thread built and never evaluated; the tests here read that refusal.
    Nothing of the engine is imported: the probe holds on its own."""
    table = Table(lazy())
    with pytest.raises(Exception, match="in current thread"):
        computed_in_another_thread(table.values)


def test_a_plain_attribute_of_a_module_is_evaluated():
    from mlx_beam.engine.arrays import materialize

    module = nn.Linear(2, 2)
    module.table = Table(lazy())
    assert "table" not in module and "table" in vars(module)
    assert materialize(module) == 3  # weight, bias and the table
    assert computed_in_another_thread(module.table.values) == [1.0, 3.0, 5.0, 7.0]


def test_arrays_nested_in_lists_and_dicts_are_evaluated():
    from mlx_beam.engine.arrays import materialize

    holder = Table([lazy(2), {"deep": (lazy(3), "text", 7)}])
    assert materialize(holder) == 2
    assert computed_in_another_thread(holder.values[0]) == [1.0, 3.0]
    assert computed_in_another_thread(holder.values[1]["deep"][0]) == [
        1.0,
        3.0,
        5.0,
    ]


def test_a_cycle_ends_the_walk_once():
    from mlx_beam.engine.arrays import materialize

    a, b = Table(lazy()), Table(lazy(2))
    a.other, b.other, a.me = b, a, a
    assert materialize(a) == 2
    assert computed_in_another_thread(b.values) == [1.0, 3.0]


def test_an_object_of_an_excluded_library_is_not_entered():
    from mlx_beam.engine.arrays import materialize

    class Processor:
        pass

    # The rule reads the class's module name, not the class itself.
    Processor.__module__ = "transformers.models.fake.processing_fake"
    processor = Processor()
    processor.table = Table(lazy())
    holder = Table(processor)
    assert materialize(holder) == 0
    with pytest.raises(Exception, match="in current thread"):
        computed_in_another_thread(processor.table.values)
