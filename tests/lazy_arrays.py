"""The thread probe's helpers: a plain object that holds an array, an array
built here and never evaluated (it carries this thread's stream), and a
computation of it in a worker thread - which MLX refuses for a lazy array
built in another thread; the tests read that refusal."""

from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx


class Table:
    """A plain object with an array: hung on a module it lands in the
    instance's __dict__, outside parameters() and state."""

    def __init__(self, values):
        self.values = values


def lazy(n=4):
    # Built here and not evaluated: carries this thread's stream.
    return mx.arange(n, dtype=mx.float32) * 2.0


def computed_in_another_thread(array):
    """`array + 1`, built and evaluated in a worker thread. The executor
    raises what the worker raised; a bare Thread would only warn."""

    def task():
        out = array + 1
        mx.eval(out)
        return out.tolist()

    with ThreadPoolExecutor(1) as pool:
        return pool.submit(task).result()
