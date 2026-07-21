"""Tests for the asv-style benchmark class runner (wall timer, no GPU)."""

from __future__ import annotations

from usv import run_benchmarks
from usv.asv import _axes, _label


class _Suite:
    params = [2, 3]
    param_names = ["n"]

    def __init__(self):
        self.calls = []

    def setup(self, n):
        self.n = n
        self.calls.append(("setup", n))

    def teardown(self, n):
        self.calls.append(("teardown", n))

    def time_square(self, n):
        return n * n


class _NoParams:
    def time_noop(self):
        return 1


class _Cached:
    def setup_cache(self):
        return 41

    def time_plus_one(self, cache):
        return cache + 1


def test_axes_normalization():
    assert _axes(None) == []
    assert _axes([1, 2, 3]) == [[1, 2, 3]]  # flat list -> single axis
    assert _axes([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]  # list of lists -> product


def test_label_formatting():
    assert _label(_Suite, "time_square", ["n"], (2,)) == "_Suite.time_square(n=2)"
    assert _label(_NoParams, "time_noop", [], ()) == "_NoParams.time_noop"


def test_run_parametrized_class():
    res = run_benchmarks(_Suite, warmup=0, iters=2, timer="wall")
    assert set(res) == {"_Suite.time_square(n=2)", "_Suite.time_square(n=3)"}
    for m in res.values():
        assert m.n == 2


def test_run_no_params_class():
    res = run_benchmarks(_NoParams, warmup=0, iters=2, timer="wall")
    assert set(res) == {"_NoParams.time_noop"}


def test_setup_teardown_invoked():
    suite = _Suite()
    run_benchmarks(suite, warmup=0, iters=1, timer="wall")
    kinds = [c[0] for c in suite.calls]
    assert kinds.count("setup") == 2 and kinds.count("teardown") == 2


def test_setup_cache_passed_as_first_arg():
    res = run_benchmarks(_Cached, warmup=0, iters=2, timer="wall")
    assert set(res) == {"_Cached.time_plus_one"}
