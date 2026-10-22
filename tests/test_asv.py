"""Tests for the asv-style benchmark class runner (wall timer, no GPU)."""

from __future__ import annotations

import pytest

from usv import run_benchmarks
from usv.asv import _axes, _label, _param_ids, asv_bench_kwargs


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


class _Attrs:
    number = 4
    repeat = (7, 20, 10.0)
    warmup_time = 0.01
    timeout = 30

    def time_a(self):
        pass

    def time_b(self):
        pass

    time_b.number = 0
    time_b.sample_time = 0.002
    time_b.warmup_time = 0
    time_b.repeat = 9


def test_asv_attribute_mapping():
    assert asv_bench_kwargs(_Attrs, "time_a") == {
        "inner": 4, "iters": 7, "min_warmup_time": 0.01, "timeout": 30.0,
    }
    # Method attributes override the class's.
    assert asv_bench_kwargs(_Attrs, "time_b") == {
        "inner": "auto", "target_window_s": 0.002, "iters": 9, "warmup": 0, "timeout": 30.0,
    }
    assert asv_bench_kwargs(_NoParams, "time_noop") == {}


def test_attributes_apply_and_explicit_kwargs_win():
    res = run_benchmarks(_Attrs, warmup=0, timer="wall")
    assert res["_Attrs.time_a"].n == 7 and res["_Attrs.time_a"].inner == 4
    res = run_benchmarks(_Attrs, warmup=0, iters=3, timer="wall")
    assert res["_Attrs.time_a"].n == 3


def test_param_ids():
    assert _param_ids(["n", "dtype"], (1024, "bf16")) == ["1024", "bf16"]
    assert _param_ids(["shape"], ((1, 2),)) == ["shape0"]


def test_interleave_pools_rounds():
    res = run_benchmarks(_Suite, interleave=2, warmup=0, iters=3, timer="wall")
    assert {m.n for m in res.values()} == {4}  # ceil(3/2) samples x 2 rounds


def test_instance_runs_directly_without_pytest(monkeypatch):
    import usv.asv as asv

    monkeypatch.setattr(asv, "_run_via_pytest", lambda *a, **kw: pytest.fail("used pytest"))
    res = run_benchmarks(_Suite(), warmup=0, iters=1, timer="wall")
    assert set(res) == {"_Suite.time_square(n=2)", "_Suite.time_square(n=3)"}


_SUITE_FILE = '''
class Suite:
    params = [[2, 3], ["a"]]
    param_names = ["n", "tag"]

    def setup(self, n, tag):
        if n == 3:
            raise NotImplementedError

    def time_square(self, n, tag):
        return n * n


class TestNotAsv:
    def time_helper(self):
        pass

    def test_it(self):
        pass
'''

_QUIET = ("--benchmark-no-cpu-pin", "--benchmark-gpu-interference=off", "--benchmark-timer=wall")


def test_pytest_collects_asv_suites(pytester):
    pytester.makepyfile(bench_suite=_SUITE_FILE)
    result = pytester.runpytest_subprocess(*_QUIET, "--benchmark-asv", "-v")
    # n=3 is skipped by setup; TestNotAsv stays a test class.
    result.assert_outcomes(passed=2, skipped=1)
    result.stdout.fnmatch_lines([
        "*bench_suite.py::Suite::time_square?2-a? PASSED*",
        "*Suite.time_square(n=2, tag='a')*",
    ])


def test_asv_suites_need_opt_in(pytester):
    pytester.makepyfile(bench_suite=_SUITE_FILE)
    result = pytester.runpytest_subprocess(*_QUIET)
    assert result.ret == 5  # nothing collected


def test_asv_in_test_files_and_k_selection(pytester):
    pytester.makepyfile(test_suite=_SUITE_FILE)
    result = pytester.runpytest_subprocess(*_QUIET, "--benchmark-asv", "-k", "Suite and 2")
    result.assert_outcomes(passed=1, deselected=2)  # TestNotAsv::test_it is a normal test


def test_asv_interleave_rounds(pytester):
    pytester.makepyfile(bench_suite=_SUITE_FILE)
    result = pytester.runpytest_subprocess(
        *_QUIET, "--benchmark-asv", "--benchmark-interleave=3", "-v"
    )
    result.assert_outcomes(passed=4, skipped=3)  # 2 asv items x 3 rounds + test_it
    result.stdout.fnmatch_lines(["*time_square?2-a-r1? PASSED*", "*time_square?2-a-r3? PASSED*"])
