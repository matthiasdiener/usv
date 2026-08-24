"""Tests for the usv pytest plugin (run inner pytest sessions via pytester)."""

from __future__ import annotations

# The plugin is installed via its pytest11 entry point, so inner pytester
# sessions autoload it automatically -- no explicit registration needed.
_BENCH = '''
import time
import pytest

@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_kernel(usv_benchmark, dtype):
    usv_benchmark(lambda: time.sleep(0.0005), warmup=0, iters=3,
                  timer="wall", flops=1e9)

def test_plain():
    assert True
'''

_FWDBWD = '''
import time

def test_fb(usv_benchmark):
    usv_benchmark.fwd_bwd(
        lambda: time.sleep(0.0003),
        lambda: time.sleep(0.0009),
        fwd_flops=1e9, warmup=0, iters=3, timer="wall",
    )
'''

_CMP = '''
import time

def test_cmp(usv_benchmark):
    usv_benchmark(lambda: time.sleep(0.001), warmup=0, iters=3, timer="wall")
'''

_CSV_HEADER = "name,n,inner,tflops,gbps,median_ms,mean_ms,stdev_ms,min_ms,max_ms\n"


def _write_baseline(pytester, median_ms):
    results = pytester.path / "results"
    results.mkdir(exist_ok=True)
    (results / "unknown-base.csv").write_text(
        _CSV_HEADER + f"test_cmp,3,1,,,{median_ms},{median_ms},0.0,{median_ms},{median_ms}\n"
    )


def test_runs_and_reports(pytester):
    pytester.makepyfile(_BENCH)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=3)  # 2 parametrized benchmarks + 1 plain test
    result.stdout.fnmatch_lines(["*usv benchmarks*", "*test_kernel?bf16?*"])


def test_benchmark_only_deselects_non_benchmarks(pytester):
    pytester.makepyfile(_BENCH)
    result = pytester.runpytest_subprocess("--benchmark-only")
    result.assert_outcomes(passed=2, deselected=1)


def test_not_benchmark_marker_skips_benchmarks(pytester):
    pytester.makepyfile(_BENCH)
    result = pytester.runpytest_subprocess("-m", "not benchmark")
    result.assert_outcomes(passed=1, deselected=2)


def test_benchmark_save_writes_csv(pytester):
    pytester.makepyfile(_BENCH)
    result = pytester.runpytest_subprocess("--benchmark-save=unit")
    result.assert_outcomes(passed=3)
    csvs = list(pytester.path.glob("results/*.csv"))
    assert csvs, "expected a saved results CSV under ./results"


def test_fwd_bwd_records_two_rows(pytester):
    pytester.makepyfile(_FWDBWD)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    out = result.stdout.str()
    assert "test_fb [fwd]" in out and "test_fb [bwd]" in out


def test_compare_pass_when_faster_than_baseline(pytester):
    pytester.makepyfile(_CMP)
    _write_baseline(pytester, median_ms=100.0)  # baseline much slower -> no regression
    result = pytester.runpytest_subprocess(
        "--benchmark-compare=base", "--benchmark-compare-fail=median:5%"
    )
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*comparison vs*", "*test_cmp*ok*"])


def test_compare_fails_session_on_regression(pytester):
    pytester.makepyfile(_CMP)
    _write_baseline(pytester, median_ms=0.0001)  # baseline much faster -> regression
    result = pytester.runpytest_subprocess(
        "--benchmark-compare=base", "--benchmark-compare-fail=median:5%"
    )
    result.assert_outcomes(passed=1)  # the test itself still passes
    assert result.ret != 0  # but the session is gated to failure
    result.stdout.fnmatch_lines(["*benchmark regression: test_cmp*"])


def test_compare_missing_baseline_is_reported(pytester):
    pytester.makepyfile(_CMP)
    result = pytester.runpytest_subprocess("--benchmark-compare=nope")
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*no baseline found*"])

