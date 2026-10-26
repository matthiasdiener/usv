"""Tests for the usv pytest plugin (run inner pytest sessions via pytester)."""

from __future__ import annotations

import re

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

# Most tests don't exercise the session setup; keep them off the host's CPUs/SMI tools.
_QUIET = ("--benchmark-no-cpu-pin", "--benchmark-gpu-interference=off")


def _run(pytester, *args):
    return pytester.runpytest_subprocess(*_QUIET, *args)


def _write_baseline(pytester, median_ms):
    results = pytester.path / "results"
    results.mkdir(exist_ok=True)
    (results / "unknown-base.csv").write_text(
        _CSV_HEADER + f"test_cmp,3,1,,,{median_ms},{median_ms},0.0,{median_ms},{median_ms}\n"
    )


def test_runs_and_reports(pytester):
    pytester.makepyfile(_BENCH)
    result = _run(pytester)
    result.assert_outcomes(passed=3)  # 2 parametrized benchmarks + 1 plain test
    result.stdout.fnmatch_lines(["*usv benchmarks*", "*test_kernel?bf16?*"])


def test_benchmark_only_deselects_non_benchmarks(pytester):
    pytester.makepyfile(_BENCH)
    result = _run(pytester, "--benchmark-only")
    result.assert_outcomes(passed=2, deselected=1)


def test_not_benchmark_marker_skips_benchmarks(pytester):
    pytester.makepyfile(_BENCH)
    result = _run(pytester, "-m", "not benchmark")
    result.assert_outcomes(passed=1, deselected=2)


def test_benchmark_save_writes_csv(pytester):
    pytester.makepyfile(_BENCH)
    result = _run(pytester, "--benchmark-save=unit")
    result.assert_outcomes(passed=3)
    csvs = list(pytester.path.glob("results/*.csv"))
    assert csvs, "expected a saved results CSV under ./results"
    info = list(pytester.path.glob("results/*-unit-info.txt"))
    assert info and "cpus:" in info[0].read_text()


def test_fwd_bwd_records_two_rows(pytester):
    pytester.makepyfile(_FWDBWD)
    result = _run(pytester)
    result.assert_outcomes(passed=1)
    out = result.stdout.str()
    assert "test_fb [fwd]" in out and "test_fb [bwd]" in out


def test_compare_pass_when_faster_than_baseline(pytester):
    pytester.makepyfile(_CMP)
    _write_baseline(pytester, median_ms=100.0)  # baseline much slower -> no regression
    result = _run(pytester, "--benchmark-compare=base", "--benchmark-compare-fail=median:5%")
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*comparison vs*", "*test_cmp*ok*"])


def test_compare_fails_session_on_regression(pytester):
    pytester.makepyfile(_CMP)
    _write_baseline(pytester, median_ms=0.0001)  # baseline much faster -> regression
    result = _run(pytester, "--benchmark-compare=base", "--benchmark-compare-fail=median:5%")
    result.assert_outcomes(passed=1)  # the test itself still passes
    assert result.ret != 0  # but the session is gated to failure
    result.stdout.fnmatch_lines(["*benchmark regression: test_cmp*"])


def test_compare_missing_baseline_is_reported(pytester):
    pytester.makepyfile(_CMP)
    result = _run(pytester, "--benchmark-compare=nope")
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*no baseline found*"])


_ROUNDS = '''
import pytest

ORDER = []

@pytest.mark.parametrize("n", [1, 2, 3])
def test_k(usv_benchmark, n):
    ORDER.append(n)
    usv_benchmark(lambda: None, warmup=0, iters=5, timer="wall")

def test_plain():
    assert True
'''


def test_interleave_rounds_pool_samples(pytester):
    pytester.makepyfile(_ROUNDS)
    result = _run(pytester, "--benchmark-interleave=2", "--benchmark-save=il", "-v")
    result.assert_outcomes(passed=7)  # 3 benchmarks x 2 rounds + 1 plain test
    lines = [l for l in result.stdout.lines if "PASSED" in l]
    assert "test_plain" in lines[0]  # non-benchmarks first, then round 1, round 2
    assert all("-r1]" in l for l in lines[1:4]) and all("-r2]" in l for l in lines[4:7])
    (csv,) = pytester.path.glob("results/*-il.csv")
    rows = {r.split(",")[0]: r.split(",")[1] for r in csv.read_text().splitlines()[1:]}
    # One row per benchmark, pooling ceil(5/2) = 3 samples from each of 2 rounds.
    assert rows == {"test_k[1]": "6", "test_k[2]": "6", "test_k[3]": "6"}


def test_interleave_rejects_zero_rounds(pytester):
    pytester.makepyfile(_ROUNDS)
    result = _run(pytester, "--benchmark-interleave=0")
    assert result.ret == 4  # usage error


_BACKWARD = '''
import time

def test_bwd(usv_benchmark):
    usv_benchmark.backward(lambda: 1, lambda out: time.sleep(0.0002), block_calls=2,
                           min_blocks=3, min_time=0.0, timer="wall")
'''


def test_backward_records_bwd_row(pytester):
    pytester.makepyfile(_BACKWARD)
    result = _run(pytester)
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*test_bwd ?bwd?*"])


_FAKE_NEIGHBOR = '''
import usv.interference as interference
import usv.sysinfo as sysinfo

sysinfo._gpu_props = lambda index=0: {"pci": "0000:75:00.0"}
interference.snapshot_gpu_processes = lambda **kw: {}
interference.foreign_processes = lambda *a, **kw: [{"pid": 4242, "name": "hog", "memory_mb": None}]
'''


def test_interference_warns(pytester):
    pytester.makeconftest(_FAKE_NEIGHBOR)
    pytester.makepyfile(_CMP)
    result = pytester.runpytest_subprocess("--benchmark-no-cpu-pin")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*GPU INTERFERENCE: 1 other process*", "*PID 4242*hog*"])


def test_interference_abort(pytester):
    pytester.makeconftest(_FAKE_NEIGHBOR)
    pytester.makepyfile(_CMP)
    result = pytester.runpytest_subprocess(
        "--benchmark-no-cpu-pin", "--benchmark-gpu-interference=abort"
    )
    assert result.ret == 3


def test_interference_unknown_pci_is_reported(pytester):
    pytester.makeconftest(_FAKE_NEIGHBOR.replace('"pci": "0000:75:00.0"', '"name": "gpu"'))
    pytester.makepyfile(_CMP)
    result = pytester.runpytest_subprocess(
        "--benchmark-no-cpu-pin", "--benchmark-gpu-interference=abort"
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*interference check unavailable (unknown PCI address*"])


def test_cpu_pinning_applied_at_first_benchmark(pytester):
    pytester.makepyfile(
        _CMP
        + '''
import os

def test_zz_affinity():
    print("AFFINITY", ",".join(map(str, sorted(os.sched_getaffinity(0)))))
'''
    )
    result = pytester.runpytest_subprocess("--benchmark-gpu-interference=off", "-s")
    result.assert_outcomes(passed=2)
    out = result.stdout.str()
    # The number of cores depends on the host (2 on GitHub's 4-vCPU runners).
    pinned = re.search(r"usv: main thread pinned to cpus (\d+(?:,\d+)*)", out).group(1)
    assert f"AFFINITY {pinned}" in out


_SAME_NAME = '''
import time

def test_k(usv_benchmark):
    usv_benchmark(lambda: time.sleep(0.0002), warmup=0, iters=3, timer="wall")
'''


def test_same_name_in_different_files_is_not_pooled(pytester):
    pytester.makepyfile(test_a=_SAME_NAME, test_b=_SAME_NAME)
    result = _run(pytester, "--benchmark-save=two")
    result.assert_outcomes(passed=2)
    (csv,) = pytester.path.glob("results/*-two.csv")
    rows = [r.split(",")[:2] for r in csv.read_text().splitlines()[1:]]
    assert rows == [["test_k", "3"], ["test_k", "3"]]

