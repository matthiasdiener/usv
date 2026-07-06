"""Basic tests for usv (no GPU required - uses the wall-clock timer)."""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from usv import Measurement, do_bench, do_bench_many, format_table, rotating
from usv.results import load_results, save_results, save_samples
from usv.timer import get_timer


# -- Timer tests ---------------------------------------------------------


def test_wall_timer_record_n():
    timer = get_timer("wall")
    times = timer.record_n(lambda: time.sleep(0.001), 3)
    assert times.shape == (3,)
    assert all(t >= 0.0005 for t in times)


def test_wall_timer_inner_divides():
    timer = get_timer("wall")
    # one sample of two 0.5ms sleeps -> ~0.5ms per call after dividing by inner
    times = timer.record_n(lambda: time.sleep(0.0005), 1, inner=2)
    assert times.shape == (1,)
    assert 0.0003 <= times[0] <= 0.005


# -- do_bench (single) ----------------------------------------------------


def test_do_bench_returns_measurement():
    m = do_bench(lambda: time.sleep(0.001), warmup=1, iters=3, timer="wall")
    assert isinstance(m, Measurement)
    assert m.n == 3
    assert m.median >= 0.0005


def test_do_bench_inner():
    m = do_bench(lambda: time.sleep(0.0005), warmup=0, iters=2, inner=2, timer="wall")
    assert m.inner == 2
    assert m.median >= 0.0003


# -- do_bench_many (interleaving) -----------------------------------------


def test_do_bench_many_interleaved():
    fns = {
        "a": lambda: time.sleep(0.0005),
        "b": lambda: time.sleep(0.0010),
    }
    res = do_bench_many(fns, warmup=0, iters=4, interleave=True, timer="wall")
    assert set(res) == {"a", "b"}
    assert res["a"].n == 4 and res["b"].n == 4
    # 'b' sleeps twice as long as 'a'
    assert res["b"].median > res["a"].median


def test_do_bench_many_sequential():
    fns = {"a": lambda: None, "b": lambda: None}
    res = do_bench_many(fns, warmup=0, iters=3, interleave=False, timer="wall")
    assert res["a"].n == 3 and res["b"].n == 3


def test_do_bench_many_visits_round_robin():
    # Interleaving must visit every benchmark once per round (order within a
    # round is shuffled).
    order: list[str] = []
    fns = {
        "a": lambda: order.append("a"),
        "b": lambda: order.append("b"),
    }
    do_bench_many(fns, warmup=0, iters=3, interleave=True, timer="wall")
    # drop the pre-warmup pass (one discarded call per callable) before checking
    timed = order[len(fns) :]
    assert len(timed) == 6
    assert timed.count("a") == 3 and timed.count("b") == 3
    # each round (a pair) contains exactly one of each
    for i in range(0, 6, 2):
        assert set(timed[i : i + 2]) == {"a", "b"}


def test_min_iters_time_raises_count():
    # iters floor is 1, but a ~0.5ms kernel must run long enough to fill 10ms.
    m = do_bench(lambda: time.sleep(5e-4), warmup=0, iters=1, min_iters_time=0.01, timer="wall")
    assert m.n > 1


def test_min_iters_time_per_fn_dropout():
    fns = {
        "fast": lambda: time.sleep(2e-4),
        "slow": lambda: time.sleep(3e-3),
    }
    res = do_bench_many(
        fns, warmup=0, iters=1, min_iters_time=0.015, interleave=True, timer="wall"
    )
    # The faster callable needs many more samples to fill the same budget.
    assert res["fast"].n >= 1 and res["slow"].n >= 1
    assert res["fast"].n > res["slow"].n


def test_pre_warmup_call_discarded():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    m = do_bench(fn, warmup=0, iters=3, inner=1, timer="wall")
    # one pre-warmup call (discarded) plus three timed samples
    assert calls["n"] == 4
    assert m.n == 3


# -- Measurement ----------------------------------------------------------


def test_measurement_stats():
    m = Measurement(samples=np.array([1.0, 2.0, 3.0]), name="x")
    assert m.median == 2.0
    assert m.mean == 2.0
    assert m.min == 1.0 and m.max == 3.0


def test_measurement_throughput():
    m = Measurement(samples=np.array([1e-3]), flops=1e9, bytes=1e6)
    assert abs(m.tflops - 1.0) < 1e-9
    assert abs(m.gbps - 1.0) < 1e-9


def test_rotating():
    nxt = rotating([10, 20, 30])
    assert [nxt() for _ in range(7)] == [10, 20, 30, 10, 20, 30, 10]


def test_flush_bytes_sizing():
    from usv.bench import _l2_cache_size_bytes, _resolve_flush_bytes

    # An explicit size (MB) is honored exactly.
    assert _resolve_flush_bytes(128) == 128 * 1024 * 1024

    # Auto: 2x L2 when known, else a 256 MB fallback.
    auto = _resolve_flush_bytes(None)
    l2 = _l2_cache_size_bytes()
    if l2:
        assert auto == 2 * l2
    else:
        assert auto == 256 * 1024 * 1024


def test_format_table():
    ms = [
        Measurement(samples=np.array([1e-3, 1.1e-3]), name="a"),
        Measurement(samples=np.array([2e-3]), name="b", flops=1e9),
    ]
    table = format_table(ms)
    assert "benchmark" in table
    assert "TFLOP/s" in table  # shown because one measurement has flops


# -- Result I/O -----------------------------------------------------------


def test_save_load_results():
    ms = {
        "a": Measurement(samples=np.array([1e-3, 2e-3]), name="a", flops=1e9),
        "b": Measurement(samples=np.array([3e-3]), name="b"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.csv")
        assert save_results(ms, path) == path
        rows = load_results(path)
        assert {r["name"] for r in rows} == {"a", "b"}
        a = next(r for r in rows if r["name"] == "a")
        assert a["n"] == "2" and float(a["median_ms"]) > 0
        assert a["tflops"] != ""  # flops was provided
        b = next(r for r in rows if r["name"] == "b")
        assert b["tflops"] == ""  # no flops -> blank throughput


def test_save_samples():
    ms = {
        "a": Measurement(samples=np.array([1e-3, 2e-3, 3e-3]), name="a"),
        "b": Measurement(samples=np.array([5e-3]), name="b"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "samples.csv")
        assert save_samples(ms, path) == path
        rows = load_results(path)
        # one row per raw sample across all benchmarks
        assert len(rows) == 4
        assert [r["name"] for r in rows].count("a") == 3
        a0 = next(r for r in rows if r["name"] == "a" and r["sample_idx"] == "0")
        assert abs(float(a0["time_ms"]) - 1.0) < 1e-6  # 1e-3 s -> 1 ms
        b0 = next(r for r in rows if r["name"] == "b")
        assert b0["sample_idx"] == "0" and abs(float(b0["time_ms"]) - 5.0) < 1e-6
