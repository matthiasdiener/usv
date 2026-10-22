"""Tests for block-based backward timing (wall timer, no GPU needed)."""

from __future__ import annotations

from usv import do_bench_backward


def test_blocks_of_forwards_then_backwards():
    calls = []
    fwd = lambda: calls.append("f") or len(calls)  # noqa: E731
    bwd = lambda out: calls.append("b")  # noqa: E731
    m = do_bench_backward(fwd, bwd, block_calls=3, min_blocks=4, min_time=0.0, timer="wall", name="x")
    assert m.n == 4 and m.inner == 3 and m.name == "x"
    # One warmup block plus four timed ones, each three forwards then three backwards.
    assert "".join(calls) == "fffbbb" * 5


def test_runs_until_min_time():
    import time

    m = do_bench_backward(
        lambda: None, lambda out: time.sleep(0.002), block_calls=2, min_blocks=1, min_time=0.02, timer="wall"
    )
    assert m.n >= 5  # 0.02 s / (2 calls x 2 ms)
    assert 0.0015 < m.median < 0.02


def test_default_block_size():
    m = do_bench_backward(lambda: None, lambda out: None, min_blocks=2, min_time=0.0, timer="wall")
    assert m.inner == 20
