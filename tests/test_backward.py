"""Tests for block-based backward timing (wall timer, no GPU needed)."""

from __future__ import annotations

import pytest

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
    total = m.samples.sum() * m.inner
    assert total >= 0.02
    assert total - m.samples[-1] * m.inner < 0.02  # stopped at the first block that reached it
    assert m.min >= 0.002


def test_default_block_size():
    m = do_bench_backward(lambda: None, lambda out: None, min_blocks=2, min_time=0.0, timer="wall")
    assert m.inner == 20


def _fake_torch(monkeypatch, graph_bytes, free_bytes):
    import sys
    import types

    allocated = iter([0, graph_bytes, 0])  # before, after the forward, after freeing it
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        memory_allocated=lambda: next(allocated),
        memory_reserved=lambda: graph_bytes,  # the freed forward stays cached
        mem_get_info=lambda: (free_bytes, 288 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_block_size_falls_to_one_forward(monkeypatch):
    import usv.bench as bench

    _fake_torch(monkeypatch, 12 * 1024**3, 200 * 1024**3)
    # A 12 GiB forward under the 16 GiB cap: one forward per block, not two.
    assert bench._bwd_block_calls(lambda: None, 16 * 1024**3) == 1


def test_block_size_fits_free_memory(monkeypatch):
    import usv.bench as bench

    _fake_torch(monkeypatch, 1024**3, 7 * 1024**3)
    # 7 GiB free + 1 GiB cached: half of 8 GiB holds four 1 GiB forwards, not 16.
    assert bench._bwd_block_calls(lambda: None, 16 * 1024**3) == 4


@pytest.mark.parametrize("block_calls", [0, -2])
def test_rejects_non_positive_block_calls(block_calls):
    with pytest.raises(ValueError, match="block_calls"):
        do_bench_backward(lambda: None, lambda out: None, block_calls=block_calls, timer="wall")
