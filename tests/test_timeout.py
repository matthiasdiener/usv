"""Tests for the optional benchmark timeout (SIGALRM, main thread)."""

from __future__ import annotations

import time

import pytest

from usv import do_bench


def test_timeout_aborts_slow_benchmark():
    # Many slow samples would run for seconds; the timeout must abort quickly.
    with pytest.raises(TimeoutError):
        do_bench(
            lambda: time.sleep(0.05),
            warmup=0,
            iters=1000,
            timer="wall",
            timeout=0.1,
        )


def test_no_timeout_when_within_budget():
    m = do_bench(lambda: None, warmup=0, iters=2, timer="wall", timeout=5.0)
    assert m.n == 2


def test_timeout_none_is_noop():
    m = do_bench(lambda: None, warmup=0, iters=2, timer="wall", timeout=None)
    assert m.n == 2


def test_timeout_restores_signal_handler():
    import signal

    before = signal.getsignal(signal.SIGALRM)
    do_bench(lambda: None, warmup=0, iters=1, timer="wall", timeout=5.0)
    assert signal.getsignal(signal.SIGALRM) is before
