"""Test the opt-in cool-down sleep (wall timer, no GPU)."""

from __future__ import annotations

import time

from usv import do_bench


def test_cooldown_adds_idle_time():
    t0 = time.perf_counter()
    do_bench(lambda: None, warmup=0, iters=1, timer="wall", cooldown_s=0.15)
    assert time.perf_counter() - t0 >= 0.15


def test_no_cooldown_by_default():
    t0 = time.perf_counter()
    do_bench(lambda: None, warmup=0, iters=1, timer="wall")
    assert time.perf_counter() - t0 < 0.15
