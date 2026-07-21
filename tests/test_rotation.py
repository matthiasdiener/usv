"""Tests for L2-sized buffer rotation helpers (no GPU needed)."""

from __future__ import annotations

import itertools

import numpy as np

from usv import rotating_buffers, rotation_count
from usv.bench import _nbytes


def test_rotation_count_exceeds_l2():
    # 32 B buffers, want >= 2x a 100 B "L2" -> ceil(200/32) = 7.
    assert rotation_count(32, l2_mult=2.0, l2_bytes=100) == 7


def test_rotation_count_respects_min_buffers():
    # A single buffer already dwarfs L2, but we still want at least 2.
    assert rotation_count(10_000, l2_mult=2.0, min_buffers=2, l2_bytes=100) == 2


def test_rotation_count_unknown_l2_falls_back_to_min(monkeypatch):
    # When the L2 size can't be queried, fall back to min_buffers.
    monkeypatch.setattr("usv.bench._l2_cache_size_bytes", lambda: None)
    assert rotation_count(32, l2_bytes=None, min_buffers=3) == 3


def test_nbytes_numpy():
    assert _nbytes(np.zeros(4, dtype=np.float64)) == 32


def test_rotating_buffers_sizes_and_cycles():
    counter = itertools.count()
    make = lambda: np.full(4, next(counter), dtype=np.float64)  # 32 B each

    nxt = rotating_buffers(make, l2_bytes=100, l2_mult=2.0)  # -> 7 buffers
    values = [nxt()[0] for _ in range(7)]
    assert values == list(range(7))  # seven distinct buffers
    assert nxt()[0] == 0  # then it wraps around


def test_rotating_buffers_explicit_count():
    make = lambda: np.zeros(1, dtype=np.float32)
    nxt = rotating_buffers(make, count=3)
    ids = {id(nxt()) for _ in range(3)}
    assert len(ids) == 3
