"""Tests for opt-in GPU interference detection (no GPU needed)."""

from __future__ import annotations

import os
import warnings

import pytest

import usv.interference as interference
from usv import check_gpu_interference, do_bench


def test_check_excludes_self(monkeypatch):
    me = os.getpid()
    fake = [
        {"pid": me, "name": "self", "memory_mb": 10},
        {"pid": me + 1, "name": "other", "memory_mb": 20},
    ]
    monkeypatch.setattr(interference, "gpu_processes", lambda **kw: fake)
    others = check_gpu_interference()
    assert [p["pid"] for p in others] == [me + 1]


def test_check_excludes_extra_pids(monkeypatch):
    fake = [{"pid": 111, "name": "a", "memory_mb": None}, {"pid": 222, "name": "b", "memory_mb": None}]
    monkeypatch.setattr(interference, "gpu_processes", lambda **kw: fake)
    others = check_gpu_interference(exclude_pids=[111])
    assert [p["pid"] for p in others] == [222]


def test_unknown_vendor_returns_empty(monkeypatch):
    monkeypatch.setattr(interference, "gpu_vendor", lambda: None)
    assert interference.gpu_processes(vendor="auto") == []


def test_do_bench_warns_on_interference(monkeypatch):
    monkeypatch.setattr(
        interference,
        "check_gpu_interference",
        lambda **kw: [{"pid": 999, "name": "hog", "memory_mb": 1}],
    )
    with pytest.warns(UserWarning, match="other process"):
        do_bench(lambda: None, warmup=0, iters=1, timer="wall", check_interference=True)


def test_do_bench_no_warn_by_default(monkeypatch):
    monkeypatch.setattr(
        interference,
        "check_gpu_interference",
        lambda **kw: [{"pid": 999, "name": "hog", "memory_mb": 1}],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        do_bench(lambda: None, warmup=0, iters=1, timer="wall")
