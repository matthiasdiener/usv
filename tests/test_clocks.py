"""Tests for the opt-in clock-locking helpers (no GPU required)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import usv.bench as bench
import usv.clocks as clocks
from usv import do_bench, fixed_clocks, gpu_vendor


def test_gpu_vendor_returns_known_value():
    assert gpu_vendor() in (None, "nvidia", "amd")


def test_fixed_clocks_requires_detected_vendor(monkeypatch):
    # With no detectable GPU/tooling, locking can't proceed and must error.
    monkeypatch.setattr(clocks, "gpu_vendor", lambda: None)
    with pytest.raises(RuntimeError):
        with fixed_clocks():
            pass


def test_amd_fixed_clocks_uses_perf_level_high(monkeypatch):
    monkeypatch.setattr(clocks, "gpu_vendor", lambda: "amd")
    monkeypatch.setattr(
        clocks.shutil, "which", lambda n: "/usr/bin/amd-smi" if n == "amd-smi" else None
    )
    cmds: list[list[str]] = []
    monkeypatch.setattr(clocks, "_run", lambda cmd: cmds.append(cmd) or "")
    monkeypatch.setattr(clocks, "_run_ok", lambda cmd: cmds.append(cmd))
    with fixed_clocks():  # AMD -> force max via perf-level HIGH
        pass
    assert cmds[0] == ["/usr/bin/amd-smi", "set", "--gpu", "0", "--perf-level", "HIGH"]
    assert cmds[-1] == ["/usr/bin/amd-smi", "set", "--gpu", "0", "--perf-level", "AUTO"]


def _record_fixed_clocks(seen):
    @contextmanager
    def fake(sm=None, mem=None, **kw):
        seen.append((sm, mem))
        yield (sm, mem)

    return fake


def test_lock_clocks_true_requests_device_max(monkeypatch):
    seen: list = []
    monkeypatch.setattr(bench, "fixed_clocks", _record_fixed_clocks(seen))
    do_bench(lambda: None, warmup=0, iters=1, timer="wall", lock_clocks=True)
    assert seen == [(None, None)]  # None -> device max


def test_lock_clocks_none_does_not_lock(monkeypatch):
    seen: list = []
    monkeypatch.setattr(bench, "fixed_clocks", _record_fixed_clocks(seen))
    do_bench(lambda: None, warmup=0, iters=1, timer="wall")
    assert seen == []  # fixed_clocks never entered
