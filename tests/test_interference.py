"""Tests for opt-in GPU interference detection (no GPU needed)."""

from __future__ import annotations

import json
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
    monkeypatch.setattr(interference, "_torch_bdf", lambda device: None)
    monkeypatch.setattr(interference, "gpu_processes", lambda **kw: fake)
    others = check_gpu_interference()
    assert [p["pid"] for p in others] == [me + 1]


def test_check_excludes_extra_pids(monkeypatch):
    fake = [{"pid": 111, "name": "a", "memory_mb": None}, {"pid": 222, "name": "b", "memory_mb": None}]
    monkeypatch.setattr(interference, "_torch_bdf", lambda device: None)
    monkeypatch.setattr(interference, "gpu_processes", lambda **kw: fake)
    others = check_gpu_interference(exclude_pids=[111])
    assert [p["pid"] for p in others] == [222]


def test_check_matches_torch_device_by_pci(monkeypatch):
    snap = {
        "0000:75:00.0": [{"pid": 7, "name": "python", "memory_mb": None}],
        "0000:05:00.0": [{"pid": 8, "name": "python", "memory_mb": None}],
    }
    monkeypatch.setattr(interference, "_torch_bdf", lambda device: "0000:75:00.0")
    monkeypatch.setattr(interference, "snapshot_gpu_processes", lambda **kw: snap)
    assert [p["pid"] for p in check_gpu_interference()] == [7]


def test_foreign_processes_filters_tools_and_pids():
    snap = {
        "0000:75:00.0": [
            {"pid": 1, "name": "rocm-smi", "memory_mb": None},
            {"pid": 2, "name": "nvtop", "memory_mb": None},
            {"pid": 3, "name": "python3", "memory_mb": None},
            {"pid": 4, "name": "python3", "memory_mb": None},
        ]
    }
    got = interference.foreign_processes(snap, "75:00.0", exclude_pids={4})
    assert [p["pid"] for p in got] == [3]
    assert interference.foreign_processes(snap, None) == []
    assert interference.foreign_processes(None, "0000:75:00.0") == []


def test_amd_snapshot_parses_list_and_process_json(monkeypatch):
    gpus = [{"gpu": 0, "bdf": "0000:05:00.0"}, {"gpu": 1, "bdf": "0000:75:00.0"}]
    procs = [
        {"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]},
        {"gpu": 1, "process_list": [{"process_info": {"name": "python3", "pid": 99}}]},
    ]
    outputs = {"list": json.dumps(gpus), "process": json.dumps(procs)}
    monkeypatch.setattr(interference.shutil, "which", lambda exe: f"/usr/bin/{exe}")
    monkeypatch.setattr(interference, "_run", lambda cmd: outputs[cmd[1]])
    snap = interference.snapshot_gpu_processes(vendor="amd")
    assert snap == {
        "0000:05:00.0": [],
        "0000:75:00.0": [{"pid": 99, "name": "python3", "memory_mb": None}],
    }


def test_idle_amd_gpu_parses_to_empty():
    out = json.dumps([{"gpu": 0, "process_list": [{"process_info": "No running processes detected"}]}])
    assert interference._parse_amd_json(out) == []


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
