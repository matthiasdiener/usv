"""Tests for fresh-process (subprocess-isolated) benchmark execution."""

from __future__ import annotations

import os

import pytest

from usv import run_benchmarks

_MODULE_SRC = '''
import os

class Bench:
    params = [1, 2]
    param_names = ["n"]

    def time_x(self, n):
        d = os.environ["USV_TEST_PIDDIR"]
        with open(os.path.join(d, f"{n}.pid"), "w") as f:
            f.write(str(os.getpid()))
        return n * n
'''


def test_fresh_process_runs_each_in_its_own_process(tmp_path, monkeypatch):
    piddir = tmp_path / "pids"
    piddir.mkdir()
    monkeypatch.setenv("USV_TEST_PIDDIR", str(piddir))
    (tmp_path / "usv_benchmod.py").write_text(_MODULE_SRC)
    monkeypatch.syspath_prepend(str(tmp_path))

    res = run_benchmarks(
        "usv_benchmod:Bench", fresh_process=True, warmup=0, iters=2, timer="wall"
    )

    assert set(res) == {"Bench.time_x(n=1)", "Bench.time_x(n=2)"}
    for m in res.values():
        assert m.n == 2

    pids = {p.read_text() for p in piddir.glob("*.pid")}
    assert len(pids) == 2  # each benchmark ran in a separate process
    assert str(os.getpid()) not in pids  # and none of them is the parent


def test_fresh_process_rejects_instance():
    class Local:
        def time_x(self):
            return 1

    with pytest.raises(ValueError):
        run_benchmarks(Local(), fresh_process=True, timer="wall")


def test_in_process_still_default(tmp_path, monkeypatch):
    # Sanity: the same class runs in-process without fresh_process.
    monkeypatch.setenv("USV_TEST_PIDDIR", str(tmp_path))
    (tmp_path / "usv_benchmod2.py").write_text(_MODULE_SRC.replace("Bench", "Bench2"))
    monkeypatch.syspath_prepend(str(tmp_path))
    res = run_benchmarks("usv_benchmod2:Bench2", warmup=0, iters=1, timer="wall")
    assert set(res) == {"Bench2.time_x(n=1)", "Bench2.time_x(n=2)"}
