"""Tests for opt-in GPU telemetry monitoring (rocm-smi), no GPU needed."""

from __future__ import annotations

import pytest

import usv.bench as bench
import usv.monitor as monitor
from usv import GpuMonitor, do_bench
from usv.monitor import _parse_card, clock_drift_fraction


def test_parse_card_extracts_metrics():
    card = {
        "sclk clock speed": "(1500Mhz)",
        "mclk clock speed": "(1200Mhz)",
        "Average Graphics Package Power (W)": "42.0",
        "Temperature (Sensor edge) (C)": "55.0",
    }
    parsed = _parse_card(card)
    assert parsed["sclk_mhz"] == 1500.0
    assert parsed["mclk_mhz"] == 1200.0
    assert parsed["power_w"] == 42.0
    assert parsed["temp_c"] == 55.0


def test_clock_drift_fraction():
    assert clock_drift_fraction({"sclk_mhz": {"min": 1000, "max": 1500, "mean": 1250}}) == 0.4
    assert clock_drift_fraction({}) is None
    assert clock_drift_fraction(None) is None


def test_monitor_collects_and_summarizes():
    readings = [{"sclk_mhz": 1000.0, "power_w": 30.0}, {"sclk_mhz": 1400.0, "power_w": 50.0}]
    it = iter(readings)
    mon = GpuMonitor(interval_s=0.01, sampler=lambda: next(it, readings[-1]))
    with mon:
        pass
    s = mon.summary()
    assert s["n"] >= 1
    assert "sclk_mhz" in s and s["sclk_mhz"]["max"] >= s["sclk_mhz"]["min"]


def test_do_bench_attaches_monitor_summary(monkeypatch):
    monkeypatch.setattr(monitor, "sample_rocm_smi", lambda device=0: {"sclk_mhz": 1500.0})
    m = do_bench(lambda: None, warmup=0, iters=1, timer="wall", monitor=True)
    assert isinstance(m.monitor, dict) and m.monitor["n"] >= 1
    assert m.monitor["sclk_mhz"]["mean"] == 1500.0


def test_no_monitor_by_default():
    m = do_bench(lambda: None, warmup=0, iters=1, timer="wall")
    assert m.monitor is None


def test_warn_on_clock_drift():
    drifted = {"sclk_mhz": {"min": 900.0, "max": 1500.0, "mean": 1200.0, "std": 200.0}}
    with pytest.warns(UserWarning, match="sclk varied"):
        bench._warn_on_clock_drift(drifted)
