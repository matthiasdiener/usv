"""Opt-in GPU telemetry sampling during a benchmark (AMD / ``rocm-smi``).

CUTLASS' measurement guidelines stress that *locking* clocks is not enough:
power and thermal controllers can still move the frequency, so the settled clock
(and power / temperature) should be *monitored* during the profiling iterations
and results flagged if the clock drifted.

:class:`GpuMonitor` samples ``rocm-smi`` in a background thread while a block
runs and summarizes the readings.  Everything is best-effort and read-only: if
``rocm-smi`` is missing or its JSON can't be parsed, sampling yields nothing
rather than raising.  Only AMD / ``rocm-smi`` is supported for now.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading

import numpy as np

__all__ = ["GpuMonitor", "sample_rocm_smi", "clock_drift_fraction"]

_METRICS = ("sclk_mhz", "mclk_mhz", "power_w", "temp_c")


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True, timeout=10
        )
    except Exception:
        return None


def _num(value) -> float | None:
    """First number found in *value*'s text form (handles '(1500Mhz)', '35.0 W')."""
    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(m.group()) if m else None


def _flatten(card: dict, prefix: str = "") -> list[tuple[str, object]]:
    pairs: list[tuple[str, object]] = []
    for key, val in card.items():
        name = f"{prefix}{key}"
        if isinstance(val, dict):
            pairs.extend(_flatten(val, name + "."))
        else:
            pairs.append((name, val))
    return pairs


def _parse_card(card: dict) -> dict:
    """Extract sclk / mclk / power / temperature from one rocm-smi card entry."""
    out: dict[str, float] = {}
    temp_edge: float | None = None
    temp_any: float | None = None
    for key, val in _flatten(card):
        kl = key.lower()
        num = _num(val)
        if num is None:
            continue
        if "sclk" in kl and "sclk_mhz" not in out:
            out["sclk_mhz"] = num
        elif "mclk" in kl and "mclk_mhz" not in out:
            out["mclk_mhz"] = num
        elif "power" in kl and "power_w" not in out:
            out["power_w"] = num
        elif "temp" in kl:
            if "edge" in kl and temp_edge is None:
                temp_edge = num
            elif temp_any is None:
                temp_any = num
    temp = temp_edge if temp_edge is not None else temp_any
    if temp is not None:
        out["temp_c"] = temp
    return out


def sample_rocm_smi(device: int = 0) -> dict | None:
    """One telemetry reading for *device* via ``rocm-smi``, or ``None``.

    Best-effort: returns a dict with any of ``sclk_mhz``, ``mclk_mhz``,
    ``power_w``, ``temp_c`` that could be parsed, else ``None``.
    """
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    out = _run([exe, "-d", str(device), "--showclocks", "--showpower", "--showtemp", "--json"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    card = data.get(f"card{device}")
    if not isinstance(card, dict):
        card = next((v for v in data.values() if isinstance(v, dict)), None)
    if not isinstance(card, dict):
        return None
    return _parse_card(card) or None


def clock_drift_fraction(summary: dict | None) -> float | None:
    """Relative sclk spread ``(max - min) / mean`` from a monitor summary, or ``None``."""
    sclk = (summary or {}).get("sclk_mhz")
    if not sclk or not sclk.get("mean"):
        return None
    return (sclk["max"] - sclk["min"]) / sclk["mean"]


class GpuMonitor:
    """Sample GPU telemetry in a background thread for the duration of a block.

    ::

        with GpuMonitor(device=0) as mon:
            do_bench(fn)
        print(mon.summary())     # {'n': .., 'sclk_mhz': {min/mean/max/std}, ...}

    Pass a custom *sampler* (a zero-arg callable returning a reading dict) to
    monitor a different source or for testing.
    """

    def __init__(self, *, device: int = 0, interval_s: float = 0.05, sampler=None) -> None:
        self.device = device
        self.interval_s = interval_s
        self._sampler = sampler if sampler is not None else (
            lambda: sample_rocm_smi(device=device)
        )
        self.samples: list[dict] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _take(self) -> None:
        try:
            reading = self._sampler()
        except Exception:
            reading = None
        if reading:
            self.samples.append(reading)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.interval_s)
            if self._stop.is_set():
                break
            self._take()

    def start(self) -> "GpuMonitor":
        self.samples = []
        self._stop.clear()
        self._take()  # one synchronous reading, so short runs still have a sample
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 5 + 1.0)
            self._thread = None

    def __enter__(self) -> "GpuMonitor":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def summary(self) -> dict:
        """Per-metric ``{min, mean, max, std}`` over the collected samples."""
        out: dict[str, object] = {"n": len(self.samples)}
        for key in _METRICS:
            vals = [s[key] for s in self.samples if s.get(key) is not None]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[key] = {
                    "min": float(arr.min()),
                    "mean": float(arr.mean()),
                    "max": float(arr.max()),
                    "std": float(arr.std(ddof=0)),
                }
        return out
