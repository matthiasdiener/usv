"""Write benchmark results (Measurements) to CSV."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usv.bench import Measurement

__all__ = ["save_results", "save_samples", "load_results", "find_baseline"]

_FIELDS = [
    "name",
    "n",
    "inner",
    "tflops",
    "gbps",
    "median_ms",
    "mean_ms",
    "stdev_ms",
    "min_ms",
    "max_ms",
]

_SAMPLE_FIELDS = ["name", "sample_idx", "tflops", "gbps", "time_ms"]


def _get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _as_list(measurements) -> list:
    """Normalize the accepted input shapes into a list of Measurements."""
    if isinstance(measurements, dict):
        return list(measurements.values())
    if hasattr(measurements, "samples"):  # a single Measurement
        return [measurements]
    return list(measurements)


def _resolve_path(
    path: str | None,
    results_dir: str | None,
    label: str | None,
    suffix: str,
) -> str:
    """Return an explicit *path* or ``<results_dir>/<commit>[-<label>]<suffix>.csv``."""
    if path is not None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path
    results_dir = results_dir or os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = ""
    if label:
        tag = "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return os.path.join(results_dir, f"{_get_commit_hash()[:8]}{tag}{suffix}.csv")


def _row(m: "Measurement") -> dict[str, object]:
    return {
        "name": m.name,
        "n": m.n,
        "inner": m.inner,
        "tflops": f"{m.tflops:.4f}" if m.tflops is not None else "",
        "gbps": f"{m.gbps:.4f}" if m.gbps is not None else "",
        "median_ms": f"{m.median * 1e3:.6f}",
        "mean_ms": f"{m.mean * 1e3:.6f}",
        "stdev_ms": f"{m.std * 1e3:.6f}",
        "min_ms": f"{m.min * 1e3:.6f}",
        "max_ms": f"{m.max * 1e3:.6f}",
    }


def _sample_throughput(work: float | None, sample_s: float, scale: float) -> str:
    if not work or sample_s <= 0:
        return ""
    return f"{work / sample_s / scale:.4f}"


def save_results(
    measurements,
    path: str | None = None,
    *,
    results_dir: str | None = None,
    label: str | None = None,
) -> str:
    """Write *measurements* to a CSV file, one row per benchmark.

    *measurements* may be a single :class:`~usv.Measurement`, an iterable of
    them, or the ``dict[str, Measurement]`` returned by
    :func:`~usv.do_bench_many`.  Columns are ``name, n, inner, tflops, gbps,
    median_ms, mean_ms, stdev_ms, min_ms, max_ms`` (throughput cells are
    blank when ``flops`` / ``bytes`` were not supplied).

    With *path* the CSV is written there; otherwise it goes to
    ``<results_dir or ./results>/<commit>[-<label>].csv``.  Returns the path.
    """
    items = _as_list(measurements)
    path = _resolve_path(path, results_dir, label, "")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for m in items:
            writer.writerow(_row(m))
    return path


def save_samples(
    measurements,
    path: str | None = None,
    *,
    results_dir: str | None = None,
    label: str | None = None,
) -> str:
    """Write raw per-call timing samples to a CSV, one row per sample.

    Mirrors the ``--csv-samples`` output: for every benchmark, each of its
    per-call timings (``Measurement.samples``) is written as its own row with
    columns ``name, sample_idx, tflops, gbps, time_ms``.  Throughput cells are
    blank when ``flops`` / ``bytes`` were not supplied.  This preserves the full
    timing distribution for downstream analysis, whereas :func:`save_results`
    keeps only summary statistics.

    *measurements* accepts the same shapes as :func:`save_results` (a single
    :class:`~usv.Measurement`, an iterable of them, or the
    ``dict[str, Measurement]`` from :func:`~usv.do_bench_many`).

    With *path* the CSV is written there; otherwise it goes to
    ``<results_dir or ./results>/<commit>[-<label>]-samples.csv``.  Returns the
    path.
    """
    items = _as_list(measurements)
    path = _resolve_path(path, results_dir, label, "-samples")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SAMPLE_FIELDS)
        writer.writeheader()
        for m in items:
            for i, sample_s in enumerate(m.samples):
                sample_s = float(sample_s)
                writer.writerow(
                    {
                        "name": m.name,
                        "sample_idx": i,
                        "tflops": _sample_throughput(m.flops, sample_s, 1e12),
                        "gbps": _sample_throughput(m.bytes, sample_s, 1e9),
                        "time_ms": f"{sample_s * 1e3:.6f}",
                    }
                )
    return path


def load_results(path: str) -> list[dict[str, str]]:
    """Load a CSV written by :func:`save_results` into a list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_baseline(selector: str, *, results_dir: str | None = None) -> str | None:
    """Resolve a baseline *selector* to a results CSV path (or ``None``).

    *selector* may be an explicit path to a CSV, the special value ``"last"``
    (the most recent non-sample CSV in *results_dir*), or a label written by
    :func:`save_results` (the most recent ``*-<label>.csv``).  *results_dir*
    defaults to ``./results``.
    """
    if selector and os.path.isfile(selector):
        return selector
    results_dir = results_dir or os.path.join(os.getcwd(), "results")
    if not os.path.isdir(results_dir):
        return None
    csvs = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and not f.endswith("-samples.csv")
    ]
    if selector and selector != "last":
        tag = "-" + re.sub(r"[^A-Za-z0-9._-]+", "_", selector).strip("_") + ".csv"
        csvs = [p for p in csvs if p.endswith(tag)]
    if not csvs:
        return None
    return max(csvs, key=os.path.getmtime)

