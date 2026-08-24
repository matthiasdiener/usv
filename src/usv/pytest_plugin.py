"""pytest plugin: run usv GPU microbenchmarks as tests.

Exposes the ``usv_benchmark`` fixture -- a thin wrapper over :func:`usv.do_bench`
-- so a benchmark is just an ordinary pytest test.  Selection/filtering is plain
pytest (``-k`` / ``-m`` / ``parametrize``); recorded results are printed as an
end-of-session table and, with ``--benchmark-save``, written to a CSV in usv's
:func:`usv.save_results` format::

    def test_gemm(usv_benchmark):
        x = torch.randn(4096, 4096, device="cuda")
        usv_benchmark(lambda: x @ x, flops=2 * 4096**3)

    # bf16 only, no fp8:   pytest -k bf16
    # only benchmarks:     pytest --benchmark-only
    # skip benchmarks:     pytest -m "not benchmark"

The fixture also offers ``.fwd_bwd(fwd, fwd_bwd, fwd_flops=...)`` (records a
forward row plus a derived backward row) and ``.many({name: fn})`` (interleaved
group timing via :func:`usv.do_bench_many`).

Regression gating compares against a saved baseline and fails the session when a
metric regresses past a threshold::

    pytest --benchmark-only --benchmark-save=main            # record a baseline
    pytest --benchmark-only --benchmark-compare=main \\
           --benchmark-compare-fail=median:5%                # gate on +5% median

``--benchmark-compare`` accepts a label, an explicit CSV path, or ``last``;
``--benchmark-compare-fail`` is repeatable and takes ``METRIC:PCT`` where METRIC
is one of ``median|mean|min|max`` (lower is better) or ``tflops|gbps`` (higher is
better).  Without a fail spec, deltas are shown but the session is not gated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from usv import (
    do_bench,
    do_bench_many,
    find_baseline,
    format_table,
    load_results,
    save_results,
)
from usv.bench import Measurement

__all__ = ["usv_benchmark"]

# Metric name -> CSV column; time metrics are lower-is-better, rates higher.
_TIME_METRICS = {"median": "median_ms", "mean": "mean_ms", "min": "min_ms", "max": "max_ms"}
_RATE_METRICS = {"tflops": "tflops", "gbps": "gbps"}



class _Store:
    """Session-wide collector of recorded Measurements."""

    def __init__(self) -> None:
        self.measurements: list[Measurement] = []


class _Bench:
    """Returned by the ``usv_benchmark`` fixture; call it to time a callable."""

    def __init__(self, request, store: _Store, default_timer: str) -> None:
        self._request = request
        self._store = store
        self._default_timer = default_timer

    @property
    def _base_name(self) -> str:
        # node name carries the parametrize id, e.g. "test_gemm[bf16-N4096]"
        return self._request.node.name

    def _record(self, m: Measurement) -> Measurement:
        self._store.measurements.append(m)
        return m

    def __call__(self, fn, *, name=None, timer=None, **kwargs) -> Measurement:
        """Time *fn* via :func:`usv.do_bench` and record the result."""
        return self._record(
            do_bench(
                fn,
                name=name or self._base_name,
                timer=self._default_timer if timer is None else timer,
                **kwargs,
            )
        )

    def many(self, fns, *, interleave=True, timer=None, **kwargs) -> dict:
        """Time a group of callables together (interleaved by default)."""
        res = do_bench_many(
            fns,
            interleave=interleave,
            timer=self._default_timer if timer is None else timer,
            **kwargs,
        )
        for m in res.values():
            self._store.measurements.append(m)
        return res

    def fwd_bwd(self, fwd, fwd_bwd, *, fwd_flops=None, name=None, timer=None, **kwargs):
        """Record a forward row and a derived backward row.

        Backward is derived as ``(fwd+bwd) - fwd`` (per sample); its flops
        default to twice the forward's (dX + dW).  Returns ``(m_fwd, m_bwd)``.
        """
        base = name or self._base_name
        t = self._default_timer if timer is None else timer
        m_fwd = do_bench(fwd, name=f"{base} [fwd]", flops=fwd_flops, timer=t, **kwargs)
        m_fb = do_bench(fwd_bwd, name=f"{base} [fwd+bwd]", timer=t, **kwargs)
        m_bwd = Measurement(
            samples=m_fb.samples - m_fwd.median,
            name=f"{base} [bwd]",
            inner=m_fb.inner,
            flops=None if fwd_flops is None else 2 * fwd_flops,
        )
        self._record(m_fwd)
        return m_fwd, self._record(m_bwd)


@dataclass
class _Cmp:
    """One benchmark/metric compared against the baseline."""

    name: str
    metric: str
    baseline: float | None
    current: float | None
    delta_pct: float | None
    regressed: bool


def _parse_criteria(specs) -> list[tuple[str, float | None]]:
    """Parse ``METRIC:PCT`` fail specs into ``(metric, threshold_pct)`` pairs."""
    out: list[tuple[str, float | None]] = []
    for spec in specs or []:
        metric, _, thr = spec.partition(":")
        metric = metric.strip().lower()
        if metric not in _TIME_METRICS and metric not in _RATE_METRICS:
            raise pytest.UsageError(
                f"--benchmark-compare-fail: unknown metric {metric!r}"
            )
        thr = thr.strip().rstrip("%")
        try:
            pct = float(thr) if thr else 0.0
        except ValueError:
            raise pytest.UsageError(
                f"--benchmark-compare-fail: bad threshold in {spec!r}"
            )
        out.append((metric, pct))
    return out


def _current_value(m: Measurement, metric: str) -> float | None:
    if metric in _TIME_METRICS:
        return getattr(m, metric) * 1e3  # seconds -> ms, matching the CSV
    return m.tflops if metric == "tflops" else m.gbps


def _baseline_value(row: dict, metric: str) -> float | None:
    col = _TIME_METRICS.get(metric) or _RATE_METRICS[metric]
    raw = row.get(col, "")
    return float(raw) if raw not in ("", None) else None


def _compare(measurements, baseline, criteria) -> list[_Cmp]:
    """Build one :class:`_Cmp` per measurement/criterion.

    A ``None`` threshold means display-only (never regressed).  Time metrics
    regress when they grow past the threshold; rate metrics when they drop.
    """
    out: list[_Cmp] = []
    for m in measurements:
        row = baseline.get(m.name)
        for metric, thr in criteria:
            cur = _current_value(m, metric)
            base = _baseline_value(row, metric) if row else None
            delta = regressed = None
            if base not in (None, 0) and cur is not None:
                delta = (cur - base) / base * 100.0
                if thr is not None:
                    regressed = delta > thr if metric in _TIME_METRICS else delta < -thr
            out.append(_Cmp(m.name, metric, base, cur, delta, bool(regressed)))
    return out


def _ensure_comparison(config) -> list[_Cmp]:
    """Compute (and cache on *config*) the baseline comparison for this session."""
    if hasattr(config, "_usv_comparison"):
        return config._usv_comparison
    store = getattr(config, "_usv_store", None)
    selector = config.getoption("--benchmark-compare")
    cmps: list[_Cmp] = []
    path = None
    if store and store.measurements and selector:
        path = find_baseline(selector)
        if path:
            baseline = {r["name"]: r for r in load_results(path)}
            criteria = _parse_criteria(config.getoption("--benchmark-compare-fail"))
            cmps = _compare(store.measurements, baseline, criteria or [("median", None)])
    config._usv_baseline_path = path
    config._usv_comparison = cmps
    return cmps


def _fmt_val(metric: str, v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}ms" if metric in _TIME_METRICS else f"{v:.1f}"


def _format_comparison(cmps: list[_Cmp]) -> str:
    header = ("benchmark", "metric", "baseline", "current", "delta", "")
    rows = [
        (
            c.name,
            c.metric,
            _fmt_val(c.metric, c.baseline),
            _fmt_val(c.metric, c.current),
            "-" if c.delta_pct is None else f"{c.delta_pct:+.1f}%",
            "FAIL" if c.regressed else ("miss" if c.baseline is None else "ok"),
        )
        for c in cmps
    ]
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]
    fmt = lambda r: "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))
    sep = "-" * (sum(widths) + 2 * (len(widths) - 1))
    return "\n".join([fmt(header), sep, *(fmt(r) for r in rows)])


def pytest_addoption(parser) -> None:
    group = parser.getgroup("usv", "usv GPU benchmarking")
    group.addoption(
        "--benchmark-only",
        action="store_true",
        default=False,
        help="run only tests that use the usv_benchmark fixture",
    )
    group.addoption(
        "--benchmark-save",
        nargs="?",
        const=True,
        default=None,
        metavar="LABEL",
        help="save collected benchmark results to a CSV (optionally labeled)",
    )
    group.addoption(
        "--benchmark-timer",
        default="auto",
        metavar="TIMER",
        help="usv timer for benchmarks: auto|torch|jax|wall (default: auto)",
    )
    group.addoption(
        "--benchmark-compare",
        default=None,
        metavar="SELECTOR",
        help="compare against a saved baseline: a label, a CSV path, or 'last'",
    )
    group.addoption(
        "--benchmark-compare-fail",
        action="append",
        default=[],
        metavar="METRIC:PCT",
        help="fail if METRIC regresses past PCT%% (e.g. median:5%%); repeatable",
    )



def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: usv GPU microbenchmark (deselect with -m 'not benchmark')",
    )
    config._usv_store = _Store()


@pytest.fixture
def usv_benchmark(request):
    """Time GPU work in a test; see the plugin module docstring for usage."""
    return _Bench(
        request,
        request.config._usv_store,
        request.config.getoption("--benchmark-timer"),
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Auto-mark fixture users as ``benchmark`` and honor ``--benchmark-only``."""
    only = config.getoption("--benchmark-only")
    selected, deselected = [], []
    for item in items:
        is_bench = "usv_benchmark" in getattr(item, "fixturenames", ())
        if is_bench:
            item.add_marker("benchmark")
        (deselected if (only and not is_bench) else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    store = getattr(config, "_usv_store", None)
    if not store or not store.measurements:
        return
    tr = terminalreporter
    tr.write_sep("-", "usv benchmarks")
    tr.write_line(format_table(store.measurements))
    save = config.getoption("--benchmark-save")
    if save:
        path = save_results(
            store.measurements, label=save if isinstance(save, str) else None
        )
        tr.write_line(f"saved {len(store.measurements)} result(s) -> {path}")
    selector = config.getoption("--benchmark-compare")
    if not selector:
        return
    cmps = _ensure_comparison(config)
    base_path = getattr(config, "_usv_baseline_path", None)
    if not base_path:
        tr.write_line(f"no baseline found for {selector!r}")
        return
    tr.write_line("")
    tr.write_line(f"comparison vs {os.path.basename(base_path)}")
    tr.write_line(_format_comparison(cmps))
    regressed = sorted({c.name for c in cmps if c.regressed})
    if regressed:
        tr.write_line(f"benchmark regression: {', '.join(regressed)}")


def pytest_sessionfinish(session, exitstatus) -> None:
    """Fail the session (non-zero exit) if any benchmark regressed."""
    config = session.config
    if not getattr(config, "_usv_store", None) or not config.getoption(
        "--benchmark-compare"
    ):
        return
    cmps = _ensure_comparison(config)
    if any(c.regressed for c in cmps) and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

