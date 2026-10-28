"""pytest plugin: the usv benchmark runner.

Exposes the ``usv_benchmark`` fixture -- a thin wrapper over :func:`usv.do_bench`
-- so a benchmark is just an ordinary pytest test.  Selection/filtering is plain
pytest (``-k`` / ``-m`` / ``parametrize``); recorded results are printed as an
end-of-session table and, with ``--benchmark-save``, written to a CSV in usv's
:func:`usv.save_results` format, next to a ``-info.txt`` describing the machine::

    def test_gemm(usv_benchmark):
        x = torch.randn(4096, 4096, device="cuda")
        usv_benchmark(lambda: x @ x, flops=2 * 4096**3)

    # bf16 only, no fp8:   pytest -k bf16
    # only benchmarks:     pytest --benchmark-only
    # skip benchmarks:     pytest -m "not benchmark"

The fixture also offers ``.backward(fwd, bwd)`` (backward timed in blocks, see
:func:`usv.do_bench_backward`), ``.fwd_bwd(fwd, fwd_bwd, fwd_flops=...)`` (a forward
row plus a backward row derived as ``(fwd+bwd) - fwd``) and ``.many({name: fn})``
(interleaved group timing via :func:`usv.do_bench_many`).  With ``--benchmark-asv``,
asv-style benchmark classes (see :mod:`usv.asv`) are collected as benchmarks too.

Before the first benchmark runs, the session is prepared (each step can be disabled):

* ``--benchmark-gpu-interference=warn|abort|off`` (default ``warn``): report other
  processes on the benchmark GPU, from a snapshot taken before this process
  touches the GPU (see :mod:`usv.interference`).
* CPU pinning (``--benchmark-no-cpu-pin`` disables it): pin to a few GPU-local
  cores at the performance governor, restored at the end (see :mod:`usv.cpupin`).

``--benchmark-interleave R`` measures every benchmark in R rounds spread over the
session (round 1 of all benchmarks, then round 2, ...; shuffled within a round), each
with 1/R of the samples, and reports the statistics of all rounds' samples, so a
transient slowdown hits a few samples of many benchmarks instead of one benchmark's
whole measurement.  ``--benchmark-fresh-process`` runs each asv benchmark in its
own Python process.

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

import inspect
import math
import os
import random
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from fnmatch import fnmatch

import numpy as np
import pytest

from usv import (
    do_bench,
    do_bench_backward,
    do_bench_many,
    find_baseline,
    format_table,
    load_results,
    save_results,
)
from usv.asv import (
    SkipBenchmark,
    _axis_indices,
    _bench_callable,
    _call,
    _enumerate_class,
    _import_path,
    _label,
    _param_ids,
    _param_spec,
    _run_in_subprocess,
    _setup,
    asv_bench_kwargs,
    is_asv_class,
)
from usv.bench import Measurement

__all__ = ["usv_benchmark"]

# Metric name -> CSV column; time metrics are lower-is-better, rates higher.
_TIME_METRICS = {"median": "median_ms", "mean": "mean_ms", "min": "min_ms", "max": "max_ms"}
_RATE_METRICS = {"tflops": "tflops", "gbps": "gbps"}

_DEFAULT_ITERS = inspect.signature(do_bench).parameters["iters"].default
_BWD_DEFAULTS = inspect.signature(do_bench_backward).parameters


def _scale_for_rounds(kwargs: dict, rounds: int) -> dict:
    """Give one interleave round 1/*rounds* of the samples."""
    kw = dict(kwargs)
    kw["iters"] = max(1, math.ceil(kw.get("iters", _DEFAULT_ITERS) / rounds))
    if kw.get("min_iters_time") is not None:
        kw["min_iters_time"] = kw["min_iters_time"] / rounds
    return kw


def _scale_backward_for_rounds(kwargs: dict, rounds: int) -> dict:
    kw = dict(kwargs)
    blocks = kw.get("min_blocks", _BWD_DEFAULTS["min_blocks"].default)
    kw["min_blocks"] = max(1, math.ceil(blocks / rounds))
    kw["min_time"] = kw.get("min_time", _BWD_DEFAULTS["min_time"].default) / rounds
    return kw


def _strip_round(name: str, rnd: int | None) -> str:
    """Drop the interleave-round id from a node name: ``t[a-r2]`` -> ``t[a]``, ``t[r2]`` -> ``t``."""
    if rnd is None:
        return name
    if name.endswith(f"[r{rnd}]"):
        return name[: -len(f"[r{rnd}]")]
    if name.endswith(f"-r{rnd}]"):
        return name[: -len(f"-r{rnd}]")] + "]"
    return name


class _Store:
    """Session-wide collector of recorded Measurements.

    The interleave rounds of one benchmark (same *key*: the test id without its round,
    plus the measurement name) are pooled into one; without a key nothing is pooled.
    """

    def __init__(self) -> None:
        self.measurements: list[Measurement] = []
        self._pooled: dict[tuple, Measurement] = {}

    def add(self, m: Measurement, key: tuple | None = None) -> None:
        if key is None:
            self.measurements.append(m)
            return
        pooled = self._pooled.get(key)
        if pooled is None:
            pooled = Measurement(
                samples=m.samples.copy(),
                name=m.name,
                inner=m.inner,
                flops=m.flops,
                bytes=m.bytes,
                monitor=m.monitor,
            )
            self._pooled[key] = pooled
            self.measurements.append(pooled)
        else:
            pooled.samples = np.concatenate([pooled.samples, m.samples])


class _Bench:
    """Returned by the ``usv_benchmark`` fixture; call it to time a callable."""

    def __init__(self, request, store: _Store, default_timer: str) -> None:
        self._request = request
        self._store = store
        self._default_timer = default_timer
        self._rounds = request.config.getoption("--benchmark-interleave")
        # Set only on items repeated for --benchmark-interleave.
        self._round = getattr(request.node, "_usv_round", None)

    @property
    def _base_name(self) -> str:
        # node name carries the parametrize id, e.g. "test_gemm[bf16-N4096]"
        return _strip_round(self._request.node.name, self._round)

    def _record(self, m: Measurement) -> Measurement:
        key = None
        if self._round is not None:
            key = (_strip_round(self._request.node.nodeid, self._round), m.name)
        self._store.add(m, key)
        return m

    def _kw(self, kwargs: dict, timer) -> dict:
        kw = dict(kwargs, timer=self._default_timer if timer is None else timer)
        return _scale_for_rounds(kw, self._rounds) if self._round is not None else kw

    def __call__(self, fn, *, name=None, timer=None, **kwargs) -> Measurement:
        """Time *fn* via :func:`usv.do_bench` and record the result."""
        return self._record(do_bench(fn, name=name or self._base_name, **self._kw(kwargs, timer)))

    def many(self, fns, *, interleave=True, timer=None, **kwargs) -> dict:
        """Time a group of callables together (interleaved by default)."""
        res = do_bench_many(fns, interleave=interleave, **self._kw(kwargs, timer))
        for m in res.values():
            self._record(m)
        return res

    def backward(self, fwd, bwd, *, name=None, timer=None, **kwargs) -> Measurement:
        """Time the backward of *fwd* via :func:`usv.do_bench_backward`; records ``<name> [bwd]``."""
        kw = dict(kwargs, timer=self._default_timer if timer is None else timer)
        if self._round is not None:
            kw = _scale_backward_for_rounds(kw, self._rounds)
        base = name or self._base_name
        return self._record(do_bench_backward(fwd, bwd, name=f"{base} [bwd]", **kw))

    def fwd_bwd(self, fwd, fwd_bwd, *, fwd_flops=None, name=None, timer=None, **kwargs):
        """Record a forward row and a derived backward row.

        Backward is derived as ``(fwd+bwd) - fwd`` (per sample); its flops
        default to twice the forward's (dX + dW).  Returns ``(m_fwd, m_bwd)``.
        :meth:`backward` times the backward directly and avoids the occasional
        outliers of the derived value.
        """
        base = name or self._base_name
        kw = self._kw(kwargs, timer)
        m_fwd = do_bench(fwd, name=f"{base} [fwd]", flops=fwd_flops, **kw)
        m_fb = do_bench(fwd_bwd, name=f"{base} [fwd+bwd]", **kw)
        m_bwd = Measurement(
            samples=m_fb.samples - m_fwd.median,
            name=f"{base} [bwd]",
            inner=m_fb.inner,
            flops=None if fwd_flops is None else 2 * fwd_flops,
        )
        self._record(m_fwd)
        return m_fwd, self._record(m_bwd)


# asv-style classes as pytest items (--benchmark-asv)


def _matches_any(name: str, patterns) -> bool:
    """pytest's ``python_classes``-style match: a prefix, or a glob with wildcards."""
    return any(
        name.startswith(p) or (any(c in p for c in "*?[") and fnmatch(name, p))
        for p in patterns
    )


def _marks(obj) -> list:
    """The ``pytestmark`` marks of *obj* (from decorators or a class attribute)."""
    marks = getattr(obj, "pytestmark", [])
    return [getattr(m, "mark", m) for m in (marks if isinstance(marks, list) else [marks])]


class AsvClass(pytest.Collector):
    """An asv benchmark class; collects one :class:`AsvItem` per method and parameters."""

    def __init__(self, *, obj: type, **kwargs) -> None:
        super().__init__(**kwargs)
        self.obj = obj
        self.own_markers.extend(_marks(obj))
        self._cache_args: tuple | None = None

    def collect(self):
        rounds = self.config.getoption("--benchmark-interleave")
        cases = []
        for method, index, combo, names in _enumerate_class(self.obj):
            axes = _param_spec(self.obj, method)[0]
            ids = "-".join(_param_ids(names, combo, _axis_indices(axes, index)))
            cases.append((method, index, combo, names, ids))
        # Equal ids (e.g. values 1 and "1") get a counter, as pytest does, skipping taken ids.
        counts = Counter((method, ids) for method, _, _, _, ids in cases)
        taken = set(counts)
        seen: Counter = Counter()
        items = []
        for method, index, combo, names, ids in cases:
            if counts[method, ids] > 1:
                base = ids
                while (method, ids) in taken:
                    ids = f"{base}_{seen[method, base]}"
                    seen[method, base] += 1
                taken.add((method, ids))
            for rnd in range(1, rounds + 1) if rounds > 1 else [None]:
                parts = [p for p in (ids, f"r{rnd}" if rnd is not None else "") if p]
                items.append(
                    AsvItem.from_parent(
                        self,
                        name=f"{method}[{'-'.join(parts)}]" if parts else method,
                        method=method,
                        combo_index=index,
                        combo=combo,
                        param_names=names,
                        rnd=rnd,
                    )
                )
        return items

    def cache_args(self) -> tuple:
        """``(setup_cache(),)`` computed once per class, or ``()`` without one."""
        if self._cache_args is None:
            inst = self.obj()
            self._cache_args = (inst.setup_cache(),) if hasattr(inst, "setup_cache") else ()
        return self._cache_args


class AsvItem(pytest.Item):
    """One asv benchmark: a ``time_*`` method with one parameter combination."""

    def __init__(self, *, method, combo_index, combo, param_names, rnd=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.method = method
        self.combo_index = combo_index
        self.combo = combo
        self.param_names = param_names
        self.own_markers.extend(_marks(getattr(self.parent.obj, method)))
        if rnd is not None:
            self._usv_round = rnd

    def runtest(self) -> None:
        config = self.config
        cls = self.parent.obj
        kwargs = {**asv_bench_kwargs(cls, self.method), **getattr(config, "_usv_asv_kwargs", {})}
        kwargs.setdefault("timer", config.getoption("--benchmark-timer"))
        if hasattr(self, "_usv_round"):
            kwargs = _scale_for_rounds(kwargs, config.getoption("--benchmark-interleave"))
        label = _label(cls, self.method, self.param_names, self.combo)
        try:
            if config.getoption("--benchmark-fresh-process"):
                m = _run_in_subprocess(_import_path(cls), self.method, self.combo_index, kwargs)
            else:
                args = self.parent.cache_args() + self.combo
                inst = cls()
                _setup(inst, args, label)
                try:
                    method = getattr(inst, self.method)
                    m = do_bench(_bench_callable(method, args), name=label, **kwargs)
                finally:
                    _call(inst, "teardown", args)
        except SkipBenchmark as e:
            pytest.skip(str(e))
        rnd = getattr(self, "_usv_round", None)
        config._usv_store.add(m, rnd and (_strip_round(self.nodeid, rnd), m.name))

    def reportinfo(self):
        code = getattr(getattr(self.parent.obj, self.method), "__code__", None)
        line = code.co_firstlineno - 1 if code is not None else 0  # 0-based, as pytest expects
        return self.path, line, f"{self.parent.obj.__name__}.{self.name}"


def _is_benchmark(item) -> bool:
    return isinstance(item, AsvItem) or "usv_benchmark" in getattr(item, "fixturenames", ())


# session setup before the first benchmark


@dataclass
class _Session:
    started: bool = False
    pin: object = None
    pin_description: str | None = None
    gpu: dict = field(default_factory=dict)


def _say(config, msg: str) -> None:
    """Print *msg* on the terminal, outside of pytest's output capturing."""
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        print(msg, file=sys.__stderr__)
        return
    capman = config.pluginmanager.get_plugin("capturemanager")
    with capman.global_and_fixture_disabled() if capman else nullcontext():
        tr.write_line(msg)


def _proc_cmdline(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            parts = [p.decode("utf-8", "replace") for p in f.read().split(b"\0") if p]
        return " ".join(parts) or None
    except OSError:  # e.g. a host PID seen from inside a container
        return None


def _format_interference(foreign, pci, gpu_was_initialized: bool) -> str:
    lines = [f"usv: GPU INTERFERENCE: {len(foreign)} other process(es) on the benchmark GPU ({pci}):"]
    for p in foreign:
        cmd = _proc_cmdline(p["pid"]) or p["name"] or "?"
        lines.append(f"  PID {p['pid']:<8} {cmd[:150]}")
    if gpu_was_initialized:
        lines.append(
            "  (the GPU was initialized before the first benchmark: in a container, this "
            "process may be listed under its host PID)"
        )
    return "\n".join(lines)


def _start_session(config) -> None:
    """Interference snapshot, GPU init, CPU pinning, then the interference report."""
    st = config._usv_session
    if st.started:
        return
    st.started = True
    from usv.cpupin import pin_cpus
    from usv.interference import foreign_processes, snapshot_gpu_processes
    from usv.sysinfo import _gpu_props, probe_gpu

    fresh = config.getoption("--benchmark-fresh-process")
    check = config.getoption("--benchmark-gpu-interference")
    pin = not config.getoption("--benchmark-no-cpu-pin")
    torch = sys.modules.get("torch")
    initialized = torch is not None and torch.cuda.is_initialized()
    # Before this process touches the GPU, the SMI tools list genuine neighbors only.
    snapshot = snapshot_gpu_processes() if check != "off" else None
    gpu = {}
    if fresh and (pin or check != "off"):
        gpu = probe_gpu()  # the parent stays off the GPU; children do the work
    elif pin or check != "off":
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.init()  # creates the runtime's helper threads, which pinning moves
        gpu = _gpu_props()
    st.gpu = gpu

    if pin:
        st.pin = pin_cpus(pci=gpu.get("pci", ""), move_helpers=not fresh)
        st.pin_description = st.pin.description
        if fresh:
            helpers = ",".join(map(str, st.pin.helpers))
            os.environ["USV_PIN_HELPERS"] = helpers
            st.pin_description = (
                f"main thread pinned to cpus {','.join(map(str, st.pin.cpus))}, helper threads "
                f"of each benchmark process to {helpers} ({st.pin.governor})"
            )
        _say(config, f"usv: {st.pin_description}")

    if check != "off":
        if snapshot is None:
            _say(config, "usv: GPU interference check unavailable (no amd-smi / nvidia-smi)")
        elif not gpu.get("pci"):
            _say(config, "usv: GPU interference check unavailable (unknown PCI address of the GPU)")
        else:
            foreign = foreign_processes(snapshot, gpu.get("pci"), exclude_pids={os.getpid()})
            if foreign:
                _say(config, _format_interference(foreign, gpu.get("pci"), initialized))
                if check == "abort":
                    pytest.exit(
                        "usv: GPU interference detected (--benchmark-gpu-interference=abort)",
                        returncode=3,
                    )


def _run_info(config) -> dict[str, str]:
    from usv.sysinfo import _gpu_props, probe_gpu, run_info

    st = config._usv_session
    gpu = st.gpu
    torch = sys.modules.get("torch")
    if not gpu and torch is not None and torch.cuda.is_initialized():
        gpu = _gpu_props()
    elif not gpu and config.getoption("--benchmark-fresh-process"):
        gpu = probe_gpu()  # the benchmarks ran in child processes
    extra = {}
    rounds = config.getoption("--benchmark-interleave")
    if rounds > 1:
        extra["interleave"] = f"{rounds} rounds"
    if config.getoption("--benchmark-fresh-process"):
        extra["fresh_process"] = "yes"
    return run_info(gpu=gpu, cpus=st.pin_description, **extra)


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

    def fmt(r):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

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
    group.addoption(
        "--benchmark-interleave",
        type=int,
        default=1,
        metavar="R",
        help="measure every benchmark in R rounds spread over the session, each with "
        "1/R of the samples, and pool the samples (default: 1)",
    )
    group.addoption(
        "--benchmark-no-cpu-pin",
        action="store_true",
        default=False,
        help="don't pin benchmarks to GPU-local cores at the performance governor",
    )
    group.addoption(
        "--benchmark-gpu-interference",
        choices=("warn", "abort", "off"),
        default="warn",
        help="report (warn), abort on, or don't check for (off) other processes on "
        "the benchmark GPU (default: warn)",
    )
    group.addoption(
        "--benchmark-asv",
        action="store_true",
        default=False,
        help="collect asv-style benchmark classes (time_* methods) as benchmarks",
    )
    group.addoption(
        "--benchmark-fresh-process",
        action="store_true",
        default=False,
        help="run each asv benchmark in its own Python process",
    )
    parser.addini(
        "usv_asv_files",
        type="args",
        default=["bench*.py"],
        help="file patterns searched for asv classes with --benchmark-asv "
        "(besides python_files)",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: usv GPU microbenchmark (deselect with -m 'not benchmark')",
    )
    if config.getoption("--benchmark-interleave") < 1:
        raise pytest.UsageError("--benchmark-interleave expects a number of rounds >= 1")
    config._usv_store = _Store()
    config._usv_session = _Session()


@pytest.fixture
def usv_benchmark(request):
    """Time GPU work in a test; see the plugin module docstring for usage."""
    return _Bench(
        request,
        request.config._usv_store,
        request.config.getoption("--benchmark-timer"),
    )


@pytest.fixture
def _usv_round(request):
    return getattr(request, "param", 1)


@pytest.hookimpl(trylast=True)
def pytest_generate_tests(metafunc) -> None:
    rounds = metafunc.config.getoption("--benchmark-interleave")
    if rounds > 1 and "usv_benchmark" in metafunc.fixturenames:
        metafunc.fixturenames.append("_usv_round")
        metafunc.parametrize(
            "_usv_round", range(1, rounds + 1), indirect=True, ids=lambda r: f"r{r}"
        )


def pytest_collect_file(file_path, parent):
    """With --benchmark-asv, also collect asv suite files (``usv_asv_files``)."""
    config = parent.config
    if (
        not config.getoption("--benchmark-asv")
        or file_path.suffix != ".py"
        or parent.session.isinitpath(file_path)
        or any(fnmatch(file_path.name, p) for p in config.getini("python_files"))
    ):
        return None  # pytest's own collection handles these
    if any(fnmatch(file_path.name, p) for p in config.getini("usv_asv_files")):
        return pytest.Module.from_parent(parent, path=file_path)
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makeitem(collector, name, obj):
    """With --benchmark-asv, collect classes with ``time_*`` methods as asv suites."""
    if not collector.config.getoption("--benchmark-asv") or not is_asv_class(obj):
        return None
    if obj.__module__ != getattr(collector.obj, "__name__", None):
        return None  # imported from elsewhere, or nested in a class
    if _matches_any(name, collector.config.getini("python_classes")):
        return None  # a test class: leave it to pytest
    return AsvClass.from_parent(collector, name=name, obj=obj)


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    """Mark benchmarks, honor ``--benchmark-only``, then order interleave rounds.

    Marking runs before the other plugins (so ``-m "not benchmark"`` works); the
    round ordering after them (so it sees the final selection).
    """
    only = config.getoption("--benchmark-only")
    selected, deselected = [], []
    for item in items:
        is_bench = _is_benchmark(item)
        if is_bench:
            item.add_marker("benchmark")
            params = getattr(getattr(item, "callspec", None), "params", {})
            if "_usv_round" in params:
                item._usv_round = params["_usv_round"]
        (deselected if (only and not is_bench) else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    yield

    rounds = config.getoption("--benchmark-interleave")
    if rounds > 1:
        by_round: list[list] = [[] for _ in range(rounds)]
        others = []
        for item in items:
            rnd = getattr(item, "_usv_round", None)
            (by_round[rnd - 1] if rnd else others).append(item)
        # Fixed seeds: a rerun visits the benchmarks in the same order.
        for r, round_items in enumerate(by_round):
            random.Random(r).shuffle(round_items)
        items[:] = others + [item for round_items in by_round for item in round_items]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item) -> None:
    if _is_benchmark(item):
        _start_session(item.config)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    store = getattr(config, "_usv_store", None)
    if not store or not store.measurements:
        return
    tr = terminalreporter
    tr.write_sep("-", "usv benchmarks")
    tr.write_line(format_table(store.measurements))
    save = config.getoption("--benchmark-save")
    path = getattr(config, "_usv_saved_path", None)
    if save and path:
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
    """Undo the CPU pinning, save results, and fail the session if a benchmark regressed.

    Saving happens here rather than in the terminal summary, so it also works without
    the terminal plugin (as in :func:`usv.run_benchmarks`).
    """
    config = session.config
    st = getattr(config, "_usv_session", None)
    if st is not None and st.pin is not None:
        st.pin.restore()
        st.pin = None
        os.environ.pop("USV_PIN_HELPERS", None)
    store = getattr(config, "_usv_store", None)
    if not store or not store.measurements:
        return
    compare = config.getoption("--benchmark-compare")
    # Resolve the baseline before saving, so "last" can't pick this run's own file.
    cmps = _ensure_comparison(config) if compare else []
    save = config.getoption("--benchmark-save")
    if save:
        from usv.sysinfo import write_run_info

        path = save_results(store.measurements, label=save if isinstance(save, str) else None)
        write_run_info(os.path.splitext(path)[0] + "-info.txt", _run_info(config))
        config._usv_saved_path = path
    if any(c.regressed for c in cmps) and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

