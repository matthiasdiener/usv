"""Run ``asv``-style benchmark classes on top of :func:`usv.do_bench`.

`airspeed-velocity <https://github.com/airspeed-velocity/asv>`_ organizes
benchmarks as *classes*: methods named ``time_*`` are the benchmarks, with
optional ``setup`` / ``teardown`` hooks, a ``setup_cache`` that runs once, and
``params`` / ``param_names`` for parameter sweeps.  This module discovers such
classes and runs their ``time_*`` methods through usv's timer, so existing asv
benchmark suites (or that familiar structure) can be reused.

The pytest plugin is the runner: ``pytest --benchmark-asv`` collects asv classes as
benchmark items (one per method and parameter combination), so they get the same
session features as fixture-based benchmarks.  :func:`run_benchmarks` is the
programmatic entry point and runs importable classes through that runner too::

    class MatmulSuite:
        params = [512, 1024, 2048]
        param_names = ["n"]

        def setup(self, n):
            self.a = torch.randn(n, n, device="cuda")

        def time_matmul(self, n):
            return self.a @ self.a

    from usv.asv import run_benchmarks
    results = run_benchmarks(MatmulSuite)     # -> {name: Measurement}

The asv timing attributes ``number``, ``sample_time``, ``repeat``, ``warmup_time`` and
``timeout`` (on the method or the class) map to :func:`usv.do_bench` arguments; see
:func:`asv_bench_kwargs`.  Everything is opt-in - importing usv does not scan or run
anything.
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import json
import os
import subprocess
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from usv.bench import Measurement, do_bench

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["run_benchmarks", "asv_bench_kwargs"]


def _load_class(path: str) -> type:
    """Import ``"module:QualName"`` (or ``"module.QualName"``) and return the class."""
    if ":" in path:
        mod_name, _, qual = path.partition(":")
    else:
        mod_name, _, qual = path.rpartition(".")
    if not mod_name or not qual:
        raise ValueError(f"cannot resolve benchmark class from {path!r}")
    obj: object = importlib.import_module(mod_name)
    for part in qual.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"{path!r} is not a class")
    return obj


def _discover_classes(module: ModuleType) -> list[type]:
    """Return classes in *module* that define at least one ``time_*`` method."""
    found = []
    for obj in vars(module).values():
        if is_asv_class(obj):
            found.append(obj)
    return found


def is_asv_class(obj) -> bool:
    """True for a class with at least one ``time_*`` attribute."""
    return isinstance(obj, type) and any(a.startswith("time_") for a in dir(obj))


def _resolve(target) -> list[tuple[type, object | None]]:
    """Normalize *target* to a list of ``(class, instance_or_None)`` pairs."""
    if isinstance(target, str):
        if ":" in target or _looks_like_class_path(target):
            return [(_load_class(target), None)]
        module = importlib.import_module(target)
        return [(c, None) for c in _discover_classes(module)]
    if isinstance(target, ModuleType):
        return [(c, None) for c in _discover_classes(target)]
    if isinstance(target, type):
        return [(target, None)]
    # An already-constructed instance: reuse it (preserves any state).
    return [(type(target), target)]


def _looks_like_class_path(path: str) -> bool:
    """Heuristic: a dotted path whose last segment names a class (CapWords)."""
    tail = path.rpartition(".")[2]
    return bool(tail) and tail[:1].isupper()


def _axes(params) -> list[list]:
    """Normalize asv ``params`` into a list of value-axes (Cartesian factors)."""
    if not params:
        return []
    if all(isinstance(p, (list, tuple)) for p in params):
        return [list(p) for p in params]
    return [list(params)]  # a flat list is a single axis


def _param_spec(cls: type, method_name: str) -> tuple[list[list], list[str]]:
    """Resolve the (axes, names) for a benchmark, method-level overriding class-level."""
    method = getattr(cls, method_name, None)
    params = getattr(method, "params", None)
    names = getattr(method, "param_names", None)
    if params is None:
        params = getattr(cls, "params", None)
        names = getattr(cls, "param_names", None)
    axes = _axes(params)
    names = list(names) if names else []
    return axes, names


def _label(cls: type, method_name: str, names: list[str], combo: tuple) -> str:
    base = f"{cls.__name__}.{method_name}"
    if not combo:
        return base
    if len(names) == len(combo):
        inside = ", ".join(f"{n}={v!r}" for n, v in zip(names, combo))
    else:
        inside = ", ".join(repr(v) for v in combo)
    return f"{base}({inside})"


def _param_ids(names: list[str], combo: tuple) -> list[str]:
    """pytest-style ids for a parameter combination (values, or ``name<index>``)."""
    ids = []
    for i, v in enumerate(combo):
        if v is None or isinstance(v, (str, int, float, bool)):
            ids.append(str(v))
        else:
            ids.append(f"{names[i] if i < len(names) else 'p'}{i}")
    return ids


_UNSET = object()


def _asv_attr(cls: type, method_name: str, attr: str):
    """An asv benchmark attribute: the method's own, else the class's, else unset."""
    method = getattr(cls, method_name, None)
    value = getattr(method, attr, _UNSET)
    if value is _UNSET:
        value = getattr(cls, attr, _UNSET)
    return value


def asv_bench_kwargs(cls: type, method_name: str) -> dict:
    """Map the asv timing attributes of a benchmark to :func:`usv.do_bench` arguments.

    Only attributes set on the method (or else the class) are mapped:

    * ``number`` - calls per sample: ``inner=number``; ``0`` means automatic, i.e.
      ``inner="auto"``.
    * ``sample_time`` - target duration of one sample: ``inner="auto"`` with
      ``target_window_s=sample_time`` (unless ``number`` fixes the count).
    * ``repeat`` - number of samples: ``iters=repeat``; for asv's
      ``(min_repeat, max_repeat, max_time)`` form, ``iters=min_repeat``.
    * ``warmup_time`` - seconds of warmup: ``min_warmup_time``; ``0`` disables
      warmup, a negative value keeps the default.
    * ``timeout`` - seconds before the benchmark is aborted: ``timeout``.
    """
    kw: dict = {}
    number = _asv_attr(cls, method_name, "number")
    if number is not _UNSET and number is not None:
        kw["inner"] = int(number) if int(number) > 0 else "auto"
    sample_time = _asv_attr(cls, method_name, "sample_time")
    if sample_time is not _UNSET and sample_time:
        kw["target_window_s"] = float(sample_time)
        kw.setdefault("inner", "auto")
    repeat = _asv_attr(cls, method_name, "repeat")
    if isinstance(repeat, (tuple, list)) and repeat:
        repeat = repeat[0]
    if repeat is not _UNSET and repeat:
        kw["iters"] = int(repeat)
    warmup_time = _asv_attr(cls, method_name, "warmup_time")
    if warmup_time is not _UNSET and warmup_time is not None:
        if warmup_time == 0:
            kw["warmup"] = 0
        elif warmup_time > 0:
            kw["min_warmup_time"] = float(warmup_time)
    timeout = _asv_attr(cls, method_name, "timeout")
    if timeout is not _UNSET and timeout:
        kw["timeout"] = float(timeout)
    return kw


def _call(inst, name: str, args: tuple) -> None:
    """Invoke an optional hook ``inst.name(*args)`` if it exists."""
    hook = getattr(inst, name, None)
    if hook is not None:
        hook(*args)


def _bench_callable(method: "Callable", args: tuple) -> "Callable[[], object]":
    return lambda m=method, a=args: m(*a)


def _enumerate_class(cls: type) -> list[tuple[str, int, tuple, list[str]]]:
    """List ``(method_name, combo_index, combo, param_names)`` for *cls*.

    Deterministic (sorted methods, product order) so a subprocess can select a
    benchmark by ``(method_name, combo_index)`` without serializing param values.
    """
    out = []
    for method_name in sorted(m for m in dir(cls) if m.startswith("time_")):
        axes, names = _param_spec(cls, method_name)
        combos = list(itertools.product(*axes)) if axes else [()]
        for index, combo in enumerate(combos):
            out.append((method_name, index, combo, names))
    return out


def _run_single(cls: type, method_name: str, combo: tuple, kwargs: dict) -> Measurement:
    """Run one benchmark on a *fresh* instance (setup_cache -> setup -> time -> teardown)."""
    inst = cls()
    cache_args = (inst.setup_cache(),) if hasattr(inst, "setup_cache") else ()
    args = cache_args + combo
    names = _param_spec(cls, method_name)[1]
    label = _label(cls, method_name, names, combo)
    kwargs = {**asv_bench_kwargs(cls, method_name), **kwargs}
    _call(inst, "setup", args)
    try:
        method = getattr(inst, method_name)
        return do_bench(_bench_callable(method, args), name=label, **kwargs)
    finally:
        _call(inst, "teardown", args)


def run_one_by_index(target: str, method_name: str, combo_index: int, kwargs: dict) -> Measurement:
    """Load *target* and run the single benchmark ``(method_name, combo_index)``.

    Used by the fresh-process worker (:mod:`usv._subproc`).
    """
    cls = _load_class(target)
    for name, index, combo, _names in _enumerate_class(cls):
        if name == method_name and index == combo_index:
            return _run_single(cls, method_name, combo, kwargs)
    raise LookupError(f"no benchmark {method_name!r}[{combo_index}] on {target!r}")


def _import_path(cls: type) -> str:
    """Return a re-importable ``"module:QualName"`` for *cls*, or raise."""
    mod = getattr(cls, "__module__", None)
    qual = getattr(cls, "__qualname__", None)
    if not mod or not qual or mod == "__main__":
        raise ValueError(
            f"fresh_process needs an importable benchmark class; {cls!r} is not "
            "importable (define it in a module, or pass a 'module:ClassName' string)"
        )
    path = f"{mod}:{qual}"
    try:
        resolved = _load_class(path)
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError(f"fresh_process could not re-import {path!r}") from e
    if resolved is not cls:
        raise ValueError(f"fresh_process could not re-import {path!r} to the same class")
    return path


def _run_in_subprocess(path: str, method_name: str, combo_index: int, kwargs: dict) -> Measurement:
    try:
        payload = json.dumps(
            {"target": path, "method": method_name, "combo_index": combo_index, "kwargs": kwargs}
        )
    except TypeError as e:
        raise TypeError(
            "fresh_process requires JSON-serializable do_bench kwargs "
            "(e.g. timer='wall', not a GPUTimer instance)"
        ) from e

    # Propagate the parent's import path so the worker can import the same modules.
    env = os.environ.copy()
    parent_path = os.pathsep.join(p for p in sys.path if p)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = parent_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = parent_path

    # If a per-benchmark timeout is set, the child aborts itself first; give the
    # parent a small margin before it force-kills a wedged (hung-GPU) child.
    timeout = kwargs.get("timeout")
    run_timeout = timeout + 10.0 if timeout else None

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "usv._subproc", payload],
            capture_output=True,
            text=True,
            env=env,
            timeout=run_timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            f"fresh-process benchmark {method_name!r}[{combo_index}] of {path!r} "
            f"exceeded {run_timeout:g}s and was killed (GPU hang?)"
        ) from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"fresh-process benchmark {method_name!r}[{combo_index}] of {path!r} failed:\n"
            f"{proc.stderr.strip()}"
        )
    try:
        data = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        raise RuntimeError(
            f"fresh-process worker produced no result for {method_name!r}[{combo_index}]:\n"
            f"{proc.stdout.strip()}"
        ) from e
    return Measurement(
        np.asarray(data["samples"], dtype=np.float64),
        name=data["name"],
        inner=data["inner"],
        flops=data["flops"],
        bytes=data["bytes"],
    )


def _run_class(cls: type, inst, kwargs: dict) -> dict[str, Measurement]:
    """Run every ``time_*`` method of *cls* over its parameter grid, in process."""
    if inst is None:
        inst = cls()
    has_cache = hasattr(inst, "setup_cache")
    cache_args = (inst.setup_cache(),) if has_cache else ()

    results: dict[str, Measurement] = {}
    for method_name in sorted(m for m in dir(inst) if m.startswith("time_")):
        method = getattr(inst, method_name)
        axes, names = _param_spec(cls, method_name)
        for combo in itertools.product(*axes) if axes else [()]:
            args = cache_args + combo
            _call(inst, "setup", args)
            try:
                label = _label(cls, method_name, names, combo)
                results[label] = do_bench(
                    _bench_callable(method, args),
                    name=label,
                    **{**asv_bench_kwargs(cls, method_name), **kwargs},
                )
            finally:
                _call(inst, "teardown", args)
    return results


def _pytest_nodeid(cls: type) -> str | None:
    """``"<file>::<Class>"`` if pytest can collect *cls* from its module file, else None."""
    module = sys.modules.get(getattr(cls, "__module__", ""))
    if module is None or module.__name__ == "__main__" or "." in cls.__qualname__:
        return None
    if getattr(module, cls.__name__, None) is not cls:
        return None
    try:
        path = inspect.getsourcefile(module)
    except TypeError:
        return None
    if not path or not path.endswith(".py"):
        return None
    return f"{os.path.abspath(path)}::{cls.__name__}"


class _RunCapture:
    """pytest plugin object that hands kwargs to asv items and collects the outcome."""

    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs
        self.errors: list[str] = []
        self.measurements: list[Measurement] = []

    def pytest_configure(self, config) -> None:
        config._usv_asv_kwargs = self.kwargs

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.errors.append(f"collecting {report.nodeid}:\n{report.longreprtext}")

    def pytest_runtest_logreport(self, report) -> None:
        if report.failed:
            self.errors.append(f"{report.nodeid} ({report.when}):\n{report.longreprtext}")

    def pytest_sessionfinish(self, session) -> None:
        store = getattr(session.config, "_usv_store", None)
        self.measurements = list(store.measurements) if store else []


def _run_via_pytest(
    nodeids: list[str],
    kwargs: dict,
    *,
    fresh_process: bool,
    interleave: int,
    cpu_pin: bool,
    gpu_interference: str,
    pytest_args,
) -> dict[str, Measurement]:
    import pytest

    rootdir = os.path.commonpath([os.path.dirname(n.partition("::")[0]) for n in nodeids])
    args = [
        *nodeids,
        "--benchmark-asv",
        f"--benchmark-interleave={interleave}",
        f"--benchmark-gpu-interference={gpu_interference}",
        f"--rootdir={rootdir}",
        "-o", "addopts=",  # ignore the project's pytest addopts
        "--noconftest",
        "-p", "no:cacheprovider",
        "-p", "no:terminal",
        # Only asv classes: no test functions/classes of the target modules.
        "-o", "python_functions=_usv_none_",
        "-o", "python_classes=_usv_none_",
    ]
    if fresh_process:
        args.append("--benchmark-fresh-process")
    if not cpu_pin:
        args.append("--benchmark-no-cpu-pin")
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        args += ["-p", "usv.pytest_plugin"]
    args += list(pytest_args)

    capture = _RunCapture(kwargs)
    rc = pytest.main(args, plugins=[capture])
    if capture.errors:
        raise RuntimeError("usv: benchmark run failed:\n" + "\n\n".join(capture.errors))
    if rc not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        raise RuntimeError(f"usv: benchmark run failed (pytest exit code {int(rc)})")
    return {m.name: m for m in capture.measurements}


def run_benchmarks(
    target,
    *,
    fresh_process: bool = False,
    interleave: int = 1,
    cpu_pin: bool = False,
    gpu_interference: str = "off",
    pytest_args=(),
    **do_bench_kwargs,
) -> dict[str, Measurement]:
    """Discover and run ``asv``-style benchmark classes, returning Measurements.

    *target* may be a benchmark class, an instance, a module (all of its
    benchmark classes are run), or a string ``"module:ClassName"`` /
    ``"module"``.  Recognized asv members: ``time_*`` methods (the benchmarks),
    ``setup(self, *params)`` / ``teardown(self, *params)`` hooks, ``setup_cache``
    (run once; its result is passed as the first argument to setup/teardown and
    each benchmark), ``params`` / ``param_names`` for sweeps (a flat ``params``
    list is one axis; a list of lists is a Cartesian product), and the timing
    attributes mapped by :func:`asv_bench_kwargs`.

    Classes defined at module scope in a ``.py`` file run through the pytest runner
    (``pytest --benchmark-asv``): each benchmark gets a fresh instance, a ``setup``
    that raises ``NotImplementedError`` skips it, and the session options apply:
    *interleave* (``--benchmark-interleave``: measure each benchmark in that many
    rounds spread over the run), *cpu_pin* (pin to GPU-local cores at the
    performance governor, restored afterwards) and *gpu_interference* (``"warn"``,
    ``"abort"`` or ``"off"``).  *pytest_args* are passed on to pytest.  A pre-built
    instance, or a class pytest can't collect (in ``__main__`` or a local scope),
    runs directly instead, sharing one instance per class, without those options.

    With ``fresh_process=True`` each benchmark runs in its own freshly spawned
    Python process (asv's default isolation), so per-benchmark warmup state,
    allocator fragmentation, and module-level caches never leak between
    benchmarks.  This requires an *importable* target (a ``"module:ClassName"``
    string, or a class defined at module scope - not in ``__main__`` or a local
    scope) and JSON-serializable ``do_bench`` keywords (e.g. ``timer='wall'``).

    Extra keyword arguments are forwarded to :func:`usv.do_bench` (``warmup``,
    ``iters``, ``timer``, ``cache_flush``, ...) and take precedence over the asv
    attributes.  Benchmarks are named ``ClassName.time_method`` with a parameter
    suffix.
    """
    results: dict[str, Measurement] = {}
    nodeids: list[str] = []
    for cls, inst in _resolve(target):
        nodeid = _pytest_nodeid(cls) if inst is None else None
        if nodeid is not None:
            nodeids.append(nodeid)
        elif fresh_process:
            if inst is not None:
                raise ValueError(
                    "fresh_process needs an importable class or 'module:ClassName' "
                    "string, not a pre-built instance"
                )
            path = _import_path(cls)
            for method_name, combo_index, combo, names in _enumerate_class(cls):
                label = _label(cls, method_name, names, combo)
                results[label] = _run_in_subprocess(
                    path, method_name, combo_index, do_bench_kwargs
                )
        else:
            results.update(_run_class(cls, inst, do_bench_kwargs))
    if nodeids:
        results.update(
            _run_via_pytest(
                nodeids,
                do_bench_kwargs,
                fresh_process=fresh_process,
                interleave=interleave,
                cpu_pin=cpu_pin,
                gpu_interference=gpu_interference,
                pytest_args=pytest_args,
            )
        )
    return results
