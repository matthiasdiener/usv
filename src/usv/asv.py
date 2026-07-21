"""Run ``asv``-style benchmark classes on top of :func:`usv.do_bench`.

`airspeed-velocity <https://github.com/airspeed-velocity/asv>`_ organizes
benchmarks as *classes*: methods named ``time_*`` are the benchmarks, with
optional ``setup`` / ``teardown`` hooks, a ``setup_cache`` that runs once, and
``params`` / ``param_names`` for parameter sweeps.  This module discovers such
classes and runs their ``time_*`` methods through usv's timer, so existing asv
benchmark suites (or that familiar structure) can be reused.

::

    class MatmulSuite:
        params = [512, 1024, 2048]
        param_names = ["n"]

        def setup(self, n):
            self.a = torch.randn(n, n, device="cuda")

        def time_matmul(self, n):
            return self.a @ self.a

    from usv.asv import run_benchmarks
    results = run_benchmarks(MatmulSuite)     # -> {name: Measurement}

Everything is opt-in - importing usv does not scan or run anything.
"""

from __future__ import annotations

import importlib
import itertools
from types import ModuleType
from typing import TYPE_CHECKING

from usv.bench import Measurement, do_bench

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["run_benchmarks"]


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
        if isinstance(obj, type) and any(a.startswith("time_") for a in dir(obj)):
            found.append(obj)
    return found


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


def _call(inst, name: str, args: tuple) -> None:
    """Invoke an optional hook ``inst.name(*args)`` if it exists."""
    hook = getattr(inst, name, None)
    if hook is not None:
        hook(*args)


def _bench_callable(method: "Callable", args: tuple) -> "Callable[[], object]":
    return lambda m=method, a=args: m(*a)


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
                    _bench_callable(method, args), name=label, **kwargs
                )
            finally:
                _call(inst, "teardown", args)
    return results


def run_benchmarks(target, **do_bench_kwargs) -> dict[str, Measurement]:
    """Discover and run ``asv``-style benchmark classes, returning Measurements.

    *target* may be a benchmark class, an instance, a module (all of its
    benchmark classes are run), or a string ``"module:ClassName"`` /
    ``"module"``.  Recognized asv members: ``time_*`` methods (the benchmarks),
    ``setup(self, *params)`` / ``teardown(self, *params)`` hooks, ``setup_cache``
    (run once; its result is passed as the first argument to setup/teardown and
    each benchmark), and ``params`` / ``param_names`` for sweeps (a flat
    ``params`` list is one axis; a list of lists is a Cartesian product).

    Extra keyword arguments are forwarded to :func:`usv.do_bench` (``warmup``,
    ``iters``, ``timer``, ``cache_flush``, ...).  Benchmarks are named
    ``ClassName.time_method`` with a parameter suffix.
    """
    results: dict[str, Measurement] = {}
    for cls, inst in _resolve(target):
        results.update(_run_class(cls, inst, do_bench_kwargs))
    return results
