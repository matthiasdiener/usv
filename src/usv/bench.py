"""Function-style GPU micro-benchmarking.

The core API mirrors ``triton.testing.do_bench`` / ``torch.utils.benchmark``:
time a plain callable and get statistics back.  Two entry points:

* :func:`do_bench` - time a single callable.  There is only one kernel, so
  there is nothing to interleave with.
* :func:`do_bench_many` - time a *group* of named callables together.  By
  handing the runner every callable at once it can collect samples
  round-robin, so time-correlated GPU noise (thermal ramp, DVFS, a neighbor
  job) is spread across every benchmark instead of biasing one contiguous
  block.  This is how interleaving is expressed in a functional API.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from usv.clocks import fixed_clocks
from usv.timer import GPUTimer, get_timer

__all__ = [
    "Measurement",
    "do_bench",
    "do_bench_many",
    "rotating",
    "format_table",
]


# result container


@dataclass
class Measurement:
    """Per-call timing samples for one benchmark, with lazy statistics.

    ``samples`` holds per-call seconds (already divided by ``inner``).
    Optional ``flops`` / ``bytes`` describe the work of a single call and, if
    given, enable the :attr:`tflops` / :attr:`gbps` throughput helpers.
    """

    samples: np.ndarray
    name: str = ""
    inner: int = 1
    flops: float | None = None
    bytes: float | None = None

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float64)

    @property
    def n(self) -> int:
        return int(self.samples.size)

    @property
    def median(self) -> float:
        return float(np.median(self.samples))

    @property
    def mean(self) -> float:
        return float(self.samples.mean())

    @property
    def std(self) -> float:
        return float(self.samples.std(ddof=0))

    @property
    def min(self) -> float:
        return float(self.samples.min())

    @property
    def max(self) -> float:
        return float(self.samples.max())

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    @property
    def tflops(self) -> float | None:
        m = self.median
        return self.flops / m / 1e12 if (self.flops and m > 0) else None

    @property
    def gbps(self) -> float | None:
        m = self.median
        return self.bytes / m / 1e9 if (self.bytes and m > 0) else None

    def __repr__(self) -> str:
        tag = f"{self.name}: " if self.name else ""
        out = f"{tag}{self.median * 1e3:.4f} ms +/- {self.std * 1e3:.4f} (median+/-std, n={self.n}"
        if self.inner > 1:
            out += f", inner={self.inner}"
        out += ")"
        if self.tflops is not None:
            out += f"  {self.tflops:.2f} TFLOP/s"
        if self.gbps is not None:
            out += f"  {self.gbps:.1f} GB/s"
        return out


# helpers


def rotating(items: list) -> Callable[[], object]:
    """Return a callable that cycles through *items* round-robin.

    Use it to rotate input buffers so successive kernel launches touch
    different memory, reducing cache-residency bias without a full flush::

        bufs = [torch.randn(N, N, device="cuda") for _ in range(8)]
        nxt = rotating(bufs)
        do_bench(lambda: nxt() @ nxt())
    """
    it = itertools.cycle(items)
    return lambda: next(it)


def _l2_cache_size_bytes() -> int | None:
    """Best-effort L2 cache size of the current GPU in bytes (via torch).

    Works for both NVIDIA and AMD/ROCm devices whose ``torch`` build exposes
    ``L2_cache_size`` on the device properties; returns ``None`` otherwise.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        size = getattr(props, "L2_cache_size", None)
        return int(size) if size else None
    except Exception:
        return None


def _resolve_flush_bytes(mb: int | None) -> int:
    """Size of the cache-flush buffer in bytes.

    An explicit *mb* wins.  Otherwise the size is derived from the device L2
    cache (2x, for set-associativity headroom), falling back to 256 MB when
    the L2 size cannot be queried.
    """
    if mb is not None:
        return int(mb) * 1024 * 1024
    l2 = _l2_cache_size_bytes()
    return 2 * l2 if l2 else 256 * 1024 * 1024


def _make_scratch(mb: int | None):
    """Allocate a scratch buffer for cache flushing (torch), or ``None``.

    *mb* is the size in MB, or ``None`` to size it from the device L2 cache.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        n = max(1, _resolve_flush_bytes(mb) // 4)
        return torch.empty(n, dtype=torch.float32, device="cuda")
    except Exception:
        return None


def _estimate_per_call(
    timer: GPUTimer,
    fn: Callable[[], object],
    reps: int = 5,
) -> float:
    """Rough per-call seconds from a short probe, used to size counts.

    Times *reps* back-to-back calls in a single window.  The caller is expected
    to have pre-warmed *fn* already (see the pre-warmup pass in :func:`do_bench_many`),
    so this need only be good enough to convert a time budget or
    ``target_window_s`` into a call count.
    """
    handle = timer.open(fn, reps)
    timer.synchronize()
    return timer.value(handle)


def _inner_for_window(
    per_call_s: float,
    target_s: float,
    max_inner: int = 100_000,
) -> int:
    """Inner count so one timed window lasts about *target_s*."""
    if per_call_s <= 0:
        return 1
    return max(1, min(max_inner, int(target_s / per_call_s) + 1))


def _clock_lock(lock_clocks: bool):
    """Return a context manager that pins GPU clocks to the device max, or a no-op.

    Use :func:`usv.fixed_clocks` directly for a specific SM / memory clock.
    """
    return fixed_clocks() if lock_clocks else nullcontext()


# -- single-callable timing -----------------------------------------------


def do_bench(
    fn: Callable[[], object],
    *,
    warmup: int = 50,
    iters: int = 100,
    inner: int | str = 1,
    target_window_s: float = 1e-3,
    min_warmup_time: float | None = None,
    min_iters_time: float | None = None,
    cache_flush: bool = False,
    flush_mb: int | None = None,
    lock_clocks: bool = False,
    timer: GPUTimer | str = "auto",
    name: str = "",
    flops: float | None = None,
    bytes: float | None = None,
) -> Measurement:
    """Time a single callable, returning a :class:`Measurement`.

    A first *pre-warmup* call is always discarded (absorbing one-shot costs like JIT
    compilation, algorithm selection, or lazy init); then ``warmup`` untimed calls precede ``iters``
    timed samples.  ``inner`` is the number of back-to-back calls per timed
    sample - raise it, or pass ``"auto"`` (fills ``target_window_s``), for
    kernels faster than the timer's resolution.  ``min_warmup_time`` /
    ``min_iters_time`` (seconds) turn
    ``warmup`` / ``iters`` into floors: the phase runs until at least that much
    kernel time elapses, so counts scale up for very short kernels.
    ``cache_flush`` zeroes a scratch buffer before each sample to measure
    cold-cache cost; ``flush_mb`` sets its size in MB and, when ``None``, is
    taken from the device L2 cache.  ``lock_clocks=True`` pins the GPU clock to
    the device max for the run; for a specific frequency, or to lock once across
    many calls, use the :func:`usv.fixed_clocks` context manager instead.

    This is a thin wrapper over :func:`do_bench_many` (a single-entry group
    with no interleaving), unwrapped to one :class:`Measurement`.
    """
    (measurement,) = do_bench_many(
        {name: fn},
        warmup=warmup,
        iters=iters,
        inner=inner,
        target_window_s=target_window_s,
        min_warmup_time=min_warmup_time,
        min_iters_time=min_iters_time,
        interleave=False,
        cache_flush=cache_flush,
        flush_mb=flush_mb,
        lock_clocks=lock_clocks,
        timer=timer,
        flops={name: flops} if flops is not None else None,
        bytes={name: bytes} if bytes is not None else None,
    ).values()
    return measurement


# -- interleaved multi-callable timing ------------------------------------


def do_bench_many(
    fns: dict[str, Callable[[], object]],
    *,
    warmup: int = 50,
    iters: int = 100,
    inner: int | str = 1,
    target_window_s: float = 1e-3,
    min_warmup_time: float | None = None,
    min_iters_time: float | None = None,
    interleave: bool = False,
    cache_flush: bool = False,
    flush_mb: int | None = None,
    lock_clocks: bool = False,
    timer: GPUTimer | str = "auto",
    flops: dict[str, float] | None = None,
    bytes: dict[str, float] | None = None,
) -> dict[str, Measurement]:
    """Time a group of named callables, returning one :class:`Measurement` each.

    By default (``interleave=False``) each callable is timed to completion in
    turn, as if calling :func:`do_bench` repeatedly.  With ``interleave=True``
    samples are instead collected round-robin: each round enqueues one sample
    from every still-active callable - in a freshly shuffled order (seeded for
    reproducibility) - before moving to the next round.  Slow drift in GPU
    clocks or neighboring load then lands on one sample of *each* benchmark
    rather than corrupting a single benchmark's contiguous block.

    A first *pre-warmup* call per callable is always discarded (JIT / lazy init).
    ``warmup`` / ``iters`` are per-callable count floors.  Set
    ``min_warmup_time`` / ``min_iters_time`` (seconds) to instead run each
    callable until that much kernel time elapses; the count is derived per
    callable from a short probe, so a fast kernel takes more samples than a
    slow one for the same budget.  Under interleaving each callable drops out
    of the rotation once it meets its own budget.

    ``inner="auto"`` is resolved per callable.  Throughput columns come from
    the optional ``flops`` / ``bytes`` maps keyed by benchmark name.

    ``lock_clocks=True`` pins the GPU clock to the device max around the whole
    group (via :func:`usv.fixed_clocks`); for a specific frequency, or to lock
    once across many separate calls, use the ``fixed_clocks`` context manager
    directly.
    """
    with _clock_lock(lock_clocks):
        tm = timer if isinstance(timer, GPUTimer) else get_timer(timer)
        names = list(fns)
        flops = flops or {}
        bytes = bytes or {}

        scratch = _make_scratch(flush_mb) if cache_flush else None
        before = (lambda: scratch.zero_()) if scratch is not None else None

        # Pre-warmup: discard one (possibly compiling or lazily-initialized) first
        # call per callable before estimating, warming, or timing anything.
        for nm in names:
            fns[nm]()
        tm.synchronize()

        # A per-call time estimate is needed to resolve inner="auto" and to turn
        # min_warmup_time / min_iters_time into per-callable counts.
        need_est = inner == "auto" or min_warmup_time is not None or min_iters_time is not None
        est_by = {nm: _estimate_per_call(tm, fns[nm]) for nm in names} if need_est else {}

        if inner == "auto":
            inner_by = {nm: _inner_for_window(est_by[nm], target_window_s) for nm in names}
        else:
            inner_by = {nm: max(1, int(inner)) for nm in names}

        def _counts(min_time: float | None, floor: int) -> dict[str, int]:
            """Per-callable count: a plain floor, or enough to fill *min_time*."""
            if min_time is None:
                return {nm: floor for nm in names}
            counts: dict[str, int] = {}
            for nm in names:
                sample_s = est_by[nm] * inner_by[nm]
                n = math.ceil(min_time / sample_s) if sample_s > 0 else floor
                counts[nm] = max(floor, n)
            return counts

        warmup_by = _counts(min_warmup_time, warmup)
        iters_by = _counts(min_iters_time, iters)

        # Warmup every callable (per-callable count) before any timing.
        for nm in names:
            for _ in range(warmup_by[nm]):
                fns[nm]()
        tm.synchronize()

        handles: list[tuple[str, object]] = []
        if interleave:
            # Dropout round-robin: each round samples every still-active
            # callable (shuffled); a callable retires once it meets its budget.
            budget = dict(iters_by)
            active = [nm for nm in names if budget[nm] > 0]
            rng = random.Random(0)
            while active:
                order = active[:]
                if len(order) > 1:
                    rng.shuffle(order)
                for nm in order:
                    handles.append((nm, tm.open(fns[nm], inner_by[nm], before)))
                    budget[nm] -= 1
                active = [nm for nm in active if budget[nm] > 0]
        else:
            for nm in names:
                for _ in range(iters_by[nm]):
                    handles.append((nm, tm.open(fns[nm], inner_by[nm], before)))

        tm.synchronize()

        buckets: dict[str, list[float]] = {nm: [] for nm in names}
        for nm, handle in handles:
            buckets[nm].append(tm.value(handle))

        return {
            nm: Measurement(
                samples=np.asarray(buckets[nm], dtype=np.float64),
                name=nm,
                inner=inner_by[nm],
                flops=flops.get(nm),
                bytes=bytes.get(nm),
            )
            for nm in names
        }


# -- pretty table ---------------------------------------------------------


def format_table(measurements: Iterable[Measurement]) -> str:
    """Return an aligned table of measurements, sorted by name."""
    ms = sorted(measurements, key=lambda m: m.name)
    if not ms:
        return "(no measurements)"

    show_tflops = any(m.tflops is not None for m in ms)
    show_gbps = any(m.gbps is not None for m in ms)
    w = max(max(len(m.name) for m in ms), len("benchmark"))

    header = (
        f"{'benchmark':<{w}}  {'median':>10}  {'mean':>10}  "
        f"{'stdev':>10}  {'min':>10}  {'n':>5}"
    )
    if show_tflops:
        header += f"  {'TFLOP/s':>9}"
    if show_gbps:
        header += f"  {'GB/s':>9}"
    lines = [header, "-" * len(header)]

    for m in ms:
        row = (
            f"{m.name:<{w}}  {m.median * 1e3:>8.3f}ms  {m.mean * 1e3:>8.3f}ms  "
            f"{m.std * 1e3:>8.3f}ms  {m.min * 1e3:>8.3f}ms  {m.n:>5}"
        )
        if show_tflops:
            row += f"  {m.tflops:>9.2f}" if m.tflops is not None else f"  {'':>9}"
        if show_gbps:
            row += f"  {m.gbps:>9.1f}" if m.gbps is not None else f"  {'':>9}"
        lines.append(row)
    return "\n".join(lines)
