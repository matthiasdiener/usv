"""GPU timer abstraction.

Provides event-based GPU timing that works with both CUDA (NVIDIA) and
HIP/ROCm (AMD).  The primary backend uses ``torch.cuda.Event`` which
works on both vendors when PyTorch is installed.  A ``time.perf_counter``
wall-clock backend is available for development off-GPU.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["GPUTimer", "get_timer"]


class GPUTimer(ABC):
    """Abstract GPU timer.

    The low-level protocol is :meth:`open` / :meth:`value`.  ``open`` enqueues
    one timed *sample* - an optional untimed ``before`` callback, then *inner*
    back-to-back calls of ``fn`` bracketed by timing - and returns an opaque
    handle; ``value`` returns the per-call seconds for that handle after a
    :meth:`synchronize`.  Separating *enqueue* from *read* is what makes
    interleaving possible: samples from many callables can be enqueued on the
    stream in round-robin order and resolved with a single synchronize.
    """

    @abstractmethod
    def synchronize(self) -> None: ...

    @abstractmethod
    def open(
        self,
        fn: Callable[[], object],
        inner: int = 1,
        before: Callable[[], object] | None = None,
    ) -> Any:
        """Enqueue one timed sample; return an opaque handle for :meth:`value`."""
        ...

    @abstractmethod
    def value(self, handle: Any) -> float:
        """Per-call seconds for *handle* (call :meth:`synchronize` first)."""
        ...

    def record_n(
        self,
        fn: Callable[[], object],
        n: int,
        *,
        inner: int = 1,
        before: Callable[[], object] | None = None,
    ) -> np.ndarray:
        """Convenience: time *n* contiguous samples of *fn* (no interleaving)."""
        handles = [self.open(fn, inner, before) for _ in range(n)]
        self.synchronize()
        return np.array([self.value(h) for h in handles], dtype=np.float64)


# torch backend


class _TorchTimer(GPUTimer):
    """Per-iteration CUDA-event timing via ``torch.cuda.Event``."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        # Pre-warm the CUDA context.
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA/ROCm device found")
        torch.cuda.synchronize()

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()

    def open(self, fn, inner=1, before=None):
        torch = self._torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if before is not None:
            before()
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        return (start, end, inner)

    def value(self, handle) -> float:
        start, end, inner = handle
        # elapsed_time is milliseconds -> seconds, per inner call.
        return start.elapsed_time(end) / 1000.0 / inner


# wall-clock fallback (CPU-only or unsupported GPU)


class _WallClockTimer(GPUTimer):
    """Fallback: ``time.perf_counter`` with no GPU synchronization.

    Timing is synchronous, so :meth:`open` measures immediately and
    :meth:`value` just returns the stored result.  Interleaving still works
    (round-robin CPU timing); it simply cannot overlap with the device.
    """

    def synchronize(self) -> None:
        pass

    def open(self, fn, inner=1, before=None):
        if before is not None:
            before()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        t1 = time.perf_counter()
        return (t1 - t0) / inner

    def value(self, handle) -> float:
        return handle


# factory


def get_timer(backend: str = "auto") -> GPUTimer:
    """Return a :class:`GPUTimer` for the requested *backend*.

    Parameters
    ----------
    backend
        ``"torch"`` - use ``torch.cuda.Event`` (works on NVIDIA & AMD).
        ``"wall"``  - plain ``time.perf_counter`` (no GPU sync).
        ``"auto"``  - try ``torch``, then fall back to ``wall``.
    """
    if backend == "torch":
        return _TorchTimer()
    if backend == "wall":
        return _WallClockTimer()
    if backend != "auto":
        raise ValueError(f"Unknown timer backend: {backend!r}")

    # auto: prefer torch events, then wall-clock.
    for factory in (_TorchTimer, _WallClockTimer):
        try:
            return factory()
        except Exception:
            continue
    return _WallClockTimer()
