"""Opt-in CUDA/HIP graph capture for launch-overhead-free timing.

For very short kernels the Python-side launch cost can dominate the measured
time.  Capturing the work into a CUDA graph (HIP graph on ROCm) and timing
*replays* removes per-launch CPU overhead, so the measurement reflects device
execution.  This mirrors ``triton.testing.do_bench_cudagraphs``.

It is never used automatically - pass ``cudagraph=True`` to :func:`usv.do_bench`
/ :func:`usv.do_bench_many`, or wrap a callable with :func:`graph_replay`
directly.  The callable must be graph-capturable: static shapes, no host
synchronization, and it must write into pre-allocated buffers (a fresh
allocation per call cannot be captured).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["graph_replay", "ensure_available"]


def _torch_cuda():
    """Return the torch module, requiring a working CUDA/ROCm device."""
    try:
        import torch
    except Exception as e:  # pragma: no cover - torch missing
        raise RuntimeError("cudagraph mode requires torch") from e
    if not torch.cuda.is_available():
        raise RuntimeError("cudagraph mode requires a CUDA/ROCm device")
    return torch


def ensure_available() -> None:
    """Raise ``RuntimeError`` unless CUDA-graph capture is possible.

    Used to fail fast before any of the user's benchmark runs.
    """
    _torch_cuda()


def graph_replay(fn: "Callable[[], object]", *, warmup: int = 3) -> "Callable[[], None]":
    """Capture *fn* into a CUDA/HIP graph and return a callable that replays it.

    *warmup* untimed calls run on a side stream first (so any lazy init / autotune
    settles) before the graph is captured on the current stream, following the
    standard PyTorch capture pattern.  The returned callable takes no arguments
    and replays the captured work; timing it measures replay (device) cost with
    no Python launch overhead.

    ::

        from usv import do_bench
        from usv.cudagraph import graph_replay

        step = graph_replay(lambda: torch.mm(a, b, out=c))
        m = do_bench(step)          # or: do_bench(orig_fn, cudagraph=True)
    """
    torch = _torch_cuda()

    # Warm up on a side stream so capture sees a settled state, then rejoin.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    # The bound method keeps `graph` alive for as long as the callable exists.
    return graph.replay
