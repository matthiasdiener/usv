"""Tests for opt-in CUDA/HIP-graph timing.

The guards are exercised without a GPU; the real capture/replay path is
GPU-gated and runs on a CUDA/ROCm device.
"""

from __future__ import annotations

import pytest

import usv.cudagraph as cg
from usv import do_bench


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _raise(*_a, **_k):
    raise RuntimeError("no cuda")


def test_cudagraph_off_by_default():
    # Default path never needs cudagraph; the wall timer works with no GPU.
    m = do_bench(lambda: None, warmup=0, iters=2, timer="wall")
    assert m.n == 2


def test_ensure_available_raises_without_cuda(monkeypatch):
    monkeypatch.setattr(cg, "_torch_cuda", _raise)
    with pytest.raises(RuntimeError):
        cg.ensure_available()


def test_do_bench_cudagraph_fails_fast_without_cuda(monkeypatch):
    # cudagraph=True must raise before running the callable when capture is impossible.
    monkeypatch.setattr(cg, "_torch_cuda", _raise)
    with pytest.raises(RuntimeError):
        do_bench(lambda: None, warmup=0, iters=1, timer="wall", cudagraph=True)


@pytest.mark.skipif(not _has_cuda(), reason="needs a CUDA/ROCm device")
def test_cudagraph_replays_on_gpu():
    import torch

    a = torch.randn(256, 256, device="cuda")
    c = torch.empty_like(a)
    m = do_bench(lambda: torch.add(a, a, out=c), warmup=2, iters=10, cudagraph=True)
    assert m.n == 10 and m.median >= 0.0
