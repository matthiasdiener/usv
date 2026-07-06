#!/usr/bin/env python3
"""Example: functional GPU micro-benchmarks with usv.

Run standalone:
    python examples/bench_matmul.py
"""

import torch

from usv import do_bench, do_bench_many, format_table, rotating

N_BUFS = 8  # rotating buffer ring size


def make_matmul(N):
    """Return a callable that times one NxN matmul with rotating inputs."""
    bufs = [torch.randn(N, N, device="cuda") for _ in range(N_BUFS)]
    nxt = rotating(bufs)
    return lambda: nxt() @ nxt()


def make_add(N):
    """Return a callable timing an element-wise add (memory-bound)."""
    xs = [torch.randn(N, N, device="cuda") for _ in range(N_BUFS)]
    ys = [torch.randn(N, N, device="cuda") for _ in range(N_BUFS)]
    nx, ny = rotating(xs), rotating(ys)
    return lambda: nx() + ny()


if __name__ == "__main__":
    # 1) Single kernel, Triton-style:
    m = do_bench(make_matmul(1024), iters=50, flops=2 * 1024**3, name="matmul[N=1024]")
    print(m)

    # 2) Interleaved sweep across every size/op at once.  Passing all the
    #    callables together is what lets usv round-robin the samples.
    fns = {f"matmul[N={N}]": make_matmul(N) for N in (512, 1024, 2048)}
    fns.update({f"add[N={N}]": make_add(N) for N in (1024, 4096, 8192)})
    flops = {f"matmul[N={N}]": 2 * N**3 for N in (512, 1024, 2048)}
    bytes_ = {f"add[N={N}]": 3 * N * N * 4 for N in (1024, 4096, 8192)}
    results = do_bench_many(
        fns,
        iters=50,
        interleave=True,
        flops=flops,
        bytes=bytes_,
    )
    print(format_table(results.values()))
