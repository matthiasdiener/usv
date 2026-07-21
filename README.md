# usv - Unladen Swallow Velocity

GPU micro-benchmarking for AMD and NVIDIA, in the spirit of
[`asv`](https://github.com/airspeed-velocity/asv) /
[`torch.utils.benchmark`](https://docs.pytorch.org/docs/stable/benchmark_utils.html) /
[`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html).

Optional accuracy features:

- **Interleaved scheduling** - time a *group* of callables together and collect
  samples round-robin, so time-correlated noise (thermal ramp, DVFS,
  neighboring jobs) spreads across every benchmark instead of biasing one
  contiguous block.
- **Cache flushing** - zero an L2-sized scratch buffer before each sample to
  measure cold-cache kernel cost.
- **CUDA/HIP graphs** - capture the callable into a graph and time replays,
  removing per-launch CPU overhead for very short kernels (opt-in,
  `cudagraph=True`).
- **Interference detection** - check the vendor SMI for *other* processes on the
  GPU before trusting a number (opt-in, `check_interference=True`).
- **Rotating buffers** - cycle inputs through a ring so back-to-back launches see
  different memory, reducing cache-residency bias without a full flush.
- **Event-based GPU timing** - per-call `cuda.Event` timing (works on NVIDIA and
  AMD/ROCm via PyTorch), with a CPU wall-clock fallback for development off-GPU.
- **JAX timing** - an opt-in `timer="jax"` backend that synchronizes with
  `jax.block_until_ready` for benchmarking JAX callables.

## Install

```bash
pip install -e .           # runtime (numpy + torch)
pip install -e ".[dev]"    # + test tooling
```

## Quick start

Time a single kernel:

```python
import torch
from usv import do_bench, rotating

bufs = [torch.randn(1024, 1024, device="cuda") for _ in range(8)]
nxt = rotating(bufs)                       # rotating input buffers

m = do_bench(lambda: nxt() @ nxt(), flops=2 * 1024**3)
print(m)                                   # 17.4 TFLOP/s  (0.1234 ms +/- 0.0021, median+/-std, n=100)
print(m.median, m.tflops)
```

## Interleaving

A single `do_bench(fn)` can't interleave - it only sees one kernel. To
interleave, hand **all** the callables to `do_bench_many` at once; it collects
samples round-robin (one per callable per round, in a reshuffled order), so slow
drift lands on one sample of *each* benchmark rather than a contiguous block of
one:

```python
from usv import do_bench_many, format_table

fns = {f"matmul[N={N}]": make_matmul(N) for N in (512, 1024, 2048)}
results = do_bench_many(fns, iters=100, interleave=True)   # dict[name -> Measurement]
print(format_table(results.values()))
```

Interleaving is opt-in. By default (`interleave=False`) each callable is timed
to completion in turn (equivalent to calling `do_bench` in a loop).

## Writing benchmarks

A benchmark is just a callable of no arguments that launches the work to time.
Close over any state (tensors, streams) you set up beforehand:

```python
def make_matmul(N, nbuf=8):
    bufs = [torch.randn(N, N, device="cuda") for _ in range(nbuf)]
    nxt = rotating(bufs)
    return lambda: nxt() @ nxt()
```

## API

### `do_bench(fn, ...) -> Measurement`

| Argument | Default | Meaning |
| --- | --- | --- |
| `fn` | - | Callable of no args to time. |
| `warmup` | `50` | Untimed calls before timing (a first pre-warmup call is always discarded). |
| `iters` | `100` | Number of timed samples. |
| `inner` | `1` | Calls per timed sample, or `"auto"` to fill `target_window_s`. |
| `target_window_s` | `1e-3` | Target window duration for `inner="auto"`. |
| `min_warmup_time` | `None` | If set (s), warm up until this much kernel time elapses (floor on `warmup`). |
| `min_iters_time` | `None` | If set (s), sample until this much kernel time elapses (floor on `iters`). |
| `cache_flush` | `False` | Zero an L2-sized buffer before each sample. |
| `flush_mb` | `None` | Flush buffer size in MB; `None` sizes it from the device L2 cache. |
| `lock_clocks` | `False` | Pin supported GPU clocks for the run (see `fixed_clocks`). |
| `cudagraph` | `False` | Capture into a CUDA/HIP graph and time replays (no launch overhead). |
| `timer` | `"auto"` | `auto` \| `torch` \| `jax` \| `wall`, or a `GPUTimer`. |
| `name` | `""` | Label for printing. |
| `flops` / `bytes` | `None` | Per-call work -> `TFLOP/s` / `GB/s` columns. |

### `do_bench_many(fns, ...) -> dict[str, Measurement]`

Same knobs as `do_bench`, plus:

| Argument | Default | Meaning |
| --- | --- | --- |
| `fns` | - | `{name: callable}` to time together. |
| `interleave` | `False` | Collect samples round-robin across `fns`. |
| `flops` / `bytes` | `None` | `{name: value}` maps for throughput columns. |

### `Measurement`

Holds the raw per-call `samples` (a numpy array, already divided by `inner`) and
exposes `median`, `mean`, `std`, `min`, `max`, `n`, `quantile(q)`, and - when
`flops`/`bytes` are set - `tflops` / `gbps`.

## Stable clocks

A big source of run-to-run variance in GPU micro-benchmarks is usually not
the kernel - it's the GPU changing its clock (DVFS / boost / thermal throttle)
between runs. Interleaving (above) spreads that noise across benchmarks, but for
*reproducible absolute* numbers you want to pin the clocks to a fixed frequency
before measuring.

- **AMD** - see
  [AMD SMI - performance determinism](https://rocm.docs.amd.com/projects/amdsmi/en/develop/conceptual/perf-determinism.html)

Triton does this in an opt-in helper,
`triton.testing.set_gpu_clock(ref_sm_clock, ref_mem_clock)`, which hard-codes a
reference clock. `usv` provides `usv.fixed_clocks()` - a context manager
that forces performance level HIGH and restores it on exit:

```python
from usv import do_bench, fixed_clocks

with fixed_clocks():   # needs privileges (usually sudo)
    m = do_bench(lambda: x @ x)
```

Because it changes global GPU state and usually needs `sudo`, it is never run
automatically - use it only when you need reproducible absolute numbers.

For a single measurement you can pass it inline instead:
`do_bench(..., lock_clocks=True)` does the same lock around that one call; use
the `fixed_clocks()` context manager to lock once around a loop or sweep.

## CUDA/HIP graphs

For very short kernels the Python-side launch cost can dominate the measured
time. Capturing the work into a CUDA graph (HIP graph on ROCm) and timing
*replays* removes that per-launch overhead, mirroring
`triton.testing.do_bench_cudagraphs`:

```python
from usv import do_bench

m = do_bench(lambda: torch.add(a, a, out=c), cudagraph=True)
```

The callable must be graph-capturable: static shapes, no host synchronization,
and it must write into pre-allocated buffers (a fresh allocation per call cannot
be captured). It is off by default and needs a CUDA/ROCm device. You can also
capture manually with `usv.graph_replay(fn)`.

## Interference detection

Another process sharing the GPU (a leftover training job, a second benchmark, a
compositor) quietly biases every measurement. `usv` can check the vendor SMI
tool for other compute processes on the device and warn before timing:

```python
from usv import do_bench

m = do_bench(lambda: x @ x, check_interference=True)   # warns if the GPU is shared
```

Or query it yourself:

```python
from usv import check_gpu_interference, gpu_processes

print(gpu_processes())               # every compute process on the GPU
others = check_gpu_interference()    # everything except this process
if others:
    raise RuntimeError(f"GPU is busy: {others}")
```

It is best-effort and read-only: if no SMI tool is available (or its output
can't be parsed) it returns an empty list rather than failing. Off by default.

## JAX

JAX callables can be timed with the opt-in `timer="jax"` backend. JAX dispatches
asynchronously and exposes no CUDA events, so it times wall-clock around the
calls and blocks on the result with `jax.block_until_ready` (the usual JAX
micro-benchmark pattern). The callable should *return* its output so there is
something to block on, and should be `jit`-compiled for realistic numbers:

```python
import jax, jax.numpy as jnp
from usv import do_bench

f = jax.jit(lambda: x @ x)
m = do_bench(f, timer="jax")
```

Install the extra with `pip install -e ".[jax]"`. `"auto"` never selects JAX -
it is opt-in via `timer="jax"`.

## License

MIT
