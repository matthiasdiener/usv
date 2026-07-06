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
- **Rotating buffers** - cycle inputs through a ring so back-to-back launches see
  different memory, reducing cache-residency bias without a full flush.
- **Event-based GPU timing** - per-call `cuda.Event` timing (works on NVIDIA and
  AMD/ROCm via PyTorch), with a CPU wall-clock fallback for development off-GPU.

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
print(m)                                   # 0.1234 ms +/- 0.0021 (median+/-std, n=100)  17.4 TFLOP/s
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
| `timer` | `"auto"` | `auto` \| `torch` \| `wall`, or a `GPUTimer`. |
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

## License

MIT
