# usv - Unladen Swallow Velocity

GPU micro-benchmarking, in the spirit of
[`asv`](https://github.com/airspeed-velocity/asv) /
[`torch.utils.benchmark`](https://docs.pytorch.org/docs/stable/benchmark_utils.html) /
[`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html).

Optional accuracy features:

- **pytest runner** - benchmarks are ordinary pytest tests (or asv-style classes),
  selected with `-k` / `-m`, with result tables, saved baselines and regression
  gating. Before the first benchmark it pins the process to GPU-local CPU cores at
  the performance governor and checks the GPU for other processes.
- **Interleaved scheduling** - time a *group* of callables together and collect
  samples round-robin, so time-correlated noise (thermal ramp, DVFS,
  neighboring jobs) spreads across every benchmark instead of biasing one
  contiguous block; or, in the pytest runner, measure every benchmark in several
  rounds spread over the session.
- **Cache flushing** - zero an L2-sized scratch buffer before each sample to
  measure cold-cache kernel cost.
- **CUDA/HIP graphs** - capture the callable into a graph and time replays,
  removing per-launch CPU overhead for very short kernels (opt-in,
  `cudagraph=True`).
- **Interference detection** - check the vendor SMI for *other* processes on the
  GPU before trusting a number (opt-in, `check_interference=True`).
- **Clock/power/temp monitoring** - sample `rocm-smi` during timing and warn if
  the GPU clock drifted, since locking clocks alone isn't enough (opt-in,
  `monitor=True`, AMD only).
- **Rotating buffers** - cycle inputs through a ring so back-to-back launches see
  different memory, reducing cache-residency bias without a full flush;
  `rotating_buffers` auto-sizes the ring to exceed L2.
- **Event-based GPU timing** - per-call `cuda.Event` timing, with a CPU wall-clock fallback for development off-GPU.
- **Backward timing** - time autograd backward passes in blocks of back-to-back
  backwards, instead of deriving them as `(fwd+bwd) - fwd`.
- **JAX timing** - an opt-in `timer="jax"` backend that synchronizes with
  `jax.block_until_ready` for benchmarking JAX callables.

## Install

```bash
pip install -e .           # numpy, torch, pytest
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

## pytest runner

usv installs a pytest plugin. A benchmark is a test that uses the `usv_benchmark`
fixture (a wrapper over `do_bench`):

```python
@pytest.mark.parametrize("n", [1024, 4096])
def test_matmul(usv_benchmark, n):
    a = torch.randn(n, n, device="cuda")
    usv_benchmark(lambda: a @ a, flops=2 * n**3)
```

```bash
pytest bench/ -k 4096                     # select with -k / -m like any test
pytest --benchmark-only --benchmark-save=main
pytest --benchmark-only --benchmark-compare=main --benchmark-compare-fail=median:5%
```

The fixture also has `.backward(fwd, bwd)` (see [Backward timing](#backward-timing)),
`.fwd_bwd(fwd, fwd_bwd)` (backward derived as `(fwd+bwd) - fwd`) and
`.many({name: fn})` (interleaved group). Results are printed as a table at the
end; `--benchmark-save` writes the CSV plus a `-info.txt` with the host, GPU,
software versions and CPU pinning.

Before the first benchmark runs, the runner prepares the session:

- **GPU interference check** - lists other processes on the benchmark GPU, from a
  snapshot taken before this process touches the GPU (see
  [Interference detection](#interference-detection)).
- **CPU pinning** - see [CPU pinning](#cpu-pinning); restored at the end.

| Option | Meaning |
| --- | --- |
| `--benchmark-only` | Run only benchmarks. |
| `--benchmark-timer` | `auto` \| `torch` \| `jax` \| `wall`. |
| `--benchmark-save [LABEL]` | Write results to `results/<commit>[-LABEL].csv` (+ `-info.txt`). |
| `--benchmark-compare SELECTOR` | Compare with a saved baseline (label, CSV path or `last`). |
| `--benchmark-compare-fail METRIC:PCT` | Fail the session on a regression (repeatable). |
| `--benchmark-interleave R` | Measure every benchmark in R rounds spread over the session. |
| `--benchmark-no-cpu-pin` | Don't pin CPUs or change the governor. |
| `--benchmark-gpu-interference` | `warn` (default), `abort` or `off`. |
| `--benchmark-asv` | Also collect asv-style classes (see [asv-style benchmark classes](#asv-style-benchmark-classes)). |
| `--benchmark-fresh-process` | Run each asv benchmark in its own process. |

`--benchmark-interleave R` runs round 1 of all selected benchmarks, then round 2,
and so on (shuffled within a round, with a fixed seed), each round with 1/R of the
samples, and reports the statistics of all rounds' samples. A transient slowdown
then hits a few samples of many benchmarks instead of one benchmark's whole
measurement. Setup and warmup repeat every round, so runs take longer.

## CPU pinning

Launch-bound benchmarks are timed largely on the host. With the default
`schedutil` governor, work split across threads on different cores (e.g.
PyTorch's autograd thread) can leave every core partly idle and clocked low, so
launch-bound timings can flip between runs by up to 2x. `pin_cpus` (used by the
pytest runner) therefore:

- pins the main thread to 4 physical cores on the GPU's NUMA node, offset by the
  GPU index so runs on other GPUs use other cores; threads created later inherit
  them;
- moves the GPU runtime's helper threads to those cores' SMT siblings (multi-stream
  work ran ~1.6x slower when one of them landed on another CCD or socket);
- sets the cores to the `performance` governor (needs root; otherwise only the
  pinning applies) and restores the previous governors afterwards, via a watchdog
  process if the benchmark is killed.

It stays within the CPUs the process may use (`taskset`, cgroup cpusets).

```python
from usv import do_bench, pinned_cpus

with pinned_cpus() as pin:      # initializes the GPU runtime via torch
    print(pin.description)
    m = do_bench(fn)
```

In a container, the governor can only be changed with `--cap-add SYS_ADMIN` (or
`--privileged`); otherwise `/sys` is read-only.

## Writing benchmarks

A benchmark is just a callable of no arguments that launches the work to time.
Close over any state (tensors, streams) you set up beforehand:

```python
def make_matmul(N, nbuf=8):
    bufs = [torch.randn(N, N, device="cuda") for _ in range(nbuf)]
    nxt = rotating(bufs)
    return lambda: nxt() @ nxt()
```

To defeat L2 residency the way the
[CUTLASS methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
recommends, the ring should span at least 2x the L2 cache. `rotating_buffers`
sizes it for you from a buffer factory (inferring the per-buffer bytes and the
device L2 size), so a buffer is evicted before it is reused:

```python
from usv import rotating_buffers

nxt = rotating_buffers(lambda: torch.randn(N, N, device="cuda"))   # >= 2x L2
m = do_bench(lambda: nxt() @ nxt())
```

`rotation_count(bytes_per_buffer, l2_mult=2.0)` exposes just the sizing math if
you build the ring yourself.

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
| `cooldown_s` | `0.0` | Idle sleep (s) after timing, to cool the GPU between benchmarks. |
| `monitor` | `False` | Sample `rocm-smi` during timing; attach summary, warn on clock drift (AMD). |
| `timeout` | `None` | Abort with `TimeoutError` if timing exceeds this many seconds (GPU hang guard). |
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

## Backward timing

`do_bench_backward(fwd, bwd)` times the backward pass of `fwd`. `bwd` takes one
result of `fwd`, runs its backward and clears the gradients it produced. Each block
runs up to 20 forwards untimed (fewer if their saved state would exceed 16 GB or
half the free GPU memory),
then times their backwards back to back as one sample, so the host runs ahead of
the GPU as in forward timing. The first block is warmup; timing stops after at
least 15 blocks and 0.2 s.

```python
from usv import do_bench_backward

def bwd(out):
    out.sum().backward()
    x.grad = None
    lin.weight.grad = lin.bias.grad = None

m = do_bench_backward(lambda: lin(x), bwd, flops=4 * n**3)
```

Deriving backward as `(fwd+bwd) - fwd` gives a stable median but occasional large
outliers in single runs, and repeating one backward with `retain_graph` fails for
modules that free saved state in backward.

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
Monitoring tools (`nvtop`, `amd-smi`, `rocm-smi`) are ignored. Once torch has
initialized the GPU, the device is matched by PCI address, since the SMI tools'
indices ignore `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`.

Inside a container the SMI tools report host PIDs, so this process can't be told
apart from a neighbor once it has used the GPU. Take a snapshot first:

```python
from usv import foreign_processes, snapshot_gpu_processes
from usv.sysinfo import gpu_pci

snap = snapshot_gpu_processes()          # SMI tools only: doesn't touch the GPU
others = foreign_processes(snap, gpu_pci())   # gpu_pci() initializes the GPU
```

The pytest runner does this before the first benchmark.

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

## asv-style benchmark classes

If you already structure benchmarks the [`asv`](https://github.com/airspeed-velocity/asv)
way - a class with `time_*` methods, `setup` / `teardown`, `setup_cache`, and
`params` / `param_names` - the pytest runner collects them with `--benchmark-asv`:

```python
# bench_matmul.py
class MatmulSuite:
    params = [512, 1024, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.a = torch.randn(n, n, device="cuda")

    def time_matmul(self, n):
        return self.a @ self.a
```

```bash
pytest --benchmark-asv bench_matmul.py -k 1024   # item bench_matmul.py::MatmulSuite::time_matmul[1024]
```

Classes are collected from test files and from files matching the
`usv_asv_files` ini option (default `bench*.py`). Each benchmark gets a fresh
instance; `setup_cache` runs once per class and its result is passed as the first
argument to `setup` / `teardown` and each benchmark; a `setup` that raises
`NotImplementedError` skips the benchmark (as in asv). A flat `params` list is one
axis; a list of lists is a Cartesian product. Results are named
`MatmulSuite.time_matmul(n=1024)`.

asv's timing attributes, on the method or the class, map to `do_bench`:

| asv attribute | usv |
| --- | --- |
| `number` | `inner` (`0`: `inner="auto"`) |
| `sample_time` | `inner="auto"`, `target_window_s` |
| `repeat` | `iters` (`(min, max, max_time)`: `iters=min`) |
| `warmup_time` | `min_warmup_time` (`0`: no warmup) |
| `timeout` | `timeout` |

`usv.run_benchmarks` runs them programmatically, through the same runner:

```python
from usv import run_benchmarks

results = run_benchmarks(MatmulSuite, iters=50)   # {"MatmulSuite.time_matmul(n=512)": Measurement, ...}
results = run_benchmarks("mypkg.bench:MatmulSuite", interleave=3, cpu_pin=True)
```

It accepts a class, an instance, a module (all of its benchmark classes), or a
`"module:ClassName"` string. Extra keywords (`iters`, `warmup`, `timer`,
`cache_flush`, ...) go to `do_bench` and take precedence over the asv attributes.
`interleave`, `cpu_pin` (default off), `gpu_interference` (default `"off"`) and
`pytest_args` control the session. A pre-built instance, or a class pytest can't
collect (defined in `__main__` or a local scope), runs directly instead, sharing
one instance per class.

### Fresh-process isolation

Passing `fresh_process=True` (or `--benchmark-fresh-process` to pytest) runs
*each* benchmark in its own freshly spawned Python process - asv's default
methodology. This prevents per-benchmark warmup state, allocator fragmentation,
and module-level caches from leaking between benchmarks:

```python
results = run_benchmarks("mypkg.benchmarks:MatmulSuite", fresh_process=True, iters=50)
```

It requires an *importable* target (a `"module:ClassName"` string, or a class
defined at module scope - not in `__main__` or a local scope) and
JSON-serializable `do_bench` keywords (e.g. `timer="wall"`, not a `GPUTimer`
instance). Off by default. With CPU pinning, the parent pins the CPUs (inherited
by the children) without touching the GPU, and each child moves its own GPU
runtime helper threads.

## Clock / power / temperature monitoring

The [CUTLASS methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
points out that *locking* clocks isn't enough - power and thermal controllers
can still move the frequency, so the settled clock should be monitored during
the profiling iterations. With `monitor=True`, usv samples `rocm-smi` in a
background thread while timing, attaches a summary to each `Measurement`, and
warns if the GPU clock drifted (AMD only):

```python
from usv import do_bench

m = do_bench(lambda: x @ x, monitor=True)   # warns if sclk drifts > 5%
print(m.monitor)   # {'n': 20, 'sclk_mhz': {'min':.., 'mean':.., 'max':.., 'std':..}, 'power_w': {..}, 'temp_c': {..}}
```

Or sample it yourself:

```python
from usv import GpuMonitor, sample_rocm_smi

print(sample_rocm_smi(device=0))   # one reading, or None
with GpuMonitor(device=0) as mon:
    ...
print(mon.summary())
```

Best-effort and read-only: if `rocm-smi` is missing or unparsable, sampling
yields nothing rather than failing. Off by default; AMD / `rocm-smi` only for now.

## Sources and References

- [`asv` (airspeed velocity)](https://github.com/airspeed-velocity/asv)
- [`torch.utils.benchmark`](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)
- [CUTLASS GEMM performance measurement methodology](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html)
- [hipBLASLt standard benchmarking](https://amd.atlassian.net/wiki/spaces/MLSE/pages/744192648/hipBLASLt+Standard+Benchmarking)
- [HipKittens benchmarking](https://github.com/HazyResearch/HipKittens/tree/main/analysis/baselines)

## License

MIT
