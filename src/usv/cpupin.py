"""Pin the benchmark process to a few GPU-local CPU cores at the performance governor.

Launch-bound GPU benchmarks are timed largely on the host: how fast the CPU issues
kernel launches.  With the default ``schedutil`` governor, a process whose work is
split across threads on different cores (e.g. PyTorch's autograd thread) can leave
every core partly idle and clocked low, so its timings flip between runs by up to 2x.
:func:`pin_cpus` therefore:

* pins the main thread to *ncpus* physical cores on the GPU's NUMA node (threads it
  creates later inherit them), offset by the GPU index so runs on other GPUs of the
  node use different cores;
* moves the GPU runtime's helper threads (HIP/ROCr, one of which polls events) to
  those cores' SMT siblings - on another CCD or socket they slow multi-stream work;
* sets those cores to the ``performance`` cpufreq governor (needs root; without it
  only the pinning is applied) and restores the previous governors afterwards, via a
  small watchdog process if the benchmark is killed or crashes.

Everything respects the CPUs the process may use (``taskset``, cgroup cpusets).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

__all__ = [
    "CpuPin",
    "pin_cpus",
    "pinned_cpus",
    "cpu_group",
    "gpu_local_cores",
    "move_helper_threads",
]

DEFAULT_NCPUS = 4
_CPU_SYSFS = Path("/sys/devices/system/cpu")
_NODE_SYSFS = Path("/sys/devices/system/node")
_PCI_SYSFS = Path("/sys/bus/pci/devices")

# Runs as a separate process: EOF on stdin without a prior message means the benchmark
# process died without calling restore(), so put the governors ("path=old" args) back.
# Reads one byte, not up to EOF: forked workers may hold the pipe open after restore().
_GOVERNOR_GUARD = r"""
import sys
if not sys.stdin.buffer.read(1):
    for arg in sys.argv[1:]:
        path, _, old = arg.rpartition("=")
        try:
            open(path, "w").write(old)
        except OSError:
            pass
"""


def parse_cpulist(text: str) -> set[int]:
    """Parse a sysfs CPU list such as ``"0-3,8,10-11"``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _siblings(cpu: int) -> set[int]:
    try:
        return parse_cpulist((_CPU_SYSFS / f"cpu{cpu}/topology/thread_siblings_list").read_text())
    except (OSError, ValueError):
        return {cpu}


def visible_gpu_index() -> int:
    """Physical index of the first visible GPU (``HIP/ROCR/CUDA_VISIBLE_DEVICES``), else 0."""
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        first = os.environ.get(var, "").split(",")[0].strip()
        if first:
            try:
                return int(first)
            except ValueError:
                return 0
    return 0


def gpu_local_cores(pci: str | None = None, allowed: set[int] | None = None) -> list[int]:
    """Physical cores (one CPU per SMT group) we may use, on the NUMA node of GPU *pci*.

    Each core is represented by its lowest *allowed* SMT sibling.  Without *pci*, or if
    its NUMA node is unknown, all allowed CPUs are considered.
    """
    allowed = set(os.sched_getaffinity(0)) if allowed is None else set(allowed)
    cpus = allowed
    if pci:
        try:
            node = int((_PCI_SYSFS / pci / "numa_node").read_text())
            if node >= 0:
                node_cpus = parse_cpulist((_NODE_SYSFS / f"node{node}/cpulist").read_text())
                cpus = (node_cpus & allowed) or allowed
        except (OSError, ValueError):
            pass
    return [cpu for cpu in sorted(cpus) if cpu == min(_siblings(cpu) & cpus)]


def move_helper_threads(helpers) -> None:
    """Move every thread of this process except the calling one to the *helpers* CPUs."""
    me = threading.get_native_id()
    for tid in os.listdir("/proc/self/task"):
        if int(tid) != me:
            try:
                os.sched_setaffinity(int(tid), helpers)
            except OSError:
                pass


@dataclass
class CpuPin:
    """Result of :func:`pin_cpus`; call :meth:`restore` (or use :func:`pinned_cpus`)."""

    cpus: list[int]
    helpers: list[int]
    governor: str
    description: str
    _restore: Callable[[], None] = field(repr=False, default=lambda: None)

    def restore(self) -> None:
        """Put back the previous governors and thread affinities (idempotent)."""
        self._restore()
        self._restore = lambda: None


def _set_governor(cpus) -> tuple[dict[Path, str], OSError | None]:
    previous: dict[Path, str] = {}
    error = None
    for cpu in cpus:
        # CPUs that share a cpufreq policy link to the same governor file.
        path = (_CPU_SYSFS / f"cpu{cpu}/cpufreq/scaling_governor").resolve()
        if path in previous:
            continue
        try:
            old = path.read_text().strip()
            path.write_text("performance")
            previous[path] = old
        except OSError as e:
            error = e
    return previous, error


def cpu_group(cores: list[int], ncpus: int, gpu: int) -> list[int]:
    """The *ncpus* cores for GPU *gpu*: whole groups, so GPUs get disjoint cores while
    there are enough.  Group 0 (often busy with OS work) is used last."""
    groups = len(cores) // ncpus
    if groups < 1:
        return list(cores)
    start = ncpus * ((gpu + 1) % groups)
    return cores[start : start + ncpus]


def pin_cpus(
    ncpus: int = DEFAULT_NCPUS,
    *,
    pci: str | None = None,
    gpu_index: int | None = None,
    governor: bool = True,
    move_helpers: bool = True,
) -> CpuPin:
    """Pin the calling thread to *ncpus* GPU-local physical cores; see the module docstring.

    *pci* is the GPU's PCI address (``"0000:75:00.0"``); by default it is read from
    torch's ``cuda:0``, which initializes the GPU runtime.  Initialize it before
    calling (as the default does) so the runtime's helper threads exist and can be
    moved; ``move_helpers=False`` leaves them alone (e.g. in a parent process whose
    children move their own).  *gpu_index* (default: the first visible GPU) selects a
    different group of cores per GPU, so concurrent runs on other GPUs don't share them.
    """
    if pci is None:
        from usv.sysinfo import gpu_pci

        pci = gpu_pci()
    allowed = set(os.sched_getaffinity(0))
    cores = gpu_local_cores(pci, allowed)
    gpu = visible_gpu_index() if gpu_index is None else gpu_index
    cpus = cpu_group(cores, ncpus, gpu)
    siblings: set[int] = set()
    for cpu in cpus:
        siblings |= _siblings(cpu)
    # Without allowed siblings the helpers share the main thread's cores.
    helpers = sorted((siblings & allowed) - set(cpus)) or list(cpus)

    me = threading.get_native_id()
    old_affinity: dict[int, set[int]] = {}
    for tid in map(int, os.listdir("/proc/self/task")):
        if tid == me or move_helpers:
            try:
                old_affinity[tid] = os.sched_getaffinity(tid)
            except OSError:
                pass
    if move_helpers:
        move_helper_threads(helpers)
    os.sched_setaffinity(0, cpus)

    previous, error = _set_governor(cpus) if governor else ({}, None)
    guard = None

    def restore() -> None:
        for path, old in previous.items():
            try:
                path.write_text(old)
            except OSError:
                pass
        if guard is not None:
            guard.communicate(b"restored")
        for tid, mask in old_affinity.items():
            try:
                os.sched_setaffinity(tid, mask)
            except OSError:
                pass
        # Threads created while pinned inherited a pinned mask; leave other masks alone.
        pinned_masks = (set(cpus), set(helpers) if move_helpers else set(cpus))
        for tid in map(int, os.listdir("/proc/self/task")):
            try:
                if tid not in old_affinity and os.sched_getaffinity(tid) in pinned_masks:
                    os.sched_setaffinity(tid, old_affinity[me])
            except (OSError, KeyError):
                pass

    if previous:
        try:
            # Own session, so a terminal Ctrl-C doesn't kill the guard along with the benchmark.
            guard = subprocess.Popen(
                [sys.executable, "-c", _GOVERNOR_GUARD, *(f"{p}={old}" for p, old in previous.items())],
                stdin=subprocess.PIPE,
                start_new_session=True,
            )
        except BaseException:
            restore()
            raise

    if not governor:
        gov = "governor unchanged"
    elif error is None:
        gov = "performance governor"
    else:
        gov = f"governor unchanged: {error}"
    desc = (
        f"main thread pinned to cpus {','.join(map(str, cpus))}, helper threads "
        f"{'to ' + ','.join(map(str, helpers)) if move_helpers else 'not moved'} ({gov})"
    )
    return CpuPin(list(cpus), helpers, gov, desc, restore)


@contextmanager
def pinned_cpus(ncpus: int = DEFAULT_NCPUS, **kwargs) -> Iterator[CpuPin]:
    """Context manager around :func:`pin_cpus` that restores everything on exit::

    with pinned_cpus() as pin:
        print(pin.description)
        m = do_bench(fn)
    """
    pin = pin_cpus(ncpus, **kwargs)
    try:
        yield pin
    finally:
        pin.restore()
