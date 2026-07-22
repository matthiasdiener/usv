"""Opt-in AMD GPU clock locking for reproducible absolute timings.

DVFS / boost / thermal throttling make the GPU's clock the dominant source of
run-to-run variance.  Locking it to a fixed frequency removes that, at the cost
of needing elevated privileges (the vendor SMI tools usually require ``sudo``)
and mutating *global* device state.  This is therefore never done automatically
- wrap a benchmark in :func:`fixed_clocks` explicitly.  Clocks are restored on
exit.

* AMD: forces the maximum via performance level HIGH
  (``amd-smi set --perf-level HIGH``).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["fixed_clocks", "gpu_vendor"]


def gpu_vendor() -> str | None:
    """Return ``"nvidia"``, ``"amd"``, or ``None`` from torch / SMI tooling."""
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return "amd"
        if getattr(torch.version, "cuda", None):
            return "nvidia"
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("amd-smi") or shutil.which("rocm-smi"):
        return "amd"
    return None


def _run(cmd: list[str]) -> str:
    """Run *cmd*, returning stdout; raise a clear RuntimeError on failure."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found on PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (clock control usually needs sudo):\n"
            f"{(e.output or '').strip()}"
        ) from e


def _run_ok(cmd: list[str]) -> None:
    """Best-effort command whose failure must not mask a primary error."""
    try:
        _run(cmd)
    except Exception:
        pass


# AMD


@contextmanager
def _amd_fixed(device: int) -> Iterator[None]:
    # Force clocks to maximum with performance level HIGH ("forces clocks to
    # maximum regardless of workload").  For the full set of AMD frequency
    # and power controls (perf levels, clock/power caps, determinism), see
    # https://rocm.docs.amd.com/projects/amdsmi/en/develop/conceptual/perf-determinism.html
    amd = shutil.which("amd-smi")
    rocm = None if amd else shutil.which("rocm-smi")
    if amd:
        set_cmd = [amd, "set", "--gpu", str(device), "--perf-level", "HIGH"]
        reset_cmd = [amd, "set", "--gpu", str(device), "--perf-level", "AUTO"]
    elif rocm:
        set_cmd = [rocm, "-d", str(device), "--setperflevel", "high"]
        reset_cmd = [rocm, "-d", str(device), "--setperflevel", "auto"]
    else:
        raise RuntimeError("neither amd-smi nor rocm-smi found on PATH")

    _run(set_cmd)
    try:
        yield
    finally:
        _run_ok(reset_cmd)


# public API


@contextmanager
def fixed_clocks(*, device: int = 0, vendor: str = "auto") -> Iterator[None]:
    """Pin supported GPU clocks for the duration of the block.

    Removes DVFS / boost variance for reproducible absolute timings on AMD GPUs by
    forcing performance level HIGH.  Use *device* to pick a GPU and *vendor* to
    override auto-detection.

    Needs privileges - the vendor SMI tools usually require ``sudo`` - and
    changes global GPU state, so use it only around benchmarking.  Clocks are
    restored on exit.

    ::

        from usv import do_bench, fixed_clocks

        with fixed_clocks():
            m = do_bench(lambda: x @ x)
    """
    v = vendor if vendor != "auto" else gpu_vendor()
    if v == "amd":
        with _amd_fixed(device):
            yield
    elif v == "nvidia":
        raise RuntimeError("fixed_clocks: NVIDIA clock control is not supported")
    else:
        raise RuntimeError(
            "fixed_clocks: could not detect an AMD GPU / SMI tool "
            "(pass vendor='amd' explicitly)"
        )
