"""Opt-in GPU clock locking for reproducible absolute timings.

DVFS / boost / thermal throttling make the GPU's clock the dominant source of
run-to-run variance.  Locking it to a fixed frequency removes that, at the cost
of needing elevated privileges (the vendor SMI tools usually require ``sudo``)
and mutating *global* device state.  This is therefore never done automatically
- wrap a benchmark in :func:`fixed_clocks` explicitly.  Clocks are restored on
exit.

* NVIDIA: locks SM + memory clocks to the device max
  (``nvidia-smi --lock-gpu-clocks`` / ``--lock-memory-clocks``).
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


# -- NVIDIA ---------------------------------------------------------------


def _nvidia_max_clocks(device: int) -> tuple[int, int]:
    out = _run(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-gpu=clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    sm, mem = (int(x) for x in out.split(","))
    return sm, mem


def _nvidia_persistence(device: int) -> str | None:
    try:
        return _run(
            [
                "nvidia-smi",
                "-i",
                str(device),
                "--query-gpu=persistence_mode",
                "--format=csv,noheader",
            ]
        ).strip()
    except Exception:
        return None


@contextmanager
def _nvidia_fixed(device: int) -> Iterator[None]:
    sm, mem = _nvidia_max_clocks(device)
    prev_pm = _nvidia_persistence(device)

    _run_ok(["nvidia-smi", "-i", str(device), "-pm", "1"])
    try:
        _run(["nvidia-smi", "-i", str(device), f"--lock-gpu-clocks={sm},{sm}"])
        _run(["nvidia-smi", "-i", str(device), f"--lock-memory-clocks={mem},{mem}"])
        yield
    finally:
        _run_ok(["nvidia-smi", "-i", str(device), "-rgc"])
        _run_ok(["nvidia-smi", "-i", str(device), "-rmc"])
        if prev_pm == "Disabled":
            _run_ok(["nvidia-smi", "-i", str(device), "-pm", "0"])


# -- AMD ------------------------------------------------------------------


@contextmanager
def _amd_fixed(device: int) -> Iterator[None]:
    # Force clocks to maximum with performance level HIGH ("forces clocks to
    # maximum regardless of workload").
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


# -- public API -----------------------------------------------------------


@contextmanager
def fixed_clocks(*, device: int = 0, vendor: str = "auto") -> Iterator[None]:
    """Lock the GPU clock to the device maximum for the duration of the block.

    Removes DVFS / boost variance for reproducible absolute timings.  NVIDIA
    locks SM + memory clocks to the device max; AMD forces performance level
    HIGH.  Use *device* to pick a GPU and *vendor* to override auto-detection.

    Needs privileges - the vendor SMI tools usually require ``sudo`` - and
    changes global GPU state, so use it only around benchmarking.  Clocks are
    restored on exit.

    ::

        from usv import do_bench, fixed_clocks

        with fixed_clocks():
            m = do_bench(lambda: x @ x)
    """
    v = vendor if vendor != "auto" else gpu_vendor()
    if v == "nvidia":
        with _nvidia_fixed(device):
            yield
    elif v == "amd":
        with _amd_fixed(device):
            yield
    else:
        raise RuntimeError(
            "fixed_clocks: could not detect an NVIDIA or AMD GPU / SMI tool "
            "(pass vendor='nvidia' or 'amd' explicitly)"
        )
