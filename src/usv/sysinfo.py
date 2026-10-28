"""Machine and GPU metadata for benchmark runs."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from importlib import metadata

__all__ = ["gpu_pci", "probe_gpu", "run_info", "write_run_info"]

# Prints one JSON line describing cuda:0, in a child process so the caller never
# initializes the GPU runtime itself.
_PROBE = r"""
import json, torch
p = torch.cuda.get_device_properties(0)
print(json.dumps({"pci": "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id),
                  "name": p.name, "arch": getattr(p, "gcnArchName", "") or ""}))
"""


def gpu_pci(index: int = 0) -> str | None:
    """PCI address of torch's ``cuda:index`` (``"0000:75:00.0"``), or ``None``.

    Initializes the GPU runtime.  Honors ``HIP/CUDA_VISIBLE_DEVICES``, so ``cuda:0``
    is the GPU the benchmark uses.
    """
    return _gpu_props(index).get("pci")


def _gpu_props(index: int = 0) -> dict:
    try:
        import torch

        p = torch.cuda.get_device_properties(index)
        return {
            "pci": "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id),
            "name": p.name,
            "arch": getattr(p, "gcnArchName", "") or "",
        }
    except Exception:
        return {}


def probe_gpu(timeout: float = 120.0) -> dict:
    """``{"pci", "name", "arch"}`` of ``cuda:0`` from a short-lived child process.

    For callers that must not initialize the GPU runtime themselves (e.g. a parent
    spawning fresh benchmark processes).  Returns ``{}`` if the probe fails.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return {}


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def run_info(*, gpu: dict | None = None, cpus: str | None = None, **extra: str) -> dict[str, str]:
    """Describe the machine, GPU and software of a run as ``{key: value}`` strings.

    *gpu* is a :func:`probe_gpu`-style dict; by default it is read from torch's
    ``cuda:0`` in this process (initializing the runtime).  *cpus* describes the CPU
    pinning; *extra* entries are appended as given.
    """
    gpu = _gpu_props() if gpu is None else gpu
    torch_hip = ""
    if "torch" in sys.modules:
        torch_hip = getattr(sys.modules["torch"].version, "hip", "") or ""
    info = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": os.uname().nodename.split(".")[0],
        "visible_gpu": (
            os.environ.get("HIP_VISIBLE_DEVICES")
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            or "all"
        ),
        "gpu_name": gpu.get("name", ""),
        "gpu_arch": gpu.get("arch", ""),
        "gpu_pci": gpu.get("pci", ""),
        "rocm": _read("/opt/rocm/.info/version") or torch_hip,
        "amdgpu_drv": _read("/sys/module/amdgpu/version"),
        "kernel": os.uname().release,
        "python": sys.version.split()[0],
        "torch": _version("torch"),
        "numpy": _version("numpy"),
        "usv": _version("usv"),
        "commit": _git_commit(),
        "cpus": cpus or "unpinned",
    }
    info.update(extra)
    return {k: str(v) for k, v in info.items() if v not in (None, "")}


def write_run_info(path: str, info: dict[str, str]) -> str:
    """Write *info* as ``key: value`` lines to *path*; returns *path*."""
    width = max((len(k) for k in info), default=0) + 2
    with open(path, "w") as f:
        f.writelines(f"{k + ':':{width}}{v}\n" for k, v in info.items())
    return path
