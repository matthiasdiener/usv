"""Opt-in detection of other processes sharing the GPU.

A common (and easy to miss) source of noisy or biased micro-benchmarks is
*another* process running on the same GPU - a leftover training job, a second
benchmark, a display compositor.  This module queries the vendor SMI tool for
the compute processes on a device so you can check for such interference before
trusting a measurement.

Everything here is best-effort and read-only: if the SMI tool is missing or its
output can't be parsed, the functions return an empty list rather than raising,
so detection never breaks a benchmark.  It is also entirely opt-in - pass
``check_interference=True`` to :func:`usv.do_bench` / :func:`usv.do_bench_many`,
or call :func:`check_gpu_interference` directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from usv.clocks import gpu_vendor

__all__ = ["gpu_processes", "check_gpu_interference"]


def _run(cmd: list[str]) -> str | None:
    """Run *cmd* and return stdout, or ``None`` on any failure (best-effort)."""
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True, timeout=10
        )
    except Exception:
        return None


def _nvidia_processes(device: int) -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run(
        [
            exe,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
            "-i",
            str(device),
        ]
    )
    if not out:
        return []
    procs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        name = parts[1] if len(parts) > 1 else ""
        mem = None
        if len(parts) > 2 and parts[2].isdigit():
            mem = int(parts[2])
        procs.append({"pid": pid, "name": name, "memory_mb": mem})
    return procs


def _amd_processes(device: int) -> list[dict]:
    import json

    amd = shutil.which("amd-smi")
    if amd:
        out = _run([amd, "process", "--gpu", str(device), "--json"])
        procs = _parse_amd_json(out)
        if procs is not None:
            return procs
    rocm = shutil.which("rocm-smi")
    if rocm:
        out = _run([rocm, "--showpids", "--json"])
        if out:
            try:
                return _parse_rocm_pids(json.loads(out))
            except Exception:
                return []
    return []


def _parse_amd_json(out: str | None) -> list[dict] | None:
    """Parse ``amd-smi process --json`` output; ``None`` if it can't be read."""
    if not out:
        return None
    import json

    try:
        data = json.loads(out)
    except Exception:
        return None
    procs: list[dict] = []
    # amd-smi emits a list of per-GPU objects, each with a "process_list".
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        plist = entry.get("process_list", []) if isinstance(entry, dict) else []
        for item in plist:
            info = item.get("process_info", item) if isinstance(item, dict) else {}
            pid = info.get("pid") or info.get("process_id")
            if pid is None:
                continue
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            procs.append(
                {
                    "pid": pid,
                    "name": info.get("name") or info.get("process_name") or "",
                    "memory_mb": None,
                }
            )
    return procs


def _parse_rocm_pids(data: dict) -> list[dict]:
    """Parse ``rocm-smi --showpids --json`` output."""
    procs: list[dict] = []
    system = data.get("system", data) if isinstance(data, dict) else {}
    for key, val in system.items():
        if not key.upper().startswith("PID"):
            continue
        pid_str = key.split()[-1] if " " in key else key[3:]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        name = ""
        if isinstance(val, dict):
            name = val.get("name") or val.get("Name") or ""
        procs.append({"pid": pid, "name": name, "memory_mb": None})
    return procs


def gpu_processes(*, device: int = 0, vendor: str = "auto") -> list[dict]:
    """Return compute processes running on *device* (best-effort, may be empty).

    Each entry is ``{"pid": int, "name": str, "memory_mb": int | None}``.  Uses
    ``nvidia-smi`` on NVIDIA and ``amd-smi`` / ``rocm-smi`` on AMD; returns an
    empty list if no SMI tool is available or its output can't be parsed.
    """
    v = vendor if vendor != "auto" else gpu_vendor()
    if v == "nvidia":
        return _nvidia_processes(device)
    if v == "amd":
        return _amd_processes(device)
    return []


def check_gpu_interference(
    *, device: int = 0, vendor: str = "auto", exclude_pids: "list[int] | None" = None
) -> list[dict]:
    """Return *other* processes on *device*, excluding this process.

    A non-empty result means something else is using the GPU and the benchmark
    may be contended.  This process' PID is always excluded; pass additional
    ``exclude_pids`` (e.g. known helper processes) to ignore them too.
    """
    exclude = {os.getpid()} | set(exclude_pids or ())
    return [
        p for p in gpu_processes(device=device, vendor=vendor) if p["pid"] not in exclude
    ]
