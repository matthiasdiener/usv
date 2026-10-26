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
Processes are matched to the benchmark GPU by PCI address, because the SMI tools'
device indices ignore ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``.  Inside a
container the SMI tools report host PIDs, so this process can't be told apart from a
neighbor once it has touched the GPU; take a :func:`snapshot_gpu_processes` *before*
the first GPU call to get a list free of this process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from usv.clocks import gpu_vendor

__all__ = [
    "gpu_processes",
    "check_gpu_interference",
    "snapshot_gpu_processes",
    "foreign_processes",
]

# Monitoring tools open every GPU but aren't compute interference.  Matched after
# stripping non-alphanumerics, so e.g. "rocm-smi" -> "rocmsmi".
_MONITORING_TOOLS = ("nvtop", "amdsmi", "rocmsmi", "nvidiasmi")


def _is_monitoring_tool(name: str) -> bool:
    base = "".join(ch for ch in str(name or "").lower() if ch.isalnum())
    return bool(base) and base.startswith(_MONITORING_TOOLS)


def _norm_bdf(bdf: str) -> str:
    """Normalize a PCI address to ``dddd:bb:dd.f`` (lowercase, 4-digit domain)."""
    bdf = str(bdf).strip().lower()
    if bdf.count(":") == 1:
        bdf = "0000:" + bdf
    domain, _, rest = bdf.partition(":")
    return f"{domain[-4:].zfill(4)}:{rest}"


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
    try:
        data = json.loads(out)
    except Exception:
        return None
    return _amd_entry_processes(data)


def _amd_entry_processes(data) -> list[dict]:
    procs: list[dict] = []
    # amd-smi emits a list of per-GPU objects, each with a "process_list".
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        plist = entry.get("process_list", []) if isinstance(entry, dict) else []
        for item in plist:
            info = item.get("process_info", item) if isinstance(item, dict) else {}
            if not isinstance(info, dict):  # "No running processes detected"
                continue
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


def _amd_snapshot() -> dict[str, list[dict]] | None:
    amd = shutil.which("amd-smi")
    if not amd:
        return None
    gpu_out = _run([amd, "list", "--json"])
    proc_out = _run([amd, "process", "--json"])
    if gpu_out is None or proc_out is None:
        return None
    try:
        bdfs = {e["gpu"]: _norm_bdf(e["bdf"]) for e in json.loads(gpu_out)}
        entries = json.loads(proc_out)
    except Exception:
        return None
    snap: dict[str, list[dict]] = {bdf: [] for bdf in bdfs.values()}
    for entry in entries if isinstance(entries, list) else [entries]:
        bdf = bdfs.get(entry.get("gpu")) if isinstance(entry, dict) else None
        if bdf is not None:
            snap[bdf] = _amd_entry_processes(entry)
    return snap


def _nvidia_snapshot() -> dict[str, list[dict]] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = _run(
        [
            exe,
            "--query-compute-apps=gpu_bus_id,pid,process_name",
            "--format=csv,noheader",
        ]
    )
    if out is None:
        return None
    snap: dict[str, list[dict]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        name = parts[2] if len(parts) > 2 else ""
        snap.setdefault(_norm_bdf(parts[0]), []).append(
            {"pid": int(parts[1]), "name": name, "memory_mb": None}
        )
    return snap


def snapshot_gpu_processes(*, vendor: str = "auto") -> dict[str, list[dict]] | None:
    """Compute processes on every GPU, keyed by PCI address; ``None`` if unavailable.

    Uses only the SMI command-line tools, so calling it does not touch the GPU from
    this process: taken before the first GPU call, it lists genuine neighbors only.
    """
    v = vendor if vendor != "auto" else gpu_vendor()
    if v == "amd":
        return _amd_snapshot()
    if v == "nvidia":
        return _nvidia_snapshot()
    return None


def foreign_processes(
    snapshot: dict[str, list[dict]] | None,
    bdf: str | None,
    *,
    exclude_pids=None,
) -> list[dict]:
    """Processes of *snapshot* on the GPU at PCI address *bdf*, minus monitoring tools.

    Returns ``[]`` when the snapshot or *bdf* is missing, rather than risk a false
    positive.  PIDs in *exclude_pids* are dropped as well.
    """
    if not snapshot or not bdf:
        return []
    exclude = set(exclude_pids or ())
    return [
        p
        for p in snapshot.get(_norm_bdf(bdf), [])
        if p["pid"] not in exclude and not _is_monitoring_tool(p["name"])
    ]


def _torch_bdf(device: int) -> str | None:
    """PCI address of torch's ``cuda:device`` if torch already initialized the GPU."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_initialized():
            return None
        p = torch.cuda.get_device_properties(device)
        return "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)
    except Exception:
        return None


def check_gpu_interference(
    *, device: int = 0, vendor: str = "auto", exclude_pids: "list[int] | None" = None
) -> list[dict]:
    """Return *other* processes on *device*, excluding this process.

    A non-empty result means something else is using the GPU and the benchmark
    may be contended.  This process' PID is always excluded; pass additional
    ``exclude_pids`` (e.g. known helper processes) to ignore them too.  Monitoring
    tools (``nvtop``, ``amd-smi``, ...) are ignored.  If torch has already
    initialized the GPU, *device* is torch's ``cuda:device`` and is matched by PCI
    address; otherwise it is the SMI tool's own device index.
    """
    exclude = {os.getpid()} | set(exclude_pids or ())
    bdf = _torch_bdf(device)
    if bdf is not None:
        snap = snapshot_gpu_processes(vendor=vendor)
        if snap is not None:
            return foreign_processes(snap, bdf, exclude_pids=exclude)
    return [
        p
        for p in gpu_processes(device=device, vendor=vendor)
        if p["pid"] not in exclude and not _is_monitoring_tool(p["name"])
    ]
