"""Tests for CPU pinning and the governor handling (fake sysfs, no root needed)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import usv.cpupin as cpupin

PCI = "0000:75:00.0"


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    """Fake topology: CPUs 0-7 on NUMA node 0, SMT pairs (n, n+4); 0/1 share a policy."""
    cpu_root = tmp_path / "cpu"
    for cpu in range(8):
        topo = cpu_root / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        topo.joinpath("thread_siblings_list").write_text(f"{cpu % 4},{cpu % 4 + 4}\n")
    policies = tmp_path / "cpufreq"
    for policy in ("policy01", "policy2", "policy3"):
        (policies / policy).mkdir(parents=True)
        (policies / policy / "scaling_governor").write_text("schedutil\n")
    for cpu, policy in ((0, "policy01"), (1, "policy01"), (2, "policy2"), (3, "policy3")):
        (cpu_root / f"cpu{cpu}" / "cpufreq").symlink_to(policies / policy)
    node = tmp_path / "node" / "node0"
    node.mkdir(parents=True)
    node.joinpath("cpulist").write_text("0-7\n")
    pci = tmp_path / "pci" / PCI
    pci.mkdir(parents=True)
    pci.joinpath("numa_node").write_text("0\n")
    monkeypatch.setattr(cpupin, "_CPU_SYSFS", cpu_root)
    monkeypatch.setattr(cpupin, "_NODE_SYSFS", tmp_path / "node")
    monkeypatch.setattr(cpupin, "_PCI_SYSFS", tmp_path / "pci")
    return policies


def test_parse_cpulist():
    assert cpupin.parse_cpulist("0-2,5,7-8\n") == {0, 1, 2, 5, 7, 8}


def test_gpu_local_cores_one_cpu_per_core(sysfs):
    assert cpupin.gpu_local_cores(PCI, allowed=set(range(8))) == [0, 1, 2, 3]


def test_gpu_local_cores_only_second_siblings_allowed(sysfs):
    # The lowest *allowed* sibling represents each core, so this is never empty.
    assert cpupin.gpu_local_cores(PCI, allowed={4, 5, 6, 7}) == [4, 5, 6, 7]


def test_visible_gpu_index(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    assert cpupin.visible_gpu_index() == 0
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,4")
    assert cpupin.visible_gpu_index() == 3


@pytest.mark.skipif(not set(range(8)) <= os.sched_getaffinity(0), reason="needs CPUs 0-7")
def test_pin_sets_and_restores(sysfs):
    before = os.sched_getaffinity(0)
    pin = cpupin.pin_cpus(2, pci=PCI, gpu_index=0)
    try:
        # Offset by GPU 0: cores [0,1,2,3] -> start 2 -> cpus 2,3, helpers their siblings.
        assert pin.cpus == [2, 3]
        assert pin.helpers == [6, 7]
        assert os.sched_getaffinity(0) == {2, 3}
        assert (sysfs / "policy2" / "scaling_governor").read_text() == "performance"
        assert (sysfs / "policy3" / "scaling_governor").read_text() == "performance"
        assert (sysfs / "policy01" / "scaling_governor").read_text() == "schedutil\n"
        assert "performance governor" in pin.description
    finally:
        pin.restore()
    assert os.sched_getaffinity(0) == before
    assert (sysfs / "policy2" / "scaling_governor").read_text() == "schedutil"
    pin.restore()  # idempotent


@pytest.mark.skipif(not {0, 1, 2, 3} <= os.sched_getaffinity(0), reason="needs CPUs 0-3")
def test_shared_policy_restored_to_original(sysfs):
    pin = cpupin.pin_cpus(4, pci=PCI, gpu_index=0)  # cpus 0-3; 0 and 1 share policy01
    try:
        assert (sysfs / "policy01" / "scaling_governor").read_text() == "performance"
    finally:
        pin.restore()
    assert (sysfs / "policy01" / "scaling_governor").read_text() == "schedutil"


def test_governor_guard_restores_when_parent_dies(tmp_path):
    gov = tmp_path / "scaling_governor"
    gov.write_text("performance")
    guard = subprocess.Popen(
        [sys.executable, "-c", cpupin._GOVERNOR_GUARD, f"{gov}=schedutil"],
        stdin=subprocess.PIPE,
    )
    guard.stdin.close()  # EOF without the "restored" message: the parent died
    guard.wait(timeout=30)
    assert gov.read_text() == "schedutil"


def test_governor_guard_leaves_restored_state(tmp_path):
    gov = tmp_path / "scaling_governor"
    gov.write_text("schedutil")
    guard = subprocess.Popen(
        [sys.executable, "-c", cpupin._GOVERNOR_GUARD, f"{gov}=ondemand"],
        stdin=subprocess.PIPE,
    )
    guard.communicate(b"restored", timeout=30)
    assert gov.read_text() == "schedutil"


def test_pinned_cpus_without_governor(sysfs):
    with cpupin.pinned_cpus(1, pci=PCI, governor=False, move_helpers=False) as pin:
        assert os.sched_getaffinity(0) == set(pin.cpus)
        assert "governor unchanged" in pin.description
    assert (sysfs / "policy2" / "scaling_governor").read_text() == "schedutil\n"
