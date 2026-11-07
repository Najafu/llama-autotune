"""Tests for resources.py probes and the benchmark watchdog."""

import subprocess
import sys

from llama_autotune.benchmark import watch_process
from llama_autotune.resources import cpu_topology, measure_bandwidth, memory_status, power_status


def test_memory_status_is_sane():
    m = memory_status()
    assert m.total > 0
    assert 0 < m.available <= m.total


def test_power_status_returns_flags():
    p = power_status()
    assert isinstance(p.on_battery, bool)
    assert not (p.on_battery and not p.has_battery)


def test_cpu_topology_counts():
    t = cpu_topology()
    assert t.physical_cores >= 1
    assert t.logical_cores >= t.physical_cores
    assert t.performance_cores + t.efficiency_cores == t.physical_cores
    assert t.performance_logical >= 1


def test_measure_bandwidth_is_positive():
    assert measure_bandwidth(size_mib=16, repeats=2) > 0


def _sleeper(seconds: float) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


def test_watchdog_lets_a_normal_process_finish():
    report = watch_process(_sleeper(0.3), timeout=30, min_available_mib=0)
    assert report.stopped_by == ""


def test_watchdog_stops_on_timeout():
    proc = _sleeper(30)
    report = watch_process(proc, timeout=0.5, min_available_mib=0)
    assert report.stopped_by == "timeout"
    assert proc.poll() is not None


def test_watchdog_stops_when_ram_floor_is_crossed():
    proc = _sleeper(30)
    report = watch_process(proc, timeout=30, min_available_mib=10**9)
    assert report.stopped_by in ("memory_guard", "memory_guard_external")
    assert "available RAM fell" in report.detail
    assert proc.poll() is not None
