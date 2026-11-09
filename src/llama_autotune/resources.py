"""Live system resources: memory, power state, CPU topology, bandwidth.

Windows is probed through Win32 APIs with ctypes (no administrator
rights, no PowerShell start-up delay); other platforms and any failed
call fall back to psutil. Every probe degrades to a safe default instead
of raising.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import dataclass

import psutil

MIB = 1024 * 1024


@dataclass
class MemoryStatus:
    """Physical memory and commit state in bytes."""

    total: int = 0
    available: int = 0
    commit_total: int = 0
    commit_limit: int = 0

    @property
    def commit_headroom(self) -> int:
        """Bytes that can still be committed before the page file is full."""
        return max(0, self.commit_limit - self.commit_total) if self.commit_limit else 0


@dataclass
class PowerStatus:
    """Power source of the machine."""

    on_battery: bool = False
    has_battery: bool = False
    battery_saver: bool = False


@dataclass
class CpuTopology:
    """Physical layout of the CPU cores visible to this process."""

    physical_cores: int = 0
    logical_cores: int = 0
    performance_cores: int = 0
    efficiency_cores: int = 0
    perf_logical: int = 0
    numa_nodes: int = 0

    @property
    def is_hybrid(self) -> bool:
        return self.efficiency_cores > 0

    @property
    def performance_logical(self) -> int:
        """Logical processors of the performance cores (all of them if not hybrid)."""
        return self.perf_logical or self.logical_cores


# ── memory ────────────────────────────────────────────────────────────


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong),
    ]


def memory_status() -> MemoryStatus:
    """Return total and available RAM and the system commit state.

    On Windows, available RAM is free plus standby pages
    (``GlobalMemoryStatusEx``), so a model file cached by an earlier run
    still counts as available.
    """
    if sys.platform == "win32":
        try:
            ms = _MEMORYSTATUSEX()
            ms.dwLength = ctypes.sizeof(ms)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                status = MemoryStatus(total=ms.ullTotalPhys, available=ms.ullAvailPhys)
                pi = _PERFORMANCE_INFORMATION()
                pi.cb = ctypes.sizeof(pi)
                if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
                    status.commit_total = pi.CommitTotal * pi.PageSize
                    status.commit_limit = pi.CommitLimit * pi.PageSize
                return status
        except (OSError, AttributeError):
            pass
    vm = psutil.virtual_memory()
    status = MemoryStatus(total=vm.total, available=vm.available)
    try:
        sw = psutil.swap_memory()
        status.commit_limit = vm.total + sw.total
        status.commit_total = (vm.total - vm.available) + sw.used
    except (OSError, RuntimeError):
        pass
    return status


# ── power ─────────────────────────────────────────────────────────────


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def power_status() -> PowerStatus:
    """Return whether the machine runs on battery and battery saver is on."""
    if sys.platform == "win32":
        try:
            ps = _SYSTEM_POWER_STATUS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(ps)):
                has_battery = ps.BatteryFlag != 128 and ps.BatteryFlag != 255
                return PowerStatus(
                    on_battery=has_battery and ps.ACLineStatus == 0,
                    has_battery=has_battery,
                    battery_saver=bool(ps.SystemStatusFlag & 1),
                )
        except (OSError, AttributeError):
            pass
    try:
        bat = psutil.sensors_battery()
    except (AttributeError, OSError, RuntimeError):
        bat = None
    if bat is None:
        return PowerStatus()
    return PowerStatus(on_battery=not bat.power_plugged, has_battery=True)


# ── CPU topology ──────────────────────────────────────────────────────

_RELATION_PROCESSOR_CORE = 0


def cpu_topology() -> CpuTopology:
    """Return core counts, separating performance and efficiency cores.

    Windows: ``GetLogicalProcessorInformationEx`` reports an efficiency
    class per core; the highest class is the performance cores. Linux
    hybrid Intel CPUs list their cores under ``/sys/devices/cpu_core``.
    Elsewhere every core counts as a performance core. The number of
    NUMA nodes (memory controllers; usually one per CPU socket) is
    reported too.
    """
    topo = None
    if sys.platform == "win32":
        topo = _topology_windows()
    elif sys.platform.startswith("linux"):
        topo = _topology_linux()
    if topo is None or topo.physical_cores == 0:
        physical = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or physical
        topo = CpuTopology(physical_cores=physical, logical_cores=logical, performance_cores=physical)
    if not topo.numa_nodes:
        topo.numa_nodes = numa_node_count()
    return topo


def numa_node_count() -> int:
    """Number of NUMA nodes; 1 on ordinary desktops and laptops."""
    if sys.platform == "win32":
        try:
            highest = ctypes.c_ulong(0)
            if ctypes.windll.kernel32.GetNumaHighestNodeNumber(ctypes.byref(highest)):
                return highest.value + 1
        except (OSError, AttributeError):
            pass
        return 1
    try:
        nodes = [n for n in os.listdir("/sys/devices/system/node") if n.startswith("node") and n[4:].isdigit()]
        return max(1, len(nodes))
    except OSError:
        return 1


def parse_core_records(buf: bytes, ptr_size: int = 8) -> list[tuple[int, int]]:
    """Parse ``RelationProcessorCore`` records from ``GetLogicalProcessorInformationEx``.

    Each ``PROCESSOR_RELATIONSHIP`` holds the affinity mask of one core in
    its processor group. Windows splits machines with more than 64
    logical processors into several groups, so threads are counted in
    every group, not only group 0.

    Args:
        buf: The raw buffer filled by the API.
        ptr_size: Size of a pointer (``KAFFINITY``) in bytes.

    Returns:
        One ``(efficiency_class, logical_processor_count)`` per core.
    """
    cores: list[tuple[int, int]] = []
    offset = 0
    while offset + 8 <= len(buf):
        relationship = int.from_bytes(buf[offset:offset + 4], "little")
        size = int.from_bytes(buf[offset + 4:offset + 8], "little")
        if size == 0:
            break
        if relationship == _RELATION_PROCESSOR_CORE:
            body = offset + 8
            eff_class = buf[body + 1]
            group_count = int.from_bytes(buf[body + 22:body + 24], "little")
            threads = 0
            for g in range(group_count):
                entry = body + 24 + g * (ptr_size + 8)
                threads += bin(int.from_bytes(buf[entry:entry + ptr_size], "little")).count("1")
            cores.append((eff_class, threads))
        offset += size
    return cores


def topology_from_cores(cores: list[tuple[int, int]], numa_nodes: int = 0) -> CpuTopology | None:
    """Build a :class:`CpuTopology` from ``(efficiency_class, threads)`` per core."""
    if not cores:
        return None
    top_class = max(c for c, _ in cores)
    perf = [t for c, t in cores if c == top_class]
    hybrid = len(perf) < len(cores)
    return CpuTopology(
        physical_cores=len(cores),
        logical_cores=sum(t for _, t in cores),
        performance_cores=len(perf),
        efficiency_cores=len(cores) - len(perf),
        perf_logical=sum(perf) if hybrid else 0,
        numa_nodes=numa_nodes,
    )


def _topology_windows() -> CpuTopology | None:
    try:
        kernel32 = ctypes.windll.kernel32
        length = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, None, ctypes.byref(length))
        buf = ctypes.create_string_buffer(length.value)
        if not kernel32.GetLogicalProcessorInformationEx(_RELATION_PROCESSOR_CORE, buf, ctypes.byref(length)):
            return None
    except (OSError, AttributeError):
        return None
    return topology_from_cores(parse_core_records(buf.raw[:length.value], ctypes.sizeof(ctypes.c_void_p)))


def _read_cpu_list(path: str) -> list[int]:
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return []
    cpus: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def _topology_linux() -> CpuTopology | None:
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    p_cpus = _read_cpu_list("/sys/devices/cpu_core/cpus")
    e_cpus = _read_cpu_list("/sys/devices/cpu_atom/cpus")
    if not p_cpus or not e_cpus:
        return CpuTopology(physical_cores=physical, logical_cores=logical, performance_cores=physical)
    p_cores = {open(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read().strip() for c in p_cpus
               if os.path.exists(f"/sys/devices/system/cpu/cpu{c}/topology/core_id")}
    return CpuTopology(
        physical_cores=physical, logical_cores=logical,
        performance_cores=len(p_cores) or physical - len(e_cpus),
        efficiency_cores=len(e_cpus), perf_logical=len(p_cpus),
    )


# ── load and bandwidth ────────────────────────────────────────────────


def cpu_load(seconds: float = 1.0) -> float:
    """Return the system-wide CPU load in percent over a short interval."""
    return psutil.cpu_percent(interval=seconds)


def measure_bandwidth(size_mib: int = 128, repeats: int = 6, threads: int = 1) -> float:
    """Measure memory copy bandwidth in GB/s (read plus write bytes).

    With one thread a plain ``bytearray`` copy runs as ``memcpy``; on the
    test PC it gave 16.2 to 17.3 GB/s in about 0.2 s, the same as a numpy
    copy. One core cannot saturate the memory of workstations and
    servers (many channels, several sockets), so with ``threads`` > 1
    each thread copies its own buffers through ``ctypes.memmove``, which
    releases the GIL, and the higher of the two figures is reported. The
    best of several runs counts, as in the STREAM rules.

    Args:
        size_mib: Size of the single-thread buffer; keep it far above the
            CPU cache.
        repeats: Number of copies; the fastest one counts.
        threads: Copy threads for the multi-thread pass (capped at 16).

    Returns:
        Bandwidth in GB/s, or 0.0 if the buffers cannot be allocated.
    """
    n = size_mib * MIB
    best = 0.0
    try:
        src = bytearray(b"\x01") * n
        dst = bytearray(n)
    except MemoryError:
        return 0.0
    for _ in range(repeats):
        t = time.perf_counter()
        dst[:] = src
        dt = time.perf_counter() - t
        if dt > 0:
            best = max(best, 2 * n / dt / 1e9)
    del src, dst
    threads = min(threads, 16)
    if threads > 1:
        best = max(best, _threaded_bandwidth(threads, max(32, size_mib // 2), repeats))
    return round(best, 1)


def _threaded_bandwidth(threads: int, size_mib: int, repeats: int) -> float:
    n = size_mib * MIB
    try:
        bufs = [(ctypes.create_string_buffer(n), ctypes.create_string_buffer(n)) for _ in range(threads)]
    except MemoryError:
        return 0.0
    for src, dst in bufs:  # touch every page before timing
        ctypes.memmove(dst, src, n)
    best = 0.0
    for _ in range(repeats):
        start = threading.Barrier(threads + 1)

        def copy(src, dst):
            start.wait()
            ctypes.memmove(dst, src, n)

        workers = [threading.Thread(target=copy, args=pair) for pair in bufs]
        for w in workers:
            w.start()
        start.wait()
        t = time.perf_counter()
        for w in workers:
            w.join()
        dt = time.perf_counter() - t
        if dt > 0:
            best = max(best, 2 * n * threads / dt / 1e9)
    return best
