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
    performance_mask: int = 0

    @property
    def is_hybrid(self) -> bool:
        return self.efficiency_cores > 0

    @property
    def performance_logical(self) -> int:
        return bin(self.performance_mask).count("1") if self.performance_mask else self.logical_cores


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
_LTP_PC_SMT = 0x1


def cpu_topology() -> CpuTopology:
    """Return core counts, separating performance and efficiency cores.

    Windows: ``GetLogicalProcessorInformationEx`` reports an efficiency
    class per core; the highest class is the performance cores. Linux
    hybrid Intel CPUs list their cores under ``/sys/devices/cpu_core``.
    Elsewhere every core counts as a performance core.
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
    return topo


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

    cores: list[tuple[int, int]] = []  # (efficiency class, affinity mask in group 0)
    offset = 0
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    while offset < length.value:
        relationship, size = ctypes.c_ulong.from_buffer(buf, offset).value, ctypes.c_ulong.from_buffer(buf, offset + 4).value
        if size == 0:
            break
        if relationship == _RELATION_PROCESSOR_CORE:
            body = offset + 8
            eff_class = ctypes.c_ubyte.from_buffer(buf, body + 1).value
            group_count = ctypes.c_ushort.from_buffer(buf, body + 22).value
            mask = 0
            for g in range(group_count):
                entry = body + 24 + g * (ptr_size + 8)
                if ctypes.c_ushort.from_buffer(buf, entry + ptr_size).value == 0:
                    mask |= ctypes.c_size_t.from_buffer(buf, entry).value
            cores.append((eff_class, mask))
        offset += size

    if not cores:
        return None
    top_class = max(c for c, _ in cores)
    perf = [m for c, m in cores if c == top_class]
    perf_mask = 0
    for m in perf:
        perf_mask |= m
    logical = sum(bin(m).count("1") for _, m in cores) or (psutil.cpu_count(logical=True) or len(cores))
    return CpuTopology(
        physical_cores=len(cores),
        logical_cores=logical,
        performance_cores=len(perf),
        efficiency_cores=len(cores) - len(perf),
        performance_mask=perf_mask if len(perf) < len(cores) else 0,
    )


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
    mask = 0
    for c in p_cpus:
        mask |= 1 << c
    return CpuTopology(
        physical_cores=physical, logical_cores=logical,
        performance_cores=len(p_cores) or physical - len(e_cpus),
        efficiency_cores=len(e_cpus), performance_mask=mask,
    )


# ── load and bandwidth ────────────────────────────────────────────────


def cpu_load(seconds: float = 1.0) -> float:
    """Return the system-wide CPU load in percent over a short interval."""
    return psutil.cpu_percent(interval=seconds)


def measure_bandwidth(size_mib: int = 128, repeats: int = 6) -> float:
    """Measure memory copy bandwidth in GB/s (read plus write bytes).

    A plain ``bytearray`` copy runs as ``memcpy``; on the test PC it gave
    16.2 to 17.3 GB/s in about 0.2 s, the same as a numpy copy. The best
    of several runs is reported, as in the STREAM rules.

    Args:
        size_mib: Size of each buffer; keep it far above the CPU cache.
        repeats: Number of copies; the fastest one counts.

    Returns:
        Bandwidth in GB/s, or 0.0 if the buffers cannot be allocated.
    """
    n = size_mib * MIB
    try:
        src = bytearray(b"\x01") * n
        dst = bytearray(n)
    except MemoryError:
        return 0.0
    best = 0.0
    for _ in range(repeats):
        t = time.perf_counter()
        dst[:] = src
        dt = time.perf_counter() - t
        if dt > 0:
            best = max(best, 2 * n / dt / 1e9)
    return round(best, 1)
