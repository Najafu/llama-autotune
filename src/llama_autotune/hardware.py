"""Cross-platform hardware detection.

What llama.cpp itself reports is the authority: the devices listed by
``--list-devices`` decide the backend and VRAM, and the CPU variant it
loads decides the instruction-set tier. OS probes fill in the rest (CPU
name, core topology, RAM). WMI ``AdapterRAM`` is not used for VRAM
because it is a 32-bit field that misreports cards above 4 GB.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys

from .compat import _NO_WINDOW, Capabilities, detect_capabilities
from .models import Backend, GpuVendor, HardwareInfo
from .resources import cpu_topology, memory_status

GIB = 1024**3


def detect_hardware(caps: Capabilities | None = None, probe_llama: bool = True) -> HardwareInfo:
    """Detect the hardware of the current machine.

    Args:
        caps: Capabilities of the installed llama.cpp build; probed when
            None and ``probe_llama`` is True.
        probe_llama: Ask llama.cpp for its devices and CPU variant. When
            False, GPUs are guessed from OS names (informational only).

    Returns:
        A populated :class:`HardwareInfo`.
    """
    info = HardwareInfo(cpu_name=_get_cpu_name())
    topo = cpu_topology()
    info.physical_cores = topo.physical_cores
    info.logical_cores = topo.logical_cores
    info.performance_cores = topo.performance_cores
    info.efficiency_cores = topo.efficiency_cores
    info.performance_threads = topo.performance_logical
    info.numa_nodes = topo.numa_nodes or 1
    mem = memory_status()
    info.ram_gb = round(mem.total / GIB, 1)
    info.ram_available_gb = round(mem.available / GIB, 2)

    if caps is None and probe_llama:
        caps = detect_capabilities()
    if caps is not None:
        # Only devices llama.cpp lists can be used; an empty list means CPU,
        # even if the OS reports a GPU (or the probe timed out).
        info.cpu_variant, info.isa_tier = caps.cpu_variant, caps.isa_tier
        for dev in caps.devices:
            info.gpu_count += 1
            info.gpu_models.append(dev.description)
            info.vram_per_gpu.append(round(dev.total_mib / 1024, 1))
            info.gpu_vendor = _classify_gpu(dev.description)
        info.backend = _backend_from_devices([d.name for d in caps.devices])
        return info

    _detect_gpu_names(info)
    _determine_backend(info)
    return info


def hardware_fingerprint(hw: HardwareInfo) -> str:
    """Stable id for a machine: CPU, total RAM, GPUs and operating system family.

    The OS release is left out on purpose: it changes with every Linux kernel
    update and with a Windows upgrade, and the hardware is still the same.
    """
    data = json.dumps([hw.cpu_name, round(hw.ram_gb), sorted(hw.gpu_models), platform.system()])
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def legacy_hardware_fingerprint(hw: HardwareInfo) -> str:
    """The id used up to v0.5.0 (it included the OS release), to adopt old results."""
    data = json.dumps([hw.cpu_name, round(hw.ram_gb), sorted(hw.gpu_models), platform.system(), platform.release()])
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _backend_from_devices(names: list[str]) -> Backend:
    for prefix, backend in (("CUDA", Backend.CUDA), ("ROCm", Backend.ROCM), ("HIP", Backend.ROCM),
                            ("Metal", Backend.METAL), ("MTL", Backend.METAL), ("Vulkan", Backend.VULKAN),
                            ("SYCL", Backend.SYCL), ("OpenCL", Backend.OPENCL)):
        if any(n.startswith(prefix) for n in names):
            return backend
    return Backend.CPU


def _get_cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown CPU"


def _detect_gpu_names(info: HardwareInfo) -> None:
    """Fallback when llama.cpp is unavailable: GPU names only, no VRAM."""
    names: list[str] = []
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                text=True, timeout=15, creationflags=_NO_WINDOW)
            names = [n.strip() for n in out.splitlines() if n.strip()]
        elif sys.platform.startswith("linux"):
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                          text=True, timeout=10)
            names = [n.strip() for n in out.splitlines() if n.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    for name in names:
        vendor = _classify_gpu(name)
        if vendor != GpuVendor.UNKNOWN:
            info.gpu_count += 1
            info.gpu_models.append(name)
            info.gpu_vendor = vendor


def _classify_gpu(name: str) -> GpuVendor:
    """Classify a GPU vendor by keywords in its name."""
    n = name.lower()
    if any(x in n for x in ("nvidia", "geforce", "rtx", "gtx", "tesla", "quadro")):
        return GpuVendor.NVIDIA
    if any(x in n for x in ("amd", "radeon", "rx ", "instinct")):
        return GpuVendor.AMD
    if any(x in n for x in ("intel", "arc", "iris", "uhd")):
        return GpuVendor.INTEL
    if any(x in n for x in ("apple", "m1", "m2", "m3", "m4", "m5")):
        return GpuVendor.APPLE
    return GpuVendor.UNKNOWN


def _determine_backend(info: HardwareInfo) -> None:
    """Fallback backend guess from the GPU vendor (no llama.cpp available)."""
    info.backend = {
        GpuVendor.NVIDIA: Backend.CUDA,
        GpuVendor.AMD: Backend.ROCM,
        GpuVendor.INTEL: Backend.VULKAN,
        GpuVendor.APPLE: Backend.METAL,
    }.get(info.gpu_vendor, Backend.CPU)
