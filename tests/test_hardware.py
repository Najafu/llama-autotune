from llama_autotune.hardware import (
    _classify_gpu,
    _determine_backend,
    detect_hardware,
)
from llama_autotune.models import Backend, GpuVendor, HardwareInfo


def test_detect_hardware():
    hw = detect_hardware()
    assert hw.physical_cores > 0
    assert hw.logical_cores > 0
    assert hw.ram_gb > 0
    assert hw.cpu_name != ""


def test_classify_nvidia():
    assert _classify_gpu("NVIDIA GeForce RTX 4090") == GpuVendor.NVIDIA
    assert _classify_gpu("Tesla V100") == GpuVendor.NVIDIA


def test_classify_amd():
    assert _classify_gpu("AMD Radeon RX 7900 XTX") == GpuVendor.AMD


def test_classify_intel():
    assert _classify_gpu("Intel Arc A770") == GpuVendor.INTEL
    assert _classify_gpu("Intel UHD Graphics") == GpuVendor.INTEL


def test_classify_apple():
    assert _classify_gpu("Apple M1") == GpuVendor.APPLE
    assert _classify_gpu("Apple M2 Max") == GpuVendor.APPLE


def test_classify_unknown():
    assert _classify_gpu("Unknown GPU") == GpuVendor.UNKNOWN


def test_determine_backend_nvidia():
    hw = HardwareInfo(gpu_vendor=GpuVendor.NVIDIA)
    _determine_backend(hw)
    assert hw.backend == Backend.CUDA


def test_determine_backend_cpu():
    hw = HardwareInfo()
    _determine_backend(hw)
    assert hw.backend == Backend.CPU


def test_determine_backend_apple():
    hw = HardwareInfo(gpu_vendor=GpuVendor.APPLE)
    _determine_backend(hw)
    assert hw.backend == Backend.METAL
