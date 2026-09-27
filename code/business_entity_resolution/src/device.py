"""CUDA GPU acceleration detection and tensor helper utilities."""

from __future__ import annotations

import os
from typing import Any

# Ensure CUDA runtime DLLs are found on Windows
_CUDA_PATHS = [
    r"C:\AMAZON_ML_CHALLENGE\Amazon_ML_Challenge\venv\Lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\AMAZON_ML_CHALLENGE\Amazon_ML_Challenge\venv\Lib\site-packages\nvidia\cuda_nvrtc\bin",
]
for _p in _CUDA_PATHS:
    if os.path.isdir(_p):
        try:
            os.add_dll_directory(_p)
        except (AttributeError, OSError):
            pass


def get_cuda_status() -> dict[str, Any]:
    """Detect CUDA availability and return GPU hardware metadata."""
    status: dict[str, Any] = {
        "cuda_available": False,
        "device_type": "cpu",
        "device_name": "CPU",
        "memory_total_mb": 0.0,
        "memory_free_mb": 0.0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            status["cuda_available"] = True
            status["device_type"] = "cuda"
            status["device_name"] = torch.cuda.get_device_name(0)
            status["device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            status["memory_total_mb"] = props.total_memory / (1024 ** 2)
            mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
            status["memory_free_mb"] = status["memory_total_mb"] - mem_alloc
            status["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    return status


def print_cuda_info() -> str:
    """Print hardware device and CUDA configuration banner."""
    info = get_cuda_status()
    if info["cuda_available"]:
        banner = (
            f"[CUDA ACCELERATION ENABLED]\n"
            f"  Device:         {info['device_name']} (Device 0)\n"
            f"  VRAM:           {info['memory_total_mb']:.0f} MB total\n"
            f"  CUDA Version:   {info.get('cuda_version', '12.x')}\n"
        )
    else:
        banner = (
            f"[HARDWARE ACCELERATION: CPU FALLBACK]\n"
            f"  Device: Multi-core CPU with SIMD vectorization\n"
        )
    return banner
