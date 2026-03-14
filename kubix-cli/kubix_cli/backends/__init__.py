"""
Kubix CLI - GPU Backend Abstraction Layer

Unified interface for multiple GPU compute platforms:
- NVIDIA CUDA
- AMD ROCm
- Intel oneAPI (SYCL)
- Vulkan Compute (cross-platform fallback)
"""

from .base import GPUBackend, DeviceInfo, CompileOptions, KernelMetrics

__all__ = [
    "GPUBackend",
    "DeviceInfo",
    "CompileOptions",
    "KernelMetrics",
]
