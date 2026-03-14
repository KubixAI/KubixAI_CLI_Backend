"""
Kubix CLI - GPU Backend Base Classes

Abstract interface for GPU compute backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class BackendType(Enum):
    """Supported GPU backend types."""
    CUDA = "cuda"
    ROCM = "rocm"
    SYCL = "sycl"
    VULKAN = "vulkan"
    METAL = "metal"


@dataclass
class DeviceInfo:
    """Information about a GPU device."""
    id: int
    name: str
    compute_capability: Tuple[int, int]  # (major, minor)
    total_memory_mb: float
    multiprocessor_count: int
    max_threads_per_block: int
    max_shared_memory_per_block: int
    max_registers_per_block: int
    warp_size: int
    backend_type: BackendType
    pci_bus_id: Optional[str] = None
    driver_version: Optional[str] = None

    def __str__(self):
        return (
            f"{self.name} (compute {self.compute_capability[0]}.{self.compute_capability[1]}, "
            f"{self.total_memory_mb / 1024:.1f} GB)"
        )


@dataclass
class CompileOptions:
    """Compilation options for kernel."""
    optimization_level: int = 3  # O0, O1, O2, O3
    use_fast_math: bool = True
    generate_line_info: bool = True
    max_registers: Optional[int] = None
    target_arch: Optional[str] = None
    extra_flags: List[str] = None

    def __post_init__(self):
        if self.extra_flags is None:
            self.extra_flags = []


@dataclass
class KernelMetrics:
    """Runtime metrics for a kernel."""
    execution_time_ms: float
    memory_bandwidth_gbps: float
    occupancy_percent: float
    register_usage: int
    shared_memory_bytes: int
    local_memory_bytes: int
    achieved_occupancy: float
    warp_efficiency: float
    branch_divergence_percent: float


class GPUBackend(ABC):
    """
    Abstract base class for GPU compute backends.

    All GPU backends (CUDA, ROCm, SYCL, Vulkan) must implement this interface
    to ensure consistent behavior across platforms.
    """

    def __init__(self):
        self._devices: List[DeviceInfo] = []
        self._current_device: Optional[DeviceInfo] = None
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the backend and detect available devices.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available on the system.

        Returns:
            True if backend can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """Get the type of this backend."""
        pass

    @abstractmethod
    def get_devices(self) -> List[DeviceInfo]:
        """
        Get list of all available GPU devices.

        Returns:
            List of DeviceInfo objects
        """
        pass

    @abstractmethod
    def set_device(self, device_id: int) -> None:
        """
        Set the current active device.

        Args:
            device_id: ID of device to activate

        Raises:
            DeviceNotFoundError: If device_id is invalid
        """
        pass

    @abstractmethod
    def get_current_device(self) -> DeviceInfo:
        """
        Get information about the current active device.

        Returns:
            DeviceInfo for current device

        Raises:
            BackendError: If no device is set
        """
        pass

    @abstractmethod
    def compile_kernel(
        self,
        source_code: str,
        kernel_name: str,
        options: Optional[CompileOptions] = None
    ) -> Any:
        """
        Compile kernel source code.

        Args:
            source_code: Kernel source code
            kernel_name: Name of the kernel function
            options: Compilation options

        Returns:
            Compiled kernel object (backend-specific)

        Raises:
            CompilationError: If compilation fails
        """
        pass

    @abstractmethod
    def launch_kernel(
        self,
        kernel: Any,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        args: List[Any],
        shared_memory_bytes: int = 0
    ) -> None:
        """
        Launch a compiled kernel.

        Args:
            kernel: Compiled kernel object
            grid_size: Grid dimensions (x, y, z)
            block_size: Block dimensions (x, y, z)
            args: Kernel arguments
            shared_memory_bytes: Dynamic shared memory size

        Raises:
            BackendError: If launch fails
        """
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Wait for all operations on current device to complete."""
        pass

    @abstractmethod
    def allocate_memory(self, size_bytes: int) -> Any:
        """
        Allocate device memory.

        Args:
            size_bytes: Number of bytes to allocate

        Returns:
            Device memory handle (backend-specific)

        Raises:
            MemoryError: If allocation fails
        """
        pass

    @abstractmethod
    def copy_to_device(self, host_data: Any, device_ptr: Any) -> None:
        """
        Copy data from host to device.

        Args:
            host_data: Host data (numpy array or similar)
            device_ptr: Device memory pointer
        """
        pass

    @abstractmethod
    def copy_from_device(self, device_ptr: Any, host_data: Any) -> None:
        """
        Copy data from device to host.

        Args:
            device_ptr: Device memory pointer
            host_data: Host buffer to receive data
        """
        pass

    @abstractmethod
    def free_memory(self, device_ptr: Any) -> None:
        """
        Free device memory.

        Args:
            device_ptr: Device memory pointer to free
        """
        pass

    @abstractmethod
    def benchmark_kernel(
        self,
        kernel: Any,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        args: List[Any],
        iterations: int = 100,
        warmup: int = 10
    ) -> float:
        """
        Benchmark kernel execution time.

        Args:
            kernel: Compiled kernel
            grid_size: Grid dimensions
            block_size: Block dimensions
            args: Kernel arguments
            iterations: Number of timing iterations
            warmup: Number of warmup iterations

        Returns:
            Median execution time in milliseconds
        """
        pass

    @abstractmethod
    def profile_kernel(
        self,
        kernel: Any,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        args: List[Any]
    ) -> KernelMetrics:
        """
        Profile kernel and collect detailed metrics.

        Args:
            kernel: Compiled kernel
            grid_size: Grid dimensions
            block_size: Block dimensions
            args: Kernel arguments

        Returns:
            KernelMetrics with detailed profiling data
        """
        pass

    @abstractmethod
    def get_kernel_attributes(self, kernel: Any) -> Dict[str, Any]:
        """
        Get attributes of compiled kernel.

        Args:
            kernel: Compiled kernel

        Returns:
            Dictionary with kernel attributes (registers, shared memory, etc.)
        """
        pass

    @abstractmethod
    def validate_launch_config(
        self,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        shared_memory_bytes: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate kernel launch configuration.

        Args:
            grid_size: Grid dimensions
            block_size: Block dimensions
            shared_memory_bytes: Dynamic shared memory size

        Returns:
            (is_valid, error_message) tuple
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up backend resources."""
        pass

    # ========================================================================
    # Utility Methods (with default implementations)
    # ========================================================================

    def get_device_count(self) -> int:
        """Get number of available devices."""
        return len(self.get_devices())

    def get_memory_info(self) -> Tuple[float, float]:
        """
        Get memory information for current device.

        Returns:
            (free_mb, total_mb) tuple
        """
        device = self.get_current_device()
        return (device.total_memory_mb, device.total_memory_mb)

    def supports_feature(self, feature: str) -> bool:
        """
        Check if backend supports a specific feature.

        Args:
            feature: Feature name (e.g., 'tensor_cores', 'fp16', 'int8')

        Returns:
            True if feature is supported
        """
        return False  # Override in subclasses

    def get_backend_version(self) -> str:
        """Get backend version string."""
        return "unknown"

    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
