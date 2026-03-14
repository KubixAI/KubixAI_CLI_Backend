"""
Kubix CLI - NVIDIA CUDA Backend

Implementation of GPUBackend for NVIDIA CUDA using PyCUDA.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import statistics

from .base import (
    GPUBackend, DeviceInfo, CompileOptions, KernelMetrics, BackendType
)
from ..exceptions import (
    BackendNotAvailableError, BackendInitializationError,
    DeviceNotFoundError, CompilationError, MemoryError as KubixMemoryError
)


class CUDABackend(GPUBackend):
    """NVIDIA CUDA backend implementation using PyCUDA."""

    def __init__(self):
        super().__init__()
        self._cuda = None
        self._context = None
        self._module_cache: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """Initialize CUDA backend and detect devices."""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

            self._cuda = cuda
            self._SourceModule = SourceModule

            # Initialize CUDA
            cuda.init()

            # Detect all devices
            self._devices = []
            for i in range(cuda.Device.count()):
                device = cuda.Device(i)
                compute_cap = device.compute_capability()
                total_mem = device.total_memory() / (1024 ** 2)  # Convert to MB

                device_info = DeviceInfo(
                    id=i,
                    name=device.name(),
                    compute_capability=compute_cap,
                    total_memory_mb=total_mem,
                    multiprocessor_count=device.get_attribute(
                        cuda.device_attribute.MULTIPROCESSOR_COUNT
                    ),
                    max_threads_per_block=device.get_attribute(
                        cuda.device_attribute.MAX_THREADS_PER_BLOCK
                    ),
                    max_shared_memory_per_block=device.get_attribute(
                        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
                    ),
                    max_registers_per_block=device.get_attribute(
                        cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
                    ),
                    warp_size=device.get_attribute(
                        cuda.device_attribute.WARP_SIZE
                    ),
                    backend_type=BackendType.CUDA,
                    pci_bus_id=device.pci_bus_id()
                )
                self._devices.append(device_info)

            # Set first device as current
            if self._devices:
                self._current_device = self._devices[0]
                self._initialized = True
                return True

            return False

        except ImportError as e:
            raise BackendNotAvailableError(
                "cuda",
                reason=f"PyCUDA not installed: {e}"
            )
        except Exception as e:
            raise BackendInitializationError("cuda", e)

    def is_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            return cuda.Device.count() > 0
        except:
            return False

    def get_backend_type(self) -> BackendType:
        """Get backend type."""
        return BackendType.CUDA

    def get_devices(self) -> List[DeviceInfo]:
        """Get list of available CUDA devices."""
        if not self._initialized:
            self.initialize()
        return self._devices

    def set_device(self, device_id: int) -> None:
        """Set current active device."""
        if device_id < 0 or device_id >= len(self._devices):
            raise DeviceNotFoundError("cuda")

        self._current_device = self._devices[device_id]

        # Switch CUDA context
        if self._cuda:
            device = self._cuda.Device(device_id)
            if self._context:
                self._context.pop()
            self._context = device.make_context()

    def get_current_device(self) -> DeviceInfo:
        """Get current active device."""
        if not self._current_device:
            raise DeviceNotFoundError("cuda")
        return self._current_device

    def compile_kernel(
        self,
        source_code: str,
        kernel_name: str,
        options: Optional[CompileOptions] = None
    ) -> Any:
        """Compile CUDA kernel from source."""
        if not self._initialized:
            self.initialize()

        options = options or CompileOptions()

        # Build compilation options
        compile_opts = []

        if options.optimization_level == 0:
            compile_opts.append("-O0")
        elif options.optimization_level == 1:
            compile_opts.append("-O1")
        elif options.optimization_level == 2:
            compile_opts.append("-O2")
        else:
            compile_opts.append("-O3")

        if options.use_fast_math:
            compile_opts.append("-use_fast_math")

        if options.generate_line_info:
            compile_opts.append("-lineinfo")

        if options.max_registers:
            compile_opts.append(f"-maxrregcount={options.max_registers}")

        if options.target_arch:
            compile_opts.append(f"-arch={options.target_arch}")
        else:
            # Use current device compute capability
            major, minor = self._current_device.compute_capability
            compile_opts.append(f"-arch=sm_{major}{minor}")

        compile_opts.extend(options.extra_flags)

        # Prepare kernel code with includes
        full_code = self._prepare_kernel_code(source_code)

        # Compile
        try:
            module = self._SourceModule(
                full_code,
                options=compile_opts,
                no_extern_c=False
            )

            # Cache the module
            cache_key = f"{kernel_name}_{hash(source_code)}"
            self._module_cache[cache_key] = module

            # Get kernel function
            kernel_func = module.get_function(kernel_name)

            return {
                'module': module,
                'function': kernel_func,
                'kernel_name': kernel_name,
                'source_code': source_code,
                'options': options
            }

        except Exception as e:
            # Parse compilation errors
            error_msg = str(e)
            errors = self._parse_compilation_errors(error_msg)
            raise CompilationError(source_code, errors)

    def launch_kernel(
        self,
        kernel: Any,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        args: List[Any],
        shared_memory_bytes: int = 0
    ) -> None:
        """Launch compiled kernel."""
        if not self._initialized:
            self.initialize()

        kernel_func = kernel['function']

        # Validate launch configuration
        is_valid, error = self.validate_launch_config(
            grid_size, block_size, shared_memory_bytes
        )
        if not is_valid:
            raise ValueError(f"Invalid launch configuration: {error}")

        # Convert args to CUDA-compatible format
        cuda_args = self._prepare_kernel_args(args)

        # Launch kernel
        try:
            kernel_func(
                *cuda_args,
                block=block_size,
                grid=grid_size,
                shared=shared_memory_bytes
            )
        except Exception as e:
            raise RuntimeError(f"Kernel launch failed: {e}")

    def synchronize(self) -> None:
        """Synchronize device."""
        if self._cuda:
            self._cuda.Context.synchronize()

    def allocate_memory(self, size_bytes: int) -> Any:
        """Allocate device memory."""
        if not self._initialized:
            self.initialize()

        try:
            return self._cuda.mem_alloc(size_bytes)
        except Exception as e:
            available_mb = self._current_device.total_memory_mb
            required_mb = size_bytes / (1024 ** 2)
            raise KubixMemoryError(required_mb, available_mb)

    def copy_to_device(self, host_data: Any, device_ptr: Any) -> None:
        """Copy data from host to device."""
        if isinstance(host_data, np.ndarray):
            self._cuda.memcpy_htod(device_ptr, host_data)
        else:
            # Try to convert to numpy array
            arr = np.array(host_data, dtype=np.float32)
            self._cuda.memcpy_htod(device_ptr, arr)

    def copy_from_device(self, device_ptr: Any, host_data: Any) -> None:
        """Copy data from device to host."""
        if isinstance(host_data, np.ndarray):
            self._cuda.memcpy_dtoh(host_data, device_ptr)
        else:
            raise ValueError("host_data must be a numpy array")

    def free_memory(self, device_ptr: Any) -> None:
        """Free device memory."""
        if device_ptr:
            device_ptr.free()

    def benchmark_kernel(
        self,
        kernel: Any,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        args: List[Any],
        iterations: int = 100,
        warmup: int = 10
    ) -> float:
        """Benchmark kernel execution time."""
        if not self._initialized:
            self.initialize()

        kernel_func = kernel['function']
        cuda_args = self._prepare_kernel_args(args)

        # Warmup
        for _ in range(warmup):
            kernel_func(
                *cuda_args,
                block=block_size,
                grid=grid_size
            )
        self.synchronize()

        # Benchmark using CUDA events
        times = []
        for _ in range(iterations):
            start_event = self._cuda.Event()
            end_event = self._cuda.Event()

            start_event.record()
            kernel_func(
                *cuda_args,
                block=block_size,
                grid=grid_size
            )
            end_event.record()
            end_event.synchronize()

            elapsed_ms = start_event.time_till(end_event)
            times.append(elapsed_ms)

        return statistics.median(times)

    def profile_kernel(
        self,
        kernel: Any,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        args: List[Any]
    ) -> KernelMetrics:
        """Profile kernel and collect metrics."""
        if not self._initialized:
            self.initialize()

        # Get kernel attributes
        kernel_func = kernel['function']
        attrs = self.get_kernel_attributes(kernel)

        # Benchmark execution time
        exec_time = self.benchmark_kernel(
            kernel, grid_size, block_size, args,
            iterations=10, warmup=5
        )

        # Calculate occupancy
        threads_per_block = block_size[0] * block_size[1] * block_size[2]
        occupancy = self._calculate_occupancy(
            threads_per_block,
            attrs['num_regs'],
            attrs['shared_size_bytes']
        )

        # Estimate memory bandwidth (simplified)
        # This would need actual memory transfer sizes for accuracy
        memory_bandwidth = 0.0  # TODO: Calculate from actual transfers

        return KernelMetrics(
            execution_time_ms=exec_time,
            memory_bandwidth_gbps=memory_bandwidth,
            occupancy_percent=occupancy * 100,
            register_usage=attrs['num_regs'],
            shared_memory_bytes=attrs['shared_size_bytes'],
            local_memory_bytes=attrs['local_size_bytes'],
            achieved_occupancy=occupancy,
            warp_efficiency=0.95,  # Placeholder
            branch_divergence_percent=5.0  # Placeholder
        )

    def get_kernel_attributes(self, kernel: Any) -> Dict[str, Any]:
        """Get kernel attributes."""
        kernel_func = kernel['function']

        return {
            'num_regs': kernel_func.num_regs,
            'shared_size_bytes': kernel_func.shared_size_bytes,
            'const_size_bytes': kernel_func.const_size_bytes,
            'local_size_bytes': kernel_func.local_size_bytes,
            'max_threads_per_block': kernel_func.max_threads_per_block
        }

    def validate_launch_config(
        self,
        grid_size: Tuple[int, int, int],
        block_size: Tuple[int, int, int],
        shared_memory_bytes: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """Validate kernel launch configuration."""
        device = self._current_device

        # Check threads per block
        threads = block_size[0] * block_size[1] * block_size[2]
        if threads > device.max_threads_per_block:
            return False, f"Threads per block {threads} exceeds max {device.max_threads_per_block}"

        # Check shared memory
        if shared_memory_bytes > device.max_shared_memory_per_block:
            return False, f"Shared memory {shared_memory_bytes} exceeds max {device.max_shared_memory_per_block}"

        # Check grid dimensions
        max_grid_dim = 2147483647
        for dim in grid_size:
            if dim > max_grid_dim:
                return False, f"Grid dimension {dim} exceeds max {max_grid_dim}"

        return True, None

    def cleanup(self) -> None:
        """Clean up CUDA resources."""
        if self._context:
            self._context.pop()
            self._context = None
        self._module_cache.clear()
        self._initialized = False

    def get_backend_version(self) -> str:
        """Get CUDA version."""
        if self._cuda:
            version = self._cuda.get_version()
            return f"{version[0]}.{version[1]}"
        return "unknown"

    def supports_feature(self, feature: str) -> bool:
        """Check if backend supports specific feature."""
        if not self._current_device:
            return False

        major, minor = self._current_device.compute_capability

        features = {
            'tensor_cores': major >= 7,  # Volta and later
            'fp16': major >= 5,  # Maxwell and later
            'fp64': major >= 1,  # All support double precision
            'int8': major >= 6,  # Pascal and later
            'async_copy': major >= 8,  # Ampere and later
            'cooperative_groups': major >= 6,
        }

        return features.get(feature.lower(), False)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _prepare_kernel_code(self, source_code: str) -> str:
        """Prepare kernel code with necessary includes."""
        includes = """
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>

#define FULL_MASK 0xffffffff

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

"""
        if "#include" not in source_code:
            return includes + source_code
        return source_code

    def _prepare_kernel_args(self, args: List[Any]) -> List[Any]:
        """Convert arguments to CUDA-compatible format."""
        cuda_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                # Allocate and copy
                d_arg = self.allocate_memory(arg.nbytes)
                self.copy_to_device(arg, d_arg)
                cuda_args.append(d_arg)
            elif hasattr(arg, '__cuda_array_interface__'):
                # Already a CUDA array (from PyTorch, CuPy, etc.)
                cuda_args.append(arg)
            else:
                # Scalar value
                cuda_args.append(arg)
        return cuda_args

    def _parse_compilation_errors(self, error_msg: str) -> List[str]:
        """Parse compilation error messages."""
        errors = []
        lines = error_msg.split('\n')

        for line in lines:
            if 'error:' in line.lower():
                errors.append(line.strip())

        return errors if errors else [error_msg]

    def _calculate_occupancy(
        self,
        threads_per_block: int,
        registers_per_thread: int,
        shared_memory_bytes: int
    ) -> float:
        """Calculate theoretical occupancy."""
        device = self._current_device

        # Warp size
        warp_size = device.warp_size

        # Warps per block
        warps_per_block = (threads_per_block + warp_size - 1) // warp_size

        # Max warps per SM
        max_warps_per_sm = device.max_threads_per_block // warp_size

        # Register limit
        max_regs_per_sm = device.max_registers_per_block
        if registers_per_thread > 0:
            blocks_per_sm_regs = max_regs_per_sm // (registers_per_thread * threads_per_block)
        else:
            blocks_per_sm_regs = float('inf')

        # Shared memory limit
        max_shared_per_sm = device.max_shared_memory_per_block
        if shared_memory_bytes > 0:
            blocks_per_sm_smem = max_shared_per_sm // shared_memory_bytes
        else:
            blocks_per_sm_smem = float('inf')

        # Max blocks per SM
        blocks_per_sm = min(blocks_per_sm_regs, blocks_per_sm_smem, 32)  # 32 is HW limit

        # Calculate occupancy
        active_warps = min(blocks_per_sm * warps_per_block, max_warps_per_sm)
        occupancy = active_warps / max_warps_per_sm

        return occupancy

    def get_memory_info(self) -> Tuple[float, float]:
        """Get memory info for current device."""
        if not self._initialized:
            self.initialize()

        free_bytes, total_bytes = self._cuda.mem_get_info()
        free_mb = free_bytes / (1024 ** 2)
        total_mb = total_bytes / (1024 ** 2)

        return (free_mb, total_mb)
