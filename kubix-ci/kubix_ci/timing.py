"""CUDA timing wrapper generation and output parsing.

This module wraps user CUDA kernels with accurate CUDA event-based timing.
Outputs structured KUBIXCI_* markers to stdout for parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TimingResult:
    """Parsed timing results from kernel execution."""

    status: str  # "success" or "error"
    error: str | None = None
    device_name: str | None = None
    compute_capability: str | None = None
    memory_mb: int | None = None
    driver_version: str | None = None
    runtime_version: str | None = None
    median_ms: float | None = None
    mean_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    runs: int | None = None
    all_times: list[float] | None = None


# Template for wrapping kernels that already have a main() function
# We inject timing around the kernel launch
TIMING_WRAPPER_STANDALONE = '''
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>

// === KUBIXCI TIMING INFRASTRUCTURE ===

void kubix_ci_print_device_info() {{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("KUBIXCI_DEVICE_NAME=%s\\n", prop.name);
    printf("KUBIXCI_COMPUTE_CAPABILITY=%d.%d\\n", prop.major, prop.minor);
    printf("KUBIXCI_MEMORY_MB=%lu\\n", prop.totalGlobalMem / (1024*1024));

    int driver_version, runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf("KUBIXCI_DRIVER_VERSION=%d.%d\\n", driver_version/1000, (driver_version%100)/10);
    printf("KUBIXCI_RUNTIME_VERSION=%d.%d\\n", runtime_version/1000, (runtime_version%100)/10);
}}

void kubix_ci_check_error(const char* msg) {{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        printf("KUBIXCI_STATUS=error\\n");
        printf("KUBIXCI_ERROR=%s: %s\\n", msg, cudaGetErrorString(err));
        exit(1);
    }}
}}

// Timing infrastructure
float kubix_ci_times[{benchmark_runs}];
int kubix_ci_time_idx = 0;

void kubix_ci_record_time(float ms) {{
    if (kubix_ci_time_idx < {benchmark_runs}) {{
        kubix_ci_times[kubix_ci_time_idx++] = ms;
    }}
}}

void kubix_ci_print_results() {{
    if (kubix_ci_time_idx == 0) {{
        printf("KUBIXCI_STATUS=error\\n");
        printf("KUBIXCI_ERROR=No timing data collected\\n");
        return;
    }}

    // Calculate stats
    float total = 0, min_t = FLT_MAX, max_t = 0;
    for (int i = 0; i < kubix_ci_time_idx; i++) {{
        total += kubix_ci_times[i];
        if (kubix_ci_times[i] < min_t) min_t = kubix_ci_times[i];
        if (kubix_ci_times[i] > max_t) max_t = kubix_ci_times[i];
    }}

    // Sort for median
    std::sort(kubix_ci_times, kubix_ci_times + kubix_ci_time_idx);
    float median = kubix_ci_times[kubix_ci_time_idx / 2];
    float mean = total / kubix_ci_time_idx;

    printf("KUBIXCI_STATUS=success\\n");
    printf("KUBIXCI_TIMING_MEDIAN_MS=%.6f\\n", median);
    printf("KUBIXCI_TIMING_MEAN_MS=%.6f\\n", mean);
    printf("KUBIXCI_TIMING_MIN_MS=%.6f\\n", min_t);
    printf("KUBIXCI_TIMING_MAX_MS=%.6f\\n", max_t);
    printf("KUBIXCI_TIMING_RUNS=%d\\n", kubix_ci_time_idx);

    printf("KUBIXCI_TIMING_ALL=");
    for (int i = 0; i < kubix_ci_time_idx; i++) {{
        printf("%.6f", kubix_ci_times[i]);
        if (i < kubix_ci_time_idx - 1) printf(",");
    }}
    printf("\\n");
}}

#define KUBIXCI_WARMUP_START() for (int _kubix_ci_w = 0; _kubix_ci_w < {warmup_runs}; _kubix_ci_w++) {{
#define KUBIXCI_WARMUP_END() cudaDeviceSynchronize(); }}

#define KUBIXCI_BENCHMARK_START() for (int _kubix_ci_i = 0; _kubix_ci_i < {benchmark_runs}; _kubix_ci_i++) {{ \\
    cudaEvent_t _kubix_ci_start, _kubix_ci_stop; \\
    cudaEventCreate(&_kubix_ci_start); \\
    cudaEventCreate(&_kubix_ci_stop); \\
    cudaEventRecord(_kubix_ci_start);

#define KUBIXCI_BENCHMARK_END() \\
    cudaEventRecord(_kubix_ci_stop); \\
    cudaEventSynchronize(_kubix_ci_stop); \\
    float _kubix_ci_ms = 0; \\
    cudaEventElapsedTime(&_kubix_ci_ms, _kubix_ci_start, _kubix_ci_stop); \\
    kubix_ci_record_time(_kubix_ci_ms); \\
    cudaEventDestroy(_kubix_ci_start); \\
    cudaEventDestroy(_kubix_ci_stop); \\
}}

// === END KUBIXCI TIMING INFRASTRUCTURE ===

// === USER KERNEL CODE ===
{user_kernel_code}
// === END USER KERNEL CODE ===
'''

# Full wrapper for kernels without main() - generates synthetic test harness
TIMING_WRAPPER_FULL = '''
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>

// === USER KERNEL CODE ===
{user_kernel_code}
// === END USER KERNEL CODE ===

void kubix_ci_print_device_info() {{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("KUBIXCI_DEVICE_NAME=%s\\n", prop.name);
    printf("KUBIXCI_COMPUTE_CAPABILITY=%d.%d\\n", prop.major, prop.minor);
    printf("KUBIXCI_MEMORY_MB=%lu\\n", prop.totalGlobalMem / (1024*1024));

    int driver_version, runtime_version;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf("KUBIXCI_DRIVER_VERSION=%d.%d\\n", driver_version/1000, (driver_version%100)/10);
    printf("KUBIXCI_RUNTIME_VERSION=%d.%d\\n", runtime_version/1000, (runtime_version%100)/10);
}}

int main() {{
    kubix_ci_print_device_info();

    // Default test configuration
    const int N = 1 << 20;  // 1M elements
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // Allocate memory
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        printf("KUBIXCI_STATUS=error\\n");
        printf("KUBIXCI_ERROR=Memory allocation failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }}

    // Initialize with some data
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup runs
    for (int w = 0; w < {warmup_runs}; w++) {{
        {kernel_name}<<<numBlocks, blockSize>>>(d_data, N);
        cudaDeviceSynchronize();
    }}

    err = cudaGetLastError();
    if (err != cudaSuccess) {{
        printf("KUBIXCI_STATUS=error\\n");
        printf("KUBIXCI_ERROR=Kernel execution failed: %s\\n", cudaGetErrorString(err));
        return 1;
    }}

    // Timed runs
    float times[{benchmark_runs}];
    float total = 0, min_t = FLT_MAX, max_t = 0;

    for (int i = 0; i < {benchmark_runs}; i++) {{
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        {kernel_name}<<<numBlocks, blockSize>>>(d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }}

    // Sort for median
    std::sort(times, times + {benchmark_runs});
    float median = times[{benchmark_runs} / 2];
    float mean = total / {benchmark_runs};

    printf("KUBIXCI_STATUS=success\\n");
    printf("KUBIXCI_TIMING_MEDIAN_MS=%.6f\\n", median);
    printf("KUBIXCI_TIMING_MEAN_MS=%.6f\\n", mean);
    printf("KUBIXCI_TIMING_MIN_MS=%.6f\\n", min_t);
    printf("KUBIXCI_TIMING_MAX_MS=%.6f\\n", max_t);
    printf("KUBIXCI_TIMING_RUNS=%d\\n", {benchmark_runs});

    printf("KUBIXCI_TIMING_ALL=");
    for (int i = 0; i < {benchmark_runs}; i++) {{
        printf("%.6f", times[i]);
        if (i < {benchmark_runs} - 1) printf(",");
    }}
    printf("\\n");

    // Cleanup
    cudaFree(d_data);
    free(h_data);

    return 0;
}}
'''


def has_main_function(source: str) -> bool:
    """Check if source contains a main() function."""
    # Match: int main, void main, with optional whitespace and parameters
    pattern = r'\b(int|void)\s+main\s*\('
    return bool(re.search(pattern, source))


def find_kernel_name(source: str) -> str | None:
    """Find the first __global__ kernel function name."""
    # Match: __global__ void kernel_name(
    pattern = r'__global__\s+\w+\s+(\w+)\s*\('
    match = re.search(pattern, source)
    return match.group(1) if match else None


def wrap_kernel_with_timing(
    kernel_source: str,
    warmup: int = 3,
    runs: int = 10,
) -> str:
    """
    Wrap a CUDA kernel source with timing infrastructure.

    Two modes:
    1. If kernel has main(): Inject timing macros for user to use
    2. If kernel is just __global__ function: Generate full test harness
    """
    if has_main_function(kernel_source):
        # User has their own main - provide macros they can use
        # For now, we still wrap with our infrastructure but preserve their main
        return TIMING_WRAPPER_STANDALONE.format(
            user_kernel_code=kernel_source,
            warmup_runs=warmup,
            benchmark_runs=runs,
        )
    else:
        # No main - generate synthetic test harness
        kernel_name = find_kernel_name(kernel_source)
        if not kernel_name:
            # Fallback: assume there's a kernel, try to compile anyway
            kernel_name = "kernel"

        return TIMING_WRAPPER_FULL.format(
            user_kernel_code=kernel_source,
            kernel_name=kernel_name,
            warmup_runs=warmup,
            benchmark_runs=runs,
        )


def parse_timing_output(stdout: str) -> TimingResult:
    """Parse KUBIXCI_* markers from stdout into TimingResult."""
    result = TimingResult(status="error")

    lines = stdout.strip().split('\n')

    for line in lines:
        if not line.startswith('KUBIXCI_'):
            continue

        if '=' not in line:
            continue

        key, value = line.split('=', 1)
        value = value.strip()

        if key == 'KUBIXCI_STATUS':
            result.status = value
        elif key == 'KUBIXCI_ERROR':
            result.error = value
        elif key == 'KUBIXCI_DEVICE_NAME':
            result.device_name = value
        elif key == 'KUBIXCI_COMPUTE_CAPABILITY':
            result.compute_capability = value
        elif key == 'KUBIXCI_MEMORY_MB':
            try:
                result.memory_mb = int(value)
            except ValueError:
                pass
        elif key == 'KUBIXCI_DRIVER_VERSION':
            result.driver_version = value
        elif key == 'KUBIXCI_RUNTIME_VERSION':
            result.runtime_version = value
        elif key == 'KUBIXCI_TIMING_MEDIAN_MS':
            try:
                result.median_ms = float(value)
            except ValueError:
                pass
        elif key == 'KUBIXCI_TIMING_MEAN_MS':
            try:
                result.mean_ms = float(value)
            except ValueError:
                pass
        elif key == 'KUBIXCI_TIMING_MIN_MS':
            try:
                result.min_ms = float(value)
            except ValueError:
                pass
        elif key == 'KUBIXCI_TIMING_MAX_MS':
            try:
                result.max_ms = float(value)
            except ValueError:
                pass
        elif key == 'KUBIXCI_TIMING_RUNS':
            try:
                result.runs = int(value)
            except ValueError:
                pass
        elif key == 'KUBIXCI_TIMING_ALL':
            try:
                result.all_times = [float(t) for t in value.split(',') if t]
            except ValueError:
                pass

    return result
