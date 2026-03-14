/**
 * Minimal kernel example - no main() function.
 *
 * gpuci will auto-generate a test harness for this kernel.
 * Useful for quick kernel benchmarking without boilerplate.
 */

__global__ void simple_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Simple computation: square each element
        data[idx] = data[idx] * data[idx];
    }
}
