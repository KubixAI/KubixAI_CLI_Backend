/**
 * Simple vector addition kernel for gpuci testing.
 *
 * This kernel adds two vectors element-wise:
 *   C[i] = A[i] + B[i]
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 24;  // 16M elements
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch configuration
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // ========== WARMUP ==========
    GPUCI_WARMUP_START()
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    GPUCI_WARMUP_END()

    // Check for errors after warmup
    gpuci_check_error("Warmup");

    // ========== BENCHMARK ==========
    GPUCI_BENCHMARK_START()
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    GPUCI_BENCHMARK_END()

    // Check for errors after benchmark
    gpuci_check_error("Benchmark");

    // Print results
    gpuci_print_results();

    // Verify result (spot check)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }
    printf("GPUCI_VERIFY=%s\n", correct ? "pass" : "fail");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
