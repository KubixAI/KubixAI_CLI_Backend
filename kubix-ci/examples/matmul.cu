/**
 * Matrix multiplication kernel for gpuci testing.
 *
 * C = A * B where A is MxK, B is KxN, C is MxN
 * Uses shared memory tiling for better performance.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

__global__ void matmul(const float* A, const float* B, float* C,
                       int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    // Matrix dimensions
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    // Initialize matrices
    for (int i = 0; i < M * K; i++) h_A[i] = 0.01f * (i % 100);
    for (int i = 0; i < K * N; i++) h_B[i] = 0.01f * (i % 100);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // ========== WARMUP ==========
    GPUCI_WARMUP_START()
    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    GPUCI_WARMUP_END()

    gpuci_check_error("Warmup");

    // ========== BENCHMARK ==========
    GPUCI_BENCHMARK_START()
    matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    GPUCI_BENCHMARK_END()

    gpuci_check_error("Benchmark");
    gpuci_print_results();

    // Calculate GFLOPS
    // 2 * M * N * K floating point operations
    // (multiply-add counts as 2 ops)
    double gflops = (2.0 * M * N * K) / 1e9;
    printf("GPUCI_GFLOPS_THEORETICAL=%.2f\n", gflops);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
