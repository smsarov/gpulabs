#include "matmul_shared.cuh"

__global__ void __launch_bounds__(BLOCK_THREADS) matmul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N)
{
    __shared__ float As[TILE_DIM][TILE_STRIDE];
    __shared__ float Bs[TILE_DIM][TILE_STRIDE];

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int a_col = tile * TILE_DIM + threadIdx.x;
        const int b_row = tile * TILE_DIM + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_shared_launch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    cudaStream_t stream)
{
    const dim3 block(TILE_DIM, TILE_DIM);
    const dim3 grid(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM);

    matmul_shared_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N);
}
