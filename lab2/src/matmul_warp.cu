#include "matmul_warp.cuh"

__inline__ __device__ float warpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void __launch_bounds__(BLOCK_THREADS) matmul_warp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int K,
    int N)
{
    __shared__ float A_tile[WARPS_PER_BLOCK][TILE_STRIDE];
    __shared__ float B_tile[TILE_DIM][WARPS_PER_BLOCK + 1];

    const int row = blockIdx.y * WARPS_PER_BLOCK + threadIdx.y;
    const int col = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;

    float sum = 0.0f;
    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int a_col = tile * TILE_DIM + threadIdx.x;
        const int b_row = tile * TILE_DIM + threadIdx.x;

        A_tile[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        B_tile[threadIdx.x][threadIdx.y] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        const float partial =
            A_tile[threadIdx.y][threadIdx.x] * B_tile[threadIdx.x][threadIdx.y];
        sum += warpReduceSum(partial);

        __syncthreads();
    }

    if (threadIdx.x == 0 && row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_warp_launch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    cudaStream_t stream)
{
    const dim3 block(TILE_DIM, WARPS_PER_BLOCK);
    const dim3 grid(
        (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    matmul_warp_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, K, N);
}
