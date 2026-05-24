#pragma once

#include <cuda_runtime.h>

constexpr int TILE_DIM = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int BLOCK_THREADS = TILE_DIM * WARPS_PER_BLOCK;

constexpr int TILE_STRIDE = TILE_DIM + 1;

static_assert(BLOCK_THREADS % 32 == 0, "block size must be a multiple of warp size (32)");

void matmul_warp_launch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    cudaStream_t stream = 0);
