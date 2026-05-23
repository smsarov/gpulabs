#pragma once

#include <cuda_runtime.h>

constexpr int TILE_DIM = 16;
constexpr int BLOCK_THREADS = TILE_DIM * TILE_DIM;

constexpr int TILE_STRIDE = TILE_DIM + 1;

static_assert(BLOCK_THREADS % 32 == 0, "block size must be a multiple of warp size (32)");

void matmul_shared_launch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    cudaStream_t stream = 0);
