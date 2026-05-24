#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <math_constants.h>

constexpr int FLASH_BR = 32;
constexpr int FLASH_BC = 32;
constexpr int FLASH_D_MAX = 64;

namespace {

__global__ void init_neg_inf_kernel(float* m, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        m[idx] = -CUDART_INF_F;
    }
}

__device__ float block_row_max(const float* row_scores, int count)
{
    float value = row_scores[0];
    for (int c = 1; c < count; ++c) {
        value = fmaxf(value, row_scores[c]);
    }
    return value;
}

__device__ float block_row_sum(const float* row_scores, int count)
{
    float value = 0.0f;
    for (int c = 0; c < count; ++c) {
        value += row_scores[c];
    }
    return value;
}

// Algorithm 1 (forward pass), Dao et al. 2022.
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ l,
    float* __restrict__ m,
    int N,
    int d)
{
    __shared__ float q_tile[FLASH_BR][FLASH_D_MAX];
    __shared__ float k_tile[FLASH_BC][FLASH_D_MAX];
    __shared__ float v_tile[FLASH_BC][FLASH_D_MAX];
    __shared__ float s_tile[FLASH_BR][FLASH_BC];

    const int q_block = blockIdx.x;
    const int row_base = q_block * FLASH_BR;
    if (row_base >= N) {
        return;
    }

    const int br = min(FLASH_BR, N - row_base);
    const int tc = (N + FLASH_BC - 1) / FLASH_BC;

    for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
        const int r = idx / d;
        const int k = idx % d;
        q_tile[r][k] = Q[(row_base + r) * d + k];
    }
    __syncthreads();

    for (int j = 0; j < tc; ++j) {
        const int col_base = j * FLASH_BC;
        const int bc = min(FLASH_BC, N - col_base);

        for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
            const int r = idx / d;
            const int k = idx % d;
            k_tile[r][k] = K[(col_base + r) * d + k];
            v_tile[r][k] = V[(col_base + r) * d + k];
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                sum += q_tile[r][k] * k_tile[c][k];
            }
            s_tile[r][c] = sum;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < br; r += blockDim.x) {
            const int global_row = row_base + r;

            const float m_i = m[global_row];
            const float l_i = l[global_row];

            const float m_tilde = block_row_max(&s_tile[r][0], bc);
            for (int c = 0; c < bc; ++c) {
                s_tile[r][c] = expf(s_tile[r][c] - m_tilde);
            }
            const float l_tilde = block_row_sum(&s_tile[r][0], bc);

            const float m_new = fmaxf(m_i, m_tilde);
            const float alpha = expf(m_i - m_new);
            const float beta = expf(m_tilde - m_new);
            const float l_new = alpha * l_i + beta * l_tilde;

            float* out_row = O + global_row * d;
            for (int k = 0; k < d; ++k) {
                float pv = 0.0f;
                for (int c = 0; c < bc; ++c) {
                    pv += s_tile[r][c] * v_tile[c][k];
                }
                out_row[k] = (alpha * l_i * out_row[k] + beta * pv) / l_new;
            }

            l[global_row] = l_new;
            m[global_row] = m_new;
        }
        __syncthreads();
    }
}

}  // namespace

void flash_attention_launch(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d,
    cudaStream_t stream)
{
    const int tr = (N + FLASH_BR - 1) / FLASH_BR;

    float *d_l = nullptr;
    float *d_m = nullptr;
    cudaMalloc(&d_l, static_cast<std::size_t>(N) * sizeof(float));
    cudaMalloc(&d_m, static_cast<std::size_t>(N) * sizeof(float));

    cudaMemsetAsync(d_O, 0, static_cast<std::size_t>(N) * d * sizeof(float), stream);
    cudaMemsetAsync(d_l, 0, static_cast<std::size_t>(N) * sizeof(float), stream);

    const int init_threads = 256;
    init_neg_inf_kernel<<<(N + init_threads - 1) / init_threads, init_threads, 0, stream>>>(d_m, N);

    flash_attention_kernel<<<tr, 256, 0, stream>>>(d_Q, d_K, d_V, d_O, d_l, d_m, N, d);
    cudaStreamSynchronize(stream);

    cudaFree(d_l);
    cudaFree(d_m);
}
