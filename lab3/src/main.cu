#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

constexpr int FLASH_D_MAX = 64;

void flash_attention_launch(
    const float* d_Q,
    const float* d_K,
    const float* d_V,
    float* d_O,
    int N,
    int d,
    cudaStream_t stream = 0);

// Algorithm 0 из статьи: S = QK^T, P = softmax(S), O = PV.
// CPU-версия только для проверки корректности flash-реализации.
void attention_reference_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N,
    int d)
{
    std::vector<float> scores(static_cast<std::size_t>(N));
    for (int row = 0; row < N; ++row) {
        float row_max = -INFINITY;
        for (int col = 0; col < N; ++col) {
            float score = 0.0f;
            for (int k = 0; k < d; ++k) {
                score += Q[row * d + k] * K[col * d + k];
            }
            scores[static_cast<std::size_t>(col)] = score;
            row_max = fmaxf(row_max, score);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            scores[static_cast<std::size_t>(col)] = expf(scores[static_cast<std::size_t>(col)] - row_max);
            row_sum += scores[static_cast<std::size_t>(col)];
        }

        for (int k = 0; k < d; ++k) {
            float value = 0.0f;
            for (int col = 0; col < N; ++col) {
                value += (scores[static_cast<std::size_t>(col)] / row_sum) * V[col * d + k];
            }
            O[row * d + k] = value;
        }
    }
}

int main(int argc, char** argv)
{
    int N = 512;
    int d = 64;

    if (argc == 3) {
        N = std::atoi(argv[1]);
        d = std::atoi(argv[2]);
    }

    if (N <= 0 || d <= 0 || d > FLASH_D_MAX) {
        std::fprintf(stderr, "Usage: flash_attention [N d], require 0 < d <= %d\n", FLASH_D_MAX);
        return EXIT_FAILURE;
    }

    const std::size_t qkv_size = static_cast<std::size_t>(N) * d;

    std::vector<float> h_Q(qkv_size);
    std::vector<float> h_K(qkv_size);
    std::vector<float> h_V(qkv_size);
    std::vector<float> h_O_ref(qkv_size);
    std::vector<float> h_O_flash(qkv_size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < qkv_size; ++i) {
        h_Q[i] = dist(rng);
        h_K[i] = dist(rng);
        h_V[i] = dist(rng);
    }

    attention_reference_cpu(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), N, d);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_O = nullptr;
    cudaMalloc(&d_Q, qkv_size * sizeof(float));
    cudaMalloc(&d_K, qkv_size * sizeof(float));
    cudaMalloc(&d_V, qkv_size * sizeof(float));
    cudaMalloc(&d_O, qkv_size * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    flash_attention_launch(d_Q, d_K, d_V, d_O, N, d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_O_flash.data(), d_O, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < qkv_size; ++i) {
        max_diff = fmaxf(max_diff, fabsf(h_O_flash[i] - h_O_ref[i]));
    }

    std::printf("N=%d, d=%d\n", N, d);
    std::printf("max |O_flash - O_ref| = %.6e\n", max_diff);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    return (max_diff < 1e-3f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
