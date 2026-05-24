#include "matmul_warp.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv)
{
    int M = 512;
    int K = 512;
    int N = 512;

    if (argc == 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    }

    if (M <= 0 || K <= 0 || N <= 0) {
        return EXIT_FAILURE;
    }

    const std::size_t size_a = static_cast<std::size_t>(M) * K;
    const std::size_t size_b = static_cast<std::size_t>(K) * N;
    const std::size_t size_c = static_cast<std::size_t>(M) * N;

    std::vector<float> h_A(size_a);
    std::vector<float> h_B(size_b);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : h_A) {
        value = dist(rng);
    }
    for (float& value : h_B) {
        value = dist(rng);
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size_a * sizeof(float));
    cudaMalloc(&d_B, size_b * sizeof(float));
    cudaMalloc(&d_C, size_c * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_a * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_b * sizeof(float), cudaMemcpyHostToDevice);

    matmul_warp_launch(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return EXIT_SUCCESS;
}
