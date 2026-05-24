cmake_minimum_required(VERSION 3.18)
project(gpulabs_flash_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(flash_attention
    src/main.cu
    src/flash_attention.cu
)
