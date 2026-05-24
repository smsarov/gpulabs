cmake_minimum_required(VERSION 3.18)
project(gpulabs_matmul LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(matmul_warp
    src/main.cu
    src/matmul_warp.cu
)

target_include_directories(matmul_warp PRIVATE include)

if(MSVC)
    target_compile_options(matmul_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>)
    set_target_properties(matmul_warp PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        VS_DEBUGGER_COMMAND_ARGUMENTS "512 512 512")
endif()
