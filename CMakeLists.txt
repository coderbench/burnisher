# Burnisher: a native C++/CUDA image and video generation runtime for consumer Blackwell.
#
# CUDA is OPTIONAL and off unless asked for, so the whole CPU reference path, the op registry and
# the tests build and run on a machine with no GPU. That is not a convenience: the CPU reference
# is the correctness oracle, and an oracle that only builds where the thing it is checking builds
# is not much of an oracle.
cmake_minimum_required(VERSION 3.24)
project(Burnisher VERSION 0.1.0 LANGUAGES CXX)

option(BURNISHER_BUILD_CUDA "Build the CUDA backend" OFF)
option(BURNISHER_BUILD_TESTS "Build the test binaries" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# No fast-math, ever, and this is load-bearing rather than conservative. -ffast-math permits
# reassociation, which makes a reduction's result depend on how the compiler felt about
# vectorising it that day. The determinism gate requires byte-identical replays of the same
# build; reassociation does not break that on its own, but it does make the CPU reference and a
# CUDA kernel disagree for reasons that have nothing to do with the kernel, and then the
# tolerance has to be widened to admit the compiler.
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

set(BURNISHER_SOURCES
  src/core/tensor.cpp
  src/core/impl_select.cpp
  src/core/device_stub.cpp
  src/core/safetensors.cpp
  src/cpu/ops_cpu.cpp
  src/sched/dpm_solver.cpp
  src/models/weights.cpp
  src/models/t5_encoder.cpp
  src/models/pixart_dit.cpp
  src/models/vae_decoder.cpp
  src/core/pipeline.cpp
)

add_library(burnisher_core STATIC ${BURNISHER_SOURCES})
# Weight conversion runs on every host core (src/core/tensor.cpp).
find_package(Threads REQUIRED)
target_link_libraries(burnisher_core PUBLIC Threads::Threads)
target_include_directories(burnisher_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if(BURNISHER_BUILD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    # sm_120 is RTX 5090 / RTX PRO 6000 Blackwell; sm_121 is DGX Spark (GB10).
    set(CMAKE_CUDA_ARCHITECTURES 120)
  endif()
  # device.cu carries the allocator and the device boundary; ops_cuda.cu carries the kernels.
  # Both are gated behind the correctness gate before anything they produce is timed: shipping
  # unverified kernels as though they worked is the failure this repository exists to avoid.
  target_sources(burnisher_core PRIVATE src/cuda/device.cu src/cuda/ops_cuda.cu)
  target_compile_definitions(burnisher_core PUBLIC BURNISHER_CUDA=1)
  set_target_properties(burnisher_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  find_package(CUDAToolkit REQUIRED)
  # cuDNN is REQUIRED with CUDA, not optional. The `cuda` baseline's convolutions go through it,
  # and a build that quietly fell back to another kernel would be a different baseline on every
  # box that lacked it -- every score measured against it would be against the wrong thing.
  find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include x86_64-linux-gnu)
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64 lib)
  if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "The CUDA build needs cuDNN 9 and its headers. On Ubuntu, from NVIDIA's "
                        "CUDA repository: apt install libcudnn9-dev-cuda-<CUDA major>")
  endif()
  target_include_directories(burnisher_core PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(burnisher_core PUBLIC CUDA::cudart CUDA::cublas ${CUDNN_LIBRARY})
endif()

add_executable(burnisher tools/burnisher_main.cpp)
target_link_libraries(burnisher PRIVATE burnisher_core)

if(BURNISHER_BUILD_TESTS)
  enable_testing()
  foreach(t test_tensor test_ops test_scheduler test_models)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE burnisher_core)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
  if(BURNISHER_BUILD_CUDA)
    # The device kernels against the CPU oracle. Passes as skipped where there is no GPU.
    add_executable(test_cuda_ops tests/test_cuda_ops.cpp)
    target_link_libraries(test_cuda_ops PRIVATE burnisher_core)
    add_test(NAME test_cuda_ops COMMAND test_cuda_ops)
  endif()
  # Every entry point must answer --help. It is the cheapest possible check that a binary is not
  # simply broken, and it is the one that catches a bad refactor before a benchmark run does.
  add_test(NAME burnisher_help COMMAND burnisher --help)
  add_test(NAME burnisher_info COMMAND burnisher info)
endif()

install(TARGETS burnisher burnisher_core
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
