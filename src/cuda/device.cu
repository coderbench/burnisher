// Device query and the PROBE: measure what this part actually sustains.
//
// This file exists because of one line in every roofline the repository publishes:
// `peak_basis`. A ceiling computed against a vendor peak nobody reaches understates every
// achieved fraction by the same factor, and the consequence is directional and bad -- it tells a
// contributor there is more room in a cell than there is. Telling somebody there is 55% left
// when there is 8% is exactly how a subnet loses a contributor, and it is the failure mode this
// repository is built to avoid.
//
// So `burnish probe` measures two numbers on the part itself and rewrites `peak_basis` to
// `measured`:
//
//   sustained bandwidth   a grid-stride read+write over a working set far larger than L2
//   achievable GEMM rate  a large, well-shaped bf16 GEMM through cuBLASLt
//
// The second is deliberately "what a well-tuned GEMM achieves" rather than "what the ALUs could
// theoretically issue". A roofline is only useful if a contributor could in principle reach it,
// and nothing in this pipeline will beat a tuned vendor GEMM on a square problem.
//
// NOT COMPILED. There was no CUDA toolkit and no Blackwell device available when this was
// written. docs/STATUS.md says so in those words and the CI job `cuda-compile` exists to make it
// stop being true. Treat every line here as unverified until that job is green.
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <map>
#include <type_traits>
#include <utility>

namespace {

// Whether this toolkit's cudaDeviceProp still has `clockRate` (CUDA 13 removed it). Detected by
// the compiler rather than a version macro, so the probe builds against any toolkit.
template <typename P, typename = void>
struct HasClockRate : std::false_type {};
template <typename P>
struct HasClockRate<P, std::void_t<decltype(std::declval<const P&>().clockRate)>>
    : std::true_type {};

template <typename P>
void clock_json(char* out, size_t n, const P& p, std::true_type) {
    std::snprintf(out, n, "%d", static_cast<int>(p.clockRate));
}
template <typename P>
void clock_json(char* out, size_t n, const P&, std::false_type) {
    std::snprintf(out, n, "null");
}

}  // namespace
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "burnisher/device.h"

namespace burnisher {
namespace device {
namespace {

// Allocation is tracked HERE rather than read from the driver. `nvidia-smi` and
// `cudaMemGetInfo` both include the CUDA context and the allocator's own slack -- hundreds of
// megabytes a candidate cannot influence and should not be scored on. The frontier's memory
// objective is what the runtime asked for.
std::mutex g_mutex;
size_t g_live = 0;
size_t g_peak = 0;

// Freed blocks, kept by exact size and handed back to the next allocation of that size.
//
// A forward pass allocates the same shapes in the same order every time, and the VAE allocates a
// full-resolution activation for nearly every op. Returning each one to the driver cost more than
// the decode's convolutions: `cudaFree` of a large block blocks, and four decodes spent over a
// second in it. Reuse is safe because every kernel runs on the one default stream, so work queued
// against a block's previous owner finishes before anything queued against its next one.
//
// Only while memory is plentiful. Kernel launches and the vendor libraries allocate device memory
// that never passes through here, so a cache that held every freed size ran an fp32 generation --
// 22 GB of weights -- out of memory in a launch, where no retry could reach it. A block is kept
// only while this much stays free, and a new allocation that leaves less returns the whole cache.
constexpr size_t kCacheHeadroom = 4ull << 30;

size_t device_free_bytes() {
    size_t free_b = 0, total_b = 0;
    return cudaMemGetInfo(&free_b, &total_b) == cudaSuccess ? free_b : 0;
}

// Deliberately leaked, for the same reason as the size table below.
std::multimap<size_t, void*>& free_blocks() {
    static std::multimap<size_t, void*>* m = new std::multimap<size_t, void*>();
    return *m;
}

std::map<void*, size_t>& size_table() {
    // Deliberately leaked. `release` runs from a shared_ptr deleter, and a tensor can outlive
    // static destruction -- at which point a function-local static map has already been
    // destroyed and the erase corrupts the heap. The symptom is "corrupted double-linked list"
    // during teardown, which looks like a kernel bug and is not one.
    static std::map<void*, size_t>* t = new std::map<void*, size_t>();
    return *t;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what + ": " + cudaGetErrorString(e));
    }
}

}  // namespace

bool available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Return every cached block to the driver. Called when an allocation fails with blocks cached.
void release_cache() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& kv : free_blocks()) cudaFree(kv.second);
    free_blocks().clear();
}

void* alloc(size_t bytes) {
    void* p = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = free_blocks().find(bytes);
        if (it != free_blocks().end()) {
            p = it->second;
            free_blocks().erase(it);
        }
    }
    if (!p) {
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            release_cache();
            check(cudaMalloc(&p, bytes), "cudaMalloc");
        }
        if (device_free_bytes() < kCacheHeadroom) release_cache();
    }
    // ZEROED, to match the host allocator.
    //
    // `cudaMalloc` does not zero and `calloc` does, so without this the same code produces zeros
    // on the CPU and garbage on the GPU -- and only where something relied on the initial value,
    // which is the hardest possible thing to find. It costs a memset per intermediate and buying
    // that back (by not allocating per call at all) is a real optimisation, not a correctness
    // question.
    check(cudaMemset(p, 0, bytes), "cudaMemset");
    std::lock_guard<std::mutex> lock(g_mutex);
    g_live += bytes;
    g_peak = std::max(g_peak, g_live);
    // The size is recorded against the pointer so `release` can subtract it and cache the block
    // under its size. Cached blocks are not live: the memory objective is what the runtime asked
    // for, not what the cache happens to be holding.
    size_table()[p] = bytes;
    return p;
}

void release(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = size_table().find(p);
    if (it == size_table().end()) {
        cudaFree(p);
        return;
    }
    g_live -= std::min(g_live, it->second);
    if (device_free_bytes() >= kCacheHeadroom) {
        free_blocks().emplace(it->second, p);
    } else {
        cudaFree(p);
    }
    size_table().erase(it);
}

void copy_to_device(void* dst, const void* src, size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "memcpy H2D");
}

void copy_to_host(void* dst, const void* src, size_t bytes) {
    check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "memcpy D2H");
}

void synchronize() { check(cudaDeviceSynchronize(), "synchronize"); }

void* alloc_pinned(size_t bytes) {
    void* p = nullptr;
    check(cudaMallocHost(&p, bytes), "cudaMallocHost");
    return p;
}

void release_pinned(void* p) {
    if (p) cudaFreeHost(p);
}

size_t allocated_bytes() { std::lock_guard<std::mutex> l(g_mutex); return g_live; }
size_t peak_allocated_bytes() { std::lock_guard<std::mutex> l(g_mutex); return g_peak; }
void reset_peak() { std::lock_guard<std::mutex> l(g_mutex); g_peak = g_live; }

}  // namespace device

namespace {

#define CU_CHECK(expr)                                                              \
    do {                                                                            \
        cudaError_t _e = (expr);                                                    \
        if (_e != cudaSuccess) {                                                    \
            std::fprintf(stderr, "!! %s:%d %s -> %s\n", __FILE__, __LINE__, #expr,  \
                         cudaGetErrorString(_e));                                   \
            return 1;                                                               \
        }                                                                           \
    } while (0)

// Read one array and write another, grid-stride. float4 so a warp's access is a full 128-byte
// transaction; anything narrower measures the addressing rather than the memory system.
__global__ void stream_copy(const float4* __restrict__ src, float4* __restrict__ dst,
                            size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

double time_kernel(void (*launch)(void*), void* arg, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    launch(arg);                       // warm up: the first launch pays for module load
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch(arg);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(ms) / 1000.0 / iters;
}

struct CopyArgs {
    const float4* src;
    float4* dst;
    size_t n4;
    int blocks, threads;
};

void launch_copy(void* p) {
    CopyArgs* a = static_cast<CopyArgs*>(p);
    stream_copy<<<a->blocks, a->threads>>>(a->src, a->dst, a->n4);
}

}  // namespace

int probe_device_main() {
    int device = 0;
    CU_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CU_CHECK(cudaGetDeviceProperties(&prop, device));

    // The working set has to be much larger than L2, or this measures the cache. 1 GiB per
    // buffer is comfortably past 96 MiB and still fits beside a model on a 32 GB part.
    const size_t bytes = 1ull << 30;
    const size_t n4 = bytes / sizeof(float4);
    float4 *src = nullptr, *dst = nullptr;
    CU_CHECK(cudaMalloc(&src, bytes));
    CU_CHECK(cudaMalloc(&dst, bytes));
    CU_CHECK(cudaMemset(src, 1, bytes));

    CopyArgs args{src, dst, n4, 0, 256};
    args.blocks = prop.multiProcessorCount * 32;
    const double copy_s = time_kernel(launch_copy, &args, 20);
    // Read one buffer and write the other: two bytes of traffic per byte copied.
    const double bandwidth = (2.0 * bytes) / copy_s;

    CU_CHECK(cudaFree(src));
    CU_CHECK(cudaFree(dst));

    // A large square bf16 GEMM with fp32 accumulate -- the accumulate mode a diffusion GEMM
    // actually uses, and the one the published bf16 peak in configs/devices.json claims. Getting
    // the mode wrong makes the roofline wrong by 2x on the same silicon.
    const int N = 8192;
    __nv_bfloat16 *A = nullptr, *B = nullptr;
    float* C = nullptr;
    CU_CHECK(cudaMalloc(&A, sizeof(__nv_bfloat16) * N * N));
    CU_CHECK(cudaMalloc(&B, sizeof(__nv_bfloat16) * N * N));
    CU_CHECK(cudaMalloc(&C, sizeof(float) * N * N));
    CU_CHECK(cudaMemset(A, 0, sizeof(__nv_bfloat16) * N * N));
    CU_CHECK(cudaMemset(B, 0, sizeof(__nv_bfloat16) * N * N));

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "!! cublasCreate failed\n");
        return 1;
    }
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    const float alpha = 1.0f, beta = 0.0f;
    const auto gemm_once = [&]() {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                     A, CUDA_R_16BF, N, B, CUDA_R_16BF, N, &beta,
                     C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    gemm_once();
    cudaDeviceSynchronize();
    cudaEvent_t ga, gb;
    cudaEventCreate(&ga);
    cudaEventCreate(&gb);
    cudaEventRecord(ga);
    const int gemm_iters = 20;
    for (int i = 0; i < gemm_iters; ++i) gemm_once();
    cudaEventRecord(gb);
    cudaEventSynchronize(gb);
    float gms = 0.0f;
    cudaEventElapsedTime(&gms, ga, gb);
    const double gemm_s = static_cast<double>(gms) / 1000.0 / gemm_iters;
    const double flops = 2.0 * static_cast<double>(N) * N * N / gemm_s;

    cublasDestroy(handle);
    CU_CHECK(cudaFree(A));
    CU_CHECK(cudaFree(B));
    CU_CHECK(cudaFree(C));
    cudaEventDestroy(ga);
    cudaEventDestroy(gb);

    // The part's clock rate is recorded, never scored. CUDA 13 removed it from cudaDeviceProp, and
    // reading it there broke the whole CUDA build on every box with the newer toolkit, so it is
    // reported as null where the toolkit no longer has it.
    char clock_khz[24];
    clock_json(clock_khz, sizeof clock_khz, prop,
               HasClockRate<std::decay_t<decltype(prop)>>{});

    // One BURNISH_JSON line, like every other measurement command. `basis` is "measured"
    // because a run produced it -- which is the one thing that entitles a number to that word.
    std::printf(
        "BURNISH_JSON: {\"basis\":\"measured\",\"device\":{\"name\":\"%s\",\"sm\":%d,"
        "\"cc\":\"%d.%d\",\"vram_bytes\":%zu,\"l2_bytes\":%d,"
        "\"persisting_l2_bytes\":%zu,\"clock_khz\":%s},"
        "\"measured\":{\"memory_bandwidth_gbs\":%.3f,\"bf16_tensor_tflops\":%.3f,"
        "\"gemm_shape\":%d,\"bandwidth_working_set_bytes\":%zu},"
        "\"_note\":\"bandwidth is a grid-stride read+write over a working set far larger than "
        "L2; the GEMM rate is a square bf16 GEMM with fp32 accumulate through cuBLAS -- what a "
        "well-tuned kernel achieves, not what the ALUs could issue. A roofline is only useful "
        "if a contributor could in principle reach it.\"}\n",
        prop.name, prop.multiProcessorCount, prop.major, prop.minor,
        prop.totalGlobalMem, prop.l2CacheSize,
        static_cast<size_t>(prop.accessPolicyMaxWindowSize), clock_khz,
        bandwidth / 1e9, flops / 1e12, N, bytes);

    std::fprintf(stderr,
        "\n>> Copy these into configs/devices.json with source \"measured\", then re-run\n"
        "   `burnish roofline`. Until you do, every achieved fraction in the published table\n"
        "   is computed against a VENDOR peak and is a LOWER bound on how done each cell is --\n"
        "   the real remaining room is smaller than the table implies, not larger.\n");
    return 0;
}

}  // namespace burnisher
