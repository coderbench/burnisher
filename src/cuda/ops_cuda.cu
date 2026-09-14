// The CUDA op backend: every op registered under the name "cuda", beside the CPU reference.
//
// Registered beside, never instead of. The CPU implementations are the correctness ORACLE and
// they stay runnable forever, so a kernel here can be diffed against them at any shape --
// `scripts/differential_test.py` does exactly that, and the CPU side is itself verified against
// the reference implementation.
//
// These kernels are written to be CORRECT and READABLE, not fast. That is deliberate and it is
// the whole premise of the repository: v0 ships a complete, slow pipeline and contributors make
// it fast. Every one of them is a starting point with an obvious next move, and the obvious next
// moves are the backlog:
//
//   gemm        goes through cuBLAS with a SEPARATE bias-and-activation pass. Fusing that
//               epilogue is `issues/fused-adaln.md` and is measurable the day it lands.
//   attention   one block per (batch, head, query) with a streaming softmax. No tiling, no
//               shared-memory staging, no tensor cores. `issues/dit-attention.md`.
//   norm        one block per row, a naive shared-memory reduction.
//   modulate    the canonical fusion target: two flops per element and a full round trip.
//
// A contributor who beats any of them registers a new name and the harness measures the
// difference in one process, one model load, one thermal state.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cudnn.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "burnisher/device.h"
#include "burnisher/ops.h"

namespace burnisher {
namespace {

constexpr int kBlock = 256;

// The attention kernel is templated on its tile size and registered under several names, so the
// choice is measured rather than argued. See register_cuda_ops().

#define CU_OK(expr)                                                                  \
    do {                                                                             \
        cudaError_t _e = (expr);                                                     \
        if (_e != cudaSuccess) {                                                     \
            throw std::runtime_error(std::string("cuda: ") + #expr + ": " +           \
                                     cudaGetErrorString(_e));                        \
        }                                                                            \
    } while (0)

void check_launch(const char* what) {
    // Checked after every launch. A kernel that failed asynchronously otherwise surfaces as a
    // wrong number in a later op, and the harness scores it.
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cuda: ") + what + " launch: " +
                                 cudaGetErrorString(e));
    }
}

// --- dtype access -----------------------------------------------------------------------
// Values are computed in float and stored through the tensor's dtype, so a bf16 kernel rounds
// exactly where the bf16 CPU reference rounds. A kernel that kept everything in fp32 internally
// would disagree with the oracle for a reason that has nothing to do with the kernel.

__device__ __forceinline__ float ld(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, int64_t i) {
    return __bfloat162float(p[i]);
}
__device__ __forceinline__ void st(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, int64_t i, float v) {
    p[i] = __float2bfloat16(v);
}

void require_device(const Tensor& t, const char* what, const char* operand = "operand") {
    if (t.device() != Device::CUDA) {
        throw std::runtime_error(
            std::string("cuda ") + what + ": " + operand + " " + t.describe() + " is a HOST "
            "tensor. Mixing host and device operands in one op is a fault, not a slow path.\n"
            "  The usual cause is a scratch tensor in a model graph declared without a device -- "
            "and multi-declarations (`Tensor a(...), b(...);`) are where they hide, because they "
            "do not look like the single-declaration form a search finds.");
    }
}

bool is_f32(const Tensor& t) { return t.dtype() == DType::F32; }

void require_same_dtype(const Tensor& a, const Tensor& b, const char* what) {
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error(std::string("cuda ") + what + ": operands differ in dtype (" +
                                 dtype_name(a.dtype()) + " vs " + dtype_name(b.dtype()) +
                                 "). Promoting silently would change where the rounding happens "
                                 "and therefore what the oracle comparison measures.");
    }
}

#define DISPATCH(t, NAME, ...)                                                       \
    do {                                                                             \
        if (is_f32(t)) { using NAME = float; __VA_ARGS__; }                           \
        else if ((t).dtype() == DType::BF16) { using NAME = __nv_bfloat16; __VA_ARGS__; } \
        else throw std::runtime_error(std::string("cuda: dtype ") +                   \
                                      dtype_name((t).dtype()) + " has no kernel");    \
    } while (0)

// --- elementwise ------------------------------------------------------------------------

template <typename T>
__global__ void k_add(const T* a, const T* b, T* out, int64_t outer, int64_t inner,
                      float scale, bool broadcast) {
    const int64_t n = outer * inner;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t bi = broadcast ? (i % inner) : i;
        st(out, i, ld(a, i) + scale * ld(b, bi));
    }
}

void add_cuda(const AddArgs& a) {
    require_device(*a.a, "add");
    require_same_dtype(*a.a, *a.b, "add");
    const int64_t n = a.outer * a.inner;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.a, T,
             k_add<T><<<grid, kBlock>>>((const T*)a.a->data(), (const T*)a.b->data(),
                                        (T*)a.out->data(), a.outer, a.inner, a.scale,
                                        a.broadcast_b));
    check_launch("add");
}

template <typename T>
__global__ void k_chunk(const T* mod, const T* table, T* out, int64_t batch, int64_t channels,
                        int64_t chunks, int64_t chunk, int64_t trow) {
    const int64_t n = batch * channels;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t b = i / channels, c = i % channels;
        st(out, i, ld(mod, b * chunks * channels + chunk * channels + c) +
                   ld(table, trow * channels + c));
    }
}

void chunk_cuda(const ChunkArgs& a) {
    require_device(*a.modulation, "chunk");
    const int64_t trow = (a.table_chunk >= 0) ? a.table_chunk : a.chunk;
    const int64_t n = a.batch * a.channels;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.modulation, T,
             k_chunk<T><<<grid, kBlock>>>((const T*)a.modulation->data(),
                                          (const T*)a.table->data(), (T*)a.out->data(),
                                          a.batch, a.channels, a.chunks, a.chunk, trow));
    check_launch("chunk");
}

__device__ __forceinline__ float gelu_tanh_d(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float silu_d(float x) { return x / (1.0f + __expf(-x)); }

template <typename T>
__global__ void k_activation(const T* x, const T* gate, T* out, int64_t n, int kind) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float v = ld(x, i);
        if (kind == 1) v = gelu_tanh_d(v);
        else if (kind == 2) v = silu_d(v);
        else if (kind == 3) v = gelu_tanh_d(v) * ld(gate, i);
        st(out, i, v);
    }
}

void activation_cuda(const ActivationArgs& a) {
    require_device(*a.x, "activation");
    int kind = 0;
    switch (a.kind) {
        case Epilogue::None: kind = 0; break;
        case Epilogue::Gelu: kind = 1; break;
        case Epilogue::Silu: kind = 2; break;
        case Epilogue::GeluGated: kind = 3; break;
    }
    if (kind == 3 && !a.gate) throw std::runtime_error("cuda activation: GeluGated needs a gate");
    const int grid = (int)std::min<int64_t>(65535, (a.numel + kBlock - 1) / kBlock);
    DISPATCH(*a.x, T,
             k_activation<T><<<grid, kBlock>>>((const T*)a.x->data(),
                                               a.gate ? (const T*)a.gate->data() : nullptr,
                                               (T*)a.out->data(), a.numel, kind));
    check_launch("activation");
}

template <typename T>
__global__ void k_modulate(const T* x, const T* scale, const T* shift, const T* gate,
                           const T* residual, T* out, int64_t batch, int64_t tokens,
                           int64_t channels) {
    const int64_t n = batch * tokens * channels;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t c = i % channels;
        const int64_t b = i / (tokens * channels);
        float v = ld(x, i) * (1.0f + ld(scale, b * channels + c)) + ld(shift, b * channels + c);
        if (gate) v *= ld(gate, b * channels + c);
        if (residual) v += ld(residual, i);
        st(out, i, v);
    }
}

void modulate_cuda(const ModulateArgs& a) {
    require_device(*a.x, "modulate");
    const int64_t n = a.batch * a.tokens * a.channels;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.x, T,
             k_modulate<T><<<grid, kBlock>>>(
                 (const T*)a.x->data(), (const T*)a.scale->data(), (const T*)a.shift->data(),
                 a.gate ? (const T*)a.gate->data() : nullptr,
                 a.residual ? (const T*)a.residual->data() : nullptr,
                 (T*)a.out->data(), a.batch, a.tokens, a.channels));
    check_launch("modulate");
}

// --- norms ------------------------------------------------------------------------------
// One block per row, a naive shared-memory reduction, fp32 accumulators. Deterministic: the
// reduction tree is fixed, so the same build reproduces itself, which the gate requires.

template <typename T>
__global__ void k_layernorm(const T* x, const T* weight, const T* bias, T* out,
                            int64_t cols, float eps, bool rms) {
    extern __shared__ float sdata[];
    const int64_t row = blockIdx.x;
    const T* xr = x + row * cols;
    T* orow = out + row * cols;

    float sum = 0.0f, sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = ld(xr, i);
        sum += v;
        sumsq += v * v;
    }
    sdata[threadIdx.x] = sum;
    sdata[blockDim.x + threadIdx.x] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            sdata[blockDim.x + threadIdx.x] += sdata[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    const float mean = rms ? 0.0f : sdata[0] / (float)cols;
    const float var = sdata[blockDim.x] / (float)cols - mean * mean;
    const float inv = rsqrtf(var + eps);
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = (ld(xr, i) - mean) * inv;
        if (weight) v *= ld(weight, i);
        if (bias) v += ld(bias, i);
        st(orow, i, v);
    }
}

template <typename T>
__global__ void k_groupnorm(const T* x, const T* weight, const T* bias, T* out,
                            int64_t cols, int64_t groups, int64_t channels, float eps) {
    extern __shared__ float sdata[];
    const int64_t row = blockIdx.x / groups;
    const int64_t g = blockIdx.x % groups;
    const int64_t per = cols / groups;
    const int64_t base = row * cols + g * per;
    const int64_t spatial = channels ? (cols / channels) : 1;

    float sum = 0.0f, sumsq = 0.0f;
    for (int64_t i = threadIdx.x; i < per; i += blockDim.x) {
        const float v = ld(x, base + i);
        sum += v;
        sumsq += v * v;
    }
    sdata[threadIdx.x] = sum;
    sdata[blockDim.x + threadIdx.x] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
            sdata[blockDim.x + threadIdx.x] += sdata[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    const float mean = sdata[0] / (float)per;
    const float var = sdata[blockDim.x] / (float)per - mean * mean;
    const float inv = rsqrtf(var + eps);
    for (int64_t i = threadIdx.x; i < per; i += blockDim.x) {
        float v = (ld(x, base + i) - mean) * inv;
        if (weight || bias) {
            const int64_t ch = (g * per + i) / spatial;
            if (weight) v *= ld(weight, ch);
            if (bias) v += ld(bias, ch);
        }
        st(out, base + i, v);
    }
}

void norm_cuda(const NormArgs& a) {
    require_device(*a.x, "norm");
    const int threads = 256;
    const size_t shmem = 2 * threads * sizeof(float);
    if (a.groups > 0) {
        const int blocks = (int)(a.rows * a.groups);
        DISPATCH(*a.x, T,
                 k_groupnorm<T><<<blocks, threads, shmem>>>(
                     (const T*)a.x->data(), a.weight ? (const T*)a.weight->data() : nullptr,
                     a.bias ? (const T*)a.bias->data() : nullptr, (T*)a.out->data(),
                     a.cols, a.groups, a.channels, a.eps));
    } else {
        DISPATCH(*a.x, T,
                 k_layernorm<T><<<(int)a.rows, threads, shmem>>>(
                     (const T*)a.x->data(), a.weight ? (const T*)a.weight->data() : nullptr,
                     a.bias ? (const T*)a.bias->data() : nullptr, (T*)a.out->data(),
                     a.cols, a.eps, a.rms));
    }
    check_launch("norm");
}

// --- patchify / unpatchify --------------------------------------------------------------

template <typename T>
__global__ void k_patch(const T* in, T* out, int64_t batch, int64_t channels, int64_t grid_n,
                        int64_t patch, bool inverse) {
    const int64_t H = grid_n * patch;
    const int64_t per_token = channels * patch * patch;
    const int64_t n = batch * channels * H * H;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < n;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t x = idx % H;
        const int64_t y = (idx / H) % H;
        const int64_t c = (idx / (H * H)) % channels;
        const int64_t b = idx / (H * H * channels);
        const int64_t ty = y / patch, ky = y % patch;
        const int64_t tx = x / patch, kx = x % patch;
        const int64_t token = ty * grid_n + tx;
        const int64_t img = ((b * channels + c) * H + y) * H + x;
        const int64_t tok = (b * grid_n * grid_n + token) * per_token +
                            (inverse ? ((ky * patch + kx) * channels + c)
                                     : ((c * patch + ky) * patch + kx));
        if (inverse) st(out, img, ld(in, tok));
        else         st(out, tok, ld(in, img));
    }
}

void patch_cuda(const PatchArgs& a) {
    require_device(*a.in, "patch");
    const int64_t H = a.grid * a.patch;
    const int64_t n = a.batch * a.channels * H * H;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.in, T,
             k_patch<T><<<grid, kBlock>>>((const T*)a.in->data(), (T*)a.out->data(),
                                          a.batch, a.channels, a.grid, a.patch, a.inverse));
    check_launch("patch");
}

// --- attention --------------------------------------------------------------------------
// One block per (batch, head, query row), streaming online softmax. Head-LAST layout, matching
// every projection GEMM in this runtime -- see the comment on AttentionArgs.
//
// No tiling, no shared-memory staging of K/V, no tensor cores. It is correct and it is slow, and
// it is the single largest opportunity in the repository: `issues/dit-attention.md` has the
// arithmetic.

template <typename T, int kTile>
__global__ void k_attention(const T* q, const T* k, const T* v, const T* bias,
                            const T* key_mask, T* out, int64_t heads, int64_t q_len,
                            int64_t kv_len, int64_t head_dim, float scale) {
    // One block per (batch, head, query row). Tiled online softmax, DETERMINISTIC.
    //
    // The first version of this kernel accumulated the weighted values with `atomicAdd` into
    // shared memory. Every contribution was correct and the SET of contributions was fixed, but
    // the ORDER was not -- and float addition is not associative, so two runs of the same build
    // produced different last bits. `burnish gate --determinism` requires byte-identical
    // replays, and a runtime that cannot reproduce itself cannot be a reference for anything.
    //
    // So: each tile of keys is scored into shared memory, and then each thread owns a fixed set
    // of head_dim channels and walks the tile in a fixed order. No atomics, no race, one
    // summation order. It is also faster, because atomics on a hot shared-memory address
    // serialise anyway.
    const int64_t row = blockIdx.x;
    const int64_t i = row % q_len;
    const int64_t h = (row / q_len) % heads;
    const int64_t b = row / (q_len * heads);

    extern __shared__ float shared[];
    float* scores = shared;                   // kTile
    float* red = shared + kTile;              // blockDim.x
    float* acc = shared + kTile + blockDim.x; // head_dim

    const int64_t qbase = ((b * q_len + i) * heads + h) * head_dim;

    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
    float run_max = -INFINITY;
    float run_den = 0.0f;
    __syncthreads();

    for (int64_t base = 0; base < kv_len; base += kTile) {
        const int64_t tile = min((int64_t)kTile, kv_len - base);

        for (int64_t t = threadIdx.x; t < tile; t += blockDim.x) {
            const int64_t j = base + t;
            if (key_mask && ld(key_mask, b * kv_len + j) == 0.0f) {
                scores[t] = -INFINITY;
                continue;
            }
            float s = 0.0f;
            const int64_t kb = ((b * kv_len + j) * heads + h) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) s += ld(q, qbase + d) * ld(k, kb + d);
            s *= scale;
            if (bias) s += ld(bias, (h * q_len + i) * kv_len + j);
            scores[t] = s;
        }
        __syncthreads();

        // Tile maximum, by a fixed reduction tree.
        float local = -INFINITY;
        for (int64_t t = threadIdx.x; t < tile; t += blockDim.x) local = fmaxf(local, scores[t]);
        red[threadIdx.x] = local;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        const float tile_max = red[0];
        __syncthreads();

        const float new_max = fmaxf(run_max, tile_max);
        const float rescale = (run_max == -INFINITY) ? 0.0f : __expf(run_max - new_max);

        // Weights for this tile, written back over the scores.
        for (int64_t t = threadIdx.x; t < tile; t += blockDim.x) {
            scores[t] = (scores[t] == -INFINITY) ? 0.0f : __expf(scores[t] - new_max);
        }
        __syncthreads();

        float local_den = 0.0f;
        for (int64_t t = threadIdx.x; t < tile; t += blockDim.x) local_den += scores[t];
        red[threadIdx.x] = local_den;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        const float tile_den = red[0];
        __syncthreads();

        // Each thread owns a fixed set of channels and walks the tile in index order. This is
        // the part that makes the kernel reproducible.
        for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float a_d = acc[d] * rescale;
            for (int64_t t = 0; t < tile; ++t) {
                const float w = scores[t];
                if (w == 0.0f) continue;
                const int64_t kb = ((b * kv_len + base + t) * heads + h) * head_dim;
                a_d += w * ld(v, kb + d);
            }
            acc[d] = a_d;
        }
        run_den = run_den * rescale + tile_den;
        run_max = new_max;
        __syncthreads();
    }

    const float denom = run_den > 0.0f ? run_den : 1.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        st(out, qbase + d, acc[d] / denom);
    }
}

template <int kTile>
void attention_cuda_tiled(const AttentionArgs& a) {
    require_device(*a.q, "attention", "q");
    const float scale = a.scale > 0.0f ? a.scale : rsqrtf((float)a.head_dim);
    const int threads = 128;
    const int64_t rows = a.batch * a.heads * a.q_len;
    if (rows > 2147483647LL) throw std::runtime_error("cuda attention: too many rows");
    // scores tile + reduction scratch + the accumulator. Fixed by the TILE, not by kv_len, so a
    // 16384-key VAE mid-block needs no more shared memory than a 300-key cross-attention.
    const size_t shmem = (kTile + threads + a.head_dim) * sizeof(float);
    if (shmem > 48 * 1024) {
        throw std::runtime_error("cuda attention: tile " + std::to_string(kTile) + " needs " +
                                 std::to_string(shmem) + " bytes of shared memory, over the "
                                 "48 kB default limit");
    }
    DISPATCH(*a.q, T,
             k_attention<T, kTile><<<(int)rows, threads, shmem>>>(
                 (const T*)a.q->data(), (const T*)a.k->data(), (const T*)a.v->data(),
                 a.bias ? (const T*)a.bias->data() : nullptr,
                 a.key_mask ? (const T*)a.key_mask->data() : nullptr,
                 (T*)a.out->data(), a.heads, a.q_len, a.kv_len, a.head_dim, scale));
    check_launch("attention");
}

// --- gemm -------------------------------------------------------------------------------
// cuBLAS for the matmul, then a SEPARATE pass for bias and activation.
//
// Separate on purpose. Fusing the epilogue into the GEMM is one of the cheapest real wins in
// this pipeline and it is a backlog item with its own arithmetic; shipping it already fused
// would remove the opportunity and, worse, remove the ability to measure it.

cublasHandle_t handle() {
    static cublasHandle_t h = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        }
    });
    return h;
}

template <typename T>
__global__ void k_bias_act(T* c, const T* bias, int64_t m, int64_t n, int kind) {
    const int64_t total = m * n;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x) {
        float v = ld(c, i);
        if (bias) v += ld(bias, i % n);
        if (kind == 1) v = gelu_tanh_d(v);
        else if (kind == 2) v = silu_d(v);
        st(c, i, v);
    }
}

template <typename T>
__global__ void k_mul(T* c, const T* g, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        st(c, i, ld(c, i) * ld(g, i));
    }
}

void raw_gemm(const Tensor& A, const Tensor& B, Tensor& C, int64_t m, int64_t n, int64_t k,
              bool b_transposed) {
    // cuBLAS is column-major and these tensors are row-major, so the call computes
    // C^T = B^T A^T with the operands swapped. Getting this wrong does not crash; it produces a
    // transposed result that still has the right shape.
    const cudaDataType dt = (A.dtype() == DType::F32) ? CUDA_R_32F : CUDA_R_16BF;
    const float alpha = 1.0f, beta = 0.0f;
    const int ldb = b_transposed ? (int)k : (int)n;
    const cublasStatus_t st = cublasGemmEx(
        handle(), b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)m, (int)k, &alpha,
        B.data(), dt, ldb,
        A.data(), dt, (int)k,
        &beta, C.data(), dt, (int)n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx failed with status " + std::to_string((int)st));
    }
}

void gemm_cuda(const GemmArgs& a) {
    require_device(*a.a, "gemm");
    require_same_dtype(*a.a, *a.b, "gemm");
    if (a.a->numel() < a.m * a.k || a.b->numel() < a.k * a.n || a.out->numel() < a.m * a.n) {
        throw std::runtime_error("cuda gemm: operand smaller than its declared shape");
    }
    raw_gemm(*a.a, *a.b, *a.out, a.m, a.n, a.k, a.b_transposed);

    const bool gated = (a.epilogue == Epilogue::GeluGated);
    int kind = 0;
    if (a.epilogue == Epilogue::Gelu || gated) kind = 1;
    else if (a.epilogue == Epilogue::Silu) kind = 2;

    const int64_t total = a.m * a.n;
    const int grid = (int)std::min<int64_t>(65535, (total + kBlock - 1) / kBlock);
    if (a.bias || kind) {
        DISPATCH(*a.out, T,
                 k_bias_act<T><<<grid, kBlock>>>((T*)a.out->data(),
                                                 a.bias ? (const T*)a.bias->data() : nullptr,
                                                 a.m, a.n, kind));
        check_launch("gemm epilogue");
    }
    if (gated) {
        if (!a.gate_b) throw std::runtime_error("cuda gemm: GeluGated needs gate_b");
        Tensor g({a.m, a.n}, a.out->dtype(), Device::CUDA);
        raw_gemm(*a.a, *a.gate_b, g, a.m, a.n, a.k, a.b_transposed);
        if (a.gate_bias) {
            DISPATCH(g, T,
                     k_bias_act<T><<<grid, kBlock>>>((T*)g.data(),
                                                     (const T*)a.gate_bias->data(),
                                                     a.m, a.n, 0));
            check_launch("gemm gate bias");
        }
        DISPATCH(*a.out, T,
                 k_mul<T><<<grid, kBlock>>>((T*)a.out->data(), (const T*)g.data(), total));
        check_launch("gemm gate mul");
    }
}

// --- attention through cuBLAS -----------------------------------------------------------
// Scores as one batched GEMM per batch row, a masked softmax that owns each query row, and the
// weighted values as a second batched GEMM.
//
// IN FLOAT, whatever the tensors store. Values are computed in float and rounded only when they
// are stored, which is the rule every kernel here follows and the one the CPU oracle defines. So a
// bf16 call reads Q, K and V into float, and only the output is rounded.
//
// fp32 is exact and bf16 may use TF32. The correctness gate and the fidelity objective are both
// fp32 runs against the fp32 reference, which runs with TF32 off, so an fp32 call computes plain
// fp32. A bf16 call is gated by neither, and TF32's fifteen mantissa bits are still twice the
// precision of the bf16 it stores; on this card they are also the tensor cores, and plain fp32
// attention GEMMs were a third of a denoising step.
//
// No reordering copy. The head-last layout [batch, seq, heads, head_dim] already holds every
// (batch, head) matrix in column-major form: for one batch row, column j of head h starts at
// `j * heads * head_dim + h * head_dim` and runs head_dim elements. So the leading dimension is
// heads * head_dim, the stride between heads is head_dim, and cuBLAS reads the projections in
// place.
//
// DETERMINISTIC. Both GEMMs are fixed cuBLAS algorithms over fixed shapes, and the softmax is
// cuDNN's, which has no algorithm to choose. The first version reduced each row on one GPU thread
// in index order, and that single kernel was three quarters of a denoising step.

cudnnHandle_t dnn();
void dnn_ok(cudnnStatus_t st, const char* what);

// A device buffer kept across calls. The score matrix is a gigabyte at the VAE mid-block, and a
// forward pass that allocated and zeroed it per call would spend its time in the allocator.
Tensor& scratch(int slot, int64_t numel, DType dtype) {
    static Tensor slots[8];
    Tensor& t = slots[slot];
    if (!t.defined() || t.dtype() != dtype || t.numel() < numel) {
        t = Tensor();
        t = Tensor({numel}, dtype, Device::CUDA);
    }
    return t;
}

template <typename T>
__global__ void k_to_float(const T* in, float* out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        out[i] = ld(in, i);
    }
}

// A bf16 operand as float, in scratch `slot`. An fp32 operand is used where it is.
template <typename T>
const float* as_float(const T* in, int64_t n, int slot) {
    if (sizeof(T) == sizeof(float)) return (const float*)in;
    Tensor& t = scratch(slot, n, DType::F32);
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    k_to_float<T><<<grid, kBlock>>>(in, (float*)t.data(), n);
    check_launch("attention upcast");
    return (const float*)t.data();
}

// The key mask as -inf and T5's additive bias, before the softmax. One element per thread: every
// element is independent, so there is no reduction here to keep in order.
template <typename T>
__global__ void k_from_float(const float* in, T* out, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        st(out, i, in[i]);
    }
}

template <typename T>
__global__ void k_mask_and_bias(float* scores, const T* bias, const T* key_mask, int64_t n,
                                int64_t kv_len, int64_t b) {
    for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n;
         e += (int64_t)gridDim.x * blockDim.x) {
        if (key_mask && ld(key_mask, b * kv_len + e % kv_len) == 0.0f) {
            scores[e] = -INFINITY;
        } else if (bias) {
            scores[e] += ld(bias, e);   // bias is [heads, q_len, kv_len], laid out as the scores are
        }
    }
}

// A row with every key masked comes out of the softmax as 0/0. It attends to nothing.
__global__ void k_nan_to_zero(float* w, int64_t n) {
    for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n;
         e += (int64_t)gridDim.x * blockDim.x) {
        if (w[e] != w[e]) w[e] = 0.0f;
    }
}

template <typename T>
__global__ void k_to_head_last(const float* in, T* out, int64_t heads, int64_t seq, int64_t dim,
                               int64_t base) {
    const int64_t n = heads * seq * dim;
    for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < n;
         e += (int64_t)gridDim.x * blockDim.x) {
        const int64_t h = e / (seq * dim);
        const int64_t i = (e / dim) % seq;
        const int64_t d = e % dim;
        st(out, base + (i * heads + h) * dim + d, in[e]);
    }
}

template <typename T>
void attention_blas(const AttentionArgs& a) {
    const int64_t H = a.heads, S = a.q_len, K = a.kv_len, D = a.head_dim;
    const cublasComputeType_t compute =
        (sizeof(T) == sizeof(float)) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
    const float scale = a.scale > 0.0f ? a.scale : rsqrtf((float)D);
    const float zero = 0.0f, one = 1.0f;
    const int lead = (int)(H * D);
    const float* scores = (const float*)scratch(0, H * S * K, DType::F32).data();
    const float* weights = (const float*)scratch(6, H * S * K, DType::F32).data();
    const float* values = (const float*)scratch(1, H * S * D, DType::F32).data();

    // One softmax over each row of the [heads * q_len, kv_len] score matrix.
    cudnnTensorDescriptor_t rows_desc;
    dnn_ok(cudnnCreateTensorDescriptor(&rows_desc), "softmax descriptor");
    dnn_ok(cudnnSetTensor4dDescriptor(rows_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, (int)(H * S),
                                      (int)K, 1, 1), "softmax shape");
    struct Release {
        cudnnTensorDescriptor_t d;
        ~Release() { cudnnDestroyTensorDescriptor(d); }
    } release{rows_desc};

    for (int64_t b = 0; b < a.batch; ++b) {
        const float* q = as_float((const T*)a.q->data() + b * S * H * D, S * H * D, 3);
        const float* k = as_float((const T*)a.k->data() + b * K * H * D, K * H * D, 4);
        const float* v = as_float((const T*)a.v->data() + b * K * H * D, K * H * D, 5);

        // scores[h][i * K + j] = scale * q(i) . k(j): column-major K x S = scale * K^T Q.
        cublasStatus_t st = cublasGemmStridedBatchedEx(
            handle(), CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)S, (int)D, &scale,
            k, CUDA_R_32F, lead, (long long)D, q, CUDA_R_32F, lead, (long long)D, &zero,
            (float*)scores, CUDA_R_32F, (int)K, (long long)(S * K), (int)H,
            compute, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuda attention: score GEMM failed, status " +
                                     std::to_string((int)st));
        }

        const int64_t n_scores = H * S * K;
        const int mgrid = (int)std::min<int64_t>(65535, (n_scores + kBlock - 1) / kBlock);
        if (a.bias || a.key_mask) {
            k_mask_and_bias<T><<<mgrid, kBlock>>>(
                (float*)scores, a.bias ? (const T*)a.bias->data() : nullptr,
                a.key_mask ? (const T*)a.key_mask->data() : nullptr, n_scores, K, b);
            check_launch("attention mask and bias");
        }
        dnn_ok(cudnnSoftmaxForward(dnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one,
                                   rows_desc, scores, &zero, rows_desc, (float*)weights),
               "softmax");
        if (a.key_mask) {
            k_nan_to_zero<<<mgrid, kBlock>>>((float*)weights, n_scores);
            check_launch("attention masked rows");
        }

        // values[h][i * D + d] = sum_j v(j, d) w(i, j): column-major D x S = V W.
        st = cublasGemmStridedBatchedEx(
            handle(), CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)S, (int)K, &one,
            v, CUDA_R_32F, lead, (long long)D, weights, CUDA_R_32F, (int)K, (long long)(S * K),
            &zero, (float*)values, CUDA_R_32F, (int)D, (long long)(S * D), (int)H,
            compute, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuda attention: value GEMM failed, status " +
                                     std::to_string((int)st));
        }

        const int64_t n = H * S * D;
        const int vgrid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
        k_to_head_last<T><<<vgrid, kBlock>>>(values, (T*)a.out->data(), H, S, D, b * S * H * D);
        check_launch("attention scatter");
    }
}

void attention_cuda_blas(const AttentionArgs& a) {
    require_device(*a.q, "attention", "q");
    require_same_dtype(*a.q, *a.k, "attention");
    require_same_dtype(*a.q, *a.v, "attention");
    if (a.heads * a.head_dim > 2147483647LL || a.kv_len > 2147483647LL ||
        a.q_len > 2147483647LL) {
        throw std::runtime_error("cuda attention: dimension over cuBLAS's int range");
    }
    DISPATCH(*a.q, T, attention_blas<T>(a));
}

// --- conv2d through cuDNN ---------------------------------------------------------------
// One plan per shape: descriptors, the algorithm and its workspace, chosen once.
//
// The algorithm is cuDNN's HEURISTIC choice, filtered to deterministic algorithms, and never
// `cudnnFindConvolutionForwardAlgorithm`: that one times candidates on the live device, so two
// processes can pick different algorithms and produce different last bits, which the
// determinism gate refuses.
//
// IN FLOAT, as attention is: cuDNN's legacy convolution API does not take bf16 at all, and
// computing in float and rounding only the stored result is the rule the CPU oracle defines
// anyway. So every plan is an fp32 plan, and a bf16 call reads its operands into float and rounds
// its output once. As in attention, fp32 is exact and a bf16 call may use TF32.

cudnnHandle_t dnn() {
    static cudnnHandle_t h = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (cudnnCreate(&h) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnCreate failed");
    });
    return h;
}

void dnn_ok(cudnnStatus_t st, const char* what) {
    if (st != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cudnn ") + what + ": " + cudnnGetErrorString(st));
    }
}

struct ConvPlan {
    cudnnTensorDescriptor_t x, y, bias;
    cudnnFilterDescriptor_t w;
    cudnnConvolutionDescriptor_t conv;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace;
};

const ConvPlan& conv_plan(const Conv2dArgs& a, int64_t h_out, int64_t w_out, bool tf32) {
    static std::map<std::vector<int64_t>, ConvPlan> plans;
    static std::mutex m;
    const std::vector<int64_t> key{a.batch, a.c_in, a.h_in, a.w_in, a.c_out, a.k, a.pad, tf32};
    std::lock_guard<std::mutex> lock(m);
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;

    const cudnnDataType_t dt = CUDNN_DATA_FLOAT;
    ConvPlan p{};
    dnn_ok(cudnnCreateTensorDescriptor(&p.x), "input descriptor");
    dnn_ok(cudnnCreateTensorDescriptor(&p.y), "output descriptor");
    dnn_ok(cudnnCreateTensorDescriptor(&p.bias), "bias descriptor");
    dnn_ok(cudnnCreateFilterDescriptor(&p.w), "filter descriptor");
    dnn_ok(cudnnCreateConvolutionDescriptor(&p.conv), "convolution descriptor");
    dnn_ok(cudnnSetTensor4dDescriptor(p.x, CUDNN_TENSOR_NCHW, dt, (int)a.batch, (int)a.c_in,
                                      (int)a.h_in, (int)a.w_in), "input shape");
    dnn_ok(cudnnSetTensor4dDescriptor(p.y, CUDNN_TENSOR_NCHW, dt, (int)a.batch, (int)a.c_out,
                                      (int)h_out, (int)w_out), "output shape");
    dnn_ok(cudnnSetTensor4dDescriptor(p.bias, CUDNN_TENSOR_NCHW, dt, 1, (int)a.c_out, 1, 1),
           "bias shape");
    dnn_ok(cudnnSetFilter4dDescriptor(p.w, dt, CUDNN_TENSOR_NCHW, (int)a.c_out, (int)a.c_in,
                                      (int)a.k, (int)a.k), "filter shape");
    // Cross-correlation, as the reference computes and as the direct kernel indexes.
    dnn_ok(cudnnSetConvolution2dDescriptor(p.conv, (int)a.pad, (int)a.pad, 1, 1, 1, 1,
                                           CUDNN_CROSS_CORRELATION, dt), "convolution");
    dnn_ok(cudnnSetConvolutionMathType(p.conv, tf32 ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH),
           "math type");

    cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int returned = 0;
    dnn_ok(cudnnGetConvolutionForwardAlgorithm_v7(dnn(), p.x, p.w, p.conv, p.y,
                                                  CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned,
                                                  perf), "algorithm heuristic");
    bool found = false;
    for (int i = 0; i < returned && !found; ++i) {
        if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].determinism == CUDNN_DETERMINISTIC) {
            p.algo = perf[i].algo;
            found = true;
        }
    }
    if (!found) {
        throw std::runtime_error("cudnn: no deterministic forward algorithm for this convolution");
    }
    dnn_ok(cudnnGetConvolutionForwardWorkspaceSize(dnn(), p.x, p.w, p.conv, p.y, p.algo,
                                                   &p.workspace), "workspace size");
    return plans.emplace(key, p).first->second;
}

template <typename T>
void conv2d_dnn(const Conv2dArgs& a) {
    const int64_t h_out = a.h_in + 2 * a.pad - a.k + 1;
    const int64_t w_out = a.w_in + 2 * a.pad - a.k + 1;
    const int64_t n_out = a.batch * a.c_out * h_out * w_out;
    const ConvPlan& p = conv_plan(a, h_out, w_out, sizeof(T) != sizeof(float));
    const float* x = as_float((const T*)a.x->data(), a.batch * a.c_in * a.h_in * a.w_in, 3);
    const float* w = as_float((const T*)a.weight->data(), a.c_out * a.c_in * a.k * a.k, 4);
    const float* bias = a.bias ? as_float((const T*)a.bias->data(), a.c_out, 5) : nullptr;
    float* y = (sizeof(T) == sizeof(float)) ? (float*)a.out->data()
                                            : (float*)scratch(7, n_out, DType::F32).data();
    Tensor& workspace = scratch(2, (int64_t)(p.workspace / 4) + 1, DType::F32);
    const float one = 1.0f, zero = 0.0f;
    dnn_ok(cudnnConvolutionForward(dnn(), &one, p.x, x, p.w, w, p.conv, p.algo, workspace.data(),
                                   p.workspace, &zero, p.y, y), "convolution");
    if (bias) dnn_ok(cudnnAddTensor(dnn(), &one, p.bias, bias, &one, p.y, y), "bias");
    if (sizeof(T) != sizeof(float)) {
        const int grid = (int)std::min<int64_t>(65535, (n_out + kBlock - 1) / kBlock);
        k_from_float<T><<<grid, kBlock>>>(y, (T*)a.out->data(), n_out);
        check_launch("conv2d round");
    }
}

void conv2d_cuda_dnn(const Conv2dArgs& a) {
    require_device(*a.x, "conv2d");
    require_same_dtype(*a.x, *a.weight, "conv2d");
    DISPATCH(*a.x, T, conv2d_dnn<T>(a));
}

// --- conv2d -----------------------------------------------------------------------------
// Direct convolution, one thread per output element. No im2col, no implicit GEMM, no tensor
// cores. The VAE's shapes are few and fixed, which is exactly what makes a specialised path
// plausible -- `issues/vae-decode.md`.

template <typename T>
__global__ void k_conv2d(const T* x, const T* w, const T* bias, T* out, int64_t batch,
                         int64_t c_in, int64_t h_in, int64_t w_in, int64_t c_out, int64_t ksz,
                         int64_t pad, int64_t h_out, int64_t w_out) {
    const int64_t n = batch * c_out * h_out * w_out;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < n;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t ox = idx % w_out;
        const int64_t oy = (idx / w_out) % h_out;
        const int64_t oc = (idx / (w_out * h_out)) % c_out;
        const int64_t b = idx / (w_out * h_out * c_out);
        float acc = bias ? ld(bias, oc) : 0.0f;
        for (int64_t ic = 0; ic < c_in; ++ic) {
            for (int64_t ky = 0; ky < ksz; ++ky) {
                const int64_t iy = oy + ky - pad;
                if (iy < 0 || iy >= h_in) continue;
                for (int64_t kx = 0; kx < ksz; ++kx) {
                    const int64_t ix = ox + kx - pad;
                    if (ix < 0 || ix >= w_in) continue;
                    acc += ld(x, ((b * c_in + ic) * h_in + iy) * w_in + ix) *
                           ld(w, ((oc * c_in + ic) * ksz + ky) * ksz + kx);
                }
            }
        }
        st(out, idx, acc);
    }
}

void conv2d_cuda(const Conv2dArgs& a) {
    require_device(*a.x, "conv2d");
    const int64_t h_out = a.h_in + 2 * a.pad - a.k + 1;
    const int64_t w_out = a.w_in + 2 * a.pad - a.k + 1;
    const int64_t n = a.batch * a.c_out * h_out * w_out;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.x, T,
             k_conv2d<T><<<grid, kBlock>>>(
                 (const T*)a.x->data(), (const T*)a.weight->data(),
                 a.bias ? (const T*)a.bias->data() : nullptr, (T*)a.out->data(),
                 a.batch, a.c_in, a.h_in, a.w_in, a.c_out, a.k, a.pad, h_out, w_out));
    check_launch("conv2d");
}

template <typename T>
__global__ void k_scale(const T* in, T* out, int64_t n, float s) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        st(out, i, ld(in, i) * s);
    }
}

void scale_cuda(const ScaleArgs& a) {
    require_device(*a.in, "scale", "input");
    const int grid = (int)std::min<int64_t>(65535, (a.numel + kBlock - 1) / kBlock);
    DISPATCH(*a.in, T, k_scale<T><<<grid, kBlock>>>((const T*)a.in->data(),
                                                    (T*)a.out->data(), a.numel, a.scale));
    check_launch("scale");
}

template <typename T>
__global__ void k_repeat(const T* in, T* out, int64_t outer, int64_t inner) {
    const int64_t n = outer * inner;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        st(out, i, ld(in, i % inner));
    }
}

void repeat_cuda(const RepeatArgs& a) {
    require_device(*a.in, "repeat", "input");
    const int64_t n = a.outer * a.inner;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.in, T, k_repeat<T><<<grid, kBlock>>>((const T*)a.in->data(),
                                                     (T*)a.out->data(), a.outer, a.inner));
    check_launch("repeat");
}

template <typename T>
__global__ void k_guidance(const T* pred, T* out, int64_t out_channels, int64_t channels,
                           int64_t spatial, float scale) {
    const int64_t n = channels * spatial;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t c = i / spatial, s = i % spatial;
        const float u = ld(pred, (0 * out_channels + c) * spatial + s);
        const float k = ld(pred, (1 * out_channels + c) * spatial + s);
        st(out, i, u + scale * (k - u));
    }
}

void guidance_cuda(const GuidanceArgs& a) {
    require_device(*a.prediction, "guidance", "prediction");
    const int64_t n = a.channels * a.spatial;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.prediction, T,
             k_guidance<T><<<grid, kBlock>>>((const T*)a.prediction->data(),
                                             (T*)a.out->data(), a.out_channels, a.channels,
                                             a.spatial, a.scale));
    check_launch("guidance");
}

// The ids are ALWAYS fp32 and the table is whatever the model computes in, so this kernel is
// the one place in the backend with two operands of different dtypes -- and it is templated on
// only one of them.
//
// That is not a hypothetical. This kernel originally cast the ids pointer to the TABLE's type.
// At fp32 the two agree and everything works; at bf16 it read fp32 ids as bf16, produced
// garbage token ids, and the text encoder returned embeddings for the wrong tokens. The whole
// pipeline then disagreed with the reference by a relative L2 of 1.2 -- and every fp32 test in
// the repository passed, because fp32 is exactly the case where the bug cannot appear.
//
// So the id type is fixed and separate, and the args carry an explicit contract.
template <typename T>
__global__ void k_gather(const T* table, const float* ids, T* out, int64_t rows, int64_t cols) {
    const int64_t n = rows * cols;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / cols, c = i % cols;
        const int64_t id = (int64_t)ids[r];
        st(out, i, ld(table, id * cols + c));
    }
}

void gather_cuda(const GatherArgs& a) {
    require_device(*a.table, "gather", "table");
    require_device(*a.ids, "gather", "ids");
    if (a.ids->dtype() != DType::F32) {
        throw std::runtime_error(
            std::string("cuda gather: ids are ") + dtype_name(a.ids->dtype()) + ", not fp32. "
            "The ids and the table are the one pair of operands in this backend with DIFFERENT "
            "dtypes, and reading one as the other produces wrong token ids rather than an "
            "error -- which is a whole model's worth of wrong answer that fp32 testing cannot "
            "see.");
    }
    const int64_t n = a.rows * a.cols;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.table, T,
             k_gather<T><<<grid, kBlock>>>((const T*)a.table->data(),
                                           (const float*)a.ids->data(),
                                           (T*)a.out->data(), a.rows, a.cols));
    check_launch("gather");
}

template <typename T>
__global__ void k_upsample(const T* in, T* out, int64_t batch, int64_t channels, int64_t h_in,
                           int64_t w_in, int64_t factor) {
    const int64_t H = h_in * factor, W = w_in * factor;
    const int64_t n = batch * channels * H * W;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t x = i % W;
        const int64_t y = (i / W) % H;
        const int64_t c = (i / (W * H)) % channels;
        const int64_t b = i / (W * H * channels);
        st(out, i, ld(in, ((b * channels + c) * h_in + y / factor) * w_in + x / factor));
    }
}

void upsample_cuda(const UpsampleArgs& a) {
    require_device(*a.in, "upsample");
    const int64_t H = a.h_in * a.factor, W = a.w_in * a.factor;
    const int64_t n = a.batch * a.channels * H * W;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.in, T,
             k_upsample<T><<<grid, kBlock>>>((const T*)a.in->data(), (T*)a.out->data(),
                                             a.batch, a.channels, a.h_in, a.w_in, a.factor));
    check_launch("upsample");
}

template <typename T>
__global__ void k_transpose(const T* in, T* out, int64_t batch, int64_t channels,
                            int64_t spatial, bool to_channels_first) {
    const int64_t n = batch * channels * spatial;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t s = i % spatial;
        const int64_t c = (i / spatial) % channels;
        const int64_t b = i / (spatial * channels);
        const int64_t cf = (b * channels + c) * spatial + s;
        const int64_t sf = (b * spatial + s) * channels + c;
        if (to_channels_first) st(out, cf, ld(in, sf));
        else                   st(out, sf, ld(in, cf));
    }
}

void transpose_cuda(const TransposeArgs& a) {
    require_device(*a.in, "transpose");
    const int64_t n = a.batch * a.channels * a.spatial;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.in, T,
             k_transpose<T><<<grid, kBlock>>>((const T*)a.in->data(), (T*)a.out->data(),
                                              a.batch, a.channels, a.spatial,
                                              a.to_channels_first));
    check_launch("transpose");
}

}  // namespace

void register_cuda_ops() {
    static bool done = false;
    if (done) return;
    done = true;
    register_impl<GemmArgs>("gemm", "cuda", gemm_cuda,
                            "cuBLAS matmul with a SEPARATE bias/activation pass; fusing that "
                            "epilogue is issues/fused-adaln.md");
    // Three tile sizes, registered as three names.
    //
    // This is what the registry is FOR, and it is the smallest honest demonstration of the whole
    // mechanism: three kernels that compute the same thing, differing in one constant, A/B'd in
    // one process against one another. The tile decides how much shared memory a block holds and
    // how many synchronisation points a row costs, and which one wins is a question for the
    // hardware rather than for an argument.
    register_impl<AttentionArgs>("attention", "cuda", attention_cuda_tiled<256>,
                                 "tiled online softmax, 256 keys per tile, deterministic; "
                                 "no tensor cores -- issues/dit-attention.md");
    register_impl<AttentionArgs>("attention", "cuda-vendor", attention_cuda_blas,
                                 "cuBLAS batched scores and values, one masked softmax per row");
    register_impl<Conv2dArgs>("conv2d", "cuda-vendor", conv2d_cuda_dnn,
                              "cuDNN, deterministic heuristic algorithm, bias via cudnnAddTensor");
    register_impl<AttentionArgs>("attention", "cuda-tile64", attention_cuda_tiled<64>,
                                 "the same kernel at 64 keys per tile: less shared memory, more "
                                 "synchronisation points per row");
    register_impl<AttentionArgs>("attention", "cuda-tile1024", attention_cuda_tiled<1024>,
                                 "the same kernel at 1024 keys per tile: fewer barriers, four "
                                 "times the shared memory, fewer blocks resident");
    register_impl<NormArgs>("norm", "cuda", norm_cuda,
                            "one block per row, naive shared-memory reduction, fp32 accumulators");
    register_impl<ModulateArgs>("modulate", "cuda", modulate_cuda,
                                "unfused AdaLN modulation: a full activation round trip for two "
                                "flops per element. THE fusion target");
    register_impl<ActivationArgs>("activation", "cuda", activation_cuda, "elementwise");
    register_impl<Conv2dArgs>("conv2d", "cuda", conv2d_cuda,
                              "direct convolution, one thread per output element; "
                              "issues/vae-decode.md");
    register_impl<AddArgs>("add", "cuda", add_cuda, "residual and broadcast add");
    register_impl<ChunkArgs>("chunk", "cuda", chunk_cuda, "AdaLN-single modulation chunks");
    register_impl<PatchArgs>("patch", "cuda", patch_cuda, "patchify and unpatchify");
    register_impl<GatherArgs>("gather", "cuda", gather_cuda, "embedding row gather");
    register_impl<ScaleArgs>("scale", "cuda", scale_cuda, "one multiply per element");
    register_impl<RepeatArgs>("repeat", "cuda", repeat_cuda, "guidance-batch replication");
    register_impl<GuidanceArgs>("guidance", "cuda", guidance_cuda, "classifier-free guidance");
    register_impl<UpsampleArgs>("upsample", "cuda", upsample_cuda, "nearest 2x, NCHW");
    register_impl<TransposeArgs>("transpose", "cuda", transpose_cuda,
                                 "channels-major <-> tokens-major");
}

}  // namespace burnisher
