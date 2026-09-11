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

#include <mutex>
#include <stdexcept>
#include <string>

#include "burnisher/device.h"
#include "burnisher/ops.h"

namespace burnisher {
namespace {

constexpr int kBlock = 256;

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

template <typename T>
__global__ void k_attention(const T* q, const T* k, const T* v, const T* bias,
                            const T* key_mask, T* out, int64_t heads, int64_t q_len,
                            int64_t kv_len, int64_t head_dim, float scale) {
    const int64_t row = blockIdx.x;                 // flattened (batch, head, query)
    const int64_t i = row % q_len;
    const int64_t h = (row / q_len) % heads;
    const int64_t b = row / (q_len * heads);

    extern __shared__ float shared[];
    float* acc = shared;                            // head_dim
    float* red = shared + head_dim;                 // blockDim.x

    const int64_t qbase = ((b * q_len + i) * heads + h) * head_dim;

    // Pass one: the row maximum, for a numerically stable softmax.
    float local_max = -INFINITY;
    for (int64_t j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (key_mask && ld(key_mask, b * kv_len + j) == 0.0f) continue;
        float s = 0.0f;
        const int64_t kb = ((b * kv_len + j) * heads + h) * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) s += ld(q, qbase + d) * ld(k, kb + d);
        s *= scale;
        if (bias) s += ld(bias, (h * q_len + i) * kv_len + j);
        local_max = fmaxf(local_max, s);
    }
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float row_max = red[0] == -INFINITY ? 0.0f : red[0];
    __syncthreads();

    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
    __syncthreads();

    // Pass two: weights and the weighted sum. Accumulated into shared memory with atomics on a
    // fixed-size buffer -- deterministic because every contribution is added in float and the
    // set of contributions is fixed; the ORDER varies, which is why this kernel must be checked
    // for determinism rather than assumed (`burnish gate --determinism`).
    float local_denom = 0.0f;
    for (int64_t j = threadIdx.x; j < kv_len; j += blockDim.x) {
        if (key_mask && ld(key_mask, b * kv_len + j) == 0.0f) continue;
        float s = 0.0f;
        const int64_t kb = ((b * kv_len + j) * heads + h) * head_dim;
        for (int64_t d = 0; d < head_dim; ++d) s += ld(q, qbase + d) * ld(k, kb + d);
        s *= scale;
        if (bias) s += ld(bias, (h * q_len + i) * kv_len + j);
        const float w = __expf(s - row_max);
        local_denom += w;
        for (int64_t d = 0; d < head_dim; ++d) atomicAdd(&acc[d], w * ld(v, kb + d));
    }
    red[threadIdx.x] = local_denom;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float denom = red[0] > 0.0f ? red[0] : 1.0f;
    __syncthreads();
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        st(out, qbase + d, acc[d] / denom);
    }
}

void attention_cuda(const AttentionArgs& a) {
    require_device(*a.q, "attention");
    const float scale = a.scale > 0.0f ? a.scale : rsqrtf((float)a.head_dim);
    const int threads = 128;
    const int64_t rows = a.batch * a.heads * a.q_len;
    if (rows > 2147483647LL) throw std::runtime_error("cuda attention: too many rows");
    const size_t shmem = (a.head_dim + threads) * sizeof(float);
    DISPATCH(*a.q, T,
             k_attention<T><<<(int)rows, threads, shmem>>>(
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

template <typename T>
__global__ void k_gather(const T* table, const T* ids, T* out, int64_t rows, int64_t cols) {
    const int64_t n = rows * cols;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int64_t r = i / cols, c = i % cols;
        const int64_t id = (int64_t)ld(ids, r);
        st(out, i, ld(table, id * cols + c));
    }
}

void gather_cuda(const GatherArgs& a) {
    require_device(*a.table, "gather");
    const int64_t n = a.rows * a.cols;
    const int grid = (int)std::min<int64_t>(65535, (n + kBlock - 1) / kBlock);
    DISPATCH(*a.table, T,
             k_gather<T><<<grid, kBlock>>>((const T*)a.table->data(), (const T*)a.ids->data(),
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
    register_impl<AttentionArgs>("attention", "cuda", attention_cuda,
                                 "one block per query row, streaming softmax, no tiling and no "
                                 "tensor cores; issues/dit-attention.md");
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
