// The op surface. Every one of these is a registry with at least one implementation.
//
// Shapes are explicit and destinations are explicit. There is no broadcasting engine, because a
// broadcasting engine is a layer between a contributor and the arithmetic they are trying to
// make faster.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "burnisher/registry.h"
#include "burnisher/tensor.h"

namespace burnisher {

enum class Epilogue {
    None,
    Gelu,       // tanh approximation, matching the pinned reference
    Silu,
    GeluGated,  // out = gelu(x @ w0) * (x @ w1); the T5 gated-gelu FFN
};

// C[M,N] = A[M,K] @ B[K,N] (+ bias[N]), with an optional fused activation.
//
// The epilogue is part of the op rather than a separate elementwise pass on purpose: fusing it
// is one of the cheapest real wins in this pipeline, and an op surface that made it a separate
// kernel would have hidden the opportunity behind a refactor.
struct GemmArgs {
    const Tensor* a;
    const Tensor* b;
    const Tensor* bias;     // may be null
    Tensor* out;
    int64_t m, n, k;
    // The checkpoint stores every Linear as [out_features, in_features] and the reference
    // computes `x @ W^T`. Materialising a transpose per call would be a copy of the weights on
    // every forward pass -- for the text encoder that is 9.5 GB per prompt -- so the layout is
    // a flag the kernel honours instead. Getting this wrong does not crash; it produces a
    // plausible image from a transposed weight matrix.
    bool b_transposed = false;
    Epilogue epilogue = Epilogue::None;
    const Tensor* gate_b = nullptr;   // second weight matrix for GeluGated
    const Tensor* gate_bias = nullptr;
};

// Scaled dot-product attention over [batch, heads, seq, head_dim].
//
// `materialize` selects between writing the full score matrix and streaming it. Both are
// registered, because the difference between them is one of the published cells: the VAE
// mid-block's score matrix is 16384x16384, about a gigabyte, and a contributor needs to be able
// to measure the naive path to show that removing it helped.
struct AttentionArgs {
    const Tensor* q;
    const Tensor* k;
    const Tensor* v;
    Tensor* out;
    int64_t batch, heads, q_len, kv_len, head_dim;
    float scale = 0.0f;          // 0 means 1/sqrt(head_dim)
    const Tensor* bias = nullptr;  // additive, broadcast over batch; T5's relative position bias
};

// LayerNorm over the last dimension. `weight`/`bias` may be null (PixArt's norms are affine-free
// and take their scale and shift from the AdaLN modulation instead).
struct NormArgs {
    const Tensor* x;
    const Tensor* weight;
    const Tensor* bias;
    Tensor* out;
    int64_t rows, cols;
    float eps = 1e-6f;
    bool rms = false;            // T5 uses RMSNorm: no mean subtraction, scale only
    int64_t groups = 0;          // >0 selects GroupNorm over [rows, groups, cols/groups]
    // GroupNorm only: how many channels `cols` contains, so the per-channel affine can be
    // applied. `cols` is channels*spatial for an NCHW activation, and without this the op
    // cannot tell 512 channels of 128x128 from 8192 channels of 8x8.
    int64_t channels = 0;
};

// out = x * (1 + scale) + shift, with scale and shift broadcast per (batch, channel).
//
// This is AdaLN modulation, and it is the canonical fusion target in this pipeline: two flops
// per element against a full activation round trip. It is its own op so that a fused version can
// be registered beside the unfused one and the two compared directly.
struct ModulateArgs {
    const Tensor* x;
    const Tensor* scale;
    const Tensor* shift;
    Tensor* out;
    int64_t batch, tokens, channels;
    const Tensor* residual = nullptr;  // when set: out = residual + gate * modulated
    const Tensor* gate = nullptr;
};

struct ActivationArgs {
    const Tensor* x;
    Tensor* out;
    int64_t numel;
    Epilogue kind;
    const Tensor* gate = nullptr;   // for GeluGated: out = gelu(x) * gate
};

// 2D convolution, NCHW, square kernel, stride 1, `pad` on every side.
struct Conv2dArgs {
    const Tensor* x;
    const Tensor* weight;      // [c_out, c_in, k, k]
    const Tensor* bias;        // may be null
    Tensor* out;
    int64_t batch, c_in, h_in, w_in, c_out, k, pad;
};

using GemmRegistry = OpRegistry<GemmArgs>;
using AttentionRegistry = OpRegistry<AttentionArgs>;
using NormRegistry = OpRegistry<NormArgs>;
using ModulateRegistry = OpRegistry<ModulateArgs>;
using ActivationRegistry = OpRegistry<ActivationArgs>;
using Conv2dRegistry = OpRegistry<Conv2dArgs>;

// Which registered implementation each op uses for this forward pass. Carried explicitly rather
// than read from a global so that two arms could in principle run in one process without one of
// them changing the other's configuration underneath it.
struct ImplSelection {
    std::string gemm = "stock";
    std::string attention = "stock";
    std::string norm = "stock";
    std::string modulate = "stock";
    std::string activation = "stock";
    std::string conv2d = "stock";
    static ImplSelection from_request(const std::string& requested);
    std::map<std::string, std::string> as_map() const;
};

// Every registered implementation across every op, for `burnisher info --impls`.
struct OpListing {
    std::string op;
    std::vector<ImplInfo> impls;
};
std::vector<OpListing> list_all_impls();

// Resolve an implementation name for one op, honouring a per-op override.
//
// A submission selects `--impl fused-adaln`, which applies to whichever ops register that name
// and leaves the rest on their default. That is what lets one flag A/B a single kernel without
// silently changing the rest of the pipeline underneath the comparison.
std::string resolve_impl(const std::string& op, const std::string& requested);

// True when at least one op registers `name`. A name NO op registers is a typo or a kernel that
// failed to register, and resolving it to `stock` everywhere while reporting success would
// measure a configuration nobody asked for.
bool any_op_has_impl(const std::string& name);

void register_builtin_cpu_ops();
#ifdef BURNISHER_CUDA
void register_cuda_ops();
#endif

}  // namespace burnisher
