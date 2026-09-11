// Reference implementations of every op, on the CPU.
//
// These are the ORACLE, not the product. They are written to be obviously correct and are not
// written to be fast: a contributor's CUDA kernel is compared against these, so a clever CPU
// implementation that was subtly wrong would make every correctness comparison downstream wrong
// in the same direction and pass every test.
//
// Two properties are load-bearing and are tested:
//
//   * DETERMINISM. Every reduction runs in a fixed order with an fp32 accumulator. No
//     parallel-reduction nondeterminism, no fast-math reassociation, no atomics. The same build
//     must reproduce itself byte-identically or it cannot be a reference for anything.
//   * ROUNDING AT THE SAME PLACES. Values are computed in float and stored through the tensor's
//     dtype, so a bf16 reference rounds where a bf16 kernel rounds. A CPU reference that kept
//     everything in fp32 internally and only converted at the end would make the tolerance
//     comparison measure the wrong thing.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "burnisher/ops.h"

namespace burnisher {
namespace {

inline float gelu_tanh(float x) {
    // The tanh approximation, because it is what the pinned reference uses. The exact erf form
    // differs from it by enough to matter over 28 layers, and picking the other one would show
    // up as a correctness failure nobody could explain.
    const float c = 0.7978845608028654f;  // sqrt(2/pi)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x3)));
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float apply_epilogue(Epilogue e, float v) {
    switch (e) {
        case Epilogue::None: return v;
        case Epilogue::Gelu: return gelu_tanh(v);
        case Epilogue::Silu: return silu(v);
        case Epilogue::GeluGated: return gelu_tanh(v);
    }
    return v;
}

void gemm_cpu(const GemmArgs& a) {
    const int64_t M = a.m, N = a.n, K = a.k;
    if (a.a->numel() < M * K || a.b->numel() < K * N || a.out->numel() < M * N) {
        // Checked rather than trusted: a shape slip here reads adjacent weights and produces a
        // perfectly plausible image.
        throw std::runtime_error("gemm: operand smaller than its declared shape");
    }
    const bool gated = (a.epilogue == Epilogue::GeluGated);
    if (gated && !a.gate_b) throw std::runtime_error("gemm: GeluGated needs gate_b");
    const auto bi = [&](int64_t p, int64_t j) {
        return a.b_transposed ? (j * K + p) : (p * N + j);
    };
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int64_t p = 0; p < K; ++p) {
                acc += a.a->get(i * K + p) * a.b->get(bi(p, j));
            }
            if (a.bias) acc += a.bias->get(j);
            float v = apply_epilogue(a.epilogue, acc);
            if (gated) {
                float g = 0.0f;
                for (int64_t p = 0; p < K; ++p) {
                    g += a.a->get(i * K + p) * a.gate_b->get(bi(p, j));
                }
                if (a.gate_bias) g += a.gate_bias->get(j);
                v *= g;
            }
            a.out->set(i * N + j, v);
        }
    }
}

// Streaming (flash-style) attention: the score row is formed, softmaxed and consumed without the
// full [q_len, kv_len] matrix ever existing. Numerically this is the online-softmax formulation,
// so it is stable for long rows and does not depend on kv_len for its memory.
void attention_streaming(const AttentionArgs& a) {
    const float scale = a.scale > 0.0f ? a.scale
                                       : 1.0f / std::sqrt(static_cast<float>(a.head_dim));
    // Head-LAST indexing: element (b, s, h, d) of [batch, seq, heads, head_dim].
    const auto qi = [&](int64_t b, int64_t s, int64_t h, int64_t d) {
        return ((b * a.q_len + s) * a.heads + h) * a.head_dim + d;
    };
    const auto ki = [&](int64_t b, int64_t s, int64_t h, int64_t d) {
        return ((b * a.kv_len + s) * a.heads + h) * a.head_dim + d;
    };
    std::vector<float> acc(a.head_dim);
    std::vector<float> scores(a.kv_len);
    for (int64_t b = 0; b < a.batch; ++b) {
        for (int64_t h = 0; h < a.heads; ++h) {
            for (int64_t i = 0; i < a.q_len; ++i) {
                float row_max = -INFINITY;
                for (int64_t j = 0; j < a.kv_len; ++j) {
                    float s = 0.0f;
                    for (int64_t d = 0; d < a.head_dim; ++d) {
                        s += a.q->get(qi(b, i, h, d)) * a.k->get(ki(b, j, h, d));
                    }
                    s *= scale;
                    if (a.bias) s += a.bias->get((h * a.q_len + i) * a.kv_len + j);
                    // -1e9 rather than -infinity: an all-masked row would make the softmax 0/0
                    // and NaN out the whole model. A finite sentinel degrades to a uniform row.
                    if (a.key_mask && a.key_mask->get(b * a.kv_len + j) == 0.0f) s = -1e9f;
                    scores[j] = s;
                    row_max = std::max(row_max, s);
                }
                float denom = 0.0f;
                for (int64_t j = 0; j < a.kv_len; ++j) {
                    scores[j] = std::exp(scores[j] - row_max);
                    denom += scores[j];
                }
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int64_t j = 0; j < a.kv_len; ++j) {
                    const float w = scores[j];
                    for (int64_t d = 0; d < a.head_dim; ++d) {
                        acc[d] += w * a.v->get(ki(b, j, h, d));
                    }
                }
                for (int64_t d = 0; d < a.head_dim; ++d) {
                    a.out->set(qi(b, i, h, d), acc[d] / denom);
                }
            }
        }
    }
}

// The naive path: materialize the whole score matrix, then softmax it, then multiply.
//
// Registered on purpose rather than deleted. It is what a first implementation does, the VAE
// mid-block's version of it writes about a gigabyte at 1024px, and a contributor removing it
// needs something to measure against. Deleting the slow path is how a repository loses the
// ability to demonstrate that the fast one helped.
void attention_materialized(const AttentionArgs& a) {
    const float scale = a.scale > 0.0f ? a.scale
                                       : 1.0f / std::sqrt(static_cast<float>(a.head_dim));
    const auto qi = [&](int64_t b, int64_t s, int64_t h, int64_t d) {
        return ((b * a.q_len + s) * a.heads + h) * a.head_dim + d;
    };
    const auto ki = [&](int64_t b, int64_t s, int64_t h, int64_t d) {
        return ((b * a.kv_len + s) * a.heads + h) * a.head_dim + d;
    };
    std::vector<float> mat(static_cast<size_t>(a.q_len) * a.kv_len);
    for (int64_t b = 0; b < a.batch; ++b) {
        for (int64_t h = 0; h < a.heads; ++h) {
            for (int64_t i = 0; i < a.q_len; ++i) {
                for (int64_t j = 0; j < a.kv_len; ++j) {
                    float s = 0.0f;
                    for (int64_t d = 0; d < a.head_dim; ++d) {
                        s += a.q->get(qi(b, i, h, d)) * a.k->get(ki(b, j, h, d));
                    }
                    s *= scale;
                    if (a.bias) s += a.bias->get((h * a.q_len + i) * a.kv_len + j);
                    if (a.key_mask && a.key_mask->get(b * a.kv_len + j) == 0.0f) s = -1e9f;
                    mat[static_cast<size_t>(i) * a.kv_len + j] = s;
                }
            }
            for (int64_t i = 0; i < a.q_len; ++i) {
                float* row = &mat[static_cast<size_t>(i) * a.kv_len];
                float m = -INFINITY;
                for (int64_t j = 0; j < a.kv_len; ++j) m = std::max(m, row[j]);
                float denom = 0.0f;
                for (int64_t j = 0; j < a.kv_len; ++j) { row[j] = std::exp(row[j] - m);
                                                         denom += row[j]; }
                for (int64_t j = 0; j < a.kv_len; ++j) row[j] /= denom;
            }
            for (int64_t i = 0; i < a.q_len; ++i) {
                for (int64_t d = 0; d < a.head_dim; ++d) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < a.kv_len; ++j) {
                        acc += mat[static_cast<size_t>(i) * a.kv_len + j] *
                               a.v->get(ki(b, j, h, d));
                    }
                    a.out->set(qi(b, i, h, d), acc);
                }
            }
        }
    }
}

void norm_cpu(const NormArgs& a) {
    if (a.groups > 0) {
        // GroupNorm over [rows=batch, channels=cols] with a spatial extent folded into `rows`
        // is wrong; the caller passes rows = batch and cols = channels*spatial, and the group
        // split is over channels. The spatial count is cols/channels and is implied.
        const int64_t per_group = a.cols / a.groups;
        if (a.channels && (a.channels % a.groups)) {
            throw std::runtime_error("groupnorm: channels not divisible by groups");
        }
        if (per_group * a.groups != a.cols) {
            throw std::runtime_error("groupnorm: channels*spatial not divisible by groups");
        }
        for (int64_t r = 0; r < a.rows; ++r) {
            for (int64_t g = 0; g < a.groups; ++g) {
                const int64_t base = r * a.cols + g * per_group;
                double mean = 0.0;
                for (int64_t i = 0; i < per_group; ++i) mean += a.x->get(base + i);
                mean /= per_group;
                double var = 0.0;
                for (int64_t i = 0; i < per_group; ++i) {
                    double d = a.x->get(base + i) - mean;
                    var += d * d;
                }
                var /= per_group;
                const float inv = 1.0f / std::sqrt(static_cast<float>(var) + a.eps);
                const int64_t spatial = a.channels ? (a.cols / a.channels) : 1;
                for (int64_t i = 0; i < per_group; ++i) {
                    float v = (a.x->get(base + i) - static_cast<float>(mean)) * inv;
                    if (a.weight || a.bias) {
                        // Channel of this element within the whole row, not within the group.
                        const int64_t ch = (g * per_group + i) / spatial;
                        if (a.weight) v *= a.weight->get(ch);
                        if (a.bias) v += a.bias->get(ch);
                    }
                    a.out->set(base + i, v);
                }
            }
        }
        return;
    }
    for (int64_t r = 0; r < a.rows; ++r) {
        const int64_t base = r * a.cols;
        double mean = 0.0;
        if (!a.rms) {
            for (int64_t i = 0; i < a.cols; ++i) mean += a.x->get(base + i);
            mean /= a.cols;
        }
        double var = 0.0;
        for (int64_t i = 0; i < a.cols; ++i) {
            double d = a.x->get(base + i) - mean;
            var += d * d;
        }
        var /= a.cols;
        const float inv = 1.0f / std::sqrt(static_cast<float>(var) + a.eps);
        for (int64_t i = 0; i < a.cols; ++i) {
            float v = (a.x->get(base + i) - static_cast<float>(mean)) * inv;
            if (a.weight) v *= a.weight->get(i);
            if (a.bias) v += a.bias->get(i);
            a.out->set(base + i, v);
        }
    }
}

void modulate_cpu(const ModulateArgs& a) {
    for (int64_t b = 0; b < a.batch; ++b) {
        for (int64_t t = 0; t < a.tokens; ++t) {
            const int64_t base = (b * a.tokens + t) * a.channels;
            for (int64_t c = 0; c < a.channels; ++c) {
                float v = a.x->get(base + c);
                v = v * (1.0f + a.scale->get(b * a.channels + c)) +
                    a.shift->get(b * a.channels + c);
                if (a.gate) v *= a.gate->get(b * a.channels + c);
                if (a.residual) v += a.residual->get(base + c);
                a.out->set(base + c, v);
            }
        }
    }
}

void activation_cpu(const ActivationArgs& a) {
    for (int64_t i = 0; i < a.numel; ++i) {
        float v = apply_epilogue(a.kind, a.x->get(i));
        if (a.kind == Epilogue::GeluGated) {
            if (!a.gate) throw std::runtime_error("activation: GeluGated needs a gate");
            v *= a.gate->get(i);
        }
        a.out->set(i, v);
    }
}

void conv2d_cpu(const Conv2dArgs& a) {
    const int64_t h_out = a.h_in + 2 * a.pad - a.k + 1;
    const int64_t w_out = a.w_in + 2 * a.pad - a.k + 1;
    for (int64_t b = 0; b < a.batch; ++b) {
        for (int64_t oc = 0; oc < a.c_out; ++oc) {
            for (int64_t oy = 0; oy < h_out; ++oy) {
                for (int64_t ox = 0; ox < w_out; ++ox) {
                    float acc = a.bias ? a.bias->get(oc) : 0.0f;
                    for (int64_t ic = 0; ic < a.c_in; ++ic) {
                        for (int64_t ky = 0; ky < a.k; ++ky) {
                            const int64_t iy = oy + ky - a.pad;
                            if (iy < 0 || iy >= a.h_in) continue;
                            for (int64_t kx = 0; kx < a.k; ++kx) {
                                const int64_t ix = ox + kx - a.pad;
                                if (ix < 0 || ix >= a.w_in) continue;
                                acc += a.x->get(((b * a.c_in + ic) * a.h_in + iy) * a.w_in + ix) *
                                       a.weight->get(((oc * a.c_in + ic) * a.k + ky) * a.k + kx);
                            }
                        }
                    }
                    a.out->set(((b * a.c_out + oc) * h_out + oy) * w_out + ox, acc);
                }
            }
        }
    }
}

void add_cpu(const AddArgs& a) {
    for (int64_t r = 0; r < a.outer; ++r) {
        for (int64_t i = 0; i < a.inner; ++i) {
            const int64_t at = r * a.inner + i;
            const int64_t bt = a.broadcast_b ? i : at;
            a.out->set(at, a.a->get(at) + a.scale * a.b->get(bt));
        }
    }
}

void chunk_cpu(const ChunkArgs& a) {
    const int64_t trow = (a.table_chunk >= 0) ? a.table_chunk : a.chunk;
    for (int64_t b = 0; b < a.batch; ++b) {
        for (int64_t c = 0; c < a.channels; ++c) {
            a.out->set(b * a.channels + c,
                       a.modulation->get(b * a.chunks * a.channels + a.chunk * a.channels + c) +
                       a.table->get(trow * a.channels + c));
        }
    }
}

void patch_cpu(const PatchArgs& a) {
    const int64_t P = a.patch, G = a.grid, C = a.channels, H = G * P;
    const int64_t per_token = C * P * P;
    for (int64_t b = 0; b < a.batch; ++b) {
        for (int64_t ty = 0; ty < G; ++ty) {
            for (int64_t tx = 0; tx < G; ++tx) {
                const int64_t token = ty * G + tx;
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t ky = 0; ky < P; ++ky) {
                        for (int64_t kx = 0; kx < P; ++kx) {
                            const int64_t img =
                                ((b * C + c) * H + ty * P + ky) * H + tx * P + kx;
                            // Forward: (channel, row, col) -- a conv weight's layout.
                            // Inverse: (row, col, channel) -- the reference's output einsum.
                            const int64_t tok = (b * G * G + token) * per_token +
                                                (a.inverse ? ((ky * P + kx) * C + c)
                                                           : ((c * P + ky) * P + kx));
                            if (a.inverse) a.out->set(img, a.in->get(tok));
                            else           a.out->set(tok, a.in->get(img));
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

void register_builtin_cpu_ops() {
    static bool done = false;
    if (done) return;
    done = true;
    register_impl<GemmArgs>("gemm", "stock", gemm_cpu,
                            "reference triple loop, fp32 accumulator, fixed reduction order");
    register_impl<AttentionArgs>("attention", "stock", attention_streaming,
                                 "online-softmax streaming; the score matrix never exists");
    register_impl<AttentionArgs>("attention", "materialized", attention_materialized,
                                 "writes the full score matrix, then softmaxes it. Kept as a "
                                 "measurable baseline for the tiling work, not as a default");
    register_impl<NormArgs>("norm", "stock", norm_cpu,
                            "LayerNorm / RMSNorm / GroupNorm, fp64 accumulators");
    register_impl<ModulateArgs>("modulate", "stock", modulate_cpu,
                                "unfused AdaLN modulation: a full activation round trip for two "
                                "flops per element. THE fusion target");
    register_impl<ActivationArgs>("activation", "stock", activation_cpu,
                                  "gelu-tanh / silu / gated-gelu, elementwise");
    register_impl<Conv2dArgs>("conv2d", "stock", conv2d_cpu,
                              "direct convolution, NCHW, stride 1");
    register_impl<AddArgs>("add", "stock", add_cpu,
                           "residual and broadcast add; glue, but glue that has to be an op to "
                           "reach device memory");
    register_impl<ChunkArgs>("chunk", "stock", chunk_cpu,
                             "AdaLN-single's six modulation chunks plus the per-layer table");
    register_impl<PatchArgs>("patch", "stock", patch_cpu,
                             "patchify and unpatchify; the two orderings differ, see ops.h");
}

std::vector<OpListing> list_all_impls() {
    return {
        {GemmRegistry::instance().op_name(), GemmRegistry::instance().list()},
        {AttentionRegistry::instance().op_name(), AttentionRegistry::instance().list()},
        {NormRegistry::instance().op_name(), NormRegistry::instance().list()},
        {ModulateRegistry::instance().op_name(), ModulateRegistry::instance().list()},
        {ActivationRegistry::instance().op_name(), ActivationRegistry::instance().list()},
        {Conv2dRegistry::instance().op_name(), Conv2dRegistry::instance().list()},
        {AddRegistry::instance().op_name(), AddRegistry::instance().list()},
        {ChunkRegistry::instance().op_name(), ChunkRegistry::instance().list()},
        {PatchRegistry::instance().op_name(), PatchRegistry::instance().list()},
    };
}

}  // namespace burnisher
