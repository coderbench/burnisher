// T5 v1.1 encoder: RMSNorm, no biases anywhere, gated-gelu FFN, relative position bias.
//
// Runs ONCE per prompt and holds 89% of the pinned checkpoint's parameters. Both halves of that
// sentence matter: it is nearly the whole model and nearly none of the wall clock, which is why
// its cells are on the memory axis of the frontier rather than the latency one.
#include <cmath>
#include <stdexcept>

#include "burnisher/models.h"

namespace burnisher {
namespace {

const char* kPrefix = "encoder.block.";

// T5's pad id. Padding is the only thing that uses it, so a zero in the ids is a pad position.
constexpr int64_t kPadTokenId = 0;

std::string blk(int i, const std::string& tail) {
    return kPrefix + std::to_string(i) + "." + tail;
}

// T5's relative position bucketing, bidirectional. Reproduced rather than approximated: the
// bias enters every layer's scores, so a bucketing that is off by one shifts every attention
// distribution slightly and shows up only as a failed latent comparison.
}  // namespace

int t5_relative_bucket(int relative_position, int num_buckets, int max_distance) {
    int ret = 0;
    int n = relative_position;
    num_buckets /= 2;
    ret += (n > 0) ? num_buckets : 0;
    n = std::abs(n);
    const int max_exact = num_buckets / 2;
    if (n < max_exact) return ret + n;
    const double val = std::log(static_cast<double>(n) / max_exact) /
                       std::log(static_cast<double>(max_distance) / max_exact) *
                       (num_buckets - max_exact);
    int large = max_exact + static_cast<int>(val);
    if (large > num_buckets - 1) large = num_buckets - 1;
    return ret + large;
}

namespace {
}  // namespace

T5Encoder::T5Encoder(T5Config cfg, const WeightSource& w, DType compute)
    : cfg_(std::move(cfg)), w_(w), dtype_(compute) {}

Tensor T5Encoder::forward(const Tensor& token_ids, const ImplSelection& impls) const {
    const auto& gemm = GemmRegistry::instance().get(impls.gemm);
    const auto& attn = AttentionRegistry::instance().get(impls.attention);
    const auto& norm = NormRegistry::instance().get(impls.norm);
    const auto& act = ActivationRegistry::instance().get(impls.activation);

    const int64_t B = token_ids.dim(0), S = token_ids.dim(1);
    const int64_t d = cfg_.d_model, inner = static_cast<int64_t>(cfg_.num_heads) * cfg_.d_kv;
    const int64_t M = B * S;

    // Embedding gather. The table is `vocab x d` resident and M rows are read; a loader that
    // materialised the whole thing per call would move 263 MB to produce 2.4 MB of output.
    // The ids are read on the host -- they are a handful of integers and the bounds check is
    // worth more than the copy -- and the gather itself is an op, because the table is 263 MB
    // and lives wherever the model does.
    Tensor ids_host = (token_ids.device() == Device::CUDA) ? token_ids.to_host() : token_ids;
    for (int64_t i = 0; i < M; ++i) {
        const int64_t id = static_cast<int64_t>(ids_host.get(i));
        if (id < 0 || id >= cfg_.vocab_size) {
            throw std::runtime_error("t5: token id " + std::to_string(id) + " is outside the "
                                     "vocabulary; a tokenizer mismatch is a changed oracle");
        }
    }
    Tensor embed = w_.require("shared.weight");
    Tensor x({M, d}, dtype_, impls.device);
    GatherRegistry::instance().get(impls.gather)(
        GatherArgs{&embed, &token_ids, &x, M, d});

    // Relative position bias, computed once and added into every layer's scores.
    Tensor bias_host({static_cast<int64_t>(cfg_.num_heads), S, S}, dtype_);
    {
        Tensor& bias = bias_host;
        // 32 x heads of parameters. Pulled to the host because the bucketing is a scalar
        // computation over sequence positions and the result is uploaded once for every layer to
        // share -- doing it per layer on the device would be a kernel for 2 kB of data.
        Tensor rel_dev = w_.require(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
        Tensor rel = (rel_dev.device() == Device::CUDA) ? rel_dev.to_host() : rel_dev;
        const int buckets = static_cast<int>(rel.dim(0));
        for (int64_t q = 0; q < S; ++q) {
            for (int64_t k = 0; k < S; ++k) {
                const int b = t5_relative_bucket(static_cast<int>(k - q), buckets, 128);
                for (int h = 0; h < cfg_.num_heads; ++h) {
                    bias.set((h * S + q) * S + k, rel.get(b * cfg_.num_heads + h));
                }
            }
        }
    }
    Tensor bias = (impls.device == Device::CUDA) ? bias_host.to_device() : bias_host;

    // The padding mask, PER BATCH ROW. Not optional and not a detail: a prompt is padded to a
    // fixed 300 tokens, so most of a short caption's sequence is padding, and attending to it
    // changes every hidden state in the encoder.
    //
    // By KEY only. Padded query rows still produce output and that output is meaningless; it is
    // masked out downstream by the caption mask in cross-attention, which is what the reference
    // does too.
    Tensor mask_host({B, S}, dtype_);
    for (int64_t i = 0; i < B * S; ++i) {
        mask_host.set(i, static_cast<int64_t>(ids_host.get(i)) == kPadTokenId ? 0.0f : 1.0f);
    }
    Tensor key_mask = (impls.device == Device::CUDA) ? mask_host.to_device() : mask_host;

    Tensor normed({M, d}, dtype_, impls.device);
    Tensor q({M, inner}, dtype_), k({M, inner}, dtype_), v({M, inner}, dtype_);
    Tensor ctx({M, inner}, dtype_), proj({M, d}, dtype_);
    Tensor h0({M, static_cast<int64_t>(cfg_.d_ff)}, dtype_, impls.device);
    Tensor h1({M, static_cast<int64_t>(cfg_.d_ff)}, dtype_, impls.device);
    Tensor hact({M, static_cast<int64_t>(cfg_.d_ff)}, dtype_, impls.device);

    for (int layer = 0; layer < cfg_.num_layers; ++layer) {
        Tensor ln0 = w_.require(blk(layer, "layer.0.layer_norm.weight"));
        NormArgs na{&x, &ln0, nullptr, &normed, M, d, static_cast<float>(cfg_.eps), true, 0};
        norm(na);

        Tensor wq = w_.require(blk(layer, "layer.0.SelfAttention.q.weight"));
        Tensor wk = w_.require(blk(layer, "layer.0.SelfAttention.k.weight"));
        Tensor wv = w_.require(blk(layer, "layer.0.SelfAttention.v.weight"));
        Tensor wo = w_.require(blk(layer, "layer.0.SelfAttention.o.weight"));
        gemm(GemmArgs{&normed, &wq, nullptr, &q, M, inner, d, true});
        gemm(GemmArgs{&normed, &wk, nullptr, &k, M, inner, d, true});
        gemm(GemmArgs{&normed, &wv, nullptr, &v, M, inner, d, true});

        // T5 does NOT scale the attention scores by 1/sqrt(d_kv); the scaling is folded into
        // the query projection's initialisation. Applying it here would be a quiet, uniform
        // temperature change across every layer.
        AttentionArgs aa{&q, &k, &v, &ctx, B, cfg_.num_heads, S, S, cfg_.d_kv, 1.0f, &bias,
                         &key_mask};
        attn(aa);
        gemm(GemmArgs{&ctx, &wo, nullptr, &proj, M, d, inner, true});
        for (int64_t i = 0; i < M * d; ++i) x.set(i, x.get(i) + proj.get(i));

        Tensor ln1 = w_.require(blk(layer, "layer.1.layer_norm.weight"));
        NormArgs nb{&x, &ln1, nullptr, &normed, M, d, static_cast<float>(cfg_.eps), true, 0};
        norm(nb);

        Tensor wi0 = w_.require(blk(layer, "layer.1.DenseReluDense.wi_0.weight"));
        Tensor wo2 = w_.require(blk(layer, "layer.1.DenseReluDense.wo.weight"));
        gemm(GemmArgs{&normed, &wi0, nullptr, &h0, M, cfg_.d_ff, d, true});
        if (cfg_.gated) {
            Tensor wi1 = w_.require(blk(layer, "layer.1.DenseReluDense.wi_1.weight"));
            gemm(GemmArgs{&normed, &wi1, nullptr, &h1, M, cfg_.d_ff, d, true});
            ActivationArgs ag{&h0, &hact, M * cfg_.d_ff, Epilogue::GeluGated, &h1};
            act(ag);
        } else {
            ActivationArgs ag{&h0, &hact, M * cfg_.d_ff, Epilogue::Gelu, nullptr};
            act(ag);
        }
        gemm(GemmArgs{&hact, &wo2, nullptr, &proj, M, d, cfg_.d_ff, true});
        for (int64_t i = 0; i < M * d; ++i) x.set(i, x.get(i) + proj.get(i));
    }

    Tensor lnf = w_.require("encoder.final_layer_norm.weight");
    NormArgs nf{&x, &lnf, nullptr, &normed, M, d, static_cast<float>(cfg_.eps), true, 0};
    norm(nf);
    return normed.reshape({B, S, d});
}

}  // namespace burnisher
