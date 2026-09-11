#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "burnisher/models.h"

namespace burnisher {

Tensor WeightSource::require(const std::string& name) const {
    if (!has(name)) {
        throw std::runtime_error(
            "weights: '" + name + "' is missing. A model that ran with a missing weight would "
            "produce a plausible image from wrong numbers, so this is fatal rather than a "
            "zero-fill.");
    }
    Tensor t = get(name);
    if (!t.defined()) throw std::runtime_error("weights: '" + name + "' resolved to nothing");
    return t;
}

CheckpointWeights::CheckpointWeights(std::vector<std::string> paths) {
    for (const auto& p : paths) shards_.push_back(SafeTensors::open(p));
    if (shards_.empty()) throw std::runtime_error("weights: no shards given");
}

bool CheckpointWeights::has(const std::string& name) const {
    for (const auto& s : shards_) if (s.has(name)) return true;
    return false;
}

Tensor CheckpointWeights::get(const std::string& name) const {
    for (const auto& s : shards_) if (s.has(name)) return s.get(name);
    // Route through the shard's own error, which names near misses.
    return shards_.front().get(name);
}

size_t CheckpointWeights::total_bytes() const {
    size_t n = 0;
    for (const auto& s : shards_) n += s.total_bytes();
    return n;
}

SyntheticWeights::SyntheticWeights(DType dtype, uint64_t seed) : dtype_(dtype), seed_(seed) {}

void SyntheticWeights::declare(const std::string& name, std::vector<int64_t> shape) {
    shapes_[name] = std::move(shape);
}

Tensor SyntheticWeights::get(const std::string& name) const {
    auto it = shapes_.find(name);
    if (it == shapes_.end()) {
        throw std::runtime_error("synthetic weights: '" + name + "' was never declared. The "
                                 "test fixture and the model disagree about the graph, which is "
                                 "worth finding out about loudly.");
    }
    Tensor t(it->second, dtype_);
    // splitmix64 keyed by the tensor NAME, so the same weight is the same numbers in every
    // process and every run. A test comparing two evaluations must be comparing the graph.
    uint64_t h = seed_;
    for (unsigned char c : name) h = h * 1099511628211ull ^ c;
    const int64_t n = t.numel();
    // Scaled like a real initialisation so activations neither vanish nor explode through
    // 28 layers -- a synthetic fixture that saturated would test the graph's plumbing and
    // nothing about its numerics.
    const float scale = 1.0f / std::sqrt(static_cast<float>(
        t.rank() >= 2 ? t.dim(t.rank() - 1) : 1));
    for (int64_t i = 0; i < n; ++i) {
        h += 0x9E3779B97F4A7C15ull;
        uint64_t z = h;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= (z >> 31);
        const float u = static_cast<float>((z >> 11) * (1.0 / 9007199254740992.0));
        t.set(i, (u * 2.0f - 1.0f) * scale);
    }
    return t;
}

Tensor sinusoidal_timestep_embedding(double t, int dim, DType dtype) {
    // The reference's construction: half the channels cosine, half sine, with a log-spaced
    // frequency ladder over max_period 10000. The ORDER of the halves matters -- swapping them
    // is a change nobody sees until the latents disagree.
    if (dim % 2) throw std::runtime_error("timestep embedding needs an even dimension");
    Tensor e({1, dim}, dtype);
    const int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        const double freq = std::exp(-std::log(10000.0) * static_cast<double>(i) / half);
        const double a = t * freq;
        e.set(i, static_cast<float>(std::cos(a)));
        e.set(half + i, static_cast<float>(std::sin(a)));
    }
    return e;
}

Tensor patchify(const Tensor& x, int patch, DType dtype) {
    // [B, C, H, W] -> [B, (H/p)*(W/p), C*p*p], channels-last within a patch, matching the
    // reference's conv-based patch embedding when that conv is read as a linear map.
    const int64_t B = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
    if (H % patch || W % patch) throw std::runtime_error("patchify: size not divisible by patch");
    const int64_t gh = H / patch, gw = W / patch;
    Tensor out({B, gh * gw, C * patch * patch}, dtype);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t py = 0; py < gh; ++py) {
            for (int64_t px = 0; px < gw; ++px) {
                const int64_t token = py * gw + px;
                for (int64_t c = 0; c < C; ++c) {
                    for (int64_t ky = 0; ky < patch; ++ky) {
                        for (int64_t kx = 0; kx < patch; ++kx) {
                            const int64_t src =
                                ((b * C + c) * H + py * patch + ky) * W + px * patch + kx;
                            const int64_t dst = (b * gh * gw + token) * (C * patch * patch) +
                                                (c * patch + ky) * patch + kx;
                            out.set(dst, x.get(src));
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor unpatchify(const Tensor& x, int batch, int channels, int grid, int patch, DType dtype) {
    const int64_t H = static_cast<int64_t>(grid) * patch;
    Tensor out({batch, channels, H, H}, dtype);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t py = 0; py < grid; ++py) {
            for (int64_t px = 0; px < grid; ++px) {
                const int64_t token = py * grid + px;
                for (int64_t c = 0; c < channels; ++c) {
                    for (int64_t ky = 0; ky < patch; ++ky) {
                        for (int64_t kx = 0; kx < patch; ++kx) {
                            const int64_t src =
                                (b * grid * grid + token) * (channels * patch * patch) +
                                (c * patch + ky) * patch + kx;
                            const int64_t dst =
                                ((b * channels + c) * H + py * patch + ky) * H + px * patch + kx;
                            out.set(dst, x.get(src));
                        }
                    }
                }
            }
        }
    }
    return out;
}

void declare_pixart_shapes(SyntheticWeights& w, const T5Config& t5, const DiTConfig& dit,
                           const VaeConfig& vae) {
    const int64_t inner = static_cast<int64_t>(t5.num_heads) * t5.d_kv;
    w.declare("shared.weight", {t5.vocab_size, t5.d_model});
    w.declare("encoder.final_layer_norm.weight", {t5.d_model});
    w.declare("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
              {32, t5.num_heads});
    for (int i = 0; i < t5.num_layers; ++i) {
        const std::string b = "encoder.block." + std::to_string(i) + ".";
        w.declare(b + "layer.0.layer_norm.weight", {t5.d_model});
        w.declare(b + "layer.1.layer_norm.weight", {t5.d_model});
        for (const char* p : {"q", "k", "v"}) {
            w.declare(b + "layer.0.SelfAttention." + p + ".weight", {inner, t5.d_model});
        }
        w.declare(b + "layer.0.SelfAttention.o.weight", {t5.d_model, inner});
        w.declare(b + "layer.1.DenseReluDense.wi_0.weight", {t5.d_ff, t5.d_model});
        if (t5.gated) w.declare(b + "layer.1.DenseReluDense.wi_1.weight", {t5.d_ff, t5.d_model});
        w.declare(b + "layer.1.DenseReluDense.wo.weight", {t5.d_model, t5.d_ff});
    }

    const int64_t d = dit.d(), dff = dit.d_ff(), p2 = dit.patch_size * dit.patch_size;
    // [out, in, kh, kw], the checkpoint's own conv layout -- NOT the flattened [out, in*kh*kw].
    // Contiguously they are the same bytes in the same order, and the patch embedding reads it
    // as a matrix, so the runtime works either way. The declaration still has to match the
    // checkpoint: it is what scripts/verify_checkpoint_layout.py compares against, and a shape
    // this file gets wrong is a shape nothing else can catch.
    w.declare("pos_embed.proj.weight", {d, dit.in_channels, dit.patch_size, dit.patch_size});
    w.declare("pos_embed.proj.bias", {d});
    w.declare("adaln_single.emb.timestep_embedder.linear_1.weight", {d, 256});
    w.declare("adaln_single.emb.timestep_embedder.linear_1.bias", {d});
    w.declare("adaln_single.emb.timestep_embedder.linear_2.weight", {d, d});
    w.declare("adaln_single.emb.timestep_embedder.linear_2.bias", {d});
    w.declare("adaln_single.linear.weight", {6 * d, d});
    w.declare("adaln_single.linear.bias", {6 * d});
    w.declare("caption_projection.linear_1.weight", {d, dit.caption_channels});
    w.declare("caption_projection.linear_1.bias", {d});
    w.declare("caption_projection.linear_2.weight", {d, d});
    w.declare("caption_projection.linear_2.bias", {d});
    w.declare("scale_shift_table", {2, d});
    w.declare("proj_out.weight", {p2 * dit.out_channels, d});
    w.declare("proj_out.bias", {p2 * dit.out_channels});
    for (int i = 0; i < dit.num_layers; ++i) {
        const std::string b = "transformer_blocks." + std::to_string(i) + ".";
        w.declare(b + "scale_shift_table", {6, d});
        for (const char* a : {"attn1", "attn2"}) {
            for (const char* p : {"to_q", "to_k", "to_v"}) {
                w.declare(b + a + "." + p + ".weight", {d, d});
                w.declare(b + a + "." + p + ".bias", {d});
            }
            w.declare(std::string(b) + a + ".to_out.0.weight", {d, d});
            w.declare(std::string(b) + a + ".to_out.0.bias", {d});
        }
        w.declare(b + "ff.net.0.proj.weight", {dff, d});
        w.declare(b + "ff.net.0.proj.bias", {dff});
        w.declare(b + "ff.net.2.weight", {d, dff});
        w.declare(b + "ff.net.2.bias", {d});
    }

    const int64_t zc = vae.latent_channels;
    std::vector<int> rev(vae.block_out_channels.rbegin(), vae.block_out_channels.rend());
    const auto conv_w = [&](const std::string& n, int64_t ci, int64_t co, int64_t k) {
        w.declare(n + ".weight", {co, ci, k, k});
        w.declare(n + ".bias", {co});
    };
    const auto gn = [&](const std::string& n, int64_t ch) {
        w.declare(n + ".weight", {ch});
        w.declare(n + ".bias", {ch});
    };
    const auto res = [&](const std::string& n, int64_t ci, int64_t co) {
        gn(n + ".norm1", ci);
        conv_w(n + ".conv1", ci, co, 3);
        gn(n + ".norm2", co);
        conv_w(n + ".conv2", co, co, 3);
        if (ci != co) conv_w(n + ".conv_shortcut", ci, co, 1);
    };
    conv_w("post_quant_conv", zc, zc, 1);
    conv_w("decoder.conv_in", zc, rev[0], 3);
    res("decoder.mid_block.resnets.0", rev[0], rev[0]);
    res("decoder.mid_block.resnets.1", rev[0], rev[0]);
    gn("decoder.mid_block.attentions.0.group_norm", rev[0]);
    for (const char* p : {"to_q", "to_k", "to_v"}) {
        w.declare(std::string("decoder.mid_block.attentions.0.") + p + ".weight",
                  {rev[0], rev[0]});
        w.declare(std::string("decoder.mid_block.attentions.0.") + p + ".bias", {rev[0]});
    }
    w.declare("decoder.mid_block.attentions.0.to_out.0.weight", {rev[0], rev[0]});
    w.declare("decoder.mid_block.attentions.0.to_out.0.bias", {rev[0]});
    int64_t prev = rev[0];
    for (size_t i = 0; i < rev.size(); ++i) {
        const int64_t co = rev[i];
        for (int j = 0; j <= vae.layers_per_block; ++j) {
            res("decoder.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j),
                j == 0 ? prev : co, co);
        }
        prev = co;
        if (i + 1 != rev.size()) {
            conv_w("decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv", co, co, 3);
        }
    }
    gn("decoder.conv_norm_out", prev);
    conv_w("decoder.conv_out", prev, 3, 3);
}

ImplSelection ImplSelection::from_request(const std::string& requested) {
    // Checked HERE, not only in resolve_all(): this is the entry point every model and the CLI
    // actually use, and until a test caught it an unknown name resolved silently to `stock` for
    // every op and the run reported success. A silent fallback is the one failure mode the
    // registry exists to prevent.
    if (!requested.empty() && requested != "stock" && !any_op_has_impl(requested)) {
        throw std::runtime_error(
            "no op registers an implementation named '" + requested + "'. This is a typo or a "
            "kernel that failed to register; running 'stock' everywhere and reporting success "
            "would measure a configuration nobody asked for. `burnisher info` lists what this "
            "build actually contains.");
    }
    ImplSelection s;
    s.gemm = resolve_impl("gemm", requested);
    s.attention = resolve_impl("attention", requested);
    s.norm = resolve_impl("norm", requested);
    s.modulate = resolve_impl("modulate", requested);
    s.activation = resolve_impl("activation", requested);
    s.conv2d = resolve_impl("conv2d", requested);
    return s;
}

std::map<std::string, std::string> ImplSelection::as_map() const {
    return {{"gemm", gemm}, {"attention", attention}, {"norm", norm},
            {"modulate", modulate}, {"activation", activation}, {"conv2d", conv2d}};
}

}  // namespace burnisher
