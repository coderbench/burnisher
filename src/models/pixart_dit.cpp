// PixArt-Sigma's Transformer2DModel with ada_norm_single.
//
// This is the cell that holds 94% of the predicted wall clock at twenty steps, so it is the one
// the roofline table is really about. The op sequence below is meant to be readable side by side
// with `t5/dit` enumeration in eval/burnscore/geometry.py -- if the two drift, the published
// ceiling stops describing the thing that runs, and nothing else in the repository would notice.
//
// Several details here are the oracle rather than the design, and each is marked. They look
// arbitrary because they are: they are whatever the pinned reference does, and a "cleaner"
// choice is a silently different model.
#include <cmath>
#include <stdexcept>
#include <vector>

#include "burnisher/models.h"

namespace burnisher {
namespace {

std::string blk(int i, const std::string& tail) {
    return "transformer_blocks." + std::to_string(i) + "." + tail;
}

// ORACLE: 1D sin/cos position embedding, SIN first then COS.
//
// The timestep embedding in the same model is COS first, because it is built with
// flip_sin_to_cos=True and this one is not. Two conventions in one model is not a design; it is
// history, and matching it is not optional.
void sincos_1d(int dim, const std::vector<double>& pos, std::vector<double>* out) {
    const int half = dim / 2;
    out->assign(pos.size() * dim, 0.0);
    for (size_t m = 0; m < pos.size(); ++m) {
        for (int i = 0; i < half; ++i) {
            const double omega = 1.0 / std::pow(10000.0, static_cast<double>(i) / half);
            const double a = pos[m] * omega;
            (*out)[m * dim + i] = std::sin(a);
            (*out)[m * dim + half + i] = std::cos(a);
        }
    }
}

}  // namespace

// ORACLE: the 2D grid. meshgrid(w, h) in 'xy' order, then the FIRST mesh feeds the first half of
// the channels. diffusers names those halves `emb_h`/`emb_w` the other way round; the behaviour
// is what has to match, not the naming.
std::vector<double> dit_position_embedding(int dim, int grid, int base_size,
                                           double interpolation_scale) {
    std::vector<double> axis(grid);
    for (int i = 0; i < grid; ++i) {
        axis[i] = static_cast<double>(i) /
                  (static_cast<double>(grid) / base_size) / interpolation_scale;
    }
    std::vector<double> mesh_first(grid * grid), mesh_second(grid * grid);
    for (int r = 0; r < grid; ++r) {
        for (int c = 0; c < grid; ++c) {
            mesh_first[r * grid + c] = axis[c];   // w varies fastest
            mesh_second[r * grid + c] = axis[r];
        }
    }
    std::vector<double> a, b;
    sincos_1d(dim / 2, mesh_first, &a);
    sincos_1d(dim / 2, mesh_second, &b);
    std::vector<double> out(static_cast<size_t>(grid) * grid * dim);
    const int half = dim / 2;
    for (int m = 0; m < grid * grid; ++m) {
        for (int i = 0; i < half; ++i) out[m * dim + i] = a[m * half + i];
        for (int i = 0; i < half; ++i) out[m * dim + half + i] = b[m * half + i];
    }
    return out;
}

namespace {
}  // namespace

PixArtDiT::PixArtDiT(DiTConfig cfg, const WeightSource& w, DType compute)
    : cfg_(std::move(cfg)), w_(w), dtype_(compute) {}

Tensor PixArtDiT::forward(const Tensor& latent, double timestep, const Tensor& caption,
                          const Tensor& caption_mask, const ImplSelection& impls) const {
    const auto& gemm = GemmRegistry::instance().get(impls.gemm);
    const auto& attn = AttentionRegistry::instance().get(impls.attention);
    const auto& norm = NormRegistry::instance().get(impls.norm);
    const auto& modulate = ModulateRegistry::instance().get(impls.modulate);
    const auto& act = ActivationRegistry::instance().get(impls.activation);

    const int64_t B = latent.dim(0), H = latent.dim(2);
    const int64_t d = cfg_.d(), dff = cfg_.d_ff();
    const int64_t patch = cfg_.patch_size;
    const int64_t grid = H / patch;
    const int64_t N = grid * grid;
    const int64_t M = B * N;
    const int64_t cap_len = caption.dim(1);
    const int64_t Mcap = B * cap_len;

    // --- patch embedding ---
    Tensor patches = patchify(latent, static_cast<int>(patch), dtype_);
    Tensor x({M, d}, dtype_);
    {
        // [d, C, p, p] in the checkpoint. Read as a [d, C*p*p] matrix: contiguously those are
        // the same bytes in the same order, and `patchify` lays each token out as (c, ky, kx) to
        // match. The gemm below is therefore the patch-embedding convolution, exactly.
        Tensor pw = w_.require("pos_embed.proj.weight")
                        .reshape({d, cfg_.in_channels * patch * patch});
        Tensor pb = w_.require("pos_embed.proj.bias");
        Tensor flat = patches.reshape({M, cfg_.in_channels * patch * patch});
        gemm(GemmArgs{&flat, &pw, &pb, &x, M, d, cfg_.in_channels * patch * patch, true});
        const int base = cfg_.sample_size / static_cast<int>(patch);
        // ORACLE: interpolation_scale is 2 for the 1024px checkpoint and is part of the pin.
        std::vector<double> pe = dit_position_embedding(
            static_cast<int>(d), static_cast<int>(grid), base, 2.0);
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t t = 0; t < N; ++t) {
                for (int64_t c = 0; c < d; ++c) {
                    const int64_t i = (b * N + t) * d + c;
                    x.set(i, x.get(i) + static_cast<float>(pe[t * d + c]));
                }
            }
        }
    }

    // --- AdaLN-single: ONE modulation projection per forward, not per layer ---
    Tensor embedded_t({B, d}, dtype_);   // fed to the OUTPUT modulation
    Tensor modulation({B, 6 * d}, dtype_);
    {
        Tensor proj = sinusoidal_timestep_embedding(timestep, 256, dtype_);
        Tensor broad({B, 256}, dtype_);
        for (int64_t b = 0; b < B; ++b)
            for (int64_t i = 0; i < 256; ++i) broad.set(b * 256 + i, proj.get(i));
        Tensor w1 = w_.require("adaln_single.emb.timestep_embedder.linear_1.weight");
        Tensor b1 = w_.require("adaln_single.emb.timestep_embedder.linear_1.bias");
        Tensor w2 = w_.require("adaln_single.emb.timestep_embedder.linear_2.weight");
        Tensor b2 = w_.require("adaln_single.emb.timestep_embedder.linear_2.bias");
        Tensor h({B, d}, dtype_);
        gemm(GemmArgs{&broad, &w1, &b1, &h, B, d, 256, true, Epilogue::Silu});
        gemm(GemmArgs{&h, &w2, &b2, &embedded_t, B, d, d, true});
        Tensor silu({B, d}, dtype_);
        act(ActivationArgs{&embedded_t, &silu, B * d, Epilogue::Silu, nullptr});
        Tensor wl = w_.require("adaln_single.linear.weight");
        Tensor bl = w_.require("adaln_single.linear.bias");
        gemm(GemmArgs{&silu, &wl, &bl, &modulation, B, 6 * d, d, true});
    }

    // --- caption projection: T5 hidden 4096 -> d, once per forward ---
    Tensor cap({Mcap, d}, dtype_);
    {
        Tensor flat = caption.reshape({Mcap, cfg_.caption_channels});
        Tensor w1 = w_.require("caption_projection.linear_1.weight");
        Tensor b1 = w_.require("caption_projection.linear_1.bias");
        Tensor w2 = w_.require("caption_projection.linear_2.weight");
        Tensor b2 = w_.require("caption_projection.linear_2.bias");
        Tensor h({Mcap, d}, dtype_);
        gemm(GemmArgs{&flat, &w1, &b1, &h, Mcap, d, cfg_.caption_channels, true,
                      Epilogue::Gelu});
        gemm(GemmArgs{&h, &w2, &b2, &cap, Mcap, d, d, true});
    }

    // Scratch, allocated once. A forward pass that allocated per layer would spend its time in
    // the allocator and make every kernel measurement noisier than the thing being measured.
    Tensor normed({M, d}, dtype_), modded({M, d}, dtype_);
    Tensor q({M, d}, dtype_), k({M, d}, dtype_), v({M, d}, dtype_);
    Tensor kc({Mcap, d}, dtype_), vc({Mcap, d}, dtype_);
    Tensor ctx({M, d}, dtype_), proj({M, d}, dtype_);
    Tensor ff0({M, dff}, dtype_), ffa({M, dff}, dtype_);
    Tensor scale({B, d}, dtype_), shift({B, d}, dtype_), gate({B, d}, dtype_);

    // Cross-attention mask over the caption keys, per batch row. Under classifier-free guidance
    // the negative prompt is usually far shorter than the positive one, so a single shared mask
    // would either attend to the negative branch's padding or drop the positive branch's real
    // tokens.
    if (caption_mask.numel() != B * cap_len) {
        throw std::runtime_error(
            "dit: caption_mask has " + std::to_string(caption_mask.numel()) +
            " entries, expected batch x caption_len = " + std::to_string(B * cap_len) +
            ". Cross-attention over unmasked padding is a different model that still produces a "
            "plausible image.");
    }

    // ORACLE: the six chunks are (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
    // gate_mlp), in that order, and the per-layer scale_shift_table is ADDED to the shared
    // modulation before chunking. Reordering them is invisible in a diff and fatal to the image.
    const auto take = [&](int chunk, const Tensor& table, Tensor* dst) {
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t c = 0; c < d; ++c) {
                dst->set(b * d + c,
                         modulation.get(b * 6 * d + chunk * d + c) + table.get(chunk * d + c));
            }
        }
    };

    for (int layer = 0; layer < cfg_.num_layers; ++layer) {
        Tensor table = w_.require(blk(layer, "scale_shift_table"));   // [6, d]

        // 1. self-attention, modulated
        norm(NormArgs{&x, nullptr, nullptr, &normed, M, d,
                      static_cast<float>(cfg_.eps), false, 0});
        take(0, table, &shift);
        take(1, table, &scale);
        take(2, table, &gate);
        modulate(ModulateArgs{&normed, &scale, &shift, &modded, B, N, d, nullptr, nullptr});

        Tensor wq = w_.require(blk(layer, "attn1.to_q.weight"));
        Tensor bq = w_.require(blk(layer, "attn1.to_q.bias"));
        Tensor wk = w_.require(blk(layer, "attn1.to_k.weight"));
        Tensor bk = w_.require(blk(layer, "attn1.to_k.bias"));
        Tensor wv = w_.require(blk(layer, "attn1.to_v.weight"));
        Tensor bv = w_.require(blk(layer, "attn1.to_v.bias"));
        Tensor wo = w_.require(blk(layer, "attn1.to_out.0.weight"));
        Tensor bo = w_.require(blk(layer, "attn1.to_out.0.bias"));
        gemm(GemmArgs{&modded, &wq, &bq, &q, M, d, d, true});
        gemm(GemmArgs{&modded, &wk, &bk, &k, M, d, d, true});
        gemm(GemmArgs{&modded, &wv, &bv, &v, M, d, d, true});
        attn(AttentionArgs{&q, &k, &v, &ctx, B, cfg_.num_heads, N, N, cfg_.head_dim, 0.0f,
                           nullptr});
        gemm(GemmArgs{&ctx, &wo, &bo, &proj, M, d, d, true});
        // out = x + gate * attn_out. Expressed through the modulate op with a unit scale so the
        // gated residual is one kernel a contributor can fuse, rather than two loops here.
        {
            Tensor zero({B, d}, dtype_);
            modulate(ModulateArgs{&proj, &zero, &zero, &proj, B, N, d, &x, &gate});
        }
        for (int64_t i = 0; i < M * d; ++i) x.set(i, proj.get(i));

        // 2. cross-attention. ORACLE: PixArt does NOT normalise before attn2 -- the block feeds
        // `hidden_states` in directly. Adding the norm that every other DiT has here is the most
        // natural possible mistake and produces a subtly wrong image.
        Tensor wq2 = w_.require(blk(layer, "attn2.to_q.weight"));
        Tensor bq2 = w_.require(blk(layer, "attn2.to_q.bias"));
        Tensor wk2 = w_.require(blk(layer, "attn2.to_k.weight"));
        Tensor bk2 = w_.require(blk(layer, "attn2.to_k.bias"));
        Tensor wv2 = w_.require(blk(layer, "attn2.to_v.weight"));
        Tensor bv2 = w_.require(blk(layer, "attn2.to_v.bias"));
        Tensor wo2 = w_.require(blk(layer, "attn2.to_out.0.weight"));
        Tensor bo2 = w_.require(blk(layer, "attn2.to_out.0.bias"));
        gemm(GemmArgs{&x, &wq2, &bq2, &q, M, d, d, true});
        gemm(GemmArgs{&cap, &wk2, &bk2, &kc, Mcap, d, d, true});
        gemm(GemmArgs{&cap, &wv2, &bv2, &vc, Mcap, d, d, true});
        attn(AttentionArgs{&q, &kc, &vc, &ctx, B, cfg_.num_heads, N, cap_len, cfg_.head_dim,
                           0.0f, nullptr, &caption_mask});
        gemm(GemmArgs{&ctx, &wo2, &bo2, &proj, M, d, d, true});
        for (int64_t i = 0; i < M * d; ++i) x.set(i, x.get(i) + proj.get(i));

        // 3. feed-forward, modulated
        norm(NormArgs{&x, nullptr, nullptr, &normed, M, d,
                      static_cast<float>(cfg_.eps), false, 0});
        take(3, table, &shift);
        take(4, table, &scale);
        take(5, table, &gate);
        modulate(ModulateArgs{&normed, &scale, &shift, &modded, B, N, d, nullptr, nullptr});
        Tensor w0 = w_.require(blk(layer, "ff.net.0.proj.weight"));
        Tensor b0 = w_.require(blk(layer, "ff.net.0.proj.bias"));
        Tensor w2 = w_.require(blk(layer, "ff.net.2.weight"));
        Tensor b2 = w_.require(blk(layer, "ff.net.2.bias"));
        gemm(GemmArgs{&modded, &w0, &b0, &ff0, M, dff, d, true, Epilogue::Gelu});
        gemm(GemmArgs{&ff0, &w2, &b2, &proj, M, d, dff, true});
        {
            Tensor zero({B, d}, dtype_);
            modulate(ModulateArgs{&proj, &zero, &zero, &proj, B, N, d, &x, &gate});
        }
        for (int64_t i = 0; i < M * d; ++i) x.set(i, proj.get(i));
    }

    // --- output: norm, modulate from the OUTPUT table, project, unpatchify ---
    Tensor out_table = w_.require("scale_shift_table");   // [2, d]
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t c = 0; c < d; ++c) {
            shift.set(b * d + c, out_table.get(c) + embedded_t.get(b * d + c));
            scale.set(b * d + c, out_table.get(d + c) + embedded_t.get(b * d + c));
        }
    }
    norm(NormArgs{&x, nullptr, nullptr, &normed, M, d, static_cast<float>(cfg_.eps), false, 0});
    modulate(ModulateArgs{&normed, &scale, &shift, &modded, B, N, d, nullptr, nullptr});
    const int64_t out_per_token = patch * patch * cfg_.out_channels;
    Tensor tokens({M, out_per_token}, dtype_);
    Tensor wp = w_.require("proj_out.weight");
    Tensor bp = w_.require("proj_out.bias");
    gemm(GemmArgs{&modded, &wp, &bp, &tokens, M, out_per_token, d, true});
    return unpatchify(tokens.reshape({B, N, out_per_token}), static_cast<int>(B),
                      cfg_.out_channels, static_cast<int>(grid), static_cast<int>(patch),
                      dtype_);
}

}  // namespace burnisher
