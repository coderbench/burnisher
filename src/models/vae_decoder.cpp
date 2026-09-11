// AutoencoderKL decoder: 4-channel latent -> RGB pixels.
//
// Its shape is the whole story and the roofline table says so: the last two up-blocks run at
// 512x512 and 1024x1024, one 1024x1024x128 activation is 268 MB in bf16, and a ResNet block
// touches several of them. That is why the stage is tiling-sensitive, and it is why the
// difference between its whole-stage ceiling and the sum of its per-op bounds is the largest of
// any cell in the generation.
//
// One finding worth carrying here because it corrects a common belief: on a 5090 this stage is
// COMPUTE-bound at its arithmetic ceiling, not memory-bound. The convolutions have arithmetic
// intensities in the thousands against a ridge point near 117. What makes it feel memory-bound
// in practice is that a direct convolution at 256 and 128 channels reaches a small fraction of
// tensor peak, which is a statement about the achieved fraction rather than about the bound --
// and it is exactly the room the cell has.
#include <cmath>
#include <stdexcept>
#include <vector>

#include "burnisher/models.h"

namespace burnisher {
namespace {

struct Ctx {
    const WeightSource* w;
    DType dtype;
    const ImplSelection* impls;
    int groups;
    double eps;
};

Tensor conv(const Ctx& c, const Tensor& x, const std::string& name, int64_t c_in, int64_t c_out,
            int64_t h, int64_t k, int64_t pad) {
    const auto& fn = Conv2dRegistry::instance().get(c.impls->conv2d);
    Tensor wt = c.w->require(name + ".weight");
    Tensor bs = c.w->require(name + ".bias");
    const int64_t h_out = h + 2 * pad - k + 1;
    Tensor out({1, c_out, h_out, h_out}, c.dtype);
    fn(Conv2dArgs{&x, &wt, &bs, &out, 1, c_in, h, h, c_out, k, pad});
    return out;
}

Tensor group_norm(const Ctx& c, const Tensor& x, const std::string& name, int64_t ch,
                  int64_t h) {
    const auto& fn = NormRegistry::instance().get(c.impls->norm);
    Tensor wt = c.w->require(name + ".weight");
    Tensor bs = c.w->require(name + ".bias");
    Tensor out(x.shape(), c.dtype);
    NormArgs a{&x, &wt, &bs, &out, 1, ch * h * h, static_cast<float>(c.eps), false,
               static_cast<int64_t>(c.groups), ch};
    fn(a);
    return out;
}

Tensor silu(const Ctx& c, const Tensor& x) {
    const auto& fn = ActivationRegistry::instance().get(c.impls->activation);
    Tensor out(x.shape(), c.dtype);
    fn(ActivationArgs{&x, &out, x.numel(), Epilogue::Silu, nullptr});
    return out;
}

Tensor resnet(const Ctx& c, const Tensor& x, const std::string& name, int64_t c_in,
              int64_t c_out, int64_t h) {
    Tensor t = group_norm(c, x, name + ".norm1", c_in, h);
    t = silu(c, t);
    t = conv(c, t, name + ".conv1", c_in, c_out, h, 3, 1);
    t = group_norm(c, t, name + ".norm2", c_out, h);
    t = silu(c, t);
    t = conv(c, t, name + ".conv2", c_out, c_out, h, 3, 1);
    Tensor skip = x;
    if (c_in != c_out) {
        skip = conv(c, x, name + ".conv_shortcut", c_in, c_out, h, 1, 0);
    }
    Tensor out({1, c_out, h, h}, c.dtype);
    for (int64_t i = 0; i < out.numel(); ++i) out.set(i, skip.get(i) + t.get(i));
    return out;
}

// Spatial self-attention over every pixel of the mid-block. At 1024px that is 16384 positions,
// so a materializing implementation writes a 16384x16384 score matrix -- about a gigabyte. Both
// implementations are registered; which one runs is the contributor's choice and is measured.
Tensor spatial_attention(const Ctx& c, const Tensor& x, const std::string& name, int64_t ch,
                         int64_t h) {
    const auto& gemm = GemmRegistry::instance().get(c.impls->gemm);
    const auto& attn = AttentionRegistry::instance().get(c.impls->attention);
    const int64_t n = h * h;
    Tensor normed = group_norm(c, x, name + ".group_norm", ch, h);
    // NCHW -> [tokens, channels]. The transpose is real traffic and is part of what the fused
    // version of this op would remove.
    Tensor seq({n, ch}, c.dtype);
    for (int64_t p = 0; p < n; ++p)
        for (int64_t j = 0; j < ch; ++j) seq.set(p * ch + j, normed.get(j * n + p));

    Tensor q({n, ch}, c.dtype), k({n, ch}, c.dtype), v({n, ch}, c.dtype), o({n, ch}, c.dtype);
    for (const auto& pr : {std::pair<const char*, Tensor*>{"to_q", &q},
                           {"to_k", &k}, {"to_v", &v}}) {
        Tensor wt = c.w->require(name + "." + pr.first + ".weight");
        Tensor bs = c.w->require(name + "." + pr.first + ".bias");
        gemm(GemmArgs{&seq, &wt, &bs, pr.second, n, ch, ch, true});
    }
    attn(AttentionArgs{&q, &k, &v, &o, 1, 1, n, n, ch, 0.0f, nullptr});
    Tensor projd({n, ch}, c.dtype);
    Tensor wo = c.w->require(name + ".to_out.0.weight");
    Tensor bo = c.w->require(name + ".to_out.0.bias");
    gemm(GemmArgs{&o, &wo, &bo, &projd, n, ch, ch, true});

    Tensor out({1, ch, h, h}, c.dtype);
    for (int64_t j = 0; j < ch; ++j)
        for (int64_t p = 0; p < n; ++p) out.set(j * n + p, x.get(j * n + p) + projd.get(p * ch + j));
    return out;
}

Tensor upsample_nearest(const Tensor& x, int64_t ch, int64_t h, DType dtype) {
    const int64_t h2 = h * 2;
    Tensor out({1, ch, h2, h2}, dtype);
    for (int64_t j = 0; j < ch; ++j)
        for (int64_t y = 0; y < h2; ++y)
            for (int64_t xx = 0; xx < h2; ++xx)
                out.set((j * h2 + y) * h2 + xx, x.get((j * h + y / 2) * h + xx / 2));
    return out;
}

}  // namespace

VaeDecoder::VaeDecoder(VaeConfig cfg, const WeightSource& w, DType compute)
    : cfg_(std::move(cfg)), w_(w), dtype_(compute) {}

Tensor VaeDecoder::forward(const Tensor& latent, const ImplSelection& impls) const {
    if (latent.dim(0) != 1) {
        throw std::runtime_error("vae: batch 1 only -- classifier-free guidance is resolved "
                                 "into a single latent before the decoder runs, and pretending "
                                 "otherwise would double this stage's cost in every roofline");
    }
    Ctx c{&w_, dtype_, &impls, cfg_.norm_num_groups, cfg_.eps};
    const int64_t zc = cfg_.latent_channels;
    int64_t h = latent.dim(2);

    std::vector<int> rev(cfg_.block_out_channels.rbegin(), cfg_.block_out_channels.rend());

    Tensor x = conv(c, latent, "post_quant_conv", zc, zc, h, 1, 0);
    x = conv(c, x, "decoder.conv_in", zc, rev[0], h, 3, 1);

    x = resnet(c, x, "decoder.mid_block.resnets.0", rev[0], rev[0], h);
    x = spatial_attention(c, x, "decoder.mid_block.attentions.0", rev[0], h);
    x = resnet(c, x, "decoder.mid_block.resnets.1", rev[0], rev[0], h);

    int64_t prev = rev[0];
    for (size_t i = 0; i < rev.size(); ++i) {
        const int64_t cout = rev[i];
        const bool is_final = (i + 1 == rev.size());
        for (int j = 0; j <= cfg_.layers_per_block; ++j) {
            const std::string name = "decoder.up_blocks." + std::to_string(i) + ".resnets." +
                                     std::to_string(j);
            x = resnet(c, x, name, j == 0 ? prev : cout, cout, h);
        }
        prev = cout;
        if (!is_final) {
            x = upsample_nearest(x, cout, h, dtype_);
            h *= 2;
            // ORACLE: the upsampler's convolution runs at the NEW resolution, after the
            // interpolation. Running it before would be four times cheaper and a different model.
            x = conv(c, x, "decoder.up_blocks." + std::to_string(i) + ".upsamplers.0.conv",
                     cout, cout, h, 3, 1);
        }
    }

    x = group_norm(c, x, "decoder.conv_norm_out", prev, h);
    x = silu(c, x);
    return conv(c, x, "decoder.conv_out", prev, 3, h, 3, 1);
}

}  // namespace burnisher
