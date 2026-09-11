#include <cmath>
#include <vector>

#include "burnisher/ops.h"
#include "check.h"

using namespace burnisher;

namespace {

Tensor filled(std::vector<int64_t> shape, std::initializer_list<float> vals) {
    Tensor t(shape, DType::F32);
    int64_t i = 0;
    for (float v : vals) t.set(i++, v);
    return t;
}

Tensor ramp(std::vector<int64_t> shape, double k) {
    Tensor t(shape, DType::F32);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.set(i, static_cast<float>(std::sin(static_cast<double>(i) * k)));
    }
    return t;
}

}  // namespace

int main() {
    register_builtin_cpu_ops();
    const auto& gemm = GemmRegistry::instance().get("stock");
    const auto& norm = NormRegistry::instance().get("stock");
    const auto& conv = Conv2dRegistry::instance().get("stock");
    const auto& mod = ModulateRegistry::instance().get("stock");

    // --- gemm, both weight layouts ---
    {
        Tensor a = filled({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor b = filled({3, 2}, {1, 2, 3, 4, 5, 6});       // [K, N]
        Tensor bt = filled({2, 3}, {1, 3, 5, 2, 4, 6});      // the same matrix as [N, K]
        Tensor out({2, 2}, DType::F32), out_t({2, 2}, DType::F32);
        gemm(GemmArgs{&a, &b, nullptr, &out, 2, 2, 3});
        gemm(GemmArgs{&a, &bt, nullptr, &out_t, 2, 2, 3, true});
        CHECK_NEAR(out.get(0), 22.0, 1e-5);   // 1*1+2*3+3*5
        CHECK_NEAR(out.get(1), 28.0, 1e-5);
        CHECK_NEAR(out.get(3), 64.0, 1e-5);
        // The transposed path must agree exactly. A checkpoint stores [out, in]; getting this
        // wrong does not crash, it produces a plausible image from a transposed weight matrix.
        for (int i = 0; i < 4; ++i) CHECK_NEAR(out.get(i), out_t.get(i), 1e-5);
    }
    {
        // An operand smaller than its declared shape is checked, not trusted: a shape slip reads
        // adjacent weights and produces a perfectly plausible result.
        Tensor a = filled({2, 2}, {1, 2, 3, 4});
        Tensor b = filled({2, 2}, {1, 0, 0, 1});
        Tensor out({2, 2}, DType::F32);
        CHECK_THROWS(gemm(GemmArgs{&a, &b, nullptr, &out, 4, 2, 2}));
    }

    // --- the two registered attention implementations must agree ---
    //
    // This is the single most valuable test in the file. The whole A/B mechanism rests on two
    // named implementations of one op being interchangeable; if they are not, every paired
    // measurement between them is measuring a behaviour change rather than a kernel.
    {
        const int64_t B = 2, H = 3, S = 7, M = 5, D = 4;
        Tensor q = ramp({B, H, S, D}, 0.31), k = ramp({B, H, M, D}, 0.17),
               v = ramp({B, H, M, D}, 0.53);
        Tensor a({B, H, S, D}, DType::F32), b({B, H, S, D}, DType::F32);
        AttentionRegistry::instance().get("stock")(
            AttentionArgs{&q, &k, &v, &a, B, H, S, M, D, 0.0f, nullptr});
        AttentionRegistry::instance().get("materialized")(
            AttentionArgs{&q, &k, &v, &b, B, H, S, M, D, 0.0f, nullptr});
        double worst = 0.0;
        for (int64_t i = 0; i < a.numel(); ++i) {
            worst = std::max(worst, static_cast<double>(std::fabs(a.get(i) - b.get(i))));
        }
        CHECK_MSG(worst < 1e-5,
                  "streaming and materialized attention disagree by " + std::to_string(worst) +
                  "; the A/B mechanism is only meaningful if two named implementations of one "
                  "op are interchangeable");
    }
    {
        // Softmax rows sum to one, and an additive bias shifts the distribution rather than
        // being ignored. T5 adds its relative position bias this way in every layer.
        const int64_t S = 4, D = 2;
        Tensor q = ramp({1, 1, S, D}, 0.7), k = ramp({1, 1, S, D}, 0.3);
        // Channel 0 of every value is 1, so the softmax weights sum to one there whatever the
        // distribution. Channel 1 VARIES with the key index, so it reports which keys were
        // attended to -- without that, any weighting gives the same answer and the bias check
        // below cannot fail even when the bias is being dropped.
        Tensor v({1, 1, S, D}, DType::F32);
        for (int64_t i = 0; i < S; ++i) {
            v.set(i * D, 1.0f);
            v.set(i * D + 1, static_cast<float>(i));
        }
        Tensor out({1, 1, S, D}, DType::F32);
        AttentionRegistry::instance().get("stock")(
            AttentionArgs{&q, &k, &v, &out, 1, 1, S, S, D, 0.0f, nullptr});
        for (int64_t i = 0; i < S; ++i) CHECK_NEAR(out.get(i * D), 1.0, 1e-5);

        // Drive every row's attention onto key 0, so channel 1 of the output must collapse to 0.
        Tensor bias({1, S, S}, DType::F32);
        for (int64_t i = 0; i < bias.numel(); ++i) bias.set(i, (i % S == 0) ? 50.0f : 0.0f);
        Tensor biased({1, 1, S, D}, DType::F32);
        AttentionRegistry::instance().get("stock")(
            AttentionArgs{&q, &k, &v, &biased, 1, 1, S, S, D, 0.0f, &bias});
        bool moved = false;
        for (int64_t i = 0; i < out.numel(); ++i) {
            if (std::fabs(out.get(i) - biased.get(i)) > 1e-6) moved = true;
        }
        CHECK_MSG(moved, "an attention bias that changes nothing is a bias that is being ignored");
        for (int64_t i = 0; i < S; ++i) {
            CHECK_NEAR(biased.get(i * D + 1), 0.0, 1e-4);   // all mass on key 0
        }
    }

    // --- norms ---
    {
        Tensor x = filled({1, 4}, {1, 2, 3, 4});
        Tensor out({1, 4}, DType::F32);
        norm(NormArgs{&x, nullptr, nullptr, &out, 1, 4, 1e-6f, false, 0, 0});
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < 4; ++i) mean += out.get(i);
        for (int i = 0; i < 4; ++i) var += out.get(i) * out.get(i);
        CHECK_NEAR(mean / 4.0, 0.0, 1e-5);
        CHECK_NEAR(var / 4.0, 1.0, 1e-4);

        // RMSNorm does NOT subtract the mean. A LayerNorm standing in for it changes every T5
        // activation and is invisible in a diff.
        Tensor r({1, 4}, DType::F32);
        norm(NormArgs{&x, nullptr, nullptr, &r, 1, 4, 1e-6f, true, 0, 0});
        double rmean = 0.0;
        for (int i = 0; i < 4; ++i) rmean += r.get(i);
        CHECK_MSG(std::fabs(rmean / 4.0) > 0.1, "RMSNorm must not centre its input");
    }
    {
        // GroupNorm normalizes within a group and applies a PER-CHANNEL affine. An implementation
        // that skipped the affine would still produce well-scaled activations and a wrong image.
        const int64_t ch = 4, hw = 4, groups = 2;
        Tensor x = ramp({1, ch, 2, 2}, 0.9);
        Tensor w({ch}, DType::F32), b({ch}, DType::F32);
        for (int64_t c = 0; c < ch; ++c) { w.set(c, 1.0f + c); b.set(c, -0.5f * c); }
        Tensor out({1, ch, 2, 2}, DType::F32), plain({1, ch, 2, 2}, DType::F32);
        norm(NormArgs{&x, &w, &b, &out, 1, ch * hw, 1e-6f, false, groups, ch});
        norm(NormArgs{&x, nullptr, nullptr, &plain, 1, ch * hw, 1e-6f, false, groups, ch});
        for (int64_t c = 0; c < ch; ++c) {
            for (int64_t s = 0; s < hw; ++s) {
                CHECK_NEAR(out.get(c * hw + s), plain.get(c * hw + s) * w.get(c) + b.get(c),
                           1e-4);
            }
        }
        // Each group has zero mean over its own channels.
        for (int64_t g = 0; g < groups; ++g) {
            double m = 0.0;
            const int64_t per = ch * hw / groups;
            for (int64_t i = 0; i < per; ++i) m += plain.get(g * per + i);
            CHECK_NEAR(m / per, 0.0, 1e-4);
        }
        Tensor bad({1, 3, 2, 2}, DType::F32);
        CHECK_THROWS(norm(NormArgs{&bad, nullptr, nullptr, &bad, 1, 3 * hw, 1e-6f, false,
                                   groups, 3}));
    }

    // --- modulation, with and without the gated residual ---
    {
        const int64_t B = 2, T = 3, C = 2;
        Tensor x = ramp({B, T, C}, 0.4), res = ramp({B, T, C}, 0.8);
        Tensor scale({B, C}, DType::F32), shift({B, C}, DType::F32), gate({B, C}, DType::F32);
        for (int64_t i = 0; i < B * C; ++i) {
            scale.set(i, 0.5f); shift.set(i, -0.25f); gate.set(i, 2.0f);
        }
        Tensor out({B, T, C}, DType::F32);
        mod(ModulateArgs{&x, &scale, &shift, &out, B, T, C, nullptr, nullptr});
        for (int64_t i = 0; i < out.numel(); ++i) {
            CHECK_NEAR(out.get(i), x.get(i) * 1.5f - 0.25f, 1e-5);
        }
        Tensor gated({B, T, C}, DType::F32);
        mod(ModulateArgs{&x, &scale, &shift, &gated, B, T, C, &res, &gate});
        for (int64_t i = 0; i < gated.numel(); ++i) {
            CHECK_NEAR(gated.get(i), (x.get(i) * 1.5f - 0.25f) * 2.0f + res.get(i), 1e-5);
        }
    }

    // --- conv2d ---
    {
        // A 3x3 box filter with pad 1: the centre of a 3x3 all-ones input sums all nine.
        Tensor x({1, 1, 3, 3}, DType::F32);
        for (int i = 0; i < 9; ++i) x.set(i, 1.0f);
        Tensor w({1, 1, 3, 3}, DType::F32);
        for (int i = 0; i < 9; ++i) w.set(i, 1.0f);
        Tensor out({1, 1, 3, 3}, DType::F32);
        conv(Conv2dArgs{&x, &w, nullptr, &out, 1, 1, 3, 3, 1, 3, 1});
        CHECK_NEAR(out.get(4), 9.0, 1e-5);   // centre
        CHECK_NEAR(out.get(0), 4.0, 1e-5);   // corner: four taps in range
        // A 1x1 convolution is a per-pixel channel mix.
        Tensor x2 = ramp({1, 2, 2, 2}, 0.5);
        Tensor w2 = filled({2, 2, 1, 1}, {1, 0, 0, 1});
        Tensor out2({1, 2, 2, 2}, DType::F32);
        conv(Conv2dArgs{&x2, &w2, nullptr, &out2, 1, 2, 2, 2, 2, 1, 0});
        for (int i = 0; i < 8; ++i) CHECK_NEAR(out2.get(i), x2.get(i), 1e-5);
    }

    // --- the registry itself ---
    CHECK_THROWS(GemmRegistry::instance().get("no-such-kernel"));
    CHECK(GemmRegistry::instance().has("stock"));
    CHECK(AttentionRegistry::instance().list().size() >= 2);
    // An unknown --impl is an error, not a silent fallback to stock.
    CHECK_THROWS(ImplSelection::from_request("definitely-not-registered"));
    // A name some ops register and others do not resolves per op, and reports honestly.
    CHECK(ImplSelection::from_request("materialized").attention == "materialized");
    CHECK(ImplSelection::from_request("materialized").gemm == "stock");

    return burnisher_test::summary("test_ops");
}
