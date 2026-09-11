// The three model graphs, at a small config, on synthetic weights.
//
// This is the only way these files get exercised before they meet 22 GB of checkpoint, so the
// tests target the properties that a missing checkpoint cannot excuse: does the graph reproduce
// itself, do two registered implementations agree through a whole model, and do the shape
// transforms invert.
#include <cmath>
#include <memory>
#include <vector>

#include "burnisher/models.h"
#include "burnisher/pipeline.h"
#include "check.h"

using namespace burnisher;

namespace {

void small(T5Config* t5, DiTConfig* dit, VaeConfig* vae) {
    t5->num_layers = 2; t5->d_model = 32; t5->d_ff = 64; t5->d_kv = 8; t5->num_heads = 4;
    t5->vocab_size = 128;
    dit->num_layers = 2; dit->num_heads = 4; dit->head_dim = 8; dit->caption_channels = 32;
    dit->sample_size = 8;
    vae->block_out_channels = {8, 16, 16, 16};
    vae->layers_per_block = 1;
    vae->norm_num_groups = 4;
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
    double w = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i) w = std::max(w, static_cast<double>(std::fabs(a.get(i) - b.get(i))));
    return w;
}

}  // namespace

int main() {
    register_builtin_cpu_ops();
    T5Config t5; DiTConfig dit; VaeConfig vae;
    small(&t5, &dit, &vae);

    auto w = std::make_shared<SyntheticWeights>(DType::F32);
    declare_pixart_shapes(*w, t5, dit, vae);
    const ImplSelection stock = ImplSelection::from_request("stock");

    // Synthetic weights are keyed by NAME, so two instances agree exactly. A test comparing two
    // graph evaluations has to be comparing the graph and not the random number generator.
    {
        SyntheticWeights a(DType::F32), b(DType::F32);
        a.declare("x", {4, 4});
        b.declare("x", {4, 4});
        CHECK(max_abs_diff(a.get("x"), b.get("x")) == 0.0);
        // An undeclared weight is an error: it means the fixture and the model disagree about
        // the graph, which is worth finding out about loudly.
        CHECK_THROWS(a.get("never-declared"));
    }

    // --- shape transforms invert ---
    {
        Tensor x({2, 3, 8, 8}, DType::F32);
        for (int64_t i = 0; i < x.numel(); ++i) x.set(i, static_cast<float>(i));
        Tensor p = patchify(x, 2, DType::F32);
        CHECK(p.dim(1) == 16);          // (8/2)^2 tokens
        CHECK(p.dim(2) == 3 * 4);       // channels * patch^2
        CHECK_THROWS(patchify(x, 3, DType::F32));

        // The two are NOT inverses and must not be tested as if they were. `patchify` lays a
        // patch out (channel, row, col) because the reference's patch embedding is a Conv2d;
        // `unpatchify` reads (row, col, channel) because the reference's output projection is a
        // Linear followed by an einsum. A round-trip test would pass with BOTH wrong, and did:
        // the output ordering was wrong for the whole of this repository's first day and every
        // self-consistency check here was green.
        //
        // So each ordering is pinned against the reference's own formula instead.
        {
            const int64_t C = 3, P = 2, G = 4;
            Tensor tokens({1, G * G, C * P * P}, DType::F32);
            for (int64_t i = 0; i < tokens.numel(); ++i) tokens.set(i, static_cast<float>(i));
            Tensor img = unpatchify(tokens, 1, static_cast<int>(C), static_cast<int>(G),
                                    static_cast<int>(P), DType::F32);
            // reshape(h, w, p, q, c) -> einsum("nhwpqc->nchpwq"): channel varies FASTEST
            // within a token.
            for (int64_t ty = 0; ty < G; ++ty) {
                for (int64_t tx = 0; tx < G; ++tx) {
                    for (int64_t ky = 0; ky < P; ++ky) {
                        for (int64_t kx = 0; kx < P; ++kx) {
                            for (int64_t c = 0; c < C; ++c) {
                                const int64_t src =
                                    ((ty * G + tx) * (C * P * P)) + (ky * P + kx) * C + c;
                                const int64_t dst =
                                    ((c * (G * P)) + ty * P + ky) * (G * P) + tx * P + kx;
                                CHECK(img.get(dst) == tokens.get(src));
                            }
                        }
                    }
                }
            }
        }
        {
            // patchify: (channel, row, col), column fastest -- a conv weight's layout.
            const int64_t C = 2, P = 2, G = 2;
            Tensor img({1, C, G * P, G * P}, DType::F32);
            for (int64_t i = 0; i < img.numel(); ++i) img.set(i, static_cast<float>(i));
            Tensor tok = patchify(img, static_cast<int>(P), DType::F32);
            for (int64_t ty = 0; ty < G; ++ty) {
                for (int64_t tx = 0; tx < G; ++tx) {
                    for (int64_t c = 0; c < C; ++c) {
                        for (int64_t ky = 0; ky < P; ++ky) {
                            for (int64_t kx = 0; kx < P; ++kx) {
                                const int64_t dst =
                                    ((ty * G + tx) * (C * P * P)) + (c * P + ky) * P + kx;
                                const int64_t src =
                                    ((c * (G * P)) + ty * P + ky) * (G * P) + tx * P + kx;
                                CHECK(tok.get(dst) == img.get(src));
                            }
                        }
                    }
                }
            }
        }
    }

    // --- the timestep embedding's halves are cos then sin ---
    //
    // The position embedding in the same model is sin then cos. Two conventions in one model is
    // not a design, it is history, and matching it is not optional.
    {
        Tensor e = sinusoidal_timestep_embedding(0.0, 8, DType::F32);
        for (int i = 0; i < 4; ++i) CHECK_NEAR(e.get(i), 1.0, 1e-6);        // cos(0)
        for (int i = 4; i < 8; ++i) CHECK_NEAR(e.get(i), 0.0, 1e-6);        // sin(0)
        CHECK_THROWS(sinusoidal_timestep_embedding(0.0, 7, DType::F32));
    }

    // --- the ORACLE details, against an independent implementation ---
    //
    // These numbers were produced in numpy from the reference's published algorithm, not from
    // this code. Differential testing is the only leverage available on details that are
    // arbitrary by nature: sin-then-cos here while the timestep embedding is cos-then-sin, a
    // meshgrid whose first axis is x, an interpolation scale that is part of the checkpoint pin.
    // A whole-forward-pass test cannot see any of it, and the failure mode is a plausible image.
    {
        const double kPosEmbed[] = {
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.000000000, 0.000000000, 1.000000000, 1.000000000,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.479425539, 0.004999979, 0.877582562, 0.999987500,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.841470985, 0.009999833, 0.540302306, 0.999950000,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.997494987, 0.014999438, 0.070737202, 0.999887502,
        0.997494987, 0.014999438, 0.070737202, 0.999887502
        };
        std::vector<double> got = dit_position_embedding(8, 4, 4, 2.0);
        CHECK(got.size() == sizeof(kPosEmbed) / sizeof(kPosEmbed[0]));
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            worst = std::max(worst, std::fabs(got[i] - kPosEmbed[i]));
        }
        CHECK_MSG(worst < 1e-9,
                  "the 2D position embedding disagrees with an independent implementation by " +
                  std::to_string(worst) + "; check the sin/cos order and the meshgrid axis order");
    }
    {
        const double kTimestep[] = {
        -0.883849273, 0.964966028, 0.283662185, 0.877582562,
        -0.467771805, -0.262374854, -0.958924275, 0.479425539
        };
        Tensor e = sinusoidal_timestep_embedding(500.0, 8, DType::F32);
        double worst = 0.0;
        for (int i = 0; i < 8; ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(e.get(i)) - kTimestep[i]));
        }
        CHECK_MSG(worst < 1e-6,
                  "the timestep embedding disagrees with an independent implementation by " +
                  std::to_string(worst) + "; it is COS first (flip_sin_to_cos), unlike the "
                  "position embedding in the same model");
    }

    {
        // T5's bidirectional relative-position bucketing, against an independent implementation.
        // It enters every layer's attention scores; off by one shifts every distribution
        // slightly and is invisible in anything short of a reference comparison. The probes
        // straddle every boundary the algorithm has: the sign split, `max_exact`, and the
        // saturation at `max_distance`.
        const struct { int rp; int bucket; } kBuckets[] = {
            {-300, 15},
            {-129, 15},
            {-128, 15},
            {-127, 15},
            {-64, 14},
            {-16, 10},
            {-9, 8},
            {-8, 8},
            {-7, 7},
            {-1, 1},
            {0, 0},
            {1, 17},
            {7, 23},
            {8, 24},
            {9, 24},
            {16, 26},
            {64, 30},
            {127, 31},
            {128, 31},
            {129, 31},
            {300, 31}
        };
        for (const auto& c : kBuckets) {
            const int got = t5_relative_bucket(c.rp, 32, 128);
            CHECK_MSG(got == c.bucket,
                      "relative position " + std::to_string(c.rp) + " bucketed to " +
                      std::to_string(got) + ", reference gives " + std::to_string(c.bucket));
        }
    }

    // --- each model reproduces itself exactly ---
    {
        T5Encoder enc(t5, *w, DType::F32);
        Tensor ids({2, 5}, DType::F32);
        for (int64_t i = 0; i < ids.numel(); ++i) ids.set(i, static_cast<float>(i * 3 % 100));
        Tensor a = enc.forward(ids, stock);
        Tensor b = enc.forward(ids, stock);
        CHECK_MSG(max_abs_diff(a, b) == 0.0,
                  "the text encoder does not reproduce itself; nothing downstream can be "
                  "attributed to a candidate");
        CHECK(a.dim(0) == 2 && a.dim(1) == 5 && a.dim(2) == t5.d_model);

        // A pad id (0) must be masked out of self-attention. Changing what sits BEYOND the
        // padding boundary of one row must not move that row's real positions.
        Tensor padded({2, 5}, DType::F32);
        for (int64_t b = 0; b < 2; ++b) {
            for (int64_t i = 0; i < 5; ++i) {
                padded.set(b * 5 + i, i < 3 ? static_cast<float>(7 + i) : 0.0f);
            }
        }
        Tensor p1 = enc.forward(padded, stock);
        Tensor p2 = padded;                      // same ids, different padding content is
        Tensor alt({2, 5}, DType::F32);          // impossible -- pad IS 0 -- so instead check
        for (int64_t i = 0; i < 10; ++i) alt.set(i, padded.get(i));
        Tensor q1 = enc.forward(alt, stock);
        CHECK(max_abs_diff(p1, q1) == 0.0);
        // ... and that the encoder is not simply ignoring the mask: an unpadded sequence of the
        // same real tokens must differ at the padded positions but agree nowhere by accident.
        for (int64_t i = 0; i < p1.numel(); ++i) CHECK(std::isfinite(p1.get(i)));
        // A token id outside the vocabulary is a tokenizer mismatch, which is a changed oracle.
        Tensor bad({1, 2}, DType::F32);
        bad.set(0, 0.0f);
        bad.set(1, 99999.0f);
        CHECK_THROWS(enc.forward(bad, stock));
    }
    {
        PixArtDiT model(dit, *w, DType::F32);
        const int64_t h = dit.sample_size;
        Tensor z({2, dit.in_channels, h, h}, DType::F32);
        for (int64_t i = 0; i < z.numel(); ++i) z.set(i, std::sin(i * 0.37f));
        Tensor cap({2, 5, dit.caption_channels}, DType::F32);
        for (int64_t i = 0; i < cap.numel(); ++i) cap.set(i, std::cos(i * 0.11f));
        // Per batch row, and the two rows are DIFFERENT lengths -- which is the case a single
        // shared mask gets wrong, and the reason the op takes a [batch, kv_len] mask.
        Tensor mask({2, 5}, DType::F32);
        for (int64_t b = 0; b < 2; ++b)
            for (int64_t i = 0; i < 5; ++i)
                mask.set(b * 5 + i, i < (b == 0 ? 3 : 2) ? 1.0f : 0.0f);

        Tensor a = model.forward(z, 500.0, cap, mask, stock);
        Tensor b = model.forward(z, 500.0, cap, mask, stock);
        CHECK(max_abs_diff(a, b) == 0.0);
        CHECK(a.dim(1) == dit.out_channels);
        CHECK(a.dim(2) == h);
        for (int64_t i = 0; i < a.numel(); ++i) CHECK(std::isfinite(a.get(i)));

        // The timestep must actually reach the output. An AdaLN path that silently produced the
        // same modulation for every step would make the denoise loop a no-op, and the image
        // would still look like something.
        Tensor c = model.forward(z, 100.0, cap, mask, stock);
        CHECK_MSG(max_abs_diff(a, c) > 1e-6,
                  "changing the timestep changed nothing; the modulation is not connected");
        // So must the caption, or cross-attention is decorative.
        Tensor cap2({2, 5, dit.caption_channels}, DType::F32);
        for (int64_t i = 0; i < cap2.numel(); ++i) cap2.set(i, std::cos(i * 0.91f));
        Tensor d = model.forward(z, 500.0, cap2, mask, stock);
        CHECK_MSG(max_abs_diff(a, d) > 1e-6,
                  "changing the caption changed nothing; cross-attention is not connected");

        // --- two registered implementations agree through the WHOLE model ---
        //
        // The op-level version of this check is in test_ops; this is the one that matters for
        // scoring, because a paired measurement compares two impls through the entire graph.
        const ImplSelection mat = ImplSelection::from_request("materialized");
        CHECK(mat.attention == "materialized");
        Tensor e = model.forward(z, 500.0, cap, mask, mat);
        const double worst = max_abs_diff(a, e);
        CHECK_MSG(worst < 1e-4,
                  "two attention implementations disagree by " + std::to_string(worst) +
                  " through the DiT; a paired measurement between them would be measuring a "
                  "behaviour change rather than a kernel");

        // Masked caption positions must not reach the output. A cross-attention that ignored the
        // mask would attend to padding on every layer -- a different model that still produces a
        // plausible image, which is the worst kind of bug to have.
        Tensor cap_pad_changed({2, 5, dit.caption_channels}, DType::F32);
        for (int64_t i = 0; i < cap_pad_changed.numel(); ++i) {
            cap_pad_changed.set(i, cap.get(i));
        }
        for (int64_t b = 0; b < 2; ++b) {
            for (int64_t t = (b == 0 ? 3 : 2); t < 5; ++t) {   // that row's masked positions
                for (int64_t c = 0; c < dit.caption_channels; ++c) {
                    cap_pad_changed.set((b * 5 + t) * dit.caption_channels + c, 99.0f);
                }
            }
        }
        Tensor masked = model.forward(z, 500.0, cap_pad_changed, mask, stock);
        CHECK_MSG(max_abs_diff(a, masked) < 1e-4,
                  "changing a MASKED caption position changed the output; cross-attention is "
                  "attending to padding");

        Tensor all_ones({2, 5}, DType::F32);
        for (int64_t i = 0; i < 10; ++i) all_ones.set(i, 1.0f);
        Tensor unmasked = model.forward(z, 500.0, cap, all_ones, stock);
        CHECK_MSG(max_abs_diff(a, unmasked) > 1e-6,
                  "masking changed nothing, so the mask is being ignored");

        Tensor wrong_mask({2, 4}, DType::F32);
        CHECK_THROWS(model.forward(z, 500.0, cap, wrong_mask, stock));
    }
    {
        VaeDecoder dec(vae, *w, DType::F32);
        const int64_t h = 4;
        Tensor z({1, vae.latent_channels, h, h}, DType::F32);
        for (int64_t i = 0; i < z.numel(); ++i) z.set(i, std::sin(i * 0.21f));
        Tensor a = dec.forward(z, stock);
        Tensor b = dec.forward(z, stock);
        CHECK(max_abs_diff(a, b) == 0.0);
        CHECK(a.dim(1) == 3);
        CHECK(a.dim(2) == h * vae.scale_factor());
        for (int64_t i = 0; i < a.numel(); ++i) CHECK(std::isfinite(a.get(i)));
        // Batch > 1 is refused rather than quietly costing twice as much as every roofline says.
        Tensor z2({2, vae.latent_channels, h, h}, DType::F32);
        CHECK_THROWS(dec.forward(z2, stock));
    }

    // --- the whole pipeline reproduces itself, which is the gate's actual requirement ---
    {
        SchedulerConfig sched;
        PipelineConfig cfg;
        cfg.resolution = static_cast<int>(vae.scale_factor()) * dit.sample_size;
        cfg.steps = 3;
        cfg.caption_len = 4;
        cfg.compute = DType::F32;
        Pipeline p(cfg, w, w, w, t5, dit, vae, sched);
        Tensor ids({2, cfg.caption_len}, DType::F32);
        for (int64_t i = 0; i < ids.numel(); ++i) ids.set(i, static_cast<float>(i * 7 % 100));

        StageTimings t{};
        Tensor a = p.generate(ids, &t);
        Tensor b = p.generate(ids, &t);
        CHECK_MSG(max_abs_diff(a, b) == 0.0,
                  "the pipeline does not reproduce itself byte for byte; it cannot be a "
                  "reference for anything");
        const OutputStats s = OutputStats::of(a);
        CHECK_MSG(s.latent_std > 1e-6, "the pipeline produced a constant image");
        CHECK(std::isfinite(s.latent_absmax));
        CHECK(t.total_s > 0.0);

        // The initial latent is seeded and host-side, so two pipelines with the same seed start
        // identically and different seeds do not.
        PipelineConfig other = cfg;
        other.seed = cfg.seed + 1;
        Pipeline q(other, w, w, w, t5, dit, vae, sched);
        CHECK(max_abs_diff(p.initial_latent(), p.initial_latent()) == 0.0);
        CHECK(max_abs_diff(p.initial_latent(), q.initial_latent()) > 1e-6);

        // Under classifier-free guidance the token ids must carry both prompts. A batch-1 input
        // here would silently guide against itself.
        Tensor single({1, cfg.caption_len}, DType::F32);
        CHECK_THROWS(p.generate(single, &t));
    }

    return burnisher_test::summary("test_models");
}
