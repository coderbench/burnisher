#include "burnisher/pipeline.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace burnisher {

OutputStats OutputStats::of(const Tensor& t) {
    OutputStats s;
    const int64_t n = t.numel();
    if (n == 0) return s;
    double sum = 0.0, sumsq = 0.0, amax = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double v = t.get(i);
        sum += v;
        sumsq += v * v;
        amax = std::max(amax, std::abs(v));
    }
    s.latent_mean = sum / n;
    s.latent_std = std::sqrt(std::max(0.0, sumsq / n - s.latent_mean * s.latent_mean));
    s.latent_absmax = amax;
    return s;
}

Pipeline::Pipeline(PipelineConfig cfg, std::shared_ptr<WeightSource> tw,
                   std::shared_ptr<WeightSource> dw, std::shared_ptr<WeightSource> vw,
                   T5Config t5, DiTConfig dit, VaeConfig vae, SchedulerConfig sched)
    : cfg_(std::move(cfg)), tw_(std::move(tw)), dw_(std::move(dw)), vw_(std::move(vw)),
      t5cfg_(t5), ditcfg_(dit), vaecfg_(vae), schedcfg_(sched) {}

Tensor Pipeline::initial_latent() const {
    const int64_t f = vaecfg_.scale_factor();
    const int64_t h = cfg_.resolution / f;
    Tensor z({1, vaecfg_.latent_channels, h, h}, cfg_.compute);
    // Box-Muller over a splitmix64 stream. Fixed order, host side, no device RNG: the gate
    // requires byte-identical replays, and a device RNG drawn in launch order is the classic way
    // a diffusion pipeline stops reproducing itself.
    uint64_t state = cfg_.seed;
    const auto next = [&state]() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return static_cast<double>((z ^ (z >> 31)) >> 11) * (1.0 / 9007199254740992.0);
    };
    for (int64_t i = 0; i < z.numel(); i += 2) {
        double u1 = next(), u2 = next();
        if (u1 < 1e-300) u1 = 1e-300;
        const double r = std::sqrt(-2.0 * std::log(u1));
        z.set(i, static_cast<float>(r * std::cos(2.0 * M_PI * u2)));
        if (i + 1 < z.numel()) z.set(i + 1, static_cast<float>(r * std::sin(2.0 * M_PI * u2)));
    }
    return z;
}

Tensor Pipeline::generate(const Tensor& token_ids, StageTimings* timings) {
    using clock = std::chrono::steady_clock;
    const auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const ImplSelection impls = ImplSelection::from_request(cfg_.impl);
    const int64_t batch = cfg_.classifier_free_guidance ? 2 : 1;
    if (token_ids.dim(0) != batch) {
        throw std::runtime_error(
            "pipeline: token ids have batch " + std::to_string(token_ids.dim(0)) +
            " but guidance asks for " + std::to_string(batch) +
            ". Under classifier-free guidance the negative prompt comes FIRST.");
    }

    const auto t0 = clock::now();
    T5Encoder text(t5cfg_, *tw_, cfg_.compute);
    Tensor caption = text.forward(token_ids, impls);
    const auto t1 = clock::now();

    PixArtDiT dit(ditcfg_, *dw_, cfg_.compute);
    DPMSolverMultistep sched(schedcfg_);
    sched.set_timesteps(cfg_.steps);

    Tensor latent = initial_latent();
    const int64_t per = latent.numel();
    Tensor batched({batch, latent.dim(1), latent.dim(2), latent.dim(3)}, cfg_.compute);
    Tensor eps(latent.shape(), cfg_.compute);

    for (int i = 0; i < sched.steps(); ++i) {
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t j = 0; j < per; ++j) batched.set(b * per + j, latent.get(j));

        Tensor out = dit.forward(batched, static_cast<double>(sched.timestep(i)), caption,
                                 impls);
        // The DiT predicts 2*in_channels: epsilon and a learned variance. The sampler is
        // epsilon-only, so the variance half is discarded -- computing it and throwing it away
        // is what the reference does, and not computing it would be a different model.
        const int64_t oc = out.dim(1);
        const int64_t chan = latent.dim(1);
        const int64_t spatial = latent.dim(2) * latent.dim(3);
        if (cfg_.classifier_free_guidance) {
            for (int64_t c = 0; c < chan; ++c) {
                for (int64_t s = 0; s < spatial; ++s) {
                    const double uncond = out.get((0 * oc + c) * spatial + s);
                    const double cond = out.get((1 * oc + c) * spatial + s);
                    eps.set(c * spatial + s,
                            static_cast<float>(uncond + cfg_.guidance_scale * (cond - uncond)));
                }
            }
        } else {
            for (int64_t c = 0; c < chan; ++c)
                for (int64_t s = 0; s < spatial; ++s)
                    eps.set(c * spatial + s, out.get(c * spatial + s));
        }
        sched.step(eps, i, latent);
    }
    const auto t2 = clock::now();

    Tensor scaled(latent.shape(), cfg_.compute);
    for (int64_t i = 0; i < latent.numel(); ++i) {
        scaled.set(i, static_cast<float>(latent.get(i) / vaecfg_.scaling_factor));
    }
    VaeDecoder vae(vaecfg_, *vw_, cfg_.compute);
    Tensor pixels = vae.forward(scaled, impls);
    const auto t3 = clock::now();

    if (timings) {
        timings->text_encode_s = secs(t0, t1);
        timings->denoise_s = secs(t1, t2);
        timings->vae_decode_s = secs(t2, t3);
        timings->total_s = secs(t0, t3);
    }
    return pixels;
}

}  // namespace burnisher
