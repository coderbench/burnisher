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
      t5cfg_(t5), ditcfg_(dit), vaecfg_(vae), schedcfg_(sched) {
    // Resolved here rather than per-generate so that an unknown --impl, or a device baseline an
    // op cannot provide, fails at construction -- before a checkpoint is mapped and long before
    // anything is timed.
    impls_ = ImplSelection::from_request(cfg_.impl, cfg_.device);
}

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

Tensor Pipeline::generate(const Tensor& token_ids, StageTimings* timings,
                          Tensor* final_latent, const Tensor* noise) {
    using clock = std::chrono::steady_clock;
    const auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const ImplSelection& impls = impls_;
    const int64_t batch = cfg_.classifier_free_guidance ? 2 : 1;
    if (token_ids.dim(0) != batch) {
        throw std::runtime_error(
            "pipeline: token ids have batch " + std::to_string(token_ids.dim(0)) +
            " but guidance asks for " + std::to_string(batch) +
            ". Under classifier-free guidance the negative prompt comes FIRST.");
    }

    // The caption mask, derived from the ids rather than passed alongside them: a mask that can
    // disagree with the ids it describes is a mask that eventually will. Per batch row, because
    // the negative and positive prompts are different lengths.
    // Built on the host from host ids, then placed. The ids are a few hundred integers; the
    // mask they imply is the same size, and deriving it on the device would be a kernel for
    // nothing.
    Tensor ids_host = (token_ids.device() == Device::CUDA) ? token_ids.to_host() : token_ids;
    Tensor mask_host({batch, token_ids.dim(1)}, cfg_.compute);
    for (int64_t i = 0; i < mask_host.numel(); ++i) {
        mask_host.set(i, static_cast<int64_t>(ids_host.get(i)) != 0 ? 1.0f : 0.0f);
    }
    Tensor caption_mask = (impls.device == Device::CUDA) ? mask_host.to_device() : mask_host;
    Tensor ids = (impls.device == Device::CUDA && token_ids.device() != Device::CUDA)
                     ? token_ids.to_device() : token_ids;

    const auto t0 = clock::now();
    T5Encoder text(t5cfg_, *tw_, cfg_.compute);
    Tensor caption = text.forward(ids, impls);
    const auto t1 = clock::now();

    PixArtDiT dit(ditcfg_, *dw_, cfg_.compute);
    DPMSolverMultistep sched(schedcfg_);
    sched.set_timesteps(cfg_.steps);

    Tensor latent_host = noise ? noise->to(cfg_.compute) : initial_latent();
    Tensor latent = (impls.device == Device::CUDA) ? latent_host.to_device() : latent_host;
    const int64_t per = latent.numel();
    Tensor batched({batch, latent.dim(1), latent.dim(2), latent.dim(3)}, cfg_.compute,
                   impls.device);
    Tensor eps(latent.shape(), cfg_.compute, impls.device);
    const auto& repeat = RepeatRegistry::instance().get(impls.repeat);
    const auto& guidance = GuidanceRegistry::instance().get(impls.guidance);

    for (int i = 0; i < sched.steps(); ++i) {
        repeat(RepeatArgs{&latent, &batched, batch, per});

        Tensor out = dit.forward(batched, static_cast<double>(sched.timestep(i)), caption,
                                 caption_mask, impls);
        // The DiT predicts 2*in_channels: epsilon and a learned variance. The sampler is
        // epsilon-only, so the variance half is discarded -- computing it and throwing it away
        // is what the reference does, and not computing it would be a different model.
        const int64_t oc = out.dim(1);
        const int64_t chan = latent.dim(1);
        const int64_t spatial = latent.dim(2) * latent.dim(3);
        guidance(GuidanceArgs{&out, &eps, oc, chan, spatial,
                              cfg_.classifier_free_guidance
                                  ? static_cast<float>(cfg_.guidance_scale) : 0.0f});
        // The sampler is a handful of scalar coefficients over the whole latent and runs on the
        // host. At 128x128x4 that is a 256 kB round trip per step against a 13 TFLOP forward
        // pass -- immaterial, and it keeps one implementation of the ORACLE rather than two.
        Tensor eps_host = (impls.device == Device::CUDA) ? eps.to_host() : eps;
        Tensor lat_host = (impls.device == Device::CUDA) ? latent.to_host() : latent;
        sched.step(eps_host, i, lat_host);
        latent = (impls.device == Device::CUDA) ? lat_host.to_device() : lat_host;
    }
    const auto t2 = clock::now();

    // Handed back on the HOST: the gate compares it with numpy, and a device tensor there would
    // be a pointer nobody can read.
    if (final_latent) {
        *final_latent = (latent.device() == Device::CUDA) ? latent.to_host() : latent;
    }

    Tensor scaled(latent.shape(), cfg_.compute, impls.device);
    ScaleRegistry::instance().get(impls.scale)(
        ScaleArgs{&latent, &scaled, latent.numel(),
                  static_cast<float>(1.0 / vaecfg_.scaling_factor)});
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
