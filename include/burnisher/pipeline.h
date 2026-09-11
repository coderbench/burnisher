// End to end: prompt tokens -> latents -> pixels. One model, one resolution, one seed.
//
// v0 is deliberately a CORRECT, COMPLETE, SLOW pipeline. Contributors optimize; they do not
// bootstrap. Speed at launch is not the deliverable and building for it first would have meant
// shipping a fast thing nobody could show was right.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "burnisher/models.h"
#include "burnisher/scheduler.h"

namespace burnisher {

struct PipelineConfig {
    int resolution = 1024;
    int steps = 20;
    // No default that means anything: the harness always passes the generation's pinned value,
    // and a runtime-side default is a fourth place for the oracle to drift. This one is
    // deliberately NOT 4.5 so that a caller who forgets to pass it gets an obviously wrong
    // image rather than a subtly right one.
    double guidance_scale = 0.0;
    bool classifier_free_guidance = true;
    int caption_len = 300;
    uint64_t seed = 20260911;
    DType compute = DType::BF16;
    std::string impl = "stock";
    Device device = Device::CPU;
};

struct OutputStats {
    double latent_mean = 0.0;
    double latent_std = 0.0;
    double latent_absmax = 0.0;
    // A generation that produced a constant or non-finite latent ran fast and generated nothing.
    // These three numbers are how the harness tells that apart from a genuine speedup, so the
    // runtime always reports them and eval/runner.py always checks them.
    static OutputStats of(const Tensor& t);
};

struct StageTimings {
    double text_encode_s = 0.0;
    double denoise_s = 0.0;
    double vae_decode_s = 0.0;
    double total_s = 0.0;
};

class Pipeline {
  public:
    Pipeline(PipelineConfig cfg, std::shared_ptr<WeightSource> text_weights,
             std::shared_ptr<WeightSource> dit_weights,
             std::shared_ptr<WeightSource> vae_weights,
             T5Config t5, DiTConfig dit, VaeConfig vae, SchedulerConfig sched);

    // `token_ids` is [2, caption_len] under classifier-free guidance: the negative prompt first,
    // then the positive. Pre-tokenized on purpose -- the T5 tokenizer is a SentencePiece model
    // and vendoring one would put a second oracle in the repository. docs/CORRECTNESS.md has the
    // procedure for producing the ids and pinning their digest.
    // Returns the decoded pixels. `final_latent`, when given, receives the latent as it stood
    // AFTER the denoise loop and BEFORE the VAE -- which is what the correctness gate compares.
    //
    // Latents, not pixels, and the distinction is load-bearing: the VAE decode is itself one of
    // the things under optimization, so comparing images would fold two questions into one and
    // let a decoder change hide a denoiser change.
    // `noise`, when given, is the starting latent -- an INPUT rather than something the runtime
    // produces. That is what makes a comparison against a reference a comparison of the
    // runtimes: two RNGs agreeing bit for bit is not a thing to depend on, and regenerating at
    // the compute dtype starts a bf16 run and an fp32 reference from different points.
    Tensor generate(const Tensor& token_ids, StageTimings* timings,
                    Tensor* final_latent = nullptr, const Tensor* noise = nullptr);

    // Deterministic Gaussian noise from the seed. Drawn on the host, in a fixed order, and NOT
    // on the device: a sampler that drew its noise in kernel-launch order would make the whole
    // pipeline non-reproducible for a reason that has nothing to do with any kernel.
    Tensor initial_latent() const;

    const PipelineConfig& config() const { return cfg_; }

  private:
    PipelineConfig cfg_;
    std::shared_ptr<WeightSource> tw_, dw_, vw_;
    T5Config t5cfg_;
    DiTConfig ditcfg_;
    VaeConfig vaecfg_;
    SchedulerConfig schedcfg_;
};

}  // namespace burnisher
