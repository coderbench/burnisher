// DPM-Solver++ (2M, multistep) -- the sampler the pinned checkpoint ships with.
//
// This is part of the ORACLE. A sampler that differs from the reference by one index convention
// produces a plausible image from the same seed and fails the latent comparison with no clue as
// to why, so the formulation here is written to match `DPMSolverMultistepScheduler` as pinned in
// docs/CORRECTNESS.md, and `tests/test_scheduler.cpp` pins the coefficients it produces.
//
// It is deliberately NOT a general sampler framework. One sampler, one configuration, matching
// one reference. A configurable sampler would be a second place for the oracle to drift.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "burnisher/tensor.h"

namespace burnisher {

struct SchedulerConfig {
    int num_train_timesteps = 1000;
    double beta_start = 0.0001;
    double beta_end = 0.02;
    std::string beta_schedule = "linear";
    int solver_order = 2;
    std::string algorithm_type = "dpmsolver++";
    std::string solver_type = "midpoint";
    std::string prediction_type = "epsilon";
    std::string timestep_spacing = "linspace";
    bool lower_order_final = true;
};

class DPMSolverMultistep {
  public:
    explicit DPMSolverMultistep(SchedulerConfig cfg);

    void set_timesteps(int steps);
    int steps() const { return static_cast<int>(timesteps_.size()); }
    int64_t timestep(int i) const { return timesteps_.at(i); }
    double sigma(int i) const { return sigmas_.at(i); }

    // Scale the initial pure-noise latent. DPM-Solver++ takes x_T at the first sigma.
    double init_noise_sigma() const;

    // One step, in place on `sample`. `model_output` is the epsilon prediction at step `i`.
    void step(const Tensor& model_output, int i, Tensor& sample);

    void reset();

    // Exposed for the scheduler test: the coefficients are the thing that has to match the
    // reference, and a test that could only check the final image would not localise a mismatch.
    struct StepCoefficients {
        int order;
        double sigma_s0, sigma_t, alpha_t, alpha_s0;
        double exp_neg_h;   // sigma_t / sigma_s0
        double r0;          // h_0 / h; 0 for a first-order step
    };
    StepCoefficients coefficients(int i) const;

  private:
    SchedulerConfig cfg_;
    std::vector<double> alphas_cumprod_;
    std::vector<double> sigma_full_;
    std::vector<int64_t> timesteps_;
    std::vector<double> sigmas_;       // one longer than timesteps_; the last is 0
    std::vector<Tensor> history_;      // x0 predictions, most recent last
    int lower_order_nums_ = 0;

    Tensor convert_model_output(const Tensor& model_output, int i, const Tensor& sample) const;
};

}  // namespace burnisher
