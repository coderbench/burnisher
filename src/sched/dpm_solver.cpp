#include "burnisher/scheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace burnisher {

namespace {
// alpha and sigma of the variance-preserving parameterisation, from the Karras-style sigma.
//   alpha = 1 / sqrt(s^2 + 1),  sigma = s / sqrt(s^2 + 1)
inline void sigma_to_alpha_sigma(double s, double* alpha, double* sigma) {
    const double d = std::sqrt(s * s + 1.0);
    *alpha = 1.0 / d;
    *sigma = s / d;
}
}  // namespace

DPMSolverMultistep::DPMSolverMultistep(SchedulerConfig cfg) : cfg_(std::move(cfg)) {
    if (cfg_.algorithm_type != "dpmsolver++") {
        throw std::runtime_error("scheduler: only dpmsolver++ is implemented; the pinned "
                                 "checkpoint uses it and a second algorithm would be a second "
                                 "place for the oracle to drift");
    }
    if (cfg_.solver_type != "midpoint") {
        throw std::runtime_error("scheduler: only the midpoint solver type is implemented");
    }
    if (cfg_.prediction_type != "epsilon") {
        throw std::runtime_error("scheduler: only epsilon prediction is implemented");
    }
    const int N = cfg_.num_train_timesteps;
    alphas_cumprod_.resize(N);
    sigma_full_.resize(N);
    double cum = 1.0;
    for (int i = 0; i < N; ++i) {
        double beta;
        if (cfg_.beta_schedule == "linear") {
            beta = cfg_.beta_start +
                   (cfg_.beta_end - cfg_.beta_start) * (static_cast<double>(i) / (N - 1));
        } else if (cfg_.beta_schedule == "scaled_linear") {
            const double a = std::sqrt(cfg_.beta_start);
            const double b = std::sqrt(cfg_.beta_end);
            const double x = a + (b - a) * (static_cast<double>(i) / (N - 1));
            beta = x * x;
        } else {
            throw std::runtime_error("scheduler: unsupported beta_schedule '" +
                                     cfg_.beta_schedule + "'");
        }
        cum *= (1.0 - beta);
        alphas_cumprod_[i] = cum;
        sigma_full_[i] = std::sqrt((1.0 - cum) / cum);
    }
}

void DPMSolverMultistep::set_timesteps(int steps) {
    if (steps < 1) throw std::runtime_error("scheduler: at least one step");
    if (cfg_.timestep_spacing != "linspace") {
        throw std::runtime_error("scheduler: only linspace spacing is implemented; the pinned "
                                 "checkpoint uses it, and spacing changes the trajectory");
    }
    const int N = cfg_.num_train_timesteps;
    // linspace(0, N-1, steps+1), rounded, reversed, last dropped -- exactly the reference's
    // construction. Off-by-one here is the single most likely way to match the reference
    // "almost", which is the least useful outcome available.
    std::vector<int64_t> ts;
    for (int j = steps; j >= 1; --j) {
        double v = static_cast<double>(j) * (N - 1) / static_cast<double>(steps);
        ts.push_back(static_cast<int64_t>(std::llround(v)));
    }
    timesteps_ = std::move(ts);
    sigmas_.clear();
    for (int64_t t : timesteps_) sigmas_.push_back(sigma_full_[static_cast<size_t>(t)]);
    sigmas_.push_back(0.0);
    reset();
}

double DPMSolverMultistep::init_noise_sigma() const {
    // DPM-Solver++ operates on the VP parameterisation, so the initial latent is
    // alpha_T * x0 + sigma_T * noise with x0 unknown -- i.e. pure noise scaled by 1. The
    // reference returns 1.0 here and scales nothing; matching that is what matters.
    return 1.0;
}

void DPMSolverMultistep::reset() {
    history_.clear();
    lower_order_nums_ = 0;
}

Tensor DPMSolverMultistep::convert_model_output(const Tensor& eps, int i,
                                                const Tensor& sample) const {
    double alpha_s, sigma_s;
    sigma_to_alpha_sigma(sigmas_.at(static_cast<size_t>(i)), &alpha_s, &sigma_s);
    Tensor x0(sample.shape(), sample.dtype());
    for (int64_t n = 0; n < sample.numel(); ++n) {
        x0.set(n, static_cast<float>((sample.get(n) - sigma_s * eps.get(n)) / alpha_s));
    }
    return x0;
}

DPMSolverMultistep::StepCoefficients DPMSolverMultistep::coefficients(int i) const {
    const size_t idx = static_cast<size_t>(i);
    StepCoefficients c{};
    c.sigma_s0 = sigmas_.at(idx);
    c.sigma_t = sigmas_.at(idx + 1);
    sigma_to_alpha_sigma(c.sigma_t, &c.alpha_t, &c.sigma_vp_t);
    sigma_to_alpha_sigma(c.sigma_s0, &c.alpha_s0, &c.sigma_vp_s0);
    // exp(-h) = sigma_t / sigma_s0, computed as the ratio rather than through lambda.
    // Going via lambda = -log(sigma) puts an infinity in the last step, where sigma_t is
    // exactly zero, and then the arithmetic has to be special-cased anyway -- the ratio is
    // both simpler and finite everywhere.
    c.exp_neg_h = c.sigma_s0 > 0.0 ? (c.sigma_t / c.sigma_s0) : 0.0;
    const bool last = (idx + 2 == sigmas_.size());
    const bool first_order = (lower_order_nums_ < 1) ||
                             (cfg_.lower_order_final && last) ||
                             (cfg_.solver_order < 2);
    c.order = first_order ? 1 : 2;
    if (c.order == 2) {
        const double h = std::log(c.sigma_s0 / c.sigma_t);
        const double h0 = std::log(sigmas_.at(idx - 1) / c.sigma_s0);
        c.r0 = h0 / h;
    }
    return c;
}

void DPMSolverMultistep::step(const Tensor& model_output, int i, Tensor& sample) {
    if (i < 0 || static_cast<size_t>(i) >= timesteps_.size()) {
        throw std::runtime_error("scheduler: step index out of range");
    }
    Tensor x0 = convert_model_output(model_output, i, sample);
    // The solver keeps the last `order` x0 predictions. Keeping more would silently change the
    // trajectory the moment `solver_order` moved.
    history_.push_back(x0);
    if (static_cast<int>(history_.size()) > cfg_.solver_order) history_.erase(history_.begin());

    const StepCoefficients c = coefficients(i);
    const Tensor& m0 = history_.back();

    // The coefficient on `sample` is the ratio of the VARIANCE-PRESERVING sigmas, while
    // exp(-h) is the ratio of the KARRAS sigmas. They are different numbers -- at t=999, 0.99998
    // against 157 -- and using the Karras ratio in both places produces a trajectory that is
    // finite, plausible, and not the reference's. That is what this code did until
    // `scripts/differential_test.py --stage scheduler` compared it.
    const double sample_coef = (c.sigma_vp_s0 > 0.0) ? (c.sigma_vp_t / c.sigma_vp_s0) : 0.0;
    const double coef = c.alpha_t * (c.exp_neg_h - 1.0);
    if (c.order == 1) {
        for (int64_t n = 0; n < sample.numel(); ++n) {
            const double v = sample_coef * static_cast<double>(sample.get(n)) -
                             coef * m0.get(n);
            sample.set(n, static_cast<float>(v));
        }
    } else {
        const Tensor& m1 = history_[history_.size() - 2];
        for (int64_t n = 0; n < sample.numel(); ++n) {
            const double d0 = m0.get(n);
            const double d1 = (1.0 / c.r0) * (d0 - m1.get(n));
            const double v = sample_coef * static_cast<double>(sample.get(n)) -
                             coef * d0 - 0.5 * coef * d1;
            sample.set(n, static_cast<float>(v));
        }
    }
    if (lower_order_nums_ < cfg_.solver_order) ++lower_order_nums_;
}

}  // namespace burnisher
