// The sampler is part of the ORACLE. A scheduler that differs from the reference by one index
// convention produces a plausible image from the same seed and fails the latent comparison with
// no clue as to why, so these tests pin the construction rather than the output.
#include <cmath>
#include <vector>

#include "burnisher/scheduler.h"
#include "check.h"

using namespace burnisher;

int main() {
    SchedulerConfig cfg;   // the pinned PixArt-Sigma configuration
    DPMSolverMultistep s(cfg);
    s.set_timesteps(20);

    CHECK(s.steps() == 20);
    // linspace(0, 999, 21), rounded, reversed, last dropped -- exactly the reference's
    // construction. Off by one here is the single most likely way to match "almost", which is
    // the least useful outcome available.
    const std::vector<int64_t> want{999, 949, 899, 849, 799, 749, 699, 649, 599, 549,
                                    500, 450, 400, 350, 300, 250, 200, 150, 100, 50};
    for (int i = 0; i < 20; ++i) {
        CHECK_MSG(s.timestep(i) == want[i],
                  "timestep " + std::to_string(i) + " is " + std::to_string(s.timestep(i)) +
                  ", reference gives " + std::to_string(want[i]));
    }
    // Descending, and the trailing sigma is exactly zero -- the last step lands on the clean
    // sample rather than near it.
    for (int i = 1; i < 20; ++i) CHECK(s.sigma(i) < s.sigma(i - 1));
    CHECK(s.sigma(20) == 0.0);
    CHECK(s.sigma(0) > 1.0);

    // The first step must be first order (no history) and the last must be first order
    // (lower_order_final), and everything between them second order. Getting the final step
    // wrong divides by a vanishing h and produces infinities, which is at least loud; getting
    // the FIRST one wrong reads uninitialised history and is not.
    Tensor sample({4}, DType::F32), eps({4}, DType::F32);
    for (int i = 0; i < 4; ++i) { sample.set(i, 1.0f + i); eps.set(i, 0.1f * i); }
    std::vector<int> orders;
    for (int i = 0; i < s.steps(); ++i) {
        orders.push_back(s.coefficients(i).order);
        s.step(eps, i, sample);
    }
    CHECK(orders.front() == 1);
    CHECK(orders.back() == 1);
    CHECK(orders[1] == 2);
    CHECK(orders[10] == 2);
    for (int i = 0; i < 4; ++i) CHECK(std::isfinite(sample.get(i)));

    // The last step's exp(-h) is exactly zero, so the update reduces to x0. Computing it through
    // lambda = -log(sigma) would put an infinity there instead.
    {
        DPMSolverMultistep t(cfg);
        t.set_timesteps(20);
        const auto c = t.coefficients(19);
        CHECK(c.sigma_t == 0.0);
        CHECK(c.exp_neg_h == 0.0);
        CHECK_NEAR(c.alpha_t, 1.0, 1e-12);
    }

    // Determinism: the same inputs give the same trajectory, every time.
    {
        DPMSolverMultistep a(cfg), b(cfg);
        a.set_timesteps(8);
        b.set_timesteps(8);
        Tensor xa({6}, DType::F32), xb({6}, DType::F32), e({6}, DType::F32);
        for (int i = 0; i < 6; ++i) {
            xa.set(i, std::sin(i * 0.7f));
            xb.set(i, std::sin(i * 0.7f));
            e.set(i, std::cos(i * 0.3f));
        }
        for (int i = 0; i < 8; ++i) { a.step(e, i, xa); b.step(e, i, xb); }
        for (int i = 0; i < 6; ++i) CHECK(xa.get(i) == xb.get(i));
    }

    // A single step is legal and is first order; zero steps is not a schedule.
    {
        DPMSolverMultistep one(cfg);
        one.set_timesteps(1);
        CHECK(one.steps() == 1);
        CHECK(one.coefficients(0).order == 1);
        CHECK_THROWS(one.set_timesteps(0));
    }

    // Only the pinned configuration is implemented. A second algorithm would be a second place
    // for the oracle to drift, so the others are refused rather than approximated.
    {
        SchedulerConfig other = cfg;
        other.algorithm_type = "dpmsolver";
        CHECK_THROWS(DPMSolverMultistep{other});
        SchedulerConfig heun = cfg;
        heun.solver_type = "heun";
        CHECK_THROWS(DPMSolverMultistep{heun});
        SchedulerConfig vpred = cfg;
        vpred.prediction_type = "v_prediction";
        CHECK_THROWS(DPMSolverMultistep{vpred});
    }

    return burnisher_test::summary("test_scheduler");
}
