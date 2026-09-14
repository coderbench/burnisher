// The CUDA kernels against the CPU oracle, on a device. Built only with CUDA; passes as skipped on a
// machine without a GPU, because `scripts/check.sh` runs everywhere and this cannot.
//
// test_ops.cpp checks the oracle and never touches a device. This file is the other half: every
// device kernel that replaces a CPU one has to compute what the oracle computes, at shapes small
// enough to be exhaustive and odd enough to catch a stride or a layout slip, in both dtypes.
// A kernel that disagrees here produces a plausible image, so this is where it has to be caught.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "burnisher/device.h"
#include "burnisher/ops.h"
#include "check.h"

using namespace burnisher;

namespace {

Tensor ramp(std::vector<int64_t> shape, double k, DType dtype) {
    Tensor t(shape, DType::F32);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.set(i, static_cast<float>(std::sin(static_cast<double>(i) * k)));
    }
    return dtype == DType::F32 ? t : t.to(dtype);
}

// The largest elementwise difference between a host result and a device one.
double worst(const Tensor& host, const Tensor& device) {
    const Tensor d = device.to_host();
    double w = 0.0;
    for (int64_t i = 0; i < host.numel(); ++i) {
        w = std::max(w, static_cast<double>(std::fabs(host.get(i) - d.get(i))));
    }
    return w;
}

bool identical(const Tensor& a, const Tensor& b) {
    const Tensor x = a.to_host(), y = b.to_host();
    for (int64_t i = 0; i < x.numel(); ++i) {
        if (x.get(i) != y.get(i)) return false;
    }
    return true;
}

// Two dtypes, two tolerances. fp32 differs from the oracle only by reduction order. bf16 inputs
// are rounded identically on both sides and both compute in float, so what is left is the same
// reduction order plus one rounding of the output.
double tolerance(DType dt) { return dt == DType::F32 ? 2e-5 : 2e-3; }

void attention_matches_the_oracle(const std::string& impl, DType dt) {
    const std::string where = impl + " attention, " + dtype_name(dt);
    // Cross-attention as T5 and the DiT use it: a per-row key mask (the rows keep different key
    // counts, one keeps a single key) and T5's additive bias with unit scale.
    {
        const int64_t B = 2, S = 7, K = 5, H = 3, D = 4;
        Tensor q = ramp({B, S, H, D}, 0.31, dt), k = ramp({B, K, H, D}, 0.17, dt),
               v = ramp({B, K, H, D}, 0.53, dt), bias = ramp({H, S, K}, 0.07, dt);
        Tensor mask({B, K}, DType::F32);
        for (int64_t j = 0; j < K; ++j) {
            mask.set(j, j < 3 ? 1.0f : 0.0f);      // row 0 keeps three keys
            mask.set(K + j, j == 0 ? 1.0f : 0.0f);  // row 1 keeps one
        }
        if (dt != DType::F32) mask = mask.to(dt);
        Tensor expect({B, S, H, D}, dt);
        AttentionRegistry::instance().get("stock")(
            AttentionArgs{&q, &k, &v, &expect, B, H, S, K, D, 1.0f, &bias, &mask});

        Tensor dq = q.to_device(), dk = k.to_device(), dv = v.to_device(),
               dbias = bias.to_device(), dmask = mask.to_device();
        Tensor got({B, S, H, D}, dt, Device::CUDA), again({B, S, H, D}, dt, Device::CUDA);
        const AttentionArgs args{&dq, &dk, &dv, &got, B, H, S, K, D, 1.0f, &dbias, &dmask};
        AttentionRegistry::instance().get(impl)(args);
        AttentionArgs replay = args;
        replay.out = &again;
        AttentionRegistry::instance().get(impl)(replay);
        device::synchronize();

        const double w = worst(expect, got);
        CHECK_MSG(w < tolerance(dt), where + " (masked, biased) differs from the oracle by " +
                                         std::to_string(w));
        CHECK_MSG(identical(got, again), where + " is not byte-identical across two calls");
    }
    // Self-attention with the default 1/sqrt(head_dim) scale and nothing masked.
    {
        const int64_t B = 2, S = 9, H = 2, D = 6;
        Tensor q = ramp({B, S, H, D}, 0.23, dt), k = ramp({B, S, H, D}, 0.41, dt),
               v = ramp({B, S, H, D}, 0.11, dt);
        Tensor expect({B, S, H, D}, dt);
        AttentionRegistry::instance().get("stock")(
            AttentionArgs{&q, &k, &v, &expect, B, H, S, S, D, 0.0f, nullptr, nullptr});
        Tensor dq = q.to_device(), dk = k.to_device(), dv = v.to_device();
        Tensor got({B, S, H, D}, dt, Device::CUDA);
        AttentionRegistry::instance().get(impl)(
            AttentionArgs{&dq, &dk, &dv, &got, B, H, S, S, D, 0.0f, nullptr, nullptr});
        device::synchronize();
        const double w = worst(expect, got);
        CHECK_MSG(w < tolerance(dt), where + " (self) differs from the oracle by " +
                                         std::to_string(w));
    }
}

void conv2d_matches_the_oracle(const std::string& impl, DType dt) {
    const std::string where = impl + " conv2d, " + dtype_name(dt);
    // A 3x3 with padding and a 1x1 shortcut, the two shapes the VAE uses, on an odd input.
    for (const auto& kp : {std::pair<int64_t, int64_t>{3, 1}, {1, 0}}) {
        const int64_t c_in = 3, c_out = 5, h = 9, ks = kp.first, pad = kp.second;
        const int64_t h_out = h + 2 * pad - ks + 1;
        Tensor x = ramp({1, c_in, h, h}, 0.19, dt), w = ramp({c_out, c_in, ks, ks}, 0.37, dt),
               b = ramp({c_out}, 0.83, dt);
        Tensor expect({1, c_out, h_out, h_out}, dt);
        Conv2dRegistry::instance().get("stock")(
            Conv2dArgs{&x, &w, &b, &expect, 1, c_in, h, h, c_out, ks, pad});

        Tensor dx = x.to_device(), dw = w.to_device(), db = b.to_device();
        Tensor got({1, c_out, h_out, h_out}, dt, Device::CUDA),
               again({1, c_out, h_out, h_out}, dt, Device::CUDA);
        Conv2dRegistry::instance().get(impl)(
            Conv2dArgs{&dx, &dw, &db, &got, 1, c_in, h, h, c_out, ks, pad});
        Conv2dRegistry::instance().get(impl)(
            Conv2dArgs{&dx, &dw, &db, &again, 1, c_in, h, h, c_out, ks, pad});
        device::synchronize();

        const double d = worst(expect, got);
        CHECK_MSG(d < tolerance(dt), where + " k=" + std::to_string(ks) +
                                         " differs from the oracle by " + std::to_string(d));
        CHECK_MSG(identical(got, again), where + " is not byte-identical across two calls");
    }
}

}  // namespace

// `test_cuda_ops [impl...]` checks other registered names too; with none it checks the kernels
// this build ships.
int main(int argc, char** argv) {
    if (!device::available()) {
        std::fprintf(stderr, "test_cuda_ops: no CUDA device, skipped\n");
        return 0;
    }
    register_builtin_cpu_ops();
    register_cuda_ops();
    std::vector<std::string> impls(argv + 1, argv + argc);
    if (impls.empty()) impls = {"cuda-vendor"};
    for (const auto& impl : impls) {
        for (DType dt : {DType::F32, DType::BF16}) {
            if (AttentionRegistry::instance().has(impl)) attention_matches_the_oracle(impl, dt);
            if (Conv2dRegistry::instance().has(impl)) conv2d_matches_the_oracle(impl, dt);
        }
    }
    return burnisher_test::summary("test_cuda_ops");
}
