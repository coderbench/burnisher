#include <stdexcept>

#include "burnisher/safetensors.h"
#include "burnisher/tensor.h"
#include "check.h"

using namespace burnisher;

int main() {
    // bf16 rounding is round-to-nearest-even, matching the hardware instruction. Truncation is
    // the obvious implementation and it is biased; over twenty denoise steps a biased rounding
    // is a drift the correctness tolerance would have to be widened to admit.
    CHECK(bf16_to_f32(f32_to_bf16(1.0f)) == 1.0f);
    CHECK(bf16_to_f32(f32_to_bf16(0.0f)) == 0.0f);
    CHECK(bf16_to_f32(f32_to_bf16(-2.5f)) == -2.5f);
    {
        // 1 + 2^-9 sits exactly between two bf16 values; RNE takes the even one, which is 1.0.
        const float mid = 1.0f + 0.001953125f;
        CHECK_NEAR(bf16_to_f32(f32_to_bf16(mid)), 1.0f, 1e-7);
    }
    CHECK(std::isnan(bf16_to_f32(f32_to_bf16(NAN))));
    CHECK(std::isinf(bf16_to_f32(f32_to_bf16(INFINITY))));

    // A sub-byte dtype must not report half the bytes of an 8-bit one: the block scale is real
    // traffic, and a roofline built on 0.5 B/element would be wrong by 12%.
    CHECK(dtype_bytes(DType::NVFP4) > 0.5);
    CHECK_NEAR(dtype_bytes(DType::NVFP4), 0.5625, 1e-12);

    Tensor t({2, 3}, DType::F32);
    CHECK(t.numel() == 6);
    CHECK(t.nbytes() == 24);
    for (int i = 0; i < 6; ++i) t.set(i, static_cast<float>(i));
    CHECK(t.get(5) == 5.0f);
    CHECK(t.reshape({3, 2}).dim(0) == 3);
    CHECK_THROWS(t.reshape({4, 2}));

    Tensor b = t.to(DType::BF16);
    CHECK(b.dtype() == DType::BF16);
    CHECK(b.nbytes() == 12);
    CHECK_NEAR(b.get(4), 4.0f, 1e-6);

    // A packed format has no scalar accessor; reading one through get() would silently produce
    // garbage rather than an error.
    Tensor packed({4}, DType::NVFP4);
    CHECK_THROWS(packed.get(0));

    // The vendored JSON parser, which the safetensors header depends on.
    auto doc = json::parse(R"({"a":[1,2,3],"b":{"c":"x\ny"},"d":-1.5e2,"e":true,"f":null})");
    CHECK(doc.at("a").as_int_array().size() == 3);
    CHECK(doc.at("a").at(2).as_number() == 3);
    CHECK(doc.at("b").at("c").as_string() == "x\ny");
    CHECK_NEAR(doc.at("d").as_number(), -150.0, 1e-9);
    CHECK(doc.contains("e"));
    CHECK(!doc.contains("zz"));
    CHECK_THROWS(json::parse("{\"a\":1}trailing"));
    CHECK_THROWS(json::parse("{\"a\":}"));
    CHECK_THROWS(doc.at("nope"));

    return burnisher_test::summary("test_tensor");
}
