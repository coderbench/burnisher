// Element types the runtime moves. Width is not a detail: every roofline in eval/ is a ratio of
// bytes to bandwidth, and a dtype that reports the wrong width makes the ceiling wrong by
// exactly that factor without making anything look broken.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace burnisher {

enum class DType {
    F32,
    BF16,
    F16,
    FP8E4M3,
    NVFP4,   // 4-bit payload with one fp8 block scale per 16 elements
};

// Bytes per element, INCLUDING the block scale for the sub-byte formats. A format that reported
// 0.5 for NVFP4 would price a 4-bit weight at exactly half an 8-bit one, and no such format
// exists -- the scale is real traffic and the kernels really read it.
inline double dtype_bytes(DType t) {
    switch (t) {
        case DType::F32: return 4.0;
        case DType::BF16: return 2.0;
        case DType::F16: return 2.0;
        case DType::FP8E4M3: return 1.0;
        case DType::NVFP4: return 0.5 + 1.0 / 16.0;
    }
    return 0.0;
}

inline const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "fp32";
        case DType::BF16: return "bf16";
        case DType::F16: return "fp16";
        case DType::FP8E4M3: return "fp8";
        case DType::NVFP4: return "nvfp4";
    }
    return "?";
}

bool dtype_from_name(const std::string& name, DType* out);

// Storage-only bf16. The runtime computes in float and stores in bf16 wherever the pinned
// pipeline does, so the CPU reference and a CUDA kernel round at the same places -- which is
// what makes a tolerance comparison between them mean anything.
struct BF16 {
    uint16_t bits;
};

inline float bf16_to_f32(BF16 v) {
    uint32_t u = static_cast<uint32_t>(v.bits) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

// Round-to-nearest-even, matching what the hardware instruction does. Truncation is the obvious
// implementation and it is biased; over twenty denoise steps a biased rounding is a drift the
// correctness gate would have to be widened to admit.
inline BF16 f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN: keep it a NaN rather than rounding into inf
        return BF16{static_cast<uint16_t>((u >> 16) | 0x0040u)};
    }
    uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
    return BF16{static_cast<uint16_t>(rounded >> 16)};
}

}  // namespace burnisher
