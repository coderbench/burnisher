"""Element widths, in one place, because a roofline is wrong by exactly this factor.

Every bound in this package is a ratio of bytes to bandwidth or of flops to a peak, and both
terms carry a dtype. A table that silently assumed fp32 activations under a bf16 pipeline would
overstate every traffic term by 2x and understate every achieved fraction by the same amount --
and it would look completely reasonable on the page.
"""
from __future__ import annotations

# Bytes per element. NVFP4 and MXFP4 are 4-bit payloads with a block scale; the scale is real
# traffic and is counted, because a format that ignored it would price a 4-bit weight at exactly
# half a 8-bit one and no such format exists.
_WIDTH = {
    "fp32": 4.0, "f32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0, "f16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "fp8e4m3": 1.0, "fp8e5m2": 1.0,
    # 4 bits of payload + one fp8 scale per 16 elements = 0.5 + 1/16 bytes.
    "nvfp4": 0.5 + 1.0 / 16.0,
    # MXFP4: 4 bits + one E8M0 scale per 32 elements.
    "mxfp4": 0.5 + 1.0 / 32.0,
    "int8": 1.0,
}

# Which arithmetic peak a dtype is scored against. The mapping is the whole reason a dtype is
# part of a cell's identity rather than an implementation detail: moving a GEMM from bf16 to fp8
# does not make the kernel better, it changes the ceiling it is measured against.
_PEAK_KEY = {
    "fp32": "fp32_tflops", "f32": "fp32_tflops", "tf32": "fp32_tflops",
    "bf16": "bf16_tensor_tflops", "f16": "bf16_tensor_tflops", "fp16": "bf16_tensor_tflops",
    "fp8": "fp8_tensor_tflops", "fp8e4m3": "fp8_tensor_tflops", "fp8e5m2": "fp8_tensor_tflops",
    "int8": "fp8_tensor_tflops",
    "nvfp4": "fp4_tensor_tflops", "mxfp4": "fp4_tensor_tflops",
}


class DtypeError(ValueError):
    """A dtype this package has no width or no peak for."""


def width(dtype: str) -> float:
    key = str(dtype).lower()
    if key not in _WIDTH:
        raise DtypeError(f"no element width for dtype {dtype!r}; known: "
                         f"{', '.join(sorted(_WIDTH))}")
    return _WIDTH[key]


def peak_key(dtype: str) -> str:
    key = str(dtype).lower()
    if key not in _PEAK_KEY:
        raise DtypeError(f"no arithmetic peak is defined for dtype {dtype!r}. A cell scored "
                         f"against a peak nobody named would be scored against zero.")
    return _PEAK_KEY[key]


def known() -> tuple:
    return tuple(sorted(_WIDTH))
