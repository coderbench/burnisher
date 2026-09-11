#include "burnisher/tensor.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace burnisher {

int64_t numel_of(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::invalid_argument("negative dimension");
        n *= d;
    }
    return n;
}

bool dtype_from_name(const std::string& name, DType* out) {
    if (name == "fp32" || name == "f32" || name == "float32") { *out = DType::F32; return true; }
    if (name == "bf16" || name == "bfloat16") { *out = DType::BF16; return true; }
    if (name == "fp16" || name == "f16" || name == "float16") { *out = DType::F16; return true; }
    if (name == "fp8" || name == "fp8e4m3") { *out = DType::FP8E4M3; return true; }
    if (name == "nvfp4") { *out = DType::NVFP4; return true; }
    return false;
}

size_t Tensor::nbytes() const {
    // Ceil, because a sub-byte dtype with an odd element count still occupies whole bytes.
    double exact = static_cast<double>(numel_) * dtype_bytes(dtype_);
    return static_cast<size_t>(exact + 0.9999);
}

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device) {
    numel_ = numel_of(shape_);
    if (device_ != Device::CPU) {
        throw std::runtime_error(
            "Tensor: only CPU allocation is implemented in the core; device tensors come from "
            "the CUDA backend so that a build without CUDA cannot silently allocate nothing");
    }
    size_t bytes = nbytes();
    void* p = std::calloc(bytes ? bytes : 1, 1);
    if (!p) throw std::bad_alloc();
    owned_ = std::shared_ptr<void>(p, std::free);
    data_ = p;
}

Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, Device d) {
    return Tensor(std::move(shape), dtype, d);
}

Tensor Tensor::view(void* data, std::vector<int64_t> shape, DType dtype, Device d) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.numel_ = numel_of(t.shape_);
    t.dtype_ = dtype;
    t.device_ = d;
    t.data_ = data;
    return t;
}

float Tensor::get(int64_t i) const {
    switch (dtype_) {
        case DType::F32: return static_cast<const float*>(data_)[i];
        case DType::BF16: return bf16_to_f32(static_cast<const BF16*>(data_)[i]);
        default:
            throw std::runtime_error(std::string("Tensor::get: dtype ") + dtype_name(dtype_) +
                                     " has no scalar accessor; it is a packed format and must "
                                     "be read by a kernel that knows its layout");
    }
}

void Tensor::set(int64_t i, float v) {
    switch (dtype_) {
        case DType::F32: static_cast<float*>(data_)[i] = v; return;
        case DType::BF16: static_cast<BF16*>(data_)[i] = f32_to_bf16(v); return;
        default:
            throw std::runtime_error(std::string("Tensor::set: dtype ") + dtype_name(dtype_) +
                                     " is a packed format");
    }
}

Tensor Tensor::to(DType target) const {
    if (target == dtype_) return *this;
    Tensor out(shape_, target, device_);
    for (int64_t i = 0; i < numel_; ++i) out.set(i, get(i));
    return out;
}

Tensor Tensor::reshape(std::vector<int64_t> shape) const {
    if (numel_of(shape) != numel_) {
        std::ostringstream os;
        os << "reshape: " << numel_ << " elements cannot become " << numel_of(shape);
        throw std::invalid_argument(os.str());
    }
    Tensor t = *this;
    t.shape_ = std::move(shape);
    return t;
}

std::string Tensor::describe() const {
    std::ostringstream os;
    os << dtype_name(dtype_) << "[";
    for (size_t i = 0; i < shape_.size(); ++i) os << (i ? "," : "") << shape_[i];
    os << "]";
    return os.str();
}

}  // namespace burnisher
