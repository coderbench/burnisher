#include "burnisher/tensor.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#include "burnisher/device.h"

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
    const size_t bytes = nbytes();
    if (device_ == Device::CUDA) {
        void* p = device::alloc(bytes ? bytes : 1);
        owned_ = std::shared_ptr<void>(p, [](void* q) { device::release(q); });
        data_ = p;
        return;
    }
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
    if (device_ != Device::CPU) {
        throw std::runtime_error(
            "Tensor::get on a device tensor. Scalar host access to device memory is not a slow "
            "path, it is a fault -- every piece of glue in a model graph has to be an op. Copy "
            "it back with `to_host()` if you really want to look at it.");
    }
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
    if (device_ != Device::CPU) {
        throw std::runtime_error("Tensor::set on a device tensor; see Tensor::get");
    }
    switch (dtype_) {
        case DType::F32: static_cast<float*>(data_)[i] = v; return;
        case DType::BF16: static_cast<BF16*>(data_)[i] = f32_to_bf16(v); return;
        default:
            throw std::runtime_error(std::string("Tensor::set: dtype ") + dtype_name(dtype_) +
                                     " is a packed format");
    }
}

namespace {

// Run `fn(begin, end)` over [0, n) in chunks, on as many threads as the host has. Every element
// is converted by the same pure function whichever thread holds it, so the result does not depend
// on the split.
template <typename Fn>
void parallel_chunks(int64_t n, Fn fn) {
    const int64_t kMinChunk = int64_t{1} << 22;
    const int64_t threads = std::max<int64_t>(
        1, std::min<int64_t>(std::thread::hardware_concurrency(), n / kMinChunk));
    if (threads == 1) {
        fn(0, n);
        return;
    }
    const int64_t chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (int64_t t = 0; t < threads; ++t) {
        const int64_t begin = t * chunk, end = std::min(n, begin + chunk);
        if (begin < end) pool.emplace_back(fn, begin, end);
    }
    for (auto& th : pool) th.join();
}

}  // namespace

Tensor Tensor::to(DType target) const {
    if (target == dtype_) return *this;
    if (device_ != Device::CPU) {
        throw std::runtime_error("Tensor::to: convert on the host, then upload");
    }
    Tensor out(shape_, target, device_);
    // fp32 <-> bf16 is how every checkpoint weight reaches a bf16 run, and element-wise through
    // get/set it was a single core walking T5's 4.7 billion parameters: ten seconds of every
    // generation spent before the text encoder ran. The same conversion functions, over raw
    // pointers and split across the host's cores.
    if (dtype_ == DType::F32 && target == DType::BF16) {
        const float* src = static_cast<const float*>(data_);
        BF16* dst = static_cast<BF16*>(out.data_);
        parallel_chunks(numel_, [src, dst](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dst[i] = f32_to_bf16(src[i]);
        });
        return out;
    }
    if (dtype_ == DType::BF16 && target == DType::F32) {
        const BF16* src = static_cast<const BF16*>(data_);
        float* dst = static_cast<float*>(out.data_);
        parallel_chunks(numel_, [src, dst](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dst[i] = bf16_to_f32(src[i]);
        });
        return out;
    }
    for (int64_t i = 0; i < numel_; ++i) out.set(i, get(i));
    return out;
}

Tensor Tensor::to_device() const {
    if (device_ == Device::CUDA) return *this;
    Tensor out(shape_, dtype_, Device::CUDA);
    device::copy_to_device(out.data_, data_, nbytes());
    return out;
}

Tensor Tensor::to_host() const {
    if (device_ == Device::CPU) return *this;
    Tensor out(shape_, dtype_, Device::CPU);
    device::copy_to_host(out.data_, data_, nbytes());
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
