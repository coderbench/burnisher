// A tensor: shape, dtype, and bytes. Deliberately small.
//
// There is no autograd, no broadcasting engine and no lazy graph here. Every op in this runtime
// takes explicit shapes and writes an explicit destination, because the thing being optimized is
// the kernel and anything that stands between a contributor and the kernel is in the way.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "burnisher/dtype.h"

namespace burnisher {

enum class Device { CPU, CUDA };

class Tensor {
  public:
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, DType dtype, Device device = Device::CPU);

    static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device d = Device::CPU);
    // A view over memory this tensor does not own -- used by the safetensors loader so a 19 GB
    // checkpoint is mapped rather than copied.
    static Tensor view(void* data, std::vector<int64_t> shape, DType dtype,
                       Device d = Device::CPU);

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t dim(size_t i) const { return shape_.at(i); }
    size_t rank() const { return shape_.size(); }
    int64_t numel() const { return numel_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t nbytes() const;
    bool defined() const { return data_ != nullptr; }

    void* data() { return data_; }
    const void* data() const { return data_; }
    template <typename T> T* as() { return static_cast<T*>(data_); }
    template <typename T> const T* as() const { return static_cast<const T*>(data_); }

    float get(int64_t flat) const;
    void set(int64_t flat, float v);

    Tensor to(DType target) const;
    Tensor reshape(std::vector<int64_t> shape) const;
    std::string describe() const;

  private:
    std::vector<int64_t> shape_;
    int64_t numel_ = 0;
    DType dtype_ = DType::F32;
    Device device_ = Device::CPU;
    void* data_ = nullptr;
    std::shared_ptr<void> owned_;
};

int64_t numel_of(const std::vector<int64_t>& shape);

}  // namespace burnisher
