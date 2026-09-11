// Read a safetensors checkpoint by memory-mapping it.
//
// Mapped, not loaded: the pinned text encoder is 19 GB on disk and the point of a native runtime
// is that it does not need a 19 GB copy in host RAM before it can start moving tensors to the
// device. The mapping is read-only and the Tensors handed out are VIEWS into it, so a caller
// that wants a different dtype converts explicitly and can see the copy in their own code.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "burnisher/tensor.h"

namespace burnisher {

struct TensorEntry {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    size_t begin;
    size_t end;
};

class SafeTensors {
  public:
    static SafeTensors open(const std::string& path);
    ~SafeTensors();
    SafeTensors(SafeTensors&&) noexcept;
    SafeTensors& operator=(SafeTensors&&) noexcept;
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    SafeTensors() = default;

    bool has(const std::string& name) const;
    // Throws naming the closest existing keys. A checkpoint whose tensor names moved is the most
    // common way a model load goes wrong, and "key not found" without context is useless when
    // there are two thousand of them.
    Tensor get(const std::string& name) const;
    const std::vector<TensorEntry>& entries() const { return entries_; }
    size_t total_bytes() const { return size_; }
    const std::string& path() const { return path_; }

  private:
    std::string path_;
    int fd_ = -1;
    void* base_ = nullptr;
    size_t size_ = 0;
    size_t data_offset_ = 0;
    std::vector<TensorEntry> entries_;
    std::map<std::string, size_t> index_;
};

// Minimal JSON, enough for a safetensors header and for the runtime's own config files.
// Vendored rather than depended on: this runtime is meant to build from source on a pinned box
// with nothing but a compiler, and one small parser is cheaper than a submodule.
namespace json {

struct Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    enum class Kind { Null, Bool, Number, String, Array_, Object_ } kind = Kind::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::shared_ptr<Array> arr;
    std::shared_ptr<Object> obj;

    bool is_object() const { return kind == Kind::Object_; }
    bool is_array() const { return kind == Kind::Array_; }
    const Value& at(const std::string& key) const;
    const Value& at(size_t i) const;
    bool contains(const std::string& key) const;
    double as_number() const;
    const std::string& as_string() const;
    std::vector<int64_t> as_int_array() const;
};

Value parse(const std::string& text);

}  // namespace json
}  // namespace burnisher
