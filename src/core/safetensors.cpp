#include "burnisher/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace burnisher {
namespace json {

const Value& Value::at(const std::string& key) const {
    if (kind != Kind::Object_) throw std::runtime_error("json: not an object");
    auto it = obj->find(key);
    if (it == obj->end()) throw std::runtime_error("json: no key '" + key + "'");
    return it->second;
}

const Value& Value::at(size_t i) const {
    if (kind != Kind::Array_) throw std::runtime_error("json: not an array");
    if (i >= arr->size()) throw std::runtime_error("json: index out of range");
    return (*arr)[i];
}

bool Value::contains(const std::string& key) const {
    return kind == Kind::Object_ && obj->count(key) != 0;
}

double Value::as_number() const {
    if (kind != Kind::Number) throw std::runtime_error("json: not a number");
    return num;
}

const std::string& Value::as_string() const {
    if (kind != Kind::String) throw std::runtime_error("json: not a string");
    return str;
}

std::vector<int64_t> Value::as_int_array() const {
    if (kind != Kind::Array_) throw std::runtime_error("json: not an array");
    std::vector<int64_t> out;
    for (const auto& v : *arr) out.push_back(static_cast<int64_t>(v.as_number()));
    return out;
}

namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }

    [[noreturn]] void fail(const std::string& what) const {
        std::ostringstream os;
        os << "json: " << what << " at byte " << i;
        throw std::runtime_error(os.str());
    }

    char peek() { ws(); if (i >= s.size()) fail("unexpected end"); return s[i]; }

    void expect(char c) { if (peek() != c) fail(std::string("expected '") + c + "'"); ++i; }

    Value parse_value() {
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': { Value v; v.kind = Value::Kind::String; v.str = parse_string(); return v; }
            case 't': case 'f': return parse_bool();
            case 'n': {
                if (s.compare(i, 4, "null") != 0) fail("bad literal");
                i += 4; return Value{};
            }
            default: return parse_number();
        }
    }

    Value parse_object() {
        expect('{');
        Value v; v.kind = Value::Kind::Object_; v.obj = std::make_shared<Object>();
        if (peek() == '}') { ++i; return v; }
        while (true) {
            std::string key = parse_string();
            expect(':');
            (*v.obj)[key] = parse_value();
            char c = peek();
            if (c == ',') { ++i; continue; }
            if (c == '}') { ++i; return v; }
            fail("expected ',' or '}'");
        }
    }

    Value parse_array() {
        expect('[');
        Value v; v.kind = Value::Kind::Array_; v.arr = std::make_shared<Array>();
        if (peek() == ']') { ++i; return v; }
        while (true) {
            v.arr->push_back(parse_value());
            char c = peek();
            if (c == ',') { ++i; continue; }
            if (c == ']') { ++i; return v; }
            fail("expected ',' or ']'");
        }
    }

    Value parse_bool() {
        Value v; v.kind = Value::Kind::Bool;
        if (s.compare(i, 4, "true") == 0) { v.b = true; i += 4; }
        else if (s.compare(i, 5, "false") == 0) { v.b = false; i += 5; }
        else fail("bad literal");
        return v;
    }

    Value parse_number() {
        size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
                                s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) ++i;
        if (start == i) fail("expected a number");
        Value v; v.kind = Value::Kind::Number;
        v.num = std::stod(s.substr(start, i - start));
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) fail("unterminated escape");
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (i + 4 > s.size()) fail("truncated \\u escape");
                    unsigned cp = std::stoul(s.substr(i, 4), nullptr, 16);
                    i += 4;
                    // Enough UTF-8 for the BMP. Tensor names are ASCII in practice; this is here
                    // so a config file with a non-ASCII comment does not take the loader down.
                    if (cp < 0x80) { out += static_cast<char>(cp); }
                    else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: fail("bad escape");
            }
        }
        fail("unterminated string");
    }
};

}  // namespace

Value parse(const std::string& text) {
    Parser p{text};
    Value v = p.parse_value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing content");
    return v;
}

}  // namespace json

namespace {

DType dtype_from_safetensors(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "BF16") return DType::BF16;
    if (s == "F16") return DType::F16;
    if (s == "F8_E4M3") return DType::FP8E4M3;
    throw std::runtime_error(
        "safetensors: dtype '" + s + "' is not one this runtime reads. Converting a checkpoint "
        "silently would change the oracle; convert it deliberately and pin the result.");
}

}  // namespace

SafeTensors::~SafeTensors() {
    if (base_ && base_ != MAP_FAILED) ::munmap(base_, size_);
    if (fd_ >= 0) ::close(fd_);
}

SafeTensors::SafeTensors(SafeTensors&& o) noexcept { *this = std::move(o); }

SafeTensors& SafeTensors::operator=(SafeTensors&& o) noexcept {
    if (this != &o) {
        path_ = std::move(o.path_);
        fd_ = o.fd_; o.fd_ = -1;
        base_ = o.base_; o.base_ = nullptr;
        size_ = o.size_; o.size_ = 0;
        data_offset_ = o.data_offset_;
        entries_ = std::move(o.entries_);
        index_ = std::move(o.index_);
    }
    return *this;
}

SafeTensors SafeTensors::open(const std::string& path) {
    SafeTensors st;
    st.path_ = path;
    st.fd_ = ::open(path.c_str(), O_RDONLY);
    if (st.fd_ < 0) throw std::runtime_error("safetensors: cannot open " + path);
    struct stat sb {};
    if (::fstat(st.fd_, &sb) != 0) throw std::runtime_error("safetensors: cannot stat " + path);
    st.size_ = static_cast<size_t>(sb.st_size);
    if (st.size_ < 8) throw std::runtime_error("safetensors: " + path + " is too short");
    st.base_ = ::mmap(nullptr, st.size_, PROT_READ, MAP_PRIVATE, st.fd_, 0);
    if (st.base_ == MAP_FAILED) throw std::runtime_error("safetensors: cannot mmap " + path);

    uint64_t header_len = 0;
    std::memcpy(&header_len, st.base_, 8);
    if (header_len == 0 || header_len + 8 > st.size_) {
        throw std::runtime_error("safetensors: header length " + std::to_string(header_len) +
                                 " does not fit in " + std::to_string(st.size_) + " bytes");
    }
    std::string header(static_cast<const char*>(st.base_) + 8, header_len);
    st.data_offset_ = 8 + header_len;

    json::Value doc = json::parse(header);
    if (!doc.is_object()) throw std::runtime_error("safetensors: header is not an object");
    for (const auto& kv : *doc.obj) {
        if (kv.first == "__metadata__") continue;
        const json::Value& e = kv.second;
        TensorEntry t;
        t.name = kv.first;
        t.dtype = dtype_from_safetensors(e.at("dtype").as_string());
        t.shape = e.at("shape").as_int_array();
        auto off = e.at("data_offsets").as_int_array();
        if (off.size() != 2) throw std::runtime_error("safetensors: bad data_offsets for " +
                                                      t.name);
        t.begin = static_cast<size_t>(off[0]);
        t.end = static_cast<size_t>(off[1]);
        const size_t want = static_cast<size_t>(numel_of(t.shape) * dtype_bytes(t.dtype));
        if (t.end < t.begin || t.end - t.begin != want) {
            throw std::runtime_error("safetensors: " + t.name + " declares " +
                                     std::to_string(t.end - t.begin) + " bytes for a shape "
                                     "needing " + std::to_string(want));
        }
        if (st.data_offset_ + t.end > st.size_) {
            throw std::runtime_error("safetensors: " + t.name + " runs past the end of the file");
        }
        st.index_[t.name] = st.entries_.size();
        st.entries_.push_back(std::move(t));
    }
    return st;
}

bool SafeTensors::has(const std::string& name) const { return index_.count(name) != 0; }

Tensor SafeTensors::get(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        // A checkpoint whose tensor names moved is the commonest way a model load goes wrong,
        // and a bare "not found" is useless among two thousand keys. Offer the near misses.
        std::vector<std::string> near;
        for (const auto& kv : index_) {
            if (kv.first.find(name) != std::string::npos ||
                name.find(kv.first) != std::string::npos) {
                near.push_back(kv.first);
                if (near.size() >= 5) break;
            }
        }
        std::ostringstream os;
        os << "safetensors: " << path_ << " has no tensor '" << name << "'";
        if (!near.empty()) {
            os << ". Did you mean: ";
            for (size_t i = 0; i < near.size(); ++i) os << (i ? ", " : "") << near[i];
        } else {
            os << " (and nothing resembling it; the checkpoint layout may have changed, which "
                  "is a change to the oracle and needs a new pin)";
        }
        throw std::runtime_error(os.str());
    }
    const TensorEntry& e = entries_[it->second];
    void* p = static_cast<char*>(base_) + data_offset_ + e.begin;
    return Tensor::view(p, e.shape, e.dtype);
}

}  // namespace burnisher
