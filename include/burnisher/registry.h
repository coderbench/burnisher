// Named implementations of an op, selectable at run time, coexisting in one binary.
//
// This is the single most important structural decision in the runtime, and it exists to serve
// the measurement rather than the code. A contributor adds a kernel by REGISTERING a new name
// beside the old one, never by replacing a file. Three things follow, and all three are things
// a file replacement cannot give you:
//
//   * base and candidate run in ONE process, with ONE model load, under ONE thermal state --
//     so the paired interleaved delta the scorer needs is a delta between two kernels and not
//     between two program startups;
//   * the old implementation stays runnable forever, so a regression can be bisected to a
//     kernel rather than to a commit range;
//   * `burnisher bench --impl X` fails loudly when X is not registered, instead of silently
//     measuring whatever the build happened to contain.
//
// The last point is a guard with teeth: the harness compares the `impl` the runtime REPORTS
// against the one it asked for, so an arm that fell back is refused rather than scored.
#pragma once

#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace burnisher {

struct ImplInfo {
    std::string name;
    std::string doc;
};

template <typename Args>
class OpRegistry {
  public:
    using Fn = std::function<void(const Args&)>;

    static OpRegistry& instance() {
        static OpRegistry r;
        return r;
    }

    void add(const std::string& name, Fn fn, const std::string& doc) {
        if (impls_.count(name)) {
            throw std::runtime_error("duplicate implementation '" + name + "' for op '" +
                                     op_name_ + "'; two kernels under one name cannot be A/B'd");
        }
        impls_.emplace(name, std::move(fn));
        docs_.emplace(name, doc);
    }

    const Fn& get(const std::string& name) const {
        auto it = impls_.find(name);
        if (it == impls_.end()) {
            std::ostringstream os;
            os << "no implementation '" << name << "' for op '" << op_name_ << "'. Registered: ";
            bool first = true;
            for (const auto& kv : impls_) {
                os << (first ? "" : ", ") << kv.first;
                first = false;
            }
            os << ". A run that silently fell back to a different kernel would produce a "
                  "perfectly good number for a configuration nobody asked for.";
            throw std::runtime_error(os.str());
        }
        return it->second;
    }

    bool has(const std::string& name) const { return impls_.count(name) != 0; }

    std::vector<ImplInfo> list() const {
        std::vector<ImplInfo> out;
        for (const auto& kv : docs_) out.push_back({kv.first, kv.second});
        return out;
    }

    void set_op_name(const std::string& n) { op_name_ = n; }
    const std::string& op_name() const { return op_name_; }

  private:
    std::string op_name_ = "?";
    std::map<std::string, Fn> impls_;
    std::map<std::string, std::string> docs_;
};

// Registration is an explicit call, not a static initialiser.
//
// The tempting alternative is a file-scope Registrar object so a kernel registers itself next to
// its own definition. It does not survive static linking: a translation unit whose symbols are
// otherwise unreferenced is dropped from the archive, the registration vanishes, and the failure
// mode is `--impl X` reporting that X does not exist in a build that plainly contains it. Every
// backend therefore exposes one `register_*_ops()` that the runtime calls, and adding a kernel
// means adding one line there. One line of bookkeeping beats an hour of confusion.
template <typename Args>
inline void register_impl(const std::string& op, const std::string& name,
                          typename OpRegistry<Args>::Fn fn, const std::string& doc) {
    OpRegistry<Args>::instance().set_op_name(op);
    OpRegistry<Args>::instance().add(name, std::move(fn), doc);
}

}  // namespace burnisher
