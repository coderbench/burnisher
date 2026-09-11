#include <map>
#include <stdexcept>
#include <string>

#include "burnisher/ops.h"

namespace burnisher {

void register_all_ops() {
    register_builtin_cpu_ops();
#ifdef BURNISHER_CUDA
    register_cuda_ops();
#endif
}

namespace {
bool op_has(const std::string& op, const std::string& name) {
    if (op == "gemm") return GemmRegistry::instance().has(name);
    if (op == "attention") return AttentionRegistry::instance().has(name);
    if (op == "norm") return NormRegistry::instance().has(name);
    if (op == "modulate") return ModulateRegistry::instance().has(name);
    if (op == "activation") return ActivationRegistry::instance().has(name);
    if (op == "conv2d") return Conv2dRegistry::instance().has(name);
    if (op == "add") return AddRegistry::instance().has(name);
    if (op == "chunk") return ChunkRegistry::instance().has(name);
    if (op == "patch") return PatchRegistry::instance().has(name);
    if (op == "gather") return GatherRegistry::instance().has(name);
    if (op == "scale") return ScaleRegistry::instance().has(name);
    if (op == "repeat") return RepeatRegistry::instance().has(name);
    if (op == "guidance") return GuidanceRegistry::instance().has(name);
    if (op == "upsample") return UpsampleRegistry::instance().has(name);
    if (op == "transpose") return TransposeRegistry::instance().has(name);
    throw std::runtime_error("unknown op '" + op + "'");
}
}  // namespace

// One `--impl` name applies to whichever ops register it and leaves the rest on `stock`.
//
// This is what lets a submission A/B a single kernel without silently changing the rest of the
// pipeline underneath the comparison. The fallback is the dangerous part -- a silent fallback
// is how an arm comes to measure a configuration nobody asked for -- so it is not silent: the
// runtime reports the resolved name for EVERY op in its `effective.impls` block, and
// eval/runner.py refuses a run whose report disagrees with the request.
//
// The one case that is an error rather than a fallback is a name no op registers at all. That
// is a typo or a kernel that failed to register, and running `stock` everywhere while reporting
// success would be the worst possible answer.
std::string resolve_impl(const std::string& op, const std::string& requested) {
    if (requested.empty() || requested == "stock") return "stock";
    if (op_has(op, requested)) return requested;
    return "stock";
}

bool any_op_has_impl(const std::string& name) {
    for (const char* op : {"gemm", "attention", "norm", "modulate", "activation", "conv2d",
                           "add", "chunk", "patch", "upsample", "transpose", "gather",
                           "scale", "repeat", "guidance"}) {
        if (op_has(op, name)) return true;
    }
    return false;
}

std::map<std::string, std::string> resolve_all(const std::string& requested) {
    if (!requested.empty() && requested != "stock" && !any_op_has_impl(requested)) {
        throw std::runtime_error(
            "no op registers an implementation named '" + requested + "'. This is a typo or a "
            "kernel that failed to register; running 'stock' everywhere and reporting success "
            "would measure a configuration nobody asked for. `burnisher info --impls` lists "
            "what this build actually contains.");
    }
    std::map<std::string, std::string> out;
    for (const char* op : {"gemm", "attention", "norm", "modulate", "activation", "conv2d",
                           "add", "chunk", "patch", "upsample", "transpose", "gather",
                           "scale", "repeat", "guidance"}) {
        out[op] = resolve_impl(op, requested);
    }
    return out;
}

}  // namespace burnisher
