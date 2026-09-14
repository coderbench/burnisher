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

// One `--impl` name applies to whichever ops register it and leaves the rest on the baseline for
// the device the run is placed on.
//
// This is what lets a submission A/B a single kernel without silently changing the rest of the
// pipeline underneath the comparison. The fallback is the dangerous part -- a silent fallback is
// how an arm comes to measure a configuration nobody asked for -- so it is not silent: the
// runtime reports the resolved name for EVERY op in its `effective.impls` block, and
// eval/runner.py refuses a run whose report disagrees with the request.
//
// The baseline is per-device, and that is the whole point. `stock` is a set of HOST kernels. On a
// CUDA run every intermediate lives in device memory, so falling back to `stock` hands a host
// kernel device pointers -- the exact fault the Device field on ImplSelection exists to prevent.
// This is not a corner case: it is what EVERY single-kernel submission looks like. `--impl
// cuda-sdpa` registers one attention variant and nothing else, so fourteen of fifteen ops
// take the fallback, and with a host baseline fourteen of fifteen ops fault. The first candidate
// this harness ever scored died here, in the gate, before it timed anything -- which is the gate
// working, but the bug was mine and every miner would have hit it on their first submission.
//
// Two cases stay hard errors rather than fallbacks:
//   - a name no op registers at all -- a typo, or a kernel that failed to register;
//   - a device baseline an op does not provide -- there is no legal host substitute for it.
std::string resolve_impl(const std::string& op, const std::string& requested, Device device) {
    const std::string baseline = (device == Device::CUDA) ? "cuda" : "stock";
    if (!requested.empty() && requested != "stock" && op_has(op, requested)) return requested;
    if (baseline != "stock" && !op_has(op, baseline)) {
        throw std::runtime_error(
            "op '" + op + "' has no '" + baseline + "' implementation to fall back on for this "
            "device. Falling back to a host kernel would hand it device pointers; running it "
            "anyway and reporting success would measure a configuration nobody asked for.");
    }
    return baseline;
}

bool any_op_has_impl(const std::string& name) {
    for (const char* op : {"gemm", "attention", "norm", "modulate", "activation", "conv2d",
                           "add", "chunk", "patch", "upsample", "transpose", "gather",
                           "scale", "repeat", "guidance"}) {
        if (op_has(op, name)) return true;
    }
    return false;
}

}  // namespace burnisher
