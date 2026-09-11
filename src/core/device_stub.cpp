// The device boundary on a build WITHOUT CUDA.
//
// Every entry point throws. The alternative -- returning null, or a host pointer -- gives a
// tensor that points at nothing and faults later inside something that looks like a kernel bug.
// Failing at the allocation says what actually went wrong.
#include <stdexcept>

#include "burnisher/device.h"

#ifndef BURNISHER_CUDA
namespace burnisher {
namespace device {

namespace {
[[noreturn]] void no_cuda(const char* what) {
    throw std::runtime_error(
        std::string("device::") + what + ": this binary was built without CUDA. Build with "
        "scripts/build_cuda.sh. A CPU build cannot allocate, copy to, or synchronise a device, "
        "and pretending otherwise would produce a tensor pointing at nothing.");
}
}  // namespace

bool available() { return false; }
void* alloc(size_t) { no_cuda("alloc"); }
void release(void*) {}
void copy_to_device(void*, const void*, size_t) { no_cuda("copy_to_device"); }
void copy_to_host(void*, const void*, size_t) { no_cuda("copy_to_host"); }
void synchronize() {}
size_t allocated_bytes() { return 0; }
size_t peak_allocated_bytes() { return 0; }
void reset_peak() {}

}  // namespace device
}  // namespace burnisher
#endif
