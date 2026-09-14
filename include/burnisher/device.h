// The device boundary: allocation, copies, synchronisation, and the high-water mark.
//
// Declared here and defined in `src/cuda/device.cu` only when CUDA is on. Without CUDA every
// function throws rather than silently doing nothing, because a CPU build that quietly accepted
// a device allocation would produce a tensor pointing at nothing and fault three frames later in
// something that looks like a kernel bug.
//
// `peak_allocated_bytes()` is the memory objective the frontier scores. On a CPU build the
// runtime reports host peak RSS, which is a DIFFERENT resource -- scoring one where the other
// belongs would make the whole memory axis meaningless and would look entirely reasonable.
#pragma once

#include <cstddef>

namespace burnisher {
namespace device {

bool available();
void* alloc(size_t bytes);
void release(void* p);
void copy_to_device(void* dst, const void* src, size_t bytes);
void copy_to_host(void* dst, const void* src, size_t bytes);
void synchronize();

// Page-locked host memory, for staging uploads. A copy from pageable memory is staged by the driver
// through a page-locked buffer of its own first; converting straight into one of ours skips that
// copy, which was two thirds of the time a bf16 generation spent loading its weights.
void* alloc_pinned(size_t bytes);
void release_pinned(void* p);

// Bytes currently and peak allocated through `alloc`. Tracked by this layer rather than read
// from the driver: the driver's figure includes the context and the allocator's own slack, which
// a candidate cannot influence and should not be scored on.
size_t allocated_bytes();
size_t peak_allocated_bytes();
void reset_peak();

}  // namespace device
}  // namespace burnisher
