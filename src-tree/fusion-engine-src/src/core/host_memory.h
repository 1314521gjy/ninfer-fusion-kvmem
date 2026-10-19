#pragma once

#include <cstddef>
#include <span>

namespace ninfer {

// Read ahead and fault in a known, read-only host working set before latency-sensitive
// execution. Does not allocate another copy or pin pages against OS reclamation.
void warm_readonly_host_memory(std::span<const std::byte> bytes);

} // namespace ninfer
