#include "core/host_memory.h"

#include <algorithm>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace ninfer {

void warm_readonly_host_memory(std::span<const std::byte> bytes) {
    if (bytes.empty()) { return; }
#ifdef _WIN32
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    const std::size_t page_bytes = info.dwPageSize;
#else
    const std::size_t page_bytes = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#endif
    // Bounded sequential reads avoid millions of serialized random file faults.
    // Touch after the read-ahead hint as it does not itself establish the working set.
    constexpr std::size_t kReadAheadBytes = 64ULL << 20;
    for (std::size_t offset = 0; offset < bytes.size(); offset += kReadAheadBytes) {
        const auto chunk = bytes.subspan(offset, std::min(kReadAheadBytes, bytes.size() - offset));
#ifdef _WIN32
        WIN32_MEMORY_RANGE_ENTRY range{const_cast<std::byte*>(chunk.data()), chunk.size()};
        (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
#else
        const auto address = reinterpret_cast<std::uintptr_t>(chunk.data());
        const auto aligned = address - address % page_bytes;
        (void)::madvise(reinterpret_cast<void*>(aligned), chunk.size() + address - aligned,
                        MADV_WILLNEED);
#endif
        const volatile std::byte* data = chunk.data();
        for (std::size_t page = 0; page < chunk.size(); page += page_bytes) {
            (void)data[page];
        }
        (void)data[chunk.size() - 1];
    }
}

} // namespace ninfer
