#include "models/qwen3_8_flash_next/impl/load/materialized.h"

#include "artifact/typed_binding.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_8_flash_next/impl/load/quantize_nvfp4_expert_bank.h"
#include "models/qwen3_8_flash_next/impl/load/quantize_output_head.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
// WIN32_LEAN_AND_MEAN/NOMINMAX before windows.h: PrefetchVirtualMemory is the only Win32 call this
// file needs, and the lean header avoids the min/max macros colliding with <algorithm>.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ------------------------------------------------------------------------------------------------
// Declared HERE, not in artifact/materializer.h, on purpose. This tree's header dependency scan is
// broken: a header edit forces a touch-all rebuild, and missing one of those shows up as "everything
// links green, then the process reads a garbage pointer". The single extra entry point the pinned
// read path needs is therefore declared at its point of use and defined in artifact/materializer.cpp
// -- both are .cpp files, so no header churn is involved.
// ------------------------------------------------------------------------------------------------
namespace ninfer::artifact {
std::uint64_t artifact_object_offset(const MaterializedArtifact& artifact, ObjectHandle handle);
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

std::vector<std::int32_t> load_flash_next_draft_shortlist(std::uint32_t K) {
    std::vector<std::int32_t> shortlist(K);
    const std::string i32_name = (K == 32'768) ? "shortlist_32k.i32" : ((K == 65'536) ? "shortlist_65k.i32" : "");
    std::vector<std::string> candidate_paths;
    if (const char* env_path = std::getenv("NINFER_RANKING_PATH"); env_path != nullptr && *env_path != '\0') {
        candidate_paths.emplace_back(env_path);
    }
    if (!i32_name.empty()) {
        candidate_paths.push_back("tools/freq_corpus/fixtures/ranking/" + i32_name);
        candidate_paths.push_back("../tools/freq_corpus/fixtures/ranking/" + i32_name);
        candidate_paths.push_back("E:/NInfer.gemini2/tools/freq_corpus/fixtures/ranking/" + i32_name);
    }
    candidate_paths.push_back("tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64");
    candidate_paths.push_back("../tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64");
    candidate_paths.push_back("E:/NInfer.gemini2/tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64");

    for (const auto& path_str : candidate_paths) {
        std::ifstream file(path_str, std::ios::binary);
        if (!file.is_open()) { continue; }

        file.seekg(0, std::ios::end);
        const auto file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size >= static_cast<std::streamoff>(K * sizeof(std::int32_t)) &&
            path_str.ends_with(".i32")) {
            file.read(reinterpret_cast<char*>(shortlist.data()), K * sizeof(std::int32_t));
            if (file.gcount() == static_cast<std::streamsize>(K * sizeof(std::int32_t))) {
                return shortlist;
            }
        }

        constexpr std::size_t kVocab = 248'320;
        if (file_size >= static_cast<std::streamoff>(kVocab * sizeof(std::int64_t))) {
            std::vector<std::int64_t> counts(kVocab);
            file.read(reinterpret_cast<char*>(counts.data()), kVocab * sizeof(std::int64_t));
            if (file.gcount() == static_cast<std::streamsize>(kVocab * sizeof(std::int64_t))) {
                std::vector<std::int32_t> indices(kVocab);
                std::iota(indices.begin(), indices.end(), 0);
                std::stable_sort(indices.begin(), indices.end(), [&](std::int32_t a, std::int32_t b) {
                    return counts[a] > counts[b];
                });
                std::copy_n(indices.begin(), K, shortlist.begin());
                return shortlist;
            }
        }
    }

    std::iota(shortlist.begin(), shortlist.end(), 0);
    return shortlist;
}

using artifact::NumericFormat;

Tensor bf16_tensor(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                   std::initializer_list<std::int32_t> shape) {
    return artifact::materialized_tensor(backing, handle, NumericFormat::BF16, shape);
}

Weight bf16_weight(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                   std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(backing, handle, NumericFormat::BF16, rows, columns);
}

Weight fp8_weight(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                  std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(backing, handle, NumericFormat::FP8_E4M3FN_ROW_F32S, rows,
                                         columns);
}

namespace {

// Per-object byte digest export, gated by NINFER_FLASH_NEXT_OBJECT_DIGEST with `read_flag_once`
// semantics (set and not "0"). The older PLE gate only checked presence, so `=0` still turned it on.
// Each line carries the artifact's OWN object id, the byte count, a sha256 over exactly the span the
// consumer was handed, the VirtualQuery memory type of that span, and the object's byte offset in the
// entry file. An outside comparator can then check three separate things: the bytes equal the
// artifact's, the span really lives in the file mapping (MEM_MAPPED, not MEM_PRIVATE), and the span
// starts where the file says it does (offset, and AllocationBase == data - offset).
bool object_digest_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_OBJECT_DIGEST");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Whether the PLE table gets warmed into memory at load. Default OFF: with the table left in the
// container's mapping the rows are demand-paged, and warming all 128 shards touches 29.80 GiB --
// measured on this machine, touching 76 GiB of the artifact leaves `Available` at 6.18 GiB, so warming
// PLE plus routing experts would exhaust what the run needs.
bool flash_next_ple_warm_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_PLE_WARM");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

[[nodiscard]] const char* memory_type_name(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    if (::VirtualQuery(address, &info, sizeof(info)) == 0) { return "QUERY_FAILED"; }
    switch (info.Type) {
    case MEM_IMAGE: return "MEM_IMAGE";
    case MEM_MAPPED: return "MEM_MAPPED";
    case MEM_PRIVATE: return "MEM_PRIVATE";
    default: return "MEM_OTHER";
    }
}

// A mapped read that cannot name its object is useless when it fails: the error it would otherwise
// surface ("object handle does not name a mapped tensor") does not say which object, and fail-fast
// stops at the first one.
[[nodiscard]] std::span<const std::byte> mapped_bytes_named(
    const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle) {
    try {
        return backing.mapped_tensor_bytes(handle);
    } catch (const artifact::ArtifactError& error) {
        throw artifact::ArtifactError(std::string(backing.object_id(handle)) + ": " + error.what());
    }
}

void report_object_digest(const artifact::MaterializedArtifact& backing,
                          artifact::ObjectHandle handle, std::span<const std::byte> bytes) {
    if (!object_digest_enabled()) { return; }
    const std::string id(backing.object_id(handle));
    std::uint64_t file_offset = 0;
    bool is_mapped            = true;
    try {
        file_offset = backing.mapped_file_offset(handle);
    } catch (const artifact::ArtifactError&) {
        is_mapped = false;
    }
    const auto digest = qwen3_5::frontend::sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
    std::fprintf(stdout, "object_digest id=%s bytes=%llu sha256=%s source=%s offset=%s%llu\n",
                 id.c_str(), static_cast<unsigned long long>(bytes.size()),
                 qwen3_5::frontend::sha256_hex(digest).c_str(), memory_type_name(bytes.data()),
                 is_mapped ? "" : "-", static_cast<unsigned long long>(file_offset));
    std::fflush(stdout);
}

} // namespace

// Host-mapped BF16 payload of a tensor bound with retain_mapped_tensor. The size check is the
// only guard against quantizing a differently shaped tensor: the mapping is raw file bytes.
std::span<const std::byte> mapped_bf16_rows(const artifact::MaterializedArtifact& backing,
                                            artifact::ObjectHandle handle, std::int32_t rows,
                                            std::int32_t columns) {
    const std::span<const std::byte> bytes = mapped_bytes_named(backing, handle);
    const std::size_t expected =
        static_cast<std::size_t>(rows) * columns * sizeof(std::uint16_t);
    if (bytes.size() != expected) {
        throw std::logic_error("Flash-Next mapped BF16 tensor has an unexpected payload size");
    }
    report_object_digest(backing, handle, bytes);
    return bytes;
}

HyperConnectionWeights load_hyper(const HyperConnectionPlan& plan,
                                  const artifact::MaterializedArtifact& backing) {
    return {
        .block_inject   = bf16_weight(backing, plan.block_inject, 4, 10'240),
        .norm           = bf16_tensor(backing, plan.norm, {10'240}),
        .input_mix_down = bf16_weight(backing, plan.input_mix_down, 320, 10'240),
        .input_mix_up   = bf16_weight(backing, plan.input_mix_up, 10'240, 320),
    };
}

HyperMixerWeights load_mixer(const HyperMixerPlan& plan,
                             const artifact::MaterializedArtifact& backing) {
    return {
        .norm           = bf16_tensor(backing, plan.norm, {10'240}),
        .input_mix_down = bf16_weight(backing, plan.input_mix_down, 320, 10'240),
        .input_mix_up   = bf16_weight(backing, plan.input_mix_up, 10'240, 320),
    };
}

// Pinned expert banks live for the process lifetime on purpose: the model view points into them and
// there is nothing to release before exit. Keeps the cudaHostAlloc'ed blocks reachable so they are
// not dropped silently, and gives the load path something to account for.
std::vector<void*>& pinned_expert_bank_registry() {
    static std::vector<void*> registry;
    return registry;
}

// Host-mapped NVFP4 expert bank. Same geometry contract as the device view, over the artifact's
// read-only file mapping: make_nvfp4_expert_bank_view checks payload_bytes against the layout's
// encoded size and derives both per-expert strides by division, so a differently shaped mapping
// cannot be silently reinterpreted. The three pointers are host addresses; the MoE leaf stages the
// routed experts into device memory before launching (impl/expert_staging.h).
//
// With NINFER_FLASH_NEXT_EXPERTS_PIN the bank is copied once into CUDA-pinned memory and the view
// points there instead -- the community's "load-mode none + full pin" setup. A memory-mapped source
// cannot be registered, so every device copy from it goes through the driver's internal pageable
// staging; the same call sequence measured 8.61 GB/s from the mapping and 22.55 GB/s from pinned
// memory against a 23.97 GB/s one-big-copy ceiling (lever_probe.exe on this machine).
// ------------------------------------------------------------------------------------------------
// P1: read the expert bank straight out of the artifact instead of memcpy'ing it out of the file
// mapping. Community answer to exactly this problem, in order of how directly it applies:
//
//  * llama.cpp PR #26014 "Windows unbuffered model load" (still open, 2026-07-22): "For Windows this
//    change is needed, as using mmap of large models on setups with small CPU RAM lead to Sysmem
//    pressure and eviction of runtime pages", and its already-merged Linux counterpart (#18012,
//    DirectIO) took model loading on a PCIe 5.0 NVMe "from 1-3 GB/s to >9 GB/s". The mechanism is
//    FILE_FLAG_NO_BUFFERING: DMA straight into our buffer instead of copying through the cache
//    manager. Our artifact is 113 GB, so the cache cannot hold it and buffered reads just evict.
//  * Helldez/BigMoeOnEdge (Windows 11 + NVMe): a LIVE file mapping serialises concurrent unbuffered
//    reads on the same file object -- 2295-2665 MiB/s across four lanes with no mapping, 895-930 with
//    one alive, and 2402 only after dropping the mapping AND reopening the channels independently.
//    That is exactly why our earlier "split the memcpy across four threads" attempt measured no
//    change: the fix is INDEPENDENT HANDLES, not more threads.
//
// So: N independent handles, bypassing the mapping. Unbuffered needs the file offset, the length and
// the buffer all on a 4096-byte boundary, and our object offsets are only guaranteed 256-byte aligned
// (G7), so mode 2 reads through an aligned bounce buffer. Modes:
//   NINFER_FLASH_NEXT_PIN_READFILE  0/unset -> memcpy from the mapping (bit-identical to before)
//                                   1       -> N independent handles, buffered, FILE_FLAG_SEQUENTIAL_SCAN
//                                   2       -> as 1 plus FILE_FLAG_NO_BUFFERING (the community answer)
// Any failure falls back to the memcpy path, so a missing env or an unreadable file can only cost
// speed, never correctness.
int pin_readfile_mode() {
    static const int mode = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_PIN_READFILE");
        if (value == nullptr || value[0] == '\0') { return 0; }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed <= 0 ? 0 : (parsed >= 2 ? 2 : 1);
    }();
    return mode;
}

std::wstring widen_utf8_path(const char* text) {
    if (text == nullptr || text[0] == '\0') { return {}; }
    const int needed = ::MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (needed <= 1) { return {}; }
    std::wstring out(static_cast<std::size_t>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, text, -1, out.data(), needed);
    out.resize(static_cast<std::size_t>(needed - 1));
    return out;
}

// One lane = one INDEPENDENT handle reading its own contiguous slice of the range.
void read_one_lane(const std::wstring& path, std::uint64_t file_offset, std::byte* destination,
                   std::size_t begin, std::size_t end, int mode, char* ok) {
    constexpr std::size_t kChunk  = 8ULL * 1024ULL * 1024ULL;
    constexpr std::size_t kSector = 4096ULL;
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN;
    if (mode >= 2) { flags |= FILE_FLAG_NO_BUFFERING; }
    HANDLE fd = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_EXISTING, flags, nullptr);
    if (fd == INVALID_HANDLE_VALUE) { *ok = 0; return; }
    std::vector<std::byte> bounce;
    if (mode >= 2) { bounce.resize(kChunk + 2 * kSector); }
    std::size_t done = 0;
    bool good        = true;
    while (good && done < end - begin) {
        const std::size_t want = std::min(kChunk, end - begin - done);
        const std::uint64_t at = file_offset + begin + done;
        std::size_t shift      = 0;
        std::size_t read_len   = want;
        std::byte* into        = destination + begin + done;
        if (mode >= 2) {
            const std::uint64_t aligned_at = at & ~static_cast<std::uint64_t>(kSector - 1);
            shift     = static_cast<std::size_t>(at - aligned_at);
            read_len  = ((shift + want + kSector - 1) / kSector) * kSector;
            into      = bounce.data();
        }
        LARGE_INTEGER pos;
        pos.QuadPart = static_cast<LONGLONG>(mode >= 2 ? (at - shift) : at);
        if (!::SetFilePointerEx(fd, pos, nullptr, FILE_BEGIN)) { good = false; break; }
        DWORD got = 0;
        if (!::ReadFile(fd, into, static_cast<DWORD>(read_len), &got, nullptr) || got == 0) {
            good = false;
            break;
        }
        const std::size_t usable = got > shift ? std::min<std::size_t>(want, got - shift) : 0;
        if (usable == 0) { good = false; break; }
        if (mode >= 2) { std::memcpy(destination + begin + done, bounce.data() + shift, usable); }
        done += usable;
    }
    ::CloseHandle(fd);
    if (!good || done != end - begin) { *ok = 0; }
}

// Returns false (leaving the caller to fall back to memcpy) if anything is missing or fails.
bool read_bank_from_file(const std::wstring& path, std::uint64_t file_offset, void* destination,
                         std::size_t bytes, int mode, int lanes) {
    if (path.empty() || bytes == 0 || mode <= 0 || lanes < 1) { return false; }
    constexpr std::size_t kLaneQuantum = 4ULL * 1024ULL * 1024ULL;
    const std::size_t lane_target =
        (bytes + static_cast<std::size_t>(lanes) - 1) / static_cast<std::size_t>(lanes);
    const std::size_t lane_bytes =
        ((lane_target + kLaneQuantum - 1) / kLaneQuantum) * kLaneQuantum;
    std::vector<char> ok(static_cast<std::size_t>(lanes), 1);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(lanes));
    auto* base = static_cast<std::byte*>(destination);
    for (int lane = 0; lane < lanes; ++lane) {
        workers.emplace_back([&, lane] {
            const std::size_t begin = lane_bytes * static_cast<std::size_t>(lane);
            if (begin >= bytes) { return; }
            const std::size_t end = std::min(bytes, begin + lane_bytes);
            read_one_lane(path, file_offset, base, begin, end, mode, &ok[static_cast<std::size_t>(lane)]);
        });
    }
    for (auto& worker : workers) { worker.join(); }
    for (const char flag : ok) {
        if (flag == 0) { return false; }
    }
    return true;
}

Nvfp4ExpertBankView host_nvfp4_expert_bank_view(const artifact::MaterializedArtifact& backing,
                                                artifact::ObjectHandle handle,
                                                std::int32_t experts, std::int32_t rows,
                                                std::int32_t columns) {
    const std::span<const std::byte> bytes = mapped_bytes_named(backing, handle);
    const void* source                     = bytes.data();
    const std::size_t source_bytes         = bytes.size();
    report_object_digest(backing, handle, bytes);

    // Prefetch is not a pinning feature: with the bank left in the container's mapping the bytes come
    // out of that mapping, and warming its pages is exactly what this does. It used to sit inside the
    // PIN branch below, so PIN=0 -- the configuration this trip runs -- never reached it. Default OFF:
    // warming all 96 banks touches 63.28 GiB, measured to leave `Available` at ~18 GiB.
    static const bool read_ahead = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_PIN_PREFETCH");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    if (read_ahead) {
        WIN32_MEMORY_RANGE_ENTRY range{};
        range.VirtualAddress = const_cast<void*>(source);
        range.NumberOfBytes  = source_bytes;
        ::PrefetchVirtualMemory(::GetCurrentProcess(), 1, &range, 0);
    }

    if (flash_next_experts_pinned_enabled()) {
        void* pinned             = nullptr;
        const cudaError_t status = cudaHostAlloc(&pinned, source_bytes, cudaHostAllocDefault);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("Flash-Next expert bank pinning failed at ") +
                                     std::to_string(source_bytes >> 20) + " MiB: " +
                                     cudaGetErrorString(status));
        }
        const auto pinned_started = std::chrono::steady_clock::now();
        // Measured on this machine: copying from the artifact's file mapping runs at 0.72 GB/s
        // (96 banks, 88.4 s), while a plain ReadFile of the same file reaches 1.94 GB/s and four
        // independent handles reach 4.73 GB/s (artifact_read_bench.py). The difference is that a
        // mapping faults one 4 KiB page at a time, whereas a buffered read lets the cache manager
        // issue large sequential I/O. Four threads over one mapping did NOT help (Windows serialises
        // faults on a single file object), so ask the OS for bulk read-ahead instead:
        // PrefetchVirtualMemory over the whole bank, then copy. Pure hint -- it changes timing only,
        // never the bytes.
        // One-shot read-ahead over the whole bank. MEASURED, both directions:
        //   no prefetch        : 0.72 GB/s, 88.4 s of pinning, target_load_ms 119383
        //   one-shot prefetch  : 0.91 GB/s, 69.2 s,             target_load_ms 100569  <-- kept
        //   chunked pipeline   : 0.68 GB/s, 93.3 s,             target_load_ms 132698  <-- rejected
        // Chunking (prefetch the next 64 MiB while copying the current one) is worse than doing
        // nothing: interleaving the prefetch with the copy makes them contend for the same pages,
        // whereas a single large request lets the cache manager read ahead before the copy starts.
        // The remaining gap to a plain ReadFile (1.94 GB/s, four independent handles 4.73 GB/s) is
        // the mapping itself, and closing it needs per-thread handles -- see
        // docs/装载耗时-诊断与修法-20260927.md.
        // Switchable on purpose: one binary can A/B the two pinning paths back-to-back, which is the
        // only way to trust this number (the same change read +7% while another process was
        // compiling and -19% on an idle machine). NINFER_FLASH_NEXT_PIN_PREFETCH=0 disables it; the
        // read-ahead itself was hoisted above this branch so it also applies when the bank is mapped.

        // P1: read the bank directly with INDEPENDENT handles, bypassing the mapping. See the note
        // above the helpers for why this is the community's answer and why the earlier parallel-
        // memcpy attempt could not have worked.
        bool moved_directly     = false;
        const int readfile_mode = pin_readfile_mode();
        if (readfile_mode > 0) {
            const std::wstring artifact_path =
                widen_utf8_path(std::getenv("NINFER_FLASH_NEXT_ARTIFACT_PATH"));
            const std::uint64_t file_offset = artifact::artifact_object_offset(backing, handle);
            moved_directly =
                read_bank_from_file(artifact_path, file_offset, pinned, source_bytes, readfile_mode, 4);
            if (!moved_directly) {
                std::fprintf(stderr,
                             "flash_next expert_bank_readfile FALLBACK mode=%d path_set=%d off=%llu\n",
                             readfile_mode, artifact_path.empty() ? 0 : 1,
                             static_cast<unsigned long long>(file_offset));
            }
        }
        if (!moved_directly) { std::memcpy(pinned, source, source_bytes); }
        pinned_expert_bank_registry().push_back(pinned);
        const double pinned_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pinned_started)
                .count();
        std::fprintf(stderr,
                     "flash_next expert_bank_pinned bytes=%zu duration_ms=%.1f mode=%d gbps=%.2f\n",
                     source_bytes, pinned_ms, readfile_mode,
                     pinned_ms > 0.0 ? (static_cast<double>(source_bytes) / 1e9) / (pinned_ms / 1e3)
                                     : 0.0);
        source = pinned;
    }

    Nvfp4ExpertBankView view =
        make_nvfp4_expert_bank_view(source, source_bytes, experts, rows, columns);
    // The device path gets these guarantees from the arena; a file mapping only guarantees the
    // container's per-object alignment (measured 256 B, see G7), so assert it instead of trusting it.
    if (reinterpret_cast<std::uintptr_t>(view.codes) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(view.scales) % 16 != 0 ||
        reinterpret_cast<std::uintptr_t>(view.weight_scale_divisors) % 16 != 0) {
        throw std::logic_error("Flash-Next host expert bank planes are not 16-byte aligned");
    }
    view.on_host = true;
    return view;
}

// Block-5 IQ3_S: the gguf-block expert bank over the artifact's read-only mapping. Deliberately no
// pinning here -- the PIN / PIN_READFILE machinery exists to accelerate a compute path that copies
// experts to the device, and a block bank that no kernel consumes yet would pin ~47 GB for nothing.
// The 16-byte assertion is the same guarantee the nvfp4 host view makes, from the container's
// 256-byte object alignment (G7).
GgufExpertBankView host_gguf_expert_bank_view(const artifact::MaterializedArtifact& backing,
                                              artifact::ObjectHandle handle, std::int32_t experts,
                                              std::int32_t rows, std::int32_t columns,
                                              QType qtype) {
    const std::span<const std::byte> bytes = mapped_bytes_named(backing, handle);
    report_object_digest(backing, handle, bytes);
    GgufExpertBankView view =
        make_gguf_expert_bank_view(bytes.data(), bytes.size(), experts, rows, columns, qtype);
    if (reinterpret_cast<std::uintptr_t>(view.data) % 16 != 0) {
        throw std::logic_error("Flash-Next host gguf expert bank is not 16-byte aligned");
    }
    view.on_host = true;
    return view;
}

MoeWeights load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& backing) {
    MoeWeights out{
        .router             = bf16_weight(backing, plan.router, 512, 2'560),
        .shared_down        = bf16_weight(backing, plan.shared_down, 2'560, 640),
        .shared_gate        = bf16_weight(backing, plan.shared_gate, 640, 2'560),
        .shared_up          = bf16_weight(backing, plan.shared_up, 640, 2'560),
        .shared_gate_weight = bf16_weight(backing, plan.shared_gate_weight, 1, 2'560),
    };
    if (plan.experts_gguf) {
        // Block-5 IQ3_S: block rows over the file mapping (host tiering) or the device arena. The
        // nvfp4 views stay empty on purpose: the MoE leaf's exact-bank validation then refuses a
        // forward over these weights, which is the honest state until the compute wiring lands.
        out.experts_gguf = true;
        out.gguf_gate_up =
            plan.experts_on_host
                ? host_gguf_expert_bank_view(backing, plan.expert_gate_up, 512, 1'280, 2'560,
                                             plan.gguf_gate_up_format)
                : materialized_gguf_expert_bank_view(backing, plan.expert_gate_up, 512, 1'280,
                                                     2'560, plan.gguf_gate_up_format);
        out.gguf_down =
            plan.experts_on_host
                ? host_gguf_expert_bank_view(backing, plan.expert_down, 512, 2'560, 640,
                                             plan.gguf_down_format)
                : materialized_gguf_expert_bank_view(backing, plan.expert_down, 512, 2'560, 640,
                                                     plan.gguf_down_format);
        // One line, once per process, only on this path: the direct evidence that the block-5
        // binding ran, with the formats the container declared and the byte counts it will read.
        // (Default-off runs and non-gguf artifacts never reach here, so they print nothing.)
        static bool gguf_line_reported = false;
        if (!gguf_line_reported) {
            gguf_line_reported = true;
            const std::string_view gu_name = artifact::format_name(plan.gguf_gate_up_format);
            const std::string_view dn_name = artifact::format_name(plan.gguf_down_format);
            std::fprintf(stderr,
                         "flash_next gguf_experts gate_up_format=%.*s down_format=%.*s experts=%d "
                         "host=%d gate_up_bytes=%llu down_bytes=%llu\n",
                         static_cast<int>(gu_name.size()), gu_name.data(),
                         static_cast<int>(dn_name.size()), dn_name.data(), out.gguf_gate_up.experts,
                         plan.experts_on_host ? 1 : 0,
                         static_cast<unsigned long long>(out.gguf_gate_up.rows) *
                             static_cast<unsigned long long>(out.gguf_gate_up.row_bytes) *
                             static_cast<unsigned long long>(out.gguf_gate_up.experts),
                         static_cast<unsigned long long>(out.gguf_down.rows) *
                             static_cast<unsigned long long>(out.gguf_down.row_bytes) *
                             static_cast<unsigned long long>(out.gguf_down.experts));
        }
        return out;
    }
    const auto bank = [&](artifact::ObjectHandle handle, std::int32_t experts, std::int32_t rows,
                          std::int32_t columns) {
        return plan.experts_on_host
                   ? host_nvfp4_expert_bank_view(backing, handle, experts, rows, columns)
                   : materialized_nvfp4_expert_bank_view(backing, handle, experts, rows, columns);
    };
    out.expert_gate_up = bank(plan.expert_gate_up, 512, 1'280, 2'560);
    out.expert_down    = bank(plan.expert_down, 512, 2'560, 640);
    return out;
}

GdnWeights load_gdn(const GdnPlan& plan, const artifact::MaterializedArtifact& backing) {
    return {
        .a_log             = bf16_tensor(backing, plan.a_log, {48}),
        .convolution       = bf16_tensor(backing, plan.convolution, {10'240, 4}),
        .dt_bias           = bf16_tensor(backing, plan.dt_bias, {48}),
        .a_b_projection    = bf16_weight(backing, plan.a_b_projection, 96, 2'560),
        .norm              = bf16_tensor(backing, plan.norm, {128}),
        .query_key_value_z = fp8_weight(backing, plan.query_key_value_z, 16'384, 2'560),
        .output            = fp8_weight(backing, plan.output, 2'560, 6'144),
    };
}

AttentionWeights load_attention(const AttentionPlan& plan,
                                const artifact::MaterializedArtifact& backing) {
    return {
        .indexer_query_key    = bf16_weight(backing, plan.indexer_query_key, 640, 2'560),
        .indexer_key_norm     = bf16_tensor(backing, plan.indexer_key_norm, {128}),
        .indexer_query_norm   = bf16_tensor(backing, plan.indexer_query_norm, {128}),
        .key_norm             = bf16_tensor(backing, plan.key_norm, {256}),
        .query_norm           = bf16_tensor(backing, plan.query_norm, {256}),
        .query_gate_key_value = fp8_weight(backing, plan.query_gate_key_value, 13'312, 2'560),
        .output               = fp8_weight(backing, plan.output, 2'560, 6'144),
    };
}

namespace {

constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;

// FNV-1a 64, continued across calls so a whole tensor can be hashed one shard at a time in order.
[[nodiscard]] std::uint64_t fnv1a64(std::uint64_t hash, std::span<const std::byte> bytes) {
    for (const std::byte byte : bytes) {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

PleWeights load_ple(const PlePlan& plan, const artifact::MaterializedArtifact& backing) {
    PleWeights out{
        .convolution      = bf16_tensor(backing, plan.convolution, {10'240, 4}),
        .key_projection   = bf16_weight(backing, plan.key_projection, 10'240, 2'560),
        .conv_norm        = bf16_tensor(backing, plan.conv_norm, {10'240}),
        .key_norm         = bf16_tensor(backing, plan.key_norm, {10'240}),
        .query_norm       = bf16_tensor(backing, plan.query_norm, {10'240}),
        .value_projection = bf16_weight(backing, plan.value_projection, 2'560, 2'560),
    };
    // The PLE table is the one tensor this run keeps on the host, so its bytes are the ones a
    // stage-1 reading has to tie back to the artifact. Gated, because hashing 29.8 GiB of spans
    // costs seconds and is only wanted while the host path is still being proven.
    const bool hash_ple = std::getenv("NINFER_FLASH_NEXT_PLE_HASH") != nullptr;
    std::uint64_t encoded_hash = fnv_offset_basis;
    std::uint64_t encoded_bytes = 0;
    for (std::size_t shard = 0; shard < out.table.shards.size(); ++shard) {
        // The span comes from the container's read-only mapping and is hashed per object, so the
        // comparator can tie each of the 128 shards back to the file on its own (the aggregate below
        // is kept as the pre-existing control it always was).
        const auto encoded = mapped_bytes_named(backing, plan.shards[shard]);
        report_object_digest(backing, plan.shards[shard], encoded);
        if (hash_ple) {
            encoded_hash = fnv1a64(encoded_hash, encoded);
            encoded_bytes += encoded.size();
        }
        out.table.shards[shard] = make_ple_shard_view(encoded);
        // Block-7 alignment ④: record the shard payload's absolute offset in the entry file for the
        // optional direct-read path (NINFER_FLASH_NEXT_PLE_IO_DIRECT). DEFENSIVE: a query failure
        // leaves 0 ("unknown"), which the direct path refuses loudly -- the load itself and every
        // mapped-path caller are untouched by the outcome.
        try {
            out.table.shards[shard].file_offset = backing.mapped_file_offset(plan.shards[shard]);
        } catch (const artifact::ArtifactError&) {
            out.table.shards[shard].file_offset = 0;
        }
    }
    if (hash_ple) {
        std::fprintf(stdout, "ple_hash shards=%llu bytes=%llu fnv1a64=%016llx\n",
                     static_cast<unsigned long long>(out.table.shards.size()),
                     static_cast<unsigned long long>(encoded_bytes),
                     static_cast<unsigned long long>(encoded_hash));
        std::fflush(stdout);
    }
    return out;
}

VisionModelView load_vision(const VisionPlan& plan, const artifact::MaterializedArtifact& backing) {
    VisionModelView out;
    out.patch_embedding      = bf16_weight(backing, plan.patch_embedding, 1'152, 1'536);
    out.patch_embedding_bias = bf16_tensor(backing, plan.patch_embedding_bias, {1'152});
    out.position_embedding   = bf16_tensor(backing, plan.position_embedding, {2'304, 1'152});

    for (std::size_t layer = 0; layer < plan.layers.size(); ++layer) {
        const auto& src   = plan.layers[layer];
        out.layers[layer] = {
            .qkv          = bf16_weight(backing, src.qkv, 3'456, 1'152),
            .qkv_bias     = bf16_tensor(backing, src.qkv_bias, {3'456}),
            .output       = bf16_weight(backing, src.output, 1'152, 1'152),
            .output_bias  = bf16_tensor(backing, src.output_bias, {1'152}),
            .fc1          = bf16_weight(backing, src.fc1, 4'304, 1'152),
            .fc1_bias     = bf16_tensor(backing, src.fc1_bias, {4'304}),
            .fc2          = bf16_weight(backing, src.fc2, 1'152, 4'304),
            .fc2_bias     = bf16_tensor(backing, src.fc2_bias, {1'152}),
            .norm1_weight = bf16_tensor(backing, src.norm1_weight, {1'152}),
            .norm1_bias   = bf16_tensor(backing, src.norm1_bias, {1'152}),
            .norm2_weight = bf16_tensor(backing, src.norm2_weight, {1'152}),
            .norm2_bias   = bf16_tensor(backing, src.norm2_bias, {1'152}),
        };
    }

    out.merger_fc1         = bf16_weight(backing, plan.merger_fc1, 4'608, 4'608);
    out.merger_fc1_bias    = bf16_tensor(backing, plan.merger_fc1_bias, {4'608});
    out.merger_fc2         = bf16_weight(backing, plan.merger_fc2, 2'560, 4'608);
    out.merger_fc2_bias    = bf16_tensor(backing, plan.merger_fc2_bias, {2'560});
    out.merger_norm_weight = bf16_tensor(backing, plan.merger_norm_weight, {1'152});
    out.merger_norm_bias   = bf16_tensor(backing, plan.merger_norm_bias, {1'152});
    return out;
}

AttentionWeights load_mtp_attention(const AttentionPlan& plan,
                                    const artifact::MaterializedArtifact& backing) {
    return {
        .indexer_query_key    = bf16_weight(backing, plan.indexer_query_key, 640, 2'560),
        .indexer_key_norm     = bf16_tensor(backing, plan.indexer_key_norm, {128}),
        .indexer_query_norm   = bf16_tensor(backing, plan.indexer_query_norm, {128}),
        .key_norm             = bf16_tensor(backing, plan.key_norm, {256}),
        .query_norm           = bf16_tensor(backing, plan.query_norm, {256}),
        .query_gate_key_value = bf16_weight(backing, plan.query_gate_key_value, 13'312, 2'560),
        .output               = bf16_weight(backing, plan.output, 2'560, 6'144),
    };
}

MoeWeights load_mtp_moe_legacy(const MoePlan& plan, const artifact::MaterializedArtifact& backing,
                               void* expert_gate_up_nvfp4, void* expert_down_nvfp4) {
    const std::size_t gate_bytes = flash_next_nvfp4_expert_bank_payload_bytes(512, 1'280, 2'560);
    const std::size_t down_bytes = flash_next_nvfp4_expert_bank_payload_bytes(512, 2'560, 640);
    return {
        .router             = bf16_weight(backing, plan.router, 512, 2'560),
        .shared_down        = bf16_weight(backing, plan.shared_down, 2'560, 640),
        .shared_gate        = bf16_weight(backing, plan.shared_gate, 640, 2'560),
        .shared_up          = bf16_weight(backing, plan.shared_up, 640, 2'560),
        .shared_gate_weight = bf16_weight(backing, plan.shared_gate_weight, 1, 2'560),
        .expert_gate_up =
            make_nvfp4_expert_bank_view(expert_gate_up_nvfp4, gate_bytes, 512, 1'280, 2'560),
        .expert_down =
            make_nvfp4_expert_bank_view(expert_down_nvfp4, down_bytes, 512, 2'560, 640),
    };
}

MtpModelView load_mtp(const MtpPlan& plan, const artifact::MaterializedArtifact& backing,
                      void* expert_gate_up_nvfp4 = nullptr, void* expert_down_nvfp4 = nullptr) {
    return {
        .embedding_projection = bf16_weight(backing, plan.embedding_projection, 2'560, 2'560),
        .hidden_projection    = bf16_weight(backing, plan.hidden_projection, 2'560, 2'560),
        .mixer                = load_mixer(plan.mixer, backing),
        .attention_hyper      = load_hyper(plan.attention_hyper, backing),
        .attention            = load_mtp_attention(plan.attention, backing),
        .mlp_hyper            = load_hyper(plan.mlp_hyper, backing),
        .moe                  = plan.moe.experts_nvfp4
                                    ? load_moe(plan.moe, backing)
                                    : load_mtp_moe_legacy(plan.moe, backing, expert_gate_up_nvfp4,
                                                          expert_down_nvfp4),
        .embedding_norm       = bf16_tensor(backing, plan.embedding_norm, {2'560}),
        .hidden_norm          = bf16_tensor(backing, plan.hidden_norm, {10'240}),
    };
}

// One unconditional line for the ledger, carrying the three numbers that describe the public token
// domain. Two of them exist at run time only inside the frontend; the third is the engine's own
// declaration, which lives as compile-time literals inside the ported program (`impl/program.cpp:61`
// workspace sizing, `:135` kSemanticTokenDomain feeding ops::sample, `:1413` the prompt-domain guard,
// and `impl/runtime_plan.cpp:194`; the full site list is docs/34 §7.7-1). Nothing in the program reads
// FrontendResources, so the two sides were never compared at run time -- this line is that
// comparison, and it is the only reading in this round that can go red on a real defect.
//
// The literal below is an explicit DISPLAY COPY of the program's value. Moving it into a shared
// header now would rebuild every consumer of `impl/model_view.h` (21 headers include it, several of
// them compiled by nvcc), so unifying the ~10 sites stays a follow-up; the precedent for doing it is
// card KP-UNI-0204 (one geom namespace as the single home, fully qualified references because an
// unqualified name can silently bind to a member, and error text that does not spell the number).
//
// Unconditional on purpose, like the plan line (`impl/load/bindings.cpp:376-383`): every phase name on
// this path is Debug-visibility (`product/logging/startup_log.cpp:56-68`), so a gated line would be
// invisible in the log the verdict script reads.
void report_frontend_domain(const qwen3_5::FrontendResources& frontend) {
    constexpr unsigned kProgramSemanticTokenDomain = 248'077; // display copy: impl/program.cpp:135
    const std::size_t vocab = frontend.tokenizer ? frontend.tokenizer->vocab_size() : 0;
    std::fprintf(stderr,
                 "flash_next frontend tokenizer=%d public_token_count=%u tokenizer_vocab=%zu "
                 "program_semantic_domain=%u\n",
                 frontend.tokenizer ? 1 : 0,
                 static_cast<unsigned>(frontend.public_token_count), vocab,
                 kProgramSemanticTokenDomain);
    std::fflush(stderr);
}

} // namespace

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                                 bool quantize_output_head_fp8,
                                 bool quantize_token_embedding_fp8)
    : backing(std::move(materialized)) {
    frontend = qwen3_5::take_frontend_resources(backing, plan.frontend);

    // `parse_resources` is the only thing in this tree that constructs the Tokenizer and writes
    // `public_token_count` (models/qwen3_5/frontend/resources.cpp:8-17). The canonical path calls it
    // from `bind_resources` (models/qwen3_5/load/resources.cpp:20), which this port has no counterpart
    // for -- it carries no config.json to parse. Without this call `frontend.tokenizer` stays null and
    // Frontend::Impl throws "Frontend requires the parsed model tokenizer and public token domain"
    // (models/qwen3_5/frontend/frontend.cpp:705).
    //
    // Of the Config argument `parse_resources` reads exactly three members: `text.vocab_size` as an
    // upper bound (`:12`), `vision` (`:23`) and `draft` (`:42`). Because the latter two stay empty
    // here, the vision-processor-geometry check (`:23-40`) is UNREACHABLE on this path -- this artifact
    // carries no config.json at all (1560 tensors + 6 resources) -- and the DFlash2 check (`:42-44`)
    // does not run. Recorded as a deliberate, named reduction (docs/33 §5, docs/34 §7.1).
    //
    // The bound is the embedding/output-head ROW CEILING, not the public token domain: `:12` only
    // rejects count > kOutputHeadRows, so a tokenizer exposing a domain in (248'077, 248'320] would
    // pass startup and only fail per request at impl/program.cpp:1413. The line printed below is what
    // catches that.
    {
        qwen3_5::Config frontend_config;
        frontend_config.text.vocab_size = static_cast<std::uint32_t>(kOutputHeadRows);
        qwen3_5::parse_resources(frontend, frontend_config);
    }
    report_frontend_domain(frontend);

    text.weights_arena = &backing.device_arena();

    // plan.features carries the *binding* decision made in bind_artifact: when its FP8 flag is set
    // the BF16 tensor was retained in the file mapping and must be quantized from the host span.
    // The constructor arguments stay authoritative for StandaloneLoadedModel, which always binds
    // on the device and therefore quantizes device-to-device (no VRAM saving, same numerics).
    const bool embedding_mapped = plan.features.quantize_token_embedding_fp8;
    const bool head_mapped      = plan.features.quantize_output_head_fp8;

    if (embedding_mapped) {
        token_embedding_fp8 = DeviceBuffer(flash_next_fp8_head_payload_bytes(248'320, 2'560));
        quantize_bf16_rows_to_fp8_e4m3_row_f32s(
            mapped_bf16_rows(backing, plan.token_embedding, 248'320, 2'560).data(),
            token_embedding_fp8, text.token_embedding, 248'320, 2'560, cudaStream_t{});
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        text.token_embedding = bf16_weight(backing, plan.token_embedding, 248'320, 2'560);
        if (quantize_token_embedding_fp8) {
            token_embedding_fp8 = DeviceBuffer(flash_next_fp8_head_payload_bytes(248'320, 2'560));
            Weight fp8_embedding{};
            quantize_bf16_head_to_fp8_e4m3_row_f32s(text.token_embedding, token_embedding_fp8,
                                                    fp8_embedding, 248'320, 2'560, cudaStream_t{});
            CUDA_CHECK(cudaDeviceSynchronize());
            text.token_embedding = fp8_embedding;
        }
    }
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < plan.text_layers.size(); ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        TextLayerWeights& target    = text.layers[layer];
        target.attention_hyper      = load_hyper(source.attention_hyper, backing);
        target.moe                  = load_moe(source.moe, backing);
        target.mlp_hyper            = load_hyper(source.mlp_hyper, backing);
        if (source.is_full_attention) {
            text.full_attention.at(full_index++) = load_attention(source.attention, backing);
        } else {
            text.gdn.at(gdn_index++) = load_gdn(source.gdn, backing);
        }
    }
    if (full_index != text.full_attention.size() || gdn_index != text.gdn.size()) {
        throw std::logic_error("Flash-Next Text topology materialization is incomplete");
    }
    text.ple = load_ple(plan.ple, backing);

    // Mapped head: the BF16 rows stay in the file mapping, so both the FP8 head and the optimized
    // proposal head's shortlist rows are produced from `mapped_head` instead of a device view.
    std::span<const std::byte> mapped_head;
    Weight raw_bf16_head{};
    if (head_mapped) {
        mapped_head     = mapped_bf16_rows(backing, plan.output_head, 248'320, 2'560);
        output_head_fp8 = DeviceBuffer(flash_next_fp8_output_head_payload_bytes());
        quantize_bf16_rows_to_fp8_e4m3_row_f32s(mapped_head.data(), output_head_fp8,
                                                text.output_head, 248'320, 2'560, cudaStream_t{});
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        raw_bf16_head    = bf16_weight(backing, plan.output_head, 248'320, 2'560);
        text.output_head = raw_bf16_head;
        if (quantize_output_head_fp8) {
            output_head_fp8 = DeviceBuffer(flash_next_fp8_output_head_payload_bytes());
            Weight fp8_head{};
            quantize_bf16_output_head_to_fp8_e4m3_row_f32s(raw_bf16_head, output_head_fp8, fp8_head,
                                                           cudaStream_t{});
            CUDA_CHECK(cudaDeviceSynchronize());
            text.output_head = fp8_head;
        }
    }

    if (plan.features.proposal_head == ProposalHead::Optimized) {
        const std::uint32_t K =
            plan.features.draft_head_rows > 0 ? plan.features.draft_head_rows : 32'768;
        const auto shortlist_ids = load_flash_next_draft_shortlist(K);

        proposal_token_ids = DeviceBuffer(K * sizeof(std::int32_t));
        proposal_token_ids.copy_from_host(shortlist_ids.data(), K * sizeof(std::int32_t));

        proposal_head_payload =
            DeviceBuffer(static_cast<std::size_t>(K) * 2'560 * sizeof(std::uint16_t));
        Weight bf16_proposal_head{};
        if (head_mapped) {
            gather_head_rows_bf16_from_host(mapped_head, shortlist_ids.data(),
                                            static_cast<std::int32_t>(K), 248'320, 2'560,
                                            proposal_head_payload, bf16_proposal_head,
                                            cudaStream_t{});
        } else {
            gather_head_rows_bf16(raw_bf16_head,
                                  static_cast<const std::int32_t*>(proposal_token_ids.p),
                                  static_cast<std::int32_t>(K), 2'560, proposal_head_payload,
                                  bf16_proposal_head, cudaStream_t{});
        }

        Weight final_proposal_head = bf16_proposal_head;
        if (quantize_output_head_fp8 || head_mapped) {
            proposal_head_fp8 = DeviceBuffer(flash_next_fp8_head_payload_bytes(static_cast<std::int32_t>(K), 2'560));
            Weight fp8_proposal_head{};
            quantize_bf16_head_to_fp8_e4m3_row_f32s(bf16_proposal_head, proposal_head_fp8,
                                                    fp8_proposal_head, static_cast<std::int32_t>(K),
                                                    2'560, cudaStream_t{});
            final_proposal_head = fp8_proposal_head;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        OptimizedProposalWeights prop{};
        prop.head      = final_proposal_head;
        prop.token_ids = Tensor(proposal_token_ids.p, DType::I32, {static_cast<std::int32_t>(K)});
        text.proposal  = prop;
    }

    text.final_mixer = load_mixer(plan.final_mixer, backing);

    if (plan.features.mtp) {
        if (plan.mtp.moe.experts_nvfp4) {
            text.mtp = load_mtp(plan.mtp, backing);
        } else {
            mtp_expert_gate_up_nvfp4 =
                DeviceBuffer(flash_next_nvfp4_expert_bank_payload_bytes(512, 1'280, 2'560));
            mtp_expert_down_nvfp4 =
                DeviceBuffer(flash_next_nvfp4_expert_bank_payload_bytes(512, 2'560, 640));

            const auto gate_up_bytes = mapped_bytes_named(backing, plan.mtp.moe.expert_gate_up);
            const auto down_bytes    = mapped_bytes_named(backing, plan.mtp.moe.expert_down);
            report_object_digest(backing, plan.mtp.moe.expert_gate_up, gate_up_bytes);
            report_object_digest(backing, plan.mtp.moe.expert_down, down_bytes);

            quantize_bf16_expert_bank_to_nvfp4(gate_up_bytes.data(), mtp_expert_gate_up_nvfp4.p, 512,
                                               1'280, 2'560, cudaStream_t{});
            quantize_bf16_expert_bank_to_nvfp4(down_bytes.data(), mtp_expert_down_nvfp4.p, 512, 2'560,
                                               640, cudaStream_t{});
            CUDA_CHECK(cudaDeviceSynchronize());

            text.mtp = load_mtp(plan.mtp, backing, mtp_expert_gate_up_nvfp4.p,
                                mtp_expert_down_nvfp4.p);
        }
    }
    if (plan.features.vision) { vision = load_vision(plan.vision, backing); }

    // PLE is a random-access host weight table. Leaving its mapped pages cold makes
    // the first code prompt pay thousands of synchronous disk faults per token batch.
    // Warm only these persistent host weights after device materialization, before ready.
    //
    // With a pinned expert bank the arithmetic changes: 63.3 GiB pinned experts + 32 GiB of warmed
    // PLE does not fit in 95.8 GiB of RAM. PLE is read ~2.5 KB per token (16 rows x 160 B) so its
    // working set is tiny and stays hot after the first touches -- take the community's lazy-mode
    // choice and give the RAM to the pinned bank instead.
    const auto warm_started = std::chrono::steady_clock::now();
    std::size_t host_bytes = 0;
    if (flash_next_experts_pinned_enabled()) {
        std::fprintf(stderr,
                     "flash_next host_ple_warm bytes=0 duration_ms=0.000 skipped=1 "
                     "reason=pinned_expert_bank\n");
    } else if (!flash_next_ple_warm_enabled()) {
        // Mapped PLE rows are demand-paged: warming them is 29.80 GiB of pages that the routing
        // experts need instead. NINFER_FLASH_NEXT_PLE_WARM=1 restores the old behaviour.
        std::fprintf(stderr,
                     "flash_next host_ple_warm bytes=0 duration_ms=0.000 skipped=1 "
                     "reason=mapped_rows_are_demand_paged\n");
    } else {
        for (const auto& shard : text.ple.table.shards) {
            warm_readonly_host_memory(shard.codes);
            warm_readonly_host_memory(shard.scales);
            host_bytes += shard.codes.size() + shard.scales.size();
        }
        std::fprintf(stderr, "flash_next host_ple_warm bytes=%zu duration_ms=%.3f\n", host_bytes,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - warm_started).count());
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
