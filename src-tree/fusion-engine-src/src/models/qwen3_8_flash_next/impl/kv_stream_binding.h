#pragma once
//
// kv_stream_binding.h — model-side binding for the KV streaming residency map (third knife).
//
// Third-knife shape (see the plan v2.5/v2.6):
//   * M-2: TWO independent states, because prefill must double-write the host copy while the block
//     table still has its legacy physical-page meaning:
//       host_mirror -- writers copy K/V into the host pool (true from the first prefill chunk);
//       residency   -- the page table is owned by the residency map (true from the first decode
//                      round). Arming does NOT clear host_written, or every historical page would
//                      look missing and J8 would throw on the first decode.
//   * M-3: the binding allocates its OWN pinned host pool and takes the device alias with
//     cudaHostGetDevicePointer. Probe evidence (E:\infer-build\kv-stream\probe_host_ptr.cu,
//     2026-10-18): cudaHostAlloc(Portable) gives host_ptr == device_ptr under unified addressing
//     (unified_addressing=1, canMapHostMemory=1, attrs.type=1, kernel write/read both clean), so
//     cudaHostAllocMapped is NOT used. THE CALL IS STILL MADE AND MUST STILL THROW ON FAILURE: the
//     "same address" outcome is a property of the machine/driver, not of this code. Re-run that
//     probe if the card or driver changes.
//   * M-3 landmine: NINFER_HOST_PAGEABLE makes PinnedHostBuffer hand out std::malloc memory
//     (arena.cu:636-648), which is NOT device addressable; E:\infer-build\fusion-mine2\run-bin.ps1:29
//     sets it to '1'. bind() refuses that environment outright -- a safety valve, not an option.
//   * M-4: owner is checked BEFORE the already-bound early return, and shutdown_if_owner() is called
//     from the executor destructor, so a second executor can never reuse a dangling map (UAF).
//
// Header-only on purpose: the model runtime manifest lists .cpp/.cu explicitly, and adding no
// translation unit keeps that manifest untouched.

#include "core/arena.h"
#include "ninfer/ops/kv_stream.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr int kKvStreamMaxLayers      = 12;
inline constexpr int kKvStreamPlanesPerLayer = 2;

struct FlashNextKvBinding {
    // Two independent states (M-2). `enabled` is gone on purpose: one flag could not express
    // "double-write yes, table no".
    bool host_mirror = false;
    bool residency   = false;

    const void* owner = nullptr;
    void* page_table  = nullptr;  // borrowed: qsa_attention_caches[0].block_tables.data
    ops::KvStreamMap map{};
    ops::KvStreamRuns runs{};

    // M-3: self-owned pinned host pool + its device alias (see the header comment).
    void* host_base_   = nullptr;  // CPU address (cudaHostAlloc result)
    void* device_base_ = nullptr;  // device alias (what kernels must use)
    std::size_t pool_bytes_ = 0;

    // Out-of-plan device memory for the map arrays (plan §A.4: not in total_device_bytes).
    std::unique_ptr<DeviceBuffer> map_storage;

    std::int64_t host_page_stride  = 0;
    std::int64_t host_layer_stride = 0;
    std::int32_t page_plane_bytes  = 0;
    std::int32_t page_tokens       = 64;
    std::int32_t block_tokens      = 4;
    std::int32_t layers            = 0;
    const void* layer_key_pages[kKvStreamMaxLayers] = {};

    static FlashNextKvBinding& instance() {
        static FlashNextKvBinding one;
        return one;
    }

    ~FlashNextKvBinding() { release_pool(); }

    [[nodiscard]] int layer_for_key_pages(const void* key_pages_data) const {
        for (int layer = 0; layer < layers; ++layer) {
            if (layer_key_pages[layer] == key_pages_data) { return layer; }
        }
        return -1;
    }

    // DEVICE-side pointers: the double write happens inside a kernel, so these must be aliases.
    [[nodiscard]] std::uint8_t* host_k(int layer) const {
        return static_cast<std::uint8_t*>(device_base_) + host_layer_stride * layer;
    }
    [[nodiscard]] std::uint8_t* host_v(int layer) const {
        return host_k(layer) + page_plane_bytes;
    }

    // M-2 stage 1: called from the prefill path, before the first KV token is written.
    void enable_host_mirror(const void* owner_, void* page_table_, std::int32_t rows,
                            std::int64_t logical_pages, std::int32_t n_slots,
                            std::int32_t page_tokens_, std::int32_t block_tokens_,
                            std::int32_t page_plane_bytes_, int layer_count,
                            const void* const* key_pages, const void* const* value_pages) {
        if (host_mirror) {
            if (owner != owner_) { throw std::logic_error(kOwnerMessage); }
            return;
        }
        bind_impl(owner_, page_table_, rows, logical_pages, n_slots, page_tokens_, block_tokens_,
                  page_plane_bytes_, layer_count, key_pages, value_pages);
        host_mirror = true;
    }

    // M-2 stage 2: called before the first decode round. Keeps host_written intact.
    void arm_residency(cudaStream_t stream) {
        if (!host_mirror) {
            throw std::logic_error("kv_stream: arm_residency before enable_host_mirror");
        }
        if (residency) { return; }
        ops::kv_stream_reset(map, /*clear_host_written=*/false, stream);
        residency = true;
    }

    void shutdown_if_owner(const void* owner_) noexcept {
        if (owner != nullptr && owner != owner_) { return; }
        host_mirror = false;
        residency   = false;
        owner       = nullptr;
        map         = ops::KvStreamMap{};
        runs        = ops::KvStreamRuns{};
        map_storage.reset();
        release_pool();
        layers = 0;
        std::memset(layer_key_pages, 0, sizeof(layer_key_pages));
    }

private:
    static constexpr const char* kOwnerMessage =
        "kv_stream: this process already streams for another executor (third-knife M-4 guard)";

    static constexpr const char* kPageableMessage =
        "kv_stream: NINFER_HOST_PAGEABLE is set -- the streaming host pool must be pinned and "
        "device-visible, but pageable host memory is handed out by std::malloc (arena.cu:636-648) "
        "and a device dereference of it is an illegal address. Unset the variable before enabling "
        "KV streaming (E:\\infer-build\\fusion-mine2\\run-bin.ps1:29 sets it; do not copy that line).";

    void release_pool() noexcept {
        if (host_base_ != nullptr) {
            (void) cudaFreeHost(host_base_);
            host_base_ = nullptr;
        }
        device_base_ = nullptr;
        pool_bytes_  = 0;
    }

    void bind_impl(const void* owner_, void* page_table_, std::int32_t rows,
                   std::int64_t logical_pages, std::int32_t n_slots, std::int32_t page_tokens_,
                   std::int32_t block_tokens_, std::int32_t page_plane_bytes_, int layer_count,
                   const void* const* key_pages, const void* const* value_pages) {
        // M-3 landmine first: refuse before any allocation, so a mis-set environment fails loudly
        // instead of producing memory a kernel may not touch.
        const char* pageable = std::getenv("NINFER_HOST_PAGEABLE");
        if (pageable != nullptr && pageable[0] != '\0' && pageable[0] != '0') {
            throw std::runtime_error(kPageableMessage);
        }
        // M-4: owner before any "already bound" short circuit.
        if (owner != nullptr && owner != owner_) { throw std::logic_error(kOwnerMessage); }
        if (rows <= 0 || rows > 8) { throw std::invalid_argument("kv_stream: rows must be 1..8"); }
        if (logical_pages <= 0 || n_slots <= 0) {
            throw std::invalid_argument("kv_stream: geometry must be positive");
        }
        if (layer_count <= 0 || layer_count > kKvStreamMaxLayers) {
            throw std::invalid_argument("kv_stream: layer count out of range");
        }
        if (page_plane_bytes_ <= 0 || page_plane_bytes_ % 16 != 0) {
            throw std::invalid_argument("kv_stream: a plane must be a multiple of 16 bytes");
        }
        if (page_table_ == nullptr) { throw std::invalid_argument("kv_stream: null page table"); }

        page_plane_bytes  = page_plane_bytes_;
        page_tokens       = page_tokens_;
        block_tokens      = block_tokens_;
        host_page_stride  = static_cast<std::int64_t>(page_plane_bytes_) * 2;
        host_layer_stride = host_page_stride * logical_pages;
        layers            = layer_count;
        page_table        = page_table_;

        // M-3: own the pinned pool; take the device alias and refuse to continue on failure.
        pool_bytes_ = static_cast<std::size_t>(host_layer_stride * layer_count);
        if (cudaHostAlloc(&host_base_, pool_bytes_, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            host_base_ = nullptr;
            throw std::runtime_error("kv_stream: cudaHostAlloc(Portable) failed for the host pool");
        }
        const cudaError_t alias = cudaHostGetDevicePointer(&device_base_, host_base_, 0);
        if (alias != cudaSuccess || device_base_ == nullptr) {
            (void) cudaGetLastError();
            release_pool();
            throw std::runtime_error(
                "kv_stream: pinned host pool is not device-addressable (cudaHostGetDevicePointer "
                "failed); re-run E:\\infer-build\\kv-stream\\probe_host_ptr.cu on this machine");
        }

        map_storage = std::make_unique<DeviceBuffer>(
            ops::kv_stream_map_bytes(rows, logical_pages, n_slots));
        ops::kv_stream_attach(map, map_storage->p, rows, logical_pages, n_slots);
        map.page_table = static_cast<std::int32_t*>(page_table_);

        runs = ops::KvStreamRuns{};
        for (int layer = 0; layer < layer_count; ++layer) {
            const int k_run = kKvStreamPlanesPerLayer * layer;
            const int v_run = k_run + 1;
            layer_key_pages[layer]  = key_pages[layer];
            runs.host[k_run]        = host_k(layer);   // device alias, plane offset below
            runs.device[k_run]      = const_cast<void*>(key_pages[layer]);
            runs.host_stride[k_run] = host_page_stride;
            runs.host_offset[k_run] = 0;
            runs.len[k_run]         = page_plane_bytes;
            runs.host[v_run]        = host_k(layer);
            runs.device[v_run]      = const_cast<void*>(value_pages[layer]);
            runs.host_stride[v_run] = host_page_stride;
            runs.host_offset[v_run] = page_plane_bytes;
            runs.len[v_run]         = page_plane_bytes;
        }
        runs.count = kKvStreamPlanesPerLayer * layer_count;
        owner      = owner_;
    }
};

inline FlashNextKvBinding& kv_stream_binding() { return FlashNextKvBinding::instance(); }

}  // namespace ninfer::models::qwen3_8_flash_next::detail
