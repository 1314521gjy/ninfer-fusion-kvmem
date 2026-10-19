#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include "core/weight_view.h"
#include "ninfer/types.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
    // Set when the stored row-split Q8 object is requantized at load: `bytes` is then the target
    // encoding's size and the device parent carries the target geometry (see artifact/transcode.h).
    std::optional<QType> transcode;
    // Which pipeline rank's device arena holds the object; `offset` is relative to that arena.
    // Rank 0 is the primary device, and is the only rank a single-device load ever uses.
    std::size_t rank = 0;
};

// Offset inside the one page-locked Host block (Residency::Pinned).
struct PinnedPlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

// The object stays in the container's read-only file mapping (Residency::Mapped): nothing is read at
// plan time, and `file_offset` records where its bytes begin in the entry file so a reader can check
// that the span it received really lies inside that mapping.
struct MappedPlacement {
    ObjectHandle object;
    std::string id;
    std::uint64_t file_offset = 0;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct MaterializationPlan {
    const Reader* source            = nullptr;
    std::size_t object_count        = 0;
    std::uint64_t prior_read_bytes  = 0;
    std::uint64_t owned_value_bytes = 0;
    // One device arena per pipeline rank. Entry 0 is the primary device and always exists, so a
    // single-device plan has exactly one entry and rank 0 owns every device object.
    std::vector<std::uint64_t> device_capacity_by_rank{0};
    // Evict-ranked objects occupy [evictable_tail_offset, device_capacity(0)) on rank 0; the offset
    // is aligned as Binder::finish was asked. Both are zero without ranked objects. The evictable
    // tail is a rank-0 feature: an offloaded rank holds nothing a Vision window may borrow.
    std::uint64_t evictable_tail_offset = 0;
    std::uint64_t evictable_tail_bytes  = 0;
    std::uint64_t pinned_capacity_bytes = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<PinnedPlacement> pinned_objects;
    std::vector<HostPlacement> host_objects;
    // Objects that stay in the container's read-only file mapping. Distinct from `pinned_objects`,
    // which is a page-locked copy; until this change the accessor below returned the pinned list.
    std::vector<MappedPlacement> mapped_objects;
    // Upstream's plan carries one `device_capacity_bytes` for its single device. This engine splits
    // the capacity per pipeline rank, so the aggregate is derived. It is a function rather than a
    // data member on purpose: a stored copy beside `device_capacity_by_rank` would be a second
    // source of truth for the same total.
    [[nodiscard]] std::uint64_t total_device_capacity_bytes() const noexcept {
        std::uint64_t total = 0;
        for (const std::uint64_t bytes : device_capacity_by_rank) { total += bytes; }
        return total;
    }
    // Mapped objects have their own list, the way the reference keeps a `mapped_tensor_objects` list
    // beside its device and host lists (`materializer.h:51-58` of the reference). The name is kept
    // because the ported loader reads it (`impl/load/loader.cpp`, `planned_mapped_tensors_count`);
    // before this change it returned the pinned list, which read as "mapped" while the storage behind
    // it was a page-locked copy.
    [[nodiscard]] std::vector<ObjectHandle> mapped_tensor_objects() const {
        std::vector<ObjectHandle> out;
        out.reserve(mapped_objects.size());
        for (const MappedPlacement& placement : mapped_objects) { out.push_back(placement.object); }
        return out;
    }

    [[nodiscard]] std::size_t device_rank_count() const noexcept {
        return device_capacity_by_rank.empty() ? 1 : device_capacity_by_rank.size();
    }

    [[nodiscard]] std::uint64_t device_capacity(std::size_t rank = 0) const noexcept {
        return rank < device_capacity_by_rank.size() ? device_capacity_by_rank[rank] : 0;
    }
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0; // primary device (rank 0)
    // Expert-offload split only: the weight arenas held by the ranks past the primary device.
    std::uint64_t offloaded_device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t mapped_tensor_bytes   = 0; // bytes left in the read-only file mapping
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t pinned_bytes          = 0; // page-locked Host block (Residency::Pinned)
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0;
    std::size_t pinned_object_count     = 0;
    std::size_t mapped_object_count     = 0;
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    // Device base address of a materialized object. The backing store itself is immutable once
    // materialized, so this is the same const_cast the dense-Op ABI already performs in
    // core/weight_view.h (`as_dense`); kernels that merely read a device weight take it as `void*`.
    [[nodiscard]] void* device_data(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;
    [[nodiscard]] const WeightParent& pinned_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> pinned_block() const noexcept;
    // Bytes of a tensor registered through the "mapped tensor" path. Upstream returns a span into
    // the container's file mapping; this engine has no mapping lease, so the payload lives either in
    // the page-locked block (Residency::Pinned, where `retain_mapped_tensor` lands) or in the
    // retained Host copy (Residency::Host, used for resources). The storage that is actually present
    // decides, and anything else throws -- a tensor that was never retained must not silently read
    // as empty.
    [[nodiscard]] std::span<const std::byte> mapped_tensor_bytes(ObjectHandle handle) const;
    // Identity of the storage behind a handle. `object_id` is the artifact's OWN object id (not a
    // synthesized label), which is what lets a failed read name the object and a comparator look it
    // up in the digest oracle; `mapped_file_offset` is where its bytes start in the entry file.
    [[nodiscard]] std::string_view object_id(ObjectHandle handle) const;
    [[nodiscard]] std::uint64_t mapped_file_offset(ObjectHandle handle) const;
    // Holds the container's read-only mapping alive; spans from `mapped_tensor_bytes` stay valid
    // exactly as long as this lease is held (which is why the lease lives here, not on the Reader).
    [[nodiscard]] const std::shared_ptr<const void>& mapping_lease() const noexcept {
        return mapping_lease_;
    }
    // The primary device arena (rank 0). Upstream exposes a single arena; this engine keeps one per
    // pipeline rank and rank 0 always exists, so the primary device is what an offload-free model
    // touches.
    [[nodiscard]] DeviceArena& device_arena();
    // Present when the device backing was supplied by an eviction pool.
    [[nodiscard]] EvictableWeightPool* weight_pool() const noexcept { return pool_.get(); }

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&, DeviceContext&,
                                            const StartupObserver*,
                                            std::unique_ptr<EvictableWeightPool>);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> pinned;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
        // A span into the container's read-only mapping, plus the identity a reader needs to prove
        // it: the object's id and its byte offset in the entry file.
        std::span<const std::byte> mapped;
        std::string id;
        std::uint64_t mapped_file_offset = 0;
    };

    // The pool owns the physical memory behind a pool-backed arena; destroy the arena first.
    std::unique_ptr<EvictableWeightPool> pool_;
    // One arena per pipeline rank, allocated on that rank's device. Index 0 is the primary device;
    // a single-device load holds exactly one entry, which is the only one an offload-free model
    // ever touches.
    std::vector<std::unique_ptr<DeviceArena>> arenas_;
    std::unique_ptr<PinnedHostBuffer> pinned_;
    // Keeps the read-only entry mapping alive while any object points into it.
    std::shared_ptr<const void> mapping_lease_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

// `backing`, when given, supplies the device arena (its arena must cover the plan) and is owned by
// the result. Its window mirror is captured later by the owner, once the window is known.
[[nodiscard]] MaterializedArtifact
materialize(const Reader& reader, MaterializationPlan&& plan, DeviceContext& device,
            const StartupObserver* startup_observer      = nullptr,
            std::unique_ptr<EvictableWeightPool> backing = nullptr);

// Upstream's plan is a value the caller moves in. Some callers here hold it as a member they read
// again afterwards (the ported loader reports its counts after materializing), so an lvalue
// overload exists rather than forcing every call site to give up ownership. The implementation is
// shared: this one copies and delegates.
[[nodiscard]] MaterializedArtifact
materialize(const Reader& reader, const MaterializationPlan& plan, DeviceContext& device,
            const StartupObserver* startup_observer      = nullptr,
            std::unique_ptr<EvictableWeightPool> backing = nullptr);

} // namespace ninfer::artifact
