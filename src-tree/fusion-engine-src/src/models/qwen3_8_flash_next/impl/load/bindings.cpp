#include "models/qwen3_8_flash_next/impl/load/bindings.h"
#include "models/qwen3_8_flash_next/impl/expert_bank.h"
#include "models/qwen3_8_flash_next/impl/ple_index.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_8_flash_next::detail {

using ninfer::port::Binder;
using namespace ninfer::portalias;

namespace {

using portalias::NumericFormat;
using artifact::ObjectHandle;
using portalias::StorageLayout;

constexpr auto kBf16Layout   = StorageLayout::ContiguousLeV1;
constexpr auto kFp8Layout    = StorageLayout::RowScaleF32V1;
constexpr auto kExpertLayout = StorageLayout::ExpertBlockScaleK16M128x4V1;
constexpr auto kPleLayout    = StorageLayout::PackedU4G16V1;
// Block-5: the gguf-block layout the port gained for the IQ3_S artifact's expert banks.
constexpr auto kGgufLayout   = StorageLayout::GgufBlocksV1;

// Negative control (docs/32 §3, N-1): registering the mapped objects as Device again must push the
// device plan back to its old size and take the mapped count to zero. It is an env switch rather than
// a source toggle on purpose -- a source-level control would cost another full rebuild (~66 minutes).
bool force_device_residency() {
    static const bool forced = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_FORCE_DEVICE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return forced;
}

ObjectHandle bind_device(Binder& binder, std::string_view name, NumericFormat format,
                         StorageLayout layout, std::initializer_list<std::uint64_t> shape) {
    const ObjectHandle handle = binder.require_tensor(name, format, layout, shape);
    binder.materialize_on_device(handle);
    return handle;
}

ObjectHandle bind_optional_device(Binder& binder, std::string_view name,
                                  NumericFormat format, StorageLayout layout,
                                  std::initializer_list<std::uint64_t> shape, bool enabled) {
    // The residency is chosen at registration, because registration is the only call that records a
    // demand: `binder_port.h:204` defaults to Device and the four verbs are no-ops (`verify()` keeps
    // nothing). So a disabled feature has to be registered with a residency that records no placement
    // at all -- `Values` -- otherwise its bytes stay in the device plan while the feature is off.
    // On this artifact the disabled MTP and vision subtrees are 5,214,301,696 B and 897,862,112 B:
    // about 5.7 GiB that used to be demanded for nothing (docs/32 §2-1).
    return binder.require_tensor(name, format, layout, artifact::Shape(shape),
                                 enabled ? artifact::Residency::Device
                                         : artifact::Residency::Values);
}

// `mapped` means the object is left in the container's read-only file mapping: nothing is read at
// load time and no device copy is made, so the demand has to be stated at registration -- the single
// call that records one (`artifact/binder.cpp:52-65` branches on the residency once).
//   * `docs/19-蓝图-qwen4移植门面-20261010.md:150` sends `retain_mapped_tensor` to
//     `Residency::Pinned`, and the port had no mapping at all, so it landed on `Host` (a private
//     copy read at bind time). The reference gives that residency a real read-only mapping, which is
//     what `Residency::Mapped` now is (`artifact/reader.cpp`, `mapped_tensor_bytes`).
//   * Every caller of this function consumes the object through a span
//     (`mapped_tensor_bytes`: the 128 PLE shards, the three I64 side tables, and the token-embedding
//     / output-head BF16 payloads the FP8-quantizing loaders read). Registering them Device left the
//     demand saying "device" while every reader treated them as a mapping -- that is the
//     `object handle does not name a mapped tensor` failure measured at 15:48 on the real artifact.
ObjectHandle bind_mapped(Binder& binder, std::string_view name, NumericFormat format,
                         StorageLayout layout, std::initializer_list<std::uint64_t> shape) {
    const artifact::Residency residency = force_device_residency() ? artifact::Residency::Device
                                                                  : artifact::Residency::Mapped;
    return binder.require_tensor(name, format, layout, artifact::Shape(shape), residency);
}

template <std::size_t Size>
void require_i64_values(const Binder& binder, ObjectHandle handle,
                        const std::array<std::int64_t, Size>& expected, std::string_view label) {
    // Owning bytes on purpose: `payload()` spans a vector that dies at the end of its full
    // expression, so the older `const std::span bytes = binder.payload(handle).data;` dangled.
    const std::vector<std::byte> bytes = binder.payload_bytes(handle);
    if (bytes.size() != Size * sizeof(std::int64_t)) {
        throw artifact::ArtifactError(std::string(label) + ": invalid I64 payload size");
    }
    for (std::size_t index = 0; index < Size; ++index) {
        std::uint64_t bits = 0;
        for (std::size_t byte = 0; byte < sizeof(std::int64_t); ++byte) {
            bits |= static_cast<std::uint64_t>(
                        std::to_integer<std::uint8_t>(bytes[index * sizeof(std::int64_t) + byte]))
                    << (byte * 8U);
        }
        if (std::bit_cast<std::int64_t>(bits) != expected[index]) {
            throw artifact::ArtifactError(std::string(label) + ": value does not match target");
        }
    }
}


HyperConnectionPlan bind_hyper(Binder& binder, const std::string& prefix,
                               bool enabled = true) {
    const auto bind = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), NumericFormat::BF16,
                                    kBf16Layout, shape, enabled);
    };
    return {
        .block_inject   = bind("block_inject", {4, 10'240}),
        .norm           = bind("norm", {10'240}),
        .input_mix_down = bind("input_mix/down", {320, 10'240}),
        .input_mix_up   = bind("input_mix/up", {10'240, 320}),
    };
}

HyperMixerPlan bind_mixer(Binder& binder, const std::string& prefix,
                          bool enabled = true) {
    const auto bind = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), NumericFormat::BF16,
                                    kBf16Layout, shape, enabled);
    };
    return {
        .norm           = bind("norm", {10'240}),
        .input_mix_down = bind("input_mix/down", {320, 10'240}),
        .input_mix_up   = bind("input_mix/up", {10'240, 320}),
    };
}

artifact::ObjectHandle bind_optional_host(Binder& binder, std::string_view name,
                                          NumericFormat format, StorageLayout layout,
                                          std::initializer_list<std::uint64_t> shape, bool enabled) {
    // Every artifact object must be consumed by the target even when its feature is off, exactly as
    // bind_optional_device does. (Port note kept: sequence 12b returned early here and every serve
    // failed. The name of the check the note quotes was not found in this tree -- grep for it hits
    // comments only -- so the rule kept is "register, never skip".)
    // Which residency: an enabled object is read as host bytes, so it stays in the container's file
    // mapping (the reference's `retain_mapped_tensor` landing); a disabled one records no placement.
    const artifact::Residency residency =
        !enabled ? artifact::Residency::Values
                 : (force_device_residency() ? artifact::Residency::Device
                                             : artifact::Residency::Mapped);
    return binder.require_tensor(name, format, layout, artifact::Shape(shape), residency);
}

MoePlan bind_moe(Binder& binder, const std::string& prefix, NumericFormat expert_format,
                 bool enabled = true, bool retain_experts_on_host = false) {
    const auto bf16 = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), NumericFormat::BF16,
                                    kBf16Layout, shape, enabled);
    };
    // Block-5: the container is authoritative about what it stores. The MTP branch below already
    // probes the descriptor to tell NVFP4 from legacy BF16; this probe additionally recognizes the
    // gguf block family, whose per-object formats differ between gate_up and down and therefore
    // cannot be spelled by one caller-supplied default. A gguf bank is refused unless the dedicated
    // switch is on (default off, so a block-5 artifact cannot load by accident), and a bank that
    // mixes gguf with anything else is refused outright.
    NumericFormat gate_up_format = expert_format;
    NumericFormat down_format    = expert_format;
    const auto probed_format = [&](std::string_view suffix) {
        const port::ObjectDescriptor* descriptor = binder.find(prefix + std::string(suffix));
        const auto* tensor =
            descriptor == nullptr ? nullptr : std::get_if<port::TensorDescriptor>(descriptor);
        return tensor == nullptr ? expert_format : tensor->format;
    };
    const NumericFormat probed_gate_up = probed_format("experts/gate_up");
    if (is_gguf(probed_gate_up)) {
        if (!flash_next_gguf_experts_enabled()) {
            throw artifact::ArtifactError(
                "port: " + prefix + "experts/gate_up is stored in ggml blocks; set "
                "NINFER_FLASH_NEXT_GGUF_EXPERTS=1 to load a block-5 IQ3_S artifact");
        }
        // Block-5 ruling: an nvfp4-only service policy next to a gguf bank would be a silent
        // no-op, so it is refused here -- fail-fast, at the load face -- naming the switch. The
        // MoE leaf repeats the check at the arithmetic point; both read the same switch table.
        if (const char* conflict = flash_next_gguf_policy_conflict(); conflict != nullptr) {
            throw artifact::ArtifactError(
                "port: " + prefix + "gguf expert bank is loaded together with " + conflict +
                ", which is not wired for gguf blocks in this stage; unset it (refused loudly "
                "rather than silently ignored)");
        }
        const NumericFormat probed_down = probed_format("experts/down");
        if (!is_gguf(probed_down)) {
            throw artifact::ArtifactError("port: " + prefix +
                                          "expert bank mixes gguf blocks with another format");
        }
        gate_up_format = probed_gate_up;
        down_format    = probed_down;
    }
    const auto layout_of = [](NumericFormat format) {
        if (is_gguf(format)) { return kGgufLayout; }
        return format == NumericFormat::NVFP4 ? kExpertLayout : kBf16Layout;
    };
    const auto bind_expert = [&](std::string_view suffix, std::initializer_list<std::uint64_t> shape,
                                 NumericFormat format) {
        if (retain_experts_on_host) {
            return bind_optional_host(binder, prefix + std::string(suffix), format,
                                      layout_of(format), shape, enabled);
        }
        return bind_optional_device(binder, prefix + std::string(suffix), format, layout_of(format),
                                    shape, enabled);
    };
    const bool experts_gguf = is_gguf(gate_up_format);
    return {
        .router             = bf16("router", {512, 2'560}),
        .shared_down        = bf16("shared_expert/down", {2'560, 640}),
        .shared_gate        = bf16("shared_expert/gate", {640, 2'560}),
        .shared_up          = bf16("shared_expert/up", {640, 2'560}),
        .shared_gate_weight = bf16("shared_expert_gate", {1, 2'560}),
        .expert_gate_up     = bind_expert("experts/gate_up", {512, 1'280, 2'560}, gate_up_format),
        .expert_down        = bind_expert("experts/down", {512, 2'560, 640}, down_format),
        .experts_nvfp4      = !experts_gguf && (expert_format == NumericFormat::NVFP4),
        .experts_on_host    = retain_experts_on_host,
        .experts_gguf       = experts_gguf,
        .gguf_gate_up_format = gate_up_format,
        .gguf_down_format    = down_format,
    };
}

GdnPlan bind_gdn(Binder& binder, const std::string& prefix) {
    return {
        .a_log = bind_device(binder, prefix + "a_log", NumericFormat::BF16, kBf16Layout, {48}),
        .convolution = bind_device(binder, prefix + "convolution", NumericFormat::BF16, kBf16Layout,
                                   {4, 10'240}),
        .dt_bias = bind_device(binder, prefix + "dt_bias", NumericFormat::BF16, kBf16Layout, {48}),
        .a_b_projection = bind_device(binder, prefix + "a_b_projection", NumericFormat::BF16,
                                      kBf16Layout, {96, 2'560}),
        .norm = bind_device(binder, prefix + "norm", NumericFormat::BF16, kBf16Layout, {128}),
        .query_key_value_z =
            bind_device(binder, prefix + "query_key_value_z", NumericFormat::FP8_E4M3FN_ROW_F32,
                        kFp8Layout, {16'384, 2'560}),
        .output = bind_device(binder, prefix + "output", NumericFormat::FP8_E4M3FN_ROW_F32,
                              kFp8Layout, {2'560, 6'144}),
    };
}

AttentionPlan bind_attention(Binder& binder, const std::string& prefix,
                             NumericFormat projection_format, bool enabled = true) {
    const StorageLayout projection_layout =
        projection_format == NumericFormat::BF16 ? kBf16Layout : kFp8Layout;
    const auto bind = [&](std::string_view suffix, NumericFormat format, StorageLayout layout,
                          std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, prefix + std::string(suffix), format, layout, shape,
                                    enabled);
    };
    return {
        .indexer_query_key =
            bind("indexer/query_key", NumericFormat::BF16, kBf16Layout, {640, 2'560}),
        .indexer_key_norm   = bind("indexer/key_norm", NumericFormat::BF16, kBf16Layout, {128}),
        .indexer_query_norm = bind("indexer/query_norm", NumericFormat::BF16, kBf16Layout, {128}),
        .key_norm           = bind("key_norm", NumericFormat::BF16, kBf16Layout, {256}),
        .query_norm         = bind("query_norm", NumericFormat::BF16, kBf16Layout, {256}),
        .query_gate_key_value =
            bind("query_gate_key_value", projection_format, projection_layout, {13'312, 2'560}),
        .output = bind("output", projection_format, projection_layout, {2'560, 6'144}),
    };
}

PlePlan bind_ple(Binder& binder) {
    const std::string prefix = "text/layers/1/ple/";
    PlePlan out{
        .convolution = bind_device(binder, prefix + "convolution", NumericFormat::BF16, kBf16Layout,
                                   {4, 10'240}),
        .key_projection = bind_device(binder, prefix + "key_projection", NumericFormat::BF16,
                                      kBf16Layout, {10'240, 2'560}),
        .conv_norm =
            bind_device(binder, prefix + "conv_norm", NumericFormat::BF16, kBf16Layout, {10'240}),
        .key_norm =
            bind_device(binder, prefix + "key_norm", NumericFormat::BF16, kBf16Layout, {10'240}),
        .query_norm =
            bind_device(binder, prefix + "query_norm", NumericFormat::BF16, kBf16Layout, {10'240}),
        .value_projection   = bind_device(binder, prefix + "value_projection", NumericFormat::BF16,
                                          kBf16Layout, {2'560, 2'560}),
        .layer_multipliers  = bind_mapped(binder, prefix + "embedding/layer_multipliers",
                                          NumericFormat::I64, kBf16Layout, {3}),
        .ngram_head_offsets = bind_mapped(binder, prefix + "embedding/ngram_head_offsets",
                                          NumericFormat::I64, kBf16Layout, {16}),
        .ngram_head_vocab_sizes = bind_mapped(binder, prefix + "embedding/ngram_head_vocab_sizes",
                                              NumericFormat::I64, kBf16Layout, {16}),
    };
    for (std::size_t shard = 0; shard < out.shards.size(); ++shard) {
        out.shards[shard] =
            bind_mapped(binder, prefix + "embedding/shards/" + std::to_string(shard),
                        NumericFormat::U4Z8G16_F16S, kPleLayout, {2'500'012, 160});
    }
    require_i64_values(binder, out.layer_multipliers, kPleLayerMultipliers, "PLE multipliers");
    require_i64_values(binder, out.ngram_head_offsets, kPleHeadOffsets, "PLE head offsets");
    require_i64_values(binder, out.ngram_head_vocab_sizes, kPleHeadVocabSizes,
                       "PLE head vocabularies");
    return out;
}

MtpPlan bind_mtp(Binder& binder, bool enabled) {
    MtpPlan out;
    out.embedding_projection =
        bind_optional_device(binder, "mtp/embedding_projection", NumericFormat::BF16, kBf16Layout,
                             {2'560, 2'560}, enabled);
    out.hidden_projection = bind_optional_device(
        binder, "mtp/hidden_projection", NumericFormat::BF16, kBf16Layout, {2'560, 2'560}, enabled);
    out.mixer           = bind_mixer(binder, "mtp/hyper_connection/", enabled);
    out.attention_hyper = bind_hyper(binder, "mtp/layer/attention/hyper_connection/", enabled);

    // Format-tolerant MTP MoE expert binding:
    // Support both new NVFP4 spliced artifacts and legacy BF16 artifacts seamlessly.
    NumericFormat mtp_expert_format = NumericFormat::NVFP4;
    bool retain_on_host             = false;
    // The descriptor lookup is the ADAPTER's, not the engine reader's: this engine's `Reader::find`
    // hands back an `ObjectHandle`, while the upstream call site expects a descriptor it can inspect.
    const port::ObjectDescriptor* gate_up_desc = binder.find("mtp/layer/mlp/experts/gate_up");
    if (gate_up_desc != nullptr) {
        if (const auto* tensor = std::get_if<port::TensorDescriptor>(gate_up_desc)) {
            if (tensor->format == NumericFormat::BF16) {
                mtp_expert_format = NumericFormat::BF16;
                retain_on_host    = true;
            }
        }
    }
    out.moe = bind_moe(binder, "mtp/layer/mlp/", mtp_expert_format, enabled, retain_on_host);

    out.mlp_hyper       = bind_hyper(binder, "mtp/layer/mlp/hyper_connection/", enabled);
    out.attention = bind_attention(binder, "mtp/layer/attention/", NumericFormat::BF16, enabled);
    out.embedding_norm = bind_optional_device(binder, "mtp/embedding_norm", NumericFormat::BF16,
                                              kBf16Layout, {2'560}, enabled);
    out.hidden_norm    = bind_optional_device(binder, "mtp/hidden_norm", NumericFormat::BF16,
                                              kBf16Layout, {10'240}, enabled);
    return out;
}

VisionPlan bind_vision(Binder& binder, bool enabled) {
    const auto bind = [&](const std::string& name, std::initializer_list<std::uint64_t> shape) {
        return bind_optional_device(binder, name, NumericFormat::BF16, kBf16Layout, shape, enabled);
    };
    VisionPlan out;
    out.patch_embedding      = bind("vision/patch_embedding", {1'152, 1'536});
    out.patch_embedding_bias = bind("vision/patch_embedding_bias", {1'152});
    out.position_embedding   = bind("vision/position_embedding", {2'304, 1'152});
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        out.layers[layer]        = {
                   .qkv          = bind(prefix + "attention/qkv", {3'456, 1'152}),
                   .qkv_bias     = bind(prefix + "attention/qkv_bias", {3'456}),
                   .output       = bind(prefix + "attention/output", {1'152, 1'152}),
                   .output_bias  = bind(prefix + "attention/output_bias", {1'152}),
                   .fc1          = bind(prefix + "mlp/fc1", {4'304, 1'152}),
                   .fc1_bias     = bind(prefix + "mlp/fc1_bias", {4'304}),
                   .fc2          = bind(prefix + "mlp/fc2", {1'152, 4'304}),
                   .fc2_bias     = bind(prefix + "mlp/fc2_bias", {1'152}),
                   .norm1_weight = bind(prefix + "norm1/weight", {1'152}),
                   .norm1_bias   = bind(prefix + "norm1/bias", {1'152}),
                   .norm2_weight = bind(prefix + "norm2/weight", {1'152}),
                   .norm2_bias   = bind(prefix + "norm2/bias", {1'152}),
        };
    }
    out.merger_fc1         = bind("vision/merger/fc1", {4'608, 4'608});
    out.merger_fc1_bias    = bind("vision/merger/fc1_bias", {4'608});
    out.merger_fc2         = bind("vision/merger/fc2", {2'560, 4'608});
    out.merger_fc2_bias    = bind("vision/merger/fc2_bias", {2'560});
    out.merger_norm_weight = bind("vision/merger/norm/weight", {1'152});
    out.merger_norm_bias   = bind("vision/merger/norm/bias", {1'152});
    return out;
}

} // namespace

ArtifactLoadPlan bind_artifact(Binder& binder, LoadFeatures features) {    ArtifactLoadPlan out;
    BindingPlan& plan    = out.bindings;
    plan.features        = features;
    // `bind_frontend_resources` is the ported qwen3_6 target's frontend-resource binder. Upstream's
    // signature takes the upstream `artifact::Binder&`, which in this port is the adapter, so it is
    // called with the adapter and the ported body forwards to the adapter's `bind_raw_resource`.
    plan.frontend        = qwen3_5::bind_frontend_resources(binder.engine());
    // FP8 at load: keep the BF16 payload in the file mapping (retain_mapped_tensor adds zero
    // device bytes) and let LoadedModelData quantize from the host span. Uploading it first would
    // leave the BF16 copy pinned in the artifact's bump arena, which has no per-object free.
    plan.token_embedding =
        features.quantize_token_embedding_fp8
            ? bind_mapped(binder, "text/token_embedding", NumericFormat::BF16, kBf16Layout,
                          {248'320, 2'560})
            : bind_device(binder, "text/token_embedding", NumericFormat::BF16, kBf16Layout,
                          {248'320, 2'560});

    for (std::size_t layer = 0; layer < plan.text_layers.size(); ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        TextLayerPlan& target    = plan.text_layers[layer];
        target.attention_hyper   = bind_hyper(binder, prefix + "attention/hyper_connection/");
        // Host-tiered placement (NINFER_FLASH_NEXT_EXPERTS_HOST): the 48 layers' 63.3 GiB of routed
        // experts stay in the file mapping and the MoE leaf stages the routed subset per layer.
        // Without it the target plan asks for 70.01 GiB of device weights and a 32 GB card refuses.
        target.moe               = bind_moe(binder, prefix + "mlp/", NumericFormat::NVFP4,
                                             /*enabled=*/true,
                                             /*retain_experts_on_host=*/flash_next_experts_on_host_enabled());
        target.mlp_hyper         = bind_hyper(binder, prefix + "mlp/hyper_connection/");
        target.is_full_attention = layer >= 3 && (layer - 3) % 4 == 0;
        if (target.is_full_attention) {
            target.attention =
                bind_attention(binder, prefix + "attention/", NumericFormat::FP8_E4M3FN_ROW_F32);
        } else {
            target.gdn = bind_gdn(binder, prefix + "gdn/");
        }
    }

    plan.ple         = bind_ple(binder);
    plan.output_head = features.quantize_output_head_fp8
                           ? bind_mapped(binder, "text/output_head", NumericFormat::BF16,
                                         kBf16Layout, {248'320, 2'560})
                           : bind_device(binder, "text/output_head", NumericFormat::BF16,
                                         kBf16Layout, {248'320, 2'560});
    plan.final_mixer    = bind_mixer(binder, "text/hyper_connection/");
    plan.mtp            = bind_mtp(binder, features.mtp);
    plan.vision         = bind_vision(binder, features.vision);
    // The adapter's `finish` is rvalue-ref qualified, matching this engine's, which hands the
    // recorded demands to the plan. This is the one place the ported call site says `std::move`.
    out.materialization = std::move(binder).finish();
    // One line a reader can diff against the frozen expectation (docs/32 §4): the device sign is
    // what the loader's `loading weights | X GiB` phase reports, and the mapped count is what the
    // digest comparator cross-checks (registration 227 with the disabled subtrees stripped, 229
    // without; bytes actually read through the mapping 224).
    std::fprintf(stderr,
                 "flash_next plan device_bytes=%llu device_tensors=%zu mapped_tensors=%zu "
                 "pinned_tensors=%zu retained_resources=%zu\n",
                 static_cast<unsigned long long>(out.materialization.total_device_capacity_bytes()),
                 out.materialization.device_objects.size(),
                 out.materialization.mapped_objects.size(),
                 out.materialization.pinned_objects.size(),
                 out.materialization.host_objects.size());
    return out;
}

// The engine-binder overload the reference declares (`bindings.h:156`) and `loader.cpp:23` calls.
// Not in the reference's .cpp because the reference had no adapter: there, both call sites passed the
// same `artifact::Binder`. Here the body above needs the adapter, so this overload exists purely to
// accept the engine binder and hand the adapter its `engine()` view. Semantics unchanged -- the
// adapter wraps the very same binder object.
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, LoadFeatures features) {
    Binder adapted(binder);
    return bind_artifact(adapted, features);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
