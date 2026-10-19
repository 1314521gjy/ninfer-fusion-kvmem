#include "ninfer/models/qwen3_vision/vision.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/gelu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/vision_pos_embed.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_vision {
namespace {

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string("Vision ") + label + " overflows size_t");
    }
    return a + b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("Vision ") + label +
                                    " alignment must be a power of two");
    }
    return checked_add(value, alignment - 1, label) & ~(alignment - 1);
}

constexpr std::size_t kWorkspaceAlignment = 256;

struct WorkspaceLayout {
    TensorRegion position_ids;
    TensorRegion pos_indices;
    TensorRegion pos_weights;
    TensorRegion x;
    TensorRegion patch_bf16;
    TensorRegion attended;
    TensorRegion qkv;
    TensorRegion attention_norm;
    TensorRegion projected;
    TensorRegion mlp_down;
    TensorRegion mlp_up;
    TensorRegion mlp_norm;
    TensorRegion normalized;
    TensorRegion merger_hidden;
    std::size_t bytes = 0;
};

TensorRegion alias_tensor(const TensorRegion& storage, DType dtype,
                          std::initializer_list<std::int32_t> shape, const char* label) {
    Tensor tensor(nullptr, dtype, shape);
    if (tensor.bytes() > storage.region.bytes) {
        throw std::logic_error(std::string("Vision ") + label +
                               " does not fit its aliased storage");
    }
    TensorRegion out;
    out.region = LayoutRegion{storage.region.offset, tensor.bytes(), storage.region.alignment};
    out.dtype  = dtype;
    std::copy(shape.begin(), shape.end(), out.shape.begin());
    return out;
}

WorkspaceLayout build_workspace_layout(std::size_t patches64, std::size_t tokens64) {
    if (patches64 == 0 || tokens64 == 0 ||
        patches64 !=
            checked_mul(tokens64, VisionTowerConfig::merge_unit, "patch/token relation")) {
        throw std::invalid_argument("Vision workspace requires P=4V>0");
    }
    if (patches64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        tokens64 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("Vision request dimensions exceed int32");
    }
    const auto patches = static_cast<std::int32_t>(patches64);
    const auto tokens  = static_cast<std::int32_t>(tokens64);

    LayoutBuilder builder;
    WorkspaceLayout out;
    const auto add = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                         const char* label) {
        return builder.add_tensor(dtype, shape, kWorkspaceAlignment, label);
    };
    out.x = add(DType::BF16, {VisionTowerConfig::hidden, patches}, "vision residual");
    {
        auto position_lifetime = builder.scope();
        out.position_ids       = add(DType::I32, {patches, 2}, "vision position ids");
        {
            auto patch_scope = builder.scope();
            out.patch_bf16 =
                add(DType::BF16, {VisionTowerConfig::patch_dim, patches}, "vision BF16 patches");
        }
        {
            auto position_scope = builder.scope();
            out.pos_indices     = add(DType::I32, {4, patches}, "vision position indices");
            out.pos_weights     = add(DType::FP32, {4, patches}, "vision position weights");
        }
        {
            auto attention_scope = builder.scope();
            out.qkv = add(DType::BF16, {3 * VisionTowerConfig::hidden, patches}, "vision QKV");
            out.attention_norm = add(DType::BF16, {VisionTowerConfig::hidden, patches},
                                     "vision attention norm/attended");
            out.attended       = out.attention_norm;
            out.projected =
                alias_tensor(out.qkv, DType::BF16, {VisionTowerConfig::hidden, patches},
                             "attention projection output");
        }
        {
            auto mlp_scope = builder.scope();
            out.mlp_up =
                add(DType::BF16, {VisionTowerConfig::intermediate, patches}, "vision MLP up");
            out.mlp_norm =
                add(DType::BF16, {VisionTowerConfig::hidden, patches}, "vision MLP norm/down");
            out.mlp_down = out.mlp_norm;
        }
    }
    {
        auto merger_scope = builder.scope();
        out.normalized =
            add(DType::BF16, {VisionTowerConfig::hidden, patches}, "vision merger norm");
        out.merger_hidden = alias_tensor(
            out.x, DType::BF16, {VisionTowerConfig::merger_hidden, tokens}, "merger hidden");
    }
    out.bytes = builder.finish(1, "vision workspace");
    return out;
}

std::size_t output_handoff_bytes(std::size_t merged_tokens, std::int32_t output_hidden) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision output handoff extent must fit positive int32");
    }
    if (output_hidden <= 0) {
        throw std::invalid_argument("Vision output hidden dimension must be positive");
    }
    LayoutBuilder layout;
    (void)layout.add_tensor(
        DType::BF16, {output_hidden, static_cast<std::int32_t>(merged_tokens)},
        kWorkspaceAlignment, "Vision item output handoff");
    return layout.finish(kWorkspaceAlignment, "Vision item output handoff layout");
}

std::size_t merger_hidden_bytes(std::size_t merged_tokens) {
    if (merged_tokens == 0 ||
        merged_tokens > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Vision merger hidden extent must fit positive int32");
    }
    Tensor tensor(nullptr, DType::BF16,
                  {VisionTowerConfig::merger_hidden, static_cast<std::int32_t>(merged_tokens)});
    return tensor.bytes();
}

void copy_host(const void* src, Tensor& dst, cudaStream_t stream) {
    if (dst.bytes() == 0) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

void validate_weights(const Weights& weights) {
    if (weights.output_hidden <= 0) {
        throw std::invalid_argument("Vision output_hidden must be positive");
    }
    if (weights.patch_embed == nullptr || weights.patch_embed_bias == nullptr ||
        weights.position_embed == nullptr) {
        throw std::invalid_argument("Vision common embedding weights cannot be null");
    }
    for (std::size_t i = 0; i < weights.layers.size(); ++i) {
        const auto& layer = weights.layers[i];
        if (layer.norm1_weight == nullptr || layer.norm1_bias == nullptr ||
            layer.qkv == nullptr || layer.qkv_bias == nullptr ||
            layer.projection == nullptr || layer.projection_bias == nullptr ||
            layer.norm2_weight == nullptr || layer.norm2_bias == nullptr ||
            layer.fc1 == nullptr || layer.fc1_bias == nullptr ||
            layer.fc2 == nullptr || layer.fc2_bias == nullptr) {
            throw std::invalid_argument("Vision layer weights cannot be null at layer " +
                                        std::to_string(i));
        }
    }
    if (weights.merger.norm_weight == nullptr || weights.merger.norm_bias == nullptr ||
        weights.merger.fc1 == nullptr || weights.merger.fc1_bias == nullptr ||
        weights.merger.fc2 == nullptr || weights.merger.fc2_bias == nullptr) {
        throw std::invalid_argument("Vision merger weights cannot be null");
    }
}

} // namespace

Encoder::Encoder(DeviceContext& device, const Weights& weights)
    : device_(device), weights_(weights) {
    validate_weights(weights_);
}

std::size_t Encoder::workspace_bytes(std::size_t patches, std::size_t merged_tokens) {
    return build_workspace_layout(patches, merged_tokens).bytes;
}

WorkspacePlan Encoder::plan_workspace(std::uint32_t max_merged_tokens,
                                      std::size_t general_capacity_bytes,
                                      std::int32_t output_hidden) {
    if (max_merged_tokens == 0) {
        throw std::invalid_argument("Vision workspace capacity bound must be positive");
    }
    if (general_capacity_bytes == 0) {
        throw std::invalid_argument("Vision general workspace capacity must be positive");
    }
    if (output_hidden <= 0) {
        throw std::invalid_argument("Vision output hidden dimension must be positive");
    }
    WorkspacePlan out;
    out.max_merged_tokens      = max_merged_tokens;
    out.general_capacity_bytes = general_capacity_bytes;
    out.encode_peak_bytes =
        build_workspace_layout(checked_mul(max_merged_tokens, VisionTowerConfig::merge_unit,
                                           "capacity patch count"),
                               max_merged_tokens)
            .bytes;
    out.handoff_offset_bytes =
        align_up(std::max(general_capacity_bytes, merger_hidden_bytes(max_merged_tokens)),
                 kWorkspaceAlignment, "handoff offset");
    out.handoff_capacity_bytes = output_handoff_bytes(max_merged_tokens, output_hidden);
    out.capacity_bytes         = std::max(
        out.encode_peak_bytes,
        checked_add(out.handoff_offset_bytes, out.handoff_capacity_bytes, "workspace capacity"));
    return out;
}

Tensor Encoder::bind_output(DeviceSpan backing, const WorkspacePlan& plan,
                            std::size_t merged_tokens,
                            std::int32_t output_hidden) {
    if (backing.data == nullptr || backing.bytes < plan.capacity_bytes || merged_tokens == 0 ||
        merged_tokens > plan.max_merged_tokens) {
        throw std::invalid_argument("Vision output binding exceeds its workspace plan");
    }
    if (output_hidden <= 0) {
        throw std::invalid_argument("Vision output hidden dimension must be positive");
    }
    const std::size_t bytes = output_handoff_bytes(merged_tokens, output_hidden);
    if (bytes > plan.handoff_capacity_bytes) {
        throw std::logic_error("Vision output binding exceeds its handoff region");
    }
    TensorRegion region;
    region.region = LayoutRegion{plan.handoff_offset_bytes, bytes, kWorkspaceAlignment};
    region.dtype  = DType::BF16;
    region.shape  = {output_hidden, static_cast<std::int32_t>(merged_tokens), 1, 1};
    return region.bind(backing);
}

void Encoder::encode(const EncodeItemView& item, Tensor& output, DeviceSpan backing,
                     const WorkspacePlan& plan) const {
    const auto patches64 = item.patch_count;
    const auto tokens64  = item.merged_count;
    if (item.patches.size() !=
        checked_mul(patches64, VisionTowerConfig::patch_dim, "patch elements")) {
        throw std::invalid_argument("Vision processor patch buffer has invalid shape");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != weights_.output_hidden ||
        output.ne[1] != static_cast<std::int32_t>(tokens64) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("Vision output must be contiguous BF16 [H,V]");
    }
    const Tensor planned_output = bind_output(backing, plan, tokens64, weights_.output_hidden);
    if (output.data != planned_output.data || output.bytes() != planned_output.bytes()) {
        throw std::invalid_argument("Vision output does not name the planned handoff region");
    }
    const WorkspaceLayout layout = build_workspace_layout(patches64, tokens64);
    if (layout.bytes > plan.encode_peak_bytes || backing.bytes < plan.capacity_bytes) {
        throw std::invalid_argument("Vision workspace capacity is too small for request");
    }
    const auto patches  = static_cast<std::int32_t>(patches64);
    const auto tokens   = static_cast<std::int32_t>(tokens64);
    cudaStream_t stream = device_.stream;

    if (item.position_ids.size() != static_cast<std::size_t>(patches) * 2) {
        throw std::invalid_argument("Vision item position_ids has invalid size");
    }
    if (item.position_table_indices.size() != static_cast<std::size_t>(patches) * 4) {
        throw std::invalid_argument("Vision item position_table_indices has invalid size");
    }
    if (item.position_table_weights.size() != static_cast<std::size_t>(patches) * 4) {
        throw std::invalid_argument("Vision item position_table_weights has invalid size");
    }

    Tensor position_ids = layout.position_ids.bind(backing);
    copy_host(item.position_ids.data(), position_ids, stream);

    Tensor x          = layout.x.bind(backing);
    Tensor patch_bf16 = layout.patch_bf16.bind(backing);
    copy_host(item.patches.data(), patch_bf16, stream);
    ops::linear(patch_bf16, *weights_.patch_embed, x, stream);
    ops::add_bias(*weights_.patch_embed_bias, x, stream);
    // The artifact records the source table shape [rows,hidden], while Tensor's
    // contiguous matrix convention is [inner,columns]. The payload is already
    // row-major, so this is a zero-copy [hidden,rows] view, not a transpose.
    Tensor pos_indices = layout.pos_indices.bind(backing);
    Tensor pos_weights = layout.pos_weights.bind(backing);
    copy_host(item.position_table_indices.data(), pos_indices, stream);
    copy_host(item.position_table_weights.data(), pos_weights, stream);
    Tensor position_table = weights_.position_embed->reshape(
        {VisionTowerConfig::hidden, VisionTowerConfig::position_embeddings});
    ops::vision_pos_embed_add(position_table, pos_indices, pos_weights, x, stream);
    for (std::size_t layer = 0; layer < weights_.layers.size(); ++layer) {
        const LayerWeights& block = weights_.layers[layer];
        {
            Tensor attended = layout.attended.bind(backing);
            {
                Tensor qkv = layout.qkv.bind(backing);
                {
                    Tensor h = layout.attention_norm.bind(backing);
                    ops::layer_norm(x, *block.norm1_weight, *block.norm1_bias,
                                    VisionTowerConfig::norm_epsilon, h, stream);
                    ops::linear(h, *block.qkv, qkv, stream);
                }
                ops::add_bias(*block.qkv_bias, qkv, stream);
                const std::int32_t plane      = VisionTowerConfig::hidden;
                const std::size_t plane_bytes = static_cast<std::size_t>(plane) * 2;
                Tensor q(qkv.data, DType::BF16,
                         {VisionTowerConfig::head_dim, VisionTowerConfig::heads, patches});
                Tensor k(static_cast<unsigned char*>(qkv.data) + plane_bytes, DType::BF16,
                         {VisionTowerConfig::head_dim, VisionTowerConfig::heads, patches});
                Tensor v(static_cast<unsigned char*>(qkv.data) + 2 * plane_bytes, DType::BF16,
                         {VisionTowerConfig::head_dim, VisionTowerConfig::heads, patches});
                q.nb[2] = qkv.nb[1];
                k.nb[2] = qkv.nb[1];
                v.nb[2] = qkv.nb[1];
                ops::rope(position_ids, VisionTowerConfig::rotary_dim,
                          VisionTowerConfig::rope_theta, q, k, stream);
                Tensor attended_heads = attended.view(
                    {VisionTowerConfig::head_dim, VisionTowerConfig::heads, patches});
                ops::packed_softmax_attention(q, k, v,
                                              {VisionTowerConfig::head_dim,
                                               VisionTowerConfig::heads,
                                               VisionTowerConfig::heads},
                                              VisionTowerConfig::attention_scale,
                                              item.segment_length, attended_heads, stream);
            }
            Tensor projected = layout.projected.bind(backing);
            ops::linear(attended, *block.projection, projected, stream);
            ops::add_bias(*block.projection_bias, projected, stream);
            ops::residual_add(projected, x, stream);
        }
        {
            Tensor down = layout.mlp_down.bind(backing);
            Tensor up   = layout.mlp_up.bind(backing);
            {
                Tensor h = layout.mlp_norm.bind(backing);
                ops::layer_norm(x, *block.norm2_weight, *block.norm2_bias,
                                VisionTowerConfig::norm_epsilon, h, stream);
                ops::linear(h, *block.fc1, up, stream);
            }
            ops::add_bias(*block.fc1_bias, up, stream);
            ops::gelu(up, ops::GeluMode::Tanh, stream);
            ops::linear(up, *block.fc2, down, stream);
            ops::add_bias(*block.fc2_bias, down, stream);
            ops::residual_add(down, x, stream);
        }
    }

    Tensor normalized = layout.normalized.bind(backing);
    ops::layer_norm(x, *weights_.merger.norm_weight, *weights_.merger.norm_bias,
                    VisionTowerConfig::norm_epsilon, normalized, stream);
    Tensor merged = normalized.view({VisionTowerConfig::merger_hidden, tokens});
    Tensor hidden = layout.merger_hidden.bind(backing);
    ops::linear(merged, *weights_.merger.fc1, hidden, stream);
    ops::add_bias(*weights_.merger.fc1_bias, hidden, stream);
    ops::gelu(hidden, ops::GeluMode::Exact, stream);
    ops::linear(hidden, *weights_.merger.fc2, output, stream);
    ops::add_bias(*weights_.merger.fc2_bias, output, stream);
}

} // namespace ninfer::models::qwen3_vision
