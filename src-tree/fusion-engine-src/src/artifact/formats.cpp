#include "artifact/formats.h"

#include "artifact/schema.h"

#include <array>

namespace ninfer::artifact {
namespace {

constexpr std::array kFormats = {
    std::pair{QType::BF16, std::string_view{"bf16"}},
    std::pair{QType::FP32, std::string_view{"fp32"}},
    std::pair{QType::INT32, std::string_view{"int32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"q4_g64_fp16"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"q5_g64_fp16"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"q6_g64_fp16"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"q8_g32_fp16"}},
    std::pair{QType::T2_G128_FP16, std::string_view{"t2_g128_fp16"}},
    std::pair{QType::PTQ1_G128_FP16, std::string_view{"ptq1_g128_fp16"}},
    std::pair{QType::NVFP4, std::string_view{"nvfp4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"fp8_e4m3fn_row_bf16"}},
    std::pair{QType::GGUF_Q8_0, std::string_view{"gguf_q8_0"}},
    std::pair{QType::GGUF_Q2_K, std::string_view{"gguf_q2_k"}},
    std::pair{QType::GGUF_Q3_K, std::string_view{"gguf_q3_k"}},
    std::pair{QType::GGUF_Q4_K, std::string_view{"gguf_q4_k"}},
    std::pair{QType::GGUF_Q5_K, std::string_view{"gguf_q5_k"}},
    std::pair{QType::GGUF_Q6_K, std::string_view{"gguf_q6_k"}},
    std::pair{QType::GGUF_IQ2_XXS, std::string_view{"gguf_iq2_xxs"}},
    std::pair{QType::GGUF_IQ2_XS, std::string_view{"gguf_iq2_xs"}},
    std::pair{QType::GGUF_IQ2_S, std::string_view{"gguf_iq2_s"}},
    std::pair{QType::GGUF_IQ3_XXS, std::string_view{"gguf_iq3_xxs"}},
    std::pair{QType::GGUF_IQ3_S, std::string_view{"gguf_iq3_s"}},
    std::pair{QType::GGUF_IQ1_S, std::string_view{"gguf_iq1_s"}},
    std::pair{QType::GGUF_IQ1_M, std::string_view{"gguf_iq1_m"}},
    std::pair{QType::GGUF_IQ4_NL, std::string_view{"gguf_iq4_nl"}},
    std::pair{QType::GGUF_IQ4_XS, std::string_view{"gguf_iq4_xs"}},
    // Block-5 IQ3_S artifact: the v3 tooling's name for ggml Q2_0 (type id 42), written on nine
    // `ffn_down_exps` objects. The name the container carries, so the engine and the tooling agree.
    std::pair{QType::GGUF_Q2_0, std::string_view{"gguf_q2_0"}},
    // The two names the Qwen3.8-Flash-Next container writes for its PLE side data. Upstream stores
    // them in upper case and matches that spelling exactly, so the stored name is what is listed
    // here -- whoever writes a name uses this table, so the engine and the container agree.
    std::pair{QType::I64, std::string_view{"I64"}},
    std::pair{QType::U4Z8G16_F16S, std::string_view{"U4Z8G16_F16S"}},
    // Third name the same container writes: the FP32-scale row-scaled FP8 spelling. Upstream calls
    // the 2-byte one `FP8_E4M3FN_ROW_BF16S`; both spellings are listed here under the names the
    // container uses, so reading a container and writing one agree.
    std::pair{QType::FP8_E4M3FN_ROW_F32, std::string_view{"FP8_E4M3FN_ROW_F32S"}},
    // ---- The rest of the upstream container's names -------------------------------
    // The container the Qwen3.8-Flash-Next artifacts are published in spells format names in UPPER
    // CASE and layout names with hyphens, while this engine's own writer uses lower case and
    // underscores for the same encodings. Both spellings name the same values, so the table lists
    // both rather than the enum growing a second value per format. Names copied verbatim from the
    // upstream `parse_format` / `parse_layout` lists so the two containers agree on every object.
    std::pair{QType::BF16, std::string_view{"BF16"}},
    std::pair{QType::FP32, std::string_view{"FP32"}},
    std::pair{QType::INT32, std::string_view{"I32"}},
    std::pair{QType::Q4_G64_FP16, std::string_view{"Q4G64_F16S"}},
    std::pair{QType::Q5_G64_FP16, std::string_view{"Q5G64_F16S"}},
    std::pair{QType::Q6_G64_FP16, std::string_view{"Q6G64_F16S"}},
    std::pair{QType::Q8_G32_FP16, std::string_view{"W8G32_F16S"}},
    std::pair{QType::NVFP4, std::string_view{"NVFP4"}},
    std::pair{QType::FP8_E4M3FN_ROW_BF16, std::string_view{"FP8_E4M3FN_ROW_BF16S"}},
    // ---- The v3 tooling's spellings --------------------------------------------------
    // `tools/artifact/` (the Python v3 writer) names formats in lower case and layouts with
    // underscores. A v3 container written by that tooling -- which is how the Qwen3.8-Flash-Next
    // artifact is converted here -- therefore carries these names, and without them the engine
    // parses a freshly converted artifact as "unknown format": the file would be unreadable by the
    // very engine it was converted for. Same values, second spelling, so no enum gains a member.
    std::pair{QType::I64, std::string_view{"int64"}},
    std::pair{QType::U4Z8G16_F16S, std::string_view{"u4z8g16_f16"}},
    std::pair{QType::FP8_E4M3FN_ROW_F32, std::string_view{"fp8_e4m3fn_row_f32"}},
};
constexpr std::array kLayouts = {
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous_le_v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row_split_k128_v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"block_scale_k16_m128x4_v1"}},
    std::pair{QuantLayout::GgufBlocks, std::string_view{"gguf_blocks_v1"}},
    // The Qwen3.8-Flash-Next PLE shard layout. It is a stored layout -- the published artifact
    // declares it on 128 objects -- so it belongs in `kLayouts`, not in `kDeviceLayouts`.
    std::pair{QuantLayout::PackedU4G16, std::string_view{"packed-u4-g16-v1"}},
    // Row-scaled FP8 with a 4-byte scale word, under the second wire name the same container uses.
    // Same layout as `row_scale_v1`: the scale word width is a property of the FORMAT, not of the
    // layout (upstream `row_scale_geometry` picks 4 when the format ends in F32S), so this is a
    // second spelling for one layout rather than a new one. Both names must parse -- the container
    // writes this one on 96 objects.
    std::pair{QuantLayout::RowScale, std::string_view{"row-scale-f32-v1"}},
    // The container's hyphenated spellings of the other layouts, copied verbatim from the upstream
    // `parse_layout` list. Same layouts, second spelling -- see the format note above.
    std::pair{QuantLayout::Contiguous, std::string_view{"contiguous-le-v1"}},
    std::pair{QuantLayout::RowSplit, std::string_view{"row-split-k128-v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4, std::string_view{"blockscale-k16-m128x4-v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4,
              std::string_view{"expert-blockscale-k16-m128x4-v1"}},
    std::pair{QuantLayout::RowScale, std::string_view{"row-scale-v1"}},
    // The v3 tooling's underscore spellings of the same layouts (see the format note above): the
    // 4-byte row-scale pair, the rank-three PLE shard, and the expert bank the v3 writer names
    // `expert_blockscale_k16_m128x4_v1` for the layout the engine calls BlockScaleK16M128x4.
    std::pair{QuantLayout::RowScale, std::string_view{"row_scale_f32_v1"}},
    std::pair{QuantLayout::PackedU4G16, std::string_view{"packed_u4_g16_v1"}},
    std::pair{QuantLayout::BlockScaleK16M128x4,
              std::string_view{"expert_blockscale_k16_m128x4_v1"}},
};

// Layouts a device may hold but an artifact may never declare. They are produced by a load-time
// permute, so `parse_layout` deliberately does not see them: a `.ninfer` claiming one would be
// describing bytes the writer could not have produced.
constexpr std::array kDeviceLayouts = {
    std::pair{QuantLayout::RowSplitPanel, std::string_view{"row_split_panel4_v1"}},
};

} // namespace

QType parse_format(std::string_view name) {
    for (const auto& [format, spelling] : kFormats) {
        if (spelling == name) { return format; }
    }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

QuantLayout parse_layout(std::string_view name) {
    for (const auto& [layout, spelling] : kLayouts) {
        if (spelling == name) { return layout; }
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

std::string_view format_name(QType format) noexcept {
    for (const auto& [value, spelling] : kFormats) {
        if (value == format) { return spelling; }
    }
    return {};
}

std::string_view layout_name(QuantLayout layout) noexcept {
    for (const auto& [value, spelling] : kLayouts) {
        if (value == layout) { return spelling; }
    }
    for (const auto& [value, spelling] : kDeviceLayouts) {
        if (value == layout) { return spelling; }
    }
    return {};
}

} // namespace ninfer::artifact
