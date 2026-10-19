#include "models/qwen3_5/frontend/frontend_resources.h"

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"

#include <cstddef>
#include <string_view>

namespace ninfer::models::qwen3_5 {
namespace {

// The canonical `FrontendResources` in this tree holds BORROWED `std::string_view` fields -- its own
// header says so ("Byte views borrow the model's Host backing", `resources.h:13-17`) -- and the
// frontend plus its call sites consume that type. So these views must point at storage that outlives
// them: the artifact's retained Host copy, which lives as long as the `MaterializedArtifact` its
// owner holds (`impl/load/materialized.cpp` keeps it in `LoadedModelData`).
//
// This used to build the views from `read_string(...)` temporaries (`std::string` by value), which
// dangle the moment the constructor's full expression ends. The engine then parsed freed heap and
// died with `[json.exception.parse_error.101] parse error at line 287, column 16 ... last read:
// '"eos_token"e'; expected ':'` -- and the file's line 287 really is `"eos_token": "<|im_end|>",`
// with the text intact up to that byte, which is what a mostly-not-yet-reused freed buffer looks
// like. Borrowing instead of copying also drops six copies, 12.8 MB of tokenizer JSON among them.
std::string_view view_host_bytes(const artifact::MaterializedArtifact& materialized,
                                 artifact::ObjectHandle handle) {
    const std::span<const std::byte> bytes = materialized.host_bytes(handle);
    return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

// Ported from `src/targets/qwen3_6/impl/frontend/resources.cpp:20-33`. The resource names are the
// container's own object names, measured from the published artifact's directory, and this engine
// indexes resources by exactly those names (`impl/port/binder_port.h:388-397`).
FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    return FrontendResourcePlan{
        .tokenizer_json = port::bind_raw_resource(binder, "frontend/tokenizer.json"),
        .tokenizer_config_json =
            port::bind_raw_resource(binder, "frontend/tokenizer_config.json"),
        .chat_template_jinja = port::bind_raw_resource(binder, "frontend/chat_template.jinja"),
        .generation_config_json =
            port::bind_raw_resource(binder, "frontend/generation_config.json"),
        .preprocessor_config_json =
            port::bind_raw_resource(binder, "frontend/preprocessor_config.json"),
        .video_preprocessor_config_json =
            port::bind_raw_resource(binder, "frontend/video_preprocessor_config.json"),
    };
}

// Returns THIS tree's canonical `qwen3_5::FrontendResources`
// (`models/qwen3_5/frontend/resources.h`, borrowed `string_view` views plus the owned Tokenizer), not
// the upstream 6x-`std::string` struct of the same name -- see the note in `frontend_resources.h:17-24`
// for why a second definition was refused. Because the canonical type BORROWS, the fields below are
// views into the artifact's retained Host bytes; the caller keeps that artifact alive for the whole
// frontend lifetime.
FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = view_host_bytes(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = view_host_bytes(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = view_host_bytes(materialized, plan.chat_template_jinja),
        .generation_config_json   = view_host_bytes(materialized, plan.generation_config_json),
        .preprocessor_config_json = view_host_bytes(materialized, plan.preprocessor_config_json),
        .video_preprocessor_config_json =
            view_host_bytes(materialized, plan.video_preprocessor_config_json),
    };
}

} // namespace ninfer::models::qwen3_5
