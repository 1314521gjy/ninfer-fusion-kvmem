#pragma once

#include "artifact/binder.h"

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/resources.h"

namespace ninfer::artifact {
class MaterializedArtifact;
}

namespace ninfer::models::qwen3_5 {

// Frontend resource *handles* for the loader: the artifact records the six frontend resources as
// file objects, and this plan carries their handles until they are taken.
//
// Ported from the upstream export header
// (upstream src/targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend_resources.h), which also
// defines a 6x `std::string` struct named `FrontendResources` right here. That struct is
// deliberately NOT copied: in this tree the name already has exactly one definition --
// `models/qwen3_5/frontend/resources.h` (borrowed `string_view` views plus the owned Tokenizer and
// `public_token_count`) -- and that is the type the frontend and its 30+ call sites consume. A
// second definition would give one fact two sources of truth, so `take_frontend_resources` below
// returns that canonical type instead.
struct FrontendResourcePlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    artifact::ObjectHandle generation_config_json;
    artifact::ObjectHandle preprocessor_config_json;
    artifact::ObjectHandle video_preprocessor_config_json;
};

[[nodiscard]] FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder);
[[nodiscard]] FrontendResources take_frontend_resources(artifact::MaterializedArtifact& artifact,
                                                        const FrontendResourcePlan& plan);

} // namespace ninfer::models::qwen3_5
