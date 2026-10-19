#include "runtime/contract/structured_output.h"

// Verbatim from upstream (igorls/ninfer @ c81c88f6) apart from ONE substitution:
// upstream spells the no-constraint enumerator `StructuredOutputKind::Text` and this engine spells
// the same value `StructuredOutputKind::None` (same ordinal, same meaning). See the header for why
// this implementation is not folded into this engine's own `text::StructuredCompiler`.

#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>

namespace ninfer::runtime {
namespace {
using Json = nlohmann::ordered_json;
const std::set<std::string> annotations{"title", "description", "default", "examples", "$comment",
                                       "$schema", "deprecated", "readOnly", "writeOnly"};

void check_schema(const Json& schema, const std::string& path, int depth = 0) {
    if (depth > 128) { throw std::invalid_argument(path + ": schema nesting exceeds 128"); }
    if (schema.is_boolean()) {
        if (!schema.get<bool>()) { throw std::invalid_argument(path + ": false schema is unsatisfiable"); }
        return;
    }
    if (!schema.is_object()) { throw std::invalid_argument(path + ": schema must be an object or true"); }
    if (schema.contains("type")) {
        const auto valid_type = [](const Json& value) {
            static const std::set<std::string> types{"object", "array", "string", "number", "integer", "boolean", "null"};
            return value.is_string() && types.contains(value.get<std::string>());
        };
        const auto& type = schema["type"];
        if (!(valid_type(type) || (type.is_array() && !type.empty() && std::all_of(type.begin(), type.end(), valid_type)))) {
            throw std::invalid_argument(path + ": type must be a recognized type or nonempty type array");
        }
    }
    if (schema.contains("required")) {
        const auto& required = schema["required"];
        if (!required.is_array()) { throw std::invalid_argument(path + ": required must be an array"); }
        for (const auto& name : required) {
            if (!name.is_string() || !schema.contains("properties") || !schema["properties"].contains(name.get<std::string>())) {
                throw std::invalid_argument(path + ": every required name must be declared in properties");
            }
        }
    }
    static const std::set<std::string> supported{
        "type", "properties", "required", "additionalProperties", "items", "prefixItems",
        "minItems", "maxItems", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "enum", "const", "anyOf", "$ref", "$defs", "definitions"};
    for (auto it = schema.begin(); it != schema.end(); ++it) {
        const auto& key = it.key();
        if (!supported.contains(key) && !annotations.contains(key)) {
            throw std::invalid_argument(path + "/" + key + ": unsupported JSON Schema keyword");
        }
        if (key == "properties" || key == "$defs" || key == "definitions") {
            if (!it->is_object()) { throw std::invalid_argument(path + "/" + key + ": expected object"); }
            for (auto child = it->begin(); child != it->end(); ++child) {
                check_schema(child.value(), path + "/" + key + "/" + child.key(), depth + 1);
            }
        } else if (key == "items" || key == "additionalProperties") {
            if (!it->is_boolean()) { check_schema(*it, path + "/" + key, depth + 1); }
        } else if (key == "anyOf" || key == "prefixItems") {
            if (!it->is_array() || (key == "anyOf" && it->empty())) {
                throw std::invalid_argument(path + "/" + key + ": expected nonempty schema array");
            }
            for (const auto& child : *it) { check_schema(child, path + "/" + key, depth + 1); }
        }
    }
    // The native compiler prioritizes these alternatives; reject intersecting sibling assertions
    // instead of silently compiling only one side of an intersection.
    for (const auto& exclusive : {"$ref", "anyOf", "enum", "const"}) {
        if (!schema.contains(exclusive)) { continue; }
        for (auto it = schema.begin(); it != schema.end(); ++it) {
            if (it.key() == exclusive || annotations.contains(it.key()) || it.key() == "$defs" ||
                it.key() == "definitions") { continue; }
            if ((std::string_view(exclusive) == "enum" || std::string_view(exclusive) == "const") &&
                it.key() == "type") { continue; }
            throw std::invalid_argument(path + ": " + exclusive + " with sibling assertions is unsupported");
        }
    }
    if (schema.contains("$ref")) {
        if (!schema["$ref"].is_string()) { throw std::invalid_argument(path + ": $ref must be a string"); }
        const auto ref = schema["$ref"].get<std::string>();
        if (ref != "#" && (!ref.starts_with("#/") || ref.find_first_of("~%") != std::string::npos ||
            ref.find("//") != std::string::npos || ref.ends_with('/'))) {
            throw std::invalid_argument(path + ": only unescaped document-local $ref paths are supported");
        }
    }
    // XGrammar's type inference does not implement the JSON Schema applicability rules.
    // Requiring explicit types avoids silently dropping assertions on an unconstrained node.
    if (!schema.contains("type")) {
        for (const auto& key : {"properties", "required", "additionalProperties", "items", "prefixItems",
                                "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
            if (schema.contains(key)) { throw std::invalid_argument(path + "/" + key + ": explicit type is required"); }
        }
    }
    if (schema.contains("type") && (schema.contains("enum") || schema.contains("const"))) {
        const auto conforms = [&](const Json& value) {
            const auto matches = [&](const Json& t) {
                if (!t.is_string()) { return false; }
                const auto name = t.get<std::string>();
                return (name == "string" && value.is_string()) || (name == "object" && value.is_object()) ||
                       (name == "array" && value.is_array()) || (name == "boolean" && value.is_boolean()) ||
                       (name == "null" && value.is_null()) || (name == "number" && value.is_number()) ||
                       (name == "integer" && value.is_number_integer());
            };
            const auto& type = schema["type"];
            return type.is_array() ? std::any_of(type.begin(), type.end(), matches) : matches(type);
        };
        if (schema.contains("const") && !conforms(schema["const"])) {
            throw std::invalid_argument(path + ": const conflicts with type");
        }
        if (schema.contains("enum")) {
            if (!schema["enum"].is_array() || schema["enum"].empty() ||
                !std::all_of(schema["enum"].begin(), schema["enum"].end(), conforms)) {
                throw std::invalid_argument(path + ": enum conflicts with type or is empty");
            }
        }
    }
}
} // namespace

void validate_structured_output(const StructuredOutputOptions& options) {
    if (options.kind == StructuredOutputKind::None || options.kind == StructuredOutputKind::JsonObject) {
        if (!options.schema.empty()) { throw std::invalid_argument("schema is only valid for JsonSchema output"); }
        return;
    }
    if (options.kind != StructuredOutputKind::JsonSchema) { throw std::invalid_argument("invalid structured output kind"); }
    try {
        const auto root = Json::parse(options.schema);
        check_schema(root, "#");
        std::set<const Json*> visited;
        std::function<void(const Json&)> check_refs = [&](const Json& node) {
            if (!node.is_object() || !visited.insert(&node).second) { return; }
            if (node.contains("$ref")) {
                const auto ref = node["$ref"].get<std::string>();
                const auto& target = root.at(Json::json_pointer(ref.substr(1)));
                check_schema(target, ref);
                check_refs(target);
            }
            for (const auto& map : {"properties", "$defs", "definitions"}) {
                if (node.contains(map)) { for (const auto& child : node[map]) { check_refs(child); } }
            }
            for (const auto& child : {"items", "additionalProperties"}) {
                if (node.contains(child)) { check_refs(node[child]); }
            }
            for (const auto& array : {"prefixItems", "anyOf"}) {
                if (node.contains(array)) { for (const auto& child : node[array]) { check_refs(child); } }
            }
        };
        check_refs(root);
    }
    catch (const nlohmann::json::exception& error) { throw std::invalid_argument(error.what()); }
}

class CompiledOutputConstraint::Impl {
public:
    xgrammar::CompiledGrammar grammar;
    int end_thinking;
    Impl(xgrammar::CompiledGrammar value, int end) : grammar(std::move(value)), end_thinking(end) {}
};
CompiledOutputConstraint::CompiledOutputConstraint(std::shared_ptr<const Impl> value) : impl(std::move(value)) {}

class OutputConstraintCompiler::Impl {
public:
    xgrammar::GrammarCompiler compiler;
    int end_thinking;
    std::mutex mutex;
    Impl(const std::vector<std::string>& vocab, const std::vector<int>& stops, int end)
        : compiler(xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW,
                                          static_cast<int>(vocab.size()), stops), 8, true, 64LL << 20),
          end_thinking(end) {}
};
OutputConstraintCompiler::OutputConstraintCompiler(std::vector<std::string> vocab, std::vector<int> stops, int end)
    : impl_(std::make_unique<Impl>(vocab, stops, end)) {}
OutputConstraintCompiler::~OutputConstraintCompiler() = default;

std::shared_ptr<const CompiledOutputConstraint>
OutputConstraintCompiler::compile(const StructuredOutputOptions& options) {
    validate_structured_output(options);
    if (options.kind == StructuredOutputKind::None) { return nullptr; }
    const std::string schema = options.kind == StructuredOutputKind::JsonObject
        ? R"({"type":"object","additionalProperties":true})" : options.schema;
    std::lock_guard lock(impl_->mutex);
    try {
        const auto whitespace = xgrammar::Grammar::FromEBNF(R"(root ::= [ \t\r\n]*)");
        const auto value = xgrammar::Grammar::FromJSONSchema(schema, true, std::nullopt, std::nullopt, false);
        // JSON permits outer whitespace. The frontend's forced reasoning-close sequence also
        // ends with two newlines, which must advance the same matcher as ordinary answer tokens.
        auto grammar = impl_->compiler.CompileGrammar(xgrammar::Grammar::Concat({whitespace, value, whitespace}));
        return std::make_shared<const CompiledOutputConstraint>(
            std::make_shared<const CompiledOutputConstraint::Impl>(std::move(grammar), impl_->end_thinking));
    } catch (const std::exception& error) { throw std::invalid_argument(std::string("unsupported or invalid JSON Schema: ") + error.what()); }
}

class OutputConstraintState::Impl {
public:
    std::shared_ptr<const CompiledOutputConstraint> compiled;
    xgrammar::GrammarMatcher matcher;
    bool reasoning;
    std::vector<std::int32_t> mask;
    Impl(std::shared_ptr<const CompiledOutputConstraint> value, bool thinking)
        : compiled(std::move(value)), matcher(compiled->impl->grammar, std::nullopt, false, 8),
          reasoning(thinking), mask((compiled->impl->grammar.GetTokenizerInfo().GetVocabSize() + 31) / 32) {}
};
OutputConstraintState::OutputConstraintState(std::shared_ptr<const CompiledOutputConstraint> value, bool reasoning)
    : impl_(std::make_unique<Impl>(std::move(value), reasoning)) {}
OutputConstraintState::OutputConstraintState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
OutputConstraintState::~OutputConstraintState() = default;
OutputConstraintState::OutputConstraintState(OutputConstraintState&&) noexcept = default;
OutputConstraintState& OutputConstraintState::operator=(OutputConstraintState&&) noexcept = default;

std::span<const std::int32_t> OutputConstraintState::next_mask() {
    if (impl_->reasoning) {
        std::fill(impl_->mask.begin(), impl_->mask.end(), -1);
        for (int id : impl_->matcher.GetStopTokenIds()) {
            impl_->mask[id / 32] &= static_cast<std::int32_t>(~(std::uint32_t{1} << (id % 32)));
        }
    } else {
        std::int64_t size = static_cast<std::int64_t>(impl_->mask.size());
        DLTensor tensor{impl_->mask.data(), DLDevice{kDLCPU, 0}, 1, DLDataType{kDLInt, 32, 1}, &size, nullptr, 0};
        impl_->matcher.FillNextTokenBitmask(&tensor);
    }
    if (std::all_of(impl_->mask.begin(), impl_->mask.end(), [](auto word) { return word == 0; })) {
        throw std::runtime_error("structured output has no valid next token");
    }
    return impl_->mask;
}
bool OutputConstraintState::try_accept(TokenId token) {
    if (impl_->reasoning) {
        if (static_cast<int>(token) == impl_->compiled->impl->end_thinking) { impl_->reasoning = false; }
        return std::find(impl_->matcher.GetStopTokenIds().begin(), impl_->matcher.GetStopTokenIds().end(), token) == impl_->matcher.GetStopTokenIds().end();
    }
    return impl_->matcher.AcceptToken(static_cast<int>(token));
}
void OutputConstraintState::accept(std::span<const TokenId> tokens) {
    for (auto token : tokens) {
        if (!try_accept(token)) { throw std::logic_error("sampled token violated structured output mask"); }
    }
}
OutputConstraintState OutputConstraintState::fork() const {
    auto copy = std::make_unique<Impl>(impl_->compiled, impl_->reasoning);
    copy->matcher = impl_->matcher.Fork();
    return OutputConstraintState(std::move(copy));
}
bool OutputConstraintState::terminated() const { return impl_->matcher.IsTerminated(); }
} // namespace ninfer::runtime
