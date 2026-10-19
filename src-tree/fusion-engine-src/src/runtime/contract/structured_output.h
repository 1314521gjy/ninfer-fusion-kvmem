#pragma once

// Copied from upstream `src/runtime/contract/structured_output.h` (igorls/ninfer @ c81c88f6).
//
// The three classes and their member sets are upstream's, verbatim: the ported Flash-Next program
// calls `compile_output_constraint`, `OutputConstraintState::next_mask`, `accept`, `fork`,
// `try_accept` and `terminated` by these names, and `ProgramImpl` holds one of each.
//
// ONE ADAPTATION, and the reason for it
//   Upstream spells the no-constraint enumerator `StructuredOutputKind::Text`; this engine spells
//   the same value `StructuredOutputKind::None`. Same ordinal, same meaning, different name. There
//   is no `Text` enumerator here and none was added -- adding one would give one fact two spellings
//   in the engine-level type. The `.cpp` below maps `None` onto upstream's "no constraint" branch.
//
// WHY THE IMPLEMENTATION IS NOT REUSED FROM `text::StructuredCompiler`
//   This engine already compiles grammars in `src/text/structured_output.{h,cpp}`
//   (`text::StructuredCompiler` / `text::GrammarState`). Both wrap the same vendored xgrammar, but
//   they expose DIFFERENT mask contracts -- upstream hands out an `int32` bitmask via
//   `FillNextTokenBitmask` and drives the matcher directly, while `text::GrammarState` fills
//   `uint32` masks conditioned on drafts. Those are two packed-bit conventions, and guessing a
//   conversion between them would be exactly the sort of silent-wrong-bits change this port avoids.
//   So the implementation is upstream's, over the same xgrammar, and the two live side by side:
//   `ninfer::runtime::*` for the ported Flash-Next program, `ninfer::text::*` for this engine's own.

#include "ninfer/types.h"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::runtime {

// Immutable compiled vocabulary/schema; every active request creates its own matcher.
class CompiledOutputConstraint {
public:
    class Impl;
    explicit CompiledOutputConstraint(std::shared_ptr<const Impl> impl);
    std::shared_ptr<const Impl> impl;
};

// Throws invalid_argument for schema constraints that the compiler cannot enforce.
void validate_structured_output(const StructuredOutputOptions& options);

class OutputConstraintCompiler {
public:
    OutputConstraintCompiler(std::vector<std::string> decoded_vocab, std::vector<int> stop_tokens,
                             int end_thinking_token);
    ~OutputConstraintCompiler();
    std::shared_ptr<const CompiledOutputConstraint> compile(const StructuredOutputOptions& options);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class OutputConstraintState {
public:
    OutputConstraintState(std::shared_ptr<const CompiledOutputConstraint> compiled, bool reasoning);
    ~OutputConstraintState();
    OutputConstraintState(OutputConstraintState&&) noexcept;
    OutputConstraintState& operator=(OutputConstraintState&&) noexcept;
    [[nodiscard]] std::span<const std::int32_t> next_mask();
    void accept(std::span<const TokenId> tokens);
    [[nodiscard]] OutputConstraintState fork() const;
    [[nodiscard]] bool try_accept(TokenId token);
    [[nodiscard]] bool terminated() const;

private:
    class Impl;
    explicit OutputConstraintState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::runtime
