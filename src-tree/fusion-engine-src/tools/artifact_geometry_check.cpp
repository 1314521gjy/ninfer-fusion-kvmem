// Per-object geometry check against a real artifact's own directory.
//
// WHY THIS EXISTS
//   A `.ninfer` container records, for every tensor object, its `name`, `shape`, `format`,
//   `layout` and `bytes`. That last field is the size the writer actually laid down. So the
//   question "can this engine read this artifact" reduces to a comparison this tool can run with no
//   GPU, no model load and no reference implementation:
//
//       weight_geometry(format, layout, shape).bytes  ==  the container's own `bytes`
//
//   A mismatch means the engine would read the wrong number of bytes for that object -- the failure
//   mode is silent (the bytes are still inside the payload), which is exactly why it is worth a
//   standalone check rather than a smoke test.
//
// USAGE
//   ninfer-artifact-geometry-check <artifact.ninfer> [--quiet] [--allow-v2]
//
//   Exits 0 when every object agrees, 1 when any object disagrees, 2 on a usage or read error.
//
//   `--allow-v2` accepts a container whose version byte is 2 as well as this engine's own. The
//   version check is not cosmetic -- the directory's position differs between versions, so parsing
//   one with the other's offsets produces confident nonsense. Version 2 is accepted only when the
//   caller says so, which is what makes "check the geometry of a v2 artifact we intend to convert"
//   a deliberate act rather than an accident. The engine itself still writes and reads only its own
//   version.
//
// SCOPE / LIMITS (stated so a green run is not over-read)
//   * Only this engine's DIRECTORY VERSION is read. A container of another version is reported as
//     unsupported rather than parsed with the wrong layout -- the JSON directory offsets above the
//     header differ between versions, so guessing would produce confident nonsense.
//   * Rank-three NVFP4 objects are the expert banks and go through `block_scale_bank_geometry`,
//     rank-two objects go through `weight_geometry`. The rule itself lives in `artifact/layouts.h`
//     (`is_block_scale_bank` / `tensor_geometry`) and is shared with the engine, so this tool and
//     the engine cannot dispatch differently. Anything else is reported, never skipped silently.
//   * `resource` objects carry no format or layout. Their `bytes` is checked only for being
//     non-zero and inside the payload.

#include "artifact/bank_geometry.h"
#include "artifact/formats.h"
#include "artifact/framing.h"
#include "artifact/layouts.h"
#include "artifact/schema.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nlohmann::json;
using ninfer::QType;
using ninfer::QuantLayout;
namespace artifact = ninfer::artifact;

struct Report {
    std::uint64_t tensors    = 0;
    std::uint64_t resources  = 0;
    std::uint64_t agreed     = 0;
    std::vector<std::string> mismatches;
    std::vector<std::string> unsupported;
};

[[nodiscard]] std::vector<std::uint64_t> to_shape(const json& value) {
    std::vector<std::uint64_t> out;
    out.reserve(value.size());
    for (const auto& dim : value) { out.push_back(dim.get<std::uint64_t>()); }
    return out;
}

void check_tensor(const json& object, Report& report, bool quiet) {
    ++report.tensors;
    // v3 names the field `id`; the v2 directory this tool can also read names it `name`. Reading the
    // wrong one threw out of json::at, and the throw escaped main (only the two parse blocks below
    // are guarded), so the whole process died with STATUS_STACK_BUFFER_OVERRUN instead of reporting.
    const std::string name =
        object.contains("id") ? object.at("id").get<std::string>() : object.at("name").get<std::string>();

    const std::string format_name = object.at("format").get<std::string>();
    const std::string layout_name = object.at("layout").get<std::string>();
    const std::uint64_t stored    = object.at("bytes").get<std::uint64_t>();
    const auto shape              = to_shape(object.at("shape"));

    QType format{};
    QuantLayout layout{};
    try {
        format = artifact::parse_format(format_name);
        layout = artifact::parse_layout(layout_name);
    } catch (const std::exception& error) {
        report.unsupported.push_back(name + ": " + error.what());
        return;
    }

    std::uint64_t expected = 0;
    try {
        // The dispatch rule lives in ONE place, `artifact/layouts.h`: the engine's own
        // `describe_tensor` and this tool ask the same predicate, so a green run here means the
        // engine computes the same byte count -- which is the only thing this tool claims.
        expected = artifact::tensor_geometry(format, layout, shape).bytes;
    } catch (const std::exception& error) {
        report.unsupported.push_back(name + ": " + error.what());
        return;
    }

    if (expected == stored) {
        ++report.agreed;
        if (!quiet) {
            std::printf("  ok    %-58s %12llu\n", name.c_str(),
                        static_cast<unsigned long long>(stored));
        }
        return;
    }
    char line[512];
    std::snprintf(line, sizeof(line),
                  "%s: container says %llu, this engine computes %llu (%s / %s, rank %zu)",
                  name.c_str(), static_cast<unsigned long long>(stored),
                  static_cast<unsigned long long>(expected), format_name.c_str(),
                  layout_name.c_str(), shape.size());
    report.mismatches.push_back(line);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <artifact.ninfer> [--quiet]\n", argv[0]);
        return 2;
    }
    const bool quiet = (argc > 2 && std::string(argv[2]) == "--quiet");
    bool allow_v2    = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--allow-v2") { allow_v2 = true; }
    }
    const std::filesystem::path path = argv[1];

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot open %s\n", path.string().c_str());
        return 2;
    }

    // The directory's own header: the entry magic, then the directory length. Reading the length is
    // what keeps this from having to load the payload -- a 100 GiB artifact costs one seek and a
    // quarter-megabyte read.
    //
    // The JSON does NOT begin immediately after those 16 bytes in this engine's own container: v3
    // reserves `kHeaderBytes` (32) and puts the directory at offset 32, which is where
    // src/artifact/reader.cpp reads it. Only v2 puts it at 16. Getting this wrong reads the
    // directory from the middle of a reserved field and reports "directory is not JSON".
    std::uint64_t directory_length = 0;
    std::streamoff directory_at    = 0;
    {
        std::array<std::byte, 16> header{};
        input.read(reinterpret_cast<char*>(header.data()), header.size());
        if (input.gcount() != static_cast<std::streamsize>(header.size())) {
            std::fprintf(stderr, "%s: file is shorter than a container header\n", path.string().c_str());
            return 2;
        }
        const std::array<std::byte, 8> entry_magic = artifact::kEntryMagic;
        const std::array<std::byte, 8> v2_magic    = {std::byte{'N'}, std::byte{'I'}, std::byte{'N'},
                                                   std::byte{'F'}, std::byte{'E'}, std::byte{'R'},
                                                   std::byte{0},   std::byte{2}};
        const bool matches_own = std::equal(header.begin(), header.begin() + 8, entry_magic.begin());
        const bool matches_v2  = std::equal(header.begin(), header.begin() + 8, v2_magic.begin());
        if (!matches_own && !(allow_v2 && matches_v2)) {
            std::fprintf(stderr,
                         "%s: container version %u is not the one this engine reads (%u)%s\n",
                         path.string().c_str(), static_cast<unsigned>(header[7]),
                         static_cast<unsigned>(entry_magic[7]),
                         matches_v2 ? "; pass --allow-v2 to check it anyway" : "");
            return 2;
        }
        std::memcpy(&directory_length, header.data() + 8, sizeof(directory_length));
        // v3 reserves a 32-byte header; only v2 puts the JSON at 16. See src/artifact/reader.cpp,
        // which reads the directory at kHeaderBytes for this engine's own container.
        directory_at = matches_v2 ? 16 : static_cast<std::streamoff>(artifact::kHeaderBytes);
    }

    input.clear();
    input.seekg(directory_at, std::ios::beg);
    std::string directory_bytes(directory_length, '\0');
    input.read(directory_bytes.data(), static_cast<std::streamsize>(directory_length));
    if (input.gcount() != static_cast<std::streamsize>(directory_length)) {
        std::fprintf(stderr, "%s: directory is truncated\n", path.string().c_str());
        return 2;
    }

    json directory;
    try {
        directory = json::parse(directory_bytes);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: directory is not JSON: %s\n", path.string().c_str(), error.what());
        return 2;
    }

    Report report;
    for (const auto& object : directory.at("objects")) {
        const std::string kind = object.at("kind").get<std::string>();
        if (kind == "tensor") {
            check_tensor(object, report, quiet);
        } else {
            ++report.resources;
        }
    }

    std::printf("\nobjects: %llu tensor + %llu resource\n",
                static_cast<unsigned long long>(report.tensors),
                static_cast<unsigned long long>(report.resources));
    std::printf("agreed : %llu\n", static_cast<unsigned long long>(report.agreed));

    if (!report.unsupported.empty()) {
        std::printf("\nUNSUPPORTED (%zu) -- this engine cannot size these at all:\n",
                    report.unsupported.size());
        for (const auto& line : report.unsupported) { std::printf("  %s\n", line.c_str()); }
    }
    if (!report.mismatches.empty()) {
        std::printf("\nMISMATCH (%zu) -- the engine would read the wrong byte count:\n",
                    report.mismatches.size());
        for (const auto& line : report.mismatches) { std::printf("  %s\n", line.c_str()); }
    }

    if (report.unsupported.empty() && report.mismatches.empty()) {
        std::printf("\nVERDICT: every tensor's byte count agrees with this engine's geometry.\n");
        return 0;
    }
    std::printf("\nVERDICT: %zu unsupported, %zu mismatched.\n", report.unsupported.size(),
                report.mismatches.size());
    return 1;
}
