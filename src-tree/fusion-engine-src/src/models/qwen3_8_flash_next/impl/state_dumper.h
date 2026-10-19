#pragma once

#include "models/qwen3_8_flash_next/impl/text_decode.h"

#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {
inline const char* dtype_to_string(DType dt) {
    switch (dt) {
        case DType::BF16: return "BF16";
        case DType::FP32: return "FP32";
        case DType::I32:  return "I32";
        case DType::I64:  return "I64";
        case DType::FP16: return "FP16";
        case DType::U8:   return "U8";
        default:          return "UNKNOWN";
    }
}

struct TensorDumpRecord {
    std::string name;
    std::string dtype;
    std::vector<std::int32_t> shape;
    std::string file;
    std::size_t bytes = 0;
};

struct PositionDumpRecord {
    std::int32_t position = 0;
    std::int32_t token_id = 0;
    std::array<std::int32_t, 3> mrope_position{0, 0, 0};
    std::vector<TensorDumpRecord> tensors;
};

class StateDumper {
public:
    explicit StateDumper(std::string dump_dir) : dump_dir_(std::move(dump_dir)) {
        std::filesystem::create_directories(dump_dir_);
    }

    FlashNextDecodeStateSink make_sink(std::int32_t position, std::int32_t token_id,
                                       std::array<std::int32_t, 3> mrope_pos) {
        char pos_buf[32];
        std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d", position);
        const std::string pos_str = pos_buf;
        const std::filesystem::path pos_dir = std::filesystem::path(dump_dir_) / pos_str;
        std::filesystem::create_directories(pos_dir);

        const std::size_t pos_idx = positions_.size();
        auto& pos_rec = positions_.emplace_back();
        pos_rec.position = position;
        pos_rec.token_id = token_id;
        pos_rec.mrope_position = mrope_pos;

        return FlashNextDecodeStateSink{
            .on_state = [this, pos_idx, pos_str, pos_dir](std::string_view name,
                                                          const Tensor& tensor) {
                const std::string filename = std::string(name) + ".bin";
                const std::filesystem::path file_path = pos_dir / filename;

                std::vector<std::uint8_t> host_bytes(tensor.bytes());
                cudaDeviceSynchronize();
                CUDA_CHECK(cudaMemcpy(host_bytes.data(), tensor.data, tensor.bytes(),
                                      cudaMemcpyDeviceToHost));

                std::ofstream ofs(file_path, std::ios::binary);
                if (!ofs) {
                    throw std::runtime_error("Failed to open file for writing: " +
                                             file_path.string());
                }
                ofs.write(reinterpret_cast<const char*>(host_bytes.data()), host_bytes.size());

                TensorDumpRecord rec;
                rec.name  = std::string(name);
                rec.dtype = dtype_to_string(tensor.dtype);
                if (tensor.ne[3] > 1) {
                    rec.shape = {tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3]};
                } else if (tensor.ne[2] > 1) {
                    rec.shape = {tensor.ne[0], tensor.ne[1], tensor.ne[2]};
                } else if (tensor.ne[1] > 1 || tensor.ne[0] > 1) {
                    rec.shape = {tensor.ne[0], tensor.ne[1]};
                } else {
                    rec.shape = {tensor.ne[0]};
                }
                rec.file  = pos_str + "/" + filename;
                rec.bytes = tensor.bytes();
                positions_[pos_idx].tensors.push_back(std::move(rec));
            },
        };
    }

    FlashNextDecodeStateSink
    make_chunk_sink(std::int32_t first_position, std::span<const std::int32_t> token_ids,
                    std::span<const std::array<std::int32_t, 3>> mrope_positions) {
        const std::size_t count     = token_ids.size();
        const std::size_t first_idx = positions_.size();
        for (std::size_t i = 0; i < count; ++i) {
            char pos_buf[32];
            std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d",
                          first_position + static_cast<std::int32_t>(i));
            const std::string pos_str           = pos_buf;
            const std::filesystem::path pos_dir = std::filesystem::path(dump_dir_) / pos_str;
            std::filesystem::create_directories(pos_dir);

            auto& pos_rec          = positions_.emplace_back();
            pos_rec.position       = first_position + static_cast<std::int32_t>(i);
            pos_rec.token_id       = token_ids[i];
            pos_rec.mrope_position = mrope_positions[i];
        }

        return FlashNextDecodeStateSink{
            .on_state = [this, first_idx, count](std::string_view name, const Tensor& tensor) {
                std::vector<std::uint8_t> host_bytes(tensor.bytes());
                cudaDeviceSynchronize();
                CUDA_CHECK(cudaMemcpy(host_bytes.data(), tensor.data, tensor.bytes(),
                                      cudaMemcpyDeviceToHost));

                const std::string filename = std::string(name) + ".bin";
                if (tensor.ne[1] == static_cast<std::int32_t>(count)) {
                    const std::size_t col_bytes = tensor.bytes() / count;
                    for (std::size_t i = 0; i < count; ++i) {
                        auto& pos_rec = positions_[first_idx + i];
                        char pos_buf[32];
                        std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d", pos_rec.position);
                        const std::string pos_str = pos_buf;
                        const std::filesystem::path pos_dir =
                            std::filesystem::path(dump_dir_) / pos_str;
                        const std::filesystem::path file_path = pos_dir / filename;

                        std::ofstream ofs(file_path, std::ios::binary);
                        if (!ofs) {
                            throw std::runtime_error("Failed to open file for writing: " +
                                                     file_path.string());
                        }
                        ofs.write(
                            reinterpret_cast<const char*>(host_bytes.data() + i * col_bytes),
                            col_bytes);

                        TensorDumpRecord rec;
                        rec.name  = std::string(name);
                        rec.dtype = dtype_to_string(tensor.dtype);
                        rec.shape = {tensor.ne[0], 1};
                        rec.file  = pos_str + "/" + filename;
                        rec.bytes = col_bytes;
                        pos_rec.tensors.push_back(std::move(rec));
                    }
                } else {
                    // Single-token tensor (e.g. final_hidden, logits on last token)
                    const std::size_t last_idx = first_idx + count - 1;
                    auto& pos_rec              = positions_[last_idx];
                    char pos_buf[32];
                    std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d", pos_rec.position);
                    const std::string pos_str = pos_buf;
                    const std::filesystem::path pos_dir =
                        std::filesystem::path(dump_dir_) / pos_str;
                    const std::filesystem::path file_path = pos_dir / filename;

                    std::ofstream ofs(file_path, std::ios::binary);
                    if (!ofs) {
                        throw std::runtime_error("Failed to open file for writing: " +
                                                 file_path.string());
                    }
                    ofs.write(reinterpret_cast<const char*>(host_bytes.data()), host_bytes.size());

                    TensorDumpRecord rec;
                    rec.name  = std::string(name);
                    rec.dtype = dtype_to_string(tensor.dtype);
                    rec.shape = {tensor.ne[0], 1};
                    rec.file  = pos_str + "/" + filename;
                    rec.bytes = tensor.bytes();
                    pos_rec.tensors.push_back(std::move(rec));
                }
            },
        };
    }

    void write_manifest() const {
        const std::filesystem::path manifest_path =
            std::filesystem::path(dump_dir_) / "manifest.json";
        std::ofstream ofs(manifest_path);
        if (!ofs) {
            throw std::runtime_error("Failed to open manifest for writing: " +
                                     manifest_path.string());
        }
        ofs << "{\n  \"positions\": [\n";
        for (std::size_t p = 0; p < positions_.size(); ++p) {
            const auto& pos = positions_[p];
            ofs << "    {\n";
            ofs << "      \"position\": " << pos.position << ",\n";
            ofs << "      \"token_id\": " << pos.token_id << ",\n";
            ofs << "      \"mrope_position\": [" << pos.mrope_position[0] << ", "
                << pos.mrope_position[1] << ", " << pos.mrope_position[2] << "],\n";
            ofs << "      \"tensors\": [\n";
            for (std::size_t t = 0; t < pos.tensors.size(); ++t) {
                const auto& item = pos.tensors[t];
                ofs << "        {\n";
                ofs << "          \"name\": \"" << item.name << "\",\n";
                ofs << "          \"dtype\": \"" << item.dtype << "\",\n";
                ofs << "          \"shape\": [";
                for (std::size_t s = 0; s < item.shape.size(); ++s) {
                    ofs << item.shape[s] << (s + 1 < item.shape.size() ? ", " : "");
                }
                ofs << "],\n";
                ofs << "          \"file\": \"" << item.file << "\",\n";
                ofs << "          \"bytes\": " << item.bytes << "\n";
                ofs << "        }" << (t + 1 < pos.tensors.size() ? "," : "") << "\n";
            }
            ofs << "      ]\n";
            ofs << "    }" << (p + 1 < positions_.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n}\n";
    }

private:
    std::string dump_dir_;
    std::vector<PositionDumpRecord> positions_;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
