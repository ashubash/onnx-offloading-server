// cpp-server/include/npy_parser.hpp
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>
#include <onnxruntime_cxx_api.h>

// A simple C++ function to parse a .npy file in memory
// Returns an ONNX Runtime Tensor
inline Ort::Value parseNpy(const std::vector<uint8_t>& buffer) {
    if (buffer.size() < 10) {
        throw std::runtime_error("File too small to be a .npy file");
    }

    // Check magic string
    if (buffer[0] != 0x93 || buffer[1] != 0x4E || buffer[2] != 0x55 || buffer[3] != 0x4D || buffer[4] != 0x50 || buffer[5] != 0x59) {
        throw std::runtime_error("Invalid .npy magic bytes");
    }

    // Check version
    if (buffer[6] != 1 || buffer[7] != 0) {
        throw std::runtime_error("Unsupported .npy version");
    }

    // Get header length
    uint16_t header_len = *reinterpret_cast<const uint16_t*>(&buffer[8]);
    size_t header_start = 10;
    size_t header_end = header_start + header_len;

    if (header_end >= buffer.size()) {
        throw std::runtime_error("Header exceeds file size");
    }

    // Parse header
    std::string header_str(buffer.begin() + header_start, buffer.begin() + header_end - 1); // -1 to exclude \n
    std::istringstream header_ss(header_str);
    std::string item;

    std::vector<int64_t> shape;
    std::string descr;

    while (std::getline(header_ss, item, ' ')) {
        if (item.find("'descr':") != std::string::npos) {
            descr = item.substr(item.find("'") + 1, item.rfind("'") - item.find("'") - 1);
        }
        if (item.find("'shape':") != std::string::npos) {
            std::string shape_str = item.substr(item.find("(") + 1, item.find(")") - item.find("(") - 1);
            std::istringstream shape_ss(shape_str);
            std::string s;
            while (std::getline(shape_ss, s, ',')) {
                if (!s.empty()) {
                    shape.push_back(std::stoll(s));
                }
            }
        }
    }

    if (descr != "<f4" || shape.size() != 2 || shape[0] != 256 || shape[1] != 56) {
        throw std::runtime_error("Unexpected .npy format or shape");
    }

    // Calculate data start
    size_t data_start = header_end;
    size_t pad = (64 - (data_start % 64)) % 64;
    data_start += pad;

    // Create tensor
    std::vector<float> data;
    size_t num_elements = shape[0] * shape[1];
    data.resize(num_elements);
    std::memcpy(data.data(), buffer.data() + data_start, num_elements * sizeof(float));

    std::vector<int64_t> dims = {1, shape[0], shape[1]};
    return Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
        data.data(), data.size(),
        dims.data(), dims.size()
    );
}