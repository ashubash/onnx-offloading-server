// cpp-server/src/main.cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

// HTTP Server
#include "httplib.h"
// JSON Parsing
#include "nlohmann/json.hpp"
// ONNX Runtime
#include "onnxruntime_cxx_api.h"
// NPY Parser
#include "npy_parser.hpp"
// libcurl for downloading files
#include <curl/curl.h>

// Use the nlohmann namespace for convenience
using json = nlohmann::json;

// --- Global ONNX Runtime Objects ---
std::unique_ptr<Ort::Env> ort_env;
std::unique_ptr<Ort::Session> ort_session;
std::vector<const char*> input_names;
std::vector<const char*> output_names;

// --- JSON Request/Response Structures ---
// --- JSON Request/Response Structures ---
struct InferenceRequest {
    std::string npy_url;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(InferenceRequest, npy_url)
};

struct InferenceResponse {
    std::vector<float> output;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(InferenceResponse, output)
};

// --- Function to Initialize ONNX Runtime ---
void initialize_onnx() {
    std::cout << "Initializing ONNX Runtime..." << std::endl;
    ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXCppServer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    std::string model_path = "./models/student_model_verified.onnx";
    ort_session = std::make_unique<Ort::Session>(*ort_env, model_path.c_str(), session_options);
    std::cout << "Model loaded successfully from: " << model_path << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;

    // --- Use the new API to get input/output names ---
    std::vector<std::string> input_names_vec;
    std::vector<std::string> output_names_vec;

    // Get input names
    Ort::AllocatedStringPtr input_name_ptr = ort_session->GetInputNameAllocated(0, allocator);
    input_names_vec.push_back(input_name_ptr.get());

    // Get output names
    Ort::AllocatedStringPtr output_name_ptr = ort_session->GetOutputNameAllocated(0, allocator);
    output_names_vec.push_back(output_name_ptr.get());

    // --- Convert std::strings to C-style char* for the run call ---
    // We need to store the string data to keep it alive
    static std::vector<std::string> input_names_storage = std::move(input_names_vec);
    static std::vector<std::string> output_names_storage = std::move(output_names_vec);

    input_names.clear();
    output_names.clear();
    input_names.reserve(input_names_storage.size());
    output_names.reserve(output_names_storage.size());

    for (const auto& name : input_names_storage) {
        input_names.push_back(name.c_str());
    }
    for (const auto& name : output_names_storage) {
        output_names.push_back(name.c_str());
    }
}

// --- Callback function for libcurl to write data into a buffer ---
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::vector<uint8_t>*)userp)->insert(((std::vector<uint8_t>*)userp)->end(), (uint8_t*)contents, (uint8_t*)contents + size * nmemb);
    return size * nmemb;
}

// --- Function to download a file into a buffer ---
std::vector<uint8_t> download_file(const std::string& url) {
    CURL *curl;
    CURLcode res;
    std::vector<uint8_t> buffer;

    curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize curl");
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::string err = "curl_easy_perform() failed: " + std::string(curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        throw std::runtime_error(err);
    }

    curl_easy_cleanup(curl);
    return buffer;
}

// --- New Inference Handler that fetches and parses .npy ---
InferenceResponse run_inference_from_npy(const std::string& npy_url) {
    if (!ort_session) {
        throw std::runtime_error("ONNX session is not initialized.");
    }

    // 1. Download the .npy file from the provided URL
    std::cout << "Downloading file from: " << npy_url << std::endl;
    std::vector<uint8_t> npy_buffer = download_file(npy_url);

    // 2. Parse the .npy file in memory into a tensor
    Ort::Value input_tensor = parseNpy(npy_buffer);

    // 3. Run inference
    auto output_tensors = ort_session->Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), output_names.size());

    // 4. Process output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = 1;
    for (auto dim : output_shape) output_size *= dim;

    InferenceResponse response;
    response.output.assign(output_data, output_data + output_size);

    return response;
}

// --- Main Function ---
int main() {
    try {
        // Initialize libcurl globally
        curl_global_init(CURL_GLOBAL_DEFAULT);

        // Initialize ONNX Runtime
        initialize_onnx();

        // Create HTTP server
        httplib::Server server;

        // Health check endpoint
        server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status", "ok"}}.dump(), "application/json");
        });

        // NEW: Inference endpoint that takes an NPY URL
        server.Post("/inference_from_npy", [](const httplib::Request& req, httplib::Response& res) {
            try {
                auto json_req = json::parse(req.body);
                std::string npy_url = json_req["npy_url"];
                if (npy_url.empty()) {
                    throw std::runtime_error("npy_url is required.");
                }

                auto inference_res = run_inference_from_npy(npy_url);

                json json_res = inference_res;
                res.set_content(json_res.dump(), "application/json");

            } catch (const std::exception& e) {
                std::cerr << "Error during inference: " << e.what() << std::endl;
                res.status = 500;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        });

        std::cout << "Server starting on http://0.0.0.0:8080" << std::endl;
        server.listen("0.0.0.0", 8080);

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }

    curl_global_cleanup();
    return 0;
}