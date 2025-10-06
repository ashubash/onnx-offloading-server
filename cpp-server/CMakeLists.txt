cmake_minimum_required(VERSION 3.16)
project(ONNXCppServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find ONNX Runtime ---
# Set the path to the downloaded dependency
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps")

# Determine the correct directory name based on the OS
if(WIN32)
    set(ONNXRUNTIME_PATH "${ONNXRUNTIME_ROOT}/onnxruntime-win-x64-1.15.1")
elseif(APPLE)
    set(ONNXRUNTIME_PATH "${ONNXRUNTIME_ROOT}/onnxruntime-osx-x64-1.15.1")
else() # Assuming Linux
    set(ONNXRUNTIME_PATH "${ONNXRUNTIME_ROOT}/onnxruntime-linux-x64-1.15.1")
endif()

# Check if the directory exists
if(NOT EXISTS ${ONNXRUNTIME_PATH})
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_PATH}. Please run the download script first.")
endif()

# Add the library and include directories
add_library(onnxruntime STATIC IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_PATH}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_PATH}/include"
)

# Add executable
add_executable(onnx_cpp_server src/main.cpp)

# Include directories
target_include_directories(onnx_cpp_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(onnx_cpp_server PRIVATE
    onnxruntime
    pthread
)

# Copy model to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/models/student_model_verified.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/models/student_model_verified.onnx
    COPYONLY
)