#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables
ONNX_VERSION="1.15.1"
DEPS_DIR="deps"
FILE_NAME="onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${FILE_NAME}"

# Create deps directory if it doesn't exist
mkdir -p "${DEPS_DIR}"

# Check if the extracted directory already exists
if [ -d "${DEPS_DIR}/onnxruntime-linux-x64-${ONNX_VERSION}" ]; then
    echo "ONNX Runtime ${ONNX_VERSION} already exists in ${DEPS_DIR}. Skipping download."
else
    echo "Downloading ONNX Runtime ${ONNX_VERSION} for Linux x64..."
    # Download the archive
    curl -L -o "${DEPS_DIR}/${FILE_NAME}" "${DOWNLOAD_URL}"

    echo "Extracting ONNX Runtime..."
    # Extract the archive
    tar -xzf "${DEPS_DIR}/${FILE_NAME}" -C "${DEPS_DIR}"

    echo "Cleaning up archive file..."
    # Remove the .tgz file
    rm "${DEPS_DIR}/${FILE_NAME}"

    echo "ONNX Runtime downloaded and extracted successfully."
fi