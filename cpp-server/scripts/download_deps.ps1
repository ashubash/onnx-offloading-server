# Define variables
 $ONNX_VERSION = "1.15.1"
 $DEPS_DIR = "..\deps"
 $FILE_NAME = "onnxruntime-win-x64-${ONNX_VERSION}.zip"
 $DOWNLOAD_URL = "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${FILE_NAME}"

# Create deps directory if it doesn't exist
if (-not (Test-Path -Path $DEPS_DIR)) {
    New-Item -ItemType Directory -Path $DEPS_DIR
}

# Check if the extracted directory already exists
 $extractedDir = Join-Path -Path $DEPS_DIR -ChildPath "onnxruntime-win-x64-${ONNX_VERSION}"
if (Test-Path -Path $extractedDir) {
    Write-Host "ONNX Runtime ${ONNX_VERSION} already exists in ${DEPS_DIR}. Skipping download."
}
else {
    Write-Host "Downloading ONNX Runtime ${ONNX_VERSION} for Windows x64..."
    # Download the archive
    Invoke-WebRequest -Uri $DOWNLOAD_URL -OutFile (Join-Path -Path $DEPS_DIR -ChildPath $FILE_NAME)

    Write-Host "Extracting ONNX Runtime..."
    # Extract the archive
    Expand-Archive -Path (Join-Path -Path $DEPS_DIR -ChildPath $FILE_NAME) -DestinationPath $DEPS_DIR

    Write-Host "Cleaning up archive file..."
    # Remove the .zip file
    Remove-Item -Path (Join-Path -Path $DEPS_DIR -ChildPath $FILE_NAME)

    Write-Host "ONNX Runtime downloaded and extracted successfully."
}