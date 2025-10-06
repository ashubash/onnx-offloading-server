# cpp-server/Dockerfile

# Step 1: Use an official Ubuntu image as a parent image
FROM ubuntu:22.04

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install build-time dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the dependency download script and make it executable
COPY scripts/download_deps.sh .
RUN chmod +x download_deps.sh

# Step 5: Copy the rest of the application code into the container
COPY . .

# Step 6: Run the script to download ONNX Runtime
RUN ./download_deps.sh

# Step 7: Build the application using CMake
RUN mkdir build && \
    cd build && \
    cmake .. -DCMAKE_PREFIX_PATH=/app/deps && \
    make -j$(nproc)

# Step 8: Expose the port the app runs on
EXPOSE 8080

# Step 9: Command to run the executable when the container starts
CMD ["./build/onnx_offloading_server"]