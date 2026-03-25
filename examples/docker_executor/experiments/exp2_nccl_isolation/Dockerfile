FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support (CUDA 12.4 compatible)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    numpy \
    --index-url https://download.pytorch.org/whl/cu124

# Set working directory
WORKDIR /workspace

# Copy benchmark scripts
COPY benchmark.py /workspace/
COPY benchmark_docker.py /workspace/

# Set environment
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "benchmark_docker.py"]
