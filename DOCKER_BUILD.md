# DockerDistributedExecutor Build Instructions

This directory contains Docker-related files for the DockerDistributedExecutor implementation.

## Files

- `Dockerfile.docker-executor` - Builds vLLM from local source using uv (official recommended method)
- `build-docker-executor.sh` - Build script
- `DOCKER_BUILD.md` - This documentation

## Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- Local vLLM source code (this repository)
- Base image: `nvidia/cuda:12.1.0-devel-ubuntu22.04` (will be pulled if not available locally)

## Building the Image

### Quick Build

```bash
cd /home/thd/repositories/vllm-dev/vllm-source
./build-docker-executor.sh
```

This creates an image named `vllm/vllm-docker-executor:latest`.

### Custom Image Name

```bash
TARGET_IMAGE=myregistry/vllm-docker-executor:v1 ./build-docker-executor.sh
```

## Dockerfile Details

The Dockerfile follows the official vLLM recommended installation method:

1. **Base Image**: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
   - CUDA 12.1 development image with build tools

2. **uv Installation**: Uses `astral.sh/uv/install.sh` to install uv
   - uv is a fast Python package manager and environment manager

3. **Python Environment**:
   ```bash
   uv venv --python 3.12 --seed --managed-python
   ```
   - Creates an isolated Python 3.12 environment
   - `--seed` ensures pip is available
   - `--managed-python` uses uv's managed Python distribution

4. **vLLM Installation**:
   ```bash
   uv pip install --editable . --torch-backend=auto
   ```
   - Installs vLLM from local source in editable mode
   - `--torch-backend=auto` automatically selects appropriate PyTorch CUDA version
   - Changes to source code are reflected immediately without rebuild

## Usage

After building, use the custom image with vLLM:

```python
import os

# Set the custom image
os.environ['VLLM_DOCKER_IMAGE'] = 'vllm/vllm-docker-executor:latest'

from vllm import LLM

# Create LLM with docker executor backend
llm = LLM(
    model="meta-llama/Llama-2-7b",
    tensor_parallel_size=2,
    distributed_executor_backend="docker"
)
```

## Verifying the Build

### Check Image Contents

```bash
# List the image
docker images | grep vllm-docker-executor

# Verify vLLM is installed
docker run --rm vllm/vllm-docker-executor:latest python -c "import vllm; print(vllm.__version__)"

# Verify custom executor files
docker run --rm vllm/vllm-docker-executor:latest \
    python -c "from vllm.v1.executor.docker_worker_entrypoint import main; print('Entrypoint OK')"
```

## Troubleshooting

### Network Issues

If Docker Hub is unavailable and you see TLS handshake timeouts:

1. **Check if base image is available locally**:
   ```bash
   docker images | grep nvidia/cuda
   ```

2. **If the base image is not available**, you need to obtain it from an alternative source or wait for Docker Hub to be available.

3. **To use a different base image**, modify the `FROM` line in `Dockerfile.docker-executor`.

### Build Failures

If the build fails during vLLM compilation:

1. **Check CUDA version compatibility**:
   ```bash
   nvidia-smi
   ```

2. **Check available disk space**:
   ```bash
   docker system df
   ```

3. **Clean build cache and retry**:
   ```bash
   docker buildx prune -f
   ./build-docker-executor.sh
   ```

### Development Workflow

Since the Dockerfile uses `--editable` install, you can mount your source code as a volume during development:

```bash
docker run --rm -v $(pwd):/vllm -it vllm/vllm-docker-executor:latest bash
# Inside container, changes to /vllm are reflected immediately
```

## Architecture

The DockerDistributedExecutor works by:

1. **Executor (Host side)** - Runs in the main Python process, manages container lifecycle
2. **Worker Containers** - One per GPU, spawned by the executor using the custom image
3. **Communication** - ZMQ TCP for control plane, NCCL for tensor parallelism

```
Host Machine
├─ DockerDistributedExecutor (Python main process)
│  ├─ MessageQueue (ZMQ TCP broadcaster)
│  └─ Container management (docker commands)
├─ Container 0 (GPU 0)
│  └─ Worker 0 (docker_worker_entrypoint.py from custom image)
├─ Container 1 (GPU 1)
│  └─ Worker 1 (docker_worker_entrypoint.py from custom image)
└─ ...
```

## Notes

- The custom image includes both `docker_worker_entrypoint.py` and `docker_executor.py`
- Workers use the entrypoint script to connect to the executor's MessageQueue
- The editable install means you can modify source code and test without rebuilding (when using volume mounts)
