# Docker Distributed Executor for vLLM

This directory contains the Docker-based distributed executor implementation for vLLM V1. This executor runs each worker in a separate Docker container instead of a separate OS process.

## Overview

The `DockerDistributedExecutor` provides an alternative to the multiprocess executor by running workers in isolated Docker containers. This enables:

- **Better isolation**: Each worker runs in its own container environment
- **Scalability**: Workers can run on hosts (with additional networking setup)
- **Reproducibility**: Containerized environments ensure consistent dependencies

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Host Machine                                │
│                                                                     │
│  ┌─────────────────────┐         ┌─────────────────────────────┐   │
│  │  Python Process     │         │   Docker Containers         │   │
│  │  (EngineCore)       │         │                             │   │
│  │                     │         │  ┌─────────────┐            │   │
│  │  DockerDistributed  │         │  │ Worker 0    │            │   │
│  │  Executor           │         │  │ (GPU 0)     │            │   │
│  │                     │         │  └─────────────┘            │   │
│  │  ┌───────────────┐  │         │                             │   │
│  │  │ RPC Broadcast │  │◄────────┤  ┌─────────────┐            │   │
│  │  │ MessageQueue  │  │  ZMQ    │  │ Worker 1    │            │   │
│  │  └───────────────┘  │ (host)  │  │ (GPU 1)     │            │   │
│  │                     │         │  └─────────────┘            │   │
│  │  ┌───────────────┐  │         │                             │   │
│  │  │ Response MQs  │  │◄────────┤       ...                   │   │
│  │  │ (per worker)  │  │  ZMQ    │                             │   │
│  │  └───────────────┘  │ (host)  │  ┌─────────────┐            │   │
│  │                     │         │  │ Worker N    │            │   │
│  └─────────────────────┘         │  │ (GPU N)     │            │   │
│                                  │  └─────────────┘            │   │
│                                  │                             │   │
│                                  └─────────────────────────────┘   │
│                                                                     │
│  Shared Volume: /tmp/vllm_docker_shared/                           │
│  ├── worker_response_handle_{rank}.txt  (handle exchange)          │
│  ├── <uuid>                             (ZMQ IPC sockets)          │
│  └── logs/                              (worker logs)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Communication:
- Control Plane: ZMQ MessageQueue (RPC broadcast + response per worker)
- Data Plane: NCCL (GPU-GPU communication via host network)
```

## Components

### 1. DockerDistributedExecutor (`vllm/v1/executor/docker_executor.py`)

The main executor class that:
- Creates network-based MessageQueues for communication
- Spawns Docker containers for each worker
- Manages handle exchange via shared volume
- Monitors container health

Key configuration:
- `VLLM_DOCKER_IMAGE`: Docker image to use (default: `vllm/vllm-docker-executor:latest`)
- `_DOCKER_SHARED_VOLUME`: Shared volume path (default: `/tmp/vllm_docker_shared`)
- `VLLM_RPC_BASE_PATH`: Set to the shared volume so ZMQ IPC sockets are accessible across containers (see [Critical: ZMQ IPC Socket Path](#critical-zmq-ipc-socket-path-vllm_rpc_base_path))

### 2. Docker Worker Entrypoint (`vllm/v1/executor/docker_worker_entrypoint.py`)

The worker script that runs inside containers:
- Deserializes configuration from environment variables
- Initializes distributed environment (NCCL)
- Creates response MessageQueue and exports handle
- Runs main worker loop processing RPC calls

Environment variables passed to containers:
- `VLLM_WORKER_RANK`: Global rank of the worker
- `VLLM_WORKER_LOCAL_RANK`: Local rank (GPU index)
- `VLLM_WORLD_SIZE`: Total number of workers
- `VLLM_SCHEDULER_HANDLE`: Base64-encoded RPC broadcast MQ handle
- `VLLM_MASTER_ADDR`: Host IP address
- `VLLM_DISTRIBUTED_INIT_METHOD`: NCCL init method
- `VLLM_CONFIG`: Base64-encoded VllmConfig
- `VLLM_IS_DRIVER_WORKER`: Whether this is the driver worker
- `VLLM_DOCKER_SHARED_VOLUME`: Path to shared volume

### 3. Dockerfile (`Dockerfile.docker-executor`)

The Docker image definition that:
- Extends the base vLLM image
- Copies the custom executor files into the container
- Sets up the Python path for imports

### 4. Build Script (`build_docker_executor.sh`)

Builds the Docker image with the custom executor files.

## Quick Start

### 1. Build the Docker Image

```bash
./build_docker_executor.sh
```

This creates `vllm/vllm-docker-executor:latest`.

### 2. Run the Test

```bash
python examples/docker_executor/test_docker_executor.py
```

Or from the repo root:

```bash
python -m examples.docker_executor.test_docker_executor
```

### 3. Use in Your Code

```python
from vllm import LLM

llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,  # Use 2 GPUs
    distributed_executor_backend="docker",
)

outputs = llm.generate(["Hello, world!"])
```

## Benchmarking with vLLM Serve

You can benchmark the Docker executor using `vllm serve` with the official vLLM benchmark CLI.

### 1. Start vLLM Serve

```bash
vllm serve facebook/opt-125m \
    --tensor-parallel-size 2 \
    --distributed-executor-backend docker \
    --port 8000
```

### 2. Run the Benchmark

Use the official `vllm bench serve` command:

```bash
# Download the ShareGPT dataset (cached in ~/.cache/vllm/datasets/)
./examples/docker_executor/download_sharegpt.sh

# Run benchmark
vllm bench serve \
  --backend vllm \
  --model facebook/opt-125m \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ~/.cache/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 50
```

### Benchmark Options

| Option | Description |
|--------|-------------|
| `--backend` | Backend type (`vllm`, `openai-chat`, etc.) |
| `--model` | Model name |
| `--endpoint` | API endpoint (`/v1/completions`, `/v1/chat/completions`) |
| `--dataset-name` | Dataset (`sharegpt`, `random`, `custom`, `hf`) |
| `--dataset-path` | Path to dataset file |
| `--num-prompts` | Number of prompts to send |
| `--max-concurrency` | Max concurrent requests |
| `--request-rate` | Request rate (req/s, `inf` for unlimited) |

### Quick Benchmark (No Dataset Required)

```bash
vllm bench serve \
  --backend vllm \
  --model facebook/opt-125m \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 50 \
  --random-input-len 50 \
  --random-output-len 100
```

### Example Output

```
============ Serving Benchmark Result ============
Successful requests:                     50
Benchmark duration (s):                  4.23
Total input tokens:                      2456
Total generated tokens:                  4873
Request throughput (req/s):              11.82
Output token throughput (tok/s):         1151.34
Total token throughput (tok/s):          1731.45
---------------Time to First Token----------------
Mean TTFT (ms):                          45.23
Median TTFT (ms):                        42.15
P99 TTFT (ms):                           89.34
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.12
Median TPOT (ms):                        7.98
P99 TPOT (ms):                           9.45
==================================================
```

### Full Documentation

See the [official vLLM benchmark documentation](https://docs.vllm.ai/en/latest/getting_started/benchmarks.html) for more options including:
- Custom datasets
- Load patterns (burstiness, ramp-up)
- Multimodal benchmarking
- Structured output benchmarking
- Prefix caching benchmarks

## Requirements

- Docker installed and running
- NVIDIA Container Toolkit (`nvidia-docker2`) for GPU access
- Docker image built (`vllm/vllm-docker-executor:latest`)
- Sufficient GPU memory for the model

## Troubleshooting

### Container fails to start

Check Docker logs:
```bash
docker logs vllm-worker-0
```

### Timeout waiting for handle

The worker may be taking too long to load the model. Check:
1. Container is running: `docker ps`
2. Worker logs in `/tmp/vllm_docker_shared/logs/`
3. GPU memory availability

### GPU not accessible in container

Ensure NVIDIA Container Toolkit is installed:
```bash
docker info | grep nvidia
```

If not, install it:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Slow performance with Tensor Parallelism (TP > 1)

If you observe significantly slower performance when using tensor parallelism (TP=2, TP=4, etc.) compared to the default multiprocess backend:

**Symptoms:**
- Token throughput is ~3x slower than expected
- NCCL logs show `via NET/Socket` instead of `via P2P/CUMEM`
- High inter-token latency (ITL)

**Diagnosis:**

Check if NCCL is using NVLink by running with debug logging:
```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH vllm serve model --tensor-parallel-size 2 --distributed-executor-backend docker 2>&1 | grep -E "(P2P|NET|hostHash|nNodes)"
```

**Expected (NVLink working):**
```
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
NCCL INFO comm 0x... rank 0 nRanks 2 nNodes 1 localRanks 2
```

**Problem (using slow TCP):**
```
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via NET/Socket/0
NCCL INFO comm 0x... rank 0 nRanks 2 nNodes 2 localRanks 1
```

**Solution:**

The Docker executor automatically configures the following for optimal NCCL performance:
- `--ipc host --pid host --network host` (container namespace sharing)
- `NVIDIA_VISIBLE_DEVICES=all` (topology discovery)

If you're still seeing slow performance, ensure:
1. Your GPUs have NVLink connectivity: `nvidia-smi topo -m`
2. You're using the latest Docker image: `./build_docker_executor.sh`
3. The NVIDIA Container Toolkit is properly installed

### Workers hang at NCCL init with `--ipc host` (deadlock)

If workers complete NCCL initialization (showing `Init COMPLETE` and `SHM/direct/direct` transport) but then hang — never writing their response handle files — the issue is the **ZMQ IPC socket path**.

**Symptoms:**
- Executor stuck at `Waiting for worker 0 to export handle...`
- Container logs end at `Connected all rings` or `SymmMemCommunicator: Device capability X.X not supported`
- No further output from any worker container
- Handle files in `/tmp/vllm_docker_shared/` are never created

**Root Cause:**

With `--ipc host`, containers share `/dev/shm`, so vLLM's `in_the_same_node_as()` detects workers as co-located. This causes the internal `GroupCoordinator` to create `MessageQueue` instances using **local readers** (POSIX shared memory + ZMQ IPC sockets). The POSIX shared memory works fine (shared `/dev/shm`), but the ZMQ IPC sockets are created at `ipc:///tmp/{uuid}` — and `/tmp` is **container-local filesystem**, not shared between containers. The reader in container 1 cannot connect to the IPC socket in container 0's `/tmp`, causing `wait_until_ready()` to deadlock.

**Solution:**

The Docker executor sets `VLLM_RPC_BASE_PATH` to the shared Docker volume (`/tmp/vllm_docker_shared`). This ensures ZMQ IPC sockets are created on the shared volume, accessible from all containers. This is handled automatically — no user action required.

If you're building a custom executor and encounter this issue, ensure you pass:
```
-e VLLM_RPC_BASE_PATH=/path/to/shared/volume
```

## Implementation Details

### Handle Exchange

The executor and workers communicate via a shared Docker volume:

1. Worker creates a response MessageQueue
2. Worker exports the handle and writes to shared volume:
   `/tmp/vllm_docker_shared/worker_response_handle_{rank}.txt`
3. Executor polls for this file (up to 120 seconds)
4. Executor connects to worker's response MQ using the handle

### Message Queue Synchronization

Due to ZMQ's asynchronous subscription model, we skip `wait_until_ready()` on response MQs to avoid circular wait:
- Worker would wait for executor subscription
- Executor would wait for worker's READY signal
- Solution: Trust the connection is established and defer sync to actual message exchange

### Network Configuration

- **Host network mode**: Used for simplicity (ZMQ and NCCL can communicate directly)
- **Host IP detection**: Automatically detects host IP for container communication
- **Port allocation**: Dynamic port allocation for MessageQueues

### NCCL/NVLink Configuration

For optimal multi-GPU communication performance with tensor parallelism, the Docker executor configures containers to use NVLink (achieving ~85 GB/s bandwidth) instead of slow TCP sockets (~3 GB/s):

| Configuration | Purpose |
|--------------|---------|
| `--network host` | Share host network namespace for NCCL socket communication |
| `--ipc host` | Share host IPC namespace for CUDA IPC/P2P memory access and POSIX shared memory |
| `--pid host` | Share host PID namespace for NCCL process visibility |
| `NVIDIA_VISIBLE_DEVICES=all` | All containers see all GPUs for topology discovery |
| `VLLM_RPC_BASE_PATH=<shared_vol>` | ZMQ IPC sockets on shared volume so containers can communicate locally |

**Performance Impact**: This configuration enables native P2P/SHM performance in Docker containers, significantly faster than default Docker networking (NET/Socket).

To verify fast transport is being used, check the container logs for:
```
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct
```
or (with NVLink):
```
NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
```

Key indicators of successful single-node detection:
- `nNodes 1` - Containers detected as single node
- `localRanks N` - All ranks are local
- `via SHM/direct/direct` or `via P2P/CUMEM/read` - Using fast transport, not TCP

## Environment Setup Guide

Step-by-step instructions for setting up the experiment environment on a fresh machine (e.g., migrating to a new node).

### 1. Prerequisites

Ensure the following are available on the target machine:

- NVIDIA GPU with drivers installed (`nvidia-smi` should work)
- Docker Engine with `nvidia-container-toolkit` (formerly `nvidia-docker2`)
- `curl`, `git`

### 2. Clone or copy the repository

The expected layout is `~/repositories/vllm-dev/vllm-source/`.

**Option A -- git clone from origin:**

```bash
mkdir -p ~/repositories/vllm-dev
cd ~/repositories/vllm-dev
git clone <origin-url> vllm-source
```

**Option B -- rsync from an existing machine:**

```bash
rsync -az user@source-machine:~/repositories/vllm-dev/ ~/repositories/vllm-dev/
```

### 3. Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your shell or `source ~/.bashrc` / `source ~/.config/fish/config.fish` so that `uv` is on your PATH.

### 4. Create the Python virtualenv

```bash
cd ~/repositories/vllm-dev
uv venv --python 3.12 .venv
source .venv/bin/activate      # bash/zsh
# source .venv/bin/activate.fish  # fish
```

Install vllm from source (editable mode):

```bash
# If the repo was rsync'd without .git, setuptools-scm needs a version hint:
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev0

uv pip install -e vllm-source/
```

Install additional dependencies for report generation:

```bash
uv pip install matplotlib
```

### 5. Build the Docker image

```bash
cd ~/repositories/vllm-dev/vllm-source
bash examples/docker_executor/build_docker_executor.sh
```

To build with a custom tag (e.g., for a specific experiment):

```bash
TARGET_IMAGE=vllm/vllm-docker-executor:exp2-cumem \
  bash examples/docker_executor/build_docker_executor.sh
```

### 6. Download the ShareGPT dataset

```bash
bash examples/docker_executor/download_sharegpt.sh
```

The dataset is cached at `~/.cache/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`.

### 7. Download model weights

`hf` CLI is installed automatically as part of the vllm dependencies (via `huggingface_hub`).

For machines in China, use the HuggingFace mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Then download the models you need:

```bash
hf download Qwen/Qwen3-4B
hf download Qwen/Qwen3-8B
hf download Qwen/Qwen3-14B
```

Weights are cached under `~/.cache/huggingface/hub/`.

### 8. Verify the setup

```bash
# Check GPU visibility
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 nvidia-smi

# Check vllm installation
.venv/bin/python3 -c "import vllm; print(vllm.__version__)"
```

If all three commands succeed, the environment is ready to run experiments.

## Future Improvements

1. **Multi-node support**: Add etcd/consul for service discovery across hosts
2. **Proper handshake**: Replace file-based handle exchange with HTTP/gRPC handshake
3. **Container orchestration**: Support for Kubernetes/Docker Swarm
4. **Resource limits**: Add Docker resource limits (CPU, memory)
5. **Health checks**: Implement proper Docker health checks

## See Also

- [vLLM Distributed Inference Documentation](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
