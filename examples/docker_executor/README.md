# Docker Distributed Executor for vLLM

This directory contains the Docker-based distributed executor implementation for vLLM V1. This executor runs each worker in a separate Docker container instead of a separate OS process.

## Overview

The `DockerDistributedExecutor` provides an alternative to the multiprocess executor by running workers in isolated Docker containers. This enables:

- **Better isolation**: Each worker runs in its own container environment
- **Scalability**: Workers can run on different hosts (with additional networking setup)
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

### 4. Build Script (`build-docker-executor.sh`)

Builds the Docker image with the custom executor files.

## Quick Start

### 1. Build the Docker Image

```bash
./build-docker-executor.sh
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
