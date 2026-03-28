# DockerBE Single-Visible-GPU Baseline

This directory stores the DockerBE baseline variant where each worker container
sees only one GPU and runs in its own IPC namespace.

- Server placement: host `vllm serve`
- Backend: `--distributed-executor-backend docker`
- Worker GPU visibility: one host GPU per container
- IPC namespace: private per container
- Transport: TCP broadcast MQ + TCP response MQ
- Output path: `VLLM_DOCKER_ASYNC_OUTPUT_COPY=1`
- Optimization / ablation intent: force Docker workers to look like separate one-GPU nodes so vLLM falls back from the intra-node path to the inter-node communication path
- Git branch: `exp1/dockerbe_single_visible_gpu`
- Docker image tag: `tth37/vllm-docker-executor:exp1-dockerbe_single_visible_gpu`
- Published Docker Hub tag: `tth37/vllm-docker-executor:exp1-dockerbe_single_visible_gpu`
- Internal helper: [scripts/run_dockerbe_single_visible_gpu.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/scripts/run_dockerbe_single_visible_gpu.sh)
