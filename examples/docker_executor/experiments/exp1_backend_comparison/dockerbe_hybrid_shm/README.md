# DockerBE Hybrid-SHM Ablation

This directory holds the DockerBE ablation that keeps async output copy but disables SHM for worker responses.

- Server placement: host `vllm serve`
- Backend: `--distributed-executor-backend docker`
- Transport: SHM broadcast MQ and TCP worker response MQ
- Output path: `VLLM_DOCKER_ASYNC_OUTPUT_COPY=1`
- Optimizations adopted: DockerDistributedExecutor, SHM broadcast, async output copy
- Optimization removed: SHM worker response MQ
- Git branch: `exp1/dockerbe_hybrid_shm`
- Docker image tag: `vllm/vllm-docker-executor:exp1-dockerbe_hybrid_shm`
- Published Docker Hub tag: `tth37/vllm-docker-executor:exp1-dockerbe_hybrid_shm`
- Run script: [run_dockerbe_hybrid_shm.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_dockerbe_hybrid_shm.sh)
- GPUs: `1,2`
- Model: `Qwen/Qwen3-8B`
- Benchmark: first `500` ShareGPT prompts, `10 req/s`, `3` warmups, `--ignore-eos`
