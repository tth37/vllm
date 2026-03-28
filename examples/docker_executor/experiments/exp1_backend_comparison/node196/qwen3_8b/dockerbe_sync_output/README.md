# DockerBE Sync-Output Ablation

This directory holds the DockerBE ablation that removes async output copy while keeping SHM RPC enabled.

- Server placement: host `vllm serve`
- Backend: `--distributed-executor-backend docker`
- Transport: SHM broadcast MQ and SHM worker response MQ
- Output path: `VLLM_DOCKER_ASYNC_OUTPUT_COPY=0`
- Optimizations adopted: DockerDistributedExecutor, SHM broadcast, SHM response
- Optimization removed: async output copy
- Git branch: `exp1/dockerbe_sync_output`
- Docker image tag: `vllm/vllm-docker-executor:exp1-dockerbe_sync_output`
- Run script: [run_dockerbe_sync_output.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_dockerbe_sync_output.sh)
- GPUs: `1,2`
- Model: `Qwen/Qwen3-8B`
- Benchmark: first `500` ShareGPT prompts, `10 req/s`, `3` warmups, `--ignore-eos`
