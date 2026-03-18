# Active DockerBE Target

This directory is reserved for the cleaned exp1 optimized rerun.

- Server placement: host `vllm serve`
- Backend: `--distributed-executor-backend docker`
- Transport: full SHM RPC
- Output path: `VLLM_DOCKER_ASYNC_OUTPUT_COPY=1`
- GPUs: `1,2`
- Model: `Qwen/Qwen3-8B`
- Benchmark: first `500` ShareGPT prompts, `10 req/s`, `3` warmups, `--ignore-eos`

Historical `Qwen3-0.6B` results were moved to [archive/reproduced_qwen3_0p6b/dockerbe_full_shm](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/archive/reproduced_qwen3_0p6b/dockerbe_full_shm).
