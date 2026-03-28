# Active Baseline Target

This directory is reserved for the cleaned exp1 baseline rerun.

- Server placement: `vllm serve` fully inside Docker
- Backend: `--distributed-executor-backend mp`
- Optimizations adopted: containerized server placement only; no DockerBE RPC path
- Git branch: `exp1/baseline`
- Docker image tag: `vllm/vllm-docker-executor:exp1-baseline`
- Run script: [run_baseline_docker_mp.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_baseline_docker_mp.sh)
- GPUs: `1,2`
- Model: `Qwen/Qwen3-8B`
- Benchmark: first `500` ShareGPT prompts, `10 req/s`, `3` warmups, `--ignore-eos`

Historical `Qwen3-0.6B` results were moved to [archive/reproduced_qwen3_0p6b/baseline](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/archive/reproduced_qwen3_0p6b/baseline).
