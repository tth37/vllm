# Experiment 1: Baseline vs DockerBE Ablations

This directory now keeps the active baseline, the optimized DockerBE target, and two DockerBE ablations that isolate the critical optimizations:

- `baseline/`: vLLM runs fully inside Docker with the multiprocess backend
- `dockerbe_sync_output/`: DockerBE keeps SHM RPC but disables async output copy
- `dockerbe_hybrid_shm/`: DockerBE keeps async output copy but disables SHM for worker responses
- `dockerbe_full_shm/`: host vLLM uses `DockerDistributedExecutor` with full SHM RPC and async output copy

The older historical reproductions and profiling-only artifacts remain under `archive/`.

## Active Methodology

- Model: `Qwen/Qwen3-8B`
- GPU policy: always use `GPU_DEVICES=1,2` unless `ALLOW_GPU_OVERRIDE=1`
- Dataset: first `500` prompts from `ShareGPT_V3_unfiltered_cleaned_split.json`
- Load shape: `--request-rate 10`, `--num-warmups 3`, `--ignore-eos`
- Baseline server: `vllm serve` inside Docker with `--distributed-executor-backend mp`
- Optimized server: host `vllm serve` with `--distributed-executor-backend docker`

## Scripts

- [build_exp1_images.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/build_exp1_images.sh): build and tag the active exp1 Docker images
- [run_baseline_docker_mp.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_baseline_docker_mp.sh): run the inside-Docker MP baseline
- [run_dockerbe_sync_output.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_dockerbe_sync_output.sh): run the async-output ablation
- [run_dockerbe_hybrid_shm.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_dockerbe_hybrid_shm.sh): run the response-SHM ablation
- [run_dockerbe_full_shm.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_dockerbe_full_shm.sh): run the optimized DockerBE comparison
- [run_benchmark.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/run_benchmark.sh): run the baseline plus all active DockerBE variants and write a combined summary
- [exp1_common.sh](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/exp1_common.sh): shared defaults, cleanup, GPU policy, and report helpers

## How To Reproduce

Build the active images first:

```bash
cd /home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison
bash build_exp1_images.sh
```

Run the cleaned comparison suite:

```bash
bash run_benchmark.sh
```

Or run the two variants separately:

```bash
bash run_baseline_docker_mp.sh
bash run_dockerbe_sync_output.sh
bash run_dockerbe_hybrid_shm.sh
bash run_dockerbe_full_shm.sh
```

The scripts refuse to run on any GPU pair other than `1,2` unless you set `ALLOW_GPU_OVERRIDE=1`.

## Active Result Targets

- [baseline](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/baseline)
- [dockerbe_sync_output](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/dockerbe_sync_output)
- [dockerbe_hybrid_shm](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/dockerbe_hybrid_shm)
- [dockerbe_full_shm](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/dockerbe_full_shm)

These directories were reset for the cleaned `Qwen3-8B` rerun. The historical `Qwen3-0.6B` files from the earlier branch-based reproduction live under [archive/reproduced_qwen3_0p6b](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/archive/reproduced_qwen3_0p6b).

## Variant Provenance

- `baseline`: branch `exp1/baseline`, image `vllm/vllm-docker-executor:exp1-baseline`
- `dockerbe_sync_output`: branch `exp1/dockerbe_sync_output`, image `vllm/vllm-docker-executor:exp1-dockerbe_sync_output`
- `dockerbe_hybrid_shm`: branch `exp1/dockerbe_hybrid_shm`, image `vllm/vllm-docker-executor:exp1-dockerbe_hybrid_shm`
- `dockerbe_full_shm`: branch `exp1/dockerbe_full_shm`, image `vllm/vllm-docker-executor:exp1-dockerbe_full_shm`

## Historical Provenance

- The root-cause profiling pass found that the large DockerBE loss came from the old synchronous output path rather than SHM transport itself.
- The async-output A/B and the earlier intermediate SHM/TCP variants are preserved under [archive](/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/experiments/exp1_backend_comparison/archive).
- The dedicated historical branches `exp1/dockerbe_async` and `exp1/dockerbe_shm` were retired during the cleanup pass.

## Current Status

This cleanup pass prepares the codebase and scripts for the final rerun, but it does not refresh the benchmark numbers yet. The next run should be done only after GPUs `1` and `2` are both idle.
