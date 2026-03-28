# Experiment 1: Baseline vs DockerBE Ablations

Compares 4 vLLM backend configurations to quantify DockerDistributedExecutor overhead:

- **baseline**: vLLM runs fully inside Docker with the multiprocess backend (`--distributed-executor-backend mp`)
- **dockerbe_sync_output**: DockerBE with SHM RPC but synchronous output copy (`VLLM_DOCKER_ASYNC_OUTPUT_COPY=0`)
- **dockerbe_hybrid_shm**: DockerBE with async output copy but TCP worker responses (`VLLM_DOCKER_RESPONSE_MQ_SHM=0`)
- **dockerbe_full_shm**: DockerBE with full SHM RPC and async output copy (the optimized target)

## Results Summary

### node192 (2x A100-SXM4-40GB, NVLink NV12)

Tested with Qwen3-4B, 8B, and 14B at TP=1 and TP=2. GPU pair: `0,1`.

| Model | Variant | TP | Median TPOT (ms) | Output tok/s | vs Baseline |
|-------|---------|---:|------------------:|-------------:|------------:|
| **Qwen3-4B** | baseline | 1 | 9.33 | 1062.79 | — |
| | dockerbe_sync_output | 1 | 9.82 | 1058.59 | +5.3% |
| | dockerbe_hybrid_shm | 1 | 9.35 | 1061.86 | +0.2% |
| | dockerbe_full_shm | 1 | 9.35 | 1061.81 | +0.2% |
| | baseline | 2 | 6.66 | 1081.45 | — |
| | dockerbe_sync_output | 2 | 7.13 | 1079.68 | +7.1% |
| | dockerbe_hybrid_shm | 2 | 6.70 | 1083.23 | +0.6% |
| | dockerbe_full_shm | 2 | 6.71 | 1083.19 | +0.8% |
| **Qwen3-8B** | baseline | 1 | 15.31 | 1021.94 | — |
| | dockerbe_sync_output | 1 | 16.05 | 1018.70 | +4.8% |
| | dockerbe_hybrid_shm | 1 | 15.26 | 1021.65 | -0.3% |
| | dockerbe_full_shm | 1 | 15.28 | 1021.58 | -0.2% |
| | baseline | 2 | 9.02 | 1063.96 | — |
| | dockerbe_sync_output | 2 | 9.52 | 1059.08 | +5.5% |
| | dockerbe_hybrid_shm | 2 | 9.06 | 1062.34 | +0.4% |
| | dockerbe_full_shm | 2 | 9.06 | 1062.46 | +0.4% |
| **Qwen3-14B** | baseline | 1 | 32.51 | 946.52 | — |
| | dockerbe_sync_output | 1 | 33.16 | 943.43 | +2.0% |
| | dockerbe_hybrid_shm | 1 | 32.53 | 946.21 | +0.1% |
| | dockerbe_full_shm | 1 | 32.55 | 946.02 | +0.1% |
| | baseline | 2 | 15.81 | 1019.32 | — |
| | dockerbe_sync_output | 2 | 16.33 | 1014.44 | +3.3% |
| | dockerbe_hybrid_shm | 2 | 15.85 | 1017.78 | +0.3% |
| | dockerbe_full_shm | 2 | 15.85 | 1017.69 | +0.3% |

### node196 (2x A100-PCIE-40GB, PCIe only)

Tested with Qwen3-8B only. GPU pair: `1,2`. (See `node196/` for raw results.)

### Key Findings

1. **DockerBE full_shm has near-zero overhead** (<1% TPOT increase) compared to the baseline Docker+MP across all model sizes and TP configurations on NVLink hardware.

2. **Sync output is the bottleneck**, not SHM transport. The `sync_output` variant shows a consistent 2-7% TPOT penalty, confirming that async output copy is the critical optimization.

3. **NVLink delivers 29-51% TPOT reduction** when going from TP=1 to TP=2 (e.g., 14B: 32.51ms → 15.81ms). The DockerBE does not degrade this NVLink benefit.

4. **Consistent across model sizes**: The overhead pattern holds for 4B, 8B, and 14B, suggesting the result generalizes to other models.

## Methodology

- Dataset: first 500 prompts from ShareGPT_V3_unfiltered_cleaned_split.json
- Load shape: `--request-rate 10`, `--num-warmups 3`, `--ignore-eos`
- `gpu_memory_utilization=0.5` (0.9 for 14B TP=1)
- `max_model_len=512`
- All 4 Docker images are built from the same source tree and tagged differently

## Scripts

- `build_exp1_images.sh`: build and tag the exp1 Docker images
- `run_baseline_docker_mp.sh`: baseline (vLLM inside Docker with MP backend)
- `run_dockerbe_sync_output.sh`: sync-output ablation
- `run_dockerbe_hybrid_shm.sh`: hybrid-SHM ablation
- `run_dockerbe_full_shm.sh`: optimized DockerBE
- `run_benchmark.sh`: run all 4 variants sequentially
- `run_exp1_node192.sh`: multi-model sweep for node192 (GPU 0,1)
- `exp1_common.sh`: shared defaults, cleanup, GPU policy, report helpers

## How To Reproduce

```bash
cd examples/docker_executor/experiments/exp1_backend_comparison
bash build_exp1_images.sh
bash run_benchmark.sh  # single-model run on default GPUs
# Or multi-model sweep on node192:
bash run_exp1_node192.sh
# Or single model:
MODEL_FILTER=qwen3_8b bash run_exp1_node192.sh
```

## Result Directories

- `node192/qwen3_4b/` — Qwen3-4B results on node192 (A100-SXM4, NVLink)
- `node192/qwen3_8b/` — Qwen3-8B results on node192
- `node192/qwen3_14b/` — Qwen3-14B results on node192
- `node196/qwen3_8b/` — Qwen3-8B results on node196 (A100-PCIE)
- `baseline/`, `dockerbe_*/` — Original node196 raw results (kept for backward compatibility)

## Historical Notes

- The root-cause profiling pass found that the large DockerBE loss came from the old synchronous output path rather than SHM transport itself.
- The async-output A/B and earlier intermediate SHM/TCP variants are preserved under `archive/`.
