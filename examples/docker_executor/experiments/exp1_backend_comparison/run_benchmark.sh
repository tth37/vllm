#!/bin/bash
# Clean entrypoint for the active exp1 comparison.
# Runs the Docker-in-Docker MP baseline plus the DockerBE ablations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

SUMMARY_REPORT="${SUMMARY_REPORT:-$SCRIPT_DIR/summary_report.txt}"
BUILD_IMAGES="${BUILD_IMAGES:-0}"

generate_summary_report() {
    local baseline_tp1="$SCRIPT_DIR/baseline/3_docker_mp_tp1_bench.txt"
    local baseline_tp2="$SCRIPT_DIR/baseline/4_docker_mp_tp2_bench.txt"
    local sync_tp1="$SCRIPT_DIR/dockerbe_sync_output/5_host_dockerbe_tp1_bench.txt"
    local sync_tp2="$SCRIPT_DIR/dockerbe_sync_output/6_host_dockerbe_tp2_bench.txt"
    local hybrid_tp1="$SCRIPT_DIR/dockerbe_hybrid_shm/5_host_dockerbe_tp1_bench.txt"
    local hybrid_tp2="$SCRIPT_DIR/dockerbe_hybrid_shm/6_host_dockerbe_tp2_bench.txt"
    local docker_tp1="$SCRIPT_DIR/dockerbe_full_shm/5_host_dockerbe_tp1_bench.txt"
    local docker_tp2="$SCRIPT_DIR/dockerbe_full_shm/6_host_dockerbe_tp2_bench.txt"

    {
        echo "================================================================"
        echo "  Experiment 1 Active Comparison Summary"
        echo "  Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "  Git branch: $(git -C "$REPO_ROOT" branch --show-current)"
        echo "  Git commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
        echo "  Model: $MODEL"
        echo "  GPUs: CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
        echo "  Dataset: ShareGPT first $NUM_PROMPTS prompts ($SHAREGPT_PATH)"
        echo "================================================================"
        echo ""
        echo "TP=1 comparison"
        echo "  Baseline Docker+MP:       median TPOT $(metric_value 'Median TPOT (ms)' "$baseline_tp1") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$baseline_tp1")"
        echo "  DockerBE sync output:     median TPOT $(metric_value 'Median TPOT (ms)' "$sync_tp1") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$sync_tp1")"
        echo "  DockerBE hybrid SHM:      median TPOT $(metric_value 'Median TPOT (ms)' "$hybrid_tp1") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$hybrid_tp1")"
        echo "  DockerBE full SHM:        median TPOT $(metric_value 'Median TPOT (ms)' "$docker_tp1") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$docker_tp1")"
        echo ""
        echo "TP=2 comparison"
        echo "  Baseline Docker+MP:       median TPOT $(metric_value 'Median TPOT (ms)' "$baseline_tp2") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$baseline_tp2")"
        echo "  DockerBE sync output:     median TPOT $(metric_value 'Median TPOT (ms)' "$sync_tp2") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$sync_tp2")"
        echo "  DockerBE hybrid SHM:      median TPOT $(metric_value 'Median TPOT (ms)' "$hybrid_tp2") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$hybrid_tp2")"
        echo "  DockerBE full SHM:        median TPOT $(metric_value 'Median TPOT (ms)' "$docker_tp2") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$docker_tp2")"
    } > "$SUMMARY_REPORT"

    ok "Summary report saved to: $SUMMARY_REPORT"
}

main() {
    verify_experiment_prereqs
    enforce_gpu_policy

    if [[ "$BUILD_IMAGES" == "1" ]]; then
        "$SCRIPT_DIR/build_exp1_images.sh"
    fi

    "$SCRIPT_DIR/run_baseline_docker_mp.sh"
    "$SCRIPT_DIR/run_dockerbe_sync_output.sh"
    "$SCRIPT_DIR/run_dockerbe_hybrid_shm.sh"
    "$SCRIPT_DIR/run_dockerbe_full_shm.sh"
    generate_summary_report
}

main "$@"
