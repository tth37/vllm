#!/bin/bash
# Run the active exp1 comparison across multiple Qwen3 model sizes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

SWEEP_NUM_PROMPTS="${SWEEP_NUM_PROMPTS:-1000}"
SWEEP_MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
)
SWEEP_SUMMARY="${SWEEP_SUMMARY:-$SCRIPT_DIR/model_sweep_summary.txt}"

run_variant_for_model() {
    local runner="$1"
    local results_dir="$2"
    local model_name="$3"
    local gpu_memory_utilization="$4"
    local startup_timeout="$5"

    MODEL="$model_name" \
    NUM_PROMPTS="$SWEEP_NUM_PROMPTS" \
    GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" \
    SERVER_STARTUP_TIMEOUT="$startup_timeout" \
    RESULTS_DIR="$results_dir" \
    "$runner"
}

append_model_summary() {
    local model_name="$1"
    local slug="$2"
    local gpu_memory_utilization="$3"
    local baseline_dir="$SCRIPT_DIR/baseline/$slug"
    local sync_dir="$SCRIPT_DIR/dockerbe_sync_output/$slug"
    local hybrid_dir="$SCRIPT_DIR/dockerbe_hybrid_shm/$slug"
    local full_dir="$SCRIPT_DIR/dockerbe_full_shm/$slug"

    {
        echo "================================================================"
        echo "  $model_name"
        echo "  Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "  GPUs: CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
        echo "  Dataset: ShareGPT first $SWEEP_NUM_PROMPTS prompts ($SHAREGPT_PATH)"
        echo "  GPU memory utilization: $gpu_memory_utilization"
        echo "================================================================"
        echo ""
        echo "TP=1 comparison"
        echo "  Baseline Docker+MP:       median TPOT $(metric_value 'Median TPOT (ms)' "$baseline_dir/3_docker_mp_tp1_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$baseline_dir/3_docker_mp_tp1_bench.txt")"
        echo "  DockerBE sync output:     median TPOT $(metric_value 'Median TPOT (ms)' "$sync_dir/5_host_dockerbe_tp1_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$sync_dir/5_host_dockerbe_tp1_bench.txt")"
        echo "  DockerBE hybrid SHM:      median TPOT $(metric_value 'Median TPOT (ms)' "$hybrid_dir/5_host_dockerbe_tp1_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$hybrid_dir/5_host_dockerbe_tp1_bench.txt")"
        echo "  DockerBE full SHM:        median TPOT $(metric_value 'Median TPOT (ms)' "$full_dir/5_host_dockerbe_tp1_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$full_dir/5_host_dockerbe_tp1_bench.txt")"
        echo ""
        echo "TP=2 comparison"
        echo "  Baseline Docker+MP:       median TPOT $(metric_value 'Median TPOT (ms)' "$baseline_dir/4_docker_mp_tp2_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$baseline_dir/4_docker_mp_tp2_bench.txt")"
        echo "  DockerBE sync output:     median TPOT $(metric_value 'Median TPOT (ms)' "$sync_dir/6_host_dockerbe_tp2_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$sync_dir/6_host_dockerbe_tp2_bench.txt")"
        echo "  DockerBE hybrid SHM:      median TPOT $(metric_value 'Median TPOT (ms)' "$hybrid_dir/6_host_dockerbe_tp2_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$hybrid_dir/6_host_dockerbe_tp2_bench.txt")"
        echo "  DockerBE full SHM:        median TPOT $(metric_value 'Median TPOT (ms)' "$full_dir/6_host_dockerbe_tp2_bench.txt") ms, output tok/s $(metric_value 'Output token throughput (tok/s)' "$full_dir/6_host_dockerbe_tp2_bench.txt")"
        echo ""
        echo "Stored results"
        echo "  baseline: $baseline_dir"
        echo "  dockerbe_sync_output: $sync_dir"
        echo "  dockerbe_hybrid_shm: $hybrid_dir"
        echo "  dockerbe_full_shm: $full_dir"
        echo ""
    } >> "$SWEEP_SUMMARY"
}

main() {
    local model_name=""
    local slug=""
    local gpu_memory_utilization=""
    local startup_timeout=""

    trap cleanup_runtime EXIT INT TERM

    verify_experiment_prereqs
    enforce_gpu_policy
    check_selected_gpus_idle

    : > "$SWEEP_SUMMARY"

    for model_name in "${SWEEP_MODELS[@]}"; do
        slug="$(model_slug "$model_name")"
        gpu_memory_utilization="$(recommended_gpu_memory_utilization "$model_name")"
        startup_timeout="$(recommended_startup_timeout "$model_name")"

        log "Running full exp1 sweep for $model_name"
        log "Results will be stored under */$slug"

        run_variant_for_model \
            "$SCRIPT_DIR/run_baseline_docker_mp.sh" \
            "$SCRIPT_DIR/baseline/$slug" \
            "$model_name" \
            "$gpu_memory_utilization" \
            "$startup_timeout"

        run_variant_for_model \
            "$SCRIPT_DIR/run_dockerbe_sync_output.sh" \
            "$SCRIPT_DIR/dockerbe_sync_output/$slug" \
            "$model_name" \
            "$gpu_memory_utilization" \
            "$startup_timeout"

        run_variant_for_model \
            "$SCRIPT_DIR/run_dockerbe_hybrid_shm.sh" \
            "$SCRIPT_DIR/dockerbe_hybrid_shm/$slug" \
            "$model_name" \
            "$gpu_memory_utilization" \
            "$startup_timeout"

        run_variant_for_model \
            "$SCRIPT_DIR/run_dockerbe_full_shm.sh" \
            "$SCRIPT_DIR/dockerbe_full_shm/$slug" \
            "$model_name" \
            "$gpu_memory_utilization" \
            "$startup_timeout"

        append_model_summary "$model_name" "$slug" "$gpu_memory_utilization"
    done

    ok "Multi-model summary saved to: $SWEEP_SUMMARY"
}

main "$@"
