#!/bin/bash
# Clean baseline runner for exp1.
# Runs vLLM fully inside Docker with the multiprocess backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/baseline}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-exp1-baseline}"

run_config() {
    local config_id="$1"
    local tp_size="$2"
    local label="${config_id}_docker_mp_tp${tp_size}"
    local server_log="$RESULTS_DIR/${label}_server.log"
    local bench_file="$RESULTS_DIR/${label}_bench.txt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Baseline config $config_id: Docker + MP + TP=$tp_size"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup_runtime
    sleep 3

    docker run -d --rm \
        --name "$CONTAINER_NAME" \
        --gpus "\"device=$GPU_DEVICES\"" \
        --network host \
        --ipc host \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        "$BASELINE_IMAGE" \
        vllm serve "$MODEL" \
            --distributed-executor-backend mp \
            --tensor-parallel-size "$tp_size" \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        >/dev/null

    docker logs -f "$CONTAINER_NAME" > "$server_log" 2>&1 &
    LOG_FOLLOW_PID=$!

    log "Waiting for baseline server to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Baseline server did not start within ${SERVER_STARTUP_TIMEOUT}s"
        tail -40 "$server_log" 2>/dev/null || true
        return 1
    fi

    run_sharegpt_benchmark "$bench_file"
    cleanup_runtime
    sleep 3
}

main() {
    trap cleanup_runtime EXIT INT TERM

    verify_experiment_prereqs
    enforce_gpu_policy
    check_selected_gpus_idle
    require_docker_image "$BASELINE_IMAGE"
    prepare_results_dir "$RESULTS_DIR"

    run_config 3 1
    run_config 4 2

    EXPERIMENT_BRANCH_NAME="$BASELINE_BRANCH_NAME" \
    EXPERIMENT_IMAGE_TAG="$BASELINE_IMAGE" \
    EXPERIMENT_OPTIMIZATIONS="Runs inside Docker, uses MultiprocExecutor, and does not use DockerDistributedExecutor RPC." \
    generate_results_report \
        "$RESULTS_DIR" \
        "Experiment 1 Baseline Report" \
        "vLLM serve runs inside Docker with --distributed-executor-backend mp and the container sees GPUs $GPU_DEVICES."
}

main "$@"
