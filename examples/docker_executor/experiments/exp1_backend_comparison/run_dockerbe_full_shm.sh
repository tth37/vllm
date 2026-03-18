#!/bin/bash
# Clean optimized runner for exp1.
# Runs host vLLM with DockerDistributedExecutor, full SHM RPC, and async output copy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/dockerbe_full_shm}"

run_config() {
    local config_id="$1"
    local tp_size="$2"
    local label="${config_id}_host_dockerbe_tp${tp_size}"
    local server_log="$RESULTS_DIR/${label}_server.log"
    local bench_file="$RESULTS_DIR/${label}_bench.txt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Optimized config $config_id: Host + DockerBE + TP=$tp_size"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup_runtime
    sleep 3

    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        VLLM_DOCKER_IMAGE="$DOCKERBE_IMAGE" \
        VLLM_DOCKER_ASYNC_OUTPUT_COPY=1 \
        "$VLLM_CMD" serve "$MODEL" \
            --distributed-executor-backend docker \
            --tensor-parallel-size "$tp_size" \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    log "Waiting for DockerBE server to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "DockerBE server did not start within ${SERVER_STARTUP_TIMEOUT}s"
        tail -60 "$server_log" 2>/dev/null || true
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
    require_docker_image "$DOCKERBE_IMAGE"
    prepare_results_dir "$RESULTS_DIR"

    run_config 5 1
    run_config 6 2

    generate_results_report \
        "$RESULTS_DIR" \
        "Experiment 1 DockerBE Full SHM Report" \
        "Host vLLM runs from the local source tree with DockerDistributedExecutor, full SHM RPC, and VLLM_DOCKER_ASYNC_OUTPUT_COPY=1."
}

main "$@"
