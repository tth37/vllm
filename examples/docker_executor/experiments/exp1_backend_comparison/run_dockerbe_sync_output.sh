#!/bin/bash
# DockerBE ablation: keep SHM RPC enabled, disable async output copy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/dockerbe_sync_output}"

run_config() {
    local config_id="$1"
    local tp_size="$2"
    local label="${config_id}_host_dockerbe_tp${tp_size}"
    local server_log="$RESULTS_DIR/${label}_server.log"
    local bench_file="$RESULTS_DIR/${label}_bench.txt"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Ablation config $config_id: Host + DockerBE + full SHM + sync output + TP=$tp_size"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup_runtime
    sleep 3

    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        VLLM_DOCKER_IMAGE="$DOCKERBE_SYNC_OUTPUT_IMAGE" \
        VLLM_DOCKER_ASYNC_OUTPUT_COPY=0 \
        VLLM_DOCKER_BROADCAST_MQ_SHM=1 \
        VLLM_DOCKER_RESPONSE_MQ_SHM=1 \
        "$VLLM_CMD" serve "$MODEL" \
            --distributed-executor-backend docker \
            --tensor-parallel-size "$tp_size" \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    log "Waiting for sync-output DockerBE server to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Sync-output DockerBE server did not start within ${SERVER_STARTUP_TIMEOUT}s"
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
    require_docker_image "$DOCKERBE_SYNC_OUTPUT_IMAGE"
    prepare_results_dir "$RESULTS_DIR"

    run_config 5 1
    run_config 6 2

    EXPERIMENT_BRANCH_NAME="$DOCKERBE_SYNC_OUTPUT_BRANCH_NAME" \
    EXPERIMENT_IMAGE_TAG="$DOCKERBE_SYNC_OUTPUT_IMAGE" \
    EXPERIMENT_OPTIMIZATIONS="DockerDistributedExecutor on the host, SHM broadcast MQ, SHM worker response MQ, and VLLM_DOCKER_ASYNC_OUTPUT_COPY=0." \
    generate_results_report \
        "$RESULTS_DIR" \
        "Experiment 1 DockerBE Sync-Output Ablation Report" \
        "Host vLLM runs from the local source tree with DockerDistributedExecutor, full SHM RPC, and VLLM_DOCKER_ASYNC_OUTPUT_COPY=0."
}

main "$@"
