#!/bin/bash
# Phase 2 Variant: DockerBE + CUMEM + All GPUs Visible (custom allreduce enabled)
#
# Ablation to isolate whether the ~10% latency overhead comes from disabled
# custom allreduce or from container/DockerBE RPC overhead.
#
# Each worker container sees ALL GPUs (NVIDIA_VISIBLE_DEVICES=all,
# CUDA_VISIBLE_DEVICES=all) but still uses CUMEM for NCCL P2P.
# Custom allreduce is NOT disabled because all GPUs are visible.
#
# Trade-off: GPU compute isolation is sacrificed (each container can access
# any GPU), but NCCL still uses CUMEM path. If this matches baseline latency,
# the overhead in the pure CUMEM variant is from disabled custom allreduce.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp2_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$PHASE2_RESULTS_DIR/vllm_dockerbe_full_vis}"

main() {
    verify_prereqs
    print_header "DockerBE Full GPU Visibility (TP=$TP_SIZE)"

    mkdir -p "$RESULTS_DIR"
    local server_log="$RESULTS_DIR/server.log"
    local bench_file="$RESULTS_DIR/bench.txt"

    trap cleanup EXIT INT TERM
    cleanup
    sleep 3

    # Standard DockerBE mode (no CUMEM isolation) — all GPUs visible,
    # --ipc host for SHM MQs and custom allreduce IPC buffers.
    # This is the exp1 "full SHM" variant adapted for exp2.
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        VLLM_DOCKER_IMAGE="$IMAGE_TAG" \
        VLLM_DOCKER_ASYNC_OUTPUT_COPY=1 \
        VLLM_DOCKER_BROADCAST_MQ_SHM=1 \
        VLLM_DOCKER_RESPONSE_MQ_SHM=1 \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$VLLM_CMD" serve "$MODEL" \
            --tensor-parallel-size "$TP_SIZE" \
            --distributed-executor-backend docker \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    log "Waiting for server (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Server did not start"
        tail -30 "$server_log" 2>/dev/null || true
        echo "--- FAILED ---" > "$bench_file"
        cleanup
        return 1
    fi
    ok "Server ready"

    run_benchmark "$bench_file"
    cleanup
    sleep 3
}

main "$@"
