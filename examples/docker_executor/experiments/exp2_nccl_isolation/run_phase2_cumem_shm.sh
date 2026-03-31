#!/bin/bash
# Phase 2 Variant: DockerBE + CUMEM + IPC Host (SHM MQs)
# Per-GPU worker containers with NCCL CUMEM P2P recovery,
# but with --ipc host to enable SHM message queues.
#
# This ablation isolates the TCP MQ overhead: if this variant matches
# baseline latency, the ~10% overhead in the pure CUMEM variant is
# entirely due to TCP-based message queues.
#
# Trade-off: IPC isolation is sacrificed (containers share host SysV IPC
# and POSIX SHM), but PID/mount/GPU isolation is preserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp2_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$PHASE2_RESULTS_DIR/vllm_dockerbe_cumem_shm}"

main() {
    verify_prereqs
    print_header "DockerBE + CUMEM + IPC Host, SHM MQs (TP=$TP_SIZE)"

    mkdir -p "$RESULTS_DIR"
    local server_log="$RESULTS_DIR/server.log"
    local bench_file="$RESULTS_DIR/bench.txt"

    trap cleanup EXIT INT TERM
    cleanup
    sleep 3

    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        VLLM_DOCKER_IMAGE="$IMAGE_TAG" \
        VLLM_DOCKER_CUMEM_ISOLATION=1 \
        VLLM_DOCKER_CUMEM_IPC_HOST=1 \
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
