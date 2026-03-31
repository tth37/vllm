#!/bin/bash
# Phase 2 Variant: Baseline Docker+MP
# Single container running vllm serve with --distributed-executor-backend mp.
# This is the reference benchmark — no DockerBE, no isolation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp2_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$PHASE2_RESULTS_DIR/vllm_baseline}"

main() {
    verify_prereqs
    print_header "Baseline Docker+MP (TP=$TP_SIZE)"

    mkdir -p "$RESULTS_DIR"
    local server_log="$RESULTS_DIR/server.log"
    local bench_file="$RESULTS_DIR/bench.txt"

    trap cleanup EXIT INT TERM
    cleanup
    sleep 3

    docker run -d --rm \
        --name vllm-bench-server \
        --gpus "\"device=${GPU_DEVICES}\"" \
        --network host \
        --ipc host \
        --shm-size=8g \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        "$IMAGE_TAG" \
        vllm serve "$MODEL" \
            --tensor-parallel-size "$TP_SIZE" \
            --port "$PORT" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        > /dev/null 2>&1

    docker logs -f vllm-bench-server > "$server_log" 2>&1 &
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
