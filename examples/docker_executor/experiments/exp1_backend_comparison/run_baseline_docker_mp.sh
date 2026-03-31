#!/bin/bash
# Exp1 variant: Baseline Docker+MP
# vLLM serve runs inside a Docker container with --distributed-executor-backend mp.
# The container sees all selected GPUs; vLLM uses MultiprocExecutor internally.
#
# Called by run_exp1_node192.sh (or manually).
# Expects RESULTS_DIR, MODEL, and GPU_DEVICES to be set via exp1_common.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp1_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/baseline}"

EXPERIMENT_BRANCH_NAME="$BASELINE_BRANCH_NAME"
EXPERIMENT_IMAGE_TAG="$BASELINE_IMAGE"
EXPERIMENT_OPTIMIZATIONS="Runs inside Docker, uses MultiprocExecutor, and does not use DockerDistributedExecutor RPC."

run_tp() {
    local tp=$1
    local prefix=$2
    local bench_file="$RESULTS_DIR/${prefix}_bench.txt"
    local server_log="$RESULTS_DIR/${prefix}_server.log"

    cleanup_runtime
    sleep 3

    local gpu_mem="${GPU_MEMORY_UTILIZATION:-0.5}"

    log "Starting baseline Docker+MP server (TP=$tp)"
    docker run -d --rm \
        --name vllm-exp1-baseline \
        --gpus "\"device=${GPU_DEVICES}\"" \
        --network host \
        --ipc host \
        --shm-size=8g \
        -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
        "$BASELINE_IMAGE" \
        vllm serve "$MODEL" \
            --tensor-parallel-size "$tp" \
            --port "$PORT" \
            --gpu-memory-utilization "$gpu_mem" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        > /dev/null 2>&1

    docker logs -f vllm-exp1-baseline > "$server_log" 2>&1 &
    LOG_FOLLOW_PID=$!

    log "Waiting for server (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Baseline server did not start (TP=$tp)"
        tail -30 "$server_log" 2>/dev/null || true
        echo "--- FAILED ---" > "$bench_file"
        cleanup_runtime
        return 1
    fi
    ok "Server ready"

    run_sharegpt_benchmark "$bench_file"
    cleanup_runtime
    sleep 3
}

main() {
    verify_experiment_prereqs
    enforce_gpu_policy
    check_selected_gpus_idle || exit 1
    require_docker_image "$BASELINE_IMAGE"

    prepare_results_dir "$RESULTS_DIR"
    trap cleanup_runtime EXIT INT TERM

    # Determine TP sizes from the number of GPUs
    local ngpus
    ngpus=$(echo "$GPU_DEVICES" | tr ',' '\n' | wc -l)

    # TP=1
    run_tp 1 "3_docker_mp_tp1"

    # TP=2 (if we have >= 2 GPUs)
    if (( ngpus >= 2 )); then
        run_tp 2 "4_docker_mp_tp2"
    fi

    # TP=4 (if we have >= 4 GPUs)
    if (( ngpus >= 4 )); then
        run_tp 4 "5_docker_mp_tp4"
    fi

    generate_results_report "$RESULTS_DIR" \
        "Experiment 1 Baseline Report" \
        "vLLM serve runs inside Docker with --distributed-executor-backend mp and the container sees GPUs ${GPU_DEVICES}."
}

main "$@"
