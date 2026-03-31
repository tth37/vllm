#!/bin/bash
# Exp1 variant: DockerBE Full SHM
# Host vLLM with DockerDistributedExecutor, full SHM MQs (both broadcast and
# response), and ASYNC_OUTPUT_COPY=1.
# This is the fully optimized DockerBE configuration.
#
# Called by run_exp1_node192.sh (or manually).
# Expects RESULTS_DIR, MODEL, and GPU_DEVICES to be set via exp1_common.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp1_common.sh"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/dockerbe_full_shm}"

EXPERIMENT_BRANCH_NAME="$DOCKERBE_FULL_SHM_BRANCH_NAME"
EXPERIMENT_IMAGE_TAG="$DOCKERBE_IMAGE"
EXPERIMENT_OPTIMIZATIONS="DockerDistributedExecutor on the host, SHM broadcast MQ, SHM worker response MQ, and VLLM_DOCKER_ASYNC_OUTPUT_COPY=1."

run_tp() {
    local tp=$1
    local prefix=$2
    local bench_file="$RESULTS_DIR/${prefix}_bench.txt"
    local server_log="$RESULTS_DIR/${prefix}_server.log"

    cleanup_runtime
    sleep 3

    local gpu_mem="${GPU_MEMORY_UTILIZATION:-0.5}"

    log "Starting DockerBE full-SHM server (TP=$tp)"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        VLLM_DOCKER_IMAGE="$DOCKERBE_IMAGE" \
        VLLM_DOCKER_ASYNC_OUTPUT_COPY=1 \
        VLLM_DOCKER_BROADCAST_MQ_SHM=1 \
        VLLM_DOCKER_RESPONSE_MQ_SHM=1 \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$VLLM_CMD" serve "$MODEL" \
            --tensor-parallel-size "$tp" \
            --distributed-executor-backend docker \
            --port "$PORT" \
            --gpu-memory-utilization "$gpu_mem" \
            --max-model-len "$MAX_MODEL_LEN" \
            --disable-log-requests \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    log "Waiting for server (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "DockerBE full-SHM server did not start (TP=$tp)"
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
    require_docker_image "$DOCKERBE_IMAGE"

    prepare_results_dir "$RESULTS_DIR"
    trap cleanup_runtime EXIT INT TERM

    local ngpus
    ngpus=$(echo "$GPU_DEVICES" | tr ',' '\n' | wc -l)

    run_tp 1 "5_host_dockerbe_tp1"

    if (( ngpus >= 2 )); then
        run_tp 2 "6_host_dockerbe_tp2"
    fi

    if (( ngpus >= 4 )); then
        run_tp 4 "7_host_dockerbe_tp4"
    fi

    generate_results_report "$RESULTS_DIR" \
        "Experiment 1 DockerBE Full SHM Report" \
        "Host vLLM runs from the local source tree with DockerDistributedExecutor, full SHM RPC, and VLLM_DOCKER_ASYNC_OUTPUT_COPY=1."
}

main "$@"
