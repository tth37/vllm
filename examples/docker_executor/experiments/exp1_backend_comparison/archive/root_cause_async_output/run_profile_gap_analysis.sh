#!/bin/bash
# Focused profiling run for the remaining TP=2 gap:
#   baseline host multiproc vs host DockerBE full SHM.

set -euo pipefail

VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-10}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-180}"
IMAGE_TAG="${IMAGE_TAG:-vllm/vllm-docker-executor:exp1-profile-gap}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/profile_gap_analysis}"
SHAREGPT_PATH="/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/ShareGPT_V3_unfiltered_cleaned_split.json"
SERVER_PID=""

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

cleanup_all() {
    log "Cleaning up..."
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    docker rm -f vllm-bench-server 2>/dev/null || true
    docker ps -aq --filter "name=vllm-worker-" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    rm -rf /tmp/vllm_docker_shared 2>/dev/null || true
    SERVER_PID=""
}

trap cleanup_all EXIT INT TERM

wait_for_server() {
    local timeout="$1"
    local start=$SECONDS
    while (( SECONDS - start < timeout )); do
        if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

run_benchmark() {
    local output_file="$1"
    env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        timeout 300 "$VLLM_CMD" bench serve \
            --backend vllm \
            --model "$MODEL" \
            --port "$PORT" \
            --endpoint /v1/completions \
            --dataset-name sharegpt \
            --dataset-path "$SHAREGPT_PATH" \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$REQUEST_RATE" \
            --num-warmups "$NUM_WARMUPS" \
            --ignore-eos \
        2>&1 | tee "$output_file"

    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        err "Benchmark failed (exit code $rc)"
        return 1
    fi
    ok "Benchmark complete: $output_file"
}

run_host_mp_tp2() {
    local out_dir="$1"
    local log_file="$out_dir/server.log"
    local bench_file="$out_dir/bench.txt"
    local profile_dir="$out_dir/profiles"

    rm -rf "$out_dir"
    mkdir -p "$profile_dir"
    cleanup_all
    sleep 3

    log "Starting baseline host multiproc TP=2 profile run"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        VLLM_RPC_PROFILE_DIR="$profile_dir" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$VLLM_CMD" serve "$MODEL" \
            --tensor-parallel-size 2 \
            --port "$PORT" \
            --gpu-memory-utilization 0.5 \
            --max-model-len 512 \
            --disable-log-requests \
        > "$log_file" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Baseline host MP server did not start"
        tail -40 "$log_file" || true
        return 1
    fi

    run_benchmark "$bench_file"
    cleanup_all
    sleep 3
}

run_host_dockerbe_tp2() {
    local out_dir="$1"
    local log_file="$out_dir/server.log"
    local bench_file="$out_dir/bench.txt"
    local profile_dir="$out_dir/profiles"

    rm -rf "$out_dir"
    mkdir -p "$profile_dir"
    cleanup_all
    sleep 3

    log "Starting host DockerBE full SHM TP=2 profile run"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        VLLM_DOCKER_IMAGE="$IMAGE_TAG" \
        VLLM_RPC_PROFILE_DIR="$profile_dir" \
        PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "$VLLM_CMD" serve "$MODEL" \
            --tensor-parallel-size 2 \
            --distributed-executor-backend docker \
            --port "$PORT" \
            --gpu-memory-utilization 0.5 \
            --max-model-len 512 \
            --disable-log-requests \
        > "$log_file" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "DockerBE full SHM server did not start"
        tail -60 "$log_file" || true
        return 1
    fi

    run_benchmark "$bench_file"
    cleanup_all
    sleep 3
}

main() {
    if [[ ! -f "$SHAREGPT_PATH" ]]; then
        err "ShareGPT dataset not found at $SHAREGPT_PATH"
        exit 1
    fi

    mkdir -p "$RESULTS_DIR"
    local baseline_dir="$RESULTS_DIR/baseline_tp2"
    local docker_dir="$RESULTS_DIR/dockerbe_full_shm_tp2"
    local report_file="$RESULTS_DIR/report.txt"

    log "Building Docker image $IMAGE_TAG"
    TARGET_IMAGE="$IMAGE_TAG" "$REPO_ROOT/examples/docker_executor/build-docker-executor.sh"

    run_host_mp_tp2 "$baseline_dir"
    run_host_dockerbe_tp2 "$docker_dir"

    log "Analyzing RPC profiles"
    python "$SCRIPT_DIR/analyze_rpc_profiles.py" \
        --baseline-profile-dir "$baseline_dir/profiles" \
        --docker-profile-dir "$docker_dir/profiles" \
        --baseline-bench "$baseline_dir/bench.txt" \
        --docker-bench "$docker_dir/bench.txt" \
        --output "$report_file"

    ok "Profile report saved to $report_file"
    cat "$report_file"
}

main "$@"
