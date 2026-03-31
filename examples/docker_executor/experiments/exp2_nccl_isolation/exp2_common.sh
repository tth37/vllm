#!/bin/bash
# Common configuration and utilities for exp2 Phase 2 vLLM serving experiments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
REQUEST_RATE="${REQUEST_RATE:-10}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-300}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
SHAREGPT_PATH="${SHAREGPT_PATH:-$HOME/.cache/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
IMAGE_TAG="${IMAGE_TAG:-vllm/vllm-docker-executor:exp2-cumem}"
PHASE2_RESULTS_DIR="$SCRIPT_DIR/results/phase2_vllm"

SERVER_PID=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; }

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    docker rm -f vllm-bench-server 2>/dev/null || true
    docker ps -aq --filter "name=vllm-worker-" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    docker volume rm -f vllm-cumem-shm 2>/dev/null || true
    rm -rf /tmp/vllm_docker_shared 2>/dev/null || true
    SERVER_PID=""
}

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
    log "Running benchmark ($NUM_PROMPTS prompts, rate=$REQUEST_RATE req/s, $NUM_WARMUPS warmups)"
    timeout 1800 "$VLLM_CMD" bench serve \
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

verify_prereqs() {
    if [[ ! -x "$VLLM_CMD" ]]; then
        err "vLLM CLI not found at $VLLM_CMD"
        exit 1
    fi

    if [[ ! -f "$SHAREGPT_PATH" ]]; then
        log "ShareGPT dataset not found, downloading..."
        SHAREGPT_PATH="$("$REPO_ROOT/examples/docker_executor/download_sharegpt.sh")"
        if [[ ! -f "$SHAREGPT_PATH" ]]; then
            err "Failed to download ShareGPT dataset"
            exit 1
        fi
    fi

    if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
        err "Docker image not found: $IMAGE_TAG"
        err "Run build_docker_executor.sh first."
        exit 1
    fi
}

print_header() {
    local variant_name="$1"
    echo ""
    echo "================================================================"
    echo "  Phase 2: $variant_name"
    echo "  Model: $MODEL  TP=$TP_SIZE  GPUs: $GPU_DEVICES"
    echo "  Image: $IMAGE_TAG"
    echo "================================================================"
}
