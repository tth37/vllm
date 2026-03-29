#!/bin/bash
# Phase 2: End-to-end vLLM serving validation of CUMEM isolation.
#
# Compares two configurations on node192 (2x A100-SXM4, GPU 0,1, NVLink NV12):
#   1. Baseline Docker+MP  — single container, --distributed-executor-backend mp
#   2. DockerBE + CUMEM     — per-GPU worker containers, NCCL CUMEM P2P recovery
#
# Results are saved to results/vllm_<variant>/ as benchmark text files.
#
# Usage:
#   ./run_vllm_validation.sh                      # Full run (Qwen3-4B TP=2)
#   MODEL=Qwen/Qwen3-8B ./run_vllm_validation.sh  # Different model
#   ./run_vllm_validation.sh --build               # Rebuild Docker image first

set -euo pipefail

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

BUILD_IMAGE=0
if [[ "${1:-}" == "--build" ]]; then
    BUILD_IMAGE=1
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; }

SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    docker rm -f vllm-bench-server 2>/dev/null || true
    docker ps -aq --filter "name=vllm-worker-" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    rm -rf /tmp/vllm_docker_shared 2>/dev/null || true
    SERVER_PID=""
}

trap cleanup EXIT INT TERM

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

# ─────────────────────────────────────────────────────────────────────
# Config 1: Baseline Docker+MP (single container, standard TP)
# ─────────────────────────────────────────────────────────────────────
run_baseline() {
    local results_dir="$SCRIPT_DIR/results/vllm_baseline"
    mkdir -p "$results_dir"
    local server_log="$results_dir/server.log"
    local bench_file="$results_dir/bench.txt"

    echo ""
    log "━━━ Config 1: Baseline Docker+MP (TP=$TP_SIZE) ━━━"
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

# ─────────────────────────────────────────────────────────────────────
# Config 2: DockerBE + CUMEM Isolation (per-GPU containers, NVLink)
# ─────────────────────────────────────────────────────────────────────
run_dockerbe_cumem() {
    local results_dir="$SCRIPT_DIR/results/vllm_dockerbe_cumem"
    mkdir -p "$results_dir"
    local server_log="$results_dir/server.log"
    local bench_file="$results_dir/bench.txt"

    echo ""
    log "━━━ Config 2: DockerBE + CUMEM Isolation (TP=$TP_SIZE) ━━━"
    cleanup
    sleep 3

    # Host runs the vllm serve process; workers are spawned as Docker containers
    # with CUMEM isolation (private PID/IPC, per-GPU CUDA_VISIBLE_DEVICES).
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        VLLM_DOCKER_IMAGE="$IMAGE_TAG" \
        VLLM_DOCKER_CUMEM_ISOLATION=1 \
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

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
main() {
    echo "================================================================"
    echo "  Phase 2: vLLM Serving Validation of CUMEM Isolation"
    echo "  Model: $MODEL  TP=$TP_SIZE  GPUs: $GPU_DEVICES"
    echo "  Image: $IMAGE_TAG"
    echo "================================================================"

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

    if [[ $BUILD_IMAGE -eq 1 ]]; then
        log "Building Docker image: $IMAGE_TAG"
        TARGET_IMAGE="$IMAGE_TAG" "$REPO_ROOT/examples/docker_executor/build_docker_executor.sh"
        ok "Built: $IMAGE_TAG"
    fi

    if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
        err "Docker image not found: $IMAGE_TAG"
        err "Run with --build flag or build manually first."
        exit 1
    fi

    run_baseline
    run_dockerbe_cumem

    echo ""
    echo "================================================================"
    echo "  All validation runs complete!"
    echo "  Results: $SCRIPT_DIR/results/vllm_*/"
    echo "================================================================"

    # Generate a summary report
    local report="$SCRIPT_DIR/results/vllm_validation_report.txt"
    {
        echo "================================================================"
        echo "  Phase 2: vLLM Serving Validation Report"
        echo "  Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "  Git commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
        echo "  Model: $MODEL  TP=$TP_SIZE  GPUs: $GPU_DEVICES"
        echo "  Image: $IMAGE_TAG"
        echo "  Benchmark: $NUM_PROMPTS prompts, rate=$REQUEST_RATE, $NUM_WARMUPS warmups"
        echo "================================================================"
        echo ""
        for variant_dir in "$SCRIPT_DIR"/results/vllm_*/; do
            local bench="$variant_dir/bench.txt"
            [[ -f "$bench" ]] || continue
            echo "── $(basename "$variant_dir") ──"
            grep -E "(Successful requests|Failed requests|Benchmark duration|Request throughput|Output token throughput|Peak concurrent|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL)" "$bench" 2>/dev/null || true
            echo ""
        done
    } > "$report"
    ok "Summary report: $report"
}

main "$@"
