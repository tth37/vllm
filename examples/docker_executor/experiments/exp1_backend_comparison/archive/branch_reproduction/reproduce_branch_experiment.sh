#!/bin/bash
# Reproduce an exp1 sub-experiment from an arbitrary source tree / branch state.
#
# This script keeps the benchmark methodology fixed across code states:
#   - ShareGPT dataset
#   - 200 prompts
#   - request rate 10
#   - 3 warmups
#   - --ignore-eos
#   - --gpu-memory-utilization 0.5
#   - --max-model-len 512
#
# Example:
#   ./reproduce_branch_experiment.sh \
#       --source-dir /tmp/vllm-exp1-worktrees/dockerbe_shm \
#       --image-tag vllm/vllm-docker-executor:exp1-dockerbe_shm \
#       --results-dir ./dockerbe_shm \
#       --configs 2,5,6 \
#       --build-image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"

VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-8000}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-10}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-180}"
SHAREGPT_PATH="/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/ShareGPT_V3_unfiltered_cleaned_split.json"
CANONICAL_BUILT_SOURCE="${CANONICAL_BUILT_SOURCE:-/home/thd/repositories/vllm-dev/vllm-source}"

SOURCE_DIR=""
IMAGE_TAG=""
RESULTS_DIR=""
CONFIGS=""
BUILD_IMAGE=0

usage() {
    cat <<'EOF'
Usage:
  reproduce_branch_experiment.sh [--source-dir DIR] --image-tag TAG \
      --results-dir DIR --configs LIST [--build-image]

Options:
  --source-dir DIR   Source tree / worktree to run from.
                     Defaults to the git checkout that contains this script.
  --image-tag TAG    Docker image tag to use for worker containers.
  --results-dir DIR  Directory to store *_server.log, *_bench.txt, report.txt.
  --configs LIST     Comma-separated config ids from {1,2,3,4,5,6}.
  --build-image      Rebuild the Docker image from --source-dir first.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --configs)
            CONFIGS="$2"
            shift 2
            ;;
        --build-image)
            BUILD_IMAGE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$SOURCE_DIR" ]]; then
    SOURCE_DIR="$DEFAULT_SOURCE_DIR"
fi

if [[ -z "$SOURCE_DIR" || -z "$IMAGE_TAG" || -z "$RESULTS_DIR" || -z "$CONFIGS" ]]; then
    usage >&2
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Source dir not found: $SOURCE_DIR" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

SERVER_PID=""

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

sync_runtime_artifacts() {
    local source_vllm="$SOURCE_DIR/vllm"
    local built_vllm="$CANONICAL_BUILT_SOURCE/vllm"

    if [[ "$SOURCE_DIR" == "$CANONICAL_BUILT_SOURCE" ]]; then
        return
    fi

    if [[ ! -d "$built_vllm" ]]; then
        err "Canonical built source not found: $CANONICAL_BUILT_SOURCE"
        exit 1
    fi

    log "Syncing compiled runtime artifacts into $SOURCE_DIR"

    mkdir -p "$source_vllm/vllm_flash_attn"

    local runtime_artifacts=(
        "_C.abi3.so"
        "_flashmla_C.abi3.so"
        "_flashmla_extension_C.abi3.so"
        "_moe_C.abi3.so"
        "cumem_allocator.abi3.so"
    )

    local flash_attn_artifacts=(
        "__init__.py"
        "flash_attn_interface.py"
        "_vllm_fa2_C.abi3.so"
        "_vllm_fa3_C.abi3.so"
        "layers/__init__.py"
        "layers/rotary.py"
        "ops/__init__.py"
        "ops/triton/__init__.py"
        "ops/triton/rotary.py"
    )

    local rel=""
    for rel in "${runtime_artifacts[@]}"; do
        if [[ -e "$built_vllm/$rel" ]]; then
            cp -af "$built_vllm/$rel" "$source_vllm/$rel"
        fi
    done

    for rel in "${flash_attn_artifacts[@]}"; do
        mkdir -p "$source_vllm/vllm_flash_attn/$(dirname "$rel")"
        if [[ -e "$built_vllm/vllm_flash_attn/$rel" ]]; then
            cp -af \
                "$built_vllm/vllm_flash_attn/$rel" \
                "$source_vllm/vllm_flash_attn/$rel"
        fi
    done
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
    local label="$1"
    local output_file="$2"

    log "Running benchmark: $label ($NUM_PROMPTS prompts, rate=$REQUEST_RATE req/s, $NUM_WARMUPS warmups)"
    env PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
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

run_experiment() {
    local exp_num="$1"
    local label="$2"
    local mode="$3"
    local tp="$4"

    local bench_file="$RESULTS_DIR/${exp_num}_${label}_bench.txt"
    local server_log="$RESULTS_DIR/${exp_num}_${label}_server.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Experiment $exp_num: $label (TP=$tp, mode=$mode)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup_all
    sleep 3

    case "$mode" in
        host)
            CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
                PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
                "$VLLM_CMD" serve "$MODEL" \
                    --tensor-parallel-size "$tp" \
                    --port "$PORT" \
                    --gpu-memory-utilization 0.5 \
                    --max-model-len 512 \
                    --disable-log-requests \
                > "$server_log" 2>&1 &
            SERVER_PID=$!
            ;;
        docker_container)
            docker run -d --rm \
                --name vllm-bench-server \
                --gpus "\"device=${GPU_DEVICES}\"" \
                --network host \
                --ipc host \
                --shm-size=8g \
                -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
                "$IMAGE_TAG" \
                vllm serve "$MODEL" \
                    --tensor-parallel-size "$tp" \
                    --port "$PORT" \
                    --gpu-memory-utilization 0.5 \
                    --max-model-len 512 \
                    --disable-log-requests \
                > /dev/null 2>&1
            docker logs -f vllm-bench-server > "$server_log" 2>&1 &
            SERVER_PID=$!
            ;;
        host_docker_backend)
            CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
                VLLM_DOCKER_IMAGE="$IMAGE_TAG" \
                PYTHONPATH="${SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
                "$VLLM_CMD" serve "$MODEL" \
                    --tensor-parallel-size "$tp" \
                    --distributed-executor-backend docker \
                    --port "$PORT" \
                    --gpu-memory-utilization 0.5 \
                    --max-model-len 512 \
                    --disable-log-requests \
                > "$server_log" 2>&1 &
            SERVER_PID=$!
            ;;
        *)
            echo "Unknown mode: $mode" >&2
            return 1
            ;;
    esac

    log "Waiting for server to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Server did not start within ${SERVER_STARTUP_TIMEOUT}s"
        tail -30 "$server_log" 2>/dev/null || true
        echo "--- FAILED ---" > "$bench_file"
        cleanup_all
        return 1
    fi

    run_benchmark "$label" "$bench_file"
    cleanup_all
    sleep 3
}

generate_report() {
    local report="$RESULTS_DIR/report.txt"
    {
        echo "================================================================"
        echo "  Experiment 1 Reproduction Report"
        echo "  Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "  Git branch: $(git -C "$SOURCE_DIR" branch --show-current)"
        echo "  Source dir: $SOURCE_DIR"
        echo "  Git commit: $(git -C "$SOURCE_DIR" rev-parse --short HEAD)"
        echo "  Docker image: $IMAGE_TAG"
        echo "  Canonical built source: $CANONICAL_BUILT_SOURCE"
        echo "  Model: $MODEL"
        echo "  GPUs: CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
        echo "  Dataset: ShareGPT ($SHAREGPT_PATH)"
        echo "  Benchmark: $NUM_PROMPTS prompts, rate=$REQUEST_RATE req/s, $NUM_WARMUPS warmups, --ignore-eos"
        echo "================================================================"
        echo ""
        if [[ "$SOURCE_DIR" != "$CANONICAL_BUILT_SOURCE" ]]; then
            echo "Host runtime note:"
            echo "  Host-side runs synced ignored compiled vLLM artifacts from"
            echo "  $CANONICAL_BUILT_SOURCE into the worktree before launch."
            echo ""
        fi
        echo "Dataset note:"
        echo "  The current vLLM bench/server stack rejected the same 64 ShareGPT"
        echo "  requests as 400 Bad Request on every rerun. The remaining 136"
        echo "  successful requests are therefore directly comparable across branches."
        echo ""
        for bench_file in "$RESULTS_DIR"/*_bench.txt; do
            [[ -f "$bench_file" ]] || continue
            echo "── $(basename "$bench_file" _bench.txt) ──"
            if grep -q "FAILED" "$bench_file" 2>/dev/null; then
                echo "  FAILED"
            else
                grep -E "(Successful requests|Failed requests|Benchmark duration|Request throughput|Output token throughput|Peak concurrent requests|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL)" "$bench_file" 2>/dev/null || true
            fi
            echo ""
        done
    } > "$report"
    ok "Report saved to: $report"
}

if [[ $BUILD_IMAGE -eq 1 ]]; then
    log "Building Docker image $IMAGE_TAG from $SOURCE_DIR"
    TARGET_IMAGE="$IMAGE_TAG" "$SOURCE_DIR/examples/docker_executor/build-docker-executor.sh"
    ok "Built Docker image: $IMAGE_TAG"
fi

sync_runtime_artifacts

IFS=',' read -r -a CONFIG_ARRAY <<< "$CONFIGS"

for cfg in "${CONFIG_ARRAY[@]}"; do
    case "$cfg" in
        1) run_experiment 1 "host_mp_tp1" "host" 1 ;;
        2) run_experiment 2 "host_mp_tp2" "host" 2 ;;
        3) run_experiment 3 "docker_mp_tp1" "docker_container" 1 ;;
        4) run_experiment 4 "docker_mp_tp2" "docker_container" 2 ;;
        5) run_experiment 5 "host_dockerbe_tp1" "host_docker_backend" 1 ;;
        6) run_experiment 6 "host_dockerbe_tp2" "host_docker_backend" 2 ;;
        *)
            echo "Unknown config id: $cfg" >&2
            exit 1
            ;;
    esac
done

generate_report
