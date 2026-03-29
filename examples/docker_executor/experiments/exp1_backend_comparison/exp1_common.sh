#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
EXPECTED_GPU_DEVICES="${EXPECTED_GPU_DEVICES:-1,2}"
GPU_DEVICES="${GPU_DEVICES:-$EXPECTED_GPU_DEVICES}"
ALLOW_GPU_OVERRIDE="${ALLOW_GPU_OVERRIDE:-0}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
REQUEST_RATE="${REQUEST_RATE:-10}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-300}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
SHAREGPT_PATH="${SHAREGPT_PATH:-$HOME/.cache/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
BASELINE_IMAGE="${BASELINE_IMAGE:-vllm/vllm-docker-executor:exp1-baseline}"
DOCKERBE_SYNC_OUTPUT_IMAGE="${DOCKERBE_SYNC_OUTPUT_IMAGE:-vllm/vllm-docker-executor:exp1-dockerbe_sync_output}"
DOCKERBE_HYBRID_SHM_IMAGE="${DOCKERBE_HYBRID_SHM_IMAGE:-vllm/vllm-docker-executor:exp1-dockerbe_hybrid_shm}"
DOCKERBE_IMAGE="${DOCKERBE_IMAGE:-vllm/vllm-docker-executor:exp1-dockerbe_full_shm}"
BASELINE_BRANCH_NAME="${BASELINE_BRANCH_NAME:-exp1/baseline}"
DOCKERBE_SYNC_OUTPUT_BRANCH_NAME="${DOCKERBE_SYNC_OUTPUT_BRANCH_NAME:-exp1/dockerbe_sync_output}"
DOCKERBE_HYBRID_SHM_BRANCH_NAME="${DOCKERBE_HYBRID_SHM_BRANCH_NAME:-exp1/dockerbe_hybrid_shm}"
DOCKERBE_FULL_SHM_BRANCH_NAME="${DOCKERBE_FULL_SHM_BRANCH_NAME:-exp1/dockerbe_full_shm}"
MODEL_CACHE_HINT="/home/thd/.cache/huggingface/hub/models--Qwen--Qwen3-8B"

SERVER_PID=""
LOG_FOLLOW_PID=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"
}

ok() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $*"
}

err() {
    echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

require_tools() {
    local tool=""
    for tool in curl docker git nvidia-smi; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            err "Required command not found: $tool"
            exit 1
        fi
    done

    if [[ ! -x "$VLLM_CMD" ]]; then
        err "vLLM CLI not found at $VLLM_CMD"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        err "Cannot connect to the Docker daemon"
        exit 1
    fi
}

verify_experiment_prereqs() {
    require_tools

    if [[ ! -f "$SHAREGPT_PATH" ]]; then
        log "ShareGPT dataset not found, downloading..."
        SHAREGPT_PATH="$("$REPO_ROOT/examples/docker_executor/download_sharegpt.sh")"
        if [[ ! -f "$SHAREGPT_PATH" ]]; then
            err "Failed to download ShareGPT dataset"
            exit 1
        fi
    fi

    if [[ ! -d "$MODEL_CACHE_HINT" ]]; then
        warn "Model cache hint not found at $MODEL_CACHE_HINT"
    fi
}

enforce_gpu_policy() {
    if [[ "$GPU_DEVICES" != "$EXPECTED_GPU_DEVICES" && "$ALLOW_GPU_OVERRIDE" != "1" ]]; then
        err "Refusing to run with GPU_DEVICES=$GPU_DEVICES"
        err "Expected the standard pair $EXPECTED_GPU_DEVICES."
        err "Set ALLOW_GPU_OVERRIDE=1 to bypass this safeguard."
        exit 1
    fi

    if [[ "$GPU_DEVICES" != "$EXPECTED_GPU_DEVICES" ]]; then
        warn "Using overridden GPU pair: $GPU_DEVICES"
    else
        ok "Using standardized GPU pair: $GPU_DEVICES"
    fi
}

check_selected_gpus_idle() {
    local -A selected_uuid=()
    local requested=""
    local index=""
    local uuid=""
    local busy=0

    while IFS=',' read -r index uuid; do
        index="$(trim "$index")"
        uuid="$(trim "$uuid")"
        [[ -z "$index" || -z "$uuid" ]] && continue
        for requested in ${GPU_DEVICES//,/ }; do
            if [[ "$index" == "$requested" ]]; then
                selected_uuid["$uuid"]="$index"
            fi
        done
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)

    if [[ ${#selected_uuid[@]} -eq 0 ]]; then
        warn "Could not map selected GPUs to UUIDs; skipping idle check"
        return 0
    fi

    while IFS=',' read -r uuid pid process_name used_mem; do
        uuid="$(trim "$uuid")"
        pid="$(trim "$pid")"
        process_name="$(trim "$process_name")"
        used_mem="$(trim "$used_mem")"
        [[ -z "$uuid" || -z "${selected_uuid[$uuid]:-}" ]] && continue

        if [[ $busy -eq 0 ]]; then
            err "Selected GPUs are not idle:"
        fi
        busy=1
        echo "  GPU ${selected_uuid[$uuid]}: PID $pid ($process_name, ${used_mem} MiB)" >&2
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)

    return $busy
}

prepare_results_dir() {
    local results_dir="$1"
    mkdir -p "$results_dir"
    find "$results_dir" -maxdepth 1 -type f \
        \( -name '*_bench.txt' -o -name '*_server.log' -o -name 'report.txt' \) \
        -delete
}

cleanup_runtime() {
    if [[ -n "${LOG_FOLLOW_PID:-}" ]] && kill -0 "$LOG_FOLLOW_PID" 2>/dev/null; then
        kill "$LOG_FOLLOW_PID" 2>/dev/null || true
        wait "$LOG_FOLLOW_PID" 2>/dev/null || true
    fi

    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    docker rm -f vllm-exp1-baseline vllm-bench-server 2>/dev/null || true
    docker ps -aq --filter "name=vllm-worker-" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    rm -rf /tmp/vllm_docker_shared 2>/dev/null || true

    SERVER_PID=""
    LOG_FOLLOW_PID=""
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

run_sharegpt_benchmark() {
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

require_docker_image() {
    local image_tag="$1"
    if ! docker image inspect "$image_tag" >/dev/null 2>&1; then
        err "Docker image not found: $image_tag"
        err "Run build_exp1_images.sh before benchmarking."
        exit 1
    fi
}

metric_value() {
    local key="$1"
    local bench_file="$2"
    grep -F "$key" "$bench_file" 2>/dev/null | head -n 1 | awk -F: '{print $2}' | xargs
}

generate_results_report() {
    local results_dir="$1"
    local title="$2"
    local run_note="$3"
    local report="$results_dir/report.txt"
    local bench_file=""
    local variant_branch="${EXPERIMENT_BRANCH_NAME:-}"
    local variant_image="${EXPERIMENT_IMAGE_TAG:-}"
    local optimization_set="${EXPERIMENT_OPTIMIZATIONS:-}"

    {
        echo "================================================================"
        echo "  $title"
        echo "  Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "  Git branch: $(git -C "$REPO_ROOT" branch --show-current)"
        echo "  Git commit: $(git -C "$REPO_ROOT" rev-parse --short HEAD)"
        echo "  Model: $MODEL"
        echo "  GPUs: CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
        echo "  Dataset: ShareGPT first $NUM_PROMPTS prompts ($SHAREGPT_PATH)"
        echo "  Benchmark: rate=$REQUEST_RATE req/s, $NUM_WARMUPS warmups, --ignore-eos"
        if [[ -n "$variant_branch" ]]; then
            echo "  Experiment branch: $variant_branch"
        fi
        if [[ -n "$variant_image" ]]; then
            echo "  Docker image: $variant_image"
        fi
        echo "================================================================"
        echo ""
        echo "Run note:"
        echo "  $run_note"
        if [[ -n "$optimization_set" ]]; then
            echo ""
            echo "Optimization set:"
            echo "  $optimization_set"
        fi
        echo ""
        for bench_file in "$results_dir"/*_bench.txt; do
            [[ -f "$bench_file" ]] || continue
            echo "── $(basename "$bench_file" _bench.txt) ──"
            grep -E "(Successful requests|Failed requests|Benchmark duration|Request throughput|Output token throughput|Peak concurrent requests|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL)" "$bench_file" 2>/dev/null || true
            echo ""
        done
    } > "$report"

    ok "Report saved to: $report"
}
