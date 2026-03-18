#!/bin/bash
# Experiment 1: Backend Comparison Benchmark
#
# Compares vLLM serve performance across 6 configurations:
#   1. Host + default backend + TP=1
#   2. Host + default backend + TP=2
#   3. Docker container + default backend + TP=1
#   4. Docker container + default backend + TP=2
#   5. Host + docker executor backend + TP=1
#   6. Host + docker executor backend + TP=2
#
# Uses the first 2 GPUs (CUDA_VISIBLE_DEVICES=0,1).
# Model: Qwen/Qwen3-0.6B (small, fast loading)
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --model Qwen/Qwen3-0.6B --num-prompts 50

set -uo pipefail

# ──────────────────────────── Configuration ────────────────────────────
VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT=8000
DOCKER_IMAGE="vllm/vllm-docker-executor:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
GPU_DEVICES="0,1"

# Benchmark parameters
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN=128
RANDOM_OUTPUT_LEN=128

# Timeouts (seconds)
SERVER_STARTUP_TIMEOUT=180
BENCHMARK_TIMEOUT=180
EXPERIMENT_TIMEOUT=420   # 7 minutes max per experiment

# ──────────────────────────── Helpers ──────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

cleanup_all() {
    log "Cleaning up..."
    # Kill any leftover vllm serve processes we spawned
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Kill docker container if running
    docker rm -f vllm-bench-server 2>/dev/null || true
    # Kill docker executor worker containers
    docker ps -aq --filter "name=vllm-worker-" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    rm -rf /tmp/vllm_docker_shared 2>/dev/null || true
    SERVER_PID=""
}

trap cleanup_all EXIT INT TERM

wait_for_server() {
    local timeout=$1
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
    local label=$1
    local output_file=$2

    log "Running benchmark: $label"
    timeout "$BENCHMARK_TIMEOUT" \
        "$VLLM_CMD" bench serve \
            --backend vllm \
            --model "$MODEL" \
            --port "$PORT" \
            --endpoint /v1/completions \
            --dataset-name random \
            --num-prompts "$NUM_PROMPTS" \
            --random-input-len "$RANDOM_INPUT_LEN" \
            --random-output-len "$RANDOM_OUTPUT_LEN" \
        2>&1 | tee "$output_file"

    local rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        err "Benchmark failed (exit code $rc)"
        return 1
    fi
    ok "Benchmark complete: $output_file"
    return 0
}

# ──────────────────────── Experiment Runner ─────────────────────────────

run_experiment() {
    local exp_num=$1
    local label=$2
    local mode=$3       # "host" | "docker_container" | "host_docker_backend"
    local tp=$4

    local bench_file="$RESULTS_DIR/${exp_num}_$(echo "$label" | tr ' ' '_')_bench.txt"
    local server_log="$RESULTS_DIR/${exp_num}_$(echo "$label" | tr ' ' '_')_server.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Experiment $exp_num: $label (TP=$tp, mode=$mode)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup_all

    local exp_start=$SECONDS

    # ── Start server ──
    case "$mode" in
        host)
            log "Starting vllm serve on host (TP=$tp)..."
            CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
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
            log "Starting vllm serve inside Docker container (TP=$tp)..."
            docker run -d --rm \
                --name vllm-bench-server \
                --gpus "\"device=${GPU_DEVICES}\"" \
                --network host \
                --ipc host \
                --shm-size=8g \
                -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
                "$DOCKER_IMAGE" \
                vllm serve "$MODEL" \
                    --tensor-parallel-size "$tp" \
                    --port "$PORT" \
                    --gpu-memory-utilization 0.5 \
                    --max-model-len 512 \
                    --disable-log-requests \
                > /dev/null 2>&1
            # Follow logs in background
            docker logs -f vllm-bench-server > "$server_log" 2>&1 &
            SERVER_PID=$!
            ;;

        host_docker_backend)
            log "Starting vllm serve on host with docker backend (TP=$tp)..."
            CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
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
    esac

    # ── Wait for server ready ──
    log "Waiting for server to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    if ! wait_for_server "$SERVER_STARTUP_TIMEOUT"; then
        err "Server did not start within ${SERVER_STARTUP_TIMEOUT}s"
        echo "--- Server log (last 30 lines) ---"
        tail -30 "$server_log" 2>/dev/null || true
        echo "--- FAILED ---" > "$bench_file"
        cleanup_all
        return 1
    fi
    ok "Server is ready (took $((SECONDS - exp_start))s)"

    # ── Run benchmark ──
    if ! run_benchmark "$label" "$bench_file"; then
        err "Benchmark failed for $label"
        cleanup_all
        return 1
    fi

    local elapsed=$((SECONDS - exp_start))
    ok "Experiment $exp_num done in ${elapsed}s"

    # ── Cleanup ──
    cleanup_all
    sleep 3
    return 0
}

# ──────────────────────────── Report ───────────────────────────────────

generate_report() {
    local report="$RESULTS_DIR/report.txt"

    {
        echo "================================================================"
        echo "  Experiment 1: Backend Comparison Benchmark Report"
        echo "  Generated: $(date)"
        echo "  Model: $MODEL"
        echo "  GPUs: CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
        echo "  Benchmark: $NUM_PROMPTS prompts, ${RANDOM_INPUT_LEN} input / ${RANDOM_OUTPUT_LEN} output tokens"
        echo "================================================================"
        echo ""

        for bench_file in "$RESULTS_DIR"/*_bench.txt; do
            [[ -f "$bench_file" ]] || continue
            local name
            name=$(basename "$bench_file" _bench.txt)
            echo "── $name ──"
            # Extract the key metrics
            if grep -q "FAILED" "$bench_file" 2>/dev/null; then
                echo "  FAILED"
            else
                grep -E "(Successful requests|Benchmark duration|Request throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL)" "$bench_file" 2>/dev/null || echo "  (no metrics found)"
            fi
            echo ""
        done

        echo "================================================================"
        echo "  End of Report"
        echo "================================================================"
    } > "$report"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat "$report"
    echo ""
    ok "Report saved to: $report"
    ok "Individual results in: $RESULTS_DIR/"
}

# ──────────────────────────── Main ─────────────────────────────────────

main() {
    echo ""
    echo "================================================================"
    echo "  Experiment 1: Backend Comparison Benchmark"
    echo "================================================================"
    echo "  Model:      $MODEL"
    echo "  GPUs:       $GPU_DEVICES"
    echo "  Prompts:    $NUM_PROMPTS"
    echo "  Input len:  $RANDOM_INPUT_LEN"
    echo "  Output len: $RANDOM_OUTPUT_LEN"
    echo "  Image:      $DOCKER_IMAGE"
    echo "================================================================"
    echo ""

    mkdir -p "$RESULTS_DIR"

    local total_start=$SECONDS
    local pass=0
    local fail=0

    # Experiment 1: Host + default (mp) + TP=1
    if run_experiment 1 "host_mp_tp1" "host" 1; then
        ((pass++))
    else
        ((fail++))
    fi

    # Experiment 2: Host + default (mp) + TP=2
    if run_experiment 2 "host_mp_tp2" "host" 2; then
        ((pass++))
    else
        ((fail++))
    fi

    # Experiment 3: Docker container + default (mp) + TP=1
    if run_experiment 3 "docker_mp_tp1" "docker_container" 1; then
        ((pass++))
    else
        ((fail++))
    fi

    # Experiment 4: Docker container + default (mp) + TP=2
    if run_experiment 4 "docker_mp_tp2" "docker_container" 2; then
        ((pass++))
    else
        ((fail++))
    fi

    # Experiment 5: Host + docker backend + TP=1
    if run_experiment 5 "host_dockerbe_tp1" "host_docker_backend" 1; then
        ((pass++))
    else
        ((fail++))
    fi

    # Experiment 6: Host + docker backend + TP=2
    if run_experiment 6 "host_dockerbe_tp2" "host_docker_backend" 2; then
        ((pass++))
    else
        ((fail++))
    fi

    local total_elapsed=$((SECONDS - total_start))
    echo ""
    ok "All experiments complete: $pass passed, $fail failed (total ${total_elapsed}s)"

    generate_report
}

main "$@"
