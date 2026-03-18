#!/bin/bash
# Profile the DockerBE TP=2 overhead by testing different configurations:
#
# Test A: Baseline (current code: NCCL_DEBUG=INFO, TCP MessageQueue)
# Test B: NCCL_DEBUG=WARN (disable NCCL info logging)
# Test C: NCCL_DEBUG=WARN + n_local_reader=world_size (shared memory MQ)
#
# The script patches docker_executor.py for each test and restores it after.

set -uo pipefail

VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"
MODEL="Qwen/Qwen3-0.6B"
PORT=8000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
EXECUTOR_PY="/home/thd/repositories/vllm-dev/vllm-source/vllm/v1/executor/docker_executor.py"
GPU_DEVICES="0,1"
NUM_PROMPTS="${NUM_PROMPTS:-50}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
    fi
    docker ps -aq --filter "name=vllm-worker-" 2>/dev/null | xargs -r docker rm -f 2>/dev/null || true
    rm -rf /tmp/vllm_docker_shared 2>/dev/null || true
    SERVER_PID=""
}

trap cleanup EXIT INT TERM

# Save original file
cp "$EXECUTOR_PY" "$EXECUTOR_PY.bak"
restore_executor() {
    cp "$EXECUTOR_PY.bak" "$EXECUTOR_PY"
    log "Restored docker_executor.py"
}

run_bench_test() {
    local label=$1
    local bench_file="$RESULTS_DIR/profile_${label}_bench.txt"
    local server_log="$RESULTS_DIR/profile_${label}_server.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Test: $label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup
    sleep 3

    local start=$SECONDS

    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
        "$VLLM_CMD" serve "$MODEL" \
            --tensor-parallel-size 2 \
            --distributed-executor-backend docker \
            --port "$PORT" \
            --gpu-memory-utilization 0.5 \
            --max-model-len 512 \
            --disable-log-requests \
        > "$server_log" 2>&1 &
    SERVER_PID=$!

    log "Waiting for server (PID=$SERVER_PID)..."
    for i in $(seq 1 90); do
        if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
            ok "Server ready in $((SECONDS - start))s"
            break
        fi
        sleep 2
    done

    if ! curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then
        err "Server failed to start"
        tail -30 "$server_log"
        echo "FAILED" > "$bench_file"
        cleanup
        return 1
    fi

    log "Running benchmark..."
    "$VLLM_CMD" bench serve \
        --backend vllm \
        --model "$MODEL" \
        --port "$PORT" \
        --endpoint /v1/completions \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len 128 \
        --random-output-len 128 \
    2>&1 | tee "$bench_file"

    ok "Test $label done in $((SECONDS - start))s"
    cleanup
    sleep 3
}

extract_metrics() {
    local file=$1
    if [[ ! -f "$file" ]] || grep -q "FAILED" "$file" 2>/dev/null; then
        echo "FAILED"
        return
    fi
    local tput med_tpot med_itl
    tput=$(grep "Output token throughput" "$file" | head -1 | awk '{print $NF}')
    med_tpot=$(grep "Median TPOT" "$file" | awk '{print $NF}')
    med_itl=$(grep "Median ITL" "$file" | awk '{print $NF}')
    echo "throughput=${tput} tok/s | Median TPOT=${med_tpot}ms | Median ITL=${med_itl}ms"
}

main() {
    mkdir -p "$RESULTS_DIR"

    echo "================================================================"
    echo "  DockerBE TP=2 Overhead Profiling"
    echo "================================================================"

    # ── Test A: Baseline (current code) ──
    log "Test A: Baseline (NCCL_DEBUG=INFO, TCP MessageQueue)"
    restore_executor
    run_bench_test "A_baseline"

    # ── Test B: NCCL_DEBUG=WARN ──
    log "Test B: Patching NCCL_DEBUG to WARN..."
    restore_executor
    sed -i 's/"NCCL_DEBUG=INFO"/"NCCL_DEBUG=WARN"/' "$EXECUTOR_PY"
    log "Patched: NCCL_DEBUG=WARN"
    run_bench_test "B_nccl_warn"

    # ── Test C: NCCL_DEBUG=WARN + SHM MessageQueue ──
    log "Test C: Patching to use shared memory MQ + NCCL_DEBUG=WARN..."
    restore_executor
    sed -i 's/"NCCL_DEBUG=INFO"/"NCCL_DEBUG=WARN"/' "$EXECUTOR_PY"
    sed -i 's/n_local_reader=0,  # No local shared memory readers - all network/n_local_reader=self.world_size,  # PROFILING: Use shared memory/' "$EXECUTOR_PY"
    log "Patched: SHM MQ + NCCL_DEBUG=WARN"
    run_bench_test "C_shm_mq"

    # Restore original
    restore_executor
    rm -f "$EXECUTOR_PY.bak"

    # ── Report ──
    echo ""
    echo "================================================================"
    echo "  Profiling Results"
    echo "================================================================"
    echo ""
    echo "Test A (baseline):    $(extract_metrics "$RESULTS_DIR/profile_A_baseline_bench.txt")"
    echo "Test B (NCCL WARN):   $(extract_metrics "$RESULTS_DIR/profile_B_nccl_warn_bench.txt")"
    echo "Test C (SHM MQ):      $(extract_metrics "$RESULTS_DIR/profile_C_shm_mq_bench.txt")"
    echo ""
    echo "================================================================"

    # Save report
    {
        echo "DockerBE TP=2 Overhead Profiling - $(date)"
        echo ""
        echo "Test A (baseline: NCCL_DEBUG=INFO + TCP MQ):  $(extract_metrics "$RESULTS_DIR/profile_A_baseline_bench.txt")"
        echo "Test B (NCCL_DEBUG=WARN + TCP MQ):            $(extract_metrics "$RESULTS_DIR/profile_B_nccl_warn_bench.txt")"
        echo "Test C (NCCL_DEBUG=WARN + SHM MQ):            $(extract_metrics "$RESULTS_DIR/profile_C_shm_mq_bench.txt")"
        echo ""
        echo "If B ≈ A: NCCL logging is not the bottleneck"
        echo "If B >> A: NCCL_DEBUG=INFO adds significant overhead"
        echo "If C >> B: SHM MQ significantly improves latency"
        echo "If C ≈ B: TCP vs SHM MQ is not the main bottleneck"
    } > "$RESULTS_DIR/profile_report.txt"

    ok "Profiling report: $RESULTS_DIR/profile_report.txt"
}

main "$@"
