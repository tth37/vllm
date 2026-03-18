#!/bin/bash
# Benchmark: DockerBE with SHM broadcast MQ fix
# Runs configs 5 (TP=1) and 6 (TP=2) and also Host+MP TP=2 as baseline

set -uo pipefail

VENV_DIR="/home/thd/repositories/vllm-dev/.venv"
VLLM_CMD="$VENV_DIR/bin/vllm"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT=8000
DOCKER_IMAGE="vllm/vllm-docker-executor:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_v4_shm"
GPU_DEVICES="0,1"

SHAREGPT_PATH="/home/thd/repositories/vllm-dev/vllm-source/examples/docker_executor/ShareGPT_V3_unfiltered_cleaned_split.json"

mkdir -p "$RESULTS_DIR"

source "$VENV_DIR/bin/activate"

cleanup() {
    pkill -f "vllm serve.*$PORT" 2>/dev/null || true
    docker rm -f vllm-worker-0 vllm-worker-1 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local timeout=${1:-120}
    for i in $(seq 1 $timeout); do
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "TIMEOUT waiting for server"
    return 1
}

run_bench() {
    local label="$1"
    local output="$RESULTS_DIR/${label}_bench.txt"
    echo "=== Benchmarking: $label ==="
    timeout 300 $VLLM_CMD bench serve \
        --backend vllm \
        --model "$MODEL" \
        --port "$PORT" \
        --endpoint /v1/completions \
        --dataset-name sharegpt \
        --dataset-path "$SHAREGPT_PATH" \
        --num-prompts 200 \
        --request-rate 10 \
        --num-warmups 3 \
        2>&1 | tee "$output"
    echo ""
}

trap cleanup EXIT

# === Baseline: Host + MP + TP=2 ===
echo "=========================================="
echo " Config 2: Host + MP + TP=2 (baseline)"
echo "=========================================="
cleanup
CUDA_VISIBLE_DEVICES=$GPU_DEVICES $VLLM_CMD serve "$MODEL" \
    --max-model-len 512 --tensor-parallel-size 2 --port $PORT \
    > "$RESULTS_DIR/2_host_mp_tp2_server.log" 2>&1 &
SERVER_PID=$!
if wait_for_server 120; then
    run_bench "2_host_mp_tp2"
fi
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
cleanup

# === Config 5: Host + DockerBE + TP=1 ===
echo "=========================================="
echo " Config 5: Host + DockerBE + TP=1"
echo "=========================================="
cleanup
CUDA_VISIBLE_DEVICES=$GPU_DEVICES VLLM_DOCKER_IMAGE=$DOCKER_IMAGE \
    $VLLM_CMD serve "$MODEL" \
    --distributed-executor-backend docker \
    --max-model-len 512 --tensor-parallel-size 1 --port $PORT \
    > "$RESULTS_DIR/5_host_dockerbe_tp1_server.log" 2>&1 &
SERVER_PID=$!
if wait_for_server 120; then
    run_bench "5_host_dockerbe_tp1"
fi
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
cleanup

# === Config 6: Host + DockerBE + TP=2 ===
echo "=========================================="
echo " Config 6: Host + DockerBE + TP=2"
echo "=========================================="
cleanup
CUDA_VISIBLE_DEVICES=$GPU_DEVICES VLLM_DOCKER_IMAGE=$DOCKER_IMAGE \
    $VLLM_CMD serve "$MODEL" \
    --distributed-executor-backend docker \
    --max-model-len 512 --tensor-parallel-size 2 --port $PORT \
    > "$RESULTS_DIR/6_host_dockerbe_tp2_server.log" 2>&1 &
SERVER_PID=$!
if wait_for_server 120; then
    run_bench "6_host_dockerbe_tp2"
fi
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
cleanup

echo ""
echo "=========================================="
echo " Results saved to: $RESULTS_DIR"
echo "=========================================="
