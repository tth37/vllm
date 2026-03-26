#!/bin/bash
# Phase 1: Namespace Isolation Matrix
# Runs all 6 configs sequentially, captures NCCL transport selection and benchmark results.
#
# Usage:
#   ./run_matrix.sh [--smoke] [--rebuild] [--config=NAME]
#
# Options:
#   --smoke     Quick smoke test (1 all-reduce) instead of full benchmark
#   --rebuild   Force rebuild of Docker image
#   --config=X  Run only config X (baseline, private_ipc, private_net, private_both, single_gpu, shared_shm)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP2_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$EXP2_DIR/results"

SMOKE=false
REBUILD=false
ONLY_CONFIG=""

for arg in "$@"; do
    case $arg in
        --smoke) SMOKE=true ;;
        --rebuild) REBUILD=true ;;
        --config=*) ONLY_CONFIG="${arg#*=}" ;;
    esac
done

# All Phase 1 configs in order
ALL_CONFIGS=(baseline private_ipc private_net private_both single_gpu shared_shm)

# Filter to single config if specified
if [ -n "$ONLY_CONFIG" ]; then
    CONFIGS=("$ONLY_CONFIG")
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Build Docker image
cd "$EXP2_DIR"
if [ "$REBUILD" = "true" ]; then
    echo "Building Docker image..."
    docker build -t gpu-comm-benchmark:latest .
elif ! docker image inspect gpu-comm-benchmark:latest &>/dev/null; then
    echo "Building Docker image (first time)..."
    docker build -t gpu-comm-benchmark:latest .
else
    echo "Using existing gpu-comm-benchmark:latest image (use --rebuild to force)"
fi

echo ""
echo "=============================================================================="
echo "  Phase 1: Namespace Isolation Matrix"
echo "  Configs to run: ${CONFIGS[*]}"
echo "  Mode: $([ "$SMOKE" = "true" ] && echo "SMOKE TEST" || echo "FULL BENCHMARK")"
echo "=============================================================================="

# Track results
declare -A TRANSPORT_MAP
declare -A STATUS_MAP

for config in "${CONFIGS[@]}"; do
    COMPOSE_FILE="$SCRIPT_DIR/compose.${config}.yml"

    if [ ! -f "$COMPOSE_FILE" ]; then
        echo "ERROR: Compose file not found: $COMPOSE_FILE"
        STATUS_MAP[$config]="MISSING"
        continue
    fi

    echo ""
    echo "======================================================================"
    echo "  Config: $config"
    echo "  Compose: compose.${config}.yml"
    echo "======================================================================"

    # Clean up any leftover containers from previous runs
    docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true

    # Run benchmark, capture both stdout and stderr (NCCL debug goes to stderr)
    LOG_FILE="$RESULTS_DIR/${config}_log.txt"

    if [ "$SMOKE" = "true" ]; then
        set +e
        SMOKE_TEST=true docker compose -f "$COMPOSE_FILE" up \
            --abort-on-container-exit 2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e
    else
        set +e
        docker compose -f "$COMPOSE_FILE" up \
            --abort-on-container-exit 2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
        set -e
    fi

    # Extract NCCL transport from logs
    TRANSPORT="UNKNOWN"
    if grep -q "via P2P" "$LOG_FILE" 2>/dev/null; then
        TRANSPORT="P2P"
    elif grep -q "via SHM" "$LOG_FILE" 2>/dev/null; then
        TRANSPORT="SHM"
    elif grep -q "via NET" "$LOG_FILE" 2>/dev/null; then
        TRANSPORT="NET/Socket"
    fi

    TRANSPORT_MAP[$config]="$TRANSPORT"

    if [ "$EXIT_CODE" -eq 0 ]; then
        STATUS_MAP[$config]="OK"
        echo "  => Status: OK, Transport: $TRANSPORT"
    else
        STATUS_MAP[$config]="FAILED (exit $EXIT_CODE)"
        echo "  => Status: FAILED (exit $EXIT_CODE), Transport: $TRANSPORT"
    fi

    # Cleanup
    docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true

    echo ""
done

# Summary table
echo ""
echo "=============================================================================="
echo "  Phase 1 Results Summary"
echo "=============================================================================="
printf "%-20s %-12s %-10s\n" "Config" "Transport" "Status"
printf "%-20s %-12s %-10s\n" "------" "---------" "------"
for config in "${CONFIGS[@]}"; do
    printf "%-20s %-12s %-10s\n" \
        "$config" \
        "${TRANSPORT_MAP[$config]:-N/A}" \
        "${STATUS_MAP[$config]:-N/A}"
done
echo "=============================================================================="
echo ""
echo "Detailed logs: $RESULTS_DIR/*_log.txt"
echo "Benchmark JSON: $RESULTS_DIR/*_results.json"
echo ""
echo "Next: run analyze_exp2.py to compare bandwidth numbers across configs."
