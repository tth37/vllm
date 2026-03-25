#!/bin/bash
# Phase 2: SHM Transport with Shared Volumes
# Tests forced SHM transport and SHM+P2P combo configurations.
#
# Usage:
#   ./run_phase2.sh [--smoke] [--rebuild] [--config=NAME]

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

ALL_CONFIGS=(shm_forced shm_p2p_combo)

if [ -n "$ONLY_CONFIG" ]; then
    CONFIGS=("$ONLY_CONFIG")
else
    CONFIGS=("${ALL_CONFIGS[@]}")
fi

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
    echo "Using existing gpu-comm-benchmark:latest image"
fi

echo ""
echo "=============================================================================="
echo "  Phase 2: SHM Transport with Shared Volumes"
echo "  Configs: ${CONFIGS[*]}"
echo "  Mode: $([ "$SMOKE" = "true" ] && echo "SMOKE TEST" || echo "FULL BENCHMARK")"
echo "=============================================================================="

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
    echo "======================================================================"

    docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true

    LOG_FILE="$RESULTS_DIR/${config}_log.txt"

    if [ "$SMOKE" = "true" ]; then
        SMOKE_TEST=true docker compose -f "$COMPOSE_FILE" up \
            --abort-on-container-exit 2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    else
        docker compose -f "$COMPOSE_FILE" up \
            --abort-on-container-exit 2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}
    fi

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
    else
        STATUS_MAP[$config]="FAILED (exit $EXIT_CODE)"
    fi

    echo "  => Status: ${STATUS_MAP[$config]}, Transport: $TRANSPORT"
    docker compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
done

echo ""
echo "=============================================================================="
echo "  Phase 2 Results Summary"
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
