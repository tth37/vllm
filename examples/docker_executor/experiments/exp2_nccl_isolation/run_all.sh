#!/bin/bash
# Exp2: Run all 3 NCCL isolation benchmarks on node192 (2x A100-SXM4, NVLink)
#
# Usage:
#   ./run_all.sh [--smoke] [--rebuild]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SMOKE_TEST="false"
REBUILD=""

for arg in "$@"; do
    case $arg in
        --smoke) SMOKE_TEST="true" ;;
        --rebuild) REBUILD="--build" ;;
    esac
done

echo "=============================================================================="
echo "  Exp2: NCCL Transport Under Per-GPU Container Isolation (NVLink)"
echo "  Testbed: node192 — 2x A100-SXM4-40GB, NV12 (12x NVLink)"
echo "=============================================================================="

CONFIGS=(
    "compose.baseline.yml:Baseline (no isolation, P2P/NVLink)"
    "compose.cumem_isolation.yml:CUMEM Isolation (per-GPU + CUMEM P2P recovery)"
    "compose.shm_isolation.yml:SHM Isolation (per-GPU + shared /dev/shm)"
    "compose.p2p_isolation.yml:Naive Isolation (per-GPU, TCP fallback)"
)

for entry in "${CONFIGS[@]}"; do
    IFS=: read -r compose_file description <<< "$entry"

    echo ""
    echo "======================================================================"
    echo "  $description"
    echo "  Config: $compose_file"
    echo "======================================================================"

    SMOKE_TEST="$SMOKE_TEST" docker compose -f "$compose_file" up \
        --abort-on-container-exit $REBUILD 2>&1

    docker compose -f "$compose_file" down --volumes 2>/dev/null || true
    echo "[OK] $description — done"
done

# Analysis
echo ""
echo "======================================================================"
echo "  Generating Analysis Report"
echo "======================================================================"
python3 analyze_exp2.py --results-dir=results/

echo ""
echo "Done! Results in: $SCRIPT_DIR/results/"
