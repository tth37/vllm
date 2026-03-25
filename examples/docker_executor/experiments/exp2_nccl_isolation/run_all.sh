#!/bin/bash
# Exp2 Master Orchestrator
# Runs Phase 1 and Phase 2 sequentially, then generates analysis report.
# Phase 3 (NCCL patching) is manual — only run if Phases 1-2 are insufficient.
#
# Usage:
#   ./run_all.sh [--smoke] [--rebuild] [--phase=1|2|3]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SMOKE=""
REBUILD=""
ONLY_PHASE=""

for arg in "$@"; do
    case $arg in
        --smoke) SMOKE="--smoke" ;;
        --rebuild) REBUILD="--rebuild" ;;
        --phase=*) ONLY_PHASE="${arg#*=}" ;;
    esac
done

echo "=============================================================================="
echo "  Exp2: NCCL High-Bandwidth Communication with Per-GPU Container Isolation"
echo "=============================================================================="
echo ""

run_phase() {
    local phase=$1
    local script=$2

    echo ""
    echo "======================================================================"
    echo "  Running Phase $phase"
    echo "======================================================================"

    chmod +x "$script"
    bash "$script" $SMOKE $REBUILD
}

# Phase 1: Namespace Isolation Matrix
if [ -z "$ONLY_PHASE" ] || [ "$ONLY_PHASE" = "1" ]; then
    run_phase 1 "$SCRIPT_DIR/phase1_isolation_matrix/run_matrix.sh"
fi

# Phase 2: SHM Transport
if [ -z "$ONLY_PHASE" ] || [ "$ONLY_PHASE" = "2" ]; then
    run_phase 2 "$SCRIPT_DIR/phase2_shm_transport/run_phase2.sh"
fi

# Phase 3: manual (NCCL patching)
if [ "$ONLY_PHASE" = "3" ]; then
    echo ""
    echo "Phase 3 (NCCL patching) requires manual steps:"
    echo "  1. cd phase3_nccl_patch/"
    echo "  2. docker build -f Dockerfile.nccl-dev -t nccl-dev:latest ."
    echo "  3. Add patches to patches/"
    echo "  4. Run: docker compose -f compose.patched_nccl.yml up --abort-on-container-exit"
    echo ""
    exit 0
fi

# Analysis
echo ""
echo "======================================================================"
echo "  Generating Analysis Report"
echo "======================================================================"
cd "$SCRIPT_DIR"
python3 analyze_exp2.py --results-dir=results/

echo ""
echo "Done! Results in: $SCRIPT_DIR/results/"
