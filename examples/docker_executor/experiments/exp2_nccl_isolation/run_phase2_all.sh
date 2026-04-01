#!/bin/bash
# Run all Phase 2 vLLM serving variants sequentially.
#
# Usage:
#   ./run_phase2_all.sh             # All variants
#   ./run_phase2_all.sh baseline    # Just baseline
#   ./run_phase2_all.sh cumem       # Just CUMEM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER="${1:-all}"

run_variant() {
    local name="$1"
    local script="$2"
    if [[ "$FILTER" != "all" && "$FILTER" != "$name" ]]; then
        echo "Skipping $name (filter=$FILTER)"
        return 0
    fi
    echo ""
    echo "################################################################"
    echo "  Running: $name"
    echo "################################################################"
    bash "$SCRIPT_DIR/$script"
}

run_variant "baseline"       "run_phase2_baseline.sh"
run_variant "dockerbe_full"  "run_phase2_dockerbe_full_vis.sh"
run_variant "cumem"          "run_phase2_cumem.sh"

echo ""
echo "################################################################"
echo "  All Phase 2 variants complete!"
echo "  Results: $(cd "$SCRIPT_DIR" && pwd)/results/phase2_vllm/"
echo "################################################################"
