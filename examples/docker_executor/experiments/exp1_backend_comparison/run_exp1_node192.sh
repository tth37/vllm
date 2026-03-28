#!/bin/bash
# Run full exp1 comparison on node192 (2x A100-SXM4 NVLink, GPU 0,1).
# Sweeps 3 models: Qwen3-4B, 8B, 14B with TP=1 and TP=2.
# Results are stored in node192/<model>/<variant>/ to keep separate from node196.
#
# For 14B with TP=1, GPU_MEMORY_UTILIZATION is raised to 0.9 to fit the model.
#
# Usage:
#   ./run_exp1_node192.sh              # Run all models
#   MODEL_FILTER=qwen3_4b ./run_exp1_node192.sh  # Run only 4B

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp1_common.sh"

# Node192-specific overrides
export EXPECTED_GPU_DEVICES="0,1"
export GPU_DEVICES="0,1"

NODE_LABEL="node192"
MODEL_FILTER="${MODEL_FILTER:-}"

declare -A MODEL_MAP=(
    [qwen3_4b]="Qwen/Qwen3-4B"
    [qwen3_8b]="Qwen/Qwen3-8B"
    [qwen3_14b]="Qwen/Qwen3-14B"
)
MODEL_ORDER=(qwen3_4b qwen3_8b qwen3_14b)

VARIANT_SCRIPTS=(
    "baseline|run_baseline_docker_mp.sh"
    "dockerbe_sync_output|run_dockerbe_sync_output.sh"
    "dockerbe_hybrid_shm|run_dockerbe_hybrid_shm.sh"
    "dockerbe_full_shm|run_dockerbe_full_shm.sh"
)

main() {
    require_tools
    enforce_gpu_policy

    for model_key in "${MODEL_ORDER[@]}"; do
        if [[ -n "$MODEL_FILTER" && "$MODEL_FILTER" != "$model_key" ]]; then
            continue
        fi

        local model="${MODEL_MAP[$model_key]}"
        export MODEL="$model"
        export MODEL_CACHE_HINT="$HOME/.cache/huggingface/hub/models--${model//\//--}"

        echo ""
        echo "################################################################"
        echo "  Model: $model ($model_key) on $NODE_LABEL"
        echo "################################################################"

        for entry in "${VARIANT_SCRIPTS[@]}"; do
            local variant_name="${entry%%|*}"
            local script_name="${entry##*|}"

            local results_dir="$SCRIPT_DIR/${NODE_LABEL}/${model_key}/${variant_name}"
            export RESULTS_DIR="$results_dir"

            echo ""
            echo "================================================================"
            echo "  $model | $variant_name | $NODE_LABEL"
            echo "  Results: $results_dir"
            echo "================================================================"

            # For 14B with TP=1, we need more GPU memory
            if [[ "$model_key" == "qwen3_14b" ]]; then
                export GPU_MEMORY_UTILIZATION="0.9"
            else
                export GPU_MEMORY_UTILIZATION="0.5"
            fi

            bash "$SCRIPT_DIR/$script_name"
        done
    done

    echo ""
    echo "################################################################"
    echo "  All $NODE_LABEL experiments complete!"
    echo "  Results: $SCRIPT_DIR/$NODE_LABEL/"
    echo "################################################################"
}

main "$@"
