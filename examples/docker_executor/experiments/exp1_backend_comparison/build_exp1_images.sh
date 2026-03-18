#!/bin/bash
# Build the Docker images used by the active exp1 comparison.
# Both tags point at the same cleaned source tree on main.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

PRIMARY_IMAGE="${PRIMARY_IMAGE:-$BASELINE_IMAGE}"
IMAGE_TAGS=(
    "$BASELINE_IMAGE"
    "$DOCKERBE_SYNC_OUTPUT_IMAGE"
    "$DOCKERBE_HYBRID_SHM_IMAGE"
    "$DOCKERBE_IMAGE"
)

main() {
    verify_experiment_prereqs

    log "Building primary exp1 image: $PRIMARY_IMAGE"
    TARGET_IMAGE="$PRIMARY_IMAGE" "$REPO_ROOT/examples/docker_executor/build-docker-executor.sh"

    local image_tag=""
    for image_tag in "${IMAGE_TAGS[@]}"; do
        if [[ "$image_tag" == "$PRIMARY_IMAGE" ]]; then
            continue
        fi
        log "Tagging $PRIMARY_IMAGE as $image_tag"
        docker tag "$PRIMARY_IMAGE" "$image_tag"
    done

    ok "Active exp1 image tags are ready:"
    for image_tag in "${IMAGE_TAGS[@]}"; do
        echo "  - $image_tag"
    done
}

main "$@"
