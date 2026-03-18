#!/bin/bash
# Build the Docker images used by the active exp1 comparison.
# Both tags point at the same cleaned source tree on main.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./exp1_common.sh
source "$SCRIPT_DIR/exp1_common.sh"

PRIMARY_IMAGE="${PRIMARY_IMAGE:-$BASELINE_IMAGE}"
SECONDARY_IMAGE="${SECONDARY_IMAGE:-$DOCKERBE_IMAGE}"

main() {
    verify_experiment_prereqs

    log "Building primary exp1 image: $PRIMARY_IMAGE"
    TARGET_IMAGE="$PRIMARY_IMAGE" "$REPO_ROOT/examples/docker_executor/build-docker-executor.sh"

    if [[ "$SECONDARY_IMAGE" != "$PRIMARY_IMAGE" ]]; then
        log "Tagging $PRIMARY_IMAGE as $SECONDARY_IMAGE"
        docker tag "$PRIMARY_IMAGE" "$SECONDARY_IMAGE"
    fi

    ok "Active exp1 image tags are ready:"
    echo "  - $PRIMARY_IMAGE"
    echo "  - $SECONDARY_IMAGE"
}

main "$@"
