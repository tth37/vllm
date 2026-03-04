#!/bin/bash
# Build script for vLLM Docker executor image
# This builds a custom Docker image from local vLLM source code using uv.
#
# Usage:
#   ./build-docker-executor.sh                    # Build with default settings
#   TARGET_IMAGE=my-image:latest ./build-docker-executor.sh  # Custom image name
#   ./build-docker-executor.sh --help             # Show help

set -e

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [--help]"
            echo ""
            echo "Builds a Docker image for vLLM DockerDistributedExecutor from local source."
            echo ""
            echo "Options:"
            echo "  --help    Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  TARGET_IMAGE       Target image name (default: vllm/vllm-docker-executor:latest)"
            echo ""
            echo "Example:"
            echo "  ./build-docker-executor.sh"
            echo "  TARGET_IMAGE=myregistry/vllm-docker:v1 ./build-docker-executor.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Configuration
TARGET_IMAGE="${TARGET_IMAGE:-vllm/vllm-docker-executor:latest}"
DOCKERFILE_PATH="$(dirname "$0")/Dockerfile.docker-executor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building vLLM Docker Executor Image${NC}"
echo "========================================"
echo "Target image: $TARGET_IMAGE"
echo "Dockerfile: $DOCKERFILE_PATH"
echo "Build context: $(dirname "$0")"
echo ""

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}Error: Dockerfile not found: $DOCKERFILE_PATH${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if we can connect to Docker
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to Docker daemon${NC}"
    exit 1
fi

# Check if nvidia/cuda base image exists (since Docker Hub may be unavailable)
if ! docker inspect "nvidia/cuda:12.1.0-devel-ubuntu22.04" &> /dev/null; then
    echo -e "${YELLOW}Warning: Base image nvidia/cuda:12.1.0-devel-ubuntu22.04 not found locally${NC}"
    echo "The build will fail if the image cannot be pulled from Docker Hub."
    echo "If Docker Hub is unavailable, you may need to:"
    echo "  1. Download the base image from an alternative source"
    echo "  2. Use a different base image that is available locally"
fi

# Determine build context (repo root)
SCRIPT_DIR="$(dirname "$0")"
if [ -f "$SCRIPT_DIR/../../vllm/__init__.py" ]; then
    # Running from examples/docker_executor/
    BUILD_CONTEXT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    # Running from repo root
    BUILD_CONTEXT="$(pwd)"
fi

# Build the custom image
echo ""
echo -e "${YELLOW}Building image: $TARGET_IMAGE${NC}"
echo "Build context: $BUILD_CONTEXT"
echo "This may take several minutes..."
echo ""

if docker build \
    -t "$TARGET_IMAGE" \
    -f "$DOCKERFILE_PATH" \
    "$BUILD_CONTEXT"; then
    echo ""
    echo -e "${GREEN}Build successful!${NC}"
    echo "========================================"
    echo "Image: $TARGET_IMAGE"
    echo ""
    echo "To verify the image:"
    echo "  docker run --rm $TARGET_IMAGE python -c \"import vllm; print(vllm.__version__)\""
    echo ""
    echo "To use with vLLM:"
    echo "  export VLLM_DOCKER_IMAGE=$TARGET_IMAGE"
    echo "  python -c \"from vllm import LLM; llm = LLM(..., distributed_executor_backend='docker')\""
else
    echo ""
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
