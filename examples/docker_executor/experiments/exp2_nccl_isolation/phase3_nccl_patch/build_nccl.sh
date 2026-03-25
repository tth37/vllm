#!/bin/bash
# Clone, patch, and build NCCL from source.
#
# Usage:
#   ./build_nccl.sh [--clean]
#
# Output:
#   /workspace/nccl/build/lib/libnccl.so.2
#
# To use the patched NCCL:
#   export LD_PRELOAD=/workspace/nccl/build/lib/libnccl.so.2

set -euo pipefail

NCCL_DIR="/workspace/nccl"
NCCL_VERSION="v2.21.5-1"
PATCHES_DIR="/workspace/patches"

if [ "${1:-}" = "--clean" ] && [ -d "$NCCL_DIR/src" ]; then
    echo "Cleaning previous build..."
    cd "$NCCL_DIR"
    make clean || true
fi

# Clone NCCL if not already cloned
if [ ! -f "$NCCL_DIR/Makefile" ]; then
    echo "Cloning NCCL $NCCL_VERSION..."
    cd /workspace
    rm -rf nccl
    git clone https://github.com/NVIDIA/nccl.git -b "$NCCL_VERSION" --depth 1 nccl
fi

cd "$NCCL_DIR"

# Apply patches
if [ -d "$PATCHES_DIR" ] && ls "$PATCHES_DIR"/*.patch 1>/dev/null 2>&1; then
    echo "Applying patches..."
    for patch in "$PATCHES_DIR"/*.patch; do
        echo "  Applying: $(basename "$patch")"
        git apply "$patch" || {
            echo "  WARNING: Patch $(basename "$patch") failed to apply (may already be applied)"
        }
    done
else
    echo "No patches found in $PATCHES_DIR"
fi

# Build NCCL
echo ""
echo "Building NCCL..."
echo "  CUDA_HOME: ${CUDA_HOME:-/usr/local/cuda}"

make -j"$(nproc)" src.build \
    CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"

echo ""
echo "Build complete!"
echo "  Library: $NCCL_DIR/build/lib/libnccl.so.2"
echo ""
echo "To test: LD_PRELOAD=$NCCL_DIR/build/lib/libnccl.so.2 python3 benchmark_docker.py"
