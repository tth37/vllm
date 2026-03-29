#!/bin/bash
# Download the ShareGPT benchmark dataset to the local cache.
#
# Default location: ~/.cache/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
# Override with: SHAREGPT_CACHE_DIR=/path/to/dir ./download_sharegpt.sh
#
# The script is idempotent — it skips the download if the file already exists.

set -euo pipefail

DATASET_NAME="ShareGPT_V3_unfiltered_cleaned_split.json"
CACHE_DIR="${SHAREGPT_CACHE_DIR:-$HOME/.cache/vllm/datasets}"
DATASET_PATH="${CACHE_DIR}/${DATASET_NAME}"
DOWNLOAD_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/${DATASET_NAME}"

if [[ -f "$DATASET_PATH" ]]; then
    echo "$DATASET_PATH"
    exit 0
fi

echo "Downloading ShareGPT dataset to $DATASET_PATH ..." >&2
mkdir -p "$CACHE_DIR"
curl -L --progress-bar -o "${DATASET_PATH}.tmp" "$DOWNLOAD_URL"
mv "${DATASET_PATH}.tmp" "$DATASET_PATH"
echo "Done." >&2
echo "$DATASET_PATH"
