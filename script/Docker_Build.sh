#!/bin/bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-hecate-compiler}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
TARGET="${TARGET:-cuda13}"

echo -e "\033[1;32m======Build ${IMAGE_NAME}:${IMAGE_TAG} for ${TARGET}======\033[0m"

# Build the image
docker build \
    --target "${TARGET}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${SCRIPT_PATH}"

# Clean up dangling images
if [ "$(docker images -f 'dangling=true' -q)" ]; then
    echo "Removing dangling Docker images..."
    docker rmi $(docker images -f 'dangling=true' -q) 2>/dev/null || true
fi

echo -e "\033[1;32m✅ Build completed: ${IMAGE_NAME}:${IMAGE_TAG}\033[0m"
