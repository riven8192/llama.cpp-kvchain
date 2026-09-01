#!/usr/bin/env bash
# Configure + build llama-server (Vulkan, Release).
# Usage:
#   llama_build.sh
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

JOBS="$(nproc)"

cd "${LLAMA_ROOT}"

echo "removing ${LLAMA_BUILD_DIR}"
rm -rf "${LLAMA_BUILD_DIR}"

echo "configuring (build dir: ${LLAMA_BUILD_DIR})"
cmake -S . -B "${LLAMA_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_BORINGSSL=ON

echo "building with -j${JOBS}"
# capture output to a log so a failure can be inspected; tail the last lines
BUILD_LOG="${LLAMA_ROOT}/devops/.llama-build.log"
if ! cmake --build "${LLAMA_BUILD_DIR}" -j"${JOBS}" > "${BUILD_LOG}" 2>&1; then
  echo "BUILD FAILED - last 40 lines of ${BUILD_LOG}:" >&2
  tail -40 "${BUILD_LOG}" >&2
  exit 1
fi

echo "build OK - server: ${LLAMA_SERVER_BIN}"
tail -3 "${BUILD_LOG}"
