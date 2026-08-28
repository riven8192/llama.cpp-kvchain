#!/usr/bin/env bash
# Configure + build llama-server (Vulkan, Release).
# Usage:
#   llama_build.sh            # incremental build
#   llama_build.sh --clean    # remove the build dir first
#   llama_build.sh -j N       # override parallelism
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

CLEAN=0
JOBS="$(nproc)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) CLEAN=1; shift ;;
    -j)      JOBS="${2:?-j needs a value}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "${LLAMA_ROOT}"

if [[ ${CLEAN} -eq 1 ]]; then
  echo "removing ${LLAMA_BUILD_DIR}"
  rm -rf "${LLAMA_BUILD_DIR}"
fi

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
