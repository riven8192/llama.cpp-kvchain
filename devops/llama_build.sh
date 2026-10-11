#!/usr/bin/env bash
set -euo pipefail

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

JOBS="$(( $(nproc) / 2 ))"
BUILD_LOG="${LLAMA_ROOT}/devops/.llama-build.log"
LOCK_FILE="${LLAMA_ROOT}/devops/.llama-build.lock"

exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
        echo "A build is already running. Aborting." >&2
        exit 1
fi

cd "${LLAMA_ROOT}"

echo "Flushing build cache..."
rm -rf "${LLAMA_BUILD_DIR}"
mkdir -p "${LLAMA_BUILD_DIR}"

# -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON (use if not linked statically)


echo "Configuring (build dir: ${LLAMA_BUILD_DIR})"
cmake -S . -B "${LLAMA_BUILD_DIR}" -G Ninja \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_BORINGSSL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

echo "Building with ${JOBS} jobs..."
build_exit=0
cmake --build "${LLAMA_BUILD_DIR}" -j"${JOBS}" 2>&1 | tee "${BUILD_LOG}" || build_exit=$?

echo "build_exit=${build_exit}"
exit "${build_exit}"
