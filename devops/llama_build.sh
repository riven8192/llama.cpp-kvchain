#!/usr/bin/env bash
# Configure + build llama-server (Vulkan, Release).
# Usage:
#   llama_build.sh
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

JOBS="$(nproc)"

cd "${LLAMA_ROOT}"

# Wipe the build dir EXCEPT the entries in KEEP: a plain full `rm -rf` also
# throws away expensive, source-independent build products every time (the
# BoringSSL fetch+build in _deps/, ~826M, and the Vulkan shader pipeline
# under ggml/ - the vulkan-shaders-gen ExternalProject + the compiled SPIR-V
# shaders, ~315M) - neither depends on anything under src/tools/common, so
# rebuilding them on every source-only change is pure waste.
#
# Deliberately an ALLOWLIST of what to KEEP (not a denylist of what to
# delete): a denylist silently stops covering new build subdirectories CMake
# adds later (a new target, a new vendored dep, ...), so stale objects from
# those would quietly survive a "clean" build again - the exact bug this
# script exists to avoid. An allowlist fails safe: anything not explicitly
# known-safe-to-keep gets wiped, same as the old rm -rf.
#
# CMakeCache.txt/build.ninja/CMakeFiles are NOT kept: removing them forces a
# full `cmake` reconfigure below, which re-runs file(GLOB ...) in
# src/CMakeLists.txt (models/*.cpp) and regenerates build-info.cpp - the
# actual source of the stale-binary bug this rm -rf was working around.
KEEP=(_deps ggml Release tools)
KEEP=()

echo "removing ${LLAMA_BUILD_DIR} (except: ${KEEP[*]})"
if [[ -d "${LLAMA_BUILD_DIR}" ]]; then
  for entry in "${LLAMA_BUILD_DIR}"/* "${LLAMA_BUILD_DIR}"/.[!.]*; do
    [[ -e "${entry}" ]] || continue
    name="$(basename "${entry}")"
    skip=0
    for k in "${KEEP[@]}"; do
      [[ "${name}" == "${k}" ]] && skip=1 && break
    done
    [[ ${skip} -eq 1 ]] && continue
    echo rm -rf "${entry}"
    rm -rf "${entry}"
  done
fi


echo "configuring (build dir: ${LLAMA_BUILD_DIR})"
cmake -S . -B "${LLAMA_BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_OPENSSL=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_BORINGSSL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

echo "building with -j${JOBS}"

BUILD_LOG="${LLAMA_ROOT}/devops/.llama-build.log"
cmake --build "${LLAMA_BUILD_DIR}" -j"${JOBS}" >"${BUILD_LOG}" 2>&1 &
build_pid=$!
tail -n 999 -F "${BUILD_LOG}" &
tail_pid=$!

wait "${build_pid}"
build_exit=$?
kill "${tail_pid}"

echo "build_exit=${build_exit}"
exit "${build_exit}"
