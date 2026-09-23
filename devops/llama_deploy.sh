#!/bin/bash

set -euo pipefail


SRC_DIR=~/opencode/kv-cache-qwen/llama.cpp-kv-qwen/llama.cpp/build-vulkan/
DST_DIR=~/llama.cpp-kvlive/

rm -rf "${DST_DIR}/build-vulkan/" # retain dir: kvcache
mkdir -p "${DST_DIR}"
cp -r "${SRC_DIR}" "${DST_DIR}"

cd "${DST_DIR}" && find .

echo "Deployed to: ${DST_DIR}"
