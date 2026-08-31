# Shared config for devops scripts. This file is meant to be SOURCED, not run:
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# It only sets defaults (via : "${VAR:=...}"), never exits or returns, so it
# is safe to source. Override any value from the environment before sourcing.
# If executed directly (a mistake), fail loudly instead of doing nothing.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "env.sh must be sourced, not executed" >&2
  return 1 2>/dev/null || exit 1
fi

: "${LLAMA_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${LLAMA_BUILD_DIR:=${LLAMA_ROOT}/build-vulkan}"
: "${LLAMA_SERVER_BIN:=${LLAMA_BUILD_DIR}/bin/llama-server}"

# HF model ref "<user>/<repo>:<quant>" - llama-server resolves it to the exact
# cached file (and downloads it if missing), picking the right snapshot revision
# instead of guessing 'head -1' out of the snapshots dir.
: "${LLAMA_HF_REF:=unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL}"
# : "${LLAMA_HF_REF:=unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q8_K_XL}"
# : "${LLAMA_HF_REF:=unsloth/Qwen3-4B-Instruct-2507-GGUF}"


: "${LLAMA_HOST:=127.0.0.1}"
: "${LLAMA_PORT:=50081}"
: "${LLAMA_CTX:=4096}"
: "${LLAMA_NGL:=99}"
: "${LLAMA_THREADS:=16}"
: "${LLAMA_PARALLEL:=1}"

: "${KV_CACHE_DIR:=${LLAMA_ROOT}/devops/.kv-cache}"
: "${KV_CHAIN_LIMIT_GB:=100}"
: "${LLAMA_LOG:=${LLAMA_ROOT}/devops/.llama-server.log}"
: "${LLAMA_PIDFILE:=${LLAMA_ROOT}/devops/.llama-server.pid}"

LLAMA_URL="http://${LLAMA_HOST}:${LLAMA_PORT}"
