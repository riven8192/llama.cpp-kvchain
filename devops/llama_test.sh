#!/usr/bin/env bash
# Hash-chain KV cache restore test.
#
# Usage:
#   llama_test.sh [--flush] "prompt1" "[restart]" "prompt2" ... [-- extra llama_run.sh args]
#     --flush   start from an empty disk cache (default: keep existing)
#     [restart] restart the server (keeping the disk cache) before the next prompt
#     -- ...    extra args forwarded to llama_run.sh (e.g. "-ub 8")

set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

FLUSH=0
if [ "${1:-}" = "--flush" ]; then
    FLUSH=1
    shift
fi

# split off a trailing `-- extra args` for llama_run.sh
RUN_ARGS=()
PROMPTS=()
SEEN_DD=0
for arg in "$@"; do
    if [ "${SEEN_DD}" -eq 1 ]; then
        RUN_ARGS+=("${arg}")
    elif [ "${arg}" = "--" ]; then
        SEEN_DD=1
    else
        PROMPTS+=("${arg}")
    fi
done
RUN_ARGS_STR="${RUN_ARGS[*]:-}"

run_server() {
    if [ -n "${RUN_ARGS_STR}" ]; then
        "${DEVS}/llama_run.sh" --keep-cache -- ${RUN_ARGS[@]} | tail -3
    else
        "${DEVS}/llama_run.sh" --keep-cache | tail -3
    fi
}

echo "=== [1/5] flush kv-cache ==="
if [ "${FLUSH}" -eq 1 ]; then
    rm -rf "${KV_CACHE_DIR}"
    echo "flushed ${KV_CACHE_DIR}"
else
    echo "keeping existing cache: ${KV_CACHE_DIR}"
    ls -la "${KV_CACHE_DIR}" 2>/dev/null || echo "(no cache dir yet)"
fi
echo ""

echo "=== [2/5] llama_build.sh ==="
"${DEVS}/llama_build.sh" | tail -3
echo ""

echo "=== [3/5] llama_run.sh ==="
run_server
echo ""

echo "=== [4/5] llama_wait.sh ==="
"${DEVS}/llama_wait.sh" 300
echo ""

counter=1
for prompt in "${PROMPTS[@]}"; do
    if [ "${prompt}" = '[restart]' ]; then
        echo "=== [restart] ==="
        run_server
        "${DEVS}/llama_wait.sh"
        echo ""
    else
        echo "=== [5/5] llama_prompt.sh ==="
        echo "prompt: '${prompt}'"
        echo "log-file: ${DEVS}/prompt-${counter}.log"
        "${DEVS}/llama_prompt.sh" "${prompt}" 2>&1 | tee "${DEVS}/prompt-${counter}.log" || true
        echo ""
        counter="$((counter + 1))"
    fi
done

echo "=== chunks on disk ==="
ls -la "${KV_CACHE_DIR}" 2>/dev/null | grep kvchunk || echo "(none)"
