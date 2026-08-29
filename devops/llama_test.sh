#!/usr/bin/env bash
# Hash-chain KV cache restore test.
#
# Usage:
#   llama_test.sh "prompt1" "[restart]" "prompt2" ... [-- extra llama_run.sh args]
#     [restart]   restart the server (keeping the disk cache) before the next prompt
#     -- ...      extra args forwarded to llama_run.sh (e.g. "-ub 8")

set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

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
    # llama_run.sh waits for the server to be ready before returning
    if [ -n "${RUN_ARGS_STR}" ]; then
        "${DEVS}/llama_run.sh" --keep-cache -- "${RUN_ARGS[@]}"
    else
        "${DEVS}/llama_run.sh" --keep-cache
    fi
}

echo "=== [1/4] flush kv-cache ==="
du -h "${KV_CACHE_DIR}"
rm -rf "${KV_CACHE_DIR}"
rm -f "${DEVS}"/prompt-*.log
echo ""

echo "=== [2/4] llama_build.sh ==="
"${DEVS}/llama_build.sh" | tail -3
echo ""

echo "=== [3/4] llama_run.sh ==="
run_server
echo ""

counter=1
for prompt in "${PROMPTS[@]}"; do
    if [ "${prompt}" = '[restart]' ]; then
        echo "=== [restart] ==="
        run_server
        echo ""
    else
        echo "=== [4/4] llama_prompt.sh ==="
        echo "prompt: '${prompt}'"
        echo "log-file: ${DEVS}/prompt-${counter}.log"
        "${DEVS}/llama_prompt.sh" "${prompt}" 2>&1 | tee "${DEVS}/prompt-${counter}.log" || true
        echo ""
        counter="$((counter + 1))"
    fi
done

echo "=== chunks on disk ==="
find "${KV_CACHE_DIR}" -name '*.kvchunk' 2>/dev/null -exec ls -la {} \; || echo "(none)"
