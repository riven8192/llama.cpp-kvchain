#!/usr/bin/env bash
# Hash-chain KV cache restore test.
#
# Usage:
#   llama_test.sh [--flush] "prompt1" "[restart]" "prompt2" ...
#     --flush   start from an empty disk cache (default: keep existing)
#     [restart] restart the server (keeping the disk cache) before the next prompt

set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

# keep the disk cache across the run; pass --flush to start from an empty cache
FLUSH=0
if [ "${1:-}" = "--flush" ]; then
    FLUSH=1
    shift
fi

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
"${DEVS}/llama_run.sh" --keep-cache | tail -3
echo ""

echo "=== [4/5] llama_wait.sh ==="
"${DEVS}/llama_wait.sh" 300
echo ""

counter=1
for prompt in "$@" ; do
    if [ "${prompt}" = '[restart]' ] ; then
        echo "=== [restart] ==="
        "${DEVS}/llama_run.sh" --keep-cache | tail -3
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
