#!/usr/bin/env bash
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PASSAGE=$(cat "${DEVS}/prompt_7sentences.txt")
PROMPT="${PASSAGE} -- ignore the above text, please only reply with listing 10 colors"

echo "=== smoke-test (naming colors) ==="
LLAMA_CTX=512 "${DEVS}/llama_test.sh" \
	"[restart-no-kv]" "[flush]" "${PROMPT}" "${PROMPT}" \
	"[restart]"       "[flush]" "${PROMPT}" "${PROMPT}" \
        -- -ub 32 -b 32 >/dev/null 2>&1
echo ""

echo "prompt 1 and 2: kv-cache disabled"
echo "prompt 2 and 3: kv-cache enabled"

"${DEVS}/llama_prompt_response_analysis.sh" \
	"${DEVS}/prompt-1.log" \
	"${DEVS}/prompt-2.log" \
	"${DEVS}/prompt-3.log" \
	"${DEVS}/prompt-4.log"

