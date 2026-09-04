#!/usr/bin/env bash
# Unit test 6: prompt whose token length is an EXACT multiple of the ubatch size.
#
# Bug (NOTES.md 4b): a prompt that tokenizes to exactly N*bs tokens restores
# fully (0 tokens left to prefill), but the full-restore branch then jumps
# straight to decode without ever running a forward pass over the prompt's
# final position -> sampler unseeded. Qwen3.8: GGML_ASSERT(t_prompt_last > 0)
# or an empty/garbage first token. DSV4F: coherent-looking nonsense.
#
# The prompt (built by llama_make_exact_prompt_len.sh) tokenizes to EXACTLY
# 256 tokens = 4 complete chunks at -ub 64. prime + resend: the resend must
# restore all 256 tokens and still answer 'OK' (not crash, not nonsense).
#
# Usage:  devops/llama_unittest_6.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

UB=64
TARGET=256   # exact multiple of $UB

# build the exact-length prompt (server must be running; llama_test.sh starts
# its own later, so start one here first, then let llama_test.sh re-spawn)
"${DEVS}/llama_run.sh" --no-kv-chain -- -ub ${UB} -b ${UB} >/dev/null 2>&1

PROMPT_LARGE=$("${DEVS}/llama_make_exact_prompt_len.sh" "$(( TARGET + 1 ))")
PROMPT_LARGE="${PROMPT_LARGE#*: [}"
PROMPT_LARGE="${PROMPT_LARGE%]}"

PROMPT_EXACT=$("${DEVS}/llama_make_exact_prompt_len.sh" "${TARGET}")
PROMPT_EXACT="${PROMPT_EXACT#*: [}"
PROMPT_EXACT="${PROMPT_EXACT%]}"

echo "prompt token count: ${TARGET} (ubatch=${UB})"

# --- run: prime, resend (no restart) ---
"${DEVS}/llama_test.sh" "${PROMPT_LARGE}" "${PROMPT_LARGE}" "[restart]" "[flush]" "${PROMPT_EXACT}" "${PROMPT_EXACT}" -- -ub ${UB} -b ${UB}

echo ""
echo "=== analysis ==="
PASS=1

get_cached_tokens_from_prompt_file() {
    grep -oE 'cached_tokens: [0-9]+' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+' || echo 0
}

get_response_tokens_from_prompt_file() {
    (grep -v 'cached_tokens' "$1" 2>/dev/null || true) | sed 's/^\s\+//'| sed 's/\s\+$//' | sed '/^$/d'
}

for i in $(seq 1 4) ; do
   echo "cached-tokens ${i}: $( get_cached_tokens_from_prompt_file "${DEVS}/prompt-${i}.log" )"
   echo "response-tokens ${i}: [$( get_response_tokens_from_prompt_file "${DEVS}/prompt-${i}.log" )]"
done

exit 0

