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

# every prompt in this test ends with the instruction "... just respond with
# 'OK'". a correct response is (up to whitespace/punctuation) exactly "OK" -
# anything else (empty, a rambling unrelated essay, json/xml/comment-block
# garbage, etc.) means the restored/replayed state was NOT equivalent to a
# fresh prefill, whether that shows up as a crash, an empty reply, or (the
# sneaky case) confidently-wrong prose that looks fine at a glance but ignored
# the instruction because the model was conditioned on a corrupted context.
is_ok_response() {
    local r
    r="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:][:punct:]')"
    [[ "${r}" == "OK" ]]
}

for i in $(seq 1 4) ; do
   c="$( get_cached_tokens_from_prompt_file "${DEVS}/prompt-${i}.log" )"
   r="$( get_response_tokens_from_prompt_file "${DEVS}/prompt-${i}.log" )"
   echo "cached-tokens ${i}: ${c}"
   echo "response-tokens ${i}: [${r}]"
   eval "C${i}=\"\${c}\""
   eval "R${i}=\"\${r}\""
done

# --- assertions -----------------------------------------------------------
# prompt 1 (LARGE, cold prime): nothing on disk yet
if [[ "${C1}" -ne 0 ]]; then
    echo "FAIL prompt1 (cold prime) expected cached_tokens=0, got ${C1}"
    PASS=0
fi
# prompt 3 ([restart]+[flush]+EXACT, cold prime): fresh cache dir
if [[ "${C3}" -ne 0 ]]; then
    echo "FAIL prompt3 (EXACT cold prime) expected cached_tokens=0, got ${C3}"
    PASS=0
fi
# prompt 4 (EXACT resend, same session) is THE BUG CASE (NOTES.md 4b): a
# full-chain hit (n_saved == n_prompt == TARGET tokens). we don't pin the
# exact cached_tokens value here (that depends on the eventual fix's
# strategy - e.g. rolling back 1 token vs a whole chunk), only that a real
# restore actually happened (>0) - the content check below is what actually
# proves correctness.
if [[ "${C4}" -le 0 ]]; then
    echo "FAIL prompt4 (EXACT resend, the bug case) expected cached_tokens>0 (a disk restore should have been attempted), got ${C4}"
    PASS=0
fi
# the actual correctness oracle: every response must be exactly "OK" (mod
# whitespace/punctuation/case). checked for ALL FOUR prompts, not just the
# bug case - a regression could just as easily show up on the "normal" paths.
for i in 1 2 3 4; do
    eval "r=\"\${R${i}}\""
    if is_ok_response "${r}"; then
        echo "PASS prompt${i} response is OK"
    else
        echo "FAIL prompt${i} response is NOT 'OK' (got: [${r:0:120}$([[ ${#r} -gt 120 ]] && echo '...')])"
        PASS=0
    fi
done

echo ""
if [[ "${PASS}" -eq 1 ]]; then
    echo "=== RESULT: PASS ==="
    exit 0
else
    echo "=== RESULT: FAIL ==="
    exit 1
fi

