#!/usr/bin/env bash
# Ubatch-boundary edge cases: prompt lengths of TARGET-1, TARGET+1 and TARGET,
# where TARGET is an EXACT multiple of the ubatch size.
#
# The exact-multiple case is the interesting one (was the NOTES.md 4b bug): a
# prompt of exactly N*bs tokens would restore in full, leaving nothing to run a
# forward pass over -> no logits -> the slot never samples (empty reply on Qwen,
# nonsense on DSV4F). load_prefix now caps the restore by one chunk so at least
# one token is always re-prefilled. The -1/+1 neighbours are the controls: they
# must be UNaffected by that cap.
#
# Each length is primed cold and then resent, with a [restart]+[flush] between
# the three groups so every prime starts from an empty cache dir. Both the
# response text AND the cached-token count are asserted (see the checks below).
#
# Usage:  devops/llama_unittest_ubatch_edge.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

UB=64
TARGET=256   # exact multiple of $UB

# build the exact-length prompt (server must be running; llama_test.sh starts
# its own later, so start one here first, then let llama_test.sh re-spawn)
"${DEVS}/llama_run.sh" --no-kv-chain -- -ub ${UB} -b ${UB} >/dev/null 2>&1

echo "Generating prompt with $(( TARGET - 1 )) tokens..."
PROMPT_SMALL=$("${DEVS}/llama_make_exact_prompt_len.sh" "$(( TARGET - 1 ))")
PROMPT_SMALL="${PROMPT_SMALL#*: [}"
PROMPT_SMALL="${PROMPT_SMALL%]}"

echo "Generating prompt with $(( TARGET + 1 )) tokens..."
PROMPT_LARGE=$("${DEVS}/llama_make_exact_prompt_len.sh" "$(( TARGET + 1 ))")
PROMPT_LARGE="${PROMPT_LARGE#*: [}"
PROMPT_LARGE="${PROMPT_LARGE%]}"

echo "Generating prompt with $(( TARGET )) tokens..."
PROMPT_EXACT=$("${DEVS}/llama_make_exact_prompt_len.sh" "${TARGET}")
PROMPT_EXACT="${PROMPT_EXACT#*: [}"
PROMPT_EXACT="${PROMPT_EXACT%]}"


echo "prompt token count: ${TARGET} (ubatch=${UB})"

# N.B.: do `exact` last, becaus ethat is the edge-case, and should be easiest to introspect.
"${DEVS}/llama_test.sh" \
	"${PROMPT_SMALL}" "${PROMPT_SMALL}" "[restart]" "[flush]" \
	"${PROMPT_LARGE}" "${PROMPT_LARGE}" "[restart]" "[flush]" \
	"${PROMPT_EXACT}" "${PROMPT_EXACT}" \
        -- -ub ${UB} -b ${UB}

echo ""
echo "=== analysis ==="
PASS=1

get_cached_token_count_from_prompt_file() {
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

for i in $(seq 1 6) ; do
   c="$( get_cached_token_count_from_prompt_file "${DEVS}/prompt-${i}.log" )"
   r="$( get_response_tokens_from_prompt_file "${DEVS}/prompt-${i}.log" )"
   echo "cached-token-count ${i}: ${c}"
   echo "response-tokens ${i}: [${r}]"
   eval "C${i}=\"\${c}\""
   eval "R${i}=\"\${r}\""
   echo ""
done


echo "Expected: SMALL=0/$(( TARGET - UB ))"
echo "Expected: LARGE=0/$(( TARGET      ))"
echo "Expected: EXACT=0/$(( TARGET - UB ))"
echo ""

# --- assertions -----------------------------------------------------------
# the actual correctness oracle: every response must be exactly "OK" (mod
# whitespace/punctuation/case). checked for ALL SIX prompts, not just the
# bug case - a regression could just as easily show up on the "normal" paths.
for i in $(seq 1 6) ; do
    eval "r=\"\${R${i}}\""
    if is_ok_response "${r}"; then
        echo "PASS prompt${i} response is OK"
    else
        echo "FAIL prompt${i} response is NOT 'OK' (got: [${r:0:120}$([[ ${#r} -gt 120 ]] && echo '...')])"
        PASS=0
    fi
done

echo ""

# the cached-token counts are asserted too, not just printed: a response can be
# "OK" while the restore silently degraded (e.g. the exact-multiple cap turning
# into a full re-prefill, or the cap wrongly firing on the +1/-1 cases). that
# would cost minutes of prefill per request without failing any content check.
#   prompts 1/3/5 are cold primes (cache flushed before each) -> 0
#   prompt 2 (SMALL=TARGET-1): last chunk is partial, never persisted -> TARGET-UB
#   prompt 4 (LARGE=TARGET+1): TARGET/UB whole chunks + 1 leftover token -> TARGET
#   prompt 6 (EXACT=TARGET)  : capped by one chunk (NOTES.md 4b)      -> TARGET-UB
check_cached() {
    local label="$1" got="$2" want="$3"
    if [[ "${got}" -eq "${want}" ]]; then
        echo "PASS ${label} cached_tokens=${got}"
    else
        echo "FAIL ${label} cached_tokens=${got} (expected ${want})"
        PASS=0
    fi
}

check_cached "prompt1 (SMALL cold prime)" "${C1}" 0
check_cached "prompt2 (SMALL restore)"    "${C2}" "$(( TARGET - UB ))"
check_cached "prompt3 (LARGE cold prime)" "${C3}" 0
check_cached "prompt4 (LARGE restore)"    "${C4}" "${TARGET}"
check_cached "prompt5 (EXACT cold prime)" "${C5}" 0
check_cached "prompt6 (EXACT restore)"    "${C6}" "$(( TARGET - UB ))"

echo ""
if [[ "${PASS}" -eq 1 ]]; then
    echo "=== RESULT: PASS ==="
    exit 0
else
    echo "=== RESULT: FAIL ==="
    exit 1
fi

