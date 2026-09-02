#!/usr/bin/env bash
# Unit test 4: forked hash chains.
#
# Two prompts share a trunk but diverge:
#   PROMPT_A = "'<passage>' -- the task is to ignore the above, and just respond with 'OK'"
#   PROMPT_B = "'<passage-with-a-mid-insertion>' -- ... respond with 'OK'"
# (B = A's passage with a [mid-insertion] spliced in the middle, so B's first
#  3 chunks == A's trunk, then B diverges.)
#
# The sequence PROMPT_A, PROMPT_A, PROMPT_B, PROMPT_B exercises every fork
# case in one run (no restart - the disk chain is the only prefix cache):
#   prompt 1 (A)  : cold prime of A's trunk            -> 0
#   prompt 2 (A)  : full-chain restore of A            -> 352 (11 chunks)
#   prompt 3 (B)  : reuses A's trunk (3 chunks) + saves B's diverging branch
#                                                  -> 96  (3 chunks)
#   prompt 4 (B)  : full restore of B (A's trunk + B's branch saved by prompt 3)
#                                                  -> 352 (11 chunks)
#
# A correct implementation must produce exactly 0 / 352 / 96 / 352. A wrong
# chain (e.g. hashing the whole prompt instead of the prefix, or not forking)
# would yield a different 4-tuple.
#
# Usage:  devops/llama_unittest_4.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )

PROMPT_A="'${PASSAGE}' -- the task is to ignore the above, and just respond with 'OK'"
PROMPT_B="'${PASSAGE:0:500} [mid-insertion] ${PASSAGE:500}' -- the task is to ignore the above, and just respond with 'OK'"

echo "=== unittest_4: forked chains (A, A, B, B) ==="
"${DEVS}/llama_test.sh" "${PROMPT_A}" "${PROMPT_A}" "${PROMPT_B}" "${PROMPT_B}" -- -ub 32 -b 32

echo ""
echo "=== analysis ==="
PASS=1

# the passage is ~370 tokens -> 11 complete 32-token chunks (352 tokens).
FULL=352
TRUNK=96   # 3 chunks shared before the mid-insertion diverges the chain

# read the cached_tokens line from a prompt log (0 if missing)
get_cached() {
  grep -oE 'cached_tokens: [0-9]+' "$1" 2>/dev/null | head -1 | grep -oE '[0-9]+' || echo 0
}

C1=$(get_cached "${DEVS}/prompt-1.log")
C2=$(get_cached "${DEVS}/prompt-2.log")
C3=$(get_cached "${DEVS}/prompt-3.log")
C4=$(get_cached "${DEVS}/prompt-4.log")

echo "cached_tokens: prompt1=${C1}  prompt2=${C2}  prompt3=${C3}  prompt4=${C4}"
echo "expected      : prompt1=0     prompt2=${FULL} prompt3=${TRUNK}  prompt4=${FULL}"

check_cached() {
  local label="$1" got="$2" want="$3"
  if [[ "${got}" -eq "${want}" ]]; then
    echo "PASS [${label}] cached_tokens=${got}"
  else
    echo "FAIL [${label}] cached_tokens=${got} (expected ${want})"
    PASS=0
  fi
}

check_cached "A prime (cold)"        "${C1}" 0
check_cached "A repeat (full)"       "${C2}" "${FULL}"
check_cached "B first (trunk fork)"  "${C3}" "${TRUNK}"
check_cached "B repeat (full)"       "${C4}" "${FULL}"

# the fork must actually be a PARTIAL restore (B reuses the trunk, not the whole
# chain): prompt3 must restore some but NOT all tokens.
if [[ "${C3}" -gt 0 && "${C3}" -lt "${FULL}" ]]; then
  echo "PASS [fork] prompt3 restored a proper prefix (${C3} < ${FULL})"
else
  echo "FAIL [fork] prompt3 did not restore a proper prefix (got ${C3})"
  PASS=0
fi

echo ""
if [[ "${PASS}" -eq 1 ]]; then
  echo "=== RESULT: PASS ==="
  exit 0
else
  echo "=== RESULT: FAIL ==="
  exit 1
fi
