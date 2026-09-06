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
# Usage:  devops/llama_unittest_forked_chains.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )

PROMPT_A="'${PASSAGE}' -- the task is to ignore the above, and just respond with 'OK'"
PROMPT_B="'${PASSAGE:0:500} [mid-insertion] ${PASSAGE:500}' -- the task is to ignore the above, and just respond with 'OK'"

echo "=== forked_chains: forked chains (A, A, B, B) ==="
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

# --- rs-file touch-stride check -------------------------------------------
# The restore touches the TAIL rs file PLUS every rs file at an index that is a
# multiple of 8 (KV_CHAIN_RS_TOUCH_STRIDE in kv-chain-store.cpp) below the tail.
# The chains in this test are 11 chunks (indices 0..10), so a full restore
# touches rs at 0, 8 and the tail 10 -> exactly 3 rs files touched at ~the same
# instant, while all the others keep their (earlier) save timestamps.
#
# We cannot map a file's hex hash name to a chunk index, so instead of checking
# WHICH files were touched we check the COUNT: take the 4 most recently modified
# .rscache files and look at the time gaps between them. prompt4's restore
# (the last operation) touches 3 files at once, so sorted ascending the 4 newest
# must show: a LARGE gap (>500ms) between the 4th-newest (a file that was NOT
# touched in that restore) and the 3rd-newest, then small gaps (<100ms) between
# the 3 files that WERE touched together. a wrong stride (e.g. touching only
# the tail, or touching every rs file) changes this 4-tuple's gap pattern.
#
# (sloppy by design - see the comment in project-plan: this is a local hack,
# the gap pattern is the strongest signal available without hash->index mapping)
rs_touch_gaps=$( find "${KV_CACHE_DIR}" -name '*.rscache' -printf '%T@\n' 2>/dev/null | sort -n | tail -n 4 | awk 'NR>1{printf "%.3f ", $1-p} {p=$1}' )
echo "rs-touch gaps (newest-4, oldest->newest): ${rs_touch_gaps}"
n_small=$( echo "${rs_touch_gaps}" | awk '{n=0; for (i=1; i<=NF; i++) if ($i+0 < 0.1) n++} END{print n}' )
n_large=$( echo "${rs_touch_gaps}" | awk '{n=0; for (i=1; i<=NF; i++) if ($i+0 > 0.5) n++} END{print n}' )
if [[ "${n_small}" -eq 2 && "${n_large}" -eq 1 ]]; then
  echo "PASS [rs-touch] 3 rs files touched together (2 gaps <100ms) + 1 gap >500ms to the rest"
else
  echo "FAIL [rs-touch] expected 2 gaps <100ms and 1 gap >500ms in the newest-4, got: ${rs_touch_gaps}"
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
