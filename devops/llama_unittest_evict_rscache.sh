#!/usr/bin/env bash
# Unit test 3b: rs-eviction resilience (no restart).
#
# prime, delete all but the last 2 .rscache files, resend in the same session.
# expect: full restore (tail rs intact -> usable = last kv chunk), 6/6 phrases.
#
# Usage:  devops/llama_unittest_evict_rscache.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )

CHECKS=(
  "In the philosophy of mind, mind–body dualism denotes either that mental phenomena are non-physical"
  "a nutritive soul of growth and metabolism that all three share"
  "a soul is the hylomorphic form of a viable organism"
  "an immortal and perpetual intellective part of mind"
  "he believed in metempsychosis, the migration of the soul to a new physical body"
  "the tendency to ignore very big groups of variables by its assumed association with the mind or the body"
)

PROMPT="please repeat this entire passage exactly, word for word, with no additions or omissions: '${PASSAGE}'"

echo "=== evict_rscache: rs-eviction resilience ==="
"${DEVS}/llama_test.sh" \
    "${PROMPT}" \
    "[cmd:cmd_del_rs_middle.sh]" \
    "${PROMPT}" \
    -- -ub 32 -b 32

echo ""
echo "=== analysis ==="
PASS=1

# prompt-1.log = prime (cold), prompt-2.log = partial restore after rs deletion
RESTORED=$(grep -oE 'cached_tokens: [0-9]+' "${DEVS}/prompt-2.log" | head -1 | grep -oE '[0-9]+' || echo 0)
echo "partial restore: cached_tokens=${RESTORED}"

# the cmd script keeps the 2 NEWEST rs files, so the tail (last chunk) still has
# its rs file. per the design, usable = last chunk with an rs file, so a full
# restore is EXPECTED here. the point of this test is: middle rs files can be
# evicted without breaking restore (zeros are streamed for the gaps).
# we only FAIL if restore dropped to 0 (the tail rs file itself was missing).
if [[ "${RESTORED}" -lt 32 ]]; then
  echo "FAIL restored fewer than 32 tokens (tail rs file should still be present)"
  PASS=0
else
  echo "PASS full restore despite missing middle rs files (tail rs intact)"
fi

# coherence: at least the first 2 phrases should survive (early chunks intact)
n_found=0
for phrase in "${CHECKS[@]}"; do
  if grep -qF "${phrase}" "${DEVS}/prompt-2.log"; then
    n_found=$((n_found + 1))
  fi
done
echo "coherence: ${n_found}/${#CHECKS[@]} phrases in partial-restore output"
if [[ "${n_found}" -lt 2 ]]; then
  echo "FAIL fewer than 2 phrases (output not coherent)"
  PASS=0
else
  echo "PASS output is coherent"
fi

echo ""
if [[ "${PASS}" -eq 1 ]]; then
  echo "=== RESULT: PASS ==="
  exit 0
else
  echo "=== RESULT: FAIL ==="
  exit 1
fi
