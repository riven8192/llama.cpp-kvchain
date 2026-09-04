#!/usr/bin/env bash
# Unit test 3c: kv-chain break (delete a middle .kvcache file, no restart).
#
# prime, delete the 3rd-oldest .kvcache file (chain breaks at chunk 2),
# resend in the same session. expect: cached_tokens ~ 64, coherent output.
#
# Usage:  devops/llama_unittest_evict_kvcache.sh
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

echo "=== evict_kvcache: kv-chain break ==="
"${DEVS}/llama_test.sh" \
    "${PROMPT}" \
    "[cmd:cmd_del_kv_middle.sh]" \
    "${PROMPT}" \
    -- -ub 32 -b 32

echo ""
echo "=== analysis ==="
PASS=1

RESTORED=$(grep -oE 'cached_tokens: [0-9]+' "${DEVS}/prompt-2.log" | head -1 | grep -oE '[0-9]+' || echo 0)
echo "chain-break restore: cached_tokens=${RESTORED}"

# chain breaks at chunk 2 (0-indexed) -> 2 chunks restored = 64 tokens
# allow 32-96 (1-3 chunks) to account for off-by-one in file ordering
if [[ "${RESTORED}" -lt 32 ]]; then
  echo "FAIL restored fewer than 32 tokens (expected chain break at chunk 2, >= 1 chunk restored)"
  PASS=0
fi
if [[ "${RESTORED}" -gt 96 ]]; then
  echo "FAIL restored ${RESTORED} tokens > 96 (expected chain break, should be ~64)"
  PASS=0
fi

# coherence: at least 2 phrases
n_found=0
for phrase in "${CHECKS[@]}"; do
  if grep -qF "${phrase}" "${DEVS}/prompt-2.log"; then
    n_found=$((n_found + 1))
  fi
done
echo "coherence: ${n_found}/${#CHECKS[@]} phrases"
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
