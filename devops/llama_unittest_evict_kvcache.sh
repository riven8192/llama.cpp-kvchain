#!/usr/bin/env bash
# Unit test 3c: kv-chain break (delete a middle .kvcache file, no restart).
#
# prime, delete the 3rd-oldest .kvcache file (the chain BREAKS at chunk 2:
# the files exist for chunks 0,1 and 3..N, but not 2), resend in the same
# session.
#
# EXPECTED: cached_tokens = 64 (2 chunks restored), coherent output.
#
# deleting the 3rd-oldest .kvcache (chunk 2) makes phase 1 of load_prefix
# (fs::exists() only) walk: kv found for 0,1 -> MISSING at 2 -> break. the
# replay then streams chunks 0,1 only (the tail rs = chunk 1's, present) and
# prefills the rest. this exercises the phase-1 break path, NOT the mid-replay
# failure path.
#
# the mid-replay path (a .kvcache that is present in phase 1 but fails to
# read during the replay - eviction race, or on-disk corruption) is different
# and deliberately harsher: by then the attn rows of the earlier chunks are
# already in the KV cache, but the recurrent tail (loaded only after the loop)
# is not - those rows are orphaned (no valid recurrent state to resume from),
# so a partial restore would produce garbage. the replay loop therefore wipes
# the seq and falls back to a 100% prefill (see the comment at the wipe in
# server-context.cpp). that case needs a corrupt (not missing) file and is not
# covered by a unit test (it is a corruption path, like the tail-rs-delete
# path in load_prefix).
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

# phase 1 breaks at chunk 2 (file missing) -> chunks 0,1 replayed = 64 tokens.
# the break is at chunk 2 (the deleted file is the 3rd-oldest = chunk 2's,
# since files are saved in chunk order). allow 32-96 (1-3 chunks) to account
# for off-by-one in file ordering.
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
