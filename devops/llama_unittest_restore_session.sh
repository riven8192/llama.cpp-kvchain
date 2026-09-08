#!/usr/bin/env bash
# Unit test 3a: split-file full-chain restore (no restart).
#
# prime, then resend the same prompt in the same session -> expect full chain
# restore from disk (all kv+rs files present), 6/6 phrases, cached_tokens >= 256.
#
# Usage:  devops/llama_unittest_restore_session.sh
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

echo "=== restore_session: full-chain restore (.kvcache + .rscache) ==="
"${DEVS}/llama_test.sh" "${PROMPT}" "${PROMPT}" -- -ub 32 -b 32

echo ""
echo "=== analysis ==="
PASS=1

check_passage() {
  local logfile="$1"
  local label="$2"
  local n_found=0
  local n_total=${#CHECKS[@]}
  if [[ ! -f "${logfile}" ]]; then
    echo "FAIL [${label}] ${logfile} missing"
    PASS=0
    return
  fi
  for phrase in "${CHECKS[@]}"; do
    if grep -qF "${phrase}" "${logfile}"; then
      n_found=$((n_found + 1))
    else
      echo "  MISS [${label}] not found: ${phrase:0:60}..."
    fi
  done
  if [[ "${n_found}" -eq "${n_total}" ]]; then
    echo "PASS [${label}] ${logfile} contains all ${n_total} distinctive phrases"
  else
    echo "FAIL [${label}] ${logfile} contains ${n_found}/${n_total} distinctive phrases"
    PASS=0
  fi
}

check_passage "${DEVS}/prompt-1.log" "prime"
check_passage "${DEVS}/prompt-2.log" "restore"

RESTORED=$(grep -oE 'cached_tokens: [0-9]+' "${DEVS}/prompt-2.log" | head -1 | grep -oE '[0-9]+' || echo 0)
echo "restored cached_tokens: ${RESTORED}"
if [[ "${RESTORED}" -lt 256 ]]; then
  echo "FAIL restored fewer than 256 tokens (expected full chain)"
  PASS=0
fi


./llama_prompt_response_analysis.sh ./prompt-1.log ./prompt-2.log

# verify both file types exist on disk
KV_COUNT=$(find "${KV_CACHE_DIR}" -name '*.kvcache' 2>/dev/null | wc -l)
RS_COUNT=$(find "${KV_CACHE_DIR}" -name '*.rscache' 2>/dev/null | wc -l)
echo "on disk: ${KV_COUNT} .kvcache, ${RS_COUNT} .rscache"
if [[ "${KV_COUNT}" -lt 10 || "${RS_COUNT}" -lt 10 ]]; then
  echo "FAIL expected >= 10 of each file type"
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
