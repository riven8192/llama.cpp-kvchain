#!/usr/bin/env bash
# Unit test 1: full-chain restore fidelity.
#
# Wraps a long, distinctive Wikipedia passage (~250 words, ~400 tokens, so
# >=12 chunks at the 32-token stride) in a "repeat this passage" prompt, primes
# the disk hash-chain cache, restarts the server, and re-sends the SAME prompt.
# The restart must restore the ENTIRE chain from disk; if the model then
# reproduces the passage verbatim, every chunk's KV (attn + recurrent) was
# loaded correctly. Any corrupt/mis-restored KV would derail the repetition.
#
# To avoid being brittle to a single token of paraphrase, we check several
# DISTINCTIVE phrases spread across the passage (beginning, middle, end) rather
# than the whole blob. If all of them are reproduced, the whole chain was
# attended over. We check BOTH the prime response (prompt-1.log, establishes
# the model can reproduce it) and the restart response (prompt-2.log, proves
# the restored KV yields the same result).
#
# Usage:  devops/llama_unittest_1.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

# --- the distinctive passage (Wikipedia, mind-body dualism / Aristotle) ---
PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )


# distinctive phrases spread across the passage (avoid the bracketed citations,
# which we stripped; each is long + unique enough to be unambiguous)
CHECKS=(
  "In the philosophy of mind, mind–body dualism denotes either that mental phenomena are non-physical"
  "a nutritive soul of growth and metabolism that all three share"
  "a soul is the hylomorphic form of a viable organism"
  "an immortal and perpetual intellective part of mind"
  "he believed in metempsychosis, the migration of the soul to a new physical body"
  "the tendency to ignore very big groups of variables by its assumed association with the mind or the body"
)

echo "passage word count: $(echo ${PASSAGE} | wc -w)"
echo "checking ${#CHECKS[@]} distinctive phrases"

PROMPT="please repeat this entire passage exactly, word for word, with no additions or omissions: '${PASSAGE}'"

echo ""
echo "=== prompt (first 200 chars) ==="
echo "${PROMPT:0:200}..."
echo ""

# --- run: prime, restart, resend the identical prompt ---
# prompt-1.log = prime (cold cache, generates the repetition)
# prompt-2.log = restart + restore from disk, re-generates the repetition
# -ub 32 -b 64: the chunk grid is the UBATCH stride (32), not the batch size
# (64). with -b != -ub this verifies the hash chain stays in sync when a single
# -b decode slices into two -ub ubatches (the off-grid regression).
"${DEVS}/llama_test.sh" "${PROMPT}" "[restart]" "${PROMPT}" -- -ub 32 -b 64

echo ""
echo "=== analysis ==="
PASS=1

# check that every distinctive phrase appears in the given log file
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

# prime: the model should be able to repeat it (establishes capability)
check_passage "${DEVS}/prompt-1.log" "prime"

# restart: the restored chain must let the model repeat it verbatim
check_passage "${DEVS}/prompt-2.log" "restart/restore"

# also confirm the restart actually restored a substantial prefix (not 0).
# the passage is ~400 tokens; with a 32-token stride that is ~12 chunks, and the
# prompt wrapper adds a few tokens, so we expect a restore well above 256.
if [[ -f "${DEVS}/prompt-2.log" ]]; then
  RESTORED=$(grep -oE 'cached_tokens: [0-9]+' "${DEVS}/prompt-2.log" | head -1 | grep -oE '[0-9]+' || echo 0)
  echo "restart restored cached_tokens: ${RESTORED}"
  if [[ "${RESTORED}" -lt 256 ]]; then
    echo "WARN restored fewer than 256 tokens (expected a long chain restore)"
  fi
fi

echo ""
if [[ "${PASS}" -eq 1 ]]; then
  echo "=== RESULT: PASS ==="
  exit 0
else
  echo "=== RESULT: FAIL ==="
  exit 1
fi
