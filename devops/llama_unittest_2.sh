#!/usr/bin/env bash
# Unit test 2: native (in-memory) prefix caching still works with kv-chain disabled.
#
# This is the zero-behavior-change check for the "disable native caching when
# kv-chain is enabled" change: run the server WITHOUT --kv-chain-dir (the old
# implementation), send a long distinctive passage, then send the SAME prompt a
# second time in the same session (NO restart - the in-memory KV cache does not
# survive a restart). The native get_common_prefix() must kick in and reuse the
# whole prompt from memory, and the model must reproduce the passage verbatim.
#
# The passage and checks are identical to llama_unittest_1.sh so both tests
# exercise the same prompt.
#
# Usage:  devops/llama_unittest_2.sh
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

# --- run: start the server WITHOUT the disk cache (native impl), send the same
# --- prompt twice in the same session (no restart - the in-memory KV cache
# --- does not survive a restart). prompt-1.log = prime (cold cache),
# --- prompt-2.log = native in-memory prefix reuse of the same prompt.
rm -f "${DEVS}"/prompt-*.log

echo "=== [1/4] llama_build.sh ==="
"${DEVS}/llama_build.sh" | tail -3
echo ""

echo "=== [2/4] llama_run.sh --no-kv-chain ==="
"${DEVS}/llama_run.sh" --no-kv-chain -- -ub 32 -b 32
echo ""

echo "=== [3/4] llama_prompt.sh (prime) ==="
PROMPT_START=$(date +%s)
"${DEVS}/llama_prompt.sh" "${PROMPT}" 2>&1 | tee "${DEVS}/prompt-1.log" || true
PROMPT_END=$(date +%s)
echo ""
echo "Took: $(( PROMPT_END - PROMPT_START )) seconds"
echo ""

echo "=== [4/4] llama_prompt.sh (repeat, same session, native reuse) ==="
PROMPT_START=$(date +%s)
"${DEVS}/llama_prompt.sh" "${PROMPT}" 2>&1 | tee "${DEVS}/prompt-2.log" || true
PROMPT_END=$(date +%s)
echo ""
echo "Took: $(( PROMPT_END - PROMPT_START )) seconds"
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

# repeat: the native in-memory prefix reuse must let the model repeat it verbatim
check_passage "${DEVS}/prompt-2.log" "repeat/native-reuse"

# confirm the repeat actually reused a substantial prefix from memory (not 0).
# the prompt is ~374 tokens; the native get_common_prefix() should reuse the
# whole prompt (minus the last token, which is re-evaluated for the logits).
if [[ -f "${DEVS}/prompt-2.log" ]]; then
  CACHED=$(grep -oE 'cached_tokens: [0-9]+' "${DEVS}/prompt-2.log" | head -1 | grep -oE '[0-9]+' || echo 0)
  echo "repeat reused cached_tokens: ${CACHED}"
  if [[ "${CACHED}" -lt 256 ]]; then
    echo "FAIL repeat reused fewer than 256 tokens (native in-memory prefix reuse did not kick in)"
    PASS=0
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
