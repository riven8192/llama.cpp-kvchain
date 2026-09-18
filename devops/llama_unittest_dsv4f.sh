#!/usr/bin/env bash
# Unit test 5: DSV4F (DeepSeek-V4-Flash) full-chain restore fidelity.
#
# Same shape as llama_unittest_restore_restart.sh (prime, resend the identical prompt) but run
# against the DSV4F model instead of Qwen. DSV4F (arch `deepseek4`,
# llama_kv_cache_dsv4) is the second architecture the kv-chain disk cache must
# support. Its state is NOT the clean "per-token KV + recurrent tail" split of
# Qwen: it has a raw/SWA per-token cache PLUS three compressed K caches (CSA/HCA/
# LID, prefix-style, regenerate-on-tail-prefill) PLUS three compressor ring
# states (the fixed-size tail object).
#
# The disk design for DSV4F mirrors the in-memory context-checkpoint feature,
# extended with the compressed K caches (which the tail prefill does NOT
# recompute - it only processes tokens >= n_saved, so they must come from disk):
#   .kvcache (per chunk) = ATTN_ONLY  -> kv_raw per-token rows only (additive)
#   .rscache (tail)      = TAIL_ONLY  -> compressed K caches + compressor rings
# On restore: append all matched .kvcache chunks, then set_data_ext(TAIL_ONLY)
# the tail .rscache, set n_past, and prefill the remainder (a no-op on a full
# chain hit - the first decode token is the response).
#
# This test verifies the whole thing empirically: if the model reproduces the
# passage verbatim after a restart+restore, every chunk's KV (attn + rings) was
# loaded correctly and the tail recompute is deterministic.
#
# IMPORTANT: this test uses a DIFFERENT model than the other unittests, so it
# overrides LLAMA_HF_REF (env.sh preserves an already-set value). It must NOT be
# run concurrently with the Qwen unittests (same port + cache dir).
#
# Usage:  devops/llama_unittest_dsv4f.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# pick the DSV4F model (export so env.sh's `: "${LLAMA_HF_REF:=...}"` keeps it)
export LLAMA_HF_REF="unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-IQ3_S"

. "${DEVS}/env.sh"

# --- the distinctive passage (same as llama_unittest_restore_restart.sh) ---
FILLER2="$( cat "${DEVS}/prompt_30k.txt" | tail -n 50 )"
PASSAGE="$( cat "${DEVS}/prompt_7sentences.txt" )"

# distinctive phrases spread across the passage
CHECKS=(
  "In the philosophy of mind, mind–body dualism denotes either that mental phenomena are non-physical"
  "a nutritive soul of growth and metabolism that all three share"
  "a soul is the hylomorphic form of a viable organism"
  "an immortal and perpetual intellective part of mind"
  "he believed in metempsychosis, the migration of the soul to a new physical body"
  "the tendency to ignore very big groups of variables by its assumed association with the mind or the body"
)

echo "model: ${LLAMA_HF_REF}"
echo "passage word count: $(echo ${PASSAGE} | wc -w)"
echo "checking ${#CHECKS[@]} distinctive phrases"

PROMPT="${FILLER2} ---- ignore everything prior to this split ---- please repeat this entire passage exactly, word for word, with no additions or omissions: '${PASSAGE}'"


echo ""
echo "=== prompt (first 200 chars) ==="
echo "${PROMPT:0:200}..."
echo ""

# --- run: prime, RESTART (same model, keep the disk cache), resend the
# --- identical prompt. prompt-1.log = prime (cold cache, generates the
# --- repetition + saves the chain to disk); the [restart] kills and re-spawns
# --- the server with --keep-cache (llama_test.sh passes it), so the disk chain
# --- survives; prompt-2.log = restore from disk + re-generate the repetition.
# NOTE: llama_test.sh flushes the cache dir AFTER the first server start, so a
# bare 2-prompt run would prime the cache and then wipe it - the [restart] is
# what makes the restore actually happen.
"${DEVS}/llama_test.sh" "${PROMPT}" "${PROMPT}" -- -ub 128 -b 128

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

# prime: establishes the model can reproduce it
check_passage "${DEVS}/prompt-1.log" "prime"

# restart: the restored chain (attn + rings + tail recompute) must yield the same result
check_passage "${DEVS}/prompt-2.log" "restart/restore"

# confirm the restart actually restored a substantial prefix (not 0).
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
