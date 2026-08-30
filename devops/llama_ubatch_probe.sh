#!/usr/bin/env bash
# Ubatch-boundary probe for the kv-chain hash-cache.
#
# Sends ONE long prompt (from devops/prompt_30k.txt, ~10K-30K tokens) to the dev
# server and lets the server's debug logging capture the real ubatch split. We
# then analyze the .llama-server.log for:
#   kv-chain[decode]:  one line per highlevel decode() call (the outer slice loop)
#   kv-chain[ubatch]:  one line per internal ubatch (pos, cb_n_pos_last, ub_n, ub_pos range, on/off-grid)
#
# The question under investigation: do the ubatch boundaries land on the
# n_ubatch-grid for the WHOLE prompt, or are there ragged (off-grid) boundaries
# mid-prompt? A mid-prompt off-grid boundary desyncs the hash chain. (Resolved:
# the only raggedness is the single trailing N mod n_ubatch partial chunk, once
# context-checkpoints are disabled for kv-chain - see NOTES.md section 5.)
#
# Usage:
#   1. Put a long prompt (>= ~10K tokens, i.e. ~7500+ words) in devops/prompt_30k.txt.
#   2. Make sure the server is running with -b == -ub, e.g.:
#        devops/llama_run.sh -- -ub 2048 -b 2048 -c 65536
#   3. In ANOTHER terminal, watch the log:  tail -f devops/.llama-server.log
#   4. Run this script:                    devops/llama_ubatch_probe.sh
#   5. When it returns, the log has the full boundary sequence (also printed here).
#
# NOTE: this sends a REAL prefill to the server. On this hardware a ~30K-token
# prompt at -b 2048 takes a few minutes. Do NOT point this at the user's own
# production llama-server (different port / no kv-chain); use the dev server.
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

# ---------------------------------------------------------------------------
# the long prompt lives in devops/prompt_30k.txt (git-ignored; paste your own
# long text there). resolve it against $DEVS so the script works from any CWD.
# a short instruction is appended so the model replies with just 'OK' (we only
# care about the PREFILL ubatch sequence, not the generation).
PROMPT_FILE="${DEVS}/prompt_30k.txt"
if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "error: prompt file not found: ${PROMPT_FILE}" >&2
  echo "       paste a long prompt (>= ~10K tokens) into it first." >&2
  exit 1
fi
PROMPT=$( cat "${PROMPT_FILE}" && echo "--- Ignore the above text, and respond only with 'OK'. " )

# sanity check: warn if the prompt looks too short to span many ubatches
NWORDS=$(echo "${PROMPT}" | wc -w)
echo "prompt word count: ${NWORDS} (target >= ~7500 for ~10K tokens / ~40 ubatches of 256)"
if [[ "${NWORDS}" -lt 400 ]]; then
  echo "WARNING: prompt looks short; the ubatch sequence may not be representative." >&2
fi

echo ""
echo "=== sending prompt to ${LLAMA_URL} ==="
echo "(watch: tail -f ${LLAMA_LOG})"
echo ""

# send it; llama_prompt.sh prints cached_tokens + prompt_tokens, then the text.
# we only care about the prompt_tokens count + the server log, so truncate the
# generated text in our own output (it is still fully in the response, just not
# echoed here).
PROMPT_START=$(date +%s)
RESP="$(curl -s -X POST "${LLAMA_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "${PROMPT}" '{prompt: $p, cache_prompt: true}')"
)"
PROMPT_END=$(date +%s)

echo "${RESP}" | jq -r '
  "cached_tokens: \(.usage.prompt_tokens_details.cached_tokens // 0)  " +
  "prompt_tokens: \(.usage.prompt_tokens)"
'
echo ""
echo "Took: $(( PROMPT_END - PROMPT_START )) seconds"
echo ""
echo "=== now analyze the ubatch boundaries in ${LLAMA_LOG} ==="
echo "decode() calls (one per outer slice):"
grep 'kv-chain\[decode\]' "${LLAMA_LOG}" || echo "  (none found - is kv-chain enabled?)"
echo ""
echo "ubatch boundaries (one per internal ubatch):"
grep 'kv-chain\[ubatch\]' "${LLAMA_LOG}" || echo "  (none found)"
