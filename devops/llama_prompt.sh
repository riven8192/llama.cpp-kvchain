#!/usr/bin/env bash
# Send a completion request to the dev llama-server and print the cached-token
# count plus the FULL generated text (no truncation).
# Usage:
#   llama_prompt.sh "your prompt here"
#   echo "prompt" | llama_prompt.sh -
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROMPT="${1:-}"

if [[ "${PROMPT}" == "-" || -z "${PROMPT}" ]]; then
  PROMPT="$(cat)"
fi

RESP="$(curl -s -X POST "${LLAMA_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "${PROMPT}" '{prompt: $p, cache_prompt: true}')"
)"

# print the cached/prompt token counts on one line, then the FULL generated text
# (no truncation) so callers can verify long outputs / KV-restore correctness
echo "${RESP}" | jq -r '
  "cached_tokens: \(.usage.prompt_tokens_details.cached_tokens // 0)  " +
  "prompt_tokens: \(.usage.prompt_tokens)"
'
echo "${RESP}" | jq -r '.choices[0].text'
