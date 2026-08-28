#!/usr/bin/env bash
# Send a completion request to the dev llama-server and print the cached-token
# count plus the generated text.
# Usage:
#   llama_prompt.sh "your prompt here" [max_tokens]
#   echo "prompt" | llama_prompt.sh - [max_tokens]     # read prompt from stdin
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROMPT="${1:-}"
MAX_TOKENS="${2:-16}"

if [[ "${PROMPT}" == "-" || -z "${PROMPT}" ]]; then
  PROMPT="$(cat)"
fi

RESP="$(curl -s -X POST "${LLAMA_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "${PROMPT}" --argjson m "${MAX_TOKENS}" \
       '{prompt: $p, max_tokens: $m, cache_prompt: true}')"
)"

echo "${RESP}" | jq -r '
  "cached_tokens: \(.usage.prompt_tokens_details.cached_tokens // 0)  " +
  "prompt_tokens: \(.usage.prompt_tokens)  " +
  "gen: \(.choices[0].text | gsub("\n"; " ") | .[0:120])"
'
