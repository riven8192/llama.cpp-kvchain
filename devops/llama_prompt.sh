#!/usr/bin/env bash
# Send a completion request to the dev llama-server in STREAM mode.
# Streams tokens to stdout as they arrive, prints cached-token count at the end.
# Usage:
#   llama_prompt.sh "your prompt here"
#   echo "prompt" | llama_prompt.sh -
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROMPT="${1:-}"

if [[ "${PROMPT}" == "-" || -z "${PROMPT}" ]]; then
  PROMPT="${PROMPT}"
  PROMPT="$(cat)"
fi

USAGE_LINE=""

while IFS= read -r line; do
  [[ "${line}" != data:* ]] && continue
  data="${line#data: }"
  [[ "${data}" == "[DONE]" ]] && continue
  # stream the text delta to stdout immediately (jq -j = raw, no added newline)
  text=$(echo "${data}" | jq -j '.choices[0].text // empty' 2>/dev/null || true)
  if [[ -n "${text}" ]]; then
    printf '%s' "${text}"
  fi
  # capture usage from the final chunk
  usage=$(echo "${data}" | jq -r 'if .usage then "cached_tokens: \(.usage.prompt_tokens_details.cached_tokens // 0)  prompt_tokens: \(.usage.prompt_tokens)" else empty end' 2>/dev/null || true)
  if [[ -n "${usage}" ]]; then
    USAGE_LINE="${usage}"
  fi
done < <(curl -s -N -X POST "${LLAMA_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "${PROMPT}" '{prompt: $p, cache_prompt: true, stream: true, max_tokens: 512}')")

echo ""
if [[ -n "${USAGE_LINE}" ]]; then
  echo "${USAGE_LINE}"
fi
