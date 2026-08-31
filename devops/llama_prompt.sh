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

# the SSE stream is parsed by python3 (native json lib), NOT by bash:
# a streamed text delta like "red\n" must reach stdout byte-for-byte - any
# bash string handling ($( ...), read, printf) risks mangling or stripping
# the newlines. the python side:
#   - writes each text delta to stdout IMMEDIATELY (flush per chunk), so the
#     stream stays live for long generations
#   - writes the final usage line to STDERR, so it never interleaves with
#     the (newline-free-at-the-end) streamed text on stdout
# temperature hardcoded to 0 in the body: deterministic output so the
# restore-fidelity tests (byte-identical-vs-prefill) are not flaked by sampling
curl -s -N -X POST "${LLAMA_URL}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg p "${PROMPT}" '{prompt: $p, cache_prompt: true, stream: true, max_tokens: 512, temperature: 0}')" \
| python3 -c '
import json, sys

usage = None
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"):
        continue
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        continue
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        continue
    choice = (chunk.get("choices") or [{}])[0]
    text = choice.get("text")
    if text:
        sys.stdout.write(text)
        sys.stdout.flush()
    if chunk.get("usage"):
        usage = chunk["usage"]

sys.stdout.write("\n")
sys.stdout.flush()
if usage:
    d = usage.get("prompt_tokens_details", {})
    sys.stderr.write("cached_tokens: %s  prompt_tokens: %s\n"
                     % (d.get("cached_tokens", 0), usage.get("prompt_tokens", 0)))
'
