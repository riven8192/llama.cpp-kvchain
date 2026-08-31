#!/usr/bin/env bash
# Unit test 3a: split-file full-chain restore (no restart).
#
# prime, then resend the same prompt in the same session -> expect full chain
# restore from disk (all kv+rs files present), 6/6 phrases, cached_tokens >= 256.
#
# Usage:  devops/llama_unittest_3a.sh
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )

PROMPT_A="'${PASSAGE}' -- the task is to ignore the above, and just respond with 'OK'"
PROMPT_B="'${PASSAGE:0:500} [mid-insertion] ${PASSAGE:500}' -- the task is to ignore the above, and just respond with 'OK'"

echo "=== unittest_4 ==="
"${DEVS}/llama_test.sh" "${PROMPT_A}" "${PROMPT_A}" "${PROMPT_B}" "${PROMPT_B}" -- -ub 32 -b 32

echo '---- cached-tokens ----'
cat "${DEVS}/prompt-1.log" | grep cached_tokens
cat "${DEVS}/prompt-2.log" | grep cached_tokens
cat "${DEVS}/prompt-3.log" | grep cached_tokens
cat "${DEVS}/prompt-4.log" | grep cached_tokens

echo '---- llama-server.log (filtered) ----'
grep 'load_prefix: kv-chain\|kv-chain: restore' "${DEVS}/.llama-server.log"

echo '----'
