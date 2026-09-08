#!/usr/bin/env bash

# set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )
PROMPT="${PASSAGE} -- please ignore the above text, and only reply with 'OK'"


LLAMA_PARALLEL=1 LLAMA_CTX=4096 "${DEVS}/llama_test.sh" "${PROMPT}" "${PROMPT}" -- -ub 32 -b 32
echo '#### llama-server.log (full)'
cat "${LLAMA_LOG}"
echo '#### llama-server.log (warnings)'
cat "${LLAMA_LOG}" | grep -v ' unused ' | grep ' W '
echo '#### llama-server.log (stacktrace)'
cat "${LLAMA_LOG}" | grep 'llama.cpp/build-vulkan/bin'
echo '#### prompt-1 response'
cat "${DEVS}/prompt-1.log"
echo '#### prompt-2 response'
cat "${DEVS}/prompt-2.log"
echo '#### cache files:'
ls -lh "${DEVS}/.kv-cache/"
echo '####'

echo '------------------------------------------'
echo '------------------------------------------'
echo '------------------------------------------'

LLAMA_PARALLEL=4 LLAMA_CTX=131072 "${DEVS}/llama_test.sh" "${PROMPT}" "${PROMPT}" -- -ub 32 -b 32
echo '#### llama-server.log (full)'
cat "${LLAMA_LOG}"
echo '#### llama-server.log (warnings)'
cat "${LLAMA_LOG}" | grep -v ' unused ' | grep ' W '
echo '#### llama-server.log (stacktrace)'
cat "${LLAMA_LOG}" | grep 'llama.cpp/build-vulkan/bin'
echo '#### prompt-1 response'
cat "${DEVS}/prompt-1.log"
echo '#### prompt-2 response'
cat "${DEVS}/prompt-2.log"
echo '#### cache files:'
ls -lh "${DEVS}/.kv-cache/"
echo '####'

echo '------------------------------------------'
echo '------------------------------------------'
echo '------------------------------------------'

for idx in $( seq 1 8 ) ; do
    "${DEVS}/llama_prompt.sh" "${PROMPT}" >"${DEVS}/async-${idx}.log" 2>&1 &
done

wait

echo '#### selected slots by id'
cat "${LLAMA_LOG}" | grep ' selected slot by '
