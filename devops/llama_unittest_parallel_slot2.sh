#!/usr/bin/env bash


set -uo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

UB=32

PASSAGE=$( cat "${DEVS}/prompt_30k.txt" )
PROMPT_A="please write a long story about cute fluffy bunnies from the perspective of a toddler"
PROMPT_B="${PASSAGE}"

echo "=== parallel_slot: -np 2, forced slots, cancellation race ==="
echo "prompt A (-> slot 0): '${PROMPT_A:0:60}...'"
echo "prompt B (-> slot 1): '${PROMPT_B:0:60}...'"
echo ""

echo "Starting llama-server..."
LLAMA_CTX=$((64*1024)) LLAMA_PARALLEL=2 "${DEVS}/llama_run.sh" -- -ub ${UB} -b ${UB} >/dev/null 2>&1
echo "Started llama-server."


post_prompt() { # $1 = label, $2 = slot id, $3 = prompt, $4 = outfile
  local label="$1" slotid="$2" prompt="$3" outfile="$4"
  echo "=== POST ${label} (id_slot ${slotid}) ==="
  local t0 t1
  t0=$(date +%s)
  curl -s -N -X POST "${LLAMA_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "${prompt}" --argjson s "${slotid}" \
          '{prompt: $p, cache_prompt: true, stream: false, max_tokens: 32768, temperature: 0, seed: 42, id_slot: $s}')" \
    > "${outfile}" 2>&1
  t1=$(date +%s)
  echo "took: $(( t1 - t0 )) s"
  echo ""
}

echo "Starting PROMPT A in background"
post_prompt "A (decode heavy, slot 0)" 0 "${PROMPT_A}" "${DEVS}/parallel-slot-1.json" &
prompt_a_pid=$!

echo "Starting PROMPT B after 10sec..."
sleep 10
post_prompt "B (prefill heavy, slot 1)" 1 "${PROMPT_B}" "${DEVS}/parallel-slot-2.json"

sleep 10
echo 'killing PROMPT A'
echo kill "${prompt_a_pid}"
kill "${prompt_a_pid}"




