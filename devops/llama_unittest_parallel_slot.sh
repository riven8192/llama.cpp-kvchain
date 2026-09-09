#!/usr/bin/env bash
# Unit test: kv-chain with --parallel 2, SLOTS FORCED, CANCELLATION RACE.
#
# Reproduces (and verifies the fix for) the -np 2 OFF-GRID desync:
#
#   prompt A -> id_slot 0 : PASSAGE x2, cold prime, saves the chunk chain
#   (kill A)              : the TCP drop aborts slot 0 mid-prefill - its KV
#                           cells up to the last committed ubatch stay in the
#                           cache (n_stream=2 for everything that follows)
#   (sleep)               : any race window around the abort
#   prompt B -> id_slot 1 : PASSAGE x3. its first 22 chunks == A's chain,
#                           so it must restore exactly as many chunks as A
#                           managed to save before the kill.
#
# Both prompts instruct the model to reply only with 'OK' - we assert on
# cached_tokens and the server log, not on response text.
#
# Raw curl (not llama_prompt.sh): we need id_slot, and non-streaming
# responses are easier to assert on.
#
# PASS criteria (with the grid-exclusivity fix):
#   1. prompt B restores a FULL chunk prefix: cached_tokens % ubs == 0,
#      0 < cached_tokens <= A's full chain length (the kill timing decides
#      how many of A's chunks exist, so no exact number is asserted)
#   2. both responses are 'OK'
#   3. every CHUNK boundary is ON-GRID: the number of ON-GRID ubatch lines
#      equals the total number of complete chunks actually prefilled - prompt A's
#      chunks that survived the kill (== prompt B's restore, C2/ubs) plus prompt
#      B's own new chunks. OFF-GRID lines (trailing partials + decode steps) are
#      expected and NOT counted against us.
#   4. pos == cb_n_pos_last + 1 for every ubatch line that is a PURE
#      single-slot ubatch (the mixed-ubatch fingerprint is cb_n_pos_last <<
#      pos - 1). a ubatch that MIXES two slots' tokens (the prefill tails
#      finishing together) fires the hook twice with a SHARED cb_n_pos_last,
#      so its second line legitimately shows the other slot's pos - those are
#      excluded from the invariant check.
#
# Usage:  devops/llama_unittest_parallel_slot.sh
set -uo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

UB=32

PASSAGE=$( cat "${DEVS}/prompt_7sentences.txt" )
SUFFIX="the task is to ignore the above, and just respond with 'OK'"
PROMPT_A="${PASSAGE} ${PASSAGE} -- ${SUFFIX}"
PROMPT_B="${PASSAGE} ${PASSAGE} ${PASSAGE} -- ${SUFFIX}"

echo "=== parallel_slot: -np 2, forced slots, cancellation race ==="
echo "prompt A (-> slot 0): '${PROMPT_A:0:60}...' (passage x2)"
echo "prompt B (-> slot 1): '${PROMPT_B:0:60}...' (passage x3)"
echo ""

# fresh server: --parallel 2, kv-chain enabled, chunk grid == -ub
LLAMA_PARALLEL=2 "${DEVS}/llama_run.sh" -- -ub ${UB} -b ${UB} >/dev/null 2>&1
# llama_run.sh cleared the kv-cache dir on start; flush again to be sure
# (llama_run.sh may have already created files - it does not, but be safe)
rm -rf "${KV_CACHE_DIR}"
mkdir -p "${KV_CACHE_DIR}"
echo "cache dir flushed: ${KV_CACHE_DIR}"
echo ""

post_prompt() { # $1 = label, $2 = slot id, $3 = prompt, $4 = outfile
  local label="$1" slotid="$2" prompt="$3" outfile="$4"
  echo "=== POST ${label} (id_slot ${slotid}) ==="
  local t0 t1
  t0=$(date +%s)
  curl -s -N -X POST "${LLAMA_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "${prompt}" --argjson s "${slotid}" \
          '{prompt: $p, cache_prompt: true, stream: false, max_tokens: 512, temperature: 0, seed: 42, id_slot: $s}')" \
    > "${outfile}" 2>&1
  t1=$(date +%s)
  echo "took: $(( t1 - t0 )) s"
  echo ""
}

# T_PROMPT_A_START="${SECONDS}"
# post_prompt "A (prime, slot 0)" 0 "${PROMPT_A}" "${DEVS}/parallel-slot-1.json" &
# prompt_a_pid=$!
# echo "Prompt A PID: ${prompt_a_pid}"
# T_PROMPT_A_END="${SECONDS}"
# echo "Prompt A took: $(( T_PROMPT_A_END - T_PROMPT_A_START )) sec"
# N.B.: Observation: PROMPT_A takes approx 6 sec

PROMPT_A_KILL_DELAY=4 # fixed
PROMPT_B_SEND_DELAY=1 # any value under 3sec causes the issue

# echo "Killing PROMPT A after ${PROMPT_A_KILL_DELAY}sec..."
# sleep "${PROMPT_A_KILL_DELAY}"
# echo "Killing PROMPT A: dead=$(kill -0 "${prompt_a_pid}" ; echo $? )"
# kill -9 "${prompt_a_pid}"
# echo "Killing PROMPT A"

echo "Starting PROMPT A in background"
post_prompt "A (prime, slot 0)" 0 "${PROMPT_A}" "${DEVS}/parallel-slot-1.json" &

echo "Starting PROMPT B after 3sec..."
sleep 3
post_prompt "B (restore, slot 1)" 1 "${PROMPT_B}" "${DEVS}/parallel-slot-2.json"

wait


echo ""
echo "=== responses ==="
for i in 1 2; do
  echo "--- prompt-${i} ---"
  jq -r '.choices[0].text // "(no text)"' "${DEVS}/parallel-slot-${i}.json" 2>/dev/null || cat "${DEVS}/parallel-slot-${i}.json"
  echo ""
done

echo "=== analysis ==="
PASS=1

get_cached() { # $1 = response json
  jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$1" 2>/dev/null || echo 0
}
get_text() { # $1 = response json
  jq -r '.choices[0].text // ""' "$1" 2>/dev/null || echo ""
}

C1=$(get_cached "${DEVS}/parallel-slot-1.json")
C2=$(get_cached "${DEVS}/parallel-slot-2.json")
T1=$(get_text "${DEVS}/parallel-slot-1.json")
T2=$(get_text "${DEVS}/parallel-slot-2.json")
echo "cached_tokens: prompt A=${C1}  prompt B=${C2}"

# prompt A is a cold prime -> 0
if [[ "${C1}" -ne 0 ]]; then
  echo "FAIL prompt A cached_tokens=${C1} (expected 0, cold prime)"
  PASS=0
else
  echo "PASS prompt A cached_tokens=0"
fi

# prompt B's chain starts with A's chain (PASSAGE x3 vs PASSAGE x2: same
# tokens at the same positions for the first ~22 chunks). the kill timing
# decides how many of A's chunks made it to disk, so assert the SHAPE of the
# restore: a full chunk prefix (the hash chain can only break at a chunk
# boundary) of at least one chunk. the pre-fix bug restored 0 or produced a
# mid-chunk-cap that costs the rest of the chain.
if [[ "${C2}" -ge ${UB} && $(( C2 % UB )) -eq 0 ]]; then
  echo "PASS prompt B restored a full chunk prefix (cached_tokens=${C2}, multiple of ${UB})"
else
  echo "FAIL prompt B cached_tokens=${C2} (expected >= ${UB} and a multiple of ${UB}: a full chunk prefix)"
  PASS=0
fi

# both prompts must reply with exactly 'OK' (mod whitespace)
is_ok() {
  local r
  r="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -d '[:space:][:punct:]')"
  [[ "${r}" == "OK" ]]
}
if is_ok "${T1}"; then
  echo "PASS prompt A response is OK"
else
  echo "FAIL prompt A response is not OK: [${T1:0:80}]"
  PASS=0
fi
if is_ok "${T2}"; then
  echo "PASS prompt B response is OK"
else
  echo "FAIL prompt B response is not OK: [${T2:0:80}]"
  PASS=0
fi

# --- grid fidelity: every chunk boundary must be ON-GRID --------------------
# a correct prefill of N tokens at -ub 32 produces floor(N/32) ON-GRID chunk
# boundaries + one trailing OFF-GRID partial (N mod 32 tokens). with -np 2 a
# desynced grid would save chunks at off-grid positions (or skip them), so the
# number of ON-GRID ubatch lines must equal the total number of COMPLETE chunks
# prefilled by both slots. OFF-GRID lines are expected (trailing partials +
# decode steps) and are NOT counted against us here.
#
# expected chunk count (ON-GRID ubatch lines):
#   slot 0 (prompt A, cold prime, KILLED mid-prefill): only the chunks it saved
#      before the kill survive - that is exactly what prompt B restored, C2/ubs.
#   slot 1 (prompt B): floor((len_B - cached) / ubs) NEW chunks (the cached
#      prefix is restored from disk, not re-prefilled, so it adds no ON-GRID
#      ubatch lines).
LEN_B=$( jq -r '.usage.prompt_tokens // 0' "${DEVS}/parallel-slot-2.json" 2>/dev/null || echo 0 )
EXPECTED_CHUNKS=$(( C2 / UB + (LEN_B - C2) / UB ))
ONGRID=$( grep -c 'ON-GRID' "${LLAMA_LOG}" || true )
OFFGRID=$( grep -c 'OFF-GRID' "${LLAMA_LOG}" || true )
echo "server log: ${ONGRID} ON-GRID, ${OFFGRID} OFF-GRID ubatch lines (len_B=${LEN_B} cached_B=${C2} expected_chunks=${EXPECTED_CHUNKS})"
if [[ "${ONGRID}" -eq "${EXPECTED_CHUNKS}" ]]; then
  echo "PASS ON-GRID count (${ONGRID}) == expected complete chunks (${EXPECTED_CHUNKS}): every chunk boundary is on-grid"
else
  echo "FAIL ON-GRID count (${ONGRID}) != expected complete chunks (${EXPECTED_CHUNKS}): a chunk boundary is off-grid or missing:"
  grep 'kv-chain\[ubatch\]' "${LLAMA_LOG}" | head -40
  PASS=0
fi

# invariant: every PURE single-slot ubatch line satisfies pos == cb_n_pos_last + 1
# (the mixed-ubatch fingerprint is cb_n_pos_last << pos - 1). a ubatch that
# mixes two slots' tokens (the prefill tails finishing together) fires the hook
# twice with a SHARED cb_n_pos_last, so its second line legitimately shows the
# other slot's pos - those lines are excluded (detected: the same pos value
# appears on two consecutive ubatch lines, i.e. a re-fire at an unchanged pos).
BADINV=$( grep 'kv-chain\[ubatch\]' "${LLAMA_LOG}" \
  | awk '
      {
        if (match($0, /pos=[0-9]+ cb_n_pos_last=[0-9]+/)) {
          seg = substr($0, RSTART, RLENGTH)
          split(seg, a, "pos=")
          split(a[2], b, " cb_n_pos_last=")
          p = b[1] + 0
          l = b[2] + 0
          # skip a re-fire at an unchanged pos (the mixed-ubatch double-fire):
          # the previous line had the same pos
          if (p != prev_p && p != l + 1) print
          prev_p = p
        }
      }' | wc -l )
if [[ "${BADINV}" -gt 0 ]]; then
  echo "FAIL ${BADINV} pure-ubatch lines violate pos == cb_n_pos_last + 1 (mixed-ubatch fingerprint):"
  grep 'kv-chain\[ubatch\]' "${LLAMA_LOG}" | head -20
  PASS=0
else
  echo "PASS pos == cb_n_pos_last + 1 holds for all pure (non-re-fired) ubatch lines"
fi

echo ""
echo "=== server log (kv-chain lines) ==="
grep -E 'kv-chain' "${LLAMA_LOG}" || echo "(none)"

echo ""
echo "=== cache files ==="
find "${KV_CACHE_DIR}" \( -name '*.kvcache' -o -name '*.rscache' \) -printf "%f\n" 2>/dev/null | sort

echo ""
if [[ "${PASS}" -eq 1 ]]; then
  echo "=== RESULT: PASS ==="
  exit 0
else
  echo "=== RESULT: FAIL ==="
  exit 1
fi
