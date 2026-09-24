#!/bin/bash
#
# Fast kv-chain unit tests. One server session for the whole run.
#
# Scenarios (cheapest first, strongest last):
#   hello       : trivial prompt gets a real answer
#   ubs-edge    : prompt lengths 255/256/257 around an ubs boundary
#   passage     : prime + same-session restore (0 -> 352), 2-token reply pinned
#   forked      : A/A/B/B hash-chain fork (0 / 352 / 96 / 352) + rs-touch WARN
#   evict-rs    : middle .rscache deleted -> tail rs still usable (full restore)
#   evict-kv    : middle .kvcache deleted -> chain breaks there (cached 64)
#
# Usage:  devops/llama_unittests.sh
# Exit:   0 = all PASS, 1 = at least one FAIL
# Takes a few minutes (vs the full llama_run_unittests.sh pass).
#
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

cd "${DEVS}"

output_base_dir="${DEVS}/.unittest-output"
rm -rf "${output_base_dir}"
mkdir -p "${output_base_dir}"

# --- pass/fail accounting ------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
FAILURES=()

record() { # $1 = ok(0/1), $2 = label
	if [[ "$1" -eq 0 ]]; then
		PASS_COUNT=$((PASS_COUNT + 1))
		echo "PASS  $2"
	else
		FAIL_COUNT=$((FAIL_COUNT + 1))
		FAILURES+=("$2")
		echo "FAIL  $2"
	fi
}

warn() { # non-fatal: prints, counts, never fails the run
	WARN_COUNT=$((WARN_COUNT + 1))
	echo "WARN  $1"
}

# --- per-prompt artifacts --------------------------------------------------------
get_file() {
	local label="$1"
	local what="$2"
	echo "${output_base_dir}/${label}-${what}.txt"
}

do_prompt() {
	local label="$1"
	local prompt="$2"
	echo "[prompt:${label}]"

	if [ -z "${prompt}" ] ; then\
		echo "EMPTY_PROMPT"
		exit 13
	fi

	serverlog_lines_pre="$(wc -l <"${LLAMA_LOG}")"
	timestamp_pre="${SECONDS}"

	echo "${prompt}" >"$(get_file "${label}" "prompt")"
	./llama_prompt.sh "${prompt}" \
		1>"$(get_file "${label}" "response")" \
		2>"$(get_file "${label}" "stats")" || true

	timestamp_post="${SECONDS}"
	serverlog_lines_post="$(wc -l <"${LLAMA_LOG}")"
	echo "prompt_took: $(( timestamp_post - timestamp_pre)) sec" >>"$(get_file "${label}" "stats")"

	awk "FNR>=${serverlog_lines_pre} && FNR<=${serverlog_lines_post}" "${LLAMA_LOG}" \
		1>"$(get_file "${label}" "serverlog")"

	echo "  prompt:   $(get_file "${label}" "prompt")"
	echo "  response: $(get_file "${label}" "response")"
	echo "  stats:    $(get_file "${label}" "stats")"
	echo "  server:   $(get_file "${label}" "serverlog")"
	cat "$(get_file "${label}" "stats")"
}

flush_cache() {
	echo "[flush]"
	rm -rf "${KV_CACHE_DIR}"
	mkdir -p "${KV_CACHE_DIR}"
}

# --- fixtures ---------------------------------------------------------------------
get_hosted_model() {
	curl -s "${LLAMA_URL}/v1/models" | \
		jq -r '.data[].id'
}

# build a prompt that tokenizes to EXACTLY $1 tokens against the currently
# hosted model/tokenizer, while retaining the instruction to output as few
# tokens as possible. converges via the server's prompt_tokens count.
# count the prompt_tokens the server reports for a given prompt text
count_prompt_tokens() {
	./llama_prompt.sh "$1" -- 2>&1 >/dev/null | grep -oE 'prompt_tokens: [0-9]+' | grep -oE '[0-9]+'
}

make_exact_prompt_len() {
	local target="$1"

	local prompt_base="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. Bright vixens jump; dozy fowl quack. Jackdaws love my big sphinx of quartz. The quick brown fox jumps again, twice, thrice, and once more for good measure."
	local prompt_suffix="--- the task is to ignore the above, and just respond with 'OK'"

	local prompt_smaller=""
	local prompt_try got
	while : ; do
		if [ -z "${prompt_smaller}" ] ; then
			prompt_try="${prompt_base}"
		else
			prompt_try="${prompt_smaller} ${prompt_base}"
		fi
		# the double space is intentional (same as: "... ${BASE:0:0} ...")
		got="$(count_prompt_tokens "${prompt_try}  ${prompt_suffix}")"
		if [ "${got}" -ge "${target}" ] ; then
			break
		fi
		prompt_smaller="${prompt_try}"
	done

	local sublen=$(( ${#prompt_base} / 2 ))
	local prompt_sized
	while : ; do
		prompt_sized="${prompt_smaller} ${prompt_base:0:${sublen}} ${prompt_suffix}"
		got="$(count_prompt_tokens "${prompt_sized}")"
		if [ "${got}" -gt $(( target + 10 )) ] ; then sublen=$(( sublen - 20 ))
		elif [ "${got}" -lt $(( target - 10 )) ] ; then sublen=$(( sublen + 20 ))
		elif [ "${got}" -gt $(( target + 5 )) ] ; then sublen=$(( sublen - 10 ))
		elif [ "${got}" -lt $(( target - 5 )) ] ; then sublen=$(( sublen + 10 ))
		elif [ "${got}" -gt "${target}" ] ; then sublen=$(( sublen - 1 ))
		elif [ "${got}" -lt "${target}" ] ; then sublen=$(( sublen + 1 ))
		else break ; fi
	done
	echo "${prompt_sized}"
}

construct_fixed_size_prompt() {
	local size="$1"
	local model="$(get_hosted_model)"
	local hash="$(echo "${model}-${size}-v3" | md5sum - | awk '{print $1}')"
	local filename="${DEVS}/.fixed-size-prompt-${hash}.txt"
	if [ ! -f "${filename}" ] ; then
		make_exact_prompt_len "${size}" >"${filename}"
	fi
	cat "${filename}"
}

# the 7-sentence Wikipedia passage (mind-body dualism / Aristotle) + suffix that
# pins the reply to exactly 2 tokens. prompt tokenizes to 365 = 11 full ubs
# chunks (352) + a 13-token tail (never saved).
PROMPT_PASSAGE="$(cat "${DEVS}/prompt_7sentences.txt")"
PROMPT_7L="'${PROMPT_PASSAGE}' -- the task is to ignore the above, and just respond with 'OK'"
# B = the same passage with a [mid-insertion] spliced at char 500: the first 3
# chunks (96 tokens) are identical to A, then the chain diverges. prompt B
# tokenizes to 381 = 96 shared + 285 diverging.
PROMPT_7L_B="'${PROMPT_PASSAGE:0:500}' [mid-insertion] '${PROMPT_PASSAGE:500}' -- the task is to ignore the above, and just respond with 'OK'"

# distinctive phrases spread across the passage (coherence oracle: if ALL of
# them come back, the model attended over the whole restored prefix)
CHECKS=(
	"In the philosophy of mind, mind–body dualism denotes either that mental phenomena are non-physical"
	"a nutritive soul of growth and metabolism that all three share"
	"a soul is the hylomorphic form of a viable organism"
	"an immortal and perpetual intellective part of mind"
	"he believed in metempsychosis, the migration of the soul to a new physical body"
	"the tendency to ignore very big groups of variables by its assumed association with the mind or the body"
)

# --- assertions ---------------------------------------------------------------------
# each require_* reads a per-prompt artifact, compares, and records PASS/FAIL.
# stats lines look like "cached_tokens: 352" (key: value, one key per line).
get_stat() { # $1 = label, $2 = key; prints the value, or nothing if absent
	local label="$1"
	local key="$2"
	awk -v k="${key}:" 'index($0, k) { sub(/^[^:]*: */, ""); print; exit }' \
		"$(get_file "${label}" "stats")"
}

require_stat() {
	local label="$1"
	local key="$2"
	local expected="$3"
	local actual
	actual="$(get_stat "${label}" "${key}")"
	if [ -z "${actual}" ] ; then
		record 1 "${label}: ${key} == ${expected} (stat line missing - server dead?)"
		return
	fi
	if [ "${expected}" = "${actual}" ] ; then
		record 0 "${label}: ${key} == ${expected}"
	else
		record 1 "${label}: ${key} == ${expected} (got: ${actual})"
	fi
}

require_cached_tokens() {
	require_stat "$1" cached_tokens "$2"
}

require_response_tokens() {
	require_stat "$1" response_tokens "$2"
}

require_response_contains() {
	local label="$1"
	local expected="$2"
	if grep -qF -- "${expected}" "$(get_file "${label}" "response")" 2>/dev/null ; then
		record 0 "${label}: response contains '${expected:0:40}...'"
	else
		record 1 "${label}: response contains '${expected:0:40}...'"
	fi
}

# coherence: every distinctive phrase must be in the response. used where the
# reply is a full passage repetition (NOT for the 2-token 'OK' prompts).
require_all_phrases() {
	local label="$1"
	local n_found=0
	for phrase in "${CHECKS[@]}" ; do
		if grep -qF -- "${phrase}" "$(get_file "${label}" "response")" 2>/dev/null ; then
			n_found=$((n_found + 1))
		fi
	done
	if [ "${n_found}" -eq "${#CHECKS[@]}" ] ; then
		record 0 "${label}: response has all ${#CHECKS[@]} passage phrases"
	else
		record 1 "${label}: response has ${n_found}/${#CHECKS[@]} passage phrases"
	fi
}

# the reply to PROMPT_7L* must be exactly 'OK' (mod case/whitespace/punct).
# a confident-wrong ramble is the sneaky restore-corruption failure mode.
require_ok_response() {
	local label="$1"
	local r
	r="$(tr '[:lower:]' '[:upper:]' <"$(get_file "${label}" "response")" | tr -d '[:space:][:punct:]')"
	if [ "${r}" = "OK" ] ; then
		record 0 "${label}: response is exactly OK"
	else
		record 1 "${label}: response is exactly OK (got: [$(head -c 80 "$(get_file "${label}" "response")")...])"
	fi
}

# --- the smoke oracle: N response files must be byte-identical (after dropping
# the cached_tokens line and blank lines) -----------------------------------------
smoke_compare() {
	local -a labels=("$@")
	local baseline=""
	local ok=0
	for label in "${labels[@]}" ; do
		local cleaned
		cleaned="$(grep -v 'cached_tokens' "$(get_file "${label}" "response")" | sed '/^$/d' | tr '\n' ' ' | sed 's/ \+/ /g')"
		if [ -z "${cleaned}" ] ; then
			ok=1
			break
		fi
		if [ -z "${baseline}" ] ; then
			baseline="${cleaned}"
		elif [ "${cleaned}" != "${baseline}" ] ; then
			ok=1
			break
		fi
	done
	if [ "${ok}" -eq 0 ] ; then
		record 0 "smoke: responses ${labels[*]} byte-identical"
	else
		record 1 "smoke: responses ${labels[*]} byte-identical"
	fi
}

# --- server helpers ------------------------------------------------------------------
# start the kv-chain server (the default for the whole run)
start_kv_server() {
	LLAMA_CTX="$(( 64*1024 ))" \
	LLAMA_HF_REF='unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL' \
		./llama_run.sh -- -b 32 -ub 32
}

server_alive() {
	local pid
	pid="$(cat "${LLAMA_PIDFILE}" 2>/dev/null || true)"
	[ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

check_hello() {
	echo "=== hello: trivial prompt gets a real answer ==="
	do_prompt "hello-a" "hello world"
	require_response_contains "hello-a" "help"
	require_response_contains "hello-a" "today"
	echo
}

check_ubatch_edge_cases() {
	echo "=== ubs-edge: prompt lengths 255/256/257 around the 256 (=8*ubs) boundary ==="
	#   255 -> 7 full chunks saved (224); the 1-token tail chunk is never saved
	#   256 -> exact multiple: the last token must stay re-prefilled -> 224
	#   257 -> 8 full chunks (256) + 1 leftover token
	flush_cache

	do_prompt "ubs-255t-a" "$(construct_fixed_size_prompt 255)"
	do_prompt "ubs-255t-b" "$(construct_fixed_size_prompt 255)"
	require_cached_tokens "ubs-255t-b" "$(( 256 - 32 ))"

	do_prompt "ubs-256t-a" "$(construct_fixed_size_prompt 256)"
	do_prompt "ubs-256t-b" "$(construct_fixed_size_prompt 256)"
	require_cached_tokens "ubs-256t-b" "$(( 256 - 32 ))"

	do_prompt "ubs-257t-a" "$(construct_fixed_size_prompt 257)"
	do_prompt "ubs-257t-b" "$(construct_fixed_size_prompt 257)"
	require_cached_tokens "ubs-257t-b" 256
	echo
}

check_restore() {
	echo "=== passage: prime + same-session restore (0 -> 352) ==="
	# the prompt is 365 tokens = 11 full ubs chunks (352) + a 13-token tail
	# (never saved), so the repeat restores exactly 352. the 'OK' instruction
	# pins the reply to 2 tokens: a restore-state divergence would change it.
	flush_cache

	do_prompt "passage-a" "${PROMPT_7L}"
	require_cached_tokens "passage-a" 0
	require_ok_response "passage-a"

	do_prompt "passage-b" "${PROMPT_7L}"
	require_cached_tokens "passage-b" 352
	require_ok_response "passage-b"
	echo
}


check_fork() {
	echo "=== forked: A/A/B/B hash-chain fork (expected 0 / 352 / 96 / 352) ==="
	# B shares A's first 3 chunks (96 tokens), then diverges. a wrong hash chain
	# (whole-prompt hashing, no fork, stale chain) yields a different 4-tuple.
	flush_cache

	do_prompt "fork-a1" "${PROMPT_7L}"
	require_cached_tokens "fork-a1" 0

	do_prompt "fork-a2" "${PROMPT_7L}"
	require_cached_tokens "fork-a2" 352

	do_prompt "fork-b1" "${PROMPT_7L_B}"
	require_cached_tokens "fork-b1" 96

	do_prompt "fork-b2" "${PROMPT_7L_B}"
	require_cached_tokens "fork-b2" 352

	# the restore touches the tail rs + every 8th rs below it (stride 8): for an
	# 11-chunk chain that is rs at 0, 8 and tail 10 -> 3 files touched at ~the
	# same instant, all others keeping their (earlier) save mtime. check the gap
	# pattern in the 4 newest rs files: 2 small gaps (<0.1s) + 1 large (>0.5s).
	# WARN only: fs timestamp granularity can flake this.
	rs_touch_gaps="$(find "${KV_CACHE_DIR}" -name '*.rscache' -printf '%T@\n' 2>/dev/null | sort -n | tail -n 4 | awk 'NR>1{printf "%.3f ", $1-p} {p=$1}')"
	n_small="$(echo "${rs_touch_gaps}" | awk '{n=0; for (i=1; i<=NF; i++) if ($i+0 < 0.1) n++} END{print n}')"
	n_large="$(echo "${rs_touch_gaps}" | awk '{n=0; for (i=1; i<=NF; i++) if ($i+0 > 0.5) n++} END{print n}')"
	echo "rs-touch gaps (newest-4, oldest->newest): ${rs_touch_gaps}"
	if [ "${n_small}" -eq 2 ] && [ "${n_large}" -eq 1 ] ; then
		record 0 "fork: rs-touch gap pattern (3 files touched together)"
	else
		warn "fork: rs-touch gap pattern (got: ${rs_touch_gaps})"
	fi
	echo
}

check_rscache_file_deletion() {
	echo "=== evict-rs: delete all but the 2 newest .rscache files ==="
	# the tail rs is the only one ever read; middle rs files can vanish without
	# breaking restore. expect: still a full restore (352).
	flush_cache

	do_prompt "evrs-a" "${PROMPT_7L}"
	require_cached_tokens "evrs-a" 0

	rs_files="$(ls -t "${KV_CACHE_DIR}"/*.rscache 2>/dev/null | tac)"
	n_total="$(echo "${rs_files}" | grep -c . || true)"
	if [ "${n_total}" -le 2 ] ; then
		warn "evict-rs: only ${n_total} rs files, nothing to delete"
	else
		n_del=$((n_total - 2))
		echo "${rs_files}" | head -n "${n_del}" | while read -r f ; do
			rm -f "${f}"
		done
		echo "deleted ${n_del} of ${n_total} .rscache files (kept newest 2)"
	fi

	do_prompt "evrs-b" "${PROMPT_7L}"
	require_cached_tokens "evrs-b" 352
	require_ok_response "evrs-b"
	echo
}

check_kvcache_file_deletion() {
	echo "=== evict-kv: delete the 3rd-oldest .kvcache file ==="
	# the chain breaks at chunk 2: files exist for 0,1 and 3..N but not 2.
	# phase 1 (exists-only walk) stops there -> cached 64 (2 chunks).
	flush_cache

	do_prompt "evkv-a" "${PROMPT_7L}"
	require_cached_tokens "evkv-a" 0

	kv_files="$(ls -t "${KV_CACHE_DIR}"/*.kvcache 2>/dev/null | tac)"
	n_total="$(echo "${kv_files}" | grep -c . || true)"
	if [ "${n_total}" -lt 3 ] ; then
		warn "evict-kv: only ${n_total} kvcache files, need >= 3"
	else
		target="$(echo "${kv_files}" | sed -n '3p')"
		echo "deleting $(basename "${target}") (3rd oldest of ${n_total})"
		rm -f "${target}"
	fi

	do_prompt "evkv-b" "${PROMPT_7L}"
	# allow 32-96 (1-3 chunks) to account for off-by-one in file ordering
	evkv_b="$(get_stat "evkv-b" cached_tokens)"
	if [ -n "${evkv_b}" ] && [ "${evkv_b}" -ge 32 ] && [ "${evkv_b}" -le 96 ] ; then
		record 0 "evict-kv: cached_tokens in [32,96] (got: ${evkv_b})"
	else
		record 1 "evict-kv: cached_tokens in [32,96] (got: ${evkv_b:-<missing>})"
	fi
	echo
}

check_coherence() {
	flush_cache

	FILLER="$( cat "${DEVS}/prompt_30k.txt" | tail -n 50 )"
	PASSAGE="$( cat "${DEVS}/prompt_7sentences.txt" )"
	PROMPT="---- the following is **filler** to test output coherence in a fork of llama.cpp ----
		${FILLER}
		---- now the real test ----
		please repeat this entire passage exactly, word for word, with no additions or omissions:
		'${PASSAGE}'"

	do_prompt "coherence-a" "${PROMPT}"
	require_cached_tokens "coherence-a" 0
	require_all_phrases "coherence-a"

	do_prompt "coherence-b" "${PROMPT}"
	require_cached_tokens "coherence-b" 2624
	require_all_phrases "coherence-b"
}


# ============================================================================
echo "=== starting kv-chain server (one session for the whole run) ==="
start_kv_server
echo

check_hello
check_restore
check_fork
check_ubatch_edge_cases
check_rscache_file_deletion
check_kvcache_file_deletion
check_coherence


# ============================================================================
echo "=== summary ==="
echo "PASS: ${PASS_COUNT}   FAIL: ${FAIL_COUNT}   WARN: ${WARN_COUNT}"
if [[ ${FAIL_COUNT} -gt 0 ]] ; then
	echo "failed checks:"
	printf '  - %s\n' "${FAILURES[@]}"
	exit 1
fi
echo "all checks passed"
