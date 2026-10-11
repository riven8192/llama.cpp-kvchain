#!/bin/bash
#
# Run the kv-chain unit-test suite once per (model x parallel) combination.
#
# - each combo gets its own log (.unittest-output-<model>-p<par>.log) so the
#   parallel=2 run does not clobber the parallel=1 run.
# - the suite's exit code is captured from the pipeline via PIPESTATUS: the
#   `| tee` pipeline would otherwise report tee's exit (always 0) and mask a
#   failing suite.
# - a summary table of every combo's exit code is printed at the end, and the
#   script exits non-zero if any combo failed.
#
# (intentionally no `set -e`: every combo should run, then report - not stop at
#  the first failure.)

declare -a MODELS=(
	"Qwen3.8-27B|32|unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL"
	"DSV4F|128|unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-IQ3_S"
	"Qwen3.8-FN|32|unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL"
)

RESULTS=()

run_combo() { # $1=model-name $2=ubs $3=hf-ref $4=parallel
	local name="$1" ubs="$2" ref="$3" par="$4"
	local logf="./.unittest-output-${name}-p${par}.log"
	echo "=== ${name} (ub=${ubs}, parallel=${par}) -> ${logf} ==="
	env LLAMA_PARALLEL="${par}" LLAMA_UBATCH="${ubs}" LLAMA_BATCH="${ubs}" LLAMA_HF_REF="${ref}" \
		./llama_unittests.sh 2>&1 | tee "${logf}"
	# capture the SUITE's exit code (not tee's) from the pipeline. must be done on
	# the SAME command as the expansion: a separate `local rc` line would run first
	# and reset $PIPESTATUS to 0 (local's own exit), masking a failing suite.
	local rc=${PIPESTATUS[0]}
	RESULTS+=("${name}-p${par}|${rc}")
	echo "=== ${name} (parallel=${par}) exit-code=${rc} ==="
}

for entry in "${MODELS[@]}"; do
	IFS='|' read -r name ubs ref <<< "${entry}"
	for par in 1 2; do
		run_combo "${name}" "${ubs}" "${ref}" "${par}"
	done
done

./llama_kill.sh

echo
echo "=== kv-chain suite results (per model x parallel) ==="
any_fail=0
for r in "${RESULTS[@]}"; do
	IFS='|' read -r label rc <<< "${r}"
	if [ "${rc}" -eq 0 ]; then
		status="PASS"
	else
		status="FAIL"
		any_fail=1
	fi
	printf '  %-22s exit=%s  %s\n' "${label}" "${rc}" "${status}"
done
echo
if [ "${any_fail}" -ne 0 ]; then
	echo "one or more combos FAILED (see .unittest-output-*.log)"
	exit 1
fi
echo "all combos passed"
