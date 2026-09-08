#!/bin/bash

# Run the Qwen kv-chain test suite. Execute with a ~60min timeout.
#
# Deliberately NOT `set -e`: every test must run even if an earlier one fails,
# so one run gives the full picture. Exit codes are collected instead and
# summarised at the end (and returned), so a failure is never silent.
#
# Each test writes ./out-<name>.log; grep/head/tail THAT instead of re-running
# (a full pass costs many minutes).
#
# llama_unittest_dsv4f.sh is NOT in this list: it needs a different model and
# must not run concurrently with the Qwen tests (same port + cache dir).

cd "$(dirname "${BASH_SOURCE[0]}")"

rm -f ./out-*.log

if ! ./llama_build.sh 1>./.build.log 2>&1; then
    echo "BUILD FAILED - see ./devops/.build.log"
    tail -20 ./.build.log
    exit 1
fi

# do NOT run the dsv4-flash test here, because the OOM-monitor job will kill
# random processes to free resources, when opencode/qwen 3.8 are also running.
TESTS=(
    smoke_test
    restore_restart    # full-chain restore across a server restart
    no_kvchain         # zero-behavior-change control (feature disabled)
    restore_session    # full-chain restore, same session (no restart)
    evict_rscache      # middle .rscache files deleted -> tail rs still usable
    evict_kvcache      # middle .kvcache deleted -> chain breaks there
    forked_chains      # A/A/B/B: shared trunk + diverging branch
    ubatch_edge        # prompt len of TARGET-1 / TARGET+1 / exact multiple
)

declare -A RESULT
n_fail=0

for t in "${TESTS[@]}"; do
    echo "=== running ${t} ==="
    if "./llama_unittest_${t}.sh" >"./out-${t}.log" 2>&1; then
        RESULT[$t]=PASS
    else
        RESULT[$t]=FAIL
        n_fail=$((n_fail + 1))
    fi
done

echo ""
echo "=== summary ==="
for t in "${TESTS[@]}"; do
    printf '  %-18s %s\n' "${t}" "${RESULT[$t]}"
done
echo ""
echo "results are written to ./devops/out-<name>.log"

if [[ ${n_fail} -gt 0 ]]; then
    echo "${n_fail} of ${#TESTS[@]} tests FAILED"
    exit 1
fi
echo "all ${#TESTS[@]} tests passed"
