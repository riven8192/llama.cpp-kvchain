#!/usr/bin/env bash
# Hash-chain KV cache restore test.
#
# Usage:
#   llama_test.sh "prompt1" "[restart]" "[cmd:./script.sh]" "prompt2" ... [-- extra llama_run.sh args]
#     "prompt"          send a prompt, log to prompt-N.log
#     [restart]         restart the server (keeping the disk cache) before the next prompt
#     [cmd:PATH]        run PATH as a bash script (with $KV_CACHE_DIR, $DEVS, $LLAMA_LOG in env),
#                       wait for it to finish, then continue. the script can inspect/twiddle
#                       the cache dir, the server log, etc.
#     -- ...            extra args forwarded to llama_run.sh (e.g. "-ub 32 -b 32")

set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

# split off a trailing `-- extra args` for llama_run.sh
RUN_ARGS=()
PROMPTS=()
SEEN_DD=0
for arg in "$@"; do
    if [ "${SEEN_DD}" -eq 1 ]; then
        RUN_ARGS+=("${arg}")
    elif [ "${arg}" = "--" ]; then
        SEEN_DD=1
    else
        PROMPTS+=("${arg}")
    fi
done
RUN_ARGS_STR="${RUN_ARGS[*]:-}"

run_server() {
    # llama_run.sh waits for the server to be ready before returning
    if [ -n "${RUN_ARGS_STR}" ]; then
        "${DEVS}/llama_run.sh" --keep-cache "$@" -- "${RUN_ARGS[@]}"
    else
        "${DEVS}/llama_run.sh" --keep-cache "$@"
    fi
}

flush_cache() {
    mkdir -p "${KV_CACHE_DIR}"
    du -h "${KV_CACHE_DIR}" 2>/dev/null
    rm -rf "${KV_CACHE_DIR}"
    mkdir -p "${KV_CACHE_DIR}"
}

rm -f "${DEVS}"/prompt-*.log
echo "=== [1/3] llama_run.sh ==="
run_server
echo ""

echo "=== [2/3] flush cache ==="
# we do this **after** the spawn, so that we are sure no (prev) instances are still writing into the dir
flush_cache
echo ""


prompt_counter=0
start_counter=1
for step in "${PROMPTS[@]}"; do
    if [ "${step}" = '[flush]' ]; then
        echo "=== [flush] ==="
        flush_cache
        echo ""
    elif [ "${step}" = '[restart]' ]; then
        start_counter="$((start_counter + 1))"

        echo "=== [restart] ==="
        run_server
        echo ""
    elif [ "${step}" = '[restart-no-kv]' ]; then
        start_counter="$((start_counter + 1))"

        echo "=== [restart-no-kv] ==="
        run_server --no-kv-chain
        echo ""
    elif [[ "${step}" == \[cmd:* ]]; then
        # extract the script path: [cmd:path/to/script.sh]
        # "[cmd:" is 5 chars, trailing "]" is 1 char
        local_cmd="${step:5:${#step}-6}"
        echo "=== [cmd:${local_cmd}] ==="
        # resolve relative paths against DEVS
        case "${local_cmd}" in
            /*) cmd_path="${local_cmd}" ;;
            *)  cmd_path="${DEVS}/${local_cmd}" ;;
        esac
        if [ ! -f "${cmd_path}" ]; then
            echo "ERROR: cmd script not found: ${cmd_path}" >&2
            exit 1
        fi
        bash "${cmd_path}"
        echo ""
    else
        prompt_counter="$((prompt_counter + 1))"

        echo "=== [prompt ${prompt_counter}] ==="
        echo "prompt: '${step:0:100}'"
        echo "log-file: ${DEVS}/prompt-${prompt_counter}.log"
        PROMPT_START=$(date +%s)
        "${DEVS}/llama_prompt.sh" "${step}" 2>&1 | tee "${DEVS}/prompt-${prompt_counter}.log" || true
        PROMPT_END=$(date +%s)
        echo ""
        echo "Took: $(( PROMPT_END - PROMPT_START)) seconds"
    fi
done

echo "=== chunks on disk ==="
find "${KV_CACHE_DIR}" \( -name '*.kvcache' -o -name '*.rscache' \) -printf "%f\n" 2>/dev/null

echo "=== distinct process log-files ==="
# list $DEVS (not the CWD): the log files live next to this script, and a
# missing match would otherwise trip `set -euo pipefail` (grep exit 1)
ls -a "${DEVS}" | grep '\.llama-server\.log' | sort | tail -n "${start_counter}" || true
