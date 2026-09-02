#!/usr/bin/env bash
# Start the dev llama-server in the background with the kv-chain disk cache.
# Usage:
#   llama_run.sh                 # fresh start (clears the kv cache dir)
#   llama_run.sh --keep-cache    # keep existing kv cache dir (for restore tests)
#   llama_run.sh --no-kv-chain   # run without the disk cache
# Extra args after a `--` are appended to the server command line.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

KEEP_CACHE=0
NO_KV_CHAIN=0
EXTRA_ARGS=()
SEEN_DASH_DASH=0
for arg in "$@"; do
  if [[ ${SEEN_DASH_DASH} -eq 1 ]]; then
    EXTRA_ARGS+=("${arg}")
    continue
  fi
  case "${arg}" in
    --keep-cache)  KEEP_CACHE=1 ;;
    --no-kv-chain) NO_KV_CHAIN=1 ;;
    --)            SEEN_DASH_DASH=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ ! -x "${LLAMA_SERVER_BIN}" ]]; then
  echo "server binary not found: ${LLAMA_SERVER_BIN} (run devops/build.sh first)" >&2
  exit 1
fi
if [[ -z "${LLAMA_HF_REF}" ]]; then
  echo "no HF model ref (set LLAMA_HF_REF, e.g. 'unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL')" >&2
  exit 1
fi

# stop any previous instance on this port
# (call as a subprocess: llama_kill.sh may exit non-zero, and we do not want
#  it to leak variables or exit this script)
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${DEVS}/llama_kill.sh" || echo "warning: previous server may still be running" >&2

if [[ ${NO_KV_CHAIN} -eq 0 ]]; then
  if [[ ${KEEP_CACHE} -eq 0 ]]; then
    rm -rf "${KV_CACHE_DIR}"
  fi
  mkdir -p "${KV_CACHE_DIR}"
fi

ARGS=(
  -hf "${LLAMA_HF_REF}"
  --host "${LLAMA_HOST}"
  --port "${LLAMA_PORT}"
  --timeout 86400
  --ctx-size "${LLAMA_CTX}"
  -ngl "${LLAMA_NGL}"
  -t "${LLAMA_THREADS}"
  --parallel "${LLAMA_PARALLEL}"
  # kill reasoning/thinking mode + zero its budget: reasoning models (e.g.
  # Qwen3-4B) otherwise spend the whole n_ctx on thinking tokens, get capped
  # mid-reasoning, and never emit the answer - tests then fail on a ctx issue,
  # not a kv-chain issue. temperature is pinned to 0 in llama_prompt.sh.
  --reasoning off
  --reasoning-budget 0
  --cache-ram 0
  --no-cache-idle-slots
  --fit off
  --load-mode none
  --no-mmproj
  --flash-attn on
  --jinja
)
if [[ ${NO_KV_CHAIN} -eq 0 ]]; then
  ARGS+=(--kv-chain-dir "${KV_CACHE_DIR}" --kv-chain-limit-gb "${KV_CHAIN_LIMIT_GB}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ ${NO_KV_CHAIN} -eq 1 ]]; then
  CACHE_DESC="disabled"
else
  CACHE_DESC="${KV_CACHE_DIR} (limit ${KV_CHAIN_LIMIT_GB} GiB)"
fi

echo "starting llama-server on ${LLAMA_URL}"
echo "  model : ${LLAMA_HF_REF}"
echo "  cache : ${CACHE_DESC}"
echo "  log   : ${LLAMA_LOG}"

# fully detach so the launcher shell does not wait on the server.
# IMPORTANT: the server's stdout/stderr must go to LOG FILES ONLY. if it inherited
# this script's stdout (e.g. via `tee` or an open pipe to the caller), any caller
# that waits for pipe-EOF (a pipeline, a tool that captures output) hangs forever
# because the long-lived server keeps the write end open.
#
# each run appends to its own timestamped log ($log_ts.log); $log is a SYMLINK to
# the newest one, so `tail -f $log` / the readiness wait-loop below always see the
# CURRENT run's log (a restart must not be fooled by the previous run's
# "listening on" line in a stale file).
log_ts="$(date +%Y%m%d-%H%M%S)"
: >"${LLAMA_PIDFILE}"
: >"${LLAMA_LOG}.${log_ts}.log"
ln -sf "${LLAMA_LOG}.${log_ts}.log" "${LLAMA_LOG}"

setsid bash -c '
  pidfile=$1; tslog=$2; shift 2
  { echo $BASHPID >"$pidfile"; exec "$@"; } </dev/null >>"$tslog" 2>&1
' _ "$LLAMA_PIDFILE" "${LLAMA_LOG}.${log_ts}.log" \
  "$LLAMA_SERVER_BIN" "${ARGS[@]}" &

for _ in {1..100}; do
    [[ -s ${LLAMA_PIDFILE} ]] && break
    sleep 0.05
done
pid="$(cat "${LLAMA_PIDFILE}")"
if [[ -z $pid ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "server failed to start; see ${LLAMA_LOG}" >&2
    exit 1
fi
echo "pid: $pid"

# wait until the server is ready (or it has crashed) - llama_run.sh is the
# single entry point, so the wait lives here, not in a separate script
TIMEOUT=300
START=$(date +%s)

echo "waiting for http service..."
while :; do
  if grep -q "listening on" "${LLAMA_LOG}" 2>/dev/null; then
    echo "ready: $(grep 'listening on' "${LLAMA_LOG}" | tail -1)"
    exit 0
  fi
  if grep -q "exiting due to" "${LLAMA_LOG}" 2>/dev/null; then
    echo "server exited:" >&2
    tail -15 "${LLAMA_LOG}" >&2
    exit 1
  fi
  if [[ -f "${LLAMA_PIDFILE}" ]]; then
    pid=$(cat "${LLAMA_PIDFILE}" 2>/dev/null || true)
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "server process ${pid} died:" >&2
      tail -15 "${LLAMA_LOG}" >&2
      exit 1
    fi
  fi
  now=$(date +%s)
  if (( now - START >= TIMEOUT )); then
    echo "timed out after ${TIMEOUT}s waiting for server" >&2
    tail -15 "${LLAMA_LOG}" >&2
    exit 1
  fi
  sleep 2
done
