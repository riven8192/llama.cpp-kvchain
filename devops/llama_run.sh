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
if [[ -z "${LLAMA_MODEL}" || ! -f "${LLAMA_MODEL}" ]]; then
  echo "model not found: '${LLAMA_MODEL}' (set LLAMA_MODEL)" >&2
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
  -m "${LLAMA_MODEL}"
  --host "${LLAMA_HOST}"
  --port "${LLAMA_PORT}"
  -c "${LLAMA_CTX}"
  -ngl "${LLAMA_NGL}"
  -t "${LLAMA_THREADS}"
  --parallel "${LLAMA_PARALLEL}"
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
echo "  model : ${LLAMA_MODEL}"
echo "  cache : ${CACHE_DESC}"
echo "  log   : ${LLAMA_LOG}"

# fully detach so the launcher shell does not wait on the server

log_ts="$(date +%Y%m%d-%H%M%S)"
: >"${LLAMA_PIDFILE}"

setsid bash -c '
  pidfile=$1; log=$2; tslog=$3; shift 3
  { echo $BASHPID >"$pidfile"; exec "$@"; } </dev/null 2>&1 | tee "$log" "$tslog"
' _ "$LLAMA_PIDFILE" "$LLAMA_LOG" "$LLAMA_LOG.$log_ts.log" \
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
