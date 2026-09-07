#!/usr/bin/env bash
# Stop the dev llama-server. Matches on the exact port so we never kill the
# user's own server (which runs on a different port). Uses the [p]attern trick
# so the grep/pkill does not match its own shell.
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ -f "${LLAMA_PIDFILE}" ]]; then
  pid=$(cat "${LLAMA_PIDFILE}" 2>/dev/null || true)
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "killing pid ${pid}"
    kill "${pid}" 2>/dev/null || true
  fi
  rm -f "${LLAMA_PIDFILE}"
fi

# belt and braces: anything bound to our port
# [l] bracket trick: prevents pkill from matching its own shell
pkill -f "[l]lama-server.*--port ${LLAMA_PORT}( |$)" 2>/dev/null || true

# wait for the port to be released
for _ in $(seq 1 20); do
  if ! ss -tln 2>/dev/null | grep -q ":${LLAMA_PORT}\b"; then
    echo "port ${LLAMA_PORT} is free"
    exit 0
  fi
  sleep 0.5
done

echo "warning: port ${LLAMA_PORT} still in use" >&2
exit 1
