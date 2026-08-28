#!/usr/bin/env bash
# Wait until the dev llama-server is ready (or it has crashed).
# Usage: llama_wait.sh [timeout_seconds]
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

TIMEOUT="${1:-300}"
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
