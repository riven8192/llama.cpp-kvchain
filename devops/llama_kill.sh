#!/usr/bin/env bash
# Stop the dev llama-server. Matches on the exact port so we never kill the
# user's own server (which runs on a different port).

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"


instance_pids() {
  ps faux | grep llama-server | grep "port ${LLAMA_PORT} " | awk '{print $2}'
}

echo "running instances: $( instance_pids | wc -l )"

if [[ -f "${LLAMA_PIDFILE}" ]]; then
  pid=$(cat "${LLAMA_PIDFILE}" 2>/dev/null || true)
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "killing pid: ${pid}"
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${LLAMA_PIDFILE}"
fi

echo "running instances: $( instance_pids | wc -l )"

while read pid ; do
  echo "killing pid: ${pid}"
  kill -9 "${pid}" 2>/dev/null || true
done < <( instance_pids )

echo "running instances: $( instance_pids | wc -l )"

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
