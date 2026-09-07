#!/usr/bin/env bash
# delete the 3rd-oldest .kvcache file (by mtime) to break the chain at chunk 2
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
kv_files=$(ls -t "${KV_CACHE_DIR}"/*.kvcache 2>/dev/null | tac)
n_total=$(echo "${kv_files}" | grep -c . || true)
if [ "${n_total}" -lt 3 ]; then
    echo "del_kv: only ${n_total} kvcache files, need >= 3"
    exit 0
fi
target=$(echo "${kv_files}" | sed -n '3p')
echo "del_kv: deleting $(basename "${target}") (3rd oldest of ${n_total})"
rm -f "${target}"
