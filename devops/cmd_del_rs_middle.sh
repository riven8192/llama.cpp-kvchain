#!/usr/bin/env bash
# delete all but the last 2 .rscache files (by mtime), keeping the tail
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
# sort by mtime (oldest first), keep the last 2
rs_files=$(ls -t "${KV_CACHE_DIR}"/*.rscache 2>/dev/null | tac)
n_total=$(echo "${rs_files}" | grep -c . || true)
if [ "${n_total}" -le 2 ]; then
    echo "del_rs: only ${n_total} rscache files, nothing to delete"
    exit 0
fi
n_del=$((n_total - 2))
echo "del_rs: deleting ${n_del} oldest of ${n_total} rscache files (keeping newest 2)"
echo "${rs_files}" | head -n "${n_del}" | while read -r f; do
    rm -f "${f}"
    echo "  deleted: $(basename "${f}")"
done
