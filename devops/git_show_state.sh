#!/bin/bash

set -euo pipefail

base=master

usage() {
  cat <<'EOF'
usage: git_show_state.sh [options]

  -b, --base REF       parent branch to diff against  (default: master)
  -h, --help           this message

example:
  git_show_state.sh -b main
EOF
}

quiet=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--base)    base=${2:?};    shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; break ;;
    *)            echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo '--- git status (short) ---'
git status --short
echo ''

echo '--- git log (branch) ---'
git log --oneline "${base}"..HEAD
echo ''
