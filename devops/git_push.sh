#!/usr/bin/env bash
#
# Push the current branch to the kvchain fork, authenticating with the token
# from github-key.txt. The token is passed on the command line (inline in the
# push URL) and is NEVER written to .git/config: the remote stays a plain
# https URL, and the credential is read from the file at push time.
#
# Usage:
#   devops/git_push.sh              # push the current branch
#   devops/git_push.sh <branch>     # push a specific branch
#
set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${DEVS}/.." && pwd)"

# the token lives in the project root (the dir that holds the fork checkout),
# not inside the repo. override with KVCHAIN_GIT_TOKEN_FILE if it is elsewhere.
# devops/ -> llama.cpp/ -> llama.cpp-kv-qwen/ -> <project root>/github-key.txt
KEY_FILE="${KVCHAIN_GIT_TOKEN_FILE:-${DEVS}/../../../github-key.txt}"
if [[ ! -f "${KEY_FILE}" ]]; then
	echo "error: token file not found: ${KEY_FILE}" >&2
	exit 1
fi
TOKEN="$(tr -d '[:space:]' < "${KEY_FILE}")"
if [[ -z "${TOKEN}" ]]; then
	echo "error: token file is empty: ${KEY_FILE}" >&2
	exit 1
fi

BRANCH="${1:-$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)}"
# push to a token-embedded URL, not to a named remote: git stores nothing
git -C "${REPO}" push "https://git:${TOKEN}@github.com/riven8192/llama.cpp-kvchain.git" "${BRANCH}"
