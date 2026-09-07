#!/bin/bash

set -euo pipefail

echo '--- git status (short) ---'
git status --short
echo ''

echo '--- git log (branch) ---'
git log --oneline master..HEAD
echo ''
