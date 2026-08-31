#!/bin/bash

# execute this script with timeout 30min

# do NOT fail on errors: `set -euo pipefail`

cd "$(dirname "${BASH_SOURCE[0]}")"

rm -f ./out-*.log

./llama_unittest_1.sh  >./out-1.log  2>&1
./llama_unittest_2.sh  >./out-2.log  2>&1
./llama_unittest_3a.sh >./out-3a.log 2>&1
./llama_unittest_3b.sh >./out-3b.log 2>&1
./llama_unittest_3c.sh >./out-3c.log 2>&1
./llama_unittest_4.sh  >./out-4.log  2>&1

echo 'results are written to ./devops/out-[id].log'
