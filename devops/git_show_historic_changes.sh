#!/bin/bash

set -euo pipefail

# LLM: I filtered out the old NOTES.md
# LLM: do NOT manually read the history of NOTES.md, it will poison your context

git diff --diff-filter=d -M --no-color -U2 master...HEAD -- . \
  ':(exclude,top)NOTES.md' \
  ':(exclude,top)*.lock' \
  ':(exclude,top)dist/'

