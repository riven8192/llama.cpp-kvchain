#!/bin/bash

# set -euo pipefail

LLAMA_UBATCH=32  LLAMA_BATCH=32  LLAMA_HF_REF=unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL \
	./llama_unittests.sh 2>&1 | tee ./.unittest-output-Qwen-3.8-27B.log

LLAMA_UBATCH=128 LLAMA_BATCH=128 LLAMA_HF_REF=unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-IQ3_S \
	./llama_unittests.sh 2>&1 | tee ./.unittest-output-DSV4F.log

LLAMA_UBATCH=32  LLAMA_BATCH=32  LLAMA_HF_REF=unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL \
	./llama_unittests.sh 2>&1 | tee ./.unittest-output-Qwen-3.8-FN.log

./llama_kill.sh
