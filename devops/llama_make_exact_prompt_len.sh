#!/usr/bin/env bash
# Build a prompt that tokenizes to EXACTLY 256 tokens, given the **currently** hosted
# model/tokenizer, while retaining the instruction to output as few tokens as possible.
# This requires the llama-server to be running. Converges to the answer.
# To fetch the prompt in the call-site, find the string between '[' and ']'.

set -euo pipefail
DEVS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${DEVS}/env.sh"

PROMPT_BASE="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. Bright vixens jump; dozy fowl quack. Jackdaws love my big sphinx of quartz. The quick brown fox jumps again, twice, thrice, and once more for good measure."
PROMPT_SUFFIX="--- the task is to ignore the above, and just respond with 'OK'"


TARGET="$1"
SUBLEN="$(( ${#PROMPT_BASE} / 2 ))"

count_tokens() {
    "${DEVS}/llama_prompt.sh" "$1" --  2>&1 >/dev/null | grep -oE 'prompt_tokens: [0-9]+' | grep -oE '[0-9]+'
}


PROMPT_SMALLER=""
while : ; do
    if [ -z "${PROMPT_SMALLER}" ] ; then
        PROMPT_TRY="${PROMPT_BASE}"
    else
        PROMPT_TRY="${PROMPT_SMALLER} ${PROMPT_BASE}"
    fi

    # the double space is intentional (same as: "... ${BASE:0:0} ...")
    got="$( count_tokens "${PROMPT_TRY}  ${PROMPT_SUFFIX}" )"
    echo "got=${got}"
    if [ "${got}" -ge "${TARGET}" ] ; then
        break
    fi
    PROMPT_SMALLER="${PROMPT_TRY}"
done

min="$( count_tokens "${PROMPT_SMALLER}  ${PROMPT_SUFFIX}" )"
max="$( count_tokens "${PROMPT_SMALLER} ${PROMPT_BASE} ${PROMPT_SUFFIX}" )"
echo "search range: ${min}..${max}"
if [ "${min}" -gt "${TARGET}" ] ; then echo 'oops' ; exit 1 ; fi
if [ "${max}" -lt "${TARGET}" ] ; then echo 'oops' ; exit 1 ; fi

while : ; do
   PROMPT_SIZED="${PROMPT_SMALLER} ${PROMPT_BASE:0:${SUBLEN}} ${PROMPT_SUFFIX}"
   got="$( count_tokens "${PROMPT_SIZED}" )"
   echo "sub=${SUBLEN}, got=${got}"
     if [ "${got}" -gt "$(( TARGET + 10 ))" ] ; then SUBLEN="$(( SUBLEN - 20 ))"
   elif [ "${got}" -lt "$(( TARGET - 10 ))" ] ; then SUBLEN="$(( SUBLEN + 20 ))"
   elif [ "${got}" -gt "$(( TARGET + 5 ))"  ] ; then SUBLEN="$(( SUBLEN - 10 ))"
   elif [ "${got}" -lt "$(( TARGET - 5 ))"  ] ; then SUBLEN="$(( SUBLEN + 10 ))"
   elif [ "${got}" -gt "${TARGET}"          ] ; then SUBLEN="$(( SUBLEN - 1 ))"
   elif [ "${got}" -lt "${TARGET}"          ] ; then SUBLEN="$(( SUBLEN + 1 ))"
   else echo "found match" ; break ; fi
done

echo "PROMPT with ${TARGET} tokens: [${PROMPT_SIZED}]"
