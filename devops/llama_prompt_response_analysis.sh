#!/bin/bash

cleanup_output() {
   local file="$1"
   grep -v 'cached_tokens' "${file}" | sed '/^$/d' | tr '\n' ' ' | sed 's/ \+/ /g'
}

analyze_output() {
   local file="$1"

   echo "=== analysis of ${file} ==="
   echo "metadata: $(grep 'cached_tokens' "${file}")  response_length: $(cleanup_output "${file}" | wc -c)  hash: $(cleanup_output "${file}" | md5sum - | awk '{print $1}')"
   echo "response one-liner: [$(cleanup_output "${file}")]"
   echo ""
}

echo "=== COMPARING [$@]"
echo

for prompt_file in "$@" ; do
    analyze_output "${prompt_file}"
done



failures=()

for prompt_file in "$@" ; do
    if [ "$( cleanup_output "${prompt_file}" | wc -c )" -lt 25 ] ; then
        failures+=("file ${prompt_file} response too short")
    fi
done

prev_prompt_file=''
for curr_prompt_file in "$@" ; do
    if [ -n "${prev_prompt_file}" ] && [ "$( cleanup_output "${prev_prompt_file}" )" != "$( cleanup_output "${curr_prompt_file}" )" ] ; then
        failures+=("file ${prev_prompt_file}/${curr_prompt_file} responses not equal")
    fi
    prev_prompt_file="${curr_prompt_file}"
done


echo "=== RESULT: for prompt files: $@"
if [ "${#failures[@]}" -eq 0 ] ; then
    echo "=== PASS ==="
    exit 0
else
    echo "=== FAIL ==="
    printf -- '- FAIL because %s\n' "${failures[@]}"
    exit 1
fi

