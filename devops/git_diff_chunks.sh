#!/usr/bin/env bash
#
# git-diff-chunks.sh - split a branch diff into LLM-sized chunks, one whole
# file per diff, packed into chunks of at most --max lines.
#
set -euo pipefail

cd /home/riven/opencode/kv-cache-qwen/llama.cpp-kv-qwen/llama.cpp

base=master
max=500
outdir=".diff-chunks"
context=3
excludes=()

usage() {
  cat <<'EOF'
usage: git-diff-chunks.sh [options] [-- extra git-diff args]

  -b, --base REF       parent branch to diff against      (default: master)
  -m, --max N          max lines per chunk                (default: 500)
  -o, --out DIR        output directory                   (default: .diff-chunks)
  -U, --context N      lines of context per hunk          (default: 3)
  -x, --exclude PATH   exclude a path (repeatable; git pathspec, repo-relative)
  -q, --quiet          print only the chunk paths
  -h, --help           this message

example:
  git-diff-chunks.sh -b main -U 2 -x NOTES.md -x '*.lock' -x dist/
EOF
}

quiet=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--base)    base=${2:?};    shift 2 ;;
    -m|--max)     max=${2:?};     shift 2 ;;
    -o|--out)     outdir=${2:?};  shift 2 ;;
    -U|--context) context=${2:?}; shift 2 ;;
    -x|--exclude) excludes+=("${2:?}"); shift 2 ;;
    -q|--quiet)   quiet=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; break ;;
    *)            echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mergebase=$(git merge-base "$base" HEAD) || exit 1

# ':(top)' anchors pathspecs to the repo root, so this works from subdirectories.
pathspecs=( ':(top)' )
for e in ${excludes[@]+"${excludes[@]}"}; do
  pathspecs+=( ":(exclude,top)$e" )
done

#   --diff-filter=d  drop deleted files (unlike -D, no leftover stub header)
#   --no-color       survive color.ui=always
diff_opts=( --no-color --diff-filter=d -M -U"$context" ${1+"$@"} )

mkdir -p "$outdir"
rm -f "${outdir:?}"/part-*.diff

part=0 used=0 chunk=""
new_chunk() {
  part=$((part + 1))
  chunk=$(printf '%s/part-%02d.diff' "$outdir" "$part")
  : > "$chunk"
  used=0
}
new_chunk

# --name-status -z emits "status NUL path NUL", and for renames
# "R### NUL old NUL new NUL". Renames need both paths handed to the same
# git diff, or -M can't pair them and dumps the whole file as an add.
while IFS= read -r -d '' status; do
  IFS= read -r -d '' path
  paths=( ":(top,literal)$path" )
  if [[ $status == R* ]]; then
    IFS= read -r -d '' newpath
    paths+=( ":(top,literal)$newpath" )
  fi

  git diff "${diff_opts[@]}" "$mergebase" HEAD -- "${paths[@]}" > "$outdir/.f"
  n=$(wc -l < "$outdir/.f")
  (( used > 0 && used + n > max )) && new_chunk
  cat "$outdir/.f" >> "$chunk"
  used=$((used + n))
done < <(git diff -z --name-status "${diff_opts[@]}" "$mergebase" HEAD -- "${pathspecs[@]}")

rm -f "$outdir/.f"

if [[ ! -s $chunk ]]; then
  rm -f "$chunk"
  echo "no changes between $base and HEAD" >&2
  exit 0
fi

if [[ $quiet -eq 0 ]]; then
  echo "diff vs $base ($(git rev-parse --short "$mergebase")), $part chunks, read in order:" >&2
fi

for f in "$outdir"/part-*.diff; do
  size=$(wc -l < "$f")
  printf '%s\n' "./llama.cpp-kv-qwen/llama.cpp/$f"
  # (( size > max )) && echo "  warning: $f is $size lines (one file's diff exceeds --max)" >&2
done
exit 0
