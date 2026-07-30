#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"

need=0
for cmd in pdftotext pdfinfo; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'ERROR: %s not found on PATH.\n' "$cmd" >&2
    need=1
  fi
done
if [ "$need" -ne 0 ]; then
  exit 1
fi

printf '%-62s %7s %10s %s\n' "Deck" "Pages" "LastTotal" "Status"
printf '%-62s %7s %10s %s\n' "----" "-----" "---------" "------"

status=0

while IFS= read -r -d '' pdf; do
  rel="${pdf#"$ROOT_DIR"/}"
  pages="$(pdfinfo "$pdf" 2>/dev/null | awk '/^Pages:/ {print $2}')"
  tmp="$(mktemp)"
  pdftotext -layout "$pdf" "$tmp" >/dev/null 2>&1 || true
  last_total="$(grep -E '^[[:space:]]+[0-9]+[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*$' "$tmp" | tail -n 1 | sed -E 's/.*\/[[:space:]]*([0-9]+)[[:space:]]*$/\1/')"
  if grep -Eq '^[[:space:]]+[0-9]+[[:space:]]*/[[:space:]]*100[[:space:]]*$' "$tmp"; then
    result="FAIL /100"
    status=1
  elif [ -z "$last_total" ]; then
    result="WARN no-footer"
  else
    result="OK"
  fi
  rm -f "$tmp"
  printf '%-62s %7s %10s %s\n' "$rel" "${pages:-?}" "${last_total:-?}" "$result"
done < <(find "$ROOT_DIR" \
  -path '*/archive_pre_split/*' -prune -o \
  -path '*/papers-to-discuss/*' -prune -o \
  -path '*/Lec*/Lec*_T*.pdf' -print0 | sort -z)

exit "$status"
