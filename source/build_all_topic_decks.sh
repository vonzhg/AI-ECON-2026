#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"

if ! command -v pdflatex >/dev/null 2>&1; then
  printf 'ERROR: pdflatex not found on PATH.\n' >&2
  exit 1
fi

find "$ROOT_DIR" \
  -path '*/archive_pre_split/*' -prune -o \
  -path '*/papers-to-discuss/*' -prune -o \
  -name 'MPE_Hyper_v44.tex' -prune -o \
  -path '*/Lec*/Lec*_T*.tex' -print0 \
| sort -z \
| while IFS= read -r -d '' deck; do
  dir="$(dirname "$deck")"
  file="$(basename "$deck")"
  stem="${file%.tex}"

  # Honor a "% !TEX program = lualatex|xelatex" magic comment in the first lines
  # (needed for decks with CJK, e.g. Lec02_T2); default to pdflatex otherwise.
  engine=pdflatex
  if head -n 5 "$deck" | grep -qiE '^%[[:space:]]*![[:space:]]*TEX[[:space:]]+program[[:space:]]*=[[:space:]]*(lua|xe)latex'; then
    engine="$(head -n 5 "$deck" | grep -oiE '(lua|xe)latex' | head -n1)"
    if ! command -v "$engine" >/dev/null 2>&1; then
      printf '  WARN: %s not found; falling back to pdflatex for %s\n' "$engine" "$file" >&2
      engine=pdflatex
    fi
  fi

  printf 'Building %s/%s  (%s)\n' "$dir" "$file" "$engine"
  (
    cd "$dir"
    "$engine" -interaction=nonstopmode -halt-on-error "$file" >/tmp/"${stem}".pass1.log
    "$engine" -interaction=nonstopmode -halt-on-error "$file" >/tmp/"${stem}".pass2.log
  )
done

printf 'Done. Active topic PDFs rebuilt with two LaTeX passes.\n'
