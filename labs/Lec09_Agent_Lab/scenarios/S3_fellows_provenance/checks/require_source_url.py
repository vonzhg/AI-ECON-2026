"""Gate for the Fellows source rules: reject a batch that breaks them.

  python3 checks/require_source_url.py data/pilot_batch_audit.csv
      A check inside a script or the rung-3 runner (any harness).

  As a Claude Code PreToolUse hook on Write (see ../claude_settings_example.json):
      the pending tool call arrives as JSON on stdin; the script reads
      tool_input.file_path and tool_input.content and exits 2 to block the write.
      A PostToolUse hook could only complain after the file was written
      (verified on Claude Code 2.1.269; CLI_FACTS_2026-09.md).
"""
import csv
import io
import json
import sys

NEEDS_SOURCE = {"phd_year": "source_phd", "birth_year": "source_birth",
                "gender_explicit": "source_gender", "research_field_bucket": "source_field"}


def problems(text: str) -> list[str]:
    out = []
    for n, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        for field, source in NEEDS_SOURCE.items():
            if (row.get(field) or "").strip() and not (row.get(source) or "").strip():
                out.append(f"line {n}: {field} is filled but {source} is empty")
        notes = (row.get("notes") or "").lower()
        if (row.get("gender_explicit") or "").strip() and any(w in notes for w in ("first name", "inferred", "guessed")):
            out.append(f"line {n}: gender_explicit is not from an explicit statement")
    return out


def main() -> int:
    if len(sys.argv) > 1:
        found = problems(open(sys.argv[1], encoding="utf-8").read())
    else:
        call = json.load(sys.stdin)
        tool_input = call.get("tool_input") or {}
        if not str(tool_input.get("file_path", "")).endswith(".csv"):
            return 0
        found = problems(tool_input.get("content", ""))
    for p in found:
        print(p, file=sys.stderr)
    return 2 if found else 0


if __name__ == "__main__":
    sys.exit(main())
