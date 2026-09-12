#!/usr/bin/env bash
# Re-verify the dated harness facts in source/Lec09_Agentic_AI/CLI_FACTS_2026-09.md.
# Runs in a throw-away directory; the Claude Code hook probe makes one small
# headless call (about $0.06 on 2026-09-12). Gemini and Codex probes need no login.
set -uo pipefail
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo "== versions ($(date -I))"
for c in claude gemini codex; do printf '%-7s ' "$c"; command -v "$c" >/dev/null && "$c" --version 2>/dev/null | grep -v 'arg0 temp' | head -1 || echo "not installed"; done

echo "== claude: permission modes and rung surfaces (help)"
claude --help 2>&1 | grep -E -- '--permission-mode|--agent |--agents|--bg|max-turns' || true
claude --help 2>&1 | grep -A2 -- '--permission-mode' | grep -o 'choices:.*' || true

echo "== claude: hook semantics (run)"
mkdir -p "$WORK/hooks/.claude" && cd "$WORK/hooks" && git init -q .
cat > .claude/gate.py <<'PY'
import json, sys, os
mode = sys.argv[1]; data = json.load(sys.stdin)
name = os.path.basename((data.get("tool_input") or {}).get("file_path", ""))
open(".claude/hook_log.txt", "a").write(f"{mode} {data.get('hook_event_name')} {name}\n")
if mode == "pre" and name == "blocked.csv": print("blocked by PreToolUse", file=sys.stderr); sys.exit(2)
if mode == "post" and name == "post.csv": print("flagged by PostToolUse", file=sys.stderr); sys.exit(2)
if mode == "pretsv": print("tsv matcher fired", file=sys.stderr); sys.exit(2)
PY
cat > .claude/settings.json <<'JSON'
{"hooks": {"PreToolUse": [{"matcher": "Write", "hooks": [{"type": "command", "command": "python3 .claude/gate.py pre"}]},
                          {"matcher": "Write(*.tsv)", "hooks": [{"type": "command", "command": "python3 .claude/gate.py pretsv"}]}],
           "PostToolUse": [{"matcher": "Write", "hooks": [{"type": "command", "command": "python3 .claude/gate.py post"}]}]}}
JSON
claude -p "Use the Write tool to create four files, one call each, without asking: blocked.csv ('a,b'), allowed.csv ('c,d'), post.csv ('x,y'), data.tsv ('p	q')." \
  --model sonnet --permission-mode acceptEdits --output-format json < /dev/null > run.json 2>/dev/null
echo "files written: $(ls *.csv *.tsv 2>/dev/null | tr '\n' ' ')"
echo "expected:      allowed.csv data.tsv post.csv   (blocked.csv absent = PreToolUse blocks; post.csv present = PostToolUse cannot)"
grep -q pretsv .claude/hook_log.txt && echo "Write(*.tsv) matcher FIRED (update CLI_FACTS)" || echo "Write(*.tsv) matcher did not fire (as recorded)"

echo "== gemini: help surface and trust"
command -v gemini >/dev/null && { gemini --help 2>&1 | grep -E -- '--approval-mode|--skip-trust|--output-format' ; gemini skills --help 2>&1 | grep -E '^  gemini skills' ; }
echo "== codex: help surface and features"
command -v codex >/dev/null && { codex exec --help 2>&1 | grep -E -- '--json|--sandbox|--output-schema'; (cd /tmp && codex features list 2>/dev/null | grep -E '^(hooks|multi_agent|skill_search) '); codex login status 2>&1 | grep -v 'arg0 temp' | head -1; }
