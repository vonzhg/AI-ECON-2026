# Lecture 9 CLI Facts — September 2026

The one place Lecture 9's decks and lab take dated harness facts from (plan v2 §0.3, review A8.4). Every deck cites this file once, in a footnote; nothing below is restated as undated prose on a slide.

Established on the course development machine on **2026-09-12**. Re-run `tools/lec09_cli_probes.sh` at the freshness gate (Oct 19–20) and update the dates.

**How each fact was established:** **RUN** = observed by running it · **HELP** = read from the CLI's own `--help` or feature list · **SRC** = read from the installed package (source or binary strings), not yet run · **OPEN** = not yet established.

## Versions and authentication

| Harness | Version | Headless auth on the dev machine |
|---|---|---|
| Claude Code | 2.1.269 (auto-updated to 2.1.270 the same afternoon: the version drifts under you; re-run the probes at the gate) | logged in — RUN |
| Gemini CLI | 0.47.0 | **blocked**: `security.auth.selectedType = gemini-api-key` with no `GEMINI_API_KEY`; `~/.gemini/gemini-credentials.json` reported corrupted — RUN |
| Codex CLI | 0.154.0 (installed 2026-09-12, `npm install -g @openai/codex`) | **not logged in** — RUN |

## One concept, three harnesses

| Concept | Claude Code 2.1.269 | Gemini CLI 0.47.0 | Codex CLI 0.154.0 |
|---|---|---|---|
| Project instructions file | `CLAUDE.md` (project and `~/.claude/`) — RUN (lab) | `GEMINI.md` by default; file name configurable (`contextFileName`) — SRC; whether `AGENTS.md` loads by default — OPEN | `AGENTS.md`, with `AGENTS.override.md` taking precedence — SRC |
| Custom sub-agents | `.claude/agents/*.md`, YAML front matter `name`, `description`, `tools`, `model` — RUN (lab, 2.1.261/2.1.269); `/agents`, `--agent`, `--agents` — HELP | `.gemini/agents/*.md`, YAML front matter (`name`, `description`, `tools`, `model` defaulting to `inherit`; `kind: local` or `remote`) — SRC; `experimental.enableAgents` defaults to true — SRC; **skipped in untrusted folders** — RUN | sub-agents through a `spawn_agent` tool with an `agent_type` — SRC; `multi_agent` feature stable and on — HELP; where custom agent types are defined — OPEN |
| How a dispatch appears in the transcript | an `Agent` tool call carrying `subagent_type` — RUN (lab, 2.1.261; re-run on 2.1.270 on 12 Sep 2026: a rung-1 review prompt dispatched `code-reviewer`, then `math-agent`) | OPEN | a `spawn_agent` tool call — SRC |
| Skills | `.claude/skills/<name>/SKILL.md`; `disable-model-invocation: true` — RUN (repo demo) | `.gemini/skills/<name>/SKILL.md` — SRC; `gemini skills list/enable/disable/install/link` — HELP; built-in skills listed — RUN; workspace skill not listed in an untrusted folder — RUN | `$CODEX_HOME/skills/<name>` (default `~/.codex/skills`) — SRC; `skill_search` stable and on — HELP |
| Headless run (rung 3) | `claude -p "…" --output-format json` or `stream-json` — RUN | `gemini -p "…" -o json` or `stream-json` — HELP | `codex exec "…" --json` (JSONL events), `--output-schema <file>`, `-o <last-message file>` — HELP |
| Read-only or planning mode | `--permission-mode plan`; modes: `acceptEdits`, `auto`, `bypassPermissions`, `manual`, `dontAsk`, `plan` — HELP | `--approval-mode plan`; modes: `default`, `auto_edit`, `yolo`, `plan` — HELP | `-s read-only`; sandboxes: `read-only`, `workspace-write`, `danger-full-access` — HELP |
| Tool allowlist | `--allowedTools "Read Grep Glob"`; `permissions.allow` / `deny` in `settings.json` — HELP | `--allowed-tools` is deprecated in favour of the Policy Engine — HELP | sandbox mode plus approval policy; `-c key=value` config overrides — HELP |
| Hooks | `settings.json` → `hooks`. **PreToolUse** with exit code 2 blocks the call, and the hook's stderr reaches the model; for `Write` the hook receives `tool_input.file_path` and `tool_input.content` — RUN. **PostToolUse** fires after the write: exit code 2 only reports back, the file stays written — RUN. The matcher is a tool-name pattern: a hook with matcher `Write(*.tsv)` **never fired** while `data.tsv` was written — RUN. Other hook events include `Stop`, `SubagentStop`, `PreCompact`, `SessionStart`, `UserPromptSubmit`, and `Notification` — SRC (2.1.270 binary) | events `BeforeTool`, `AfterTool`, `BeforeAgent`, `AfterAgent`, `BeforeModel`, `AfterModel`, `BeforeToolSelection`, `Notification` — SRC; **project hooks disabled in untrusted folders** — RUN; blocking semantics — OPEN | `hooks` feature stable and on — HELP; "Tool call blocked by PreToolUse hook" — SRC; configuration file and blocking semantics — OPEN |
| Settings file | `.claude/settings.json` (project) — RUN (the hook tests); `~/.claude/settings.json` (user) — present on the dev machine | `.gemini/settings.json` (workspace), `~/.gemini/settings.json` (user), plus a system file — SRC; the user file holds `security.auth.selectedType` — RUN | `~/.codex/config.toml` — SRC (not created on the dev machine before login) |
| Compaction controls | `/compact` takes optional summarization instructions — SRC (2.1.270 binary: `argumentHint` "optional custom summarization instructions"); `--autocompact <auto\|tokens>`, "Auto-compact window size (auto, or 100k–1M tokens)" — HELP (2.1.270) | OPEN | OPEN |
| Background sessions | `claude --bg`; `claude agents`, `logs <id>`, `stop <id>` — HELP | OPEN | `codex agents` browses agent sessions — HELP |
| Per-run turn cap | no `--max-turns` flag — HELP | OPEN | OPEN |
| Workspace trust | trust prompt on first interactive start | untrusted folders skip project agents, hooks, and workspace skills; `--skip-trust` trusts the workspace for one session — RUN / HELP | `--skip-git-repo-check` needed outside a git repository — HELP |

## Findings that change slides or the lab

1. **T5b's hook frame (T4-F15).** The slide's matcher `"Write(data/*.csv)"` is not a working matcher on Claude Code 2.1.269: it fails *silently* — the kind of failure the lecture warns about. Use a `Write` matcher and filter on `tool_input.file_path` inside the hook; block with **PreToolUse** (exit 2). A PostToolUse hook cannot reject a write. — RUN
2. **Gemini CLI folder trust.** Project agents, hooks, and workspace skills are ignored until the folder is trusted. The lab's Step 0 must check this, and T2b's harness table must say it. — RUN
3. **Archived mapping table.** Archive v1's `.agents/agents/` and `.agents/rules/` entries for Codex are not supported by anything found in Codex 0.154.0; treat them as wrong until a logged-in run says otherwise. — SRC
4. **Portability.** All three harnesses have an instructions file, `SKILL.md` skills, some form of sub-agent, hooks, and a headless JSON mode. Rung 3 is therefore the portable acceptance test (review A2.3); the rung-1 and rung-2 surfaces differ by harness.

## Open items (need a logged-in run)

- **Gemini CLI** (after the instructor fixes auth): `GEMINI.md` vs `AGENTS.md` loading; `.gemini/agents` dispatch and how it appears in `stream-json`; workspace skill activation by description; `BeforeTool` blocking; the `-o json` shape.
- **Codex CLI** (after `codex login`): where custom agent types live; how `spawn_agent` appears in `--json`; project skill discovery; hook configuration and PreToolUse blocking; the `codex exec --json` event shape.
- **Both:** lab scenario S1 at rung 3, k=3 fresh runs.
