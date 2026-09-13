# S3 — Fellows provenance audit

**The situation.** A batch of ten rows came back for the Econometric Society Fellows dataset (Lecture 10, T4). It is built from official and institutional pages; the project's source rules are in `rules/provenance.md`. Before it is merged, someone has to audit it.

**Your job in the Design Lab (about 50 minutes)**

1. Fill the eight cells of `AGENT_DESIGN_CANVAS.md` for a `provenance-checker`: an agent that audits a batch against the rules, row by row, and never fills or changes a field itself.
2. Write its brief and point `scenario.json` at it (replace the instructor's reference brief).
3. Run it at rung 3, three times:
   `python3 scripts/detection_matrix.py scenarios/S3_fellows_provenance --harness <yours> --k 3`
4. Then enforce the rules mechanically, without trusting any agent:
   - **Any harness:** `python3 scenarios/S3_fellows_provenance/checks/require_source_url.py scenarios/S3_fellows_provenance/data/pilot_batch_audit.csv`
   - **Claude Code:** copy `claude_settings_example.json` to `.claude/settings.json` inside this folder, then ask an agent to write a CSV row whose `phd_year` has no `source_phd` --- the PreToolUse hook blocks the write before it happens.
5. Pitch: your matrix row with rates, the control rows, and what the mechanical gate caught that the agent did not --- or the reverse.

**Why this matters.** An agent can fill 882 rows faster than you can read ten. Whether the dataset is usable depends on whether each value can be traced to a source, and whether a missing value stayed missing.

Do not open `instructor/` until your team has pitched.
