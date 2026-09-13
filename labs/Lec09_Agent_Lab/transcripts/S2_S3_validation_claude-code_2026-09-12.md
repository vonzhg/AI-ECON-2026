# S2 and S3 — validation runs

Recorded 2026-09-12 with `scripts/detection_matrix.py` on Claude Code 2.1.270, each agent on the model its brief names (`sonnet`), **one** fresh headless run per agent (k = 1: a check that each scenario is winnable and its controls are sound, not a reliability estimate). Each run sees only the scenario's `files`. Scores below are after the control fixes described at the end, re-scored with `--rescore` from the saved replies, not re-run.

## S2 — a leading question in a new costume

| | code-reviewer | domain-reviewer | econ-agent |
|---|---|---|---|
| an 11-month change labelled year over year (line 21) | **1/1** | 1/1 | 0/1 |
| a merge that shifts CPI a year forward (line 24) | **1/1** | 1/1 | 1/1 |
| a correlation printed as an effect (lines 27–28) | 0/1 | **1/1** | 1/1 |
| *control:* sorting CPI by date before differencing (line 18) | 0/1 | 0/1 | 0/1 |
| *control:* calling the Mar 2022–Jul 2023 hiking window wrong (line 33) | 0/1 | 0/1 | 0/1 |
| other findings | 1 | 1 | 0 |

## S3 — Fellows provenance audit

| | provenance-checker | econ-agent |
|---|---|---|
| a field bucket without its source URL (line 7) | **1/1** | 1/1 |
| a PhD year without its source URL (line 9) | **1/1** | 1/1 |
| gender coded from a first name, no source (line 11) | **1/1** | 1/1 |
| *control:* gender from an explicit pronoun, with its source (line 6) | 0/1 | 0/1 |
| *control:* gender left missing because no source states it (line 8) | 0/1 | 0/1 |
| other findings | 0 | 3 |

Bold: the agent the scenario expects to catch that defect. `provenance-checker` is the instructor's reference brief (`instructor/briefs/`); students replace it with their own.

**What the runs show**

- Both scenarios are winnable by one well-briefed agent, and each specialist caught its own defects.
- **A control can fire on a sound finding.** As first written, S2's line-33 control fired on `domain-reviewer`: it said the (correct) hiking window is shaded but never analyzed, which is the pooled-slope defect seen from the plot. S3's controls had the same exposure: lines 6 and 8 also lack a note on the missing birth year, and a finding about that is sound. Controls may now pair `lines` with a `pattern`, firing only on a claim of the wrong kind at those lines (dates for S2, gender for S3).
- **The right line with the wrong reason.** On S3, both agents cited line 7 but called it a 17-column row; every line has 18 fields and the agents miscounted the run of empty fields. Line-tied scoring cannot see this. The gate script, which parses the CSV, got the reason right.
- The sound "other" findings are listed in `instructor/ANSWER_KEY_S2.md` and `ANSWER_KEY_S3.md`.

## S3 at k = 3 (Claude Code 2.1.270, 12 Sep 2026)

| | provenance-checker | econ-agent |
|---|---|---|
| a field bucket without its source URL (line 7) | **3/3** | 3/3 |
| a PhD year without its source URL (line 9) | **3/3** | 3/3 |
| gender coded from a first name, no source (line 11) | **3/3** | 3/3 |
| *control:* gender from an explicit pronoun (line 6) | 0/3 | 0/3 |
| *control:* gender left missing, no source (line 8) | 0/3 | 0/3 |
| other findings | 0 | 2 |

## S2 at k = 3 (Claude Code 2.1.270, 13 Sep 2026)

A first attempt on 12 Sep failed on all nine calls with a CLI error while other headless jobs were finishing; this is the re-run on its own.

| | code-reviewer | domain-reviewer | econ-agent |
|---|---|---|---|
| an 11-month change labelled year over year (line 21) | **3/3** | 3/3 | 0/3 |
| a merge that shifts CPI a year forward (line 24) | **3/3** | 3/3 | 2/3 |
| a correlation printed as an effect (lines 27–28) | 0/3 | **3/3** | 3/3 |
| *control:* sorting CPI by date before differencing (line 18) | 0/3 | 0/3 | 0/3 |
| *control:* calling the Mar 2022–Jul 2023 hiking window wrong (line 33) | 0/3 | 0/3 | 0/3 |
| other findings | 4 | 0 | 3 |

Every specialist caught its own defects in all three runs, and no control fired.

**Still to do before Oct 22:** S1–S3 on a second harness (Gemini CLI or Codex CLI) once one is logged in.
