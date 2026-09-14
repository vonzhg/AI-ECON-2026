# Answer key — S3, Fellows provenance audit

*Instructor copy. Kept out of `scenarios/` so reviewing agents cannot read it.*

`data/pilot_batch_audit.csv` is the Lecture 10 Exercise 10 pilot batch (ten current Fellows, official and institutional sources only) with **three provenance fields altered**. No value was invented: every `phd_year`, field bucket and gender in the file is the one the pilot recorded, from an explicit official statement. What changed is the provenance, which is exactly what the rules protect.

## Planted defects

| CSV line | Row | What was altered | Rule broken |
|---|---|---|---|
| 7 | David Autor | `source_field` emptied; `research_field_bucket` = labor kept | every extracted field keeps its source URL |
| 9 | Aleh Tsyvinski | `source_phd` emptied; `phd_year` = 2003 kept | every extracted field keeps its source URL |
| 11 | Donald Andrews | `source_gender` emptied; note changed to "Gender coded from first name." | gender only from an explicit statement; and its source URL |

Line 11 is the instructive one: the value happens to be right, and the process is still a violation. A reviewer that says "looks correct" has confused the answer with the method.

## Controls (must stay silent)

| CSV line | Row | Why it is correct |
|---|---|---|
| 6 | Alberto Abadie | gender coded from an explicit pronoun on the MIT profile, with `source_gender` |
| 8 | Philip A. Haile | gender left missing because the source did not state pronouns --- missingness is a result |

The controls fire only on a claim about **gender** at those lines (`scenario.json` pairs each line with a pattern).

Birth year is missing on every row. Only lines 2 and 4 say why in `notes`; the other rows' notes are about something else (line 11's birth-year note was replaced by the planted one). Under the last rule ("say why in `notes`") that is a small gap in the pilot batch itself, not a planted defect: a finding about it is sound, scores as "other", and does not fire a control.

## The mechanical gate

`checks/require_source_url.py` reports exactly lines 7, 9 and 11 on this file and passes the unaltered pilot batch. As a Claude Code **PreToolUse** hook on `Write` it blocks the write; as a PostToolUse hook it could only complain afterwards (checked on Claude Code 2.1.269).

## What the validation runs showed (Claude Code 2.1.270, k = 1, 2026-09-12)

- `provenance-checker` (the reference brief) and `econ-agent` both cited lines 7, 9 and 11; neither fired a control.
- **Both described line 7 wrongly**: they called it a 17-column row whose fields had shifted. Every line has 18 fields (Python's `csv` module; the gate script parses it the same way) --- the agents miscounted the run of empty fields `,,,`. Line-tied scoring counts the finding, so the matrix cannot see this; reading the claim still can. It is the S3 pitch in one example: the mechanical gate got the reason right where both agents did not.
- `econ-agent`'s other findings: missing birth-year reasons on lines 5 and 10 (sound, see above, but it skipped the other rows with the same gap), and that `official_profile_url` is the same roster page on every row, so it cannot corroborate a row's `election_year` (a fair point about the pilot's design).

Replies: `results/S3_fellows_provenance/claude-20260912-194931/` (local, gitignored); summary in `transcripts/S2_S3_validation_claude-code_2026-09-12.md`.
