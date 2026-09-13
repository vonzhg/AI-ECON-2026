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

Birth year is missing on every row, each with a note: correct under the rules, and not a defect.

## The mechanical gate

`checks/require_source_url.py` reports exactly lines 7, 9 and 11 on this file and passes the unaltered pilot batch. As a Claude Code **PreToolUse** hook on `Write` it blocks the write; as a PostToolUse hook it could only complain afterwards (checked on Claude Code 2.1.269).
