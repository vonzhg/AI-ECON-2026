# Source rules for the Fellows dataset (Lecture 10, Exercise 10)

- Use only official and institutional sources for biography fields.
- Preserve the source URL for every extracted field: `phd_year` needs `source_phd`, `birth_year` needs `source_birth`, `gender_explicit` needs `source_gender`, `research_field_bucket` needs `source_field`.
- Code `gender_explicit` only when an official source states pronouns or otherwise states gender explicitly. If the source is silent, leave the field missing.
- Record missing as missing, and say why in `notes`. Missingness is a result, not a failure.
