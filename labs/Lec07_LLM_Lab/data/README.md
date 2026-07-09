# FOMC statement corpus

**Contents:** every Federal Open Market Committee (FOMC) post-meeting policy
statement from **1994-02-04 through 2026-06-17** — 245 plain-text files,
~580 KB. One file per statement, named `fomc/fomc_YYYYMMDD.txt`. Each file
starts with a 2-line provenance header (`# FOMC statement, YYYY-MM-DD` and
`# source: <url>`) followed by the statement body.

**Source & license:** fetched from the Federal Reserve Board's website
(www.federalreserve.gov) on 2026-07-09. FOMC statements are works of the
U.S. federal government and are in the public domain.

**How it was built** (a one-off build script, not shipped with the lab):

1. Statement URLs discovered from the Fed's index pages —
   `monetarypolicy/fomccalendars.htm` (2021–present) and
   `monetarypolicy/fomchistorical{YYYY}.htm` (1994–2020) — by taking each
   anchor whose link text is exactly "Statement" (plus the
   `monetaryYYYYMMDDa.htm` press-release links on the calendar page).
2. Each page fetched with a 1-second politeness delay and converted
   HTML → text with Python's stdlib `html.parser`: navigation trimmed to the
   statement body, whitespace collapsed, entities unescaped, and the text
   cut after the voting / discount-rate paragraph.
3. Companion releases (implementation notes, balance-sheet plans) are
   excluded. The corpus is post-meeting policy statements plus a handful of
   special FOMC statements issued between meetings (intermeeting moves such
   as 2007-08-17, the 2019-10-11 reserve-management announcement, the
   2025-08-22 framework-review update) — the notebooks point these out where
   they matter. Two 1994 statements are genuinely short (2 paragraphs) —
   that is how they were issued.

**To regenerate or extend:** repeat the recipe above (any HTML-to-text
approach works), or replace this folder and point the notebooks at your own
copy with the environment variable `FOMC_DATA_DIR` (see `../fomc_data.py`).
