# Who gained health coverage after 2013? — one real agent session

Recorded 2026-09-13 for the Lecture 9 cold open (T1). One session, four prompts, started in an empty folder.

| | |
|---|---|
| Harness | Claude Code 2.1.270, headless (`-p --output-format stream-json`, prompts 2–4 with `--resume`), `--permission-mode dontAsk` with an allowlist: Read, Write, Edit, Glob, Grep, WebFetch, WebSearch, and Bash only for python3, curl, ls, head, cat, pip |
| Model | claude-sonnet-5 |
| Prompt 1 — the data | 60 turns · 59 tool calls · 672 s · $1.92 — ended by asking a question, no data saved |
| Prompt 2 — the economist's decision | 11 turns · 10 tool calls · 208 s · $0.63 — CSV and SOURCES.md |
| Prompt 3 — the figure | 26 turns · 22 tool calls · 206 s · $1.08 |
| Prompt 4 — the question | 6 turns · 4 tool calls · 225 s · $0.50 |
| Total | 95 tool calls · 1,311 s (about 22 minutes) · $4.12 |

Not in the recording: a first attempt at prompt 2 that assumed prompt 1 had saved a CSV. It was stopped within a minute and removed from the session history before the real prompt 2, so the model never saw it.

## What the run shows

- **The agent found the right table fast, then hit a wall only a person could cross.** The Census data API now refuses requests without a registered key (checked independently: an unkeyed request is redirected to `missing_key.html`). The key is emailed to a person. After trying the no-key historical tables, it stopped and asked: wait for a key, or use tables whose margins of error start in 2017?
- **Two things it got wrong before asking.** It said ACS table S2701 has the same age groups with margins of error for every year from 2008. It does not: in 2013 the table's uninsured percentages have three age groups, and the bands were redrawn in 2017 (18–24 became 19–25). Its no-key fallback would have joined CPS tables across the 2014 redesign of the CPS health-insurance questions without saying so.
- **The economist's reply (prompt 2) was the judgment it lacked:** stay with one survey, check each table's age groups before combining years, and build three consistent groups from the no-key ACS tables. It did that in under four minutes and documented the derivations and margins of error.
- **The numbers check out.** Recomputed independently from Census tables HIC-4_ACS, HIC-5_ACS, and HIC-6_ACS (United States rows): all 48 rows match to 0.1 point. From 2013 to 2016 the uninsured rate fell 8.4 points for ages 19 to 64 (20.5% to 12.1%), 2.8 points under 19 (7.5% to 4.7%), and 0.2 points for 65 and over (1.0% to 0.8%, inside its margin of error).
- **One error survived into the final figure.** It labels 2020 "ACS data not collected". The 2020 data were collected; the Census Bureau did not release the standard 1-year estimates because the pandemic disrupted collection. The agent's own SOURCES.md says this correctly; the figure does not.
- **The verbs stayed descriptive.** "Gained the most coverage after 2014", with 2013–2016 changes labelled, not "the ACA caused". Contrast the GDP-by-party trace in T2a.

Files: this transcript (`.jsonl`: prompts, tool calls, results truncated to 400 characters, identifiers removed) and the final figure, `source/Lec09_Agentic_AI/figures/uninsured_by_age_agent.png`.

## Every tool call, in order

### Prompt 1

> I'm starting a new project in this empty folder. I study health insurance coverage in the United States. Find the Census Bureau's official estimates of the percentage of people without health insurance, by age group, for every year from 2008 to the latest available. Download the data and save a tidy CSV with columns year, age_group, pct_uninsured, and margin_of_error, plus a SOURCES.md that records each source URL and today's date.

1. **Bash** `curl -s -o /dev/null -w "%{http_code}\n" "https://api.census.gov/data.json" --max-time 10`
2. **ToolSearch** `select:WebFetch,WebSearch`
3. **Bash** `curl -s "https://api.census.gov/data/2023/acs/acs1/subject/variables.json" --max-time 30 -o /tmp/s2701_vars_20`
4. **Bash** `for y in 2008 2009 2010 2011 2012 2019 2020 2021 2023 2024 2025; do`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
5. **Bash** `curl -s --max-time 15 "https://api.census.gov/data/2023/acs/acs1/subject?get=NAME,S2701_C05_001E&for=us:*"`
6. **Bash** `curl -sv --max-time 15 "https://api.census.gov/data/2023/acs/acs1/subject?get=NAME,S2701_C05_001E&for=us:*" -o`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
7. **Bash** `curl -s --max-time 15 "https://api.census.gov/data/2023/acs/acs1/subject?get=NAME,S2701_C05_001E&for=us:*" -o `  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
8. **Bash** `echo hello`
9. **Bash** `curl -s --max-time 15 "https://api.census.gov/data/2023/acs/acs1/subject?get=NAME,S2701_C05_001E&for=us:*"`
10. **Bash** `curl -s -i --max-time 15 "https://api.census.gov/data/2023/acs/acs1/subject?get=NAME,S2701_C05_001E&for=us:*"`
11. **Bash** `env | grep -i census; env | grep -i CENSUS_API; echo "---"; env | grep -iE "api_key|apikey" `  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
12. **Bash** `printenv CENSUS_API_KEY`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
13. **Bash** `curl -s -i --max-time 15 -A "Mozilla/5.0 (Research; census-data-download)" "https://api.census.gov/data/2023/a`
14. **WebFetch** `https://www.census.gov/data/tables/time-series/demo/health-insurance/historical-series/hic.html`
15. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
16. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
17. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
18. **Bash** `curl -s --max-time 20 -A "Mozilla/5.0" "https://www2.census.gov/programs-surveys/demo/tables/health-insurance/`
19. **Bash** `ls -la /tmp/hic04_acs.xlsx`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
20. **Bash** `python3 -c "`  — *error returned to the model:* Exit code 1
Traceback (most recent call last):
  File "~/.local/lib/python3.9/site-package…
21. **Bash** `which libreoffice soffice ssconvert 2>&1; python3 -c "import xlrd; print('xlrd ok')" 2>&1; python3 -c "import `
22. **Bash** `python3 -c "`
23. **Bash** `python3 -c "`
24. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
25. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
26. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
27. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
28. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
29. **Bash** `curl -s --max-time 20 -A "Mozilla/5.0" "https://www2.census.gov/programs-surveys/demo/tables/health-insurance/`
30. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
31. **Bash** `curl -s -A "Mozilla/5.0" --max-time 30 "https://www.census.gov/data/tables/time-series/demo/health-insurance/h`
32. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/health-insurance/d`
33. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`
34. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`
35. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`
36. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
37. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`
38. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 -o /dev/null -w "%{http_code}\n" "https://data.census.gov/table/ACSST1Y`
39. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/topics/health/health-insurance/data/tables.html`
40. **Bash** `for n in 01 02 03 07 08 10 11; do`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
41. **Bash** `python3 -c "`
42. **Bash** `python3 -c "`
43. **Bash** `python3 -c "`
44. **Bash** `python3 -c "`
45. **Bash** `pip install --user xlrd 2>&1 | tail -5`
46. **Bash** `python3 -c "`
47. **Bash** `python3 -c "`
48. **Bash** `python3 -c "`
49. **Bash** `python3 -c "`
50. **Bash** `python3 -c "`
51. **WebFetch** `https://api.census.gov/data/2023/acs/acs1/subject?get=NAME,S2701_C05_001E,S2701_C05_011E,S2701_C05_012E,S2701_`
52. **Bash** `python3 -c "`
53. **WebSearch** `census.gov "Health Insurance Historical Tables" "by Age and Sex" table 2013 2014 2015 2016 site:census.gov`
54. **WebSearch** `Census Bureau CPS ASEC "Table HI-01" health insurance coverage by age historical archive xls`
55. **Bash** `python3 -c "`
56. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`
57. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
58. **Bash** `curl -s -A "Mozilla/5.0" --max-time 20 "https://www.census.gov/data/tables/time-series/demo/income-poverty/cps`
59. **Bash** `python3 -c "`

### Prompt 2

> No key today. Don't splice surveys: stay with the ACS, and don't assume a table's age groups are the same every year -- check before you combine years. The no-key ACS historical tables give all persons, children under 19, and people under 65 from 2008 on; build three consistent groups from them for the United States (under 19, 19 to 64, 65 and over), leave 2020 missing, and record in SOURCES.md how you derived each group and its margin of error.

60. **Bash** `python3 -c "`
61. **Bash** `python3 -c "`
62. **Bash** `python3 << 'EOF'`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
63. **Bash** `python3 -c "`
64. **Bash** `python3 -c "`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
65. **Bash** `python3 -c "`
66. **Bash** `pwd; ls -la`
67. **Bash** `python3 -c "`
68. **Write** `<run-dir>/uninsured_by_age.csv`
69. **Write** `<run-dir>/SOURCES.md`

### Prompt 3

> Make a publication-quality figure from the CSV: the uninsured rate for each group over time, with its margin of error, and a source note. Save it as uninsured_by_age.png together with the script that makes it.

70. **Skill** ``
71. **Read** `/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz/references/palette.md`
72. **Read** `/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz/references/marks-and-anatomy`
73. **Bash** `node "/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz/scripts/validate_palet`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
74. **Bash** `node "/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz/scripts/validate_palet`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
75. **Bash** `node "/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz/scripts/validate_palet`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
76. **Bash** `node --version`
77. **Bash** `cd "/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz" && node scripts/validat`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
78. **Bash** `node "/tmp/claude-77514/bundled-skills/2.1.270/360525aa3b2e5ff674b6d8394094051c/dataviz/scripts/validate_palet`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
79. **Bash** `node -e "console.log('hello node')"`  — *error returned to the model:* Permission to use Bash has been denied because Claude Code is running in don't ask mode. I…
80. **Bash** `python3 -c "import matplotlib; print(matplotlib.__version__)"`
81. **Write** `<run-dir>/make_chart.py`
82. **Edit** `<run-dir>/make_chart.py`
83. **Edit** `<run-dir>/make_chart.py`
84. **Bash** `python3 "<run-dir>/make_chart.py"`
85. **Glob** `*`
86. **Read** `<run-dir>/uninsured_by_age.png`
87. **Edit** `<run-dir>/make_chart.py`
88. **Edit** `<run-dir>/make_chart.py`
89. **Edit** `<run-dir>/make_chart.py`
90. **Bash** `python3 "<run-dir>/make_chart.py"`
91. **Read** `<run-dir>/uninsured_by_age.png`

### Prompt 4

> Now make the figure answer one question: who gained coverage after 2013, when the Marketplaces opened and Medicaid expanded? Mark January 2014, and label how many percentage points each group's rate fell from 2013 to 2016.

92. **Read** `<run-dir>/make_chart.py`
93. **Write** `<run-dir>/make_chart.py`
94. **Bash** `python3 "<run-dir>/make_chart.py"`
95. **Read** `<run-dir>/uninsured_by_age.png`
