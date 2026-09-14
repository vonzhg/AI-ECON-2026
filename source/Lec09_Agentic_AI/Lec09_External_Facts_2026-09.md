# Lecture 9 External Facts — September 2026

The outside claims Lecture 9's decks make, with the source each was checked against. The course's `claim-checker` can only search this repository, so this file is where the web checks live (review §7). The companion for harness facts is `CLI_FACTS_2026-09.md`.

All checked on **2026-09-12**. Re-check the rows marked **re-check** at the freshness gate (Oct 19–20).

**How checked:** **PAGE** = the source page was fetched and read · **SEARCH** = search results and at least one secondary report agree; the primary page was not read · **PRESS** = only press or blog reports · **BIB** = a bibliographic reference, not re-fetched.

## Research practice and publishing

| Claim, as the decks use it | Deck | Source | What the source says | How |
|---|---|---|---|---|
| Cold open: uninsured rate by age group, 2008–2024; from 2013 to 2016 it fell 8.4 points for ages 19–64 (20.5% to 12.1%), 2.8 under 19, 0.2 for 65 and over; no standard 2020 1-year estimates | T1 | Census [HIC-4_ACS](https://www2.census.gov/programs-surveys/demo/tables/health-insurance/time-series/acs/hic04_acs.xlsx), [HIC-5_ACS](https://www2.census.gov/programs-surveys/demo/tables/health-insurance/time-series/acs/hic05_acs.xlsx), [HIC-6_ACS](https://www2.census.gov/programs-surveys/demo/tables/health-insurance/time-series/acs/hic06_acs.xlsx) | recomputed from the United States rows on 13 Sep 2026; all 48 rows of the agent's CSV match to 0.1 point; 2020 columns are "N" | PAGE (files downloaded and recomputed) |
| The Census data API now requires a registered key | T1 | `api.census.gov` | an unkeyed query was redirected to `missing_key.html` with `X-DataWebAPI-KeyError: 1` (13 Sep 2026) | RUN |
| ACS table S2701 changed its age groups: three uninsured age groups in 2013; 18–24 and 25–34 through 2016; 19–25 and 26–34 from 2017 | T1 | Census API variable lists, `api.census.gov/data/<year>/acs/acs1/subject/variables.json` | as stated | PAGE |
| AEA journals: AI may not be an author; AI use in preparing the manuscript must be described at submission | T5d, T2b | [AER submission guidelines](https://www.aeaweb.org/journals/aer/submissions); [AEA journal policies](https://www.aeaweb.org/journals/policies) | "Artificial intelligence (AI) software … may not be listed as an author"; AI used in preparing the manuscript, including drafting or editing, must be briefly described at submission | PAGE |
| Economist-authored skill collections, both MIT-licensed | T3a, TA | [pedrohcgs/claude-code-my-workflow](https://github.com/pedrohcgs/claude-code-my-workflow); [chrisblattman/claudeblattman](https://github.com/chrisblattman/claudeblattman) | repository descriptions and licence files | PAGE |
| The clean-up prompt "I want your help cleaning up this project and setting up a git repo" | T5d, TA | [CEPR/VoxDev webinar slides, "AI Agents for Economics Research"](https://voxdev.org/sites/default/files/2026-03/AI_Agents_for_Economic_Research.pdf) (Panjwani) | the prompt, as an exercise for economists' messy projects | SEARCH |
| Cluster where treatment is assigned | T5c | Abadie, Athey, Imbens, and Wooldridge (2023), "When Should You Adjust Standard Errors for Clustering?", *QJE* 138(1) | — | BIB |
| Blinder and Watson (2016), cited by the recorded agent | T2a | "Presidents and the U.S. Economy: An Econometric Exploration", *AER* 106(4) | — | BIB |

## Measuring agents

| Claim | Deck | Source | What the source says | How |
|---|---|---|---|---|
| 50%-reliability time horizon doubles about every 7 months since 2019; about 4 months from 2023; about 3 months from 2024 | T1 (TA keeps the model-by-model minutes) | [METR, Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) (29 Jan 2026) | all-time doubling 196 days; post-2023 131 days; post-2024 89 days (TH1.1) | PAGE |
| τ-bench: gpt-4o solves under half the tasks; pass^8 under 25% in retail | T3b (T4 cites the definition) | [Yao et al. 2024, arXiv:2406.12045](https://arxiv.org/abs/2406.12045) | "succeed on <50% of the tasks, and are quite inconsistent (pass^8 <25% in retail)" | PAGE |
| OpenAI stopped reporting SWE-bench Verified on 23 Feb 2026; 138 hard tasks audited, 59% with flawed tests; recommends SWE-bench Pro | T5d, TA | [OpenAI, "Why SWE-bench Verified no longer measures frontier coding capabilities"](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | 23 Feb 2026; 59.4% of 138 audited tasks flawed | SEARCH |
| Berkeley RDI scored near 100% on eight agent benchmarks without solving a task (Apr 2026) | T5d, TA | [Berkeley RDI, "How We Broke Top AI Agent Benchmarks"](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/); [R&D World](https://www.rdworldonline.com/how-a-berkeley-team-broke-8-major-ai-benchmarks-six-of-them-hit-100-without-solving-a-single-task/) | eight benchmarks, among them SWE-bench Verified and Pro, Terminal-Bench, WebArena; most runs never called a model | SEARCH |
| CORE-Bench: computational reproducibility of published papers | T5d, TA | [Siegel et al. 2024, arXiv:2409.11363](https://arxiv.org/abs/2409.11363) | 270 tasks from 90 papers in computer science, social science, and medicine | SEARCH |
| Anthropic's multi-agent research system beat a single agent by 90.2% on its own eval ([hype]) | T1 | [Anthropic engineering, 13 Jun 2025](https://www.anthropic.com/engineering/multi-agent-research-system) | "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval" | PAGE |

## Security

| Claim | Deck | Source | What the source says | How |
|---|---|---|---|---|
| EchoLeak, CVE-2025-32711: zero-click data exfiltration from Microsoft 365 Copilot | T5d | [arXiv:2509.10540](https://arxiv.org/abs/2509.10540); [Checkmarx](https://checkmarx.com/zero-post/echoleak-cve-2025-32711-show-us-that-ai-security-is-challenging/) | a crafted email, no user interaction; found by Aim Labs | SEARCH |
| ShadowPrompt: zero-click prompt injection in Claude's Chrome extension | T5d | [The Hacker News, Mar 2026](https://thehackernews.com/2026/03/claude-extension-flaw-enabled-zero.html) | a permissive `*.claude.ai` origin allowlist plus an XSS on a subdomain; patched in extension 1.0.41 | SEARCH |
| GTG-1002: the first reported largely AI-orchestrated cyberattack, Nov 2025 | T5d | [Anthropic report](https://www-cdn.anthropic.com/d7dd50dd1185f59be051b307150d877f2b82bd2c.pdf); [MITRE ATT&CK C0062](https://attack.mitre.org/campaigns/C0062/) | announced Nov 2025; most intrusion work done by Claude Code with little human involvement | SEARCH |
| Agent-hijacking success rose from 11% to 81% with new attacks | T5d | [NIST technical blog, Jan 2025](https://www.nist.gov/news-events/news/2025/01/technical-blog-strengthening-ai-agent-hijacking-evaluations) (then the US AI Safety Institute, now CAISI) | 81% against baseline defenses, up from 11% | SEARCH |
| CORE-Bench: ~60% (easy) and ~21% (hard) | T5d | [Siegel et al. 2024, HTML v2](https://arxiv.org/html/2409.11363v2) | CORE-Agent with GPT-4o: 60.00% easy, 57.78% medium, 21.48% hard | PAGE |
| Adaptive attacks still succeed more than ~85% of the time | T5d | [Nasr et al. 2025, "The Attacker Moves Second", arXiv:2510.09023](https://arxiv.org/abs/2510.09023) | 12 defenses bypassed, attack success above 90% for most | SEARCH |

## Landscape and timeline (re-check)

| Claim | Deck | Source | What the source says | How |
|---|---|---|---|---|
| OpenAI function calling, 13 Jun 2023 | T1, TA | [OpenAI, "Function calling and other API updates"](https://openai.com/index/function-calling-and-other-api-updates/) | functions described to gpt-4-0613 and gpt-3.5-turbo-0613; the model returns JSON arguments | SEARCH |
| Google and Microsoft tooling support MCP | TA | [TechCrunch, 9 Apr 2025](https://techcrunch.com/2025/04/09/google-says-itll-embrace-anthropics-standard-for-connecting-ai-models-to-data/) (Gemini models and SDK); [Microsoft Copilot blog](https://www.microsoft.com/en-us/microsoft-copilot/blog/copilot-studio/introducing-model-context-protocol-mcp-in-copilot-studio-simplified-integration-with-ai-apps-and-agents/) (Copilot Studio) | Hassabis: Gemini will support MCP (Apr 2025); MCP in Copilot Studio, later generally available | SEARCH |
| MCP: Anthropic, Nov 2024; OpenAI support from March 2025; Linux Foundation AAIF, 9 Dec 2025 | T1, TA | [Linux Foundation press release](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation); [TechCrunch, 9 Dec 2025](https://techcrunch.com/2025/12/09/openai-anthropic-and-block-join-new-linux-foundation-effort-to-standardize-the-ai-agent-era/); OpenAI's MCP support reported 26 Mar 2025 (Agents SDK first) | AAIF formed 9 Dec 2025 with MCP, goose, and AGENTS.md | SEARCH |
| OWASP Top 10 for Agentic Applications, 9 Dec 2025 | T5d, TA | [OWASP GenAI Security Project](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/) | released 9 Dec 2025 | SEARCH |
| Claude Code routines, 14 Apr 2026: schedule, API call, or GitHub webhook; run on Anthropic's cloud | TA (T1 dates the rung) | [Anthropic, "Introducing routines in Claude Code"](https://claude.com/blog/introducing-routines-in-claude-code) | 14 Apr 2026; hourly/nightly/weekly schedules, per-routine API endpoint, GitHub events; "nothing depends on your laptop being open" | PAGE |
| Code with Claude, 6 May 2026: auto mode, worktrees, routines shown; Managed Agents demoed | TA | [InfoQ, 18 May 2026](https://www.infoq.com/news/2026/05/code-with-claude/) | event 6 May 2026, San Francisco | PAGE (press) |
| Codex Goal Mode generally available, May 2026 | TA (T1 dates the rung) | e.g. [DEV Community](https://dev.to/akaranjkar08/openai-codex-goal-mode-is-now-ga-multi-hour-autonomous-coding-sessions-oko); [AI Weekly](https://aiweekly.co/alerts/openai-codex-launches-appshots-and-stable-goal-mode) | GA 21 May 2026, on by default | PRESS — **re-check** against OpenAI's changelog |
| Navier–Stokes run: project began 1 Sep 2026; ~10,000 concurrent agents; ~88 h to a proof, 17 h more to formalise in Lean; ~2.7M messages and ~130B output tokens (campaign: ~4.9M, ~300B); Euler warm-up ~100 agents, ~50 h; a forced blow-up in R³ and T³; announced 8 Sep | T3b, TA | [VentureBeat, 8 Sep 2026](https://venturebeat.com/technology/openai-solves-longstanding-math-problem-with-10-000-agent-swarm-but-cant-rule-out-benefitting-from-a-researchers-private-codex-data); [The Next Web, 8 Sep 2026](https://thenextweb.com/news/openai-navier-stokes-claim-verification-credit); [AiCybr, 12 Sep 2026](https://aicybr.com/blog/openai-navier-stokes-ai-proof-10000-agents-lean). OpenAI's own page returned 403. | figures as listed; the proof was not yet published | PRESS — **re-check** |
| The credit dispute: Buckmaster (NYU) and Alpöge (Anthropic); OpenAI's response | T3b | VentureBeat, 8 Sep 2026 (above) | Buckmaster: "I do not know whether our data was used. I am not accusing anyone of anything." OpenAI (Mark Chen): "No people or AI systems searched through user data to solve this problem"; OpenAI: "we cannot rule out that de-identified data derived from their usage of our products helped improve our models." | PRESS — **re-check** |

## Corrections this check made (12 Sep 2026)

- **T3b sidebar:** start date 28 Aug → 1 Sep 2026. Traffic ~5M messages / ~300B tokens were the whole campaign; the Navier–Stokes run was ~2.7M / ~130B. The headline no longer says the proof was Lean-checked within the 88 hours. The blow-up is stated as forced.
- **T3b dispute:** "muscled in", the "AI slop" quote, and "OpenAI denies the charge" are not in the report checked; replaced by the quoted positions above.
- **T3b verifier frame:** "τ-bench ~90% at k=1 → ~57% at k=8" is not in Yao et al. 2024; replaced by the paper's own figures.
- **T1:** METR doubling times updated to Time Horizon 1.1. MCP multi-vendor dated to March 2025. Unattended runs dated to April–May 2026.
- **T2b:** "AEA disclosure (Oct 2024)" had no source; now points to T5d's quoted AER rule.
- **TA, checked and left as is:** Google and Microsoft MCP support, and OpenAI function calling (June 2023), both hold. METR's per-model minutes (Opus 4.5 ≈ 320, GPT-5 ≈ 214) were not re-checked; the row already says *stale — refresh*.
- **T5d, checked and left as is:** "eight" benchmarks (Berkeley RDI) is right; the July deck and `Lec09_讲稿.md:150` say four.
- **Not fixed here (other lectures):** the same unsourced "AEA … October 2024" claim is in `source/Lec07_LLM/Lec07_T5b_Applications_Validation.tex:665` and `source/Lec08_RAG/Lec08_T5_Research_Applications_Demo_Eval.tex:444`. `Lec09_讲稿.md:120` still dates OpenAI's MCP adoption to DevDay, 6 Oct 2025.
