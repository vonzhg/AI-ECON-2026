# Plan: Systematic rewrite of Lec09_Agentic_AI (5 decks) — "One map, five acts"

## Context

Lec09 (Agentic AI, 1-hour slot, Session 6 on Sun Jul 12 2026, followed the same morning by Lec10's 2-hour case studies + 1-hour lab) currently works but reads as an accumulation: install/init is exiled to an appendix, memory/compaction gets one frame, the researcher-as-PM idea is scattered, and the course's two best case studies (DeepRamsey, ES Fellows) barely appear. The user wants a systematic rewrite of the five core decks so students get a clear **map** — how agentic AI works, how to use it properly, and what the researcher's role is — with the two case studies woven through as running threads, and smooth logical flow deck-to-deck.

**Scope decisions (user-confirmed):**
- Rewrite ONLY the 5 listed decks; T1b_Agentic_Landscape + T6_Preview_Wrap stay as-is (bridge wording in the 5 decks must keep them true). After the decks, update the syllabus.
- Be comprehensive — flow and coverage over clock-fit (user adjusts live teaching). Target ≈102 frames total (current count), ≤30/deck.
- Compress T5's Context Dilemma to ~6 frames, merged into one "Managing Context" arc with the compaction trade-off.
- 讲稿 re-sync DEFERRED to a follow-up session (flag at the end; Lec09_讲稿.md will be desynced until then).

## Architecture: five acts, one recurring map

Each deck = one act answering one student question, with a recurring MAP slide ("you are here") replacing agenda frames:

| Act | Deck (filename unchanged) | Question | New display title (keywords preserved for T6 arc lines) |
|---|---|---|---|
| 1 MAP | Lec09_T1_Framing_Setup.tex | what is this & why care? | "Framing — What Agentic AI Is and Why It Matters" |
| 2 MACHINE | Lec09_T2_Under_the_Hood_FirstSession.tex | how does it work & how do I start? | "Under the Hood — How It Works and How to Start" |
| 3 METHOD | Lec09_T3_Homotopy_Tools.tex | how do I structure real work? | UNCHANGED: "The Homotopy Workflow and Tool Choice" |
| 4 MANAGEMENT | Lec09_T4_Project_Org.tex | how do I run a whole project? | "Project Organization — The Researcher as Project Manager" |
| 5 MINDSET | Lec09_T5_Tracing_Mistakes_Context.tex | what can go wrong & what is my job? | "Tracing, Mistakes, and Managing Context — Your Role as Researcher" |

**Two case threads** (Lec09 shows workflow ARTIFACTS; findings/figures stay in Lec10, taught the same morning — every thread frame carries a "full case: Lec10 T8/T4" pointer):
- **DeepRamsey** (credibility: proper use → fundamental discoveries): T1 teaser (weeks-not-years, 3 discoveries, 2 papers) → T3 opening prompts (restate/audit/refine) + seed criteria + three-layers-grow-together → T4 milestone ladder + division-of-labor contract + quality gates ("no compiler catches these") + orchestration (4 requirements, 32+ sweeps) → T5 "what changed vs what did NOT change".
- **ES Fellows** (hands-on, easy to grasp): T1 teaser (882 fellows, one batch is yours) + why-naive-chat-fails → T2 first-session walkthrough on the fellows roster + files-left-behind ("public-web analog of CLAUDE.md") → T3 V0=seed-roster → T4 provenance rules encoded as rules+hooks → T5 honest negative result (527 pages/316 CVs parsed, birth year still unknown 836/882 → redesign the variable, don't scrape harder).

**User's 8 topics → where they land:** (1) what is agentic AI → T1 F7–F10; (2) install & initialize → T2 F6/F8/F12 (durable steps in main flow; dated detail stays pointed at T6 App A); (3) CLI structure → T2 F9–F11; (4) file organization → T2 F7/F15 + T4 F8–F17; (5) clear goal + plan-validation criteria → T2 F13, T3 F4/F5/**F7 (dedicated "Grade the Plan" frame)**; (6) context window/compaction/trade-off → T2 F17–F18 (mechanics) + T5 F13–F19 (trade-off + dilemma); (7) workflow design → T3 entire; (8) researcher as PM → T4 F2–F5/F19/F20 + T5 F20–F21 close.

## Shared new file: lec09_map.tex

Create `Lec09_Agentic_AI/lec09_map.tex` defining `\LecNineMap{<1..5>}` (+ all-done variant for T5's closing reprise), `\input` after shared_preamble in each deck. (Filename deliberately does NOT match the build glob `Lec*_T*.tex`, so it is never compiled standalone.) Spec: row of 5 act boxes (9.1 MAP / 9.2 MACHINE / 9.3 METHOD / 9.4 MANAGEMENT / 9.5 MINDSET, each with its student question beneath); current act `fill=orange!20, draw=RedTitle, very thick` + "you are here" marker; others VeryLightGray. Dashed satellites: "9.1b landscape interlude" below the 9.1→9.2 arrow, "9.6 wrap + Lec10 cases" right of 9.5. Bottom band: three tinted underbars — HOW IT WORKS (acts 1–2), HOW TO USE IT WELL (acts 3–4), YOUR ROLE (act 5). pdflatex-safe; only existing tikzlibs (positioning, arrows.meta, shapes, calc); measure once for 16:9 and freeze.

## Per-deck frame outlines

Tags: [KEEP] existing ~as-is · [EDIT] reworked (source frame named) · [MOVE] relocated · [NEW] new (source material named). Counts exclude \makebeamertitle and \sectiondivider frames (house convention). All frames with listings get `[fragile]`. House voice: bold lead sentence, "Why this matters for economists:", `{\scriptsize Source: ...}` credits, NO \framesubtitle, tcolorboxes conceptbox/cautionbox/examplebox/takeawaybox.

### T1 — 16 frames (now 17). Sections: "Two Research Stories" / "From Chat to Agent" / "Your Safety Net"
1. [NEW] The Lecture in One Map — \LecNineMap{1}; three questions; one italic line: threads = DeepRamsey (Lec10 T8) + ES Fellows (Lec10 T4).
2. [EDIT: "Why Economists Should Care"] Execution Is the Bottleneck — keep 4 bullets; tie promise to the 3 map questions.
3. [NEW: DR F31/F39/M5–M7] Story 1 — DeepRamsey: Weeks, Not Years — examplebox: RA seed → full HA Ramsey code; 3 methodological discoveries (α-shapes, viability kernel, Lyapunov penalty) + 2 papers; "binding constraint was economist judgment." No findings/figures.
4. [NEW: ESF roster/F14] Story 2 — 882 Fellows: A Dataset You Could Build This Week — roster of 882, batches of 50, provenance per field; "hard part is bookkeeping, not intelligence"; Exercise B = your batch.
5. [EDIT: "What Modern Web Chat Already Does Well"] — replace "As of April 2026" → "as of mid-2026"; keep weak-baseline punchline.
6. [EDIT: "What Still Breaks Without a Project Harness" + ESF F4] What Breaks Without a Harness: The Fellows Test — 4 weaknesses instantiated on the fellows task.
7. [EDIT: "The Bridge: How a Chatbox Becomes a Terminal Agent"] Chat vs. Agent: Same Brain, Different Arms — keep table; DELETE "(e.g. Opus 4.5, Nov 2025)"; JSON reduced to one-line teaser → T2.
8. [EDIT: "What Makes a Workflow Agentic Under the Hood?"] The Agentic Loop — rework linear TikZ into circular goal→plan→act→observe→verify loop; keep orchestrator/memory/verifier bullets.
9. [KEEP] Four Features: Tool Use, Autonomy, Iteration, Persistence.
10. [EDIT: "Spectrum"] The Spectrum, and Our Featured Tool — the ONLY tool-comparison table in T1–T5 (advisor feedback); Claude Code featured, durable-lesson sentence.
11. [EDIT: "Four Research Patterns"] Four Research Patterns You Will Meet in Lec10 — keep table + Agentic-RAG footnote (Lec08 T4b backref).
12. [KEEP] What Remains Human — + one line: "T4 turns this into a job description: researcher as project manager."
13. [EDIT: "How Should Economists Evaluate..."] Five Tests for Any Agentic Workflow — keep criteria table.
14. [EDIT] Git Is the Safety Net — keep risk/benefit columns + data-vs-code rule.
15. [EDIT] Git in Three Minutes — keep listings + emergency brake.
16. [EDIT: "Looking Ahead"] Bridge: Widen the Lens, Then Open the Hood — T1b optional interlude (landscape/MCP) then T2; must read correctly even if T1b skipped.
CUT: agenda (→map); "How Agents Invoke Tools: JSON" (→T2 F3); "Early Demo: MEPS" (job done by story teasers); "Main Deck vs Appendix" (rule becomes footnote on T2 F6).

### T2 — 22 frames (now 20). Sections: "The Machine" / "Start Here" / "Your First Session" / "Budget and Discipline"
1. [NEW] Map: Act 2 — \LecNineMap{2}.
2. [EDIT] Three Components: Harness, LLM, Your Files — keep TikZ; durable-anatomy line.
3. [MOVE from T1 + EDIT] The Loop in Machine Language: JSON Tool Calls — Lec07 T3a callback; keep listing + loop + "every action is a next-token prediction."
4. [KEEP] Privacy Boundary: Local Execution ≠ Local Processing.
5. [EDIT] Terminal Vocabulary in 60 Seconds — keep 5-row table.
6. [NEW: distilled from T6 App A] Install Once: Three Durable Steps — runtime → agent CLI → verify & launch from project folder; no versions/prices; footnote = durable/dated rule + "exact commands: T6 Appendix A."
7. [EDIT] Files First: Global vs. Local — keep hierarchy table + AGENTS.md/GEMINI.md note. (Placed BEFORE CLI internals — advisor feedback.)
8. [EDIT] Initialization: What Happens When You Type `claude` — keep 5 steps + token-cost warning; link steps to F7's files.
9. [NEW: fresh] The CLI at a Glance: One Screen, Four Zones — TikZ terminal mock: transcript w/ [Read ...] traces, input box, status line, permission prompt. (Optional inset: figures/claude_code_terminal.png.)
10. [EDIT: "Key Modes"] Commands, Modes, Keys: The Control Surface — regroup table into modes / slash commands / interrupt keys; keep Shift+Tab, /effort, /cost, Esc, /compact, /clear.
11. [EDIT] Permissions: The Blast-Radius Dial — keep toolbox table + allowlist JSON + PATH warning.
12. [EDIT: "First Five Minutes", retargeted to ESF] First Session (1): Launch on a Real Project — `cd ~/research/es-fellows; claude`; prompt: parse fellows_page.html into seed CSV; annotated [Read ...] traces = F3's JSON. (Aiyagari stays the computational anchor in T3/T5.)
13. [EDIT: "Start Here"] First Session (2): Pen and Paper, Then Plan Mode — steps 0–2; keep "cheapest place to catch a wrong task" box. Topic-5 first landing.
14. [EDIT: "Continue"] First Session (3): Implement, Review, Iterate — steps 3–5 + loop TikZ.
15. [NEW: ESF F6] What the First Session Leaves Behind — seed CSV, batch file, CLAUDE.md note, one saved rule; "chat evaporates, files persist"; "public-web analog of CLAUDE.md"; tees T4.
16. [EDIT] Tokens and Cost on the Radar — keep; AEA disclosure line; Lec07 T3a pricing pointer.
17. [NEW: fresh; callbacks Lec07 T3b + Lec08 T1] The Context Window Is the Agent's Working Memory — stacked-bar TikZ of what fills it (system prompt, CLAUDE.md, rules, history, tool outputs); finite; mid-context sag in one line (NO reteach); long sessions degrade.
18. [NEW: split from T5 compaction frame] Compaction: What `/compact` Keeps and What It Drops — summarize-and-clear; decisions survive, reasoning often doesn't; /clear vs /compact; "state belongs in files, not in the conversation"; closing pointer to T5.
19. [EDIT: merge 5-pillar recap + research rendering] The 5-Pillar Spine, One More Time — one frame, two columns; attribution fixed: "introduced in Lec01, exercised throughout Lec06"; "pillars decide what should be built; the machine only decides how fast." Pillar 1/2/3-4 detail frames become one dense line each here.
20. [EDIT] Division of Labor: You Provide, the Agent Provides — keep table + "strong RA, but you are still the PI."
21. [KEEP] Validation Is Your Referee Checklist.
22. [EDIT] Bridge: The Machine Needs a Method — can install/launch/steer/pay; can't yet structure a week of research → T3.
CUT: agenda; Pillar 1, Pillar 2, Pillars 3–4 standalone frames (absorbed into F19).

### T3 — 19 frames (now 18). Sections unchanged: "The Homotopy Workflow" / "OLG Five-Step Demo" / "Tool Choice". Title UNCHANGED.
1. [NEW] Map: Act 3 — \LecNineMap{3}.
2. [EDIT: merge "The Homotopy Workflow" + "Why 'Homotopy'?"] The Homotopy Idea: Solve the Easy Version First — keep V0→V3 checkmark TikZ; Judd 1998 Ch. 5–7 origin in bottom conceptbox; one line: same discipline at paper scale = DeepRamsey.
3. [KEEP — HARD CONSTRAINT] The Six-Step Pipeline — boxes EXACTLY `1. Design / 2. Write Spec / 3. Choose Algorithm / 4. Build V0 / 5. Validate / 6. Extend` (Lec10_T8_Case6_DeepRamsey.tex line ~269 reproduces this verbatim as "Recap: The Homotopy Workflow (Lec09 T3)").
4. [EDIT] Step 1 — Design: State the Goal Like an Economist — keep Aiyagari anchor; add goal template: "compute X with method Y to answer Z; success looks like W."
5. [EDIT] Step 2 — The Spec Is the Contract — keep 4-part list; KEEP VERBATIM "ask the agent to restate the specification in its own words before it writes code" (Lec10_T8 line ~540 cites it as "the Lec09 best practice").
6. [NEW: DR F18] How DeepRamsey Opened: Restate, Audit, Refine — the 3 verbatim prompts in a listing; "the audit is the cheapest insurance in the whole project." Source credit.
7. [NEW: fresh — topic 5 anchor] Grade the Plan Before Any Code: Five Criteria — (i) restates your goal; (ii) steps bounded w/ named acceptance criterion; (iii) validation benchmark per step; (iv) stop/rollback points; (v) first step = smallest testable object. cautionbox: "a plan without acceptance criteria is a wish."
8. [EDIT] Step 3 — Algorithm and Pseudocode Catch Design Errors — keep phase listing.
9. [EDIT] Step 4 — V0: Simplest Version With a Benchmark You Understand — keep Aiyagari deterministic checks.
10. [NEW: DR F12] V0 at Research Scale: What Makes a Good Seed — works / documented / modular / points somewhere; "a good seed is to agentic research what a good instrument is to empirical work"; classroom V0 ≡ research seed at different scales.
11. [KEEP] V0 Across Four Tasks — table (Aiyagari/MEPS/review/fellows).
12. [EDIT] Steps 5–6 — Validate, Then Extend One Margin — keep table; add pointer: quality gates return as management principle in T4.
13. [NEW: DR F11] Three Layers Grow Together: Model, Algorithm, Code — double-arrow TikZ; discoveries flow backwards; AI keeps TeX/pseudocode/PyTorch describing the same economy.
14. [EDIT: merge "vs One-Shot" + why-it-works bullet] Why This Beats One-Shot Generation — 4 reasons + "30 min for V0 vs a day debugging broken V2."
15. [KEEP] OLG Demo: The Folder Is the Method — verified against demos/M4_olg_5step_demo/.
16. [KEEP] OLG Version Ladder V0→V5 — MUST match demo README (V0 3-cohort/1-asset/2-state ... V5 four-phase homotopy).
17. [KEEP] Demo Launch and Classroom Contract — commands verified.
18. [EDIT: merge "Choose the Tool" + "Workhorse"] Tool Choice per Pipeline Step — chat drafts steps 1–2, IDE for single-file edits, terminal agent for steps 4–6; drop "As of April 2026" dateline.
19. [EDIT] Bridge: One Ladder vs. the Whole Factory — one artifact vs many artifacts/sessions/reviewers → T4.
CUT: agenda; standalone why-name/why-works/workhorse frames (merged).

### T4 — 23 frames (now 22). Sections: "The Paradigm Shift" / "Project Organization" / "Your RA Team" / "Remote and HPC"
1. [NEW] Map: Act 4 — \LecNineMap{4}.
2. [NEW: fresh — topic 8 anchor] The Paradigm Shift: You Are the Project Manager — TikZ chain: Goal → Final Product → Intermediate Goods → Quality Gate per Good → Assignment (which agent/session) → Cross-Verification. "T3 disciplined one artifact; T4 runs the factory."
3. [NEW: DR F20/F13/F31] A Managed Project in One Slide: DeepRamsey's Milestones — M1 refined seed → M2 HA in TeX before code → M3 v1 → M4 full model → M5–M7 discoveries → paper Sec 7 + methods note; each milestone = validated, preserved artifact (intermediate goods, literally).
4. [NEW: DR F19] Write the Contract First: Who Owns What — division-of-labor table; "AI supplies throughput; the economist supplies judgment — stated before any code."
5. [NEW: DR F22/F25] Quality Gates Catch What No Compiler Catches — the 2 conceptual bugs (equation timing; transition direction); 3 compressed rows of v1-failure→v2-architecture; cautionbox: "the gate is where the economist earns authorship."
6. [EDIT: "Why Files on Disk Matter"] The PM's Instruments Live on Disk — files = delegation without re-explaining.
7. [EDIT: merge both "Why These Project Objects Exist" frames] Six Instruments, Six Pain Points — one 6-row table (CLAUDE.md/skills/agents/rules/verifiers/provenance logs).
8. [KEEP] Minimal Project Tree.
9. [EDIT] CLAUDE.md: The Standing Brief — keep listing; add fellows analog line.
10. [KEEP] Three Memories, Three Writers (CLAUDE.md / auto memory / course logs).
11. [KEEP] Skills, Agents, Rules: Three Different Jobs — TikZ + loading order.
12. [KEEP] Plain English vs. Encoded Skill — + "Capstone Track D2 = deliver exactly this."
13. [KEEP] Skill File Structure.
14. [EDIT] Skills Are Saved Playbooks — keep 4-row table.
15. [EDIT: merge "Rules, Hooks, Settings" + ESF F7/F8] Rules and Hooks: Encoding the Fellows Provenance Discipline — compressed table on top; below: `rules/provenance.md` ("every filled field carries a source URL; no inference from names, photos, or country") + a hook rejecting CSV rows missing source_url.
16. [KEEP] settings.json: Permissions and Hooks — listing + hook-type roster.
17. [EDIT] Three Lifetimes: Per-Event, Per-Session, Forever.
18. [EDIT: merge "Sub-Agents: Parallel Children" + "Agents as Specialist Reviewers"] Sub-Agents: Your Parallel RA Team — roles + separate contexts + wall-clock gain.
19. [EDIT: merge "When to Use" + "Subagents as Parallel Reviewers"] Assignment and Cross-Verification — good/bad-fit columns; "the producer is never the only reviewer"; blast-radius bullets; foreshadow "parallel is not independent → T5."
20. [NEW: DR F38] Orchestration at Research Scale — 4 requirements (seed, vision, judgment, orchestration); parallel sessions for formalization/implementation/audit/sweeps/write-up; 32+ config sweeps delegated; "frame 2's chain, instantiated on a real paper."
21. [EDIT: merge 2 HPC frames] Remote/HPC: Move the Agent to the Compute — keep Mac/login/compute TikZ + one-line rule.
22. [EDIT] Login vs. Compute Node: The Hard Boundary — slightly compressed.
23. [EDIT] Bridge: Management Assumes Trust — Verify It → T5.
CUT: agenda; merges as listed (HPC 4→2, sub-agent 4→2, why-objects 2→1).

### T5 — 22 frames (now 25). Sections: "Tracing" / "Mistakes, Limits, and Security" / "Managing Context" / "The Paradigm Shift, Completed"
1. [NEW] Map: Act 5 — \LecNineMap{5}.
2. [EDIT] The Anchor Returns: Aiyagari Through the Full Harness — + one line: it climbed T3's ladder inside T4's objects; now we audit.
3. [KEEP] Tracing (1): The File Flow.
4. [KEEP] Tracing (2): The Orchestrator Loop.
5. [KEEP] Tracing (3): Inside CLAUDE.md.
6. [KEEP] Tracing (4): Inside SKILL.md and the Guide — + "skeleton of a capstone D2 deliverable."
7. [EDIT] Three Recurring Operator Mistakes — mistake 3 (giant thread) points forward to Managing Context.
8. [KEEP] Where Agents Still Fail.
9. [KEEP] Failure Case 1: Clustering Standard Errors (Abadie–Athey–Imbens–Wooldridge 2023).
10. [KEEP] Failure Case 2: The CPI-1983 Definitional Break.
11. [NEW: ESF F9/F10/F13] Failure Case 3: When the Data Says No — 882 Fellows — 527 pages/316 CVs, birth year unknown 836/882; agent scaled the search; only the researcher could redesign the variable. examplebox; "full case: Lec10 T4."
12. [KEEP] Security: Red Lines and Workarounds.
13. [EDIT: compaction frame, discipline half — mechanics moved to T2 F18] Managing Context (1): The Compaction Trade-off — compaction frees the window BUT deletes history (corrections, rejected alternatives, caveats); keep progress.md listing; "write state to files before compacting."
14. [EDIT: merge "Why This Cautionary Section" + "Context as Implicit Bayesian Prior"] Managing Context (2): Your Context Is a Prior — keep prior/posterior TikZ + Xie et al. 2022; Lec07 T3a callback.
15. [EDIT] The Context Dilemma, Stated — "the same signal that improves the answer launders your priors into it."
16. [EDIT: merge Evidence-1 sycophancy + RLHF root cause] Evidence (1): Sycophancy Is Trained In — Sharma et al. 2023; Perez et al. 2023; Casper et al. 2023 / Wolf et al. 2023; "no 'do not flatter me' line removes it."
17. [EDIT: merge Evidence-2 self-correction + Evidence-3 debate] Evidence (2): Internal Review Has a Ceiling — Huang et al. 2024; Valmeekam et al. 2023; Du et al. 2023 (debate helps only when debaters sample differently).
18. [EDIT: "Subagent Reviewers Inherit the Same Prior" + monoculture bullet] Your RA Team Inherits Your Prior — callback to T4 F18–19; parallel executionally, not epistemically; closing bullet: field-scale monoculture (Kleinberg–Raghavan 2021; Bommasani et al. 2022).
19. [EDIT: merge Remedy + Operational Discipline + Bridge Back] Remedy: Outside Critics, Plus Operational Discipline — Korinek division of labor; five habits; takeawaybox: "human criticism is the scarce input — spend agent cycles making it ready."
20. [NEW: DR F37/F31] The Paradigm Shift, Completed: What Changed, What Did Not — months→days, executable documentation, cheap cross-literature imports, milestone as unit of progress; alertblock (cautionbox): never decided what was interesting / never caught wrong timing / never knew when a result wasn't credible; "weeks not years — binding constraint was economist judgment."
21. [NEW: fresh synthesis; MAP reprise all-done] Your Job Description: PI and Project Manager — three questions answered in one line each; set the goal, define the final product, gate every intermediate good, assign, cross-verify, remain the client of the outside view. takeawaybox = topic-8 capstone statement.
22. [EDIT] Bridge: T6 Wraps, Lec10 Shows the Real Thing — explicit hand-back of both threads to Lec10 T4/T8 + Exercise B.
CUT: agenda; monoculture standalone; evidence 3→2; remedy/discipline/bridge-back 3→1; compaction mechanics → T2.

Context Dilemma arc = T5 F14–F19 = 6 frames ✓.

## Hard integrity constraints (verified against sources)
- T3 F3 keeps the six box labels VERBATIM + the name "Homotopy Workflow" (Lec10_T8 recap frame reproduces them; also lines ~235/240/610 reference "Lec09 T3 homotopy").
- T3 F5 keeps the restate-the-spec sentence verbatim (cited by Lec10_T8 ~line 540).
- T5 keeps the name "Context Dilemma" + ICL-as-Bayesian framing (Lec07_T3a: "Lec09 returns to this as the context dilemma"); T2 F3 exists (Lec07_T3a "Lec09 preview" of JSON tool calls).
- Context-window frames cite Lec07_T3b / Lec08_T1, never reteach.
- T1 F11 keeps the four pattern names (T6 preview + Lec10 T1 use them) + Agentic-RAG footnote (Lec08_T4b bridge).
- New titles keep T6 arc-line keywords: "Framing" / "Under the Hood" / "Homotopy" / "Project Organization" / "Tracing...Context". T1 must still "frame the terminal-agent workflow" (T1b's opening line depends on it); MCP mentioned ONLY in T1b.
- T4 keeps skills/agents/rules vocabulary + validation gates + 5-pillar recap in T2 (capstone doc dependency; Track D2).
- T3 F15–F17 claims must match demos/M4_olg_5step_demo/ (already verified; don't invent new ladder rows).
- Dated strings purged: "As of April 2026" (T1, T3), "Opus 4.5, Nov 2025" (T1); max phrasing "as of mid-2026"; don't restate the ESF roster snapshot date — cite "official roster snapshot (Lec10 T4)".
- DR/ESF thread frames stay at artifact level (prompts, contracts, ladders, rules, coverage counts). NO findings figures/tables from Lec10.

## Implementation steps
0. Copy this plan to `Lec09_Agentic_AI/Lec09_Rewrite_Plan_2026-07-06.md` (project-folder plan copy, per user's standing workflow).
1. Create `lec09_map.tex` (\LecNineMap + done-variant); test-compile inside T1 first.
2. Rewrite decks in order T1 → T2 → T3 → T4 → T5 per the outlines above. Preserve per-deck skeleton (provenance header updated with `%% Rewritten 2026-07-06 ...` line, \input shared_preamble, \metroset{sectionpage=none}, graphicspath, tikzlibs, \title, \makebeamertitle). Compile each deck (pdflatex ×2) immediately after writing it; fix errors/overfull before moving on.
3. Fact-check pass on tool-mechanics claims in NEW/EDIT frames (init sequence, /compact behavior, plan mode, permission modes, hook events) via the claude-code-guide agent against current docs — keep wording durable; fix anything stale.
4. Full build of the 5 decks + `bash qa_page_numbers.sh` (Lec09 entries); confirm frame counts ≈ plan (16/22/19/23/22).
5. Visual QA: Read each PDF; spot-check the MAP frame in all 5 decks, every NEW TikZ frame (T2 F9/F17, T4 F2/F3, T5 F21), and every merged frame for overflow; fix and rebuild.
6. Consistency pass (grep): no "April 2026"/"Opus 4.5"; six-step labels intact; \framesubtitle absent; each thread's frames read coherently in sequence; bridge lines match T1b/T6 realities.
7. Update `Docs/syllabus_2026.html` Lec09 block: refresh the bullet list to the new emphases (what agentic AI is; under-the-hood + first session incl. install/init/CLI; homotopy workflow; researcher as project manager — goal → intermediate goods → quality gates → assignment → cross-verification; tracing, failure modes, managing context incl. compaction & the Context Dilemma; DeepRamsey + ES Fellows threads). Keep the 7-deck "Topic decks:" line and T1b/T6 bullets unchanged.
8. Update `Lec09_manifest.txt` to list all 7 PDFs (fixes pre-existing T1b omission).
9. Final report: per-deck frame counts, what moved where, threads summary, and follow-ups.

## Verification
- Each deck compiles clean under pdflatex ×2 (build script's engine default); zero errors, no missing-figure warnings; qa_page_numbers.sh passes.
- Read the built PDFs and visually verify: MAP renders identically across decks with correct highlighting; new TikZ frames fit 16:9; no text overflow on merged frames.
- Cross-reference greps: `grep -n "Design / 2" Lec10_T8*` boxes vs T3; `grep -rn "Lec09" Lec07_LLM Lec08_RAG Lec10_Case_Studies` sanity pass on inbound refs; `grep -n "framesubtitle\|April 2026\|Opus 4.5" Lec09_T*.tex` → empty.
- Coverage audit: each of the user's 8 topics readable at its planned frame(s); both case threads followable by reading only their frames.

## Follow-ups (out of scope, flag in final report)
- Lec09_讲稿.md re-sync (deferred by user; must happen before Jul 12 — script is title-locked to old frames).
- T1b/T6: untouched; optional later pass for tone alignment.
- Combined PDF (Lec09_Agentic_AI_combined.pdf) regeneration if the user wants a fresh concatenation.
- Pre-existing nit elsewhere: Lec10_T1 says "Three Research Patterns" vs four everywhere else (not ours to fix here).
