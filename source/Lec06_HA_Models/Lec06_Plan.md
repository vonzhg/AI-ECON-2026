# Lecture 6 — Heterogeneous Agent Models via RL: Plan and Refinement Notes

## Status

| Field | Value |
|---|---|
| Built | 2026-04-09 |
| Frames in PDF | **110** |
| Frames in `.tex` source | 109 `\begin{frame}` + 1 title frame |
| Compiles cleanly | Yes (`pdflatex` x2, no errors, no missing figures) |
| Lab notebook | `Lab06_KS_HA.ipynb` (byte-for-byte copy of `Notebooks/Lab11A_KS1998_Pytorch.ipynb`, 15 cells) |

## Context

Lec06 sits in a 24-hour graduate course (10 lectures) on AI/ML for Macroeconomics, with materials under `2026_New_Slides/`. Lec01 and Lec02 fix the visual template; the remaining lectures are being written one at a time.

This lecture is **2 hours of teaching + 1 hour of lab**. It bridges Lec05 (RL in a nutshell, infinite-horizon optimal control) into the modern computational frontier:

- Aiyagari/Bewley/Huggett incomplete-markets models
- Krusell–Smith (1998) with aggregate uncertainty
- Deep-learning solution methods (DeepHAM, DEQN)
- OLG with aggregate uncertainty
- Mean-Field Games as the continuum-agent limit

The user explicitly said: **"as detailed as possible, we can always skip slides later"** — so this deck is intentionally dense (110 slides) and the section dividers below let the instructor skip past parts in real time.

## Source materials (verified, with line ranges)

| Source | Path | Key line ranges |
|---|---|---|
| Primary HA source (2619 lines) | `11_HA_model/Lec11_ML_HA_models.tex` | Aiyagari setup 113–203; SRCE 205–249; Existence/uniqueness 251–540; Computation 540–871; Why ML / simple example 873–1197; Euler+ML 1199–1336; Python code blocks 1340–1510; RL motivation 1512–1607; Global DNN solution 1609–1993; Krusell–Smith 1995–2173; DeepHAM 2180–2454; DEQN 2456–2616 |
| MFG source (1367 lines) | `15-MFG/MFG.tex` | Motivation 92–183; HJB+KFE coupling 188–243; HJB review 247–381; Poisson HJB derivation 386–497; Numerical / upwind 504–743; Continuous-time Aiyagari 749–895; KFE 900–1021; Numerical algo (A vs Aᵀ) 1029–1120; Transition dynamics 1124–1223; Master equation 1229–1296; Deep learning for MFG 1300–1331 |
| Lab notebook (15 cells, complete) | `Notebooks/Lab11A_KS1998_Pytorch.ipynb` | KS via DEQN in PyTorch; SSJ initialisation → DEQN training loop → analysis. Self-contained, runs end-to-end. |
| Template reference | `2026_New_Slides/Lec02_What_is_AI/Lec02_What_is_AI.tex` lines 1–113 | Preamble + agenda frame style |

### Figures used

| Figure | Folder | Used at frame | Concept |
|---|---|---|---|
| `equilibrium.png` | `11_HA_model/` | Frame 15 | $K(r)$ vs.\ $A(r)$ equilibrium picture |
| `optgrowth_sto.jpg` | `11_HA_model/` | Frame 34 | Optimal growth stationary distribution under trained policy |
| `app_dist.png` | `11_HA_model/` | Frame 48 | NN approximation of the cross-sectional distribution |
| `KSwithDL_architecture_price_new.pdf` | `11_HA_model/` | Frame 67 | DeepHAM computational graph |
| `KS_0fm1gm_v1_valuegmbasis_sub.pdf` | `11_HA_model/` | Frame 69 | Generalised moment basis functions |
| `deepham.png`, `ddp_training_metrics.png`, `scatter_policy_c.png`, `scatter_policy_a1.png` | `11_HA_model/` | Frame 70 (4-up) | DeepHAM output plots |
| `vp.png` | `15-MFG/` | Frame 100 | Upwind derivative approximation |

Resolved by `\graphicspath{{../../11_HA_model/}{../../15-MFG/}}`. Available but unused (kept for future expansion): `hyper_mpe_sto_closedform.jpg`, `smoyak_1.pdf`, `scatter_value.png`.

## Files in this folder

1. **`Lec06_HA_Models.tex`** — Beamer source (~86 KB).
2. **`Lec06_HA_Models.pdf`** — compiled output (~1.37 MB, 110 pages).
3. **`Lab06_KS_HA.ipynb`** — exact copy of `Notebooks/Lab11A_KS1998_Pytorch.ipynb`.
4. **`Lec06_Plan.md`** — this file.
5. LaTeX auxiliary files (`.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc`, `.vrb`).

## Preamble

Reuses the exact preamble from `Lec02_What_is_AI.tex` (lines 1–87) with three differences:
- Header comment → `%% Lecture 6: Heterogeneous Agent Models via Reinforcement Learning`.
- `\graphicspath{{../../11_HA_model/}{../../15-MFG/}}` (multi-folder).
- Title: `6: Heterogeneous Agent Models via Reinforcement Learning`.
- Added `language=Python` to `\lstset` for the 5 code blocks in Part IV.

## Slide deck outline (frame-by-frame, with source citations)

Bracketed `[Lxxxx]` references point at line numbers in `Lec11_ML_HA_models.tex`; `[MFG Lxxxx]` points at `MFG.tex`.

### Front matter (frames 1–2)
1. Title slide (`\makebeamertitle`)
2. Lecture Agenda — 10-item enumerate

### Part I — Aiyagari/Bewley/Huggett: setup & equilibrium (frames 3–19)
3. Why heterogeneous agents? Limits of representative agent
4. Aiyagari (1994): heterogeneity among households `[L113–129]`
5. General equilibrium with ex-post heterogeneity `[L133–147]`
6. Household's recursive problem (Bellman) `[L151–166]`
7. Budget constraint, borrowing limit, asset structure `[L170–188]`
8. Firms: CRS, factor prices `[L192–203]`
9. SRCE — definition (prices) `[L205–230]`
10. SRCE definition (HH problem, market clearing, stationary distribution) `[L231–249]`
11. Stationarity and the invariant measure `[L208–217]`
12. Two markets: capital and labour `[L255–270]`
13. Determination of $r$: demand for capital $K(r)$ `[L272–288]`
14. Determination of $r$: supply of capital $A(r)$ `[L289–306]`
15. Equilibrium picture (figure: `equilibrium.png` `[L483]`)
16. Probability measures and the Markov operator $Q$ `[L307–353]`
17. Feller property → existence `[L354–373]`
18. Mixing property → uniqueness (with counterexample) `[L375–437]`
19. Monotonicity → convergence (with counterexample) `[L439–487]`

### Part II — Computation of stationary equilibrium (frames 20–26)
20. Aggregate precautionary savings, applications `[L488–538]`
21. Computation of SRCE: 3-step overview `[L540–573]`
22. Step 1: solve household VFI for fixed $r$ `[L576–608]`
23. Step 2: stationary distribution from transition matrix `[L611–632]`
24. Step 3: market clearing on $r$ — bisection `[L635–671]`
25. Implementation details: grids, monotonicity, multigrid `[L673–700]`
26. Numerical challenges → curse of dimensionality

### Part III — Why ML? Motivating example on optimal growth (frames 27–35)
27. Curse of dimensionality and the case for function approximators `[L879–895]`
28. Approximating $V$ and $g$ with neural networks `[L900–922]`
29. One-step value update ($T_{\text{sim}}=1$) `[L927–952]`
30. One-step policy update `[L957–981]`
31. Choosing the sampling distribution $d(k)$ `[L983–1000]`
32. Bias–variance tradeoff `[L1005–1023]`
33. Stationary-distribution sampling `[L1028–1048]`
34. Illustration (figure: `optgrowth_sto.jpg`)
35. Recap: what RL/DL gives us beyond grids

### Part IV — Euler-equation + ML on optimal growth (frames 36–44)
36. Euler equation as optimality condition `[L1199–1227]`
37. From Euler residual to MSE loss `[L1230–1284]`
38. Minimising the Euler-residual loss `[L1285–1336]`
39. Code 1/5: ValueNet architecture `[L1340–1369]` *(fragile frame)*
40. Code 2/5: value update step (Bellman MSE) `[L1373–1406]` *(fragile)*
41. Code 3/5: policy improvement (gradient ascent) `[L1410–1440]` *(fragile)*
42. Code 4/5: training loop `[L1444–1468]` *(fragile)*
43. Code 5/5: visualisation `[L1472–1510]` *(fragile)*
44. Actor–critic perspective: this *is* RL `[L1515–1534]`

### Part V — Global solution via DNN with full distribution dynamics (frames 45–54)
45. Why (deep) RL helps: flexible approximation, sampling, hardware `[L1540–1607]`
46. Global solution: full distribution as state `[L1611–1640]`
47. HH problem with evolving distribution `[L1643–1679]`
48. Distribution approximation (figure: `app_dist.png`) `[L1722–1729]`
49. DNN representation of $V$ and policy `[L1730–1750]`
50. Update value function via simulation + supervised learning `[L1774–1808]`
51. Update policy with terminal $V_{\text{NN}}$ `[L1810–1829]`
52. Transition function $G$ as a matrix `[L1831–1843]`
53. Conditional / unconditional transition kernels `[L1847–1883]`
54. Discretised transition (i.i.d.\ shocks, monotone policy) `[L1888–1915]`

### Part VI — Krusell–Smith (1998) (frames 55–62)
55. KS model: aggregate TFP shocks, two-state Markov `[L1999–2014]`
56. Distributions, firms, factor prices with aggregate state `[L2018–2031]`
57. HH problem with full distribution as state — curse of dimensionality `[L2032–2058]`
58. State reduction via finite moments `[L2060–2082]`
59. Postulated log-linear law of motion `[L2074–2076]`
60. KS algorithm steps 1–2 `[L2085–2135]`
61. KS algorithm steps 3–5 `[L2140–2157]`
62. Limitations of moment approximation `[L2160–2173]`

### Part VII — DeepHAM (frames 63–70)
63. DeepHAM (Han, Yang \& E 2025): intro `[L2180–2203]`
64. DeepHAM algorithm outline `[L2207–2228]`
65. Value function learning via supervised regression `[L2232–2246]`
66. Policy iteration via fictitious play `[L2250–2276]`
67. Computational graph (figure: `KSwithDL_architecture_price_new.pdf`) `[L2280–2297]`
68. DL-based generalised moments `[L2301–2325]`
69. Interpretation of generalised moments (figure: `KS_0fm1gm_v1_valuegmbasis_sub.pdf`) `[L2329–2336]`
70. DeepHAM output plots — 4-up grid `[L2373–2454]`

### Part VIII — Deep Equilibrium Nets (DEQN) (frames 71–78)
71. DEQN (Azinovic et al.\ 2022): one network, no value function `[L2459–2473]`
72. DEQN policy network architecture `[L2477–2499]`
73. Euler equation as the loss `[L2503–2529]`
74. Cloud simulation: $N$ parallel economies `[L2533–2554]`
75. Distribution transport step `[L2558–2575]`
76. DEQN algorithm outline `[L2579–2597]`
77. DEQN vs DeepHAM comparison table `[L2601–2616]`
78. Recap: the modern HA toolbox

### Part IX — OLG with aggregate uncertainty (frames 79–83)
*New content: HA `.tex` source has no OLG section. Pulled from Lab12_OLG.ipynb context plus standard OLG references.*
79. From infinite-horizon to OLG: life-cycle structure
80. OLG with aggregate uncertainty: state vector by cohort
81. Why DL helps for OLG with aggregate shocks
82. Homotopy-stabilised training (Brumm–Kubler–Scheidegger 2023)
83. `Lab12_OLG.ipynb` preview

### Part X — Mean-Field Games (frames 84–106)
84. MFG: motivation `[MFG L92–105]`
85. Core idea: continuum + mean-field feedback `[MFG L107–164]`
86. Brief history (Lasry–Lions, Caines–Huang–Malhamé) `[MFG L167–183]`
87. Two-equation MFG system `[MFG L188–223]`
88. Why HJB is familiar, why KFE is new `[MFG L225–243]`
89. HJB review for deterministic optimal control `[MFG L249–287]`
90. From discrete-time Bellman to continuous-time HJB `[MFG L316–353]`
91. Stationary HJB and FOCs `[MFG L355–381]`
92. Poisson income shocks: state space `[MFG L392–414]`
93. Deriving the HJB with jump terms `[MFG L416–457]`
94. Economic intuition of the jump term (option value) `[MFG L460–467]`
95. Continuous-time Aiyagari: setup `[MFG L756–842]`
96. The aggregation loop `[MFG L882–895]`
97. Kolmogorov forward (Fokker–Planck) equation `[MFG L900–946]`
98. Reading the KFE: drift, jumps, boundary `[MFG L952–993]`
99. Stationary KFE `[MFG L1000–1021]`
100. Discretisation & upwind scheme (figure: `vp.png` `[MFG L546]`) `[MFG L518–567]`
101. HJB and KFE: $A$ vs $A^\top$ duality `[MFG L1031–1091]`
102. Putting it all together: outer fixed-point on $r$ `[MFG L1095–1120]`
103. Transition dynamics: forward–backward system `[MFG L1124–1223]`
104. Master equation: distribution as state `[MFG L1229–1296]`
105. Deep learning for MFG `[MFG L1300–1331]`
106. MFG ↔ HA models: the link `[MFG L1351–1363]`

### Part XI — Lab preview & wrap-up (frames 107–110)
107. Lab06 preview: KS via DEQN in PyTorch
108. Lab06 step-by-step (4 parts of the notebook)
109. Optional follow-up: `Lab12_OLG.ipynb`
110. Recap of Lec06 + bridge to Lec07 (LLMs)

## Lab notebook plan

Per user decision (2026-04-09), copy `Notebooks/Lab11A_KS1998_Pytorch.ipynb` byte-for-byte to `Lec06_HA_Models/Lab06_KS_HA.ipynb`. No edits to either copy.

```
cp "Notebooks/Lab11A_KS1998_Pytorch.ipynb" \
   "2026_New_Slides/Lec06_HA_Models/Lab06_KS_HA.ipynb"
```

The 15-cell notebook covers: SSJ steady-state initialisation → PyTorch DEQN policy network → cloud-simulation training loop → policy and loss visualisation. Self-contained, GPU-friendly.

## Verification (already executed)

1. **File presence:** `ls Lec06_HA_Models/` shows `Lec06_HA_Models.tex`, `Lec06_HA_Models.pdf`, `Lab06_KS_HA.ipynb`, `Lec06_Plan.md`. ✅
2. **LaTeX compile:** `pdflatex -interaction=nonstopmode Lec06_HA_Models.tex` ran twice with no errors and no missing-figure warnings. ✅
3. **Frame count:** PDF has 110 pages, matching the design target. ✅
4. **Notebook integrity:** `python3 -c "import json; nb=json.load(open('Lab06_KS_HA.ipynb')); print(len(nb['cells']))"` reports 15. ✅

## Build notes (for future refinement)

### Compile warnings (cosmetic only)
- One overfull `\vbox` (~27pt) at line 2015 of the source — happens on the dense MFG-system frame (frame 87). Not visually disruptive but could be tightened.
- Beamer's metropolis theme prefers XeLaTeX/LuaLaTeX; pdflatex compiles fine but substitutes default fonts (consistent with how Lec01 and Lec02 are also built).
- Two minor font-shape warnings (`TS1/aess/m/n`, `T1/aett/b/n`); silent fallback.

### Areas that are deliberately thin and could be expanded
- **Part IX (OLG)**: only 5 frames because the source `.tex` has no OLG section. If a deeper OLG treatment is needed, draw from `Notebooks/Lab12_OLG.ipynb` directly for state-vector definitions, age-profile labour endowment, and the homotopy schedule.
- **DeepHAM output figures (frame 70)**: currently a single 4-up grid. Could be split into 2 frames with more text annotation if students need more context on training metrics vs.\ policy plots.
- **Lab preview (frames 107–108)**: kept short. If the lab is run live in class, consider adding an "expected output" checkpoint frame between Parts 3 and 4 of the notebook walk-through.

### Areas that may be too dense
- **Part X (MFG, 23 frames)** is the longest part. Three subsections could be cut/merged if time-pressed:
  - Frames 92–94 (Poisson HJB derivation) — could compress to 1 frame with the result and a "see appendix" pointer.
  - Frame 103 (transition dynamics) — pure preview, can be skipped without loss of continuity.
  - Frame 104 (master equation) — advanced; appropriate to skip in a first pass.
- **Code blocks (frames 39–43)** — five separate `[fragile]` frames. If students don't need to see every line, the value/policy update steps (40 \& 41) could be combined.

### Things to add when refining
- Cross-references to Lec05 (RL in a nutshell) for the actor–critic and Bellman material.
- A frame near the end with **suggested readings**: Aiyagari (1994), Krusell–Smith (1998), Achdou et al.\ (2022 REStud), Han–Yang–E (2025), Azinovic–Gaegauf–Scheidegger (2022), Brumm–Kubler–Scheidegger (2023), Lasry–Lions (2007).
- A proper title-page subtitle if the user adds one to the other lectures.
- Consider replacing the `optgrowth_sto.jpg` (frame 34) with a cleaner figure if one becomes available.
- The `\language=Python` setting in `\lstset` is new — if other lecture decks adopt code blocks, hoist it up to a shared style file.

### Source-file gaps (worth investigating later)
- `Lec11_ML_HA_models.tex` covers Aiyagari and KS deeply but has **no OLG section**. If the user wants OLG to be a first-class topic, write a new `.tex` file specifically for it.
- `Lec11_ML_HA_models.tex` discusses DeepHAM and DEQN in parallel but the comparison table at lines 2601–2616 is the only side-by-side reference. Worth expanding into a full benchmark frame if/when more deep-HA methods (e.g.\ Maliar–Maliar) are added.

### Symbol-renaming opportunities for consistency
- Lecture currently uses both `g_i(a)` (KFE notation) and `g(k)` (policy in optimal growth) — distinct contexts but the symbol overlap could confuse students. Consider renaming the optimal-growth policy to `\sigma(k)` or `c(k)` throughout Parts III–V.
- `Z` is used for both aggregate TFP (KS, Part VI) and the income-state set (MFG, Part X). Acceptable in context but could be flagged.
