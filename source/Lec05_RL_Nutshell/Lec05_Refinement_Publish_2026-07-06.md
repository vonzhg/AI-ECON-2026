# Lecture 5 (RL in a Nutshell) — refinement + publish record (2026-07-06)

Refined the three Lec05 decks to match the **Lec09 "one map, N acts"** house style, built a new
hands-on RL lab, and published everything to the course website. Model: Lec09_Agentic_AI.

## 1. Slides — structural refinement (content preserved; scaffolding added)

New shared file **`lec05_map.tex`** defines `\LecFiveMap{k}` — a recurring "one backbone, three
acts" map, the analogue of Lec09's `\LecNineMap`:

> **5.1 MODEL → 5.2 SAMPLE → 5.3 SCALE** — the lecture's actual arc (DP needs a full model →
> drop the model and sample → add function approximation to scale). Backup-taxonomy band
> (full → sample → sampled+approximated) + upstream/downstream course-flow satellites
> (Lec02/Lec04 → this → Lec06). `k∈{1,2,3}` = "you are here"; `k=0` = all-acts-complete reprise.

Per deck (`\input lec05_map.tex`, `\metroset{sectionpage=none}`, tikz libs added):

| Deck | Changes |
|---|---|
| **T1** RL Framework & DP | title → "The RL Framework and Its Link to Dynamic Programming"; plain agenda → **`\LecFiveMap{1}` map frame**; 4 `\sectiondivider`s; new **"Bridge: From a Model You Specify to Experience You Sample"** closer with arc line. 24 numbered frames. |
| **T2** MC/TD/Q-learning | agenda → **`\LecFiveMap{2}`**; 4 dividers; TD(λ) "Next" box → bias-variance takeaway; new **"Bridge: From Tables to Function Approximation"** closer. 30 frames. |
| **T3** Actor-Critic & Apps | agenda → **`\LecFiveMap{3}`**; 6 dividers; new **reprise map `\LecFiveMap{0}`** ("The Whole Lecture in One Map"); reordered so the deck **ends on the Lec06 bridge** (arc line + Session-4 lab pointer). 32 frames. |

All three compile clean (pdflatex ×2, no errors, no overfull); footers 24/30/32 (no `/100` bug);
`\framesubtitle` absent (house rule). Rebuild: `bash build_all_topic_decks.sh`.

## 2. New lab — `Notebooks/Lab4A_RL_Nutshell.ipynb` (24 cells, offline-safe)

The RL algorithms were **slide-only** before this (no hands-on RL notebook existed). New lab mirrors
the three-act map and validates every learner against a known benchmark:

- **MODEL** — 3×3 macro gridworld (from T2) + Value Iteration (exact optimum, full backup).
- **SAMPLE** — first-visit MC control + Q-learning (both recover the optimum, **greedy-policy value
  gap = 0.0001**, no model); ε-greedy vs Boltzmann reproduces the T2 slide table (0.62/0.37/0.01).
- **SCALE** — mini-DQN (experience replay + target net; recovers π* with a 64×64 net) and an
  actor–critic on a log-utility consumption–savings problem (**learned φ ≈ 0.22 vs closed-form
  φ\*=1−β=0.20**, flat across wealth).

Pure NumPy/matplotlib/PyTorch, `torch.set_num_threads(1)`, fixed seeds, **no internet / no Colab**
(Tsinghua-mirror install note included). Verified end-to-end with `jupyter nbconvert --execute`
(exit 0). Ships **without embedded outputs**.

## 3. Publish — course website (AI-ECON-2026, nested repo)

Current convention (changed 2026-07-06): **open access + diagonal copyright watermark** (no more
AES encryption). Watermarked the 3 source PDFs (`pikepdf add_overlay` of `watermark.pdf`; opens
w/o password; ~30–40 KB larger than source; "©Zhigang Feng" on every page) → `slides/`.

Edits: `slides/index.html` (new "Lecture 5 — Available" section; removed L5 coming-soon row;
heading "5–8 & 10" → "6–8 & 10"); `index.html` (L5 card badge → Available + Lab link);
`labs/index.html` (new "Session 4 — RL in a Nutshell (Lecture 5) — Available" row; KS row → Lecture
6); `syllabus.html` (split the Session-4 lab into **Lab 4A** = RL/Lec5 available, **Lab 4B** =
KS1998 DEQN/Lec6). Copied the lab to `AI-ECON-2026/labs/`. Committed + pushed to `main`.

## Follow-ups (not done here)

- ⚠️ **`Lec05_讲稿.md` is now desynced** by the map frames / dividers / bridges / retitled T1 —
  regenerate **before the Session-4 lecture (Thu Jul 9)**. Same situation as Lec09's 讲稿.
- The **FullCourse parent repo** (`AI-ML-2026`) source changes (Lec05 `.tex`, `lec05_map.tex`, new
  lab, this doc) are **left uncommitted** — the parent has an unrelated in-progress working tree; the
  user manages that repo.
- **Lab 4B (KS1998 DEQN, Lecture 6)** is still coming-soon on the site; physical file is
  `Notebooks/Lab11A_KS1998_Pytorch.ipynb` (not yet renamed to the `Lab4B_…` session name).
