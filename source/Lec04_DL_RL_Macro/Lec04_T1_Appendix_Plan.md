# Lec04_T1 — Self-Contained Foundations Appendix (Solow → Bellman) with Jump/Return Buttons

## Context

Lecture 4 (deep learning for macro) is sometimes given to **undergraduates with no dynamic-macro background**. `Lec04_T1_Motivation_Growth.tex` currently opens the optimal growth model cold (frame "The Workhorse", line 37). Add an **appendix at the end of the deck** that builds the model from first principles — Solow → preferences → sequential problem → Markov structure → Bellman → SP↔FE equivalence (the four questions) → "infinite-dim to 'finite'-dim" → contraction mapping → Blackwell → Theorem of the Maximum & inherited properties — **comprehensive but overview-level** (statements + intuition + named sources, no proofs). A **button near the beginning jumps to the appendix**; **return buttons in the appendix jump back to the beginning**.

One file changes: `FullCourse_10Lecs/Lec04_DL_RL_Macro/Lec04_T1_Motivation_Growth.tex` (additive; main narrative untouched).

## Conventions this reuses (found in repo)

- **Navigation pattern** already established in `Lec10_Case_Studies/Lec10_T8_Case6_DeepRamsey_paper.tex:466,1234–1448`:
  - main frame: `\begin{frame}[label=mainX]{...}` + `\hyperlink{appY}{\beamergotobutton{...}}`
  - `\appendix` before the appendix block
  - each appendix frame: `\begin{frame}[label=appY]{Appendix Ak --- ...}` with first body line `\hfill\hyperlink{mainX}{\beamerreturnbutton{Return}}`
- **House style**: no `\framesubtitle` (silently dropped); lead lines start with a command (`\textit{\footnotesize ...}`), never a braced group; sources named on-slide (e.g. "Blackwell (1965), *Ann. Math. Stat.*" as in `Lec05_T1_RL_Framework_DP.tex:323`; "Stokey, Lucas \& Prescott (1989, Harvard UP)" as in `Lec06_T1_Aiyagari_Computation.tex:282`); `conceptbox`/`takeawaybox`/`cautionbox`; `\sectiondivider{..}{..}` (noframenumbering) for the visual break.
- **Notation** matched to this deck + siblings (`Lec01_Quant_Macro/Lec01_T1_Intro_DP.tex`, Lec05_T1): `V(k,z)`, policy `g`, `u(c)`, `f(k)=Ak^\alpha`, `\beta,\delta`, Markov kernel `P(z'|z)`, feasible set `\Gamma(k,z)`.
- **Build**: pdflatex ×2 (appendix is English; no CJK, no magic comment needed). **QA**: Ghostscript only (`pdftoppm`/`pdftotext`/`pdfinfo` not installed).

## Changes

### 1. Two entry points in the main deck

1. **"The Workhorse: The Optimal Growth Model"** (line 37) → `\begin{frame}[label=mainWorkhorse]{...}`, and a right-aligned button under the lead line, before `\begin{columns}`:
   `\hfill\hyperlink{appFoundations}{\beamergotobutton{New to dynamic macro? Appendix: Solow \(\to\) Bellman}}`
   (frame is full — reclaim the `\vspace{0.15cm}` if it overflows; fallback: put the button at the bottom of the right column).
2. **"Bellman Equation and VFI"** (line 434) → `[label=mainVFI]`, plus a button after the Contraction-Mapping bullet ("the theoretical foundation we build on"):
   `\hyperlink{appCMT}{\beamergotobutton{Full statements: CMT + Blackwell (Appendix)}}`

### 2. Appendix block — inserted after the "Bridge" frame, before `\end{document}`

```latex
%=============================================================================
\appendix
\sectiondivider{Appendix: Dynamic Macro Foundations}{From Solow to Bellman ---
  self-contained refresher, no dynamic macro assumed (statements and intuition, no proofs)}
```

Then **12 frames**. Every frame's first body line is `\hfill\hyperlink{mainWorkhorse}{\beamerreturnbutton{Return to lecture}}` (per user request, "return to the beginning"); `appCMT` additionally gets `\hyperlink{mainVFI}{\beamerreturnbutton{Back to VFI}}`.

| # | label | title | content |
|---|-------|-------|---------|
| A0 | `appFoundations` | Appendix Roadmap: From Solow to Bellman | who this is for + chain diagram of A1→A11 in three groups: BUILD THE MODEL / TWO FORMULATIONS / WHY IT ALL WORKS |
| A1 | `appSolow` | Where It All Starts: The Solow Growth Model | Solow (1956, QJE). Mechanics + steady-state diagram |
| A2 | `appRamsey` | From Solow to Optimal Growth: Make Saving a Choice | endogenize s via preferences; Ramsey–Cass–Koopmans |
| A3 | `appSeqProblem` | The Sequential Problem: Choosing an Entire Future | deterministic SP; what a solution is; why it's hard |
| A4 | `appMarkov` | Adding Uncertainty: Shocks and the Markov Structure | exogenous Markov shock + endogenous law of motion ⇒ state (k,z) |
| A5 | `appBellman` | The Recursive Formulation: The Bellman Equation | stochastic Bellman eq. + anatomy + Principle of Optimality |
| A6 | `appEquivalence` | Are the Two Problems the Same? The Four Questions | SP↔FE equivalence, SLP Ch. 4, questions only |
| A7 | `appFunction` | What Did We Gain? Infinite Sequences → One Function | the "quote–unquote finite-dimensional" slide; bridge to the lecture's theme |
| A8 | `appCMT` | Why Iteration Works: The Contraction Mapping Theorem | Banach (1922): existence + uniqueness + geometric VFI convergence |
| A9 | `appBlackwell` | Proving a Contraction the Easy Way: Blackwell | monotonicity + discounting; two-line check for the Bellman operator |
| A10 | `appBerge` | Properties That Make Computation Work | Berge; concavity; Benveniste–Scheinkman ⇒ Euler equation |
| A11 | `appSummary` | The Complete Logical Chain --- and Where to Learn More | summary table + reading pointers + prominent Return |

**Per-frame content spec** (equations to appear; each frame ends with the named sources):

- **A1 Solow**: columns — left: $y_t=f(k_t)=Ak_t^\alpha$; fixed saving rate $s$: $i_t=s\,y_t$; accumulation $k_{t+1}=(1-\delta)k_t+sAk_t^\alpha$; steady state $sAk^{*\alpha}=\delta k^*$; concavity ⇒ unique $k^*>0$, monotone convergence. Right: small pgfplots figure ($sAk^\alpha$ vs $\delta k$, marked $k^*$, convergence arrows). Takeaway: the one behavioral primitive is an *assumed constant* $s$ — nobody in the model ever chooses. Source: Solow (1956, *QJE*).
- **A2 Solow→Ramsey**: what Solow cannot answer (is $s$ too high/low? welfare? policy responses) ⇒ keep the technology block $(f,\delta)$, replace the rule-of-thumb saver with preferences $\mathbb{E}_0\sum_t\beta^t u(c_t)$, $u$ increasing & strictly concave (consumption smoothing), $\beta\in(0,1)$ (impatience); each period split resources between $c$ (utility today) and $k'$ (resources tomorrow). Hook forward: in the lecture's benchmark ($u=\log$, $\delta=1$) the *optimal* rule is exactly a constant saving rate $\alpha\beta$ — Solow's assumption emerges as the answer ($c^*(y)=(1-\alpha\beta)y$, the deck's ground truth). Planner ⇔ competitive equilibrium here (welfare theorems, one line). Sources: Ramsey (1928, *Econ. J.*); Cass (1965, *REStud*); Koopmans (1965).
- **A3 Sequential problem** (deterministic first): $\max_{\{c_t,k_{t+1}\}_{t=0}^\infty}\sum\beta^t u(c_t)$ s.t. $c_t+k_{t+1}=Ak_t^\alpha+(1-\delta)k_t$, nonneg., $k_0$ given. A "solution" = an entire infinite list $(k_1,k_2,\dots)$ chosen at $t=0$. Why hard: infinitely many unknowns; Lagrangian route gives infinitely many Euler equations $u'(c_t)=\beta u'(c_{t+1})[\,f'(k_{t+1})+1-\delta\,]$ + a transversality condition (name only; Euler returns in Topic 4.3).
- **A4 Markov structure**: add $y_t=z_tf(k_t)$; a plan must now be *history-contingent* $k_{t+1}(z_0,\dots,z_t)$ — the object explodes. Two rescues, one per state variable: **exogenous** — $z_t$ Markov: $\Pr(z_{t+1}\mid z_t,z_{t-1},\dots)=P(z_{t+1}\mid z_t)$ (e.g., AR(1) log-TFP / finite chain); **endogenous** — $k_{t+1}$ is fully pinned down by today's choice from $\Gamma(k_t,z_t)=[0,\,z_tf(k_t)+(1-\delta)k_t]$ (law of motion). conceptbox: a **state variable** = the minimal information making the future conditionally independent of the past; here the pair $(k,z)$.
- **A5 Bellman**: $V(k,z)=\max_{k'\in\Gamma(k,z)}\{\,u(z f(k)+(1-\delta)k-k')+\beta\sum_{z'}P(z'\mid z)V(k',z')\,\}$ with under-brace annotations (flow utility today / discount / expected continuation value). Intuition bullets: $V$ compresses the infinite future into one number per state; optimality becomes a one-step trade-off; solving for the *functions* $V,g$ solves every initial condition at once. Principle of Optimality quote (Bellman 1957) — reuse the vetted wording from `Lec01_T1_Intro_DP.tex:124-127`.
- **A6 Equivalence — the four questions** (exactly the user's four): (1) does the SP value $v^*$ satisfy the FE? (2) is a solution of the FE equal to $v^*$? (needs a boundedness condition $\lim_t\beta^t v(k_t)=0$ ruling out bubble solutions); (3) does an optimal SP plan attain the FE maximum date-by-date? (4) does a plan generated by the FE policy $g$ solve the SP? All YES under standard assumptions — Stokey, Lucas \& Prescott (1989, Harvard UP), Ch. 4 (deterministic), Ch. 9 (stochastic); also Acemoglu (2009), Ch. 6. Framing: proofs belong to first-year macro theory; *we mention the questions and take the equivalence as given*. (Verification step below decides whether to print "Thms. 4.2–4.5" or cite chapter-level only.)
- **A7 One function**: side-by-side SP vs FE table — unknown: infinite contingent sequence vs **one function** $V$ (or $g$) on the state space; optimality: infinitely many Euler eqs + TVC vs one functional equation; output: a single path vs a decision rule for *every* state. The catch (the "finite" in scare quotes): one unknown object, but a function is itself **infinite-dimensional** — exact solutions are rare (our log benchmark), so computation = choosing a *finite representation*: table on a grid (Section 2 / Lab 3A), polynomial coefficients, or **neural-network weights — this lecture**. Bridge line: this slide is why Lecture 4 exists.
- **A8 CMT**: Bellman operator $T$ ("feed in any candidate $V$, get an improved one"); solving FE = fixed point $V=TV$. Contraction definition; Banach (1922): complete metric space + contraction ⇒ unique fixed point; convergence from *any* starting guess; error bound $d(V_n,V^*)\le \frac{\beta^n}{1-\beta}d(V_1,V_0)$ ⇒ VFI is guaranteed to work, geometrically at rate $\beta$. Uniqueness = "the model's prediction" is well-posed (echoes Lec05_T1 line 331).
- **A9 Blackwell**: verifying the contraction inequality directly is awkward (bounding a max inside a sup). Blackwell (1965, *Ann. Math. Stat.*): on bounded functions with the sup norm, **monotonicity** ($V\le W\Rightarrow TV\le TW$) + **discounting** ($T(V+a)\le TV+\beta a$) ⇒ contraction. Bellman-operator check is two lines (better continuation can only help; a constant added tomorrow is worth $\beta a$ today). scriptsize caveat: $\log$ utility is unbounded — standard fixes (weighted norms / compact state space), see SLP §4.3–4.4.
- **A10 Berge & properties**: Theorem of the Maximum (Berge, 1963; French orig. 1959): continuous objective + continuous compact-valued $\Gamma$ ⇒ $V$ continuous, policy correspondence u.h.c. Add strict concavity of $u$ + convex feasible set ⇒ $V$ strictly concave, policy a single-valued continuous **function** $g$. Add interiority ⇒ $V$ differentiable with envelope condition $V'(k)=u'(c)[zf'(k)+1-\delta]$ (Benveniste \& Scheinkman 1979, *Econometrica*) ⇒ the Euler equation Topic 4.3 trains on. Small table: continuity → interpolation between grid points is legitimate; concavity → unique maximizer, monotone policies, safe hill-climbing; differentiability → FOC/Euler-residual methods (and autodiff).
- **A11 Summary + reading**: table (extends `Lec01_T1_Intro_DP.tex:229`): Markov state → recursion possible; SLP Ch. 4 four theorems → SP ≡ FE; Banach CMT → existence/uniqueness/convergence; Blackwell → easy verification; Berge + B–S → continuity/concavity/differentiability → computation sound. Reading: SLP (1989) Ch. 3–4, 9; Ljungqvist \& Sargent, *Recursive Macroeconomic Theory*, Ch. 3–5; Acemoglu (2009) Ch. 6, 16; QuantEcon DP lectures (Sargent \& Stachurski, quantecon.org — free, runnable, accessible from China). Closing line hands back to the lecture's question (what replaces the grid in high dimension?) + prominent Return button.

### 3. Frame numbering

`\appendix` alone keeps counting frames into `\inserttotalframenumber`, so the main deck's footer total would jump ~20→32. If `kpsewhich appendixnumberbeamer.sty` finds the package, `\usepackage{appendixnumberbeamer}` (after the `\input{../shared_preamble.tex}`) so the main deck reads `k/20` and the appendix restarts — nicer for undergrads gauging lecture length. If not installed, accept the inflated total (the DeepRamsey paper deck already lives with this).

## Implementation steps

0. Copy this plan to `FullCourse_10Lecs/Lec04_DL_RL_Macro/Lec04_T1_Appendix_Plan.md` (user reviews plans in place — house workflow).
1. Verify the SLP theorem numbering (WebSearch; expect Thms. 4.2–4.5 for the four SP↔FE results). If confirmed, print them on A6; otherwise cite "Ch. 4" only. Check `kpsewhich appendixnumberbeamer.sty`.
2. Edit `Lec04_T1_Motivation_Growth.tex`: two labels + two goto buttons; append `\appendix` + divider + 12 frames (~330 lines).
3. Build in `Lec04_DL_RL_Macro/`: `pdflatex -interaction=nonstopmode -halt-on-error` ×2 (seconds; hyperlink targets need the second pass).

## Verification

- `.log` clean: no `undefined` hyperref destinations, no missing refs; overfull boxes ≤ a few pt.
- Rasterize with Ghostscript (`gs -sDEVICE=png16m -r110 -dFirstPage=N -dLastPage=N`) and visually check: the entry frame (button visible, no overflow), the VFI frame, and all 12 appendix frames.
- `gs -sDEVICE=txtwrite` spot-checks: button labels present on entry frames; appendix titles present; footer totals sane (no `/100`, which flags a stale single-pass build).
- Link sanity: `\beamergotobutton`/`\beamerreturnbutton` + labels follow the exact DeepRamsey markup, which compiles under this same preamble/theme today.

## Out of scope / follow-ups (flagged, not done)

- `Lec04_讲稿.md` is already flagged desynced (map-convention rebuild); this appendix widens the gap — regenerate on request.
- Course-website republish of the refreshed `Lec04_T1` PDF — on request.
- No git commit unless asked.

---

## As-built notes (implementation 2026-07-07)

Implemented as planned, with three deviations/discoveries:

1. **SLP theorem numbers**: could not be verified against a primary source (web copies are image-PDFs; secondary notes use their own numbering), so slide A6 cites **"SLP (1989), Ch. 4, §4.1 --- four theorems, one per question"** instead of printing "Thms. 4.2--4.5". Section title §4.1 "The Principle of Optimality" was verified via the publisher's table of contents.
2. **Metropolis kills the footer after `\appendix`**: the theme's `\apptocmd{\appendix}{numbering=none, progressbar=none}` re-instates its own `footline[plain]` (with empty frame numbering), silently erasing the shared-preamble "k/Total" footer on every appendix page. Fix: re-assert the house footline template immediately after `\appendix` (done in the deck, with a comment). Note: `Lec10_T8_Case6_DeepRamsey_paper.tex` has the same latent issue --- its appendix pages show no page numbers.
3. **TikZ style name `step` collides** with the built-in `/tikz/step` grid key (compile error); the roadmap style is named `astep`.

`appendixnumberbeamer` (installed) gives main deck `2/20 ... 20/20` and appendix `1/12 ... 12/12`. Verified: clean build (only a pre-existing 3.4pt hbox in the untouched AlphaGo frame), 35 pages, 27 live GoTo link annotations (2 entry buttons + 11 clickable roadmap boxes + 12 + 1 + 1 return buttons), visual QA of 9 rasterized pages.
