# Research-Grade Target — Algorithmic Recipe

These notes describe the **silent benchmark** the V0 → V5 ladder is climbing toward. They name *ingredients*, not a specific paper or codebase, so the ladder can be discussed without spoiling the destination.

The target is a **two-asset overlapping-generations economy with aggregate productivity risk** solved by neural-network policy approximation with a stabilising homotopy. The lineage runs through Deep Equilibrium Nets (Azinovic, Gaegauf, & Scheidegger, 2022) and the stabilising-homotopy refinement that followed it.

## Economic ingredients

- **Several living cohorts** indexed by age (e.g. seven), with a fixed efficiency-labour profile that peaks in middle life.
- **Two assets**: physical capital with adjustment costs, and a one-period bond in zero net supply with an endogenous price.
- **Production**: Cobb-Douglas with TFP shocks; depreciation may depend on TFP through a damping mechanism.
- **Aggregate uncertainty**: a discrete or continuous AR(1) on log-TFP. The classroom ladder uses 2-state and 4-state Markov chains so expectations stay closed-form; production-grade variants use Gauss–Hermite quadrature.
- **Borrowing limit** on bonds enforced via a complementarity residual (Fischer–Burmeister) rather than a hard projection.

## Computational ingredients

- **Cloud method.** A swarm of parallel economies, advanced one period each iteration under the current policy, provides the empirical training distribution. The swarm replaces a classical equilibrium fixed point on a histogram.
- **Single MLP policy** mapping the aggregate state to per-cohort savings rates and the bond price. Outputs are sigmoid-rescaled or otherwise constrained so feasibility (positive consumption, non-negative capital) holds by construction.
- **Market-clearing layer.** The bond's aggregate position is forced to sum to zero by construction (e.g. by reading off one cohort's bond holding from the others), so bond-market clearing never has to be enforced as a separate residual.
- **Euler-residual loss.** Mean squared dimensionless residual $1 - \beta\,\mathbb{E}[(1+r')\,u'(c')]\,/\,u'(c)$ for each Euler equation, summed across cohorts and assets.
- **Adam** with mild exponential learning-rate decay and minibatch SGD over the cloud.

## Stabilising homotopy

The hardest part of solving the full model from a blank initialisation is that the bond price, the borrowing limit, and the adjustment cost all move at once. The standard trick is to **solve a sequence of nested easier problems**:

1. **Capital-only training.** Fix bonds at zero and the bond price at a placeholder; train on the capital Euler alone. The result is a one-asset OLG that the network can learn cleanly.
2. **Bond-price pretraining.** Activate a small loss term on the bond Euler with the bond *quantity* still suppressed, so the network discovers a sensible bond price.
3. **Bond-weight homotopy.** Slowly raise the weight on the bond Euler from near-zero to one, letting the network smoothly transition into the two-asset solution.
4. **Fine-tuning.** Train at the full loss with a smaller learning rate and (optionally) the cloud refreshed.

Each phase produces a checkpoint that is cheaper to verify than the final full solution. **Do not ask the network to solve the hardest model from a blank file — build, verify, and extend.** That is the slogan the demo wants students to internalise.

## Validation discipline at the target

A solution is considered "research-grade" when:

- All Euler residuals (capital and bond, all cohorts) have RMS magnitude under a few percent on the ergodic cloud.
- Bond positions follow the textbook lifecycle pattern — young cohorts borrow, old cohorts lend.
- The solution survives a stress test where some structural parameter is perturbed.
- The deterministic limit ($\sigma_z \to 0$) collapses the path of aggregates to a single point.

## Why this demo is smaller than the target

The demo's `versions/v5/` reaches the target's *structure* (cohorts, bonds, adjustment costs, homotopy) but not its *scale* (the production model uses richer TFP, more cohorts, GPU-accelerated array libraries, and longer training). The pedagogical claim is **algorithmic parity** — every algorithmic primitive a research-grade solver uses appears in V5 — not numerical equivalence with any specific reference codebase.

## Reference

- Azinovic, M., Gaegauf, L., & Scheidegger, S. (2022). Deep Equilibrium Nets. *International Economic Review* 63(4), 1471–1525.
