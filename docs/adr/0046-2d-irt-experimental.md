# ADR-0046: 2D IRT as Experimental Extension

**Date:** 2026-02-26
**Status:** Accepted (see also ADR-0054 for pipeline integration)

## Context

The 1D Bayesian IRT model is the canonical baseline for Tallgrass. It achieves 0.98 AUC, converges cleanly, and correlates r > 0.94 across bienniums. However, it compresses multidimensional voting behavior into a single axis.

The Tyson paradox (`analysis/design/tyson_paradox.md`) demonstrates the limitation: Sen. Caryn Tyson ranks 23rd by PCA but 1st by IRT because her contrarianism on routine legislation (PC2 = -24.8, most extreme by 3x) is invisible to a 1D model. Her 74 Nay votes on low-discrimination bills don't affect her IRT score, while her perfect 81/81 record on high-discrimination bills pushes her to the extreme.

A 2D IRT model would estimate both dimensions simultaneously — ideology (Dim 1) and contrarianism (Dim 2) — properly separating the two.

Six research agents investigated the Python ecosystem, identification strategies, political science literature, and integration feasibility. Key findings:

- **No existing Python package** provides 2D Bayesian IRT with proper identification. Custom PyMC is the only path.
- **PLT (Positive Lower Triangular)** constraints are the recommended identification strategy: beta[0,1] = 0 (rotation fix) + beta[1,1] > 0 (Dim 2 sign fix), reducing 8 equivalent modes to 1.
- **Political science literature** suggests the second dimension has largely vanished in modern polarized legislatures. For Kansas, PC2 captures contrarianism (11% variance), not a traditional policy cleavage.
- **Expected classification gain**: 0.5-2%. The 1D model already captures the high-discrimination signal.

## Decision

1. **2D IRT is implemented as an experimental script** (`analysis/experimental/irt_2d_experiment.py`), NOT a numbered pipeline phase. The 1D model remains the canonical baseline.

2. **PLT identification** is used to constrain the discrimination matrix:
   - `beta[0, 1] = 0`: fixes rotation (anchor item = highest 1D discrimination)
   - `beta[1, 1] > 0`: fixes Dim 2 sign (HalfNormal prior)
   - Dim 1 sign: post-hoc verification that Republican mean is positive

3. **2D PCA initialization** (PC1 + PC2, standardized) with `adapt_diag` (no jitter), following ADR-0045.

4. **Relaxed convergence thresholds**: R-hat < 1.05, ESS > 200, divergences < 50. The 2D posterior is harder to sample than 1D.

5. **Runs on a single chamber** (Senate, where Tyson is) for speed and focus.

## Consequences

**Benefits:**
- Resolves the Tyson paradox: separates ideology from contrarianism in a single Bayesian model
- Provides uncertainty intervals on both dimensions
- Identifies which bills discriminate on the second dimension
- Validates (or invalidates) PCA's PC2 interpretation via a proper statistical model

**Trade-offs:**
- 3-6x runtime vs 1D (~25-50 min vs ~8 min for Senate)
- PLT identification is more fragile than 1D anchor identification
- Most legislators (95%+) will have Dim 2 near zero with wide HDIs — the dimension is only informative for ~3-5 extreme legislators
- ~~Not a pipeline phase: no HTML report, no RunContext integration, no automatic downstream consumption~~ **Superseded by ADR-0054:** now integrated as pipeline phase 04b with RunContext, HTML report, both chambers, and nutpie sampling
- Convergence thresholds are relaxed; results should be treated as exploratory, not definitive
- Interactive Plotly plots (hover-to-identify) added for all three scatter plots. These serve as horseshoe diagnostics — the Dim 1 vs PC1 plot reveals PCA dimension confounding in supermajority chambers. See `docs/horseshoe-effect-and-solutions.md` for the full horseshoe effect analysis and experimental results.
