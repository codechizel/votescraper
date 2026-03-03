# ADR-0063: Standalone PPC + LOO-CV Phase (Phase 4c)

**Date:** 2026-02-28
**Status:** Accepted

## Context

The pipeline has three IRT model variants (flat 1D, 2D experimental, hierarchical) that all produce ideal point estimates. External validation confirms they agree with independent data (Shor-McCarty, DIME/CFscores, W-NOMINATE/OC). What was missing is **internal validation** (does each model reproduce its own data?) and **formal model comparison** (which model fits best after accounting for complexity?).

Phase 04 had a basic PPC (Yea rate + accuracy) but Phases 04b and 10 had none, and there was no cross-model comparison.

## Decision

Create Phase 4c as a standalone validation phase (like Phase 17 W-NOMINATE):

1. **Manual log-likelihood computation** (numpy) instead of rebuilding PyMC models. The Bernoulli log-likelihood is a trivial formula (`y * log(sigmoid(eta)) + (1-y) * log(1-sigmoid(eta))`), and avoiding model reconstruction eliminates anchor reconstruction, party index rebuilding, and PyTensor compilation (~30-60s per model).

2. **Standalone phase** (not in `just pipeline`). PPC + LOO-CV is expensive and validation-only — it doesn't feed downstream phases. Users run it explicitly with `just ppc`.

3. **Graceful degradation**. Missing models are skipped with warnings. Single-model runs skip `az.compare()`. `--skip-loo` and `--skip-q3` flags for faster runs.

4. **Yen's Q3 local dependence** included as the empirical answer to the 1D vs 2D dimensionality question. If 1D shows Q3 violations (|Q3| > 0.2) that 2D resolves, the second dimension is justified.

5. **Joint hierarchical model excluded**. Known convergence issues, different legislator ordering across chambers, low ROI for model comparison purposes.

## Consequences

- Completes the internal validation story: every model now has PPC coverage
- LOO-CV provides formal model ranking with stacking weights
- Q3 gives empirical dimensionality evidence (beyond "2D captures more variance")
- Phase 06 now saves InferenceData to NetCDF (prerequisite for PPC loading)
- No R subprocess needed — Python ecosystem (ArviZ) is sufficient
