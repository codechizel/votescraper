# LOO-CV Observation Mismatch in Model Comparison

## Summary

When Phase 08 (PPC) compares IRT models via Leave-One-Out Cross-Validation, the models being compared must have identical observation sets. If they don't, `arviz.compare()` raises a `ValueError`. This document explains why mismatches occur and how the pipeline handles them.

## Background: What LOO-CV Does

LOO-CV (Vehtari et al. 2017) evaluates predictive accuracy by estimating how well each model predicts held-out observations. For each observed vote $i$, it computes the log predictive density $\log p(y_i | y_{-i})$ — how likely the vote is given all other votes. The Expected Log Pointwise Predictive Density (ELPD) summarizes these across all observations:

$$\text{ELPD} = \sum_{i=1}^{N} \log p(y_i \mid y_{-i})$$

To compare models, `arviz.compare()` builds an $N \times M$ matrix (N observations, M models) of per-observation ELPD values. This requires every model to have the same N observations — otherwise the matrix can't be constructed.

## Why Observation Counts Differ

Phase 08 loads fitted models from three upstream phases:

| Phase | Model | Filtering |
|-------|-------|-----------|
| 05 (IRT) | 1D Bayesian IRT | Standard: minority < 2.5%, legislators < 20 votes |
| 06 (IRT 2D) | 2-dimensional IRT | Same thresholds, but may drop additional items for convergence |
| 07 (Hierarchical) | Hierarchical IRT with party priors | Party-aware: may exclude Independents or small-party legislators |

All three phases apply the same baseline filters (defined in `analysis/design/irt.md`), but edge cases cause divergence:

1. **Legislator exclusion cascades.** If the hierarchical model drops an Independent legislator (too few members for party-level priors), all of that legislator's votes disappear from the observation set. The 1D model keeps them.

2. **Convergence-driven item drops.** The 2D model occasionally drops roll calls where both dimensions are unidentifiable (e.g., a vote that splits perfectly on one axis but is random on the other). The 1D model has no such constraint.

3. **Missing data handling.** Each model independently pivots the vote matrix and drops NaN entries. Floating-point edge cases in the pivot (e.g., a legislator who voted on a roll call in one model's filtered set but not another's) can cause off-by-one differences.

4. **Session-specific triggers.** These mismatches are rare — most bienniums have identical observation sets across all three models. They tend to appear in sessions with unusual characteristics: very few Independents, small chambers (Senate), or sessions with many near-threshold votes.

## How the Pipeline Handles It

When `arviz.compare()` raises `ValueError("The number of observations should be the same across all models")`, the pipeline:

1. **Skips the cross-model comparison table** — no ELPD difference ranking is produced.
2. **Retains individual model diagnostics** — each model's ELPD, p_loo, and Pareto k distribution are still computed and reported.
3. **Logs a message** — `(Skipped cross-model comparison — observation counts differ)` appears in the pipeline output.

The individual LOO diagnostics are the primary diagnostic value. The cross-model comparison table is a convenience for ranking models, but the ranking is typically stable: hierarchical IRT outperforms 1D, which outperforms 2D on most Kansas sessions (the legislature's dominant one-dimensional party structure makes the second dimension mostly noise).

## Affected Code

- `analysis/08_ppc/ppc_data.py` — `compare_models()`: catches `ValueError`, returns `None` for comparison
- `analysis/08_ppc/ppc.py` — `process_chamber()`: guards print and plot on `loo_comparison is not None`
- `analysis/08_ppc/ppc_report.py` — `_add_loo_comparison()`: comparison plot guarded by `path.exists()`

## When This Matters

For most analytical purposes, the individual model ELPD values are sufficient. The cross-model comparison adds value only when:

- You need to formally rank models for a specific session
- You're validating that a new model specification improves on the baseline

If the comparison is critical for a specific session, the fix is to ensure identical filtering across all three upstream phases. This can be done by passing a shared filtering manifest, but adds coupling between otherwise independent phases — a tradeoff that hasn't been worth making for the Kansas pipeline.

## References

- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5), 1413-1432.
- ADR-0063: PPC and LOO-CV model comparison design
- `analysis/design/ppc.md`: Phase 08 design document
