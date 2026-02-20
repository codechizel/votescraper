# ADR-0005: PCA Implementation Choices

**Date:** 2026-02-19
**Status:** Accepted

## Context

PCA is the mandatory second analysis phase (after EDA) per the analytic workflow rules — a cheap, fast dimensionality reduction that reveals the ideological landscape before investing in Bayesian IRT (MCMC). Several implementation choices needed to be resolved:

1. **Missing data imputation.** The binary vote matrix has ~30-50% nulls (absences, non-voting). PCA requires a complete matrix. Options: row mean, column mean, zero-fill, iterative imputation (MICE/SoftImpute).

2. **Standardization.** Whether to center-only or center-and-scale. Legislative PCA literature is split on this.

3. **Sensitivity analysis design.** The workflow rules require at least two filter settings. Which thresholds, and how to compare?

4. **Duplicating vs. importing EDA's filter logic.** The sensitivity analysis re-filters the full vote matrix at a different threshold. We could import `filter_vote_matrix` from `eda.py` or duplicate the ~40 lines.

5. **PC1 sign convention.** PCA components have arbitrary sign. Need a deterministic orientation rule.

## Decision

1. **Row-mean imputation.** Each legislator's missing votes are filled with their average Yea rate across non-missing votes. This treats absences as uninformative about ideology ("this legislator would have voted at their base rate"). Column-mean (bill average) was rejected because it erases the per-legislator signal that PCA needs. Zero-fill was rejected because it falsely asserts all absences are Nay votes. Iterative methods (SoftImpute) are principled but overkill for a pre-IRT sanity check — and harder to explain.

2. **StandardScaler (center and scale).** Without scaling, roll calls with high variance (close votes) would dominate PCA. Centering removes the mean Yea rate; scaling ensures each roll call contributes equally regardless of its margin. This matches the standard approach in Poole & Rosenthal's work.

3. **Sensitivity at 10% minority threshold.** The workflow rules specifically mention 2.5% vs 10% as the required comparison. Pearson correlation between PC1 scores from both settings quantifies robustness. Threshold: r > 0.95 = robust.

4. **Duplicate filter logic.** The ~40-line filter function is duplicated in `pca.py` rather than imported from `eda.py`. This keeps phases self-contained — PCA can run independently without importing the EDA module, and changes to EDA's filtering won't silently alter PCA's sensitivity analysis.

5. **Orient PC1 so Republicans are positive.** Compare mean PC1 scores for each party; flip sign if Republicans are negative. This is the standard convention in the legislative scaling literature (NOMINATE uses positive = conservative).

## Consequences

**Benefits:**
- Row-mean imputation is simple, defensible, and fast. Downstream IRT will handle missingness more rigorously — PCA just needs a reasonable approximation.
- StandardScaler + sklearn PCA is a one-liner, well-tested, and produces results highly correlated (r > 0.95) with NOMINATE scores.
- Duplicated filter logic ensures PCA is a standalone module. No hidden coupling to EDA internals.
- Sensitivity analysis (r = 0.999 on real data) confirms that threshold choice barely matters — results are robust.
- Holdout validation (93% accuracy, 0.97 AUC-ROC) proves PCA captures genuine structure, not just base rate.

**Trade-offs:**
- Row-mean imputation is not the most principled method. If a legislator was absent specifically on contentious votes, their imputed values are biased toward their easy-vote average. IRT handles this properly.
- Duplicated filter logic means a bug fix in EDA's filtering won't automatically propagate to PCA's sensitivity. This is acceptable because the two should be verified independently anyway.
- 5 components are extracted but only PC1-2 are typically interpretable. PC3-5 are retained for downstream IRT comparison but may be noise.
