# ADR-0041: MCA Implementation Choices

**Date:** 2026-02-25
**Status:** Accepted

## Context

Multiple Correspondence Analysis (MCA) is added as Phase 03 — a categorical-data analogue of PCA that uses chi-square distance instead of Euclidean distance. On binary (Yea/Nay) data, MCA is mathematically equivalent to PCA (Lebart et al. 1977). MCA only adds value when using the full categorical encoding: Yea / Nay / Absent as three distinct categories. This is the entire justification for the phase — it preserves the absence dimension that PCA imputes away.

Several implementation choices needed to be resolved:

1. **Categorical encoding.** How many categories per vote, and how to handle the 5 Kansas vote types.
2. **MCA library.** Custom NumPy implementation vs. established library.
3. **Inertia correction.** Raw MCA inertia is artificially low (the "pessimistic eigenvalue problem"). Which correction to apply.
4. **Polars-pandas boundary.** The project uses Polars throughout; MCA libraries expect pandas.
5. **Dim1 orientation.** MCA axes have arbitrary sign.
6. **Horseshoe detection.** How to identify the arch artifact common in correspondence analysis.
7. **PCA cross-validation.** How to confirm MCA and PCA agree on the ideological ordering.

## Decision

1. **Three categories: Yea / Nay / Absent.** All absence-type categories (Absent and Not Voting, Not Voting, Present and Passing) are collapsed to a canonical "Absent" label. Rationale: the distinctions between absence types are parliamentary procedure details, not ideological signals. Three categories give MCA enough categorical structure to exceed PCA while avoiding sparse rare categories.

2. **prince library (≥0.13).** `prince` is MIT-licensed, FactoMineR-validated via rpy2 test suite, implements both Benzécri and Greenacre corrections, and supports supplementary variables. The pandas dependency is acceptable — conversion happens at a single boundary function (`polars_to_pandas_categorical`). The alternative (custom NumPy SVD on an indicator matrix) would duplicate tested logic and require implementing corrections from scratch.

3. **Greenacre correction (default).** Greenacre's inertia correction divides by average off-diagonal inertia rather than the sum of corrected eigenvalues (Benzécri). Greenacre is more conservative — Benzécri tends to overestimate explained variance. Both are available via `--correction` CLI flag. See Greenacre (2006) and the comparison in `docs/mca-deep-dive.md`.

4. **Single conversion boundary.** `polars_to_pandas_categorical(matrix)` converts the Polars vote matrix to a pandas DataFrame with string dtype at a single point. All upstream data loading and downstream result handling uses Polars. The conversion builds pandas DataFrames from Python dicts to avoid a pyarrow dependency.

5. **Republicans positive on Dim1.** Same convention as PCA PC1 — positive = conservative. Applied by checking party means and flipping all coordinates if needed. This makes cross-method comparison intuitive.

6. **Quadratic R² > 0.80 flags horseshoe.** A quadratic polynomial regression of Dim2 on Dim1 detects the arch/horseshoe effect, a known mathematical artifact of correspondence analysis on gradient-structured data (Greenacre 2017, Le Roux & Rouanet 2004). The horseshoe confirms unidimensionality — Dim2 is a function of Dim1, not genuine second-dimension variation.

7. **Spearman rank correlation for PCA validation.** MCA Dim1 and PCA PC1 are compared via Spearman r on shared legislators. Expected r > 0.90 for binary-dominated data (categorical encoding adds Absent but the Yea/Nay dimension still dominates). Lower values indicate MCA found genuinely different structure. A scatter plot is produced for visual inspection.

## Consequences

**Benefits:**
- MCA positions absences in the ideological space rather than imputing them away — the absence map reveals whether high-absence legislators cluster near one party (partisan absence patterns).
- The biplot maps legislators AND vote categories into the same space — directly interpretable ("this legislator sits near the Yea categories of these bills").
- Contributions (CTR) identify which specific vote-category combinations define each dimension — more granular than PCA loadings.
- PCA validation provides a built-in cross-method check: if MCA and PCA agree (r > 0.90), PCA's linear assumptions are validated for this dataset.
- Sensitivity analysis (2.5% vs 10% contested threshold) tests robustness with minimal additional compute.

**Trade-offs:**
- prince requires pandas at the conversion boundary, violating the Polars-everywhere preference. Mitigated by isolating the conversion to a single function.
- Raw MCA inertia percentages look much lower than PCA variance explained (indicator matrix inflation). The correction compensates, but users unfamiliar with MCA may be confused by raw values. Report captions explain this.
- MCA outputs are currently terminal — not consumed by downstream phases. Future integration points: synthesis could incorporate absence dimension findings, cross-session could compare MCA dimensions, profiles could show per-legislator MCA coordinates.
- 34 new tests added (total: ~1159). Test fixtures use synthetic data with clear party structure.

**References:**
- Greenacre (2017), *Correspondence Analysis in Practice*, 3rd ed.
- Le Roux & Rouanet (2010), *Multiple Correspondence Analysis*, SAGE
- Benzécri (1979), inertia correction formula
- Lebart et al. (1977), MCA-PCA equivalence on binary data
- Full literature survey: `docs/mca-deep-dive.md`
- Design choices: `analysis/design/mca.md`
