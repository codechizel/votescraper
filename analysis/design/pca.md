# PCA Design Choices

**Script:** `analysis/02_pca/pca.py`
**Constants defined at:** `analysis/02_pca/pca.py:149-160`
**ADR:** `docs/adr/0005-pca-implementation-choices.md`
**Deep dive:** `docs/pca-deep-dive.md`

## Assumptions

1. **Linear ideology.** PCA assumes that voting behavior is a linear function of latent ideological dimensions. A legislator's vote on any bill is modeled as a linear combination of their position on each principal component. This is adequate for a first pass but misses the nonlinear relationship that IRT captures (the logistic link function).

2. **Complete data required.** PCA cannot handle nulls. Every cell in the input matrix must have a value, which requires imputation. The imputation method is itself a design choice (see below).

3. **Equal weighting of roll calls.** After standardization (StandardScaler), every contested roll call contributes equally to PCA. A bill that barely cleared the 2.5% minority threshold has the same influence as a major party-line vote.

4. **Orthogonal dimensions.** PCA forces principal components to be uncorrelated. If the true ideological structure has correlated dimensions (e.g., fiscal conservatism partially predicts social conservatism), PCA distorts this by forcing orthogonality.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `DEFAULT_N_COMPONENTS` | 5 | Extracts 5 PCs per chamber. Only PC1-2 are typically interpretable; PC3-5 retained for IRT comparison. |
| `MINORITY_THRESHOLD` | 0.025 | Inherited from EDA. PCA does not re-filter the default matrices. |
| `SENSITIVITY_THRESHOLD` | 0.10 | Per workflow rules: re-run at 10% for sensitivity analysis. |
| `MIN_VOTES` | 20 | Inherited from EDA. |
| `HOLDOUT_FRACTION` | 0.20 | Random 20% of non-null cells masked for holdout validation. |
| `HOLDOUT_SEED` | 42 | NumPy random seed for reproducible holdout and parallel analysis. |
| `PARALLEL_ANALYSIS_N_ITER` | 100 | Random matrices generated for Horn's parallel analysis. |
| `RECONSTRUCTION_ERROR_THRESHOLD_SD` | 2.0 | Flag legislators with reconstruction RMSE > mean + 2σ. |
| PC2 extreme threshold | 3σ | `detect_extreme_pc2()` flags the min-PC2 legislator only if `|PC2| > 3 × std(PC2)`. |

## Methodological Choices

### Imputation: row-mean (each legislator's Yea rate)

**Decision:** Missing values (nulls) are filled with each legislator's average Yea rate across their non-missing votes. A legislator who voted Yea 80% of the time gets their absences filled with 0.80. Implemented as vectorized NumPy (`np.nanmean` + broadcast).

**Alternatives considered:**
- Column-mean (bill's average Yea rate) — rejected because it erases per-legislator signal, which is exactly what PCA needs
- Zero-fill (treat absences as Nay) — rejected because it falsely asserts strategic opposition on every missed vote
- Iterative imputation (SoftImpute, MICE) — rejected as overkill for a pre-IRT sanity check; these methods are principled but harder to explain and debug
- Drop legislators with any nulls — rejected because it would eliminate ~30% of legislators

**Impact:** Row-mean imputation biases absent legislators toward their own base rate. If a legislator strategically missed contentious votes (avoiding recorded dissent), their imputed values will be biased toward their easy-vote average. This makes their PCA score look more moderate than they truly are. IRT handles this properly by simply not including absent cells in the likelihood.

**Key concern for downstream:** Sen. Miller (30/194 Senate votes) has 85% of his matrix imputed. His PC2 extreme is an imputation artifact, not a real voting pattern. See `docs/analytic-flags.md`.

### Standardization: center and scale (StandardScaler)

**Decision:** Each roll call column is centered (subtract mean) and scaled (divide by standard deviation) before PCA. This is correlation-matrix PCA.

**Why:** Without scaling, close votes (high variance in the binary column) dominate PCA while near-unanimous votes contribute little. Centering removes the overall Yea rate; scaling ensures each roll call contributes equally.

**Alternatives:** Center-only (covariance PCA) — naturally upweights contested votes and downweights near-unanimous ones, which aligns with the motivation behind the 2.5% filter. Some state-legislature studies use this. We chose center+scale to match Poole & Rosenthal's methodology. The sensitivity analysis (r=0.999) confirms the distinction barely matters in practice.

**Impact:** Every contested roll call has equal weight after scaling. This means a procedural vote that happened to be contested has the same influence as a major policy vote. The sensitivity analysis (10% threshold) partially addresses this by removing borderline-contested votes.

### PC1 sign convention: Republicans positive

**Decision:** After fitting PCA, compare mean PC1 scores for Republicans and Democrats. If Republicans are negative, flip the sign of PC1 scores and loadings.

**Why:** PCA components have arbitrary sign. This convention (positive = conservative) matches the NOMINATE literature and makes interpretation consistent across runs and sessions.

**Impact:** All downstream consumers of PC1 scores (IRT anchor selection, visualizations, reports) can assume positive = conservative.

### Holdout validation: mask-and-reconstruct

**Decision:** Randomly mask 20% of non-null cells, re-impute, re-fit PCA on the training set, reconstruct the full matrix, and evaluate predictions on the masked cells.

**Impact:** This tests whether PCA captures enough structure to predict held-out votes better than the ~82% Yea base rate. Current results: 93% accuracy, 0.97 AUC-ROC — PCA clearly captures real structure.

**Caveat:** The holdout cells were imputed in the training matrix, so the training PCA is slightly contaminated by the test data through row-mean imputation. The imputation itself is clean (row means exclude masked cells), but the scaler and PCA fit see imputed values that are functions of the same legislator's real votes. This is inherent to any row-mean imputation scheme and not fixable without a fundamentally different approach. Practical impact is negligible.

### Sensitivity: duplicated filter logic

**Decision:** The sensitivity analysis re-filters the full vote matrix at 10% minority threshold using ~40 lines of duplicated filter logic (not imported from `eda.py`).

**Why:** Keeps PCA self-contained. Changes to EDA's filtering won't silently alter PCA's sensitivity analysis.

**Impact:** If a filtering bug is found in EDA, it must be fixed in PCA (and IRT) separately.

### Parallel analysis (Horn 1965)

**Decision:** After fitting PCA, run Horn's parallel analysis to objectively determine the number of significant dimensions. Generate 100 random matrices of the same shape, compute their correlation-matrix eigenvalues, and compare actual eigenvalues against the 95th percentile of the random distribution.

**Why:** The scree plot is a visual diagnostic — subjective. Parallel analysis provides a statistical test: components whose eigenvalues exceed the random threshold are statistically significant. This is the recommended dimensionality selection method in the psychometric literature.

**Impact:** Purely additive diagnostic. Appears as a reference line on the scree plot and a table in the HTML report. No impact on PCA fitting or downstream phases. Provides early warning for sessions where the ideological structure differs from the expected 1-2 dimensions.

### Eigenvalue ratio (λ1/λ2)

**Decision:** Report the ratio of the first eigenvalue to the second as a single-number summary of dimensionality. Ratio > 5 = "strongly one-dimensional," 3-5 = "predominantly one-dimensional," < 3 = "meaningful second dimension."

**Why:** Immediately signals whether Kansas politics is one-dimensional without requiring interpretation of a scree plot.

### Per-legislator reconstruction error

**Decision:** After PCA reconstruction, compute per-legislator RMSE. Flag legislators with RMSE > mean + 2σ as "high error" — their voting patterns are poorly explained by the dominant dimensions.

**Why:** Complements PC2 extreme detection by identifying legislators who are anomalous for any reason. High reconstruction error candidates are likely to show IRT convergence issues or appear as synthesis outliers.

**Impact:** Saved as a separate parquet file. High-error legislators shown in the HTML report. No impact on PCA fitting or downstream phases.

## Downstream Implications

### For IRT (Phase 3)
- **PCA scores are used to select IRT anchors.** IRT picks the most-conservative (highest PC1) and most-liberal (lowest PC1) legislators as anchors, constrained to xi=+1 and xi=-1 respectively. If PCA scores are wrong, the IRT model will be anchored to the wrong legislators.
- **PCA-IRT correlation is a validation check.** Pearson r > 0.95 between IRT ideal points and PCA PC1 is expected. Lower correlation suggests IRT is capturing nonlinearities that PCA misses. The IRT report now uses a data-driven caption for the IRT-PCA scatter plot based on the actual r value (|r| > 0.95 → "High correlation", |r| > 0.85 → "Moderate", else → "Low — investigate horseshoe").
- **Row-mean imputation is NOT used by IRT.** IRT handles missing data natively. The imputation artifacts that affect PCA (e.g., Miller's PC2 extreme) will not carry over to IRT.
- **2D IRT initialization from PCA.** Phase 06 (2D IRT) uses PCA loadings for beta initialization when horseshoe is detected, avoiding contamination from 1D IRT estimates that may already be horseshoe-distorted (ADR-0112). PCA sensitivity r < 0.95 triggers a warning in the PCA report.

### For Clustering (Phase 5)
- PCA scores (PC1-2 or more) can be used as clustering input features.
- The sign convention (Republicans positive) is baked into the scores. Clustering methods that use distance are sign-agnostic, but any threshold-based cluster interpretation should account for the convention.

### For interpretation
- **PCA gives point estimates only.** No uncertainty intervals. A legislator at PC1=0.5 might truly be anywhere from 0.3 to 0.7 — PCA cannot tell you. Use IRT credible intervals for uncertainty.
- **PC2 interpretation requires examining loadings.** In the current data, Senate PC2 captures "contrarianism on routine legislation" (driven by Tyson/Thompson), not a traditional second ideological dimension. Do not over-interpret PC2 as a coherent dimension without checking what drives it.
- **Parallel analysis results** formalize what the scree plot suggests visually. If only 1 dimension is significant, PC2 should be treated as noise rather than a substantive dimension.
- **Reconstruction error** identifies legislators whose voting patterns don't fit the PCA model. These require case-by-case investigation rather than IRT interpretation.
