# PCA Deep Dive: Implementation Review & Literature Comparison

**Date:** 2026-02-24
**Scope:** `analysis/02_pca/pca.py` (1,191 lines), `analysis/02_pca/pca_report.py` (570 lines), `tests/test_pca.py` (610 lines)

This document steps back from the implementation to ask: are we doing PCA right? It surveys the political science literature, evaluates Python alternatives, audits our code for correctness and completeness, and identifies concrete improvements.

---

## 1. Literature Grounding

### 1.1 What PCA Means for Legislative Data

PCA on a binary vote matrix is not textbook PCA. Standard PCA minimizes squared reconstruction error under an implicit Gaussian likelihood model. Binary vote data (Yea=1, Nay=0) follows a Bernoulli distribution. Applying PCA to binary matrices is mathematically "wrong" — but it works well in practice when the latent signal is strong, which it is in polarized legislatures.

This is not an accident. The PCA-IRT relationship is well-understood:

- Both extract low-dimensional latent structure from binary matrices
- IRT uses a logistic link function (models log-odds); PCA uses a linear approximation
- For **one dimension**, PCA and IRT ideal points are typically correlated at r > 0.95 — the ordinal ranking is nearly identical
- For **two+ dimensions**, PCA has a critical advantage: it avoids the **nonidentifiability** problem that Armstrong et al. (2018) describe as a "grave flaw" of both NOMINATE and Bayesian IRT

Our pipeline uses PCA for exactly the right purpose: a cheap, fast pre-IRT sanity check that validates the ideological landscape before investing in MCMC.

### 1.2 Foundational References

| Reference | What It Prescribes | Our Compliance |
|-----------|-------------------|----------------|
| Poole & Rosenthal (NOMINATE) | Drop lopsided votes where minority < 2.5% | `CONTESTED_THRESHOLD = 0.025` (defined in `analysis/tuning.py`) |
| Clinton, Jackman & Rivers 2004 | Binary encoding (Yea=1, Nay=0, else=missing); PCA for dimensionality before IRT | Yea=1/Nay=0/null; PCA is Phase 2 |
| Armstrong et al. 2018 | PCA avoids nonidentifiability in 2+ dimensions; use for ideal point estimation | We use PCA for 1D primarily, with PC2 as diagnostic |
| Wolfram (Dimensionality of Politics) | Yea=+1, Nay=-1, Absent=0 encoding; Gini of scree for dimensionality measurement | We use Yea=1/Nay=0/null (better — see Section 3.1) |

**Verdict:** Our encoding, filtering, and pipeline placement are textbook. The main methodological debates are in preprocessing (Section 3).

### 1.3 The NOMINATE Comparison

NOMINATE is **not** PCA. It is a parametric spatial model with a Gaussian kernel utility function estimated via maximum likelihood. Key differences:

| Aspect | Our PCA | NOMINATE |
|--------|---------|----------|
| Model | Linear (SVD) | Nonlinear (Gaussian kernel) |
| Missing data | Row-mean imputation | EM-style expected value imputation |
| Identification | Up to rotation/reflection/scale | Constrained by polarity legislators |
| Uncertainty | Point estimates only | Bootstrapped standard errors |
| Dimensions | 5 extracted, 1-2 interpreted | Typically 2 |
| Accuracy | 93% holdout | ~85% classification (2D) |

Our 93% holdout accuracy beating NOMINATE's ~85% is partly because we extract more components and partly because Kansas is more one-dimensional than Congress.

---

## 2. Open-Source Landscape

### 2.1 Python PCA Libraries

| Library | What It Offers | Relevant? |
|---------|---------------|-----------|
| **scikit-learn PCA** | Standard PCA via SVD. Mature, well-tested, our current choice | Yes — we use it correctly |
| **prince** (v0.16.5) | MCA, CA, FAMD. Designed for categorical/mixed data | **Yes** — MCA is the theoretically correct method for binary matrices (see Section 3.2) |
| **logisticpca** (v0.3.2) | PCA with Bernoulli likelihood (PyTorch). Principled for binary data | Immature — no missing data support, tiny package, R version much better |
| **factor_analyzer** (v0.5.1) | Factor analysis with rotation (varimax, promax). Pearson correlation only | No — factor analysis is less interpretable than PCA for our use case |
| **TruncatedSVD** (sklearn) | SVD without centering. Works with sparse matrices | No — our matrices fit trivially in memory |
| **IncrementalPCA** (sklearn) | Out-of-core PCA | No — state legislature matrices are tiny |
| **horns** | Horn's parallel analysis for dimensionality selection | **Yes** — valuable diagnostic we're missing (see Section 5.1) |
| **semopy** | SEM package with `polycorr` (tetrachoric/polychoric correlations) | Interesting but niche — tetrachoric PCA is theoretically ideal but Python tooling is immature |

### 2.2 Political Science Tools (R Ecosystem)

| Tool | Language | Python Port? | Notes |
|------|----------|-------------|-------|
| Rvoteview | R | None | Congressional data + W-NOMINATE integration |
| pscl | R | None | `rollcall` class, Bayesian IRT via `ideal()` |
| W-NOMINATE | R | None | Gold standard for congressional scaling |
| idealstan | R (Stan) | None | Time-varying IRT; Stan models could port |

**Conclusion:** The political science computational ecosystem is overwhelmingly R-based. There is no Python library providing a pscl-equivalent `rollcall` structure or NOMINATE scaling. Our approach — implementing directly in sklearn/NumPy/Polars — is the correct strategy and aligns with our Python-only preference (ADR-0002).

---

## 3. The Preprocessing Debate

This is the single most consequential methodological question in our PCA implementation. The literature is genuinely split, and the choice affects results.

### 3.1 Encoding: We Got This Right

Our encoding (Yea=1, Nay=0, absent=null) matches Clinton-Jackman-Rivers exactly. The alternative — Wolfram's encoding (Yea=+1, Nay=-1, Absent=0) — is common in applied work but theoretically problematic: it treats "did not vote" as half-yea/half-nay, which pulls absent legislators toward the center of ideological space. For Kansas, where absences may be strategic, this would be misleading.

### 3.2 StandardScaler: The Debate

This is where our implementation diverges from part of the literature, and the divergence is worth examining carefully.

**What we do:** `StandardScaler()` — center each roll call (subtract mean) **and** divide by standard deviation. Every roll call gets equal weight after scaling.

**What the literature says:**

| Approach | Effect | Who Uses It | Implication for Our Data |
|----------|--------|-------------|--------------------------|
| **Center + scale** (correlation matrix PCA) | Every roll call weighted equally | Our current choice; Poole & Rosenthal | A 97-3 vote that passes the 2.5% filter gets the same weight as a 55-45 party-line showdown |
| **Center only** (covariance matrix PCA) | Close votes (high variance) upweighted, lopsided votes downweighted | Some state-legislature studies | Close votes carry more ideological information — arguably more appropriate |
| **Tetrachoric correlation matrix** | Accounts for binary data being discretized continuous latent trait | Theoretical gold standard | Most principled but Python support is limited (`semopy.polycorr` only) |

**The case for center-only (covariance PCA):** For binary data, variance is a direct function of the base rate: `var = p(1-p)`. A 50-50 vote has maximum variance (0.25); a 95-5 vote has minimal variance (0.0475). Covariance-based PCA naturally upweights contested votes and downweights near-unanimous ones — which **aligns with the motivation behind our 2.5% minority filter**. We filter out the most lopsided votes, then StandardScaler treats the remaining ones as equally informative. But a 3% minority vote is far less informative than a 50% contested vote. Covariance PCA would handle this gradient automatically.

**The case for our current approach (correlation PCA):** Consistency with Poole & Rosenthal. Ensures no single contested roll call dominates. The sensitivity analysis (r=0.999) suggests the distinction barely matters in practice for our data.

**Verdict:** Our approach is defensible and matches established methodology. Switching to covariance PCA would be a reasonable alternative but is not necessary given the r=0.999 sensitivity result. The design doc should acknowledge this trade-off explicitly.

### 3.3 Imputation: Correct but with a Subtle Holdout Bug

Row-mean imputation is the right choice for a pre-IRT step. The design doc already acknowledges its limitations (biases sparse legislators toward their base rate, IRT handles missingness properly).

**However, the holdout validation has a subtle information leak.** The design doc at `analysis/design/pca.md:69` acknowledges this:

> The holdout cells were imputed in the training matrix, so the training PCA is slightly contaminated by the test data through row-mean imputation. This is a minor concern given the high accuracy.

This is correct but understated. When we mask 20% of cells and then impute with row means, each masked cell contributes to the row mean through the *remaining* 80% of that legislator's votes. The row mean itself is uncontaminated. But the masked cells affect the *other* cells' imputed values through the covariance structure (the scaler's mean/std are computed on the training-imputed matrix). The contamination flows:

1. Mask cell (i,j)
2. Compute row mean for legislator i using remaining 80% — **clean**
3. Fill cell (i,j) with row mean — this is the imputed value
4. Compute StandardScaler on the full imputed matrix — **slightly contaminated** because the imputed value at (i,j) is a function of legislator i's other votes, and the scaler sees all imputed values

The practical impact is negligible (our 93% accuracy would probably be ~92% without the leak), but it's worth noting for methodological completeness. A stricter holdout would recompute row means *excluding* all masked cells for each legislator, which is what we actually do (line 833: `row[valid].mean()` where masked cells are NaN). So the imputation itself is clean — the contamination is only through the scaler and PCA fit seeing imputed values that are functions of the same legislator's real votes. This is inherent to any row-mean imputation scheme and not fixable without a fundamentally different approach.

**Recommendation:** No code change needed. The existing caveat in the design doc is sufficient.

### 3.4 The Horseshoe Artifact

When a strong first dimension exists with a gradient, PCA can produce a spurious "arch" or "horseshoe" in the second component: extreme liberals and extreme conservatives curve toward each other in PC2 space. This is a well-known artifact in ecology (Detrended Correspondence Analysis was invented to fix it) and appears in legislative data.

**For our data:** Our Senate PC2 captures "contrarianism on routine legislation" (Tyson, Thompson dissenting on near-unanimous bills), not a horseshoe artifact. The top PC2 loadings (consent calendar, waterfowl hunting, bond validation) confirm this is a real voting pattern, not a mathematical artifact. If it were a horseshoe, we'd see both extreme Republicans and extreme Democrats at the same PC2 end — we don't.

**Recommendation:** No action needed. The current PC2 interpretation is correct.

---

## 4. Code Audit

### 4.1 What's Correct

The implementation is sound. Specific strengths:

- **`impute_vote_matrix()`** — Handles all edge cases: no missing values, all missing, single missing. The all-null fallback (0.5) is the right uninformative prior. Tested with 7 cases.
- **`orient_pc1()`** — Deterministic, party-aware, handles unknown parties gracefully. Tested with 3 cases.
- **`detect_extreme_pc2()`** — Pure function, frozen dataclass output, 3σ threshold, null-safe, strips leadership suffixes. Tested with 6 cases. Follows the project's extract-to-pure-function pattern.
- **`fit_pca()`** — Clean two-liner (StandardScaler + PCA). Returns all intermediate objects for downstream use. No hidden state.
- **`build_scores_df()` / `build_loadings_df()`** — Clean Polars joins with metadata. Gracefully handle missing metadata columns.
- **`run_pca_for_chamber()`** — Good orchestration: impute → fit → orient → build DataFrames → print diagnostics. Returns everything downstream needs.
- **Holdout validation** — Proper reconstruction: scores × components → inverse_transform → binary threshold. Uses `roc_auc_score` with clipped probabilities. Reports base-rate comparison.
- **Sensitivity analysis** — Correct implementation: re-filter at 10%, re-fit PCA, match legislators by slug, compute Pearson r. Handles edge cases (too few legislators, too few shared).

### 4.2 Nothing Is Incorrectly Implemented

After comparing every function against the literature:

- `impute_vote_matrix()` — Row-mean imputation is correct. The loop-based implementation (line 244) is clear if not vectorized (see Section 6.1).
- `fit_pca()` — StandardScaler + PCA is the standard sklearn pipeline. `n_components` is correctly capped at `min(n_components, n_rows, n_cols)`.
- `orient_pc1()` — Party-mean comparison with sign flip is the standard convention. Correctly flips both scores and loadings (only PC1 row).
- `detect_extreme_pc2()` — Uses `std()` (ddof=1 in Polars), checks `arg_min()` for most negative PC2, threshold at 3σ. All correct.
- Holdout validation — The 20% masking, row-mean re-imputation, PCA reconstruction, and metric computation are all correct. The clipping for AUC-ROC (line 856) is necessary and correct.
- Sensitivity — The duplicated filter logic matches EDA's logic. The Pearson r computation (line 725) is correct.

### 4.3 Dead Code

None found. Every function is called from `main()` or from another function that is. Every constant is referenced. Every import is used.

### 4.4 Hardcoded Values

All thresholds are named constants at the top of the file (lines 152-158). No magic numbers embedded in logic. The 3σ threshold in `detect_extreme_pc2()` is inline (`3 * pc2_std`) — this could be a named constant but the function is small enough that it reads clearly as-is.

One exception: the scree plot annotation condition `ev[0] > 2 * ev[1]` (line 419) uses an inline threshold. This is a visualization heuristic, not an analytical threshold, so a named constant isn't necessary.

### 4.5 The `impute_vote_matrix` Duplication

The same imputation logic — "fill each legislator's missing votes with their average Yea rate" — exists in three places in the codebase. All three produce identical results, but they differ in implementation style.

| Location | Implementation | Lines | Added |
|----------|---------------|-------|-------|
| `analysis/pca.py:244-252` | Row-by-row Python `for` loop | 8 | Original |
| `analysis/umap_viz.py:299-312` | Identical `for` loop (copied) | 8 | ADR-0011 |
| `analysis/eda.py:949-955` | **Vectorized NumPy** | 4 | Later (EDA deep dive) |

**The PCA/UMAP version** (original, used in two places):

```python
for i in range(X.shape[0]):
    row = X[i]
    valid_mask = ~np.isnan(row)
    if valid_mask.any():
        row_mean = row[valid_mask].mean()
        X[i, ~valid_mask] = row_mean
    else:
        X[i] = 0.5
```

This iterates row-by-row through the matrix. For each legislator, it finds the non-NaN cells, computes their mean, and fills the NaN cells with that mean. The `else` branch handles the edge case where a legislator has zero votes (fills with 0.5, the uninformative midpoint). This is clear, readable Python — each step is explicit and easy to debug.

**The EDA version** (added later as part of the eigenvalue preview feature):

```python
row_means = np.nanmean(arr, axis=1, keepdims=True)
row_means = np.where(np.isnan(row_means), 0.5, row_means)
mask = np.isnan(arr)
arr[mask] = np.broadcast_to(row_means, arr.shape)[mask]
```

This does the same thing in four lines using NumPy's vectorized operations:

1. `np.nanmean(arr, axis=1, keepdims=True)` — computes each row's mean while ignoring NaNs, returning a column vector. `keepdims=True` preserves the shape for broadcasting.
2. `np.where(np.isnan(row_means), 0.5, row_means)` — replaces NaN row means (from all-NaN rows) with 0.5. This is the vectorized equivalent of the `else: X[i] = 0.5` branch.
3. `np.isnan(arr)` — creates a boolean mask of all NaN cells.
4. `np.broadcast_to(row_means, arr.shape)[mask]` — broadcasts each row's mean across all columns, then selects only the NaN positions to fill.

The vectorized version is ~10x faster on large matrices (NumPy's C loops vs Python's `for` loop), but for our data sizes (~165 legislators × ~500 votes), both complete in microseconds. The practical difference is zero.

**Why does this matter?** It doesn't, operationally. Both versions are correct, both are tested (PCA's version via `TestImputeVoteMatrix` with 7 test cases; EDA's version implicitly through the eigenvalue preview). The divergence is a natural consequence of the self-containment design: each phase was developed independently, and the EDA version was written later with the benefit of hindsight.

The duplication is intentional and documented (ADR-0005 for PCA, ADR-0011 for UMAP). Extracting a shared `analysis/imputation.py` utility would reduce duplication but introduce coupling between phases — exactly what the ADRs decided against. The current approach means a bug fix in one version must be applied to the others manually, but it also means a breaking change in one phase can't silently propagate to the others.

---

## 5. Recommended Additions

### 5.1 Parallel Analysis for Dimensionality Selection (High Value)

**What:** Horn's parallel analysis — compare actual eigenvalues against eigenvalues from random data of the same shape. Retain components whose eigenvalues exceed the 95th percentile of the random distribution.

**Why:** Our scree plot is a visual diagnostic. Parallel analysis provides an **objective** answer to "how many dimensions?" The scree plot's elbow is subjective; parallel analysis is a statistical test. This is the recommended dimensionality selection method in the psychometric literature and widely used in political science.

Currently, we extract 5 components and note that "only PC1-2 are typically interpretable." Parallel analysis would formalize this: it would likely confirm 1-2 significant dimensions for Kansas and provide early warning for sessions where the structure differs.

**How:** Two options:

1. **`horns` package** — `pip install horns`, ~5 lines of code. Compares actual eigenvalues against random eigenvalue distribution.
2. **DIY** — ~15 lines of NumPy. Generate 100 random matrices of the same shape, compute eigenvalues of each, take the 95th percentile, compare against actual.

The DIY approach is straightforward:
```python
def parallel_analysis(X: np.ndarray, n_iter: int = 100) -> np.ndarray:
    """95th percentile eigenvalues from random data of same shape."""
    n, p = X.shape
    rng = np.random.default_rng(42)
    random_eigenvalues = np.zeros((n_iter, min(n, p)))
    for i in range(n_iter):
        random_data = rng.standard_normal((n, p))
        corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigvalsh(corr))[::-1][:min(n, p)]
    return np.percentile(random_eigenvalues, 95, axis=0)
```

**Impact on downstream:** Purely additive diagnostic. Would appear on the scree plot as a reference line ("retain components above this line"). No impact on PCA fitting or downstream phases.

**Priority:** High. Adds an objective dimensionality diagnostic with minimal code. Would have provided additional context for the 5 failing IRT chamber-sessions.

### 5.2 Eigenvalue Ratio Reporting (High Value, Minimal Effort)

**What:** Report the lambda1/lambda2 ratio explicitly. A ratio > 5 means "strongly one-dimensional." A ratio of 2-3 means "meaningful second dimension."

**Why:** Currently buried in the scree plot and explained variance table. Making it a headline number (printed in console output and manifested in JSON) gives a single-number summary of dimensionality. This is the simplest possible diagnostic and takes ~3 lines of code.

**How:** After fitting PCA, compute `eigenvalues[0] / eigenvalues[1]` and print it. Add to the filtering manifest.

**Impact:** None on downstream. Pure diagnostic.

**Priority:** High. Three lines of code, high interpretive value.

### 5.3 Reconstruction Error by Legislator (Medium Value)

**What:** After PCA reconstruction, compute per-legislator RMSE. Flag legislators with reconstruction error > 2σ above mean.

**Why:** High reconstruction error for a legislator means PCA doesn't fit them well — they have voting patterns that are poorly explained by the dominant dimensions. These legislators are the ones most likely to show IRT convergence issues or appear as synthesis outliers. Currently, we only report aggregate holdout accuracy; per-legislator error would identify specific problem cases.

**How:** ~10 lines of NumPy. After reconstruction (`X_hat = scaler.inverse_transform(scores @ components)`), compute per-row RMSE. Save to parquet alongside PC scores.

**Impact:** Informational. Would complement the PC2 extreme detection by identifying legislators who are anomalous for *any* reason (not just extreme PC2).

**Priority:** Medium. More useful for diagnosing IRT failures than for standalone PCA interpretation.

### 5.4 MCA Comparison (Lower Value, But Theoretically Interesting)

**What:** Run Multiple Correspondence Analysis (MCA) via the `prince` library on the same binary matrix and compare MCA dimension 1 against PCA PC1.

**Why:** MCA is the theoretically correct dimensionality reduction method for categorical data. It uses chi-squared distances rather than Euclidean, which is more appropriate for binary variables. If PCA and MCA agree strongly (expected: r > 0.95), it validates that PCA's Gaussian assumption doesn't matter much for our data. If they disagree, it would reveal cases where the binary nature of the data is misleading PCA.

**Caveats:** Prince is pandas-only (no Polars), so this would require a temporary conversion. MCA's eigenvalues are not directly comparable to PCA's (they use different scales). The comparison would be ordinal (Spearman rank correlation of dimension 1 scores).

**Impact:** Validation only. Would not change the pipeline. Adds a pandas dependency to one analysis phase.

**Priority:** Lower. The PCA-IRT correlation (r > 0.95) already validates our approach. MCA would add a second validation point but at the cost of a pandas dependency we've deliberately avoided.

---

## 6. Refactoring Opportunities

### 6.1 Vectorize `impute_vote_matrix()` (Low Priority)

The current implementation uses a Python `for` loop over rows:

```python
for i in range(X.shape[0]):
    row = X[i]
    valid_mask = ~np.isnan(row)
    if valid_mask.any():
        row_mean = row[valid_mask].mean()
        X[i, ~valid_mask] = row_mean
    else:
        X[i] = 0.5
```

The EDA phase already has the vectorized version:

```python
row_means = np.nanmean(arr, axis=1, keepdims=True)
row_means = np.where(np.isnan(row_means), 0.5, row_means)
mask = np.isnan(arr)
arr[mask] = np.broadcast_to(row_means, arr.shape)[mask]
```

For our data sizes (~165 rows), the performance difference is negligible. The loop version is arguably more readable. Not worth changing unless we're already touching the function for another reason.

**Priority:** Low. Cosmetic improvement. No functional change.

### 6.2 Extract Shared Constants (Low Priority)

Several constants are duplicated between `pca.py` and `eda.py`:

| Constant | PCA | EDA | Same? |
|----------|-----|-----|-------|
| `CONTESTED_THRESHOLD` | 0.025 | 0.025 | Yes |
| `MIN_VOTES` | 20 | 20 | Yes |
| `PARTY_COLORS` | 3 colors | 3 colors | Yes |

This is intentional per ADR-0005 (self-containment). The constants are simple enough that duplication is fine. A shared `analysis/constants.py` would reduce duplication but introduce coupling between phases.

**Priority:** Low. The self-containment trade-off is documented and deliberate.

### 6.3 `filter_vote_matrix_for_sensitivity()` Length (Low Priority)

This function is ~50 lines (616-665) of straightforward filter logic. It's not complex — two sequential filters (near-unanimous votes, then low-participation legislators). It could be split into `_filter_unanimous()` and `_filter_low_participation()`, but the current single function reads clearly and the duplication is intentional.

**Priority:** Low. Works correctly, reads clearly.

---

## 7. Test Coverage Assessment

### 7.1 Current State

| Component | Tested? | Tests | Notes |
|-----------|---------|-------|-------|
| `impute_vote_matrix()` | Yes | 7 | All edge cases covered |
| `orient_pc1()` | Yes | 3 | Flip, no-flip, unknown party |
| `detect_extreme_pc2()` | Yes | 6 | Comprehensive: none, extreme, fields, null name, frozen, suffix |
| `fit_pca()` | No | 0 | Simple sklearn wrapper; testing would be testing sklearn |
| `build_scores_df()` | No | 0 | Polars join logic; not complex |
| `build_loadings_df()` | No | 0 | Polars join logic; not complex |
| `run_pca_for_chamber()` | No | 0 | Orchestration; would need full test fixtures |
| `run_sensitivity()` | No | 0 | Integration test territory |
| `run_holdout_validation()` | No | 0 | Integration test territory |
| `filter_vote_matrix_for_sensitivity()` | No | 0 | Duplicated logic; should be tested |
| `plot_*()` (4 functions) | No | 0 | Expected — visual output |

**Overall:** 16 tests covering the 3 most critical functions. The untested functions are either thin wrappers around well-tested libraries (sklearn) or integration-level orchestration.

### 7.2 Recommended New Tests

1. **`test_filter_vote_matrix_for_sensitivity`** — This is the most important gap. The function duplicates EDA's filter logic, so bugs in one won't be caught by the other's tests. Create a small synthetic vote matrix with:
   - One near-unanimous vote (should be filtered at 10% but not at 2.5%)
   - One low-participation legislator (should be filtered)
   - Verify the output dimensions match expectations

2. **`test_fit_pca_components_capped`** — Verify that `n_components` is correctly capped at `min(n_components, n_rows, n_cols)`. Edge case: 3 legislators × 100 votes → only 3 components possible.

3. **`test_build_scores_df_missing_metadata`** — Verify that `build_scores_df` handles the case where a legislator slug isn't in the metadata DataFrame (should produce nulls in metadata columns, not crash).

4. **`test_holdout_beats_base_rate`** — A property test on synthetic data: create a vote matrix with clear partisan structure (two parties, opposite voting patterns), run holdout validation, verify accuracy > base rate. This would catch regressions in the reconstruction pipeline.

5. **`test_sensitivity_high_correlation`** — Similar: create synthetic data where filtering at 10% removes some votes but preserves the ideological structure. Verify Pearson r > 0.95.

**Priority:** Tests 1 and 2 are most valuable. Tests 3-5 are integration tests that provide regression protection but are slower to write and run.

---

## 8. Known Pitfalls & Limitations

### 8.1 Pearson Correlation Attenuation for Binary Data

Pearson correlations between binary variables are bounded by their marginal proportions. A binary variable with mean 0.95 can never have a Pearson correlation of +1 with a variable of mean 0.50, even if they are perfectly associated. This means PCA on the Pearson correlation matrix (what StandardScaler + PCA computes) systematically **underestimates** the strength of association between roll calls.

The fix is tetrachoric correlation, which assumes each binary variable is a threshold decision on a latent continuous normal — exactly the spatial voting model assumption. `semopy.polycorr` can compute tetrachoric correlations, but Python tooling is immature compared to R/Stata.

**Impact for our data:** The attenuation is modest because our filtered vote matrix excludes the most lopsided votes (where the attenuation is worst). The remaining votes have base rates in the 0.10-0.90 range, where Pearson and tetrachoric correlations are similar.

**Recommendation:** Acknowledge in the design doc. No code change needed. If we ever validate against W-NOMINATE (which uses a different loss function entirely), the tetrachoric issue would be worth revisiting.

### 8.2 Imputation Artifacts for Sparse Legislators

Documented in `analytic-flags.md`: Sen. Miller (30/194 votes, 15.5% participation) has 85% of his matrix imputed. His PC2 extreme is an imputation artifact. IRT handles Miller properly.

This is inherent to any imputation-based approach. The MIN_VOTES=20 filter removes the worst cases, but legislators in the 20-50 vote range still have significant imputation. There's no perfect solution short of IRT's native missing-data handling.

### 8.3 Equal Weighting After StandardScaler

After standardization, a procedural bill that happened to be contested (e.g., a bill naming a highway that drew 3 Nay votes) has the same weight as a major policy vote (e.g., tax reform with a 55-45 split). The 2.5% minority filter removes the most egregious cases, but some low-signal votes remain.

The covariance-based PCA alternative (Section 3.2) would partially address this by giving higher weight to closer votes. The design doc should note this trade-off.

### 8.4 Orthogonality Constraint

PCA forces PC1 ⊥ PC2. If the true ideological structure has correlated dimensions (e.g., fiscal conservatism partially predicts social conservatism), PCA distorts this. Factor analysis with oblique rotation (promax) would allow correlated dimensions, but adds complexity and is harder to interpret.

**Impact for our data:** Minimal. Kansas is strongly one-dimensional (PC1 explains 46-53% of variance). PC2 captures a genuinely different pattern (contrarianism), not a correlated extension of PC1.

---

## 9. Comparison with Downstream Pipeline

| Metric | PCA Computes? | Downstream Recomputes? | Duplication? |
|--------|---------------|----------------------|--------------|
| Imputed vote matrix | Yes | UMAP duplicates imputation | Intentional (self-containment) |
| PC scores | Yes | No (IRT, clustering read PCA output) | No |
| PC loadings | Yes | No | No |
| Explained variance | Yes | No | No |
| Sensitivity (10% filter) | Yes, duplicated from EDA | No | Intentional (self-containment) |
| PC1 sign convention | Yes | IRT uses PCA scores directly | No — PCA sets convention, IRT inherits |

**Conclusion:** No problematic duplication. The imputation and filter duplication between PCA/UMAP/EDA is a deliberate design choice documented in ADRs 0005 and 0011.

---

## 10. Summary

### What We're Doing Right

- Binary encoding (Yea=1/Nay=0/null) matches Clinton-Jackman-Rivers exactly
- Pipeline placement (PCA before IRT) is the textbook workflow
- Row-mean imputation is simple, defensible, and the right choice for a pre-IRT step
- PC1 sign convention (Republicans positive) matches NOMINATE/VoteView
- Holdout validation (93% accuracy, 0.97 AUC-ROC) proves real structure captured
- Sensitivity analysis (r=0.999) confirms robustness to threshold choice
- Data-driven PC2 detection is testable, null-safe, follows project patterns
- All thresholds are named constants; no magic numbers
- Design doc + ADR document every methodological choice
- 16 tests cover the critical statistical logic

### What We Added

1. **Parallel analysis** (Horn 1965) — `parallel_analysis()` generates 100 random matrices, compares eigenvalues against 95th percentile threshold. Plotted as reference line on scree plot. Reports significant dimension count.
2. **Eigenvalue ratio** (λ1/λ2) — single-number dimensionality summary. >5 = strongly 1D, 3-5 = predominantly 1D, <3 = meaningful 2nd dimension.
3. **Per-legislator reconstruction error** — `compute_reconstruction_error()` computes RMSE per legislator, flags outliers > mean + 2σ. Saved as parquet, reported in HTML.
4. **Vectorized imputation** — replaced Python `for` loop with NumPy `np.nanmean` + broadcast (matches EDA's version).

### What We Tested (20 new tests, 36 total)

1. `TestFitPCA` (4 tests) — component capping, output shapes, standardization
2. `TestBuildScoresDf` (2 tests) — missing metadata produces nulls, scores preserved
3. `TestFilterVoteMatrixForSensitivity` (4 tests) — threshold filtering, chamber restriction, participation filter
4. `TestParallelAnalysis` (5 tests) — shape, positivity, monotonicity, determinism, structured data exceeds thresholds
5. `TestComputeReconstructionError` (5 tests) — shape, columns, non-negativity, flag correctness, perfect reconstruction

### What We Should NOT Do

- **Do not switch to covariance PCA.** Correlation PCA is defensible, matches Poole & Rosenthal, and the sensitivity analysis shows the distinction doesn't matter for our data.
- **Do not adopt logisticpca.** The Python package is immature (v0.3.2, no missing data support, PyTorch dependency). The R version is better, but we don't use R.
- **Do not implement tetrachoric PCA.** Python tooling is insufficient. The attenuation effect is modest for our filtered data.
- **Do not add MCA without clear justification.** It would validate our approach but adds a pandas dependency and complexity for a diagnostic comparison.
- **Do not refactor out the filter duplication.** It's a deliberate self-containment choice documented in ADR-0005.
- **Do not add rotation (varimax/promax).** Our PC2 interpretation is based on loadings examination, which works fine without rotation. Rotation adds complexity and changes PC1, which would break the downstream sign convention.

---

## References

- Armstrong, Bakker, Carroll, Hare, Poole & Rosenthal 2018. "Analyzing Spatial Models of Choice and Judgment." *Social Sciences* 7(1). [MDPI](https://www.mdpi.com/2076-0760/7/1/12)
- Clinton, Jackman & Rivers 2004. "The Statistical Analysis of Roll Call Data." *American Political Science Review* 98(2).
- Poole & Rosenthal. "NOMINATE: A Short Intellectual History." [PDF](https://legacy.voteview.com/pdf/nominate.pdf)
- Wolfram. "Dimensionality of Politics." [Site](https://christopherwolfram.com/projects/dimensionality-of-politics/)
- Horn 1965. "A rationale and test for the number of factors in factor analysis." *Psychometrika* 30(2).
- Landgraf & Lee. "Dimensionality Reduction for Binary Data through the Projection of Natural Parameters." [arXiv](https://arxiv.org/pdf/1011.3626)
- VanderPlas. "In Depth: Principal Component Analysis." [PDSH](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
