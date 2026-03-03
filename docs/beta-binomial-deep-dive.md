# Beta-Binomial Deep Dive

A code audit, literature review, and fresh-eyes evaluation of the Beta-Binomial Bayesian party loyalty analysis (Phase 9).

**Date:** 2026-02-25

---

## Executive Summary

The implementation is sound. The core mathematics — conjugate Beta-Binomial updating with method-of-moments empirical Bayes — is textbook-correct and well-suited to the problem. The code is clean, well-tested (26 tests), and well-documented (ADR-0015, design doc, method doc). No correctness bugs were found.

This deep dive identifies **five issues** worth addressing (two substantive, three minor), **three test gaps**, and **one refactoring opportunity**. It also surveys the Python ecosystem and evaluates whether the current approach remains the right choice.

---

## 1. Python Ecosystem Survey

### 1.1 Available Implementations

| Library | What It Provides | Relevance |
|---------|------------------|-----------|
| **scipy.stats.beta** | Beta PDF, CDF, PPF, median — used in current code | Core dependency (correct choice) |
| **scipy.stats.betabinom** | Beta-Binomial PMF, logPMF, CDF — the compound distribution itself | Available since scipy 1.4.0; not used (not needed for conjugate updating) |
| **PyMC** | `pm.BetaBinomial` likelihood + hierarchical hyperpriors via MCMC | Used in Phase 10 (hierarchical IRT); overkill for Phase 9's closed-form approach |
| **NumPyro** | `numpyro.distributions.BetaBinomial` — JAX-backed, fast GPU sampling | Alternative PPL; no advantage over PyMC for this workload |
| **Bambi** | R-formula interface to PyMC; supports beta regression | No native `BetaBinomial` family as of 2025; would need custom family |
| **statsmodels** | No Beta-Binomial model | Not applicable |
| **scikit-learn** | No Beta-Binomial model | Not applicable |

### 1.2 Open Source Projects Using Beta-Binomial for Legislative/Political Analysis

No notable open-source projects were found that apply Beta-Binomial specifically to legislative party loyalty. The closest analogues:

- **David Robinson's "Introduction to Empirical Bayes"** (R, 2017) — the canonical tutorial, using baseball batting averages. Same model structure (Binomial likelihood, Beta prior, method-of-moments estimation). The restaurant analogy in the report's "What Is Bayesian Shrinkage?" section likely draws from this tradition.
- **Shor-McCarty ideal point estimation** uses a different Bayesian approach (IRT, not Beta-Binomial) for legislative ideology.
- **Efron & Morris (1973, 1975)** established the empirical Bayes shrinkage framework that underpins this work. Their baseball example is the direct conceptual ancestor.

### 1.3 Assessment

The current approach — hand-rolled conjugate updating using `scipy.stats.beta` — is the right call. The Beta-Binomial conjugacy makes MCMC unnecessary, and the existing code is more transparent than wrapping a PPL would be. There is no off-the-shelf library that would simplify this code.

---

## 2. Code Audit

### 2.1 Correctness

The mathematical implementation is correct:

- **Posterior parameters:** `alpha_post = alpha_prior + y_i` and `beta_post = beta_prior + (n_i - y_i)` — textbook conjugate update.
- **Posterior mean:** `alpha_post / (alpha_post + beta_post)` — correct weighted average.
- **Shrinkage factor:** `(alpha_prior + beta_prior) / (alpha_prior + beta_prior + n_i)` — correct weight given to prior.
- **Credible intervals:** Equal-tailed via `scipy.stats.beta.ppf` — standard approach.
- **Method of moments:** `common = mu * (1 - mu) / var - 1; alpha = mu * common; beta = (1 - mu) * common` — correct derivation from Beta mean/variance formulas.

### 2.2 Issue 1: Method of Moments Uses Population Variance (Substantive)

**File:** `analysis/14_beta_binomial/beta_binomial.py:171`

```python
var = float(np.var(rates, ddof=0))
```

The method of moments estimates use `ddof=0` (population variance, dividing by N). For small groups — and Senate Democrats with ~10 legislators is exactly this case — sample variance (`ddof=1`, dividing by N-1) is the unbiased estimator and is standard practice for method-of-moments estimation from a sample.

**Impact:** With N=10 Senate Democrats, `ddof=0` underestimates variance by a factor of N/(N-1) = 10/9 ≈ 11%. This inflates the estimated concentration `alpha + beta`, producing a tighter prior than the data warrant. The effect cascades: tighter prior → more shrinkage → narrower credible intervals. For House Republicans (N≈90), the effect is negligible (<2%).

**The literature agrees:** Casella (1985), the standard reference for empirical Bayes with binomial proportions, uses the sample variance (ddof=1). So does the method-of-moments derivation in Gelman et al. BDA3, Chapter 5.

**Recommendation:** Change to `ddof=1`. The guard `len(y) < 2` already prevents division by zero.

### 2.3 Issue 2: Posterior Distributions Plot Reconstructs Alpha/Beta from Floats (Minor Bug)

**File:** `analysis/14_beta_binomial/beta_binomial.py:463-464`

```python
alpha_post = row["alpha_prior"] + row["n_party_votes"] * row["raw_loyalty"]
beta_post = row["beta_prior"] + row["n_party_votes"] * (1 - row["raw_loyalty"])
```

This reconstructs the posterior parameters from `n_party_votes * raw_loyalty` — but `raw_loyalty` is `y_i / n_i`, so this is computing `alpha_prior + n_i * (y_i / n_i) = alpha_prior + y_i`. Mathematically equivalent... **if there's no floating-point loss**. In practice, `n * (y/n)` can differ from `y` by a ULP or two when y and n aren't exact float representations.

This is a minor numerical hygiene issue, not a visible bug. But the real fix is simpler: store `votes_with_party` (the integer y_i) directly in the output DataFrame, then the plot function can use exact integer arithmetic. Currently the output has `n_party_votes` (n_i) and `raw_loyalty` (y_i / n_i) but not `votes_with_party` (y_i) itself.

**Recommendation:** Add `votes_with_party` as an integer column in the output. Use it directly in `plot_posterior_distributions`.

### 2.4 Issue 3: Duplicated Constants in Report Module

**File:** `analysis/14_beta_binomial/beta_binomial_report.py:29-30`

```python
# Constants duplicated to avoid circular import
MIN_PARTY_VOTES = 3
CI_LEVEL = 0.95
```

Two constants are duplicated between `beta_binomial.py` and `beta_binomial_report.py`. If either changes independently, the report will show stale parameter descriptions. This is a maintenance hazard.

**Recommendation:** Import from the main module. The "circular import" comment is stale — `beta_binomial_report.py` does not import from `beta_binomial.py`, so importing constants from it creates no cycle:

```python
from analysis.beta_binomial import CI_LEVEL, MIN_PARTY_VOTES
```

(With the existing `try`/`except` pattern for bare imports.)

### 2.5 Issue 4: Missing Goodness-of-Fit Diagnostic (Substantive Gap)

The implementation assumes the Beta-Binomial model is appropriate but never tests this assumption. Two standard diagnostics are missing:

1. **Overdispersion test.** The Beta-Binomial is justified when binomial data shows overdispersion (more variance than Binomial alone predicts). Without testing, we're asserting overdispersion exists without evidence. Tarone's score test is the standard check — it tests H0: "data are Binomial" vs H1: "data are Beta-Binomial" and requires only the simpler (Binomial) model to be fit.

2. **Posterior predictive check.** After fitting, simulate data from the fitted Beta-Binomial and compare to observed data. This catches model misspecification that the prior estimation step can't detect.

**Impact:** For Kansas data, overdispersion almost certainly exists (legislators within a party genuinely vary in loyalty), so the model choice is almost certainly correct. But documenting the evidence strengthens the analysis. The overdispersion test is particularly useful for the design doc and ADR — it transforms "we chose Beta-Binomial because it makes sense" into "we chose Beta-Binomial because Tarone's test rejected the Binomial (p < 0.001)."

**Recommendation:** Add Tarone's Z-statistic to the filtering manifest. The test is a one-liner:

```python
def tarone_test(y: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Tarone's score test for Beta-Binomial overdispersion."""
    p_hat = y.sum() / n.sum()
    expected = n * p_hat
    variance = n * p_hat * (1 - p_hat)
    z = ((y - expected) ** 2 - variance).sum() / np.sqrt(2 * (variance**2).sum())
    p_value = 1 - sp_stats.norm.cdf(z)
    return float(z), float(p_value)
```

### 2.6 Issue 5: No Effective Sample Size / Prior Strength Reporting

The prior strength is reported as `alpha_prior` and `beta_prior`, but the more interpretable quantity is the **effective sample size** of the prior: `kappa = alpha + beta`. This tells you "how many pseudo-observations does the prior contribute?" — directly comparable to `n_party_votes`.

For example, if `kappa = 15` and a legislator has `n = 50` party votes, the prior contributes the equivalent of 15 observations. If `kappa = 200`, the prior dominates even for legislators with 100+ votes (which would indicate the method-of-moments estimation found very tight clustering).

**Recommendation:** Add `prior_kappa` (or `prior_strength`) to the output parquet and the cross-chamber comparison table. This makes the prior's influence immediately interpretable without mental arithmetic.

---

## 3. Dead Code and Refactoring

### 3.1 No Dead Code Found

The implementation is lean. Every function is called, every constant is used. The `PARTY_COLORS["Independent"]` entry is present for consistency with other phases but never triggers in this phase (Independents are excluded from party-specific models). This is intentional, not dead code.

### 3.2 Refactoring Opportunity: Extract Posterior Computation

The `compute_bayesian_loyalty` function mixes three concerns:

1. Grouping by party and filtering
2. Estimating the prior (empirical Bayes)
3. Computing individual posteriors

Concern #3 (lines 232-264) is a tight inner loop that could be a standalone function, making it independently testable and reusable for hypothetical downstream use (e.g., cross-session could apply a pre-estimated prior to new data without re-estimating):

```python
def compute_posterior(
    y: float, n: float, alpha_prior: float, beta_prior: float, ci_level: float = 0.95
) -> dict[str, float]:
    """Compute Beta posterior statistics for a single legislator."""
    ...
```

**Recommendation:** Low priority. The current structure is clear enough. Worth doing only if cross-session comparison needs to apply pre-fit priors to new legislators.

---

## 4. Test Gaps

### 4.1 No Test for ddof Sensitivity

No test verifies that the method-of-moments estimation uses the correct variance estimator. A test should compare the estimated `alpha + beta` against a known analytic result for a small sample.

```python
def test_method_of_moments_uses_sample_variance(self) -> None:
    """Method of moments should use sample variance (ddof=1), not population variance."""
    # 5 rates with known sample variance
    y = np.array([80, 85, 90, 95, 100])
    n = np.array([100, 100, 100, 100, 100])
    rates = y / n
    sample_var = float(np.var(rates, ddof=1))
    mu = float(np.mean(rates))
    expected_concentration = mu * (1 - mu) / sample_var - 1

    alpha, beta = estimate_beta_params(y, n)
    actual_concentration = alpha + beta

    assert abs(actual_concentration - expected_concentration) < 0.1
```

### 4.2 No Test for Varying Sample Sizes (Variable n_i)

All test fixtures use `n=100` for every legislator. Real data has highly variable sample sizes (some legislators with 30 votes, others with 200). A test should verify correct behavior with heterogeneous n_i values, particularly that the method-of-moments estimation handles the `rates = y / n` computation correctly when n_i varies.

```python
def test_variable_sample_sizes(self) -> None:
    """Legislators with different numbers of party votes should all get valid posteriors."""
    df = _make_unity_df([
        {"legislator_slug": f"rep_{i}", "party": "Republican",
         "full_name": f"Rep {i}", "district": str(i),
         "votes_with_party": int(n * 0.85),
         "party_votes_present": n,
         "unity_score": 0.85, "maverick_rate": 0.15}
        for i, n in enumerate([5, 20, 50, 100, 200])
    ])
    result = compute_bayesian_loyalty(df, "House")
    assert result.height == 5

    # Legislators with more votes should have narrower CIs
    ci_widths = result.sort("n_party_votes")["ci_width"].to_list()
    for i in range(len(ci_widths) - 1):
        assert ci_widths[i] > ci_widths[i + 1] or ci_widths[i] < 0.01
```

### 4.3 No Test for Tarone's Overdispersion Test

Once implemented, Tarone's test needs coverage:

```python
def test_tarone_overdispersed_data(self) -> None:
    """Known overdispersed data should produce significant Tarone's Z."""
    rng = np.random.default_rng(42)
    thetas = rng.beta(5, 1, size=50)  # Overdispersed loyalty rates
    n = np.full(50, 100)
    y = rng.binomial(n, thetas)
    z, p = tarone_test(y, n)
    assert z > 2.0  # Should be highly significant
    assert p < 0.05

def test_tarone_binomial_data(self) -> None:
    """Pure binomial data (no overdispersion) should not be significant."""
    rng = np.random.default_rng(42)
    n = np.full(50, 100)
    y = rng.binomial(n, 0.9)  # Same theta for all — no overdispersion
    z, p = tarone_test(y, n)
    assert p > 0.05
```

### 4.4 Existing Test Weakness: test_high_variance_fallback

**File:** `tests/test_beta_binomial.py:52-69`

This test is weak — it doesn't actually trigger the `var >= mu * (1 - mu)` fallback path. The comment in the test acknowledges this: "var < mu*(1-mu), so it should NOT fall back here." The test then tries a more extreme case that also doesn't trigger the fallback, and settles for just asserting `alpha > 0` and `beta > 0`. To actually trigger the `var >= mu * (1 - mu)` fallback, you need rates that span 0 and 1 asymmetrically:

```python
def test_high_variance_fallback(self) -> None:
    """Variance exceeding Beta maximum should trigger fallback to (1, 1)."""
    # Rates: [0.01, 0.99] — var ≈ 0.24, mu*(1-mu) = 0.25 — still valid.
    # To truly exceed Beta max, need asymmetric rates near boundaries:
    # rates = [0.0, 0.1] — mu=0.05, mu*(1-mu)=0.0475, var=0.005 — valid.
    # Actually very hard to exceed analytically. The only way is mu near 0.5
    # with extreme spread. Use rates like [0.01, 0.99]:
    # mu=0.5, var=0.2401, mu*(1-mu)=0.25 — still valid!
    # The fallback is really only triggered by numerical edge cases or
    # when rates include exact 0.0 or 1.0 (but those hit the mu boundary check).
    # The test should verify the mu boundary checks instead.
    y = np.array([0, 100])
    n = np.array([100, 100])
    alpha, beta = estimate_beta_params(y, n)
    assert alpha == FALLBACK_ALPHA  # mu=0.5, var=0.25, mu*(1-mu)=0.25 — exact boundary
    assert beta == FALLBACK_BETA
```

---

## 5. Empirical Bayes vs. Alternatives

### 5.1 Method of Moments vs. MLE

The current code uses method of moments (MoM). Maximum likelihood estimation (MLE) is the main alternative, using `scipy.optimize.minimize` to maximize `sum(betabinom.logpmf(y, n, alpha, beta))`.

| Criterion | MoM | MLE |
|-----------|-----|-----|
| Speed | Instant (closed-form) | ~10ms (optimization) |
| Bias | Slightly biased for small N | Asymptotically unbiased |
| Simplicity | One formula | Requires optimizer (Nelder-Mead) |
| Robustness | Falls back to Beta(1,1) gracefully | Can fail to converge |
| Accuracy (N>20) | Negligible difference | Negligible difference |
| Accuracy (N≈10) | Slightly worse | Slightly better |

**Verdict:** MoM is the right choice for this project. With 4 groups and the smallest having ~10 members, the difference is marginal. MoM's simplicity and guaranteed convergence outweigh MLE's slight accuracy advantage. Wheeler (2023) notes that MoM "can often misbehave" but this is for general distributions — the Beta-Binomial MoM is well-behaved when `var < mu * (1 - mu)`, which the code already checks.

### 5.2 Empirical Bayes vs. Full Bayes

ADR-0015 made this decision correctly. For the record:

- **Empirical Bayes:** Treats the estimated prior as known. Credible intervals are too narrow by an amount proportional to 1/N_group (hyperparameter uncertainty). For N_group = 90 (House Republicans), this is <1%. For N_group = 10 (Senate Democrats), this could be 5-10%.
- **Full Bayes (PyMC):** Propagates hyperparameter uncertainty through the posterior. More honest credible intervals. Requires MCMC.

The hierarchical IRT model (Phase 10) already provides full Bayesian treatment. Phase 9's role as the fast, exploratory baseline is appropriate.

### 5.3 Alternatives to Beta-Binomial

| Alternative | When to Prefer It | Why Not Here |
|-------------|-------------------|--------------|
| **Logistic-normal** | When you need covariates (district, seniority) | No covariates needed; Beta-Binomial is simpler |
| **Dirichlet-Multinomial** | When outcomes have >2 categories | Party votes are binary (with/against) |
| **Hierarchical logistic regression** | When individual-level covariates matter | Overkill; no covariates to model |
| **Quasi-binomial GLM** | When you just want adjusted standard errors | Doesn't give posteriors or credible intervals |

**Verdict:** Beta-Binomial is the right model. It's the natural conjugate model for "successes out of trials" with group-level variation, and the closed-form posterior is a significant practical advantage.

---

## 6. Downstream Integration Assessment

### 6.1 Synthesis (Phase 11) — Used

Synthesis reads `posterior_loyalty_{chamber}.parquet` and uses `posterior_mean`, `ci_width`, and `shrinkage` to characterize legislators. The integration is optional and graceful — synthesis works with or without beta-binomial output.

### 6.2 Profiles (Phase 12) — Not Used

Despite the design doc stating "Profiles can show credible intervals alongside point estimates," the profiles phase does not read beta-binomial output. This is a missed opportunity — profile scorecards could show raw vs. Bayesian loyalty with CI bars.

### 6.3 Cross-Session (Phase 13) — Not Used

Despite the design doc stating "Cross-session comparison should use posterior means for legislators with few votes in one session," the cross-session phase does not read beta-binomial output. This is a more significant gap — the whole point of shrinkage is stabilizing estimates for low-N legislators, and cross-session comparison of returning legislators across bienniums is exactly the use case where some legislators may have few votes in one session.

**These are not bugs in Phase 9** — they're integration opportunities for downstream phases. Noting them here for completeness.

---

## 7. Summary of Recommendations

### Must Fix (Substantive)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | Change `ddof=0` to `ddof=1` in `estimate_beta_params` | Fixes ~11% variance underestimate for small groups (Senate Democrats) | 1 line |
| 2 | Add Tarone's overdispersion test to filtering manifest | Provides evidence for model choice; strengthens ADR-0015 | ~20 lines + 2 tests |

### Should Fix (Code Quality)

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 3 | Add `votes_with_party` column to output parquet | Fixes float reconstruction in plot; improves data completeness | ~5 lines |
| 4 | Import constants from main module in report | Eliminates duplicated constants | ~5 lines |
| 5 | Add `prior_kappa` to output and cross-chamber table | Makes prior strength interpretable | ~5 lines |

### Should Add (Tests)

| # | Test | What It Catches |
|---|------|-----------------|
| 6 | Method-of-moments uses ddof=1 | Regression on variance estimator |
| 7 | Variable sample sizes produce valid posteriors | Heterogeneous-n edge cases |
| 8 | Tarone's test (overdispersed + null cases) | Overdispersion diagnostic correctness |
| 9 | Fix `test_high_variance_fallback` | Currently doesn't test what it claims |

### Won't Fix (Correct As-Is)

- MoM vs. MLE: MoM is the right choice for this use case
- Empirical Bayes vs. Full Bayes: ADR-0015 is correct; Phase 10 handles full Bayes
- Beta-Binomial vs. alternatives: correct model for the data
- `PARTY_COLORS["Independent"]` unused in this phase: intentional consistency

---

## 8. Key References

- Casella, G. (1985). "An Introduction to Empirical Bayes Data Analysis." *The American Statistician*, 39(2), 83-87.
- Efron, B. & Morris, C. (1973). "Stein's Estimation Rule and Its Competitors — An Empirical Bayes Approach." *JASA*, 68(341), 117-130.
- Efron, B. & Morris, C. (1975). "Data Analysis Using Stein's Estimator and Its Generalizations." *JASA*, 70(350), 311-319.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press. Chapter 5 (hierarchical models).
- Johnson, A., Ott, M. & Dogucu, M. (2022). *Bayes Rules!* CRC Press. Chapter 3 (Beta-Binomial). https://www.bayesrulesbook.com/chapter-3
- Robinson, D. (2017). *Introduction to Empirical Bayes: Examples from Baseball Statistics*. https://drob.gumroad.com/l/empirical-bayes
- Tarone, R.E. (1979). "Testing the Goodness of Fit of the Binomial Distribution." *Biometrika*, 66(3), 585-590.
- Wheeler, A. (2023). "Fitting Beta Binomial in Python." https://andrewpwheeler.com/2023/10/18/fitting-beta-binomial-in-python-poisson-scan-stat-in-r/
- SciPy documentation: `scipy.stats.betabinom` (added v1.4.0). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html
