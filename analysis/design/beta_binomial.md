# Beta-Binomial Party Loyalty Design Choices

**Script:** `analysis/14_beta_binomial/beta_binomial.py`
**Constants defined at:** `analysis/14_beta_binomial/beta_binomial.py` (top of file)

## Assumptions

1. **Exchangeability within party-chamber group.** Legislators in the same party and chamber are treated as draws from the same Beta distribution of loyalty rates. This ignores caucus or district-level structure but is standard for empirical Bayes.

2. **Beta-Binomial conjugacy.** Each legislator's party-line votes are modeled as Binomial(n, theta), with a Beta prior on theta. The conjugate posterior is closed-form — no MCMC needed.

3. **Independence across legislators.** The empirical Bayes step estimates the prior from all legislators' rates simultaneously, but individual posteriors are computed independently. This underestimates the correlation between legislators who vote on the same bills, but is adequate for point estimates and credible intervals.

4. **Inputs from indices phase are correct.** We read `party_unity_{chamber}.parquet` directly and trust the CQ-standard party vote identification upstream.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `MIN_PARTY_VOTES` | 3 | Need at least 3 party votes for method of moments to produce meaningful rates |
| `CI_LEVEL` | 0.95 | Standard credible interval width |
| `FALLBACK_ALPHA` | 1.0 | Beta(1,1) = uniform prior when method of moments fails |
| `FALLBACK_BETA` | 1.0 | Same |
| `PARTY_COLORS` | R=#E81B23, D=#0015BC | Consistent with all prior phases |

## Diagnostics

### Tarone's Overdispersion Test

Before applying Beta-Binomial, we test whether overdispersion actually exists using Tarone's Z-statistic (Tarone 1979). This tests H0: "data are Binomial" vs H1: "data are Beta-Binomial." The test is run per party-chamber group, and results are recorded in the filtering manifest. Significant Z confirms that the Beta-Binomial model is justified over plain Binomial.

### Output Schema

The output parquet includes `votes_with_party` (integer y_i) and `prior_kappa` (alpha + beta, the effective prior sample size) alongside the existing columns. `prior_kappa` is directly interpretable as "how many pseudo-observations does the prior contribute?" — comparable to `n_party_votes`.

## Methodological Choices

### Empirical Bayes, not full MCMC

**Decision:** Use method of moments to estimate Beta(alpha, beta) hyperparameters from the data, then compute closed-form posteriors. Do not use PyMC.

**Rationale:** With ~170 legislators and a conjugate model, closed-form posteriors are exact and instant. MCMC would produce the same point estimates but with uncertainty on the hyperparameters — overkill for this exploratory phase. The roadmap reserves full hierarchical MCMC (item #3) for a separate, more ambitious model.

**Trade-off:** Empirical Bayes underestimates hyperparameter uncertainty. The credible intervals are slightly too narrow because they condition on the point-estimated prior. For formal inference, use full Bayes.

### Per-party-per-chamber priors (4 groups)

**Decision:** Estimate separate Beta priors for House Republicans, House Democrats, Senate Republicans, and Senate Democrats.

**Rationale:** These four groups have meaningfully different loyalty distributions. House Democrats (n≈38) may have different cohesion than Senate Democrats (n≈10). Pooling across chambers or parties would contaminate the prior.

**Risk:** Senate Democrats (n≈10) give an imprecise prior. The method of moments handles this gracefully — high variance triggers the Beta(1,1) fallback.

### Method of moments, not MLE

**Decision:** Use method of moments to estimate alpha and beta from the sample mean and sample variance (ddof=1) of observed rates.

**Rationale:** Simpler, faster, and more transparent than MLE. With the sample sizes we have (~40-90 legislators per group), the difference between MoM and MLE is negligible. MLE also requires optimization, adding complexity. Sample variance (ddof=1) is used per Casella (1985) and Gelman BDA3 Ch. 5 — population variance (ddof=0) underestimates by N/(N-1), which matters for small groups like Senate Democrats (N≈10). See ADR-0032.

### Shrinkage factor definition

**Definition:** `shrinkage = (alpha + beta) / (alpha + beta + n_i)`

This is the weight given to the prior in the posterior mean. Range: 0 (no shrinkage, n_i → ∞) to 1 (fully prior, n_i = 0). A shrinkage of 0.10 means the posterior is 90% data and 10% prior.

## Downstream Implications

- **Synthesis** can use `posterior_mean` as a more robust alternative to `unity_score` when discussing party loyalty. The `ci_width` column provides an uncertainty measure.
- **Profiles** can show credible intervals alongside point estimates to communicate confidence.
- **Cross-session comparison** should use posterior means (not raw rates) for legislators with few votes in one session.
- The `alpha_prior` and `beta_prior` columns in the output parquet enable reproducibility — anyone can verify the posterior computation.
