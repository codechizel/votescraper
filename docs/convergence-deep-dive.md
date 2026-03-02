# Convergence Deep Dive: Three MCMC Failures Across 8 Bienniums

**Date:** 2026-03-02
**Context:** Full pipeline audit (ADR-0072) identified three systematic convergence failures affecting all 8 bienniums (84th-91st). This article diagnoses each failure, documents root causes, and recommends resolution.

---

## Overview

The Tallgrass pipeline runs three MCMC models that require convergence diagnostics (R-hat < 1.01, ESS > 200, divergences < 0.1%):

| Model | Phase | Purpose | Status |
|-------|-------|---------|--------|
| Joint cross-chamber IRT | 10 (hierarchical) | Concurrent calibration of House + Senate on shared scale | **FAIL all 8** |
| 2D Bayesian IRT | 04b | Two-dimensional ideal points (PLT identification) | **FAIL all 8** |
| Dynamic IRT (Senate) | 16 | Martin-Quinn state-space ideal points across bienniums | **FAIL (regression)** |

All three failures are distinct in root cause but share a common lesson: **more model complexity does not improve inference when the data cannot support it.**

---

## A2: Joint Cross-Chamber IRT — Structural Over-Specification

### The Model

The joint hierarchical IRT attempts concurrent calibration: a single MCMC run estimates ideal points for both House and Senate legislators, using shared bill parameters (alpha, beta) for bills voted on by both chambers. This places both chambers on a common ideological scale without post-hoc linking.

### Convergence Evidence

| Biennium | Divergences | R-hat max | ESS min | Status |
|----------|------------|-----------|---------|--------|
| 84th | 2,041 | 1.63 | 7 | FAIL |
| 85th | 4,281 | 2.28 | 5 | CATASTROPHIC |
| 86th | 1,117 | 1.54 | 7 | FAIL |
| 87th | 2,959 | 2.63 | 5 | CATASTROPHIC |
| 89th | 639 | 1.69 | 6 | FAIL |
| 91st | 256 | 1.58 | 7 | FAIL |

The worst-performing parameters are consistently `group_offset_raw` (Senate party offsets), `mu_global`, and `sigma_chamber` — the top-level hyperparameters of the 3-level hierarchy.

### Root Cause: Three Reinforcing Problems

**Problem 1: Partial Reflection Modes.** Each of ~420 bills can independently flip sign (beta to -beta with compensating xi adjustment). The `exp(Normal)` reparameterization (`JOINT_BETA`) breaks global reflection but not partial reflections. With 420 independent reflection axes, the posterior has an astronomical number of near-equivalent modes that NUTS cannot bridge.

**Problem 2: High-Dimensional Slow Mixing.** The joint model has ~1,042 free parameters (vs. ~600-700 per chamber). NUTS mixing scales as O(d^{1/4}), so a 1,000-dimensional model mixes ~3.2x slower than a 600-dimensional one. The block-diagonal vote matrix (83% chamber-specific votes) and ~50% structural missingness (each legislator only votes in one chamber) compound the problem.

**Problem 3: Neal's Funnel.** The 3-level hierarchy (global → chamber → party → legislator) has only 2 chamber offsets and 4 group offsets informing the top-level variance parameters. `sigma_chamber` (ESS = 25 in the best case) creates extreme correlations with the ~365 bill parameters it governs. This is the classic funnel geometry where top-level variance oscillates between the few-parameter outer hierarchy and the high-dimensional inner parameter space. Non-centered parameterization helps for xi but cannot address the chamber/party levels.

### Fixes Applied (All Insufficient)

All three ADR fixes are correctly implemented in the current code:

1. **Bill matching (ADR-0043):** `_match_bills_across_chambers()` correctly finds 71-174 shared bills per session and maps both chambers' vote_ids to shared indices. Applied via the `rollcalls` parameter to `build_joint_model()`.

2. **LogNormal reparameterization:** `JOINT_BETA` uses `exp(Normal(0,1))` to eliminate boundary geometry. Mathematically sound but doesn't prevent partial reflections.

3. **PCA initialization + group-size-adaptive priors:** Both applied. Help but insufficient.

The bill-matching fix improved the model from "completely broken" (zero shared items) to "less broken" (256+ divergences with shared items). The three structural problems remain.

### The Alternative: Stocking-Lord Linking (Already Working)

`analysis/10_hierarchical/irt_linking.py` implements four IRT linking methods:

- **Stocking-Lord** (primary): TCC-based test characteristic curve matching
- **Haebara**: ICC-based item characteristic curve matching
- **Mean-Sigma / Mean-Mean**: closed-form methods using moment ratios

Sign-aware anchor extraction (lines 28-95) filters bills where chambers disagree on discrimination sign (~40% dropped), normalizes to positive discrimination, and preserves ICC invariance. All four methods produce rank-order identical results (pairwise r = 1.000).

The linking pipeline is already integrated into `hierarchical.py` (lines 1859-1986): it runs after per-chamber models, produces linked ideal points, and exports to `hierarchical_ideal_points_linked.parquet`. Per-chamber models converge perfectly (R-hat < 1.01, ESS > 400) and validate at r = 0.95-0.98 against Shor-McCarty external benchmarks.

### Diagnosis

The joint model is structurally over-specified for Kansas data. With only 17% shared bills, concurrent calibration cannot provide enough cross-chamber signal to overcome the 1,000-parameter posterior's complexity. This is not fixable by tuning priors, samplers, or parameterizations — the model asks more of the data than the data can provide.

### Recommendation: Drop Joint Model, Adopt Stocking-Lord

| Criterion | Joint Model | Stocking-Lord Linking |
|-----------|-------------|----------------------|
| Convergence | FAILS (all 8 bienniums) | Per-chamber models converge perfectly |
| Sampling time | 30-90 min (4 chains) | <5 min (2-parameter optimization) |
| Transparency | Black-box MCMC | Explicit scaling coefficients (A, B) |
| Literature | Fox (2010) — fragile with <20% shared items | Kolen & Brennan (2014) — 30 years of operational testing |
| External validation | Unknown (estimates questionable) | Per-chamber r = 0.95-0.98 vs Shor-McCarty |

**Action:** Remove joint model sampling from the production pipeline. Retain `build_joint_graph()` for experimental use. Make Stocking-Lord linking the sole production method for cross-chamber comparison.

---

## A3: 2D Bayesian IRT — Signal That Isn't There

### The Model

Phase 04b implements a multidimensional 2-parameter logistic (M2PL) IRT model with PLT identification (constraining a 2×2 submatrix of discriminations to fix rotation and reflection). The motivation was the "Tyson paradox" — a Republican who votes with Democrats on high-discrimination bills but against his own party on low-discrimination bills.

### Convergence Evidence

| Biennium | Senate ESS | House ESS | Senate R-hat | House R-hat |
|----------|-----------|----------|-------------|------------|
| 84th | 153.8 | 124.3 | 1.04 | 1.02 |
| 85th | 56.7 | 6.2 | 1.03 | 1.71 |
| 86th | 30.0 | 173.6 | 1.11 | 1.03 |
| 87th | 47.1 | 7.3 | 1.05 | 1.55 |
| 88th | 8.7 | 194.3 | 1.38 | 1.01 |
| 89th | 6.2 | 54.9 | 1.73 | 1.06 |
| 90th | 52.1 | 6.3 | 1.08 | 1.70 |
| 91st | 8.4 | 55.7 | 1.40 | 1.04 |

Senate never exceeds ESS = 154. House alternates between marginal and catastrophic failure. No session passes both thresholds for both chambers.

### Root Cause: Kansas Voting Is One-Dimensional

The second dimension has no empirical support in Kansas legislative data. This is a data limitation, not a model or sampler bug.

**Evidence 1: Variance decomposition.** In the 91st Senate, Dim 1 explains 0.605 variance units; Dim 2 explains 0.051 (8.4% of Dim 1). The orthogonal direction is noise.

**Evidence 2: HDI width.** 100% of senators (42/42) have Dim 2 HDI intervals that include zero. Average Dim 2 HDI width is 5.46 (394% of standard deviation). The posterior says: "we have no idea whether Dim 2 is positive or negative for anyone."

**Evidence 3: Unstable correlations with PCA PC2.**

| Biennium | Senate Dim2↔PC2 | House Dim2↔PC2 |
|----------|-----------------|----------------|
| 84th | 0.31 | 0.73 |
| 85th | 0.82 | 0.32 |
| 86th | 0.85 | 0.95 |
| 87th | 0.82 | 0.82 |
| 88th | -0.22 | 0.69 |
| 89th | 0.82 | 0.36 |
| 90th | 0.84 | 0.12 |
| 91st | 0.21 | 0.47 |

If Dim 2 captured real structure, correlations would be strong and stable across sessions. Instead they range from -0.22 to 0.95 — the hallmark of fitting noise.

**Evidence 4: Political science context.** Even in Congress, the second dimension has collapsed below 5% of variance since the 1990s (Poole 2005). Kansas has a 72% Republican supermajority with high party discipline. The only "cross-cutting" variation is 3-5 individual dissenters (Tyson, Thompson, Peck) — idiosyncrasy, not a structural dimension.

### Why Fixes Won't Work

| Approach | Why It Fails |
|----------|-------------|
| Stronger Dim 2 priors | Shrinks uncertainty without creating signal |
| More MCMC iterations | Can't resolve a non-existent parameter |
| Different sampler | Same posterior, same problem |
| Procrustes rotation to PCA | PCA and IRT optimize different objectives |
| Q-matrix constraints | Requires bill-text metadata we don't have |

The model is correctly implemented (PLT identification is textbook). nutpie sampling is reliable. The problem is fundamental: **a well-specified 2D model applied to 1D data will show non-identification on the second dimension.** This is correct behavior.

### The Tyson Paradox Is Already Explained

The original motivation (Tyson voting with Democrats on high-discrimination bills) is documented in the analytic flags and the 1D IRT report. The design doc already states: "The 2D model does NOT replace the 1D model... All downstream phases continue to use 1D ideal points." Phase 04b was always experimental.

### Recommendation: Drop Phase 04b

**Action:** Remove 2D IRT from the production pipeline. The model is sound but the data doesn't support it. Phase 04b consumes 20-40 minutes per biennium with no usable output. Keep the code in `analysis/04b_irt_2d/` for reference and document the negative finding.

---

## A4: Dynamic IRT Senate — A Fixable Regression

### The Model

Phase 16 implements a Martin-Quinn state-space IRT model: ideal points evolve via random walk across 8 bienniums (84th-91st). Non-centered parameterization with per-party evolution SD (tau). Bridge legislators (those serving in consecutive bienniums) link periods.

### Convergence Evidence: Before and After ADR-0070

| Metric | Run 260301.1 (before fixes) | Run 260301.2 (after fixes) | Change |
|--------|---------------------------|---------------------------|--------|
| Senate R-hat | 1.01 | 1.84 | +83% (worse) |
| Senate ESS bulk | 322 | 3 | -99% (catastrophic) |
| Senate ESS tail | 449 | 42 | -91% (worse) |
| House R-hat | 1.02 | 1.53 | +50% (worse) |
| House ESS bulk | 710 | 7 | -99% (worse) |

**The fixes caused the regression.** The pre-fix model was borderline acceptable (R-hat 1.01, ESS 322 — slow mixing in a single mode). The post-fix model exhibits mode-splitting (R-hat 1.84, ESS 3 — chains locked in different modes).

### Root Cause: Double-Standardization Bug in Informative Prior

ADR-0070 introduced an informative `xi_init` prior that loads static IRT ideal points from Phase 04 to anchor the sign convention. The implementation has a critical bug:

```python
# Step 1: Load static IRT values (already standardized within each chamber)
for t_idx, static_df in all_static_irt.items():
    for row in static_df.iter_rows(named=True):
        if nn in name_to_global:
            xi_init_mu_arr[gidx] = row[xi_col]  # Already unit-scale

# Step 2: Re-standardize GLOBALLY (double-standardization!)
nz = xi_init_mu_arr[xi_init_mu_arr != 0]
xi_init_mu_arr = (xi_init_mu_arr - nz.mean()) / max(nz.std(), 1e-6)
```

Static IRT values are already standardized to unit scale within each chamber during Phase 04. The global re-standardization rescales them a second time, misaligning the prior with the Senate's learned scale. This creates a bimodal posterior:

- **Mode A:** Chains follow the biased informative prior (wrong scale, low variance)
- **Mode B:** Chains reject the prior and follow the likelihood (correct scale, poor prior fit)

Each chain locks into one mode. R-hat measures between-chain disagreement → 1.84. ESS measures effective independent samples → 3.

### Secondary Issues

**Single-period initialization:** The code only uses the first biennium's (84th) static IRT to initialize `xi_init`. Legislators not in period 0 get `xi_init_mu = 0` (uninformative default). For Senate (~40 legislators per period), this wastes information for late-arriving legislators.

**Tight tau interaction:** The adaptive tau prior (`HalfNormal(0.15)` for Senate) is reasonable in isolation but destructive with a biased prior. A constrained tau (97.5th percentile ≈ 0.45) leaves no room for the random walk to escape the prior's wrong scale, amplifying mode-splitting.

### Bridge Coverage Is Fine

Senate bridge coverage exceeds 50% at every transition (20-23 shared legislators out of ~40 total). This meets or exceeds literature minimums (5-10%). The problem is not insufficient bridging — it's the biased prior.

### Sign-Flip Artifacts (87th/88th)

The 87th House sign flip (r = -0.937 with static IRT) is a real identification issue separate from the mode-splitting. The random walk can find a reflected mode because the positive beta constraint alone is insufficient for global sign identification across 8 periods. Post-hoc sign correction (ADR-0068) handles this correctly and transparently.

### Recommendation: Fix the Bug, Don't Drop the Phase

Unlike the joint model and 2D IRT, the dynamic IRT is **fundamentally sound** — it was working before ADR-0070's implementation bugs. The fix is clear:

**Immediate:** Revert the informative prior. Return to `xi_init ~ Normal(0, 1)` with per-party `tau ~ HalfNormal(0.5)`. Accept ESS ≈ 322 for Senate as exploratory-grade. Post-hoc sign correction handles sign flips.

**Better fix:** Remove the double-standardization. Use static IRT values directly (no global re-standardization) with a weaker prior (sigma=1.5 instead of 0.75). This preserves the sign-anchoring benefit without creating mode-splitting.

**Best fix:** Add hard anchor constraints for the longest-serving extreme legislators (mirrors MCMCpack's `theta.constraints`). This fixes sign at the model level rather than relying on post-hoc correction.

---

## Summary of Recommendations

| Issue | Model | Recommendation | Rationale |
|-------|-------|---------------|-----------|
| A2 | Joint cross-chamber IRT | **Drop** from pipeline | Structurally over-specified; Stocking-Lord linking is the proven alternative |
| A3 | 2D Bayesian IRT | **Drop** from pipeline | Kansas data is 1D; non-identification is unfixable |
| A4 | Dynamic IRT (Senate) | **Fix** the informative prior | Regression caused by implementation bug, not model flaw |

### Estimated Impact

- **Pipeline runtime reduction:** ~35-45 min per biennium (joint model 30-90 min + 2D IRT 20-40 min)
- **Full 8-biennium savings:** ~5-6 hours of eliminated wasted computation
- **Reduced false confidence:** No more "relaxed thresholds" masking non-converged results
- **Dynamic IRT Senate fix:** ESS 3 → ~322+ (100x improvement) with prior revert

### Common Thread

All three failures illustrate the same principle: **model complexity must be justified by data complexity.** Kansas legislative voting is dominated by a single partisan dimension with high party discipline. Models that assume richer structure (two dimensions, three hierarchical levels, biased priors) fail because the data cannot distinguish their additional parameters from noise. The pipeline's strength lies in its simpler, well-identified models: 1D IRT, per-chamber hierarchical IRT, and Stocking-Lord linking.

---

## References

- ADR-0042: Joint model convergence diagnosis
- ADR-0043: Bill-matching fix for cross-chamber identification
- ADR-0046: 2D IRT experimental phase
- ADR-0054: 2D IRT pipeline integration with relaxed thresholds
- ADR-0058: Dynamic IRT (Martin-Quinn state-space)
- ADR-0068: Post-hoc sign correction
- ADR-0070: Dynamic IRT convergence fixes (introduced regression)
- ADR-0072: Pipeline audit findings
- Fox, J.-P. (2010). *Bayesian Item Response Modeling*
- Kolen, M. J. & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking*
- Poole, K. T. (2005). *Spatial Models of Parliamentary Voting*
