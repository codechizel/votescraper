# ADR-0117: Hierarchical 2D IRT with PLT Identification and Party Pooling

**Date:** 2026-03-14
**Status:** Accepted

## Context

The pipeline has two complementary IRT models for handling the Kansas supermajority horseshoe problem:

- **Phase 06 (Flat 2D IRT)**: Resolves the horseshoe by separating ideology (Dim 1) from the establishment–contrarian axis (Dim 2). But Dim 2 has known convergence issues (R-hat up to 1.05, ESS ~200) due to weak signal (~11% variance). No party pooling means sparse legislators get unreliable estimates.

- **Phase 07 (Hierarchical 1D IRT)**: Party pooling regularizes sparse legislators and produces reliable 1D estimates. But the 1D model *can't* resolve the horseshoe — it conflates ideology with establishment loyalty in supermajority chambers.

Neither model alone solves both problems. A Hierarchical 2D IRT combines the strengths of both.

## Decision

Add Phase 07b: Hierarchical 2D IRT at `analysis/07b_hierarchical_2d/`.

### Model Specification

```
Party-level (per dimension):
  mu_party_dim1_raw ~ Normal(mu_07[p], 1.0)     ← informative from Phase 07
  mu_party_dim1 = sort(mu_party_dim1_raw)        ← D < R identification
  mu_party_dim2 ~ Normal(dim2_party_avg[p], 2.0) ← soft prior from Phase 06
  sigma_party_dim1 ~ HalfNormal(sigma_scale)     ← adaptive (small group = 0.5)
  sigma_party_dim2 ~ HalfNormal(sigma_scale)

Legislator (non-centered, per dimension):
  xi_offset_dim1 ~ Normal(0, 1)
  xi_offset_dim2 ~ Normal(0, 1)
  xi_dim1 = mu_party_dim1[party] + sigma_party_dim1[party] * xi_offset_dim1
  xi_dim2 = mu_party_dim2[party] + sigma_party_dim2[party] * xi_offset_dim2
  xi = stack([xi_dim1, xi_dim2], axis=1)

Bill parameters (PLT):
  alpha ~ Normal(0, 5)
  beta_col0 ~ Normal(0, 1)
  beta_col1: PLT-constrained ([0]=0, [1]=HalfNormal, rest=Normal)
  beta = stack([beta_col0, beta_col1], axis=1)

Likelihood (M2PL):
  eta = sum_d(beta[j,d] * xi[i,d]) - alpha[j]
  y ~ Bernoulli(logit(eta))
```

### Prior Chain

Phase 07 party means → Dim 1 mu prior (tight: sigma=1.0). Phase 06 Dim 2 party averages → Dim 2 mu prior (wide: sigma=2.0). When upstream output is missing, falls back to diffuse priors (sigma=2.0).

### Canonical Routing

Updated routing preference for horseshoe-affected chambers:
1. **Hierarchical 2D Dim 1** (if H2D converged at Tier 1 or 2)
2. Flat 2D Dim 1 (existing)
3. 1D IRT (fallback)

**Caveat (ADR-0123):** The party-pooling prior can force party separation on Dim 1 even when Dim 1 is not the ideology axis. In 6/28 chamber-sessions, H2D Dim 2 or another model better agrees with W-NOMINATE Dim 1. A W-NOMINATE cross-validation gate (ADR-0123) checks the selected dimension against W-NOMINATE and swaps if a better IRT dimension is available.

### PPC Integration

Phase 08 (PPC) discovers H2D as a 4th model in model comparison:
- Flat 1D, 2D IRT, Hierarchical 1D, **Hierarchical 2D**
- H2D uses 2D likelihood with hierarchical legislator filtering

## Consequences

**Positive:**
- Better Dim 2 convergence: party priors constrain the secondary axis
- Sparse legislator shrinkage in both dimensions
- Canonical routing can prefer the most regularized 2D estimate
- Clean integration with existing infrastructure (reuses `prepare_hierarchical_data`, `compute_beta_init_from_pca`, `resolve_init_source`)

**Negative:**
- Additional ~10-20 min sampling per biennium in the pipeline
- Model complexity: 4 IRT phases (1D flat, 2D flat, 1D hierarchical, 2D hierarchical) may be overkill for balanced chambers where 1D suffices
- Graceful degradation: H2D skips if either Phase 06 or Phase 07 output is missing

**Key gotchas:**
- `jitter_rvs` must exclude `xi_offset_dim1` and `xi_offset_dim2` when initialized, but *never* be `set()` (HalfNormal support point → log(0)=-inf)
- Supermajority tuning (ADR-0112): N_TUNE doubles to 4000 when majority > 70%
- Init strategy defaults to `pca-informed` (same as Phase 06) to avoid horseshoe contamination
