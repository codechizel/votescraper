# ADR-0055: Reparameterized LogNormal Beta and IRT Scale Linking

**Date:** 2026-02-28
**Status:** Accepted

## Context

The joint cross-chamber hierarchical IRT model fails convergence in all 8 bienniums (84th–91st). Three experiments revealed reinforcing root causes:

1. **Reflection mode multimodality.** With `beta ~ Normal(0, 1)`, each bill's discrimination can flip sign, creating ~365 independent reflection axes. Confirmed by the positive-beta experiment (ADR-0047): constraining beta > 0 fixes R-hat.

2. **LogNormal boundary geometry.** The ADR-0047 experiment used `pm.LogNormal(0, 0.5)` to constrain beta > 0. PyMC applies an internal log transform, creating catastrophic curvature near beta = 0: the distance between 0.01 and 0.001 in unconstrained space is 2.3 nats, vs 0.01 between 1.0 and 0.99. On the 84th biennium, divergences exploded from 10 to 2,041.

3. **Three-level hierarchy funnel.** Even with fixed sampling geometry, the chamber-level variance parameters (`sigma_chamber`, `sigma_party`) have only 2 and 4 observations respectively, creating poorly identified funnels. ESS for these hyperparameters collapses to 39-140.

Meanwhile, the per-chamber models converge perfectly (all 16 chamber-sessions pass with nutpie, ADR-0051/0053). The separate-then-link approach — well-converged per-chamber estimates plus a 2-parameter optimization — is the standard approach in psychometrics (Kolen and Brennan 2014) and is used by the most successful cross-chamber scaling methods in political science (Shor-McCarty, DW-NOMINATE Common Space).

## Decision

### 1. exp(Normal) Reparameterization (`lognormal_reparam`)

Replace `pm.LogNormal("beta", ...)` with an explicit reparameterization:

```python
log_beta = pm.Normal("log_beta", mu=0, sigma=1, shape=n_votes, dims="vote")
beta = pm.Deterministic("beta", pt.exp(log_beta), dims="vote")
```

This produces a mathematically identical distribution on beta (LogNormal(0, 1)) but the sampler works with `log_beta` — a smooth Gaussian with no boundary, no curvature explosion, no Jacobian. This is the Stan 2PL IRT tutorial approach.

Added as a new `lognormal_reparam` case in `BetaPriorSpec.build()`. The production joint model constant `JOINT_BETA = BetaPriorSpec("lognormal_reparam", {"mu": 0, "sigma": 1})` is defined in `model_spec.py`. Per-chamber models continue to use `PRODUCTION_BETA = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})` — they converge perfectly without the positive constraint.

The wider prior (sigma = 1.0 vs the ADR-0047 experiment's 0.5) gives 95% mass in [0.14, 7.39], accommodating bills with near-zero discrimination. The Stan User's Guide IRT section uses `lognormal(0.5, 1)` — sigma = 1.0.

### 2. Tighter Alpha Prior

The joint model's alpha (bill difficulty) prior tightened from `Normal(0, 5)` to `Normal(0, 2)`. Added as a configurable `alpha_sigma` parameter to `build_joint_graph()` and `build_joint_model()`, defaulting to 5.0 for backward compatibility. The production joint model call passes `alpha_sigma=2.0`. Difficulty parameters are typically in [-3, 3] for legislative votes; the wider prior added unnecessary posterior volume.

### 3. IRT Scale Linking (`irt_linking.py`)

New module `analysis/07_hierarchical/irt_linking.py` implementing four IRT linking methods for cross-chamber ideal point alignment:

- **Stocking-Lord**: Minimizes squared difference between test characteristic curves (TCCs) over a standard-normal-weighted theta grid. Most widely used in operational testing (Kolen and Brennan 2014).
- **Haebara**: Minimizes squared ICC differences per item. More robust than Stocking-Lord when individual anchor items are outliers.
- **Mean-Sigma**: Closed-form using SD ratio of difficulty parameters. This is essentially what the flat IRT's test equating uses.
- **Mean-Mean**: Closed-form using mean ratio of discrimination parameters. Simplest; sensitive to outliers.

All four methods find an affine transformation `xi_linked = A * xi_target + B` using shared (anchor) bills — the same bills identified by `_match_bills_across_chambers()`.

### 4. Sign-Aware Anchor Extraction

Per-chamber models use `beta ~ Normal(0, 1)`, so discrimination can be negative. Standard IRT linking assumes positive discrimination. `extract_anchor_params()` resolves this:

1. **Filter**: Exclude anchor items where the two chambers disagree on sign direction (a bill with positive beta in one chamber and negative in the other).
2. **Normalize**: For retained items, take `|beta|` and flip alpha accordingly (the ICC `a*theta - b` is invariant under `(a, b) → (-a, -b)`).

On the 84th biennium: 40 of 67 anchors usable (27 dropped for sign disagreement). All four methods agree perfectly on rank order (r = 1.000 pairwise).

### 5. Pipeline Integration

Stocking-Lord linking is integrated into `main()` in `hierarchical.py`. After per-chamber models converge, the pipeline:
1. Matches bills across chambers (reusing `_match_bills_across_chambers()`)
2. Runs all four linking methods as a sensitivity check
3. Reports sign-filtering diagnostics and linking coefficients
4. Saves linked ideal points to `hierarchical_ideal_points_linked.parquet`
5. Compares linked scores against the joint model (if available)

## Results (84th Biennium)

### Reparameterized Joint Model

| Metric | Baseline | LogNormal (ADR-0047) | Reparam |
|--------|----------|---------------------|---------|
| R-hat(xi) max | 1.54 | 1.22 | **1.010** |
| ESS(xi) min | 7 | 14 | **378** |
| Divergences | 10 | 2,041 | **828** |
| mu_chamber R-hat | — | — | 1.10 |
| Sign correction | Yes | No | **No** |

Major improvement but still failing: 828 divergences, hyperparameters unconverged.

### Stocking-Lord Linking

| Method | A | B |
|--------|---|---|
| Stocking-Lord | 1.03 | 0.69 |
| Haebara | 0.76 | 0.58 |
| Mean-Sigma | 0.67 | 0.63 |
| Mean-Mean | 0.86 | 0.58 |

All methods agree perfectly on rank order (r = 1.000). S-L linked vs joint: r = 0.967 (House), r = 0.894 (Senate).

## Consequences

**Positive:**
- The reparameterized LogNormal eliminates the boundary geometry catastrophe while preserving the reflection-mode fix
- IRT linking provides a robust, computationally cheap alternative to concurrent calibration for cross-chamber scaling
- Sign-aware anchor extraction handles the unconstrained per-chamber betas correctly
- Four linking methods as sensitivity check — if they agree, the linking is robust
- All 150 legislators placed on a common scale without requiring joint model convergence
- Linking infrastructure is reusable for any pair of IRT calibrations (future multi-state comparisons)

**Negative:**
- The joint model still fails convergence (828 divergences, hyperparameters unconverged) — the 3-level hierarchy's funnel geometry remains unsolved
- 40% of anchor items dropped due to sign disagreement — this may indicate genuine differential item functioning or just sign indeterminacy. A proper DIF analysis is needed.
- Linking assumes the shared items measure the same construct in both chambers (no DIF). This is weaker than concurrent calibration's joint estimation.
- Four linking methods disagree on the magnitude of A (0.67–1.03), though they agree on rank order. The "right" A depends on which method's assumptions best fit the data.

**Production path:** Stocking-Lord linking is the recommended production method for cross-chamber comparison. The joint model continues as an experimental benchmark.

## References

- Kolen, M.J. & Brennan, R.L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Kim, S. & Kolen, M.J. (2006). Robustness to format effects of IRT linking methods for mixed-format tests. *Applied Measurement in Education*, 19(4), 357-381.
- Holland, P.W. & Wainer, H. (1993). *Differential Item Functioning*. Lawrence Erlbaum.
- Shor, B. & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR*, 105(3), 530-551.
- Poole, K.T. & Rosenthal, H. (1991). Patterns of congressional voting. *AJPS*, 35(1), 228-278.
- Stan Development Team. 2PL IRT tutorial. https://mc-stan.org/users/documentation/case-studies/

## Related

- [ADR-0047](0047-positive-beta-constraint-experiment.md) — Positive beta experiment (motivation for reparameterization)
- [ADR-0048](0048-experiment-framework.md) — Experiment framework (BetaPriorSpec)
- [ADR-0043](0043-hierarchical-irt-bill-matching-and-adaptive-priors.md) — Bill-matching for shared items
- [ADR-0053](0053-nutpie-all-models.md) — nutpie for all MCMC models
- [Joint Model Deep Dive](../joint-model-deep-dive.md) — Full diagnosis and experiment results
- [Hierarchical Convergence Improvement](../hierarchical-convergence-improvement.md) — Convergence theory and improvement plan
