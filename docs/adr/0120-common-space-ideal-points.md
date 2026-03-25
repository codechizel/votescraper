# ADR-0120: Common Space Ideal Points

**Date:** 2026-03-24
**Status:** Accepted (revised: simultaneous → chained)
**Deciders:** Joseph Claeys

## Context

The pipeline produces per-biennium ideal points on independent scales. Each session's IRT model uses its own identification constraints — a score of +1.5 in the 79th Legislature does not mean the same thing as +1.5 in the 91st. This prevents cross-temporal comparison: we cannot ask whether a legislator from 2001 was more or less conservative than one from 2025, track polarization trends, or compare legislators who never served together.

The Kansas data has excellent bridge coverage (62-85% overlap between adjacent bienniums, three 28-year end-to-end legislators) and no term limits, making cross-temporal linking feasible.

## Decision

Implement Phase 28 (Common Space Ideal Points) using **pairwise chained affine alignment** of canonical ideal points via bridge legislators, with delta-method uncertainty propagation through the chain.

### Approach: Pairwise chain linking (Battauz 2023 / GLS 1999)

For each adjacent pair of bienniums, estimate an affine transformation (A, B) using trimmed OLS on bridge legislators' canonical ideal points. Chain the transformations backward from the reference session (91st) to the earliest available:

```
A_total(t→ref) = A(t→t+1) * A(t+1→t+2) * ... * A(n→ref)
B_total(t→ref) = follows the chain recursion
```

This is the standard approach in the test equating literature (Kolen & Brennan 2014, Battauz 2015/2023) and the legislative scaling literature (GLS 1999).

### Uncertainty propagation: Delta method through the chain

Two independent sources of uncertainty, combined in quadrature:

1. **IRT estimation uncertainty** (per-legislator posterior SD): `sd_irt = |A_total| * xi_sd`
2. **Alignment chain uncertainty** (from bootstrap on each link): propagated through the chain via the delta method, producing Var(A_total), Var(B_total), Cov(A_total, B_total) per session

Combined: `Var(xi_common) = A²·Var(xi) + xi²·Var(A) + Var(B) + 2·xi·Cov(A,B)`

This follows the `equateIRT` package methodology (Battauz 2015) and extends GLS 1999 by incorporating IRT posterior uncertainty (GLS treated raw scores as known).

### Key property: within-session comparisons are exact

For two legislators in the same session, the linking error cancels — both are transformed by the same (A, B). The comparison uncertainty reduces to `A² * (Var(xi_i) + Var(xi_j))`, which is just the scaled IRT uncertainty. Cross-session comparisons carry the full chain uncertainty.

### Input: Canonical ideal points

The phase consumes horseshoe-corrected canonical ideal points from the routing system (ADR-0109). For horseshoe sessions (primarily 78th-83rd Senate), this is Hierarchical 2D Dim 1; for clean sessions, it's flat 1D IRT. The common space phase does not estimate ideal points — it links existing ones.

## Revision: Simultaneous → Chained (2026-03-24)

The initial implementation used **simultaneous all-pairs alignment** — estimating (A, B) for all 13 non-reference sessions at once by minimizing total squared error across all bridge pairs. This produced degenerate solutions: A coefficients of 0.05-0.19 (House) and 0.005-0.024 (Senate), compressing non-reference sessions to a narrow range near the reference mean.

**Root cause:** The simultaneous least-squares solver minimizes total error by shrinking A toward zero and B toward the reference mean — a mathematically valid but substantively degenerate solution. With all-pairs observations, the solver can satisfy distant bridge pairs (which have the most noise) by collapsing both sessions to a constant, which dominates the loss function.

**Fix:** Switch to pairwise chain linking (adjacent pairs only). Each pairwise link has a well-conditioned design matrix (one session's scores regressed on another's for the same legislators) that cannot degenerate. Non-adjacent bridges are used for validation (cross-checking the chain), not estimation.

This aligns with the standard practice in:
- GLS 1999: pairwise linking between adjacent Congresses
- `equateIRT` (Battauz 2015, 2023): `chainec()` function chains pairwise links
- Kolen & Brennan 2014: Chapter 7 on chain equating

## Alternatives Considered

### Simultaneous all-pairs alignment (rejected after testing)

Estimate all (A, B) at once via least-squares on all bridge pairs. Appealing in theory (non-adjacent bridges should stabilize the solution), but produces degenerate coefficients in practice. The problem is ill-conditioned: the solver finds a trivial minimum by compressing all scales.

### Extended Dynamic IRT (state-space model to 14 bienniums)

Phase 27 already implements Martin-Quinn state-space IRT for 84th-91st. Extending to 78th-91st would provide smooth trajectories and full Bayesian uncertainty. However: (a) 78th-83rd lack canonical ideal points (need pipeline runs first), (b) 14 time periods with thousands of parameters would be computationally expensive, (c) the random walk assumption may not hold across structural breaks like the 2012 Republican purge. Dynamic IRT remains a complementary validation tool.

### Joint estimation (all sessions in one model)

Pool all 14 sessions into one giant IRT model with shared legislator parameters for bridge legislators. The most statistically principled approach but computationally infeasible (thousands of legislators, thousands of bills, 14 time periods).

### DW-NOMINATE linear trends

Constrain each legislator to a linear career trajectory. Fast but too rigid for a 28-year span that includes the Brownback disruption and the 2012 Republican purge.

## Cross-Chamber Unification (2026-03-25)

House and Senate are aligned independently across time (different bills, different IRT scales). After within-chamber alignment, the two scales are linked via 54 legislators who served in both chambers. An affine transform (A, B) maps Senate common-space scores onto the House scale, estimated via trimmed OLS on chamber-switchers' mean scores. This produces a unified scale for all 708 unique legislators.

Career scores are computed three ways:
- **Per-chamber** (`career_scores_house.parquet`, `career_scores_senate.parquet`): DerSimonian-Laird random-effects meta-analysis pooling per-session common-space scores within each chamber
- **Unified** (`career_scores_unified.parquet`): same RE meta-analysis but pooling across both chambers on the unified scale — one number per legislator

## Consequences

- Every Kansas legislator from 1999-2026 receives a score on a common ideological scale
- Cross-era comparison becomes possible (Huelskamp 2001 vs. any current legislator)
- Cross-chamber comparison enabled by chamber-switcher linking (54 bridges)
- 708 legislators get unified career scores — one number per person, regardless of which chamber(s) they served in
- Polarization trajectories show how party means evolved on the common scale
- Career trajectories show how individual legislators moved over their careers
- Confidence intervals honestly reflect chain distance from the reference biennium — earlier sessions have wider CIs
- Within-session rank order is preserved exactly (linking error cancels)
- Absolute ideological drift is undetectable (known limitation of all bridge-based methods)
- External validation via Shor-McCarty and DIME anchors the scale where coverage exists

## References

- Battauz, M. (2023). A general framework for chain equating under the IRT paradigm. *Psychometrika* 88(4): 1260-1287.
- Battauz, M. (2015). equateIRT: An R package for IRT test equating. *Journal of Statistical Software* 68(7): 1-22.
- Groseclose, T., Levitt, S. D., & Snyder, J. M. (1999). Comparing interest group scores across time and chambers. *APSR* 93(1): 33-50.
- Kolen, M. J., & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR* 105(3): 530-551.
- Martin, A. D., & Quinn, K. M. (2002). Dynamic ideal point estimation via MCMC for the U.S. Supreme Court. *Political Analysis* 10(2): 134-153.
- Bailey, M. A. (2007). Comparable preference estimates across time and institutions. *AJPS* 51(3): 433-448.
