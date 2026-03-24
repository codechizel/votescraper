# ADR-0120: Common Space Ideal Points

**Date:** 2026-03-24
**Status:** Accepted
**Deciders:** Joseph Claeys

## Context

The pipeline produces per-biennium ideal points on independent scales. Each session's IRT model uses its own identification constraints — a score of +1.5 in the 79th Legislature does not mean the same thing as +1.5 in the 91st. This prevents cross-temporal comparison: we cannot ask whether a legislator from 2001 was more or less conservative than one from 2025, track polarization trends, or compare legislators who never served together.

The Kansas data has excellent bridge coverage (62-85% overlap between adjacent bienniums, three 28-year end-to-end legislators) and no term limits, making cross-temporal linking feasible.

## Decision

Implement Phase 28 (Common Space Ideal Points) using **simultaneous affine alignment** of canonical ideal points via bridge legislators.

### Approach: Simultaneous alignment (GLS 1999 style)

Fix the 91st Legislature as the reference scale (A=1, B=0). For each other biennium, estimate an affine transformation (A, B) that minimizes discrepancies across all bridge pairs simultaneously — not just adjacent ones. This is a single over-determined least-squares problem with 26 unknowns (13 sessions × 2 parameters) and thousands of bridge observations.

Bootstrap resampling (N=1000) provides 95% confidence intervals on all parameters. Quality gates check party separation (Cohen's d >= 1.5) and sign consistency (R > D) per biennium.

### Input: Canonical ideal points

The phase consumes horseshoe-corrected canonical ideal points from the routing system (ADR-0109). For horseshoe sessions (primarily 78th-83rd Senate), this is Hierarchical 2D Dim 1; for clean sessions, it's flat 1D IRT. The common space phase does not estimate ideal points — it links existing ones.

## Alternatives Considered

### Sequential chaining

Link 78th→79th→80th→...→91st via pairwise affine regressions. Simpler but accumulates error — the 78th's scores pass through 13 transformations, each adding noise. Rejected because the 84th-85th gap (62.7%) would amplify error at the weakest link.

### Extended Dynamic IRT (state-space model to 14 bienniums)

Phase 27 already implements Martin-Quinn state-space IRT for 84th-91st. Extending to 78th-91st would provide smooth trajectories and full Bayesian uncertainty. However: (a) 78th-83rd lack canonical ideal points (need pipeline runs first), (b) 14 time periods with thousands of parameters would be computationally expensive, (c) the random walk assumption may not hold across structural breaks like the 2012 Republican purge. Dynamic IRT remains a complementary validation tool.

### Joint estimation (all sessions in one model)

Pool all 14 sessions into one giant IRT model with shared legislator parameters for bridge legislators. The most statistically principled approach but computationally infeasible (thousands of legislators, thousands of bills, 14 time periods).

### DW-NOMINATE linear trends

Constrain each legislator to a linear career trajectory. Fast but too rigid for a 28-year span that includes the Brownback disruption and the 2012 Republican purge.

## Consequences

- Every Kansas legislator from 1999-2026 receives a score on a common ideological scale
- Cross-era comparison becomes possible (Huelskamp 2001 vs. any current legislator)
- Polarization trajectories show how party means evolved on the common scale
- Career trajectories show how individual legislators moved over their careers
- Confidence intervals honestly reflect chain distance from the reference biennium
- Absolute ideological drift is undetectable (known limitation of all bridge-based methods)
- External validation via Shor-McCarty and DIME anchors the scale where coverage exists

## References

- Groseclose, T., Levitt, S. D., & Snyder, J. M. (1999). Comparing interest group scores across time and chambers. *APSR* 93(1): 33-50.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *APSR* 105(3): 530-551.
- Martin, A. D., & Quinn, K. M. (2002). Dynamic ideal point estimation via MCMC for the U.S. Supreme Court. *Political Analysis* 10(2): 134-153.
- Bailey, M. A. (2007). Comparable preference estimates across time and institutions. *AJPS* 51(3): 433-448.
