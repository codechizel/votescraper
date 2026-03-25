# W-NOMINATE Common Space Ideal Points

**Phase 30 — Cross-temporal W-NOMINATE alignment for the Kansas Legislature (1999-2026)**

## Why a Second Common Space?

Phase 28 produces common-space ideal points using Bayesian IRT scores aligned via pairwise chain linking. Phase 30 does the same thing with W-NOMINATE scores. Why bother?

Three reasons:

1. **Field-standard comparability.** W-NOMINATE is the dominant scaling method in political science. DW-NOMINATE common-space scores from Voteview are the canonical reference for the U.S. Congress (Poole & Rosenthal 1997, updated through the 118th Congress). Researchers who want to compare Kansas ideal points with Congressional scores will expect W-NOMINATE-scaled output. Phase 30 speaks their language.

2. **Methodological robustness.** IRT and W-NOMINATE make different assumptions. Bayesian IRT (Phase 05/07) uses probit link functions with unbounded ideal points and MCMC-estimated posterior distributions. W-NOMINATE uses Gaussian utility functions with scores bounded to [-1, +1] and frequentist MLE. If both methods produce the same common-space rank ordering, our findings are robust to modeling choices. If they diverge, the divergence itself is informative — it reveals sessions where the ideological structure is ambiguous.

3. **The bounded scale is a feature, not a bug.** W-NOMINATE's [-1, +1] constraint forces extreme legislators toward the boundary. This compression is often criticized, but for cross-temporal comparison it provides a natural normalization — a "most extreme possible" anchor that IRT's unbounded scale lacks. When party polarization increases over time, IRT scores drift outward without bound while W-NOMINATE scores compress, revealing whether the *relative* positions changed or just the scale.

## How Common-Space Estimation Works

### The Bridge Legislator Approach

The fundamental insight behind all common-space methods (GLS 1999, DW-NOMINATE, Shor-McCarty) is the same: **legislators who serve in multiple sessions anchor the cross-temporal scale**. If Senator Haley served in both the 78th (1999-2000) and the 91st (2025-2026), his estimated ideal point in each session provides a calibration pair. With enough calibration pairs, you can estimate an affine transformation mapping one session's scale onto another's.

For Kansas, adjacent bienniums share 79-145 bridge legislators — far more than the minimum needed for stable alignment. Even the weakest link (84th→85th with 79 bridges) exceeds the 20-bridge threshold used in the literature.

### Pairwise Chain Linking

Rather than estimating all sessions simultaneously (which can produce degenerate solutions — see ADR-0120's revision history), Phase 30 uses the same **pairwise chain linking** approach as Phase 28:

1. Estimate an affine transform (A, B) for each adjacent session pair using trimmed OLS on bridge legislators
2. Compose the pairwise transforms into a chain: 78th → 79th → ... → 91st (reference)
3. The composed transform for session *t* is: `xi_common = A_total * xi_session + B_total`

This follows the IRT test equating literature (Battauz 2023, Kolen & Brennan 2014) and avoids the shrinkage-to-mean problem that plagued our initial simultaneous solver.

### Uncertainty Propagation

Each pairwise link has estimation uncertainty in (A, B). As transforms are chained, uncertainty accumulates — sessions farther from the reference have wider confidence intervals. Phase 30 uses the **delta method** (same as Phase 28) to propagate Var(A), Var(B), and Cov(A,B) through the chain, combining with W-NOMINATE's bootstrap standard errors via quadrature.

### The W-NOMINATE Bounded Scale

W-NOMINATE constrains ideal points to [-1, +1]. After affine transformation, scores can theoretically exceed these bounds. This is correct behavior — the common scale is no longer the per-session W-NOMINATE scale but a derived scale anchored to the reference session. The bounds apply to per-session estimation, not to the cross-temporal projection.

## Relationship to Other Methods

### DW-NOMINATE (Poole & Rosenthal)

The gold standard for Congressional common-space scores. DW-NOMINATE estimates all sessions simultaneously, constraining each legislator to a **linear career trajectory**: `x_i(t) = a_i + b_i * t`. This linear constraint is computationally efficient but cannot capture nonlinear change — if a legislator moderated after a primary challenge and then re-radicalized, the linear trend misses it.

Phase 30 does **not** assume linear trajectories. Each session's scores are linked independently, allowing arbitrary movement between sessions. The career score (computed via random-effects meta-analysis) captures the central tendency without forcing linearity. This is more flexible than DW-NOMINATE but less parsimonious.

The `dwnominate` R package (wmay/dwnominate on GitHub) wraps Poole's original FORTRAN code and supports multi-session estimation. We don't use it because: (a) it requires FORTRAN compilation, (b) the linear trajectory assumption is too rigid for Kansas's 28-year span including the Brownback disruption and 2012 Republican purge, and (c) our pairwise chain linking is already implemented and tested.

### Shor-McCarty State Legislature Scores

Boris Shor and Nolan McCarty's common-space scores for state legislatures use a **survey bridge** rather than legislator bridges. They regress within-state W-NOMINATE scores on responses to Project Vote Smart's National Political Awareness Test (NPAT), using the regression coefficients as an affine transformation to a common scale. This allows cross-state comparison but produces career-fixed scores with no within-career dynamics.

We validate Phase 28 IRT common-space scores against Shor-McCarty in Phase 17 (external validation). Phase 30 W-NOMINATE common-space scores provide an additional validation path — if our bridge-based W-NOMINATE alignment correlates with Shor-McCarty's survey-based alignment, the two independent bridge methods agree.

### Nokken-Poole Scores

A session-specific variant of DW-NOMINATE. Bill parameters and hyperparameters are fixed at common-space values, then legislator ideal points are re-estimated for each session independently. This produces maximum session-to-session volatility while maintaining a common scale. Our approach is similar in spirit — we estimate session-specific scores and then link them — but we use W-NOMINATE (not DW-NOMINATE) as the per-session estimator.

### Martin-Quinn Dynamic IRT

Used for the U.S. Supreme Court. A Bayesian IRT model with a random-walk prior on ideal points across time. Our Phase 27 (dynamic IRT) implements this approach for Kansas. Phase 30 is complementary — it provides a frequentist W-NOMINATE counterpart to the Bayesian IRT common space.

### GLS 1999 (Groseclose, Levitt & Snyder)

The methodological ancestor of our approach. GLS derived affine transformations for ADA interest-group scores using bridge legislators, revealing a strong liberal trend invisible in raw scores. Our Phase 28 and Phase 30 apply the same logic to IRT and W-NOMINATE ideal points respectively, with bootstrap uncertainty propagation that GLS did not include.

## What Phase 30 Produces

### Per-Session Common-Space Scores

Every legislator in every biennium receives a W-NOMINATE score on a common scale anchored to the 91st Legislature (2025-2026). Within-session rank order is preserved exactly (the affine transform is monotonic). Cross-session comparisons carry chain uncertainty that grows with distance from the reference.

### Career Scores

One number per legislator across their entire career, computed via DerSimonian-Laird random-effects meta-analysis of per-session common-space scores. The I² statistic measures heterogeneity — legislators with I² < 0.25 are "stable" (consistent ideology), while I² > 0.75 flags "movers" (genuine ideological change over their career).

### Cross-Method Validation

The headline output: how well do W-NOMINATE and IRT common-space career scores agree? Within each session, IRT and W-NOMINATE typically correlate at r > 0.98 (Phase 16). But do the *cross-temporal alignments* agree?

**Results:** Unified career scores correlate at r = 0.96 (Spearman ρ = 0.95) across 696 matched legislators. Within-party: Republicans r = 0.86 (n=501), Democrats r = 0.86 (n=195). The two methods tell the same story about Kansas ideology.

The validation report highlights the **top 25 legislators with the largest rank disagreements** between methods. These divergences typically reflect W-NOMINATE's bounded [-1, +1] scale compressing extreme legislators differently than IRT's unbounded scale, or sessions where the two methods weight different voting patterns. The full searchable comparison table covers all 696 legislators.

## Known Limitations

1. **W-NOMINATE SEs are often zero in our data.** The R `wnominate` package's parametric bootstrap SEs appear as zero in many parquet exports. Phase 30 uses a minimum SE floor (same approach as Phase 28) to avoid zero-variance issues in the meta-analysis.

2. **Bounded scale compression.** Legislators near the [-1, +1] boundary in W-NOMINATE have compressed scores that understate true ideological differences. The affine transformation partially corrects this (stretching the compressed dimension), but cannot recover discrimination lost at the boundary.

3. **84th Legislature outlier.** The 84th House (2011-2012) shows IRT↔W-NOMINATE correlation of only r = 0.62, far below the typical r > 0.98. This is the post-2012 Republican purge session with extreme supermajority dynamics. Both methods struggle here, and the linking coefficient for this session will have wider confidence intervals.

4. **Pre-2011 sessions lack OCD person IDs.** The 78th-83rd bienniums (KanFocus era) rely on slug-based person matching with manual overrides. This is the same limitation as Phase 28 and affects bridge identification, not the alignment algorithm itself.

## References

- Battauz, M. (2023). A general framework for chain equating under the IRT paradigm. *Psychometrika* 88(4): 1260-1287.
- Groseclose, T., Levitt, S. D., & Snyder, J. M. (1999). Comparing interest group scores across time and chambers: adjusted ADA scores for the U.S. Congress. *American Political Science Review* 93(1): 33-50.
- Kolen, M. J. & Brennan, R. L. (2014). *Test Equating, Scaling, and Linking* (3rd ed.). Springer.
- Lewis, J. B. & Sonnet, L. (2024). Penalized spline DW-NOMINATE. Working paper.
- Nokken, T. P. & Poole, K. T. (2004). Congressional party defection in American history. *Legislative Studies Quarterly* 29(4): 545-568.
- Poole, K. T. & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Shor, B. & McCarty, N. (2011). The ideological mapping of American legislatures. *American Political Science Review* 105(3): 530-551.
