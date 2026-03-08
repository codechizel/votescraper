# IRT Identification Strategies

How Tallgrass resolves the reflection invariance problem in Bayesian IRT ideal point estimation.

## The Problem: Reflection Invariance

In a standard 2-parameter logistic (2PL) IRT model, the probability of legislator *i* voting Yea on bill *j* is:

```
P(Yea | xi_i, alpha_j, beta_j) = logit^-1(beta_j * xi_i - alpha_j)
```

The likelihood is invariant under sign reflection: negating all `xi` (ideal points) and all `beta` (discrimination parameters) simultaneously produces identical predicted vote probabilities. This means the posterior is bimodal — two mirror-image modes — and the sampler may converge to either one. Without constraints, "positive = conservative" and "positive = liberal" are equally valid solutions.

This is the **identification problem** in the IRT literature (Clinton, Jackman & Rivers 2004). Every IRT implementation must solve it. The choice of method is called the **identification strategy**.

## Why One Strategy Isn't Enough

The standard approach — fixing two anchor legislators at `xi = +1` (conservative) and `xi = -1` (liberal) based on PCA PC1 extremes — works well for balanced chambers. But Kansas has Republican supermajorities (~72% in recent sessions), and supermajority chambers produce a **horseshoe effect** in PCA:

- PC1 captures establishment-vs-rebel variation (the dominant axis within the majority party), not left-vs-right ideology.
- The most extreme PCA scores may belong to contrarian Republicans, not genuine ideological conservatives.
- Anchoring on these legislators drives the IRT model into a horseshoe-distorted solution where moderate Republicans appear more liberal than actual Democrats.

This was the **Huelskamp problem** (79th Legislature, 2001-02): Tim Huelskamp, a far-right Republican, was selected as the conservative anchor via PCA, but the horseshoe effect meant the anchor was actually capturing contrarianism. The resulting ideal points placed moderate Republicans far to the left of Democrats.

Tallgrass implements seven identification strategies and auto-selects the best one for each chamber's composition.

## The Seven Strategies

### 1. Anchor-PCA (`anchor-pca`)

**How it works:** Fix two anchor legislators at `xi = ±1`, selected as the most extreme Republican (highest PC1) and most extreme Democrat (lowest PC1), each with ≥50% vote participation.

**Literature:** Clinton, Jackman & Rivers (2004). The standard method in political science IRT.

**Strengths:**
- Simple, well-understood, widely used
- Works well for balanced (roughly 50/50) chambers

**Weaknesses:**
- Horseshoe effect in supermajority chambers distorts anchor selection
- Assumes PCA PC1 is a valid proxy for ideology (fails when intra-party variation dominates)

**When auto-selected:** Balanced chambers (no party holds ≥70% of seats) without external scores.

### 2. Anchor-Agreement (`anchor-agreement`)

**How it works:** Fix two anchor legislators at `xi = ±1`, selected by **cross-party contested vote agreement**. For each legislator, compute what fraction of contested votes (where both parties have ≥10% on each side) they agree with the opposite party's majority position. The most partisan Republican (lowest agreement with Democrats) and most partisan Democrat (lowest agreement with Republicans) become anchors.

**Literature:** Extends Clinton, Jackman & Rivers (2004) with a data-driven anchor selection criterion that is robust to the horseshoe effect. Developed for this project.

**Strengths:**
- Uses a metric external to the IRT model itself (no circular dependency)
- Immune to PCA distortion — directly measures bipartisan behavior
- Naturally selects genuine ideological extremes, not intra-party rebels

**Weaknesses:**
- Requires sufficient contested votes (≥10) and legislators with agreement data (≥6)
- Still pins two specific legislators, inheriting the conceptual limitation of hard anchors

**When auto-selected:** Supermajority chambers (≥70% single-party) with sufficient contested votes.

### 3. Sort Constraint (`sort-constraint`)

**How it works:** No individual anchors. All legislators are free parameters with `xi ~ Normal(0, 1)`. A `pm.Potential` applies a hard ordering constraint: if the mean of Republican ideal points falls below the mean of Democrat ideal points, a penalty of `-1e6` is applied to the log-posterior, effectively making that region of parameter space impossible.

**Literature:** Ordering constraints in factor analysis (Geweke & Zhou 1996). Applied to IRT as a sign identification alternative. Used in Tallgrass's own hierarchical model (Phase 07).

**Strengths:**
- No individual legislators are pinned — all are estimated freely
- Robust to supermajority distortion (constrains party means, not extremes)
- Simple to implement and understand

**Weaknesses:**
- Requires both parties to be present in the chamber
- Hard step function can cause sampling difficulties near the boundary
- Does not identify scale (only sign) — relies on the Normal(0,1) prior for scale

**When auto-selected:** Supermajority chambers where contested votes are insufficient for agreement-based anchors.

### 4. Positive Beta (`positive-beta`)

**How it works:** Force all discrimination parameters positive via `beta ~ HalfNormal(1)`. Since negating all `beta` and all `xi` yields the same likelihood, constraining `beta > 0` eliminates the reflection invariance entirely.

**Literature:** Stan User's Guide §1.11; standard in educational testing IRT. ADR-0047 documents the trade-off.

**Strengths:**
- Eliminates reflection invariance completely — no post-hoc correction needed
- No anchors or party information required

**Weaknesses:**
- **Silences D-Yea bills:** When the liberal position is Yea (e.g., expanding Medicaid), the true `beta` is negative. Forcing `beta > 0` misrepresents these bills, compressing their discrimination toward zero. In typical Kansas sessions, ~12.5% of roll calls have negative true discrimination.
- Biases ideal points for legislators who vote on many D-Yea bills

**When auto-selected:** Never (due to D-Yea bill limitation). Available as a manual override via `--identification positive-beta`.

### 5. Hierarchical Prior (`hierarchical-prior`)

**How it works:** Set party-informed priors on ideal points: `xi ~ Normal(+0.5, 1)` for Republicans, `xi ~ Normal(-0.5, 1)` for Democrats. No hard constraints. The prior provides soft identification — the data can overwhelm the prior if the evidence is strong enough.

**Literature:** Bafumi, Gelman, Park & Kaplan (2005). Hierarchical priors for ideal point models.

**Strengths:**
- Soft identification — doesn't force any specific structure on the posterior
- Both sign and approximate scale are identified by the prior
- Gracefully handles cases where a legislator's true ideology contradicts their party

**Weaknesses:**
- If the data is weak (few votes), the prior dominates and ideal points cluster near ±0.5
- Requires both parties present
- The prior magnitude (±0.5) is somewhat arbitrary

**When auto-selected:** Never (conservative design — used as a manual override). Could be promoted to auto-selection for chambers with weak party structure.

### 6. Unconstrained (`unconstrained`)

**How it works:** No identification constraints during MCMC. All parameters are free: `xi ~ Normal(0, 1)`, `beta ~ Normal(0, 1)`. After sampling, post-hoc sign correction via party means and `validate_sign()` resolves the reflection.

**Literature:** `pscl::postProcess()` in R (Jackman 2000). Post-hoc rotation of ideal point estimates.

**Strengths:**
- Fastest to implement
- No assumptions baked into the model
- If chains happen to converge to the correct mode, no distortion

**Weaknesses:**
- **Chains may not mix** in the bimodal posterior — each chain can converge to a different mode, producing spurious between-chain variation and inflated R-hat
- Relies entirely on post-hoc correction, which can fail if chain mixing is poor
- No scale identification (only sign correction)

**When auto-selected:** Never. Available as a manual override for diagnostic purposes.

### 7. External Prior (`external-prior`)

**How it works:** Use externally measured ideology scores as informative priors: `xi ~ Normal(sm_score, 0.5)`. Typically, Shor-McCarty scores from Phase 17 (external validation). The tight prior (sigma=0.5) provides strong identification of sign, location, and scale simultaneously.

**Literature:** Shor & McCarty (2011) for the external scores; Bonica & Woodruff (2015) for informative prior IRT.

**Strengths:**
- Solves sign, location, and scale simultaneously
- Places legislators on a nationally comparable scale
- Robust to all chamber compositions

**Weaknesses:**
- Requires external scores to be available (not all bienniums have Shor-McCarty coverage)
- Strong prior may suppress genuine within-session variation
- Tight sigma (0.5) means the data has limited ability to deviate from the prior

**When auto-selected:** When external scores are available (currently triggered by Phase 17 output).

## Auto-Detection Logic

When `--identification auto` (the default), Tallgrass selects a strategy based on chamber composition:

```
1. External scores available? → external-prior
2. Supermajority (≥70%) + sufficient contested votes? → anchor-agreement
3. Supermajority + insufficient contested votes + both parties? → sort-constraint
4. Balanced chamber → anchor-pca
```

The threshold for supermajority is 70% (`SUPERMAJORITY_THRESHOLD = 0.70`). "Sufficient contested votes" means ≥10 votes where both parties split ≥10% on each side, with ≥6 legislators having agreement data.

## Report Output

Every IRT run prints (and includes in the HTML report) a **strategy rationale table** showing all seven strategies with:

- Which strategy was selected (and whether by auto-detection or user override)
- Why each non-selected strategy was passed over
- Key metrics: supermajority fraction, contested vote count, party sizes

This ensures full transparency — a reader can see exactly why a particular identification method was chosen and what alternatives existed.

## CLI Usage

```bash
# Auto-detect (default)
just irt --session 2025-26

# Force a specific strategy
just irt --session 2001-02 --identification anchor-agreement
just irt --session 2025-26 --identification sort-constraint

# All options
just irt --identification {auto,anchor-pca,anchor-agreement,sort-constraint,positive-beta,hierarchical-prior,unconstrained,external-prior}
```

## Post-Hoc Safety Net: `validate_sign()`

Regardless of which identification strategy is used, every IRT run passes through `validate_sign()` as a final safety check. This function:

1. Computes cross-party contested vote agreement rates
2. Correlates Republican ideal points with their Democrat-agreement rate
3. If the correlation is positive (r > 0, p < 0.10), the sign is flipped — ideal points and discrimination parameters are negated

Correct sign → negative correlation (moderates agree more with the opposite party). Flipped sign → positive correlation (extremes agree more). This catches any residual sign errors that the primary strategy may have missed.

## Strategies Considered But Not Implemented

The literature describes additional identification strategies that were evaluated and excluded:

| Strategy | Why excluded |
|----------|-------------|
| Mean-zero constraint (xi sum = 0) | Solves location but not sign/scale |
| Variance-one constraint (xi var = 1) | Solves scale but not sign |
| Rotation to principal axis | Multivariate technique for 2D+ IRT only |
| Procrustes rotation | Requires reference configuration; not useful for 1D |
| Bridge anchoring (cross-session) | Used in Dynamic IRT (Phase 27), not per-session 1D |
| MCMC label switching moves | Complex to implement; sort constraint is simpler |
| Informative beta prior (mu=1) | Weaker than positive-beta, doesn't fully solve sign |
| Random anchor selection | Unreliable; dominated by PCA/agreement-based selection |
| Majority-party anchor only | Single anchor doesn't identify scale |
| Post-hoc rotation to DW-NOMINATE | External dependency; DW-NOMINATE not available for state legislatures |

These strategies are either inapplicable to 1D flat IRT, redundant with implemented strategies, or dominated by better alternatives.

## Key References

- **Clinton, J. D., Jackman, S., & Rivers, D.** (2004). "The Statistical Analysis of Roll Call Data." *American Political Science Review*, 98(2), 355-370.
- **Bafumi, J., Gelman, A., Park, D. K., & Kaplan, N.** (2005). "Practical Issues in Implementing and Understanding Bayesian Ideal Point Estimation." *Political Analysis*, 13(2), 171-187.
- **Shor, B., & McCarty, N.** (2011). "The Ideological Mapping of American Legislatures." *American Political Science Review*, 105(3), 530-551.
- **Morucci, M., et al.** (2024). "Bayesian Ideal Point Models: A Practical Guide." Working paper.
- **Jackman, S.** (2000). "Estimation and Inference via Bayesian Simulation: An Introduction to Markov Chain Monte Carlo." *American Journal of Political Science*, 44(2), 375-404.
- **Geweke, J., & Zhou, G.** (1996). "Measuring the Pricing Error of the Arbitrage Pricing Theory." *Review of Financial Studies*, 9(2), 557-587.
- **Bonica, A., & Woodruff, M. J.** (2015). "A Common-Space Measure of State Supreme Court Ideology." *Journal of Law, Economics, and Organization*, 31(3), 472-498.

## Related Documentation

- `analysis/design/irt.md` — IRT design document (priors, MCMC settings, convergence diagnostics)
- `docs/irt-sign-identification-deep-dive.md` — Deep dive on the sign flip problem
- `docs/79th-horseshoe-robustness-analysis.md` — Empirical robustness analysis of the 79th (horseshoe diagnostic, contested-only refit, 2D cross-reference)
- `docs/adr/0006-irt-anchor-selection.md` — Original anchor selection ADR
- `docs/adr/0047-positive-beta-tradeoff.md` — Positive beta trade-off analysis
- `docs/adr/0104-irt-robustness-flags.md` — Robustness flags system (CLI-driven diagnostics)
