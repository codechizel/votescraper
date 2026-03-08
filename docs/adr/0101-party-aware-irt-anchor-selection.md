# ADR-0101: Party-Aware IRT Anchor Selection

**Date:** 2026-03-06
**Status:** Accepted

## Context

The flat IRT model (Phase 05) uses two anchors — a conservative legislator fixed at xi=+1 and a liberal legislator fixed at xi=-1 — to resolve sign, location, and scale indeterminacy in the 2PL model. Previously, `select_anchors()` picked the legislators with the highest and lowest raw PCA PC1 scores, regardless of party.

In the 79th Kansas Senate (2001-2002), this produced incorrect anchors. With 30 Republicans and 10 Democrats (75% supermajority), the dominant source of voting variation was *within* the Republican party — the moderate establishment (Praeger, Oleen, Vratil) versus conservative rebels (Huelskamp, Pugh, Lyon). The first principal component captured this establishment-vs-rebel axis rather than the traditional left-right axis (λ1/λ2 = 1.45, indicating near-equal dimensionality).

Tim Huelskamp — a staunch fiscal conservative who later served as a Freedom Caucus member in Congress — had the most extreme negative PC1 score (-31.0) because his voting pattern (opposing Republican establishment bills) closely resembled Democrat voting. Without party filtering, he would have been selected as the "liberal" anchor, fundamentally mis-identifying the model.

The PCA `orient_pc1()` function correctly ensures Republican mean > Democrat mean on PC1, so party-level sign is correct. But in supermajority chambers, the most extreme *individual* PC1 values don't correspond to ideological extremes — they correspond to anti-establishment extremes.

## Decision

Changed `select_anchors()` to **party-aware selection**:

1. **Conservative anchor**: Republican with the highest PC1 score (most party-typical Republican)
2. **Liberal anchor**: Democrat with the lowest PC1 score (most party-typical Democrat)
3. **Fallback**: If either party has no eligible legislators (single-party chamber), fall back to raw PC1 extremes with a warning

Both anchors must still meet the existing >= 50% participation guard.

This ensures anchors come from opposite parties and represent the ideological mainstream of each party, not the anti-establishment extreme.

## Consequences

**Benefits:**
- Correct anchor selection in supermajority chambers (79th Senate: Praeger R at +1.0, Haley D at -1.0 instead of Huelskamp as "liberal" anchor)
- No change in behavior for balanced chambers — in typical sessions, the most extreme R and most extreme D *are* the raw PC1 extremes
- Minimal code change (filtering by party column before sorting, with fallback)

**Trade-offs:**
- Requires the `party` column in PCA scores DataFrame (already present via `build_scores_df()`)
- In supermajority chambers, the "most party-typical R" selected by PCA is actually the most establishment-aligned moderate, not the most ideologically conservative. This produces a sign flip where the dimension's shape is correct but its polarity is inverted (e.g., Huelskamp at xi=-3.26). A post-hoc `validate_sign()` step detects and corrects this. See below.

**Validation:**
- 79th pipeline re-run: all convergence checks pass, PCA-IRT correlation r=0.94 (Senate), r=0.97 (House)
- Sensitivity analysis robust (r > 0.99 for both chambers)
- Hierarchical IRT (Phase 07) independently confirms the same pattern (flat-hierarchical r=0.978)

## Post-Hoc Sign Validation (2026-03-07 addendum)

**Problem:** Party-aware anchor selection prevents selecting a rebel as anchor, but in supermajority chambers the PCA horseshoe effect still causes the "most party-typical" Republican to be the most moderate/establishment-aligned, not the most conservative. The resulting anchor locks in a sign flip where the dimension's shape is correct but its polarity is wrong.

**Solution:** `validate_sign()` runs after MCMC sampling and checks whether the recovered ideal points have the correct polarity:

1. Identify contested votes (both parties split, ≥10% on each side per party)
2. For each Republican, compute agreement rate with the Democrat majority on contested votes
3. Spearman-correlate R agreement with R ideal points
4. Correct sign → negative correlation (moderates agree more with opposite party)
5. Flipped sign → positive correlation (extremes agree more) → negate xi, xi_free, beta

**Guard rails:** Skips with <3 legislators per party, <10 contested votes, <5 Rs with valid data, or when p ≥ 0.10.

**Impact:** Fixes the 79th Senate Huelskamp placement (xi=-3.26 → +3.26). Does not fire on correctly-signed sessions. 6 new tests. See `docs/irt-sign-identification-deep-dive.md`.

**Note:** This fix applies only to Phase 05 flat IRT. The hierarchical model (Phase 07) uses a sort constraint (D mean < R mean) that correctly identifies global sign but cannot prevent individual-level horseshoe placement — that is a dimension collapse issue that negation cannot fix.
