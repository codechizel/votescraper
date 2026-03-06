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
- In supermajority chambers, the first IRT dimension may still capture establishment-vs-rebel rather than left-vs-right even with correct anchors — this is dimension collapse (a data feature, not a model failure). Huelskamp still appears at xi=-3.26 in the 79th Senate because his voting pattern genuinely places him there. See `docs/irt-sign-identification-deep-dive.md` for a detailed analysis.

**Validation:**
- 79th pipeline re-run: all convergence checks pass, PCA-IRT correlation r=0.94 (Senate), r=0.97 (House)
- Sensitivity analysis robust (r > 0.99 for both chambers)
- Hierarchical IRT (Phase 07) independently confirms the same pattern (flat-hierarchical r=0.978)
