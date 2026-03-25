# ADR-0110: Tiered Convergence Quality Gate for Canonical Routing

**Date:** 2026-03-11
**Status:** Accepted

## Context

ADR-0109 introduced canonical ideal point routing — auto-selecting 1D IRT or 2D Dim 1 per chamber based on horseshoe detection and convergence quality. The original quality gate required R-hat < 1.05 and ESS > 200 for the 2D model to be used as the canonical source.

Empirical evidence from 14 bienniums (78th–91st) shows this gate is too strict:

| Chamber pattern | Bienniums | Best R-hat | Best ESS |
|-----------------|-----------|------------|----------|
| Both converge well (< 1.10) | 83rd, 84th | 1.02 | 82–154 |
| House OK, Senate fails | 78th–82nd, 85th, 86th, 88th, 91st | 1.46–2.13 | 5–8 |
| Both struggle | 87th, 89th, 90th | 1.09–1.63 | 6–42 |

The Kansas Senate consistently fails the strict gate due to small chamber size (~40 members) combined with supermajority composition. However, domain experts confirmed that the 79th Senate's 2D Dim 1 ordering is ecologically valid despite R-hat ~2.0 — the forest plot shows the correct ideology ranking with honestly wide HDIs.

The key insight: **R-hat measures posterior mixing, not point estimate accuracy.** When chains find similar rankings at different scales (mode-splitting), the median point estimates can still be reliable. A rank correlation check against PCA PC1 validates this without requiring full convergence.

## Decision

### Three-tier quality gate

Replace the binary pass/fail with a tiered system:

| Tier | Name | R-hat | ESS | Extra Check | Canonical Action |
|------|------|-------|-----|-------------|------------------|
| 1 | **Converged** | < 1.10 | > 100 | — | Use 2D Dim 1 with full HDIs |
| 2 | **Point estimates credible** | < 2.50 | any | Spearman ρ(Dim1, PC1) > 0.70 | Use 2D Dim 1 point estimates; flag wide HDIs in report |
| 3 | **Failed** | ≥ 2.50 | — | or ρ < 0.70 | Fall back to 1D IRT |

### Tier 2 rank correlation check

When the 2D model doesn't fully converge but R-hat is below the catastrophic threshold (2.50), compute the Spearman rank correlation between 2D Dim 1 point estimates and PCA PC1 scores. If |ρ| > 0.70, the point estimates are capturing the primary ideological axis despite imperfect mixing.

This check requires PCA scores from Phase 02, which are always available when the pipeline runs in order.

### Constants

```python
# Tier 1: full convergence
TIER1_RHAT_THRESHOLD = 1.10
TIER1_ESS_THRESHOLD = 100

# Tier 2: point estimates credible
TIER2_RHAT_THRESHOLD = 2.50
TIER2_RANK_CORR_THRESHOLD = 0.70

# Tier 3: fall back to 1D (implicit — anything beyond Tier 2)
```

### Report integration

The routing manifest records the tier classification. The report displays:
- **Tier 1:** "Canonical source: 2D Dim 1 (converged)"
- **Tier 2:** "Canonical source: 2D Dim 1 (point estimates credible, ρ=X.XX, HDIs may be wide)"
- **Tier 3:** "Canonical source: 1D IRT (2D convergence insufficient)"

## Consequences

**Benefits:**
- The 79th Senate (and similar supermajority chambers) now correctly routes to 2D Dim 1, which separates ideology from establishment-loyalty
- Honest uncertainty: Tier 2 flags wide HDIs rather than hiding the convergence difficulty
- The rank correlation check is a principled validation — not just loosening thresholds arbitrarily
- Matches DW-NOMINATE's practical approach: use the point estimates even when the posterior is imperfect

**Costs:**
- Tier 2 results carry wider uncertainty than Tier 1 — downstream consumers should check the `source` column and routing manifest
- Requires PCA scores for the correlation check (always available in pipeline order)

**Expected tier distribution (based on 14 bienniums):**
- Tier 1: ~4 chamber-sessions (83rd both, 84th both, 88th House)
- Tier 2: ~15 chamber-sessions (most Senates, several Houses)
- Tier 3: rare (only if mode-splitting is so severe that rankings are meaningless)

**Known limitation (ADR-0123):** The tiered gate checks convergence quality and party separation, but does not validate whether the selected dimension is the correct ideology axis. The hierarchical model's party-pooling prior can create party separation on a non-ideology dimension, causing Tier 1 or Tier 2 to pass even when the dimension is wrong. A W-NOMINATE cross-validation gate (ADR-0123) adds dimension correctness checking as a final step after the convergence tiers.

**Related:**
- ADR-0109 — Canonical ideal point routing (original binary gate, now superseded by tiered gate)
- ADR-0054 — 2D IRT pipeline integration (relaxed convergence thresholds)
- ADR-0123 — W-NOMINATE cross-validation gate (dimension correctness check)
- `analysis/canonical_ideal_points.py` — Implementation
