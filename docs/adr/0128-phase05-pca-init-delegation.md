# ADR-0128: Phase 05 PCA Initialization Delegation

**Date:** 2026-03-26
**Status:** Accepted

## Context

Phase 05 flat IRT (`analysis/05_irt/irt.py`) hardcoded `["PC1"]` when initializing the MCMC sampler from PCA scores. This bypassed two critical systems:

1. **Manual PCA overrides** (`pca_overrides.yaml`) — 4 Senate sessions (79th, 80th, 82nd, 83rd) where PC2 is the ideology axis.
2. **Automated party-correlation detection** (`detect_ideology_pc()` in `init_strategy.py`) — catches axis swaps not covered by manual overrides.

Phase 07 hierarchical IRT already delegated correctly to `resolve_init_source()` with `session` and `chamber` parameters. Phase 05 only did this for the `--init-strategy 2d-dim1` path; all other strategies fell through to the hardcoded PC1 block.

### Impact

When the sampler initialized on the wrong PC, it could find a local mode with flipped signs. The post-hoc `validate_sign()` check failed to catch flips when contested votes were scarce (e.g., 91st Senate: 24 contested votes, Spearman r = +0.176 but p = 0.335 missed the p < 0.10 threshold).

Three sessions had flipped signs in Phase 05 flat IRT output:
- **79th Senate** (2001-2002) — PCA axis instability, PC2 is ideology
- **84th Senate** (2011-2012) — near-zero separation, sign essentially random
- **91st Senate** (2025-2026) — clean session (PC1 d=10.00) but sampler found wrong mode

Canonical routing (Phase 06) produced correct signs for all 28 chamber-sessions, so downstream phases (prediction, synthesis, profiles) were unaffected. The issue was cosmetic but misleading — Phase 05 forest plots and diagnostics showed inverted ideology.

## Decision

Replace the hardcoded PC1 extraction with a call to `resolve_init_source()`:

```python
ch_lower = chamber.lower()
init_vals, _, init_src = resolve_init_source(
    strategy="pca-informed",
    slugs=data["leg_slugs"],
    pca_scores=pca_scores,
    pca_column="PC1",
    session=args.session,
    chamber=ch_lower,
)
xi_init = init_vals[free_pos].astype(np.float64)
```

This mirrors the pattern already used by Phase 07 hierarchical IRT and the Phase 05 `2d-dim1` path.

## Consequences

**Fixes:** Phase 05 flat IRT now respects manual PCA overrides and automated PC detection for all identification strategies, not just `2d-dim1`. The 3 flipped sessions will produce correct signs on re-run.

**No new dependencies.** `resolve_init_source()` was already imported in `irt.py` (line 59).

**Behavioral change:** Sessions with PCA overrides (79th, 80th, 82nd, 83rd Senate) will now initialize from PC2 instead of PC1, likely improving convergence and eliminating the need for post-hoc sign correction in those chambers.

**Risk:** Minimal. The replacement function is well-tested (used by Phase 07 for all pipeline runs) and the `[free_pos]` indexing pattern matches the existing `2d-dim1` path.
