# ADR-0109: Canonical Ideal Point Routing

**Date:** 2026-03-11
**Status:** Accepted

## Context

Three attempts to fix the horseshoe effect in 1D IRT for supermajority chambers (Kansas Senate) all failed or introduced new problems:

1. **Init strategy (ADR-0107):** Using 1D IRT scores to initialize the 2D model poisoned it — the horseshoe-confounded scores contaminated the 2D posterior.
2. **Informative prior (ADR-0108):** Using 2D Dim 1 as a prior on the 1D model created a circular dependency — bad 1D → bad 2D → bad prior → bad 1D.
3. **PC2-targeted remediation (ADR-0104):** Filtering to PC2-dominant votes and applying a PC2 prior worked experimentally but was fragile and lost information.

The fundamental insight: the field solved this decades ago. DW-NOMINATE has always fit a 2D model and extracted Dimension 1 as the primary ideology score. We were trying to fix 1D when the correct approach is to use 2D Dim 1 as the canonical score for horseshoe-affected chambers.

## Decision

### Automatic per-chamber routing

After Phase 06 (2D IRT) completes, a routing function (`analysis/canonical_ideal_points.py`) automatically selects the best ideology score for each chamber:

| Condition | Canonical Source | Rationale |
|-----------|-----------------|-----------|
| Horseshoe detected + 2D converged | 2D IRT Dim 1 | DW-NOMINATE standard — separates ideology from establishment |
| Horseshoe detected + 2D failed | 1D IRT (fallback) | Better than nothing; flag in report |
| No horseshoe | 1D IRT | Simpler model is sufficient for balanced chambers |

### Horseshoe detection

Uses the same detection logic as Phase 05's `detect_horseshoe()`:
- Democrat wrong-side fraction > 20%
- Party overlap > 30%
- Any Republican more liberal than Democrat mean

### Output schema

Canonical output matches Phase 05's schema exactly (plus a `source` column), so downstream phases need zero column-level changes:

| Column | Type | Description |
|--------|------|-------------|
| `legislator_slug` | str | Unique identifier |
| `full_name` | str | Display name |
| `party` | str | Republican/Democrat/Independent |
| `xi_mean` | float | Posterior mean ideal point |
| `xi_sd` | float | Posterior SD (computed from HDI for 2D) |
| `xi_hdi_2.5` | float | 2.5% HDI bound |
| `xi_hdi_97.5` | float | 97.5% HDI bound |
| `source` | str | `"1d_irt"` or `"2d_dim1"` |

### Phase 06 init regression fix

Phase 06 now defaults to `--init-strategy pca-informed` instead of `auto`. The `auto` strategy prefers 1D IRT scores, which are horseshoe-confounded in exactly the chambers where the 2D model matters most. PCA is always safe for 2D initialization.

### Downstream integration

Synthesis (`analysis/24_synthesis/synthesis_data.py`) now checks for canonical routing output from Phase 06 before falling back to raw Phase 05 scores. The canonical output lives at `{phase_06_run_dir}/canonical_irt/`.

### Dim 1 forest plot

Phase 06 now generates a forest plot of Dim 1 ideal points per chamber — the visual counterpart to Phase 05's forest plot, but from the 2D model's ideology axis.

## Consequences

**Benefits:**
- Downstream phases automatically get the correct ideology score per chamber
- No manual configuration needed — horseshoe detection + convergence checks are automatic
- Follows the DW-NOMINATE standard (field-tested for 40+ years)
- 2D model's Dim 1 is always a correct ideology ranking, even for horseshoe-affected chambers
- Routing manifest records the decision for reproducibility

**Costs:**
- Phase 06 must run before synthesis (already required for `--promote-2d`)
- Adds ~10 seconds to Phase 06 (routing logic, not MCMC)

**Research flags retained:**
- `--dim1-prior` (ADR-0108), `--horseshoe-remediate` (ADR-0104), and `--init-strategy 2d-dim1` (ADR-0107) are retained for research but superseded for production pipelines

**Known limitation (ADR-0123):** Convergence and party-separation gates do not guarantee the selected dimension is the ideology axis. In 6/28 chamber-sessions, the hierarchical model's party-pooling prior forces party separation on a non-ideology dimension, producing a canonical source that disagrees with W-NOMINATE Dim 1 (r as low as 0.33). A W-NOMINATE cross-validation gate (ADR-0123) addresses this by checking whether an alternative IRT dimension better agrees with the unsupervised W-NOMINATE dimension ordering.

**Related:**
- `docs/canonical-ideal-points.md` — Full article documenting the journey
- `docs/canonical-ideal-points-implementation-plan.md` — Implementation plan
- ADR-0104 — Robustness flags (now research-only)
- ADR-0107 — Shared init strategy (Phase 06 default changed)
- ADR-0108 — Dim 1 informative prior (superseded)
- ADR-0123 — W-NOMINATE cross-validation gate (dimension correctness check)
