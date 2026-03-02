# ADR-0074: Resolve 3 Systematic MCMC Convergence Failures

**Date:** 2026-03-02
**Status:** Accepted

## Context

Pipeline audit (ADR-0072) identified three systematic MCMC convergence failures across all 8 bienniums:

- **A2 (Joint model):** 256-4,281 divergences, sigma_chamber R-hat 1.14-2.56, ESS 5-39 in every biennium. Root cause: structurally over-specified (~1,000 params), partial reflections, Neal's funnel geometry. All targeted fixes applied (bill-matching ADR-0043, LogNormal reparameterization ADR-0055, PCA init) but insufficient — the model is fundamentally ill-conditioned.
- **A3 (2D IRT):** Senate ESS 6-52 (threshold 200) across all bienniums. Dim 2 captures noise, not signal (Dim2-vs-PC2 r = 0.12-0.82). Kansas voting is fundamentally 1D — this is a data limitation, not a model bug.
- **A4 (Dynamic IRT Senate):** R-hat 1.84, ESS 3 despite ADR-0070 fixes (informative prior, adaptive tau). Root cause: double-standardization bug in the informative prior — static IRT values (already approximately unit-scale) were globally re-standardized, destroying the per-legislator sign information and causing mode-splitting.

## Decision

### 1. Joint Model: Off by Default

Replace `--skip-joint` with `--run-joint` — the joint model is now opt-in. Stocking-Lord IRT linking (ADR-0055) is the production cross-chamber alignment method. It uses well-converged per-chamber posteriors and a simple optimization, sidestepping the joint model's fundamental identification problems.

The hierarchical report now includes a Stocking-Lord linking section (coefficients table + linked ideal points) whenever linking results are available.

### 2. 2D IRT: Removed from Pipeline

Phase 04b (`irt-2d`) removed from `just pipeline` and from the dashboard phase listing. The standalone `just irt-2d` recipe is preserved for research use. Phase 4c (PPC) already degrades gracefully when 2D IRT results are missing.

### 3. Dynamic IRT: Informative Prior Fixed

Three changes to `analysis/16_dynamic_irt/dynamic_irt.py`:

1. **Remove global re-standardization** of static IRT values. The prior construction now uses raw static IRT xi_mean values directly. These are already approximately unit-scale from the per-biennium IRT fits.
2. **Accumulator pattern** for multi-biennium averaging: instead of first-match-only, each legislator's prior mean is the average of their xi_mean across all bienniums where they appear.
3. **Widen prior sigma** from 0.75 to 1.5 (`XI_INIT_PRIOR_SIGMA` constant). The tighter sigma, combined with the double-standardized (near-zero) means, created an overly informative prior centered on the wrong values. The wider sigma lets the sampler explore while still transferring sign identification.

## Consequences

- Joint model remains available via `--run-joint` for research/experimentation
- Pipeline runs save ~4 min/biennium by not running the failing joint model
- 2D IRT standalone use unaffected; pipeline saves ~2-5 min/biennium
- Dynamic IRT Senate should now converge (prior correctly transfers sign information)
- PPC phase 4c gracefully handles missing 2D IRT (already tested)
- Stocking-Lord linking section in hierarchical report provides the cross-chamber comparison that the joint model was meant to deliver
