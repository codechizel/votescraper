# ADR-0059: W-NOMINATE + Optimal Classification Validation Phase

**Date:** 2026-02-28
**Status:** Accepted

## Context

W-NOMINATE (Poole & Rosenthal 1997) is the field-standard method for legislative scaling — virtually every published paper on congressional voting uses it. Our Bayesian IRT (Phase 04) produces near-identical ideal points (r ≈ 0.99 in the literature), but we could not previously make that claim with our own data. Optimal Classification (Poole 2000) adds a nonparametric robustness check at near-zero marginal cost.

Several design decisions were needed: (1) how to integrate R-only packages into a Python pipeline, (2) whether WNOM/OC should be separate or bundled, (3) whether this feeds downstream or is validation-only, and (4) phase numbering.

## Decision

### Validation-only role (no downstream consumers)

W-NOMINATE and OC results do NOT feed into synthesis, profiles, or any other downstream phase. Rationale:

- **IRT is our primary scaling method.** It produces full Bayesian posteriors, integrates with hierarchical and dynamic extensions, and has already been externally validated against Shor-McCarty.
- **Avoiding circularity.** If W-NOMINATE results fed into the pipeline, we could not independently use them as validation evidence.
- **Simplicity.** No new dependencies in the synthesis or profiles phases.

### W-NOMINATE + OC bundled (not separate phases)

Both methods use the same input (pscl rollcall object), same polarity legislator, and same R subprocess. Running them together:

- Adds ~10s to a ~30s R call.
- Eliminates duplicated data loading, conversion, and report infrastructure.
- Produces a single comparative report (3×3 correlation matrix: IRT/WNOM/OC).

OC failure is non-fatal — the report shows W-NOMINATE only.

### R subprocess with CSV I/O (not rpy2)

Matches the pattern established in Phase 16 (emIRT). Advantages over rpy2:

- No compilation against user's R installation.
- No shared library path issues on macOS.
- No ABI mismatch segfaults.
- CSV I/O overhead (~10ms) is negligible.
- Portable: any system with `Rscript` on PATH works.

### Phase 17 numbering

Follows the established convention: phases numbered sequentially after Phase 16 (Dynamic IRT). Phase 17 is standalone (not in the pipeline recipe) — same pattern as Phase 14 (External Validation) and Phase 16 (Dynamic IRT).

## Consequences

### Positive

- Pipeline can now claim "our IRT ideal points correlate at r=X with W-NOMINATE" — the single most credible validation statement in political science.
- OC provides a nonparametric robustness check at no marginal cost.
- R integration pattern (subprocess + CSV) is proven by Phase 16.

### Negative

- Requires R + four packages (`wnominate`, `oc`, `pscl`, `jsonlite`). These are not Python dependencies and must be installed separately.
- W-NOMINATE SEs are parametric bootstrap, not Bayesian posteriors — users may conflate them.
- Not in the pipeline recipe (`just pipeline`), so must be run manually.

### Neutral

- Phase 17 output directory follows run-directory layout (`results/<session>/<run_id>/16_wnominate/`).
- `just wnominate --session 2025-26` recipe added for convenience.
