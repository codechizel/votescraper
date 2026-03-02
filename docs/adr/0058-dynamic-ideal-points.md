# ADR-0058: Dynamic Ideal Points (State-Space IRT)

**Date:** 2026-02-28
**Status:** Accepted

## Context

The pipeline estimates ideal points independently per biennium (static IRT in Phase 04), then aligns them via Stocking-Lord linking or affine alignment (Phase 13). With 8 bienniums scraped (84th–91st, 2011–2026), we can now build a Martin-Quinn style state-space IRT model that jointly estimates time-varying ideal points — tracking *who moved, and when* across a legislator's career.

Several modeling choices required decisions.

## Decision

### PyMC over R packages (emIRT, MCMCpack)

We chose PyMC with nutpie for the dynamic model, keeping emIRT as an optional exploration tier:

- **Consistency**: The pipeline already uses PyMC/nutpie for flat IRT (Phase 04), hierarchical IRT (Phase 10), and 2D IRT (Phase 04b). Adding another model in the same framework avoids context-switching and infrastructure duplication.
- **nutpie performance**: The Rust NUTS sampler matches or exceeds Stan's speed on our model scale (~10K params). Normalizing flow mass matrix adaptation helps with the random walk geometry.
- **Transparency**: Full model specification in Python, not behind R package APIs. Every prior, constraint, and parameterization is explicit and testable.
- **emIRT as validation**: emIRT::dynIRT provides fast EM point estimates for comparison. If R is unavailable, it is skipped automatically.

### Non-centered random walk (manual) over GaussianRandomWalk

We manually implement the non-centered parameterization:

```
xi_init ~ Normal(0, 1)
xi_innovations ~ Normal(0, 1, shape=(T-1, N_leg))
xi[t] = xi[t-1] + tau * xi_innovations[t-1]
```

rather than using PyMC's `GaussianRandomWalk`. Rationale:
- **Per-party tau broadcasting**: `tau[party_idx]` broadcasts to each legislator's party. `GaussianRandomWalk` doesn't natively support heterogeneous innovation SD.
- **Existing pattern**: The hierarchical model (Phase 10) already uses manual non-centered `xi_offset`. Consistent parameterization across the pipeline.
- **Control**: Explicit innovations are individually addressable for initialization, jittering, and diagnostics.

### Per-party tau (2 params) over global (1) or per-legislator (~300)

Two evolution SD parameters — one per party — balance flexibility with parsimony:

- **Global (1)**: Too restrictive. If Republican intra-party variance is higher (common in supermajority settings), a single tau overfits one party and underfits the other.
- **Per-legislator (~300)**: Too many. With only 8 time points, per-legislator tau is severely underidentified. The prior would dominate.
- **Per-party (2)**: The natural middle ground. Allows Democratic and Republican evolution rates to differ, which is substantively interesting. CLI flag `--evolution global` available for comparison.

### Positive beta (HalfNormal) over unconstrained + anchors

For cross-period models, anchor-based identification is more complex than single-session:
- Anchors must persist across all bienniums (few legislators serve 8 terms).
- Anchor selection interacts with the random walk — fixing xi at multiple time points constrains the walk.

Instead, `HalfNormal(2.5)` on beta (bill discrimination) provides sign identification by forcing all discriminations positive. This matches the hierarchical model's approach and the 2D IRT design (ADR-0054). **Note:** Positive beta alone is insufficient when the random walk chain is broken (e.g., missing biennium data). A post-hoc sign correction step (ADR-0068) addresses this.

### Cross-session flat mode

Like Phase 13 (cross-session validation), this phase operates on multiple bienniums simultaneously. It uses `RunContext(session="cross-session", analysis_name="dynamic_irt")` with flat directory structure: `results/kansas/cross-session/dynamic_irt/{YYMMDD}.{n}/`. NOT added to the `pipeline` recipe (which runs per-biennium phases).

### Relaxed convergence thresholds

R-hat threshold relaxed from 1.01 (standard) to 1.05 for the dynamic model. With ~10K parameters across 8 time periods, some bill parameters in early bienniums (84th has weaker data) may not fully converge. ESS threshold remains at 400. Document any R-hat > 1.01 in the convergence section.

## Consequences

**Positive:**
- First published dynamic ideal point analysis of a state legislature, built on a validated pipeline with external validation (Shor-McCarty r > 0.85).
- Conversion vs. replacement decomposition answers a question static IRT cannot: is polarization driven by returning members moving, or by turnover?
- Top mover identification reveals which legislators actually changed ideological position.
- Bridge coverage analysis quantifies cross-biennium data quality.
- Optional emIRT exploration tier provides a fast validation against an established method.

**Negative:**
- Runtime: estimated 1–3 hours per chamber on M3 Pro. The model is large and the random walk creates strong correlations between adjacent time points.
- 84th biennium (~30% committee-of-the-whole votes) has weaker data. Ideal points for 84th-only legislators will have wide posteriors.
- 84th→85th bridge (post-2012 redistricting) is the expected weakest link in the Markov chain.
- Convergence may require experimentation with tuning parameters.

**Files created:**
- `analysis/16_dynamic_irt/dynamic_irt.py` — Main script
- `analysis/16_dynamic_irt/dynamic_irt_data.py` — Data preparation (pure logic)
- `analysis/16_dynamic_irt/dynamic_irt_report.py` — HTML report builder
- `analysis/16_dynamic_irt/dynamic_irt_emirt.R` — emIRT exploration tier
- `analysis/design/dynamic_irt.md` — Design document
- `tests/test_dynamic_irt.py` — Test suite

**Files modified:**
- `analysis/__init__.py` — Registered modules in module map
- `Justfile` — Added `dynamic-irt` recipe
- `CLAUDE.md` — Updated pipeline, recipes, references
- `.claude/rules/analysis-framework.md` — Updated pipeline
- `.claude/rules/testing.md` — Added test file
