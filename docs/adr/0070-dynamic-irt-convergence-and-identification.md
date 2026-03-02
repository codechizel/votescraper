# ADR-0070: Dynamic IRT Convergence and Identification

**Date:** 2026-03-01
**Status:** Accepted

## Context

The full 8-biennium pipeline run (2026-03-01, run 260301.1) surfaced three issues:

1. **Symlink race** — Standalone phases overwrite the pipeline's `latest` symlink, breaking downstream consumers. When `just dynamic-irt` runs outside the pipeline, it auto-generates a `run_id` and overwrites the pipeline's `latest` symlink, pointing it to a directory that only contains Phase 16 output.

2. **87th sign flip** — The 87th House ideal points persistently invert (r = -0.937 with static IRT). `HalfNormal(2.5)` on beta constrains discrimination to be positive, but this alone is insufficient for sign identification when the 87th has few bridge legislators connecting it to well-identified periods.

3. **Senate convergence** — R-hat 1.84, ESS 3 (mode-splitting). Per-party tau estimated from ~70 Senate legislators is too weakly identified with the default `HalfNormal(0.5)` prior, causing the sampler to split between two modes.

Research and diagnosis documented in `docs/dynamic-irt-convergence-diagnosis.md`.

## Decision

### Decision 1: Guard `latest` symlink (pipeline-only updates)

`RunContext` now tracks `_explicit_run_id` (True when `run_id` is provided by the pipeline, False when auto-generated). Only explicit run IDs update the `latest` symlink and report convenience symlink. Auto-generated runs still create their run directory and write data — they just don't touch `latest`.

### Decision 2: Informative `xi_init` prior from static IRT

Load Phase 04 ideal points for each biennium, map to global roster via `normalize_name()` matching, standardize to unit scale, and use as prior mean: `Normal(xi_init_mu, 0.75)` instead of `Normal(0, 1)`. This transfers the well-identified sign convention from the per-biennium static IRT to the dynamic model's initial period.

### Decision 3: Adaptive tau + global tau for small chambers

- **Small chambers** (< 80 legislators, e.g., Senate): use `HalfNormal(0.15)` (tighter prior) and auto-switch from per-party to global tau. This prevents mode-splitting when per-party tau is estimated from too few data points.
- **Large chambers** (>= 80 legislators, e.g., House): keep `HalfNormal(0.5)` with per-party tau.
- MCMC budget increased from 1000/1000/2 to 2000/2000/4 for better R-hat diagnostics and ESS.
- `--tau-sigma` CLI flag for manual override.

### Post-hoc sign correction retained

The post-hoc sign correction (ADR-0068) is retained as a diagnostic safety net. With the informative prior, it should be a no-op. If it fires, that indicates the informative prior failed — a useful diagnostic signal.

## Consequences

**Positive:**
- Senate convergence should improve dramatically (from R-hat 1.84/ESS 3 to passing)
- 87th House sign flip should be eliminated by informative prior
- Standalone phases no longer break pipeline symlinks
- `--tau-sigma` allows experimentation without code changes

**Negative:**
- MCMC runtime roughly doubles (2x samples, 2x tune, 2x chains)
- Informative prior requires Phase 04 to have run first (graceful degradation: falls back to uninformative if static IRT unavailable)
- Small chamber auto-switch may mask genuine between-party evolution differences in the Senate

**Trade-offs:**
- `sigma=0.75` on the informative prior is a compromise: tight enough to transfer sign information, loose enough to allow the dynamic model to learn its own scale
- `SMALL_CHAMBER_THRESHOLD=80` chosen based on Kansas Senate (40 members) vs House (125 members); may need adjustment for other state legislatures
