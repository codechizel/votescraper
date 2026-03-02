# ADR-0068: Dynamic IRT Post-Hoc Sign Correction

**Date:** 2026-03-01
**Status:** Accepted

## Context

Audit of run `260301.1` found the 87th biennium House ideal points were inverted (r = -0.937 with static IRT). Democrats showed positive xi, Republicans negative — the opposite of the expected convention.

**Root cause:** Positive beta prior (`HalfNormal(2.5)`) alone is insufficient for sign identification when the random walk chain is broken. The 88th biennium (2019-20) data was unavailable when the dynamic IRT ran (timing: it ran before the 88th pipeline completed), creating a gap with 0 bridge legislators on both sides of period 4. This severed the Markov chain, allowing the sampler to find a sign-flipped mode for the 87th.

The Senate was unaffected — 4-year staggered terms provide higher legislator continuity, maintaining sufficient bridge coverage even across the gap.

**Downstream consequences:** The sign flip caused the House random walk tau to inflate (D=1.46, R=0.81) to absorb the artificial ±3-unit jumps at the 87th boundary, producing scale drift where the dynamic/static range ratio grew from 0.66x (84th) to 1.51x (91st).

## Decision

Add a post-hoc per-period sign correction step between sampling and post-processing. For each time period with available static IRT data:

1. Compute dynamic xi posterior means for served legislators
2. Match to static IRT by normalized name
3. Compute Pearson r
4. If r < 0, negate `xi_post[:, :, t, :]` for that period

Pattern follows `fix_joint_sign_convention()` in hierarchical IRT (ADR-0042).

**Transparency:** Every correction is documented in the HTML report:
- Table of corrected periods with r before/after and number of matched legislators
- Named reference legislators for each corrected period (the 3 highest-|xi| matches) showing dynamic vs static values
- Full model priors and tuning parameters displayed in a dedicated report section
- Methodology section updated with identification caveat

**Corrected NetCDF:** The InferenceData is saved after correction, so all downstream consumers (trajectories, decomposition, movers, tau, plots) use the corrected posterior.

## Consequences

- **Positive:** Makes the model robust to broken chains from missing data. Transparent — every correction is named and documented.
- **Negative:** Relies on static IRT existing for each biennium (graceful skip if absent). Does not fix the root cause (missing data). Adds ~1s to runtime.
- **Future:** Re-running with 88th biennium data available should eliminate the sign flip at the root, making the post-hoc correction a no-op.
