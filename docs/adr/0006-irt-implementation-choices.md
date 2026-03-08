# ADR-0006: IRT Implementation Choices

**Date:** 2026-02-19
**Status:** Accepted

## Context

Bayesian IRT is the canonical baseline analysis per the analytic workflow rules: "1D Bayesian IRT on Yea/Nay only." PCA (Phase 2) confirmed strong 1D ideological structure (PC1 explains 53% House, 46% Senate). Several implementation choices needed to be resolved:

1. **Discrimination prior.** The standard IRT formulation allows discrimination (beta) to be positive or negative. The sign of beta determines whether conservative legislators (positive beta) or liberal legislators (negative beta) are more likely to vote Yea. A sign identification problem arises when the model is not anchored: flipping the sign of all beta and xi simultaneously leaves the likelihood unchanged.

2. **Identification method.** The 2PL IRT model is not identified without constraints (sign, location, and scale invariance). Multiple approaches exist: fix two legislators, order constraints, or soft identification via priors.

3. **Number of chains.** Standard recommendation is 4 chains, but IRT on ~170 legislators x ~400 votes is expensive. Trade-off between diagnostic rigor and runtime.

4. **Holdout validation approach.** True out-of-sample validation requires refitting the model on a subset of data. For IRT this means a second expensive MCMC run.

5. **Per-chamber vs. joint model.** Could estimate ideal points jointly across both chambers (using bridging legislators like Miller) or separately.

6. **Posterior storage format.** Full posterior distributions are large (~50-100MB per chamber). Need a standard, reloadable format.

## Decision

1. **Normal(0, 1) prior for discrimination (unconstrained).** Allows beta to be positive (conservative Yea) or negative (liberal Yea). Sign identification is provided by the hard anchors (Decision 2), making a positive constraint unnecessary. The original implementation used LogNormal(0.5, 0.5) following the standard recommendation for soft-identified models. However, investigation revealed this silenced 12.5% of bills (all D-Yea votes got beta ≈ 0, treated as uninformative). With `P(Yea) = logit⁻¹(β·ξ - α)` and β > 0, the probability of Yea always increases with conservatism — alpha shifts the threshold but cannot flip the direction. Switching to Normal(0, 1) improved holdout accuracy by +3.5%, ESS by 10×, and sampling speed by 18%, with zero sign-switching. See `analysis/design/beta_prior_investigation.md` for the full investigation and `docs/lessons-learned.md` Lesson 6.

2. **PCA-based anchor method (party-aware).** After PCA orients PC1 so Republican mean > Democrat mean, select the most extreme Republican (highest PC1) as conservative anchor at xi=+1 and the most extreme Democrat (lowest PC1) as liberal anchor at xi=-1. Anchors must have >= 50% participation to ensure tight estimation. This leverages the PCA results we already have and avoids requiring manual knowledge of Kansas legislators. The party-aware selection prevents sign flip in supermajority chambers where intra-party variation dominates PC1 and raw PC1 extremes may not correspond to ideological extremes — e.g., in the 79th Senate (30R/10D), the raw PC1 extreme was a far-right Republican rebel, not the most liberal Democrat. Falls back to raw PC1 extremes for single-party chambers. The alternative (soft identification via priors + post-hoc correction) was rejected because it requires checking sign orientation across chains and can produce label-switching artifacts. See `docs/irt-sign-identification-deep-dive.md` for a detailed analysis of dimension collapse in supermajority chambers.

3. **2 chains by default, sampled in parallel.** Sufficient for R-hat and ESS computation. With `cores=n_chains`, PyMC runs chains in parallel via multiprocessing (separate processes, not threads), so 2 chains complete in roughly the wall-clock time of 1 (~5-10 min per chamber). Each chain gets its own process, memory, and deterministic per-chain seed — results are mathematically identical to sequential execution. If convergence diagnostics show problems, `--n-chains 4` is available. This follows the PyMC documentation recommendation for computationally expensive models.

4. **In-sample holdout prediction.** Use posterior means from the full model to predict a random 20% of observed cells. This is documented as in-sample (the model saw all data during fitting). The proper Bayesian validation is the posterior predictive check (PPC), which samples from the full posterior and compares replicated data to observed. A true out-of-sample holdout would require refitting — available as a future enhancement but not worth the doubled runtime for the baseline.

5. **Per-chamber models + cross-chamber test equating.** House and Senate are first fitted independently as the primary per-chamber models. A classical test-equating transformation then links the scales using (a) shared bill discrimination ratios for the scale factor and (b) bridging legislators for the location shift. A full joint MCMC model was attempted but does not converge — 71 shared bills for 169 legislators is insufficient (R-hat > 1.7 despite 4 anchors and 4 chains). Test equating uses the already-converged per-chamber posteriors and is instantaneous. Per-chamber models remain the primary output; equated scores are for cross-chamber comparisons. Use `--skip-joint` to skip equating. See `analysis/design/irt.md` for details.

6. **ArviZ NetCDF for posterior storage.** `idata.to_netcdf()` produces a self-describing file that can be reloaded with `az.from_netcdf()` for downstream analysis (e.g., posterior predictive checks on new data, model comparison via LOO-CV). NetCDF is the ArviZ default and is supported by xarray for advanced slicing.

## Consequences

**Benefits:**
- Unconstrained Normal(0, 1) discrimination uses all contested bills — both R-Yea and D-Yea votes contribute to ideal point estimation. The sign of beta encodes direction (positive = conservative Yea, negative = liberal Yea), while the magnitude encodes how partisan the vote was.
- PCA-based anchors are automated and reproducible — no manual legislator selection required. They also provide the sign identification that makes the unconstrained beta prior safe.
- 2 parallel chains provide the same diagnostic coverage as sequential chains at roughly half the wall-clock time.
- In-sample holdout + PPC together provide comprehensive validation without the cost of refitting.
- Per-chamber models are simple, standard, and sufficient for the 1D baseline.
- NetCDF files enable full posterior reuse in downstream phases (clustering, network analysis).

**Trade-offs:**
- Unconstrained beta requires hard anchors for identification. If the anchor selection is wrong (e.g., PCA PC1 sign convention flipped), the model will be mis-identified. Post-hoc `validate_sign()` provides an additional safety net by checking cross-party contested vote agreement.
- PCA anchors assume PCA and IRT agree on who is most extreme within each party. In supermajority chambers, the horseshoe effect can cause the "most party-typical" Republican to be a moderate rather than the most conservative. Party-aware selection + `validate_sign()` together handle this: anchors come from the right parties, and post-hoc validation detects and corrects any remaining polarity inversion.
- 2 chains provide less multi-modal detection power than 4. If R-hat > 1.01 or ESS < 400, the user should re-run with `--n-chains 4`.
- In-sample holdout overstates predictive performance. This is clearly documented in the output and is supplemented by PPC.
- Test equating assumes shared bills have the same ideological content in both chambers. If bills are substantially amended between chambers, the equating is weakened.
- With only 3 bridging legislators, the location shift (B) has limited precision. A single outlier can shift the entire Senate scale.

**Revision history:**
- 2026-02-19: Initial decision — LogNormal(0.5, 0.5) discrimination prior.
- 2026-02-20: Changed to Normal(0, 1) after discovering the D-Yea blind spot. See `analysis/design/beta_prior_investigation.md`.
- 2026-02-20: Added joint cross-chamber model (Decision 5 updated from per-chamber-only to per-chamber + joint).
- 2026-02-20: Changed from joint MCMC to test equating after convergence failure. See `analysis/design/irt.md` "Why Not a Joint MCMC Model?"
- 2026-02-23: Added `cores=n_chains` for parallel chain sampling (Decision 3 updated). See ADR-0022.
- 2026-02-25: Removed `--sign-constraint` flag and associated dead code (experimental soft sign constraint via `pm.Potential`). Hard anchors + PCA init make it unnecessary; 0/16 chamber-sessions needed it. Added tail-ESS to convergence diagnostics per Vehtari et al. (2021). Extracted `N_CONVERGENCE_SUMMARY` constant. See `docs/irt-deep-dive.md` for the code audit that motivated these changes.
- 2026-03-06: Changed anchor selection from raw PC1 extremes to party-aware selection (most extreme R as conservative, most extreme D as liberal). Prevents sign flip in supermajority chambers where intra-party variation dominates PC1. See `docs/irt-sign-identification-deep-dive.md`.
- 2026-03-07: Added post-hoc `validate_sign()` step after MCMC sampling. Correlates cross-party contested vote agreement with ideal points to detect and correct horseshoe-effect sign flips that party-aware anchor selection alone cannot prevent. Negates xi and beta posteriors when a flip is detected. See ADR-0101 addendum and `docs/irt-sign-identification-deep-dive.md`.
