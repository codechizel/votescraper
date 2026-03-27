# IRT Design Choices

**Script:** `analysis/05_irt/irt.py`
**Constants defined at:** `analysis/05_irt/irt.py:181-205`
**ADR:** `docs/adr/0006-irt-implementation-choices.md`

## Assumptions

1. **Unidimensional ideology.** The 2PL model assumes a single latent dimension explains all voting behavior. Legislators who deviate on a second dimension (e.g., Tyson's contrarianism on routine bills) will have their 1D ideal point estimated as a compromise between their positions on both dimensions. This is by design for the canonical baseline. An experimental 2D IRT model (`analysis/experimental/irt_2d_experiment.py`) confirms the second dimension exists (Dim 2 vs PCA PC2 r=0.81) but is noisy for most legislators — only Tyson, Thompson, and Peck show meaningful Dim 2 signal. See `docs/2d-irt-deep-dive.md` and ADR-0046.

   **Known limitation — wrong-axis estimation:** In 7/14 Kansas Senate sessions (78th-83rd, 88th), the 1D IRT captures the intra-Republican factional axis rather than the party axis (party d < 1.5 vs d > 4 in normal sessions). This happens when PCA PC1 captures within-R variance, and the IRT model's maximum-discrimination axis aligns with it. The model converges cleanly but measures the wrong latent dimension. See `docs/pca-ideology-axis-instability.md`.

2. **Yea/Nay only.** The model's likelihood is Bernoulli (binary). "Present and Passing," absences, and non-votes are excluded entirely — they do not enter the likelihood. This is a strength: no imputation artifacts.

3. **Missing at random (MAR).** Absences are assumed uninformative about ideology conditional on the observed votes. If legislators strategically avoid recorded votes on contentious bills, their ideal-point estimates may be too moderate (same concern as PCA, but IRT is less affected because it doesn't impute).

4. **Bills have fixed ideological content.** Each bill's difficulty (alpha) and discrimination (beta) are treated as fixed characteristics. In reality, legislators may interpret the same bill differently, or the bill's ideological content may not map cleanly onto a single dimension.

5. **Exchangeability of non-anchor legislators.** All non-anchor legislators share the same prior: Normal(0, 1). There is no hierarchical structure by party, region, or seniority. The data alone drives posterior differences.

## Parameters & Constants

### MCMC Sampling

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `DEFAULT_N_SAMPLES` | 2000 | Posterior draws per chain. Standard for IRT — sufficient for stable HDIs. | `irt.py:181` |
| `DEFAULT_N_TUNE` | 1000 | NUTS adaptation period (discarded). Allows the sampler to learn the posterior geometry. | `irt.py:182` |
| `DEFAULT_N_CHAINS` | 2 | Two independent chains. Sufficient for R-hat computation. See ADR-0006 for trade-off discussion. | `irt.py:183` |
| `TARGET_ACCEPT` | 0.9 | Higher than the PyMC default (0.8). Reduces divergences in the complex IRT posterior at a ~20% speed cost. | `irt.py:184` |
| `RANDOM_SEED` | 42 | For MCMC reproducibility. Same seed used consistently across phases. | `irt.py:185` |

### Priors

| Prior | Distribution | Justification |
|-------|-------------|---------------|
| xi (ideal point) | Normal(0, 1) + two anchors fixed at +1/-1 | Standard normal center with unit scale. Anchors fix location and scale. |
| alpha (difficulty) | Normal(0, 5) | Diffuse prior. Allows difficulty to range widely — some bills are easy, some are hard. The SD of 5 means 95% prior mass covers [-10, +10], which is much wider than the ideal-point scale. |
| beta (discrimination) | Normal(0, 1) | **Unconstrained.** Anchors provide sign identification; positive β = conservative Yea, negative β = liberal Yea. See `beta_prior_investigation.md` for rationale. |

### Filtering & Validation

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `CONTESTED_THRESHOLD` | 0.025 | Inherited from EDA (defined in `analysis/tuning.py`). IRT reads pre-filtered matrices. | `irt.py:187` |
| `SENSITIVITY_THRESHOLD` | 0.10 | Full MCMC re-run at 10% threshold. | `irt.py:188` |
| `MIN_VOTES` | 20 | Inherited from EDA. | `irt.py:189` |
| `HOLDOUT_FRACTION` | 0.20 | In-sample prediction on 20% of observed cells. | `irt.py:190` |
| `HOLDOUT_SEED` | 42 | For holdout reproducibility. | `irt.py:191` |
| `MIN_PARTICIPATION_FOR_ANCHOR` | 0.50 | Anchors must have voted on >= 50% of contested bills. Ensures tight estimation. | `irt.py:192` |

### Convergence Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `RHAT_THRESHOLD` | 1.01 | Rank-normalized R-hat must be < 1.01 for all parameters. Vehtari et al. (2021). |
| `ESS_THRESHOLD` | 400 | Bulk-ESS and tail-ESS > 400 for reliable inference. 100 per chain minimum. |
| `MAX_DIVERGENCES` | 10 | < 10 divergent transitions across all chains. More indicates posterior geometry problems. |
| `N_CONVERGENCE_SUMMARY` | 4 | Number of extreme-R-hat parameters shown in convergence summary output. |

**Convergence diagnostics checked:** R-hat (rank-normalized), bulk-ESS, tail-ESS, divergent transitions, E-BFMI (> 0.3). Tail-ESS (Vehtari et al. 2021) catches poor mixing in posterior tails that bulk-ESS alone can miss — particularly relevant for credible interval estimation.

## Methodological Choices

### Discrimination prior (beta)

**Current decision:** Normal(0, 1) — unconstrained. Anchors provide sign identification.

**What this does:** Allows beta to be positive (conservative Yea) or negative (liberal Yea). The sign of beta encodes which end of the spectrum favors Yea; the magnitude encodes how partisan the vote is. The two hard anchors (xi = +1 and -1) break the sign ambiguity, making a positive constraint unnecessary.

**Why Normal(0, 1):**
- **Uses all data.** With unconstrained beta, both R-Yea bills (beta > 0) and D-Yea bills (beta < 0) contribute to ideal point estimation.
- **Anchors handle identification.** The sign-switching problem that motivates positive-constrained priors is already solved by the hard anchors.
- **Best convergence.** In a 3-way experiment (LogNormal(0.5,0.5), Normal(0,2.5), Normal(0,1)), Normal(0,1) had the highest ESS, fastest sampling, and zero divergences.
- **Best accuracy.** Holdout accuracy +3.5%, AUC +0.025, PCA correlation +0.022 vs LogNormal.

**History:** The initial implementation used LogNormal(0.5, 0.5), following the standard recommendation for soft-identified IRT models. Investigation revealed this silenced 12.5% of bills (all D-Yea votes assigned beta near zero). See `analysis/design/beta_prior_investigation.md` for the full investigation, experiment results, and mathematical proof that alpha cannot compensate for constrained beta.

**Alternatives considered:**
- LogNormal(0.5, 0.5) — rejected: positive constraint silences D-Yea bills. Was standard advice but assumes soft identification, not hard anchors.
- Normal(0, 2.5) — viable but wider prior produces slightly slower convergence than Normal(0, 1) with nearly identical accuracy.
- Half-Normal — viable but lacks negative support needed for D-Yea bills.

### PCA-based anchor selection (party-aware)

**Decision:** After PCA orients PC1 so Republican mean > Democrat mean, select the most extreme Republican (highest PC1) as conservative anchor at xi=+1 and the most extreme Democrat (lowest PC1) as liberal anchor at xi=-1. Both must have >= 50% participation. Falls back to raw PC1 extremes for single-party chambers.

**Why:** Automates anchor selection using existing PCA results. No manual knowledge of Kansas politics required. The 50% participation guard ensures anchors have enough data for tight estimation (their ideal points are fixed, so their data directly informs bill parameters). The party-aware selection prevents sign flip in supermajority chambers where intra-party variation dominates PC1. For example, in the 79th Kansas Senate (30R/10D), the raw PC1 extreme was Tim Huelskamp — a far-right Republican rebel whose voting pattern (opposing establishment bills) resembled Democrats. Without party filtering, Huelskamp would be selected as the "liberal" anchor, mis-identifying the model.

**Alternatives considered:**
- Raw PC1 extremes without party filtering — the original approach; caused sign flip in the 79th Senate and potentially other supermajority chambers. Replaced in 2026-03-06.
- Manual anchor selection (e.g., "pick a Freedom Caucus member and a Lawrence Democrat") — rejected for reproducibility; requires human judgment that changes across sessions
- Soft identification via N(0,1) prior + post-hoc sign correction — rejected because it's fragile with 2 chains
- Three anchors (conservative, liberal, moderate at 0) — rejected as unnecessary for 1D; two anchors fix location, scale, and sign

**Impact:** In supermajority chambers, the horseshoe effect can cause PCA-based anchor selection to pick the wrong reference legislator (e.g., a moderate establishment Republican instead of a true conservative), producing a sign flip where individual legislators appear on the wrong side of the dimension. A post-hoc `validate_sign()` step detects and corrects this by correlating cross-party contested vote agreement with ideal point magnitude. See `docs/irt-sign-identification-deep-dive.md`.

### Post-hoc sign validation

**Decision:** After MCMC, `validate_sign()` checks whether the recovered ideal points have the correct polarity by correlating cross-party contested vote agreement with ideal point magnitude. If the correlation indicates a sign flip, all xi and beta posteriors are negated, selecting the other equally valid posterior mode.

**Why:** PCA-based anchor selection picks the "most party-typical" Republican as the conservative anchor. In supermajority chambers with a rebel faction, the horseshoe effect causes PCA PC1 to capture establishment-vs-rebel rather than left-vs-right. The "most party-typical" R becomes the most establishment-aligned moderate, not the most ideologically conservative. This produces a sign flip: ultra-conservative rebels appear at the liberal end.

**Algorithm:** (1) Identify contested votes where both parties split (≥10% threshold per side per party). (2) For each Republican, compute agreement rate with the Democrat majority position on contested votes. (3) Spearman-correlate agreement with xi. Correct sign → negative correlation (moderates agree more). Flipped → positive (extremes agree more). (4) If positive with p < 0.10, negate xi, xi_free, and beta.

**Guard rails:** Skips when fewer than 3 legislators per party, fewer than 10 contested votes, or fewer than 5 Republicans with valid agreement data.

**Impact:** Fixes the 79th Senate Huelskamp placement (xi = -3.26 → +3.26). Does not affect correctly-signed sessions (the correlation is negative, so no flip occurs). See `docs/irt-sign-identification-deep-dive.md`.

### Identification strategy system (7 strategies, auto-detection)

**Decision:** Rather than a single identification method, the model supports seven strategies with auto-detection via `--identification auto` (default). See ADR-0103 and `docs/irt-identification-strategies.md` for the full catalog, literature references, and auto-detection logic.

| Strategy | Method | Auto-selected when |
|----------|--------|--------------------|
| anchor-pca | Hard anchors from PCA PC1 party extremes | Balanced chamber, no external scores |
| anchor-agreement | Hard anchors from cross-party contested vote agreement | Supermajority + sufficient contested votes |
| sort-constraint | `pm.Potential`: D mean < R mean | Supermajority + insufficient contested votes |
| positive-beta | `beta ~ HalfNormal(1)` | Never (manual only; silences D-Yea bills) |
| hierarchical-prior | `xi ~ Normal(±0.5, 1)` by party | Never (manual only) |
| unconstrained | No constraints; post-hoc sign correction | Never (diagnostic only) |
| external-prior | `xi ~ Normal(sm_score, 0.5)` | External scores available |

All strategies pass through `validate_sign()` as a post-hoc safety net. Every run prints a rationale table showing why the selected strategy was chosen and why alternatives were passed over.

### Robustness flags (CLI-driven diagnostics)

**Decision:** Optional CLI flags enable runtime robustness analyses without code changes. All flags are always visible in the HTML report (ON/OFF), so the exact configuration is recorded with every run. See ADR-0104.

| Flag | CLI arg | Purpose |
|------|---------|---------|
| `contested_only` | `--contested-only` | Re-fit IRT on cross-party contested votes only (strips intra-party rebel dynamics) |
| `horseshoe_diagnostic` | `--horseshoe-diagnostic` | Compute 6 quantitative horseshoe metrics (Democrat wrong-side fraction, overlap, eigenvalue ratio) |
| `horseshoe_remediate` | `--horseshoe-remediate` | Auto-refit with PC2-filtered votes + PC2 informative prior when horseshoe detected (implies `--horseshoe-diagnostic`) |
| `promote_2d` | `--promote-2d` | Cross-reference 1D rankings with 2D IRT Dim 1 rankings; flag legislators with large rank shifts |

**Why flags, not always-on:** Contested-only refit doubles MCMC runtime. 2D cross-reference requires Phase 04b results. Horseshoe diagnostic is cheap but adds report clutter for balanced chambers where it always passes. Making these opt-in keeps the default report focused.

**Extensibility:** New flags follow the `RobustnessFlag` frozen dataclass + `RobustnessFlags` registry pattern. Add to `ALL_FLAGS`, add CLI arg, add logic block in `main()`.

**Always-on report additions:** The identification summary (strategy, anchors, sign flip status) and robustness flags table are always shown regardless of flag state. This closes a prior reporting oversight where anchor identity and method were not recorded.

### PCA-informed chain initialization (default: on)

**Decision:** Initialize MCMC chains with standardized PCA PC1 scores (mean-0, sd-1) as starting values for the free ideal point parameters (`xi_free`). On by default; disable with `--no-pca-init`.

**Why:** The 2PL IRT model has a reflection invariance — negating all ξ and all β leaves the likelihood unchanged. Hard anchors create an energy barrier between the two modes but cannot prevent a chain from initializing on the wrong side. With random initialization and seed 42, 5 of 16 chamber-sessions exhibited catastrophic mode-splitting (R-hat ~1.83, ESS 3, zero divergences). PCA initialization places both chains in the correct mode's basin of attraction, eliminating the problem at zero cost.

**Literature support:** Jackman's `pscl::ideal()` — the reference implementation for Bayesian IRT in political science — has used eigendecomposition (`startvals="eigen"`) as its default starting values since 2001. Bafumi, Gelman, Park, & Kaplan (2005) discuss PCA-generated starting values as one of three initialization strategies. Betancourt (2017) warns against initialization-based identification for general mixture models but acknowledges it works when modes have a known substantive ordering, which is the case for legislative ideal points. See `results/experimental_lab/2026-02-23_irt-convergence-mode-splitting/irt-convergence-investigation.md` for the full experimental validation and `results/experimental_lab/2026-02-23_irt-convergence-mode-splitting/lit-review-irt-initialization.md` for the literature review.

**Alternatives considered:**
- Random initialization (PyMC default) — caused 5/16 convergence failures
- Extended tuning (3000 draws) — zero effect; trapped chains don't cross the energy barrier
- Soft sign constraint (`pm.Potential`) — zero effect; likelihood overwhelms the penalty. Code removed (2026-02-25).
- 4 chains — partial fix; masks the problem via majority vote but destroys credible interval precision

**Impact:** Eliminates all 5 convergence failures. No known downsides when PC1 cleanly separates the ideological dimension (true for all Kansas sessions, with PCA-IRT r > 0.93 when converged).

**PCA axis delegation (ADR-0128):** Phase 05 now delegates PCA column selection to `resolve_init_source()` with `session` and `chamber` parameters. This enables manual PCA overrides (`pca_overrides.yaml`) and automated `detect_ideology_pc()` party-correlation detection. Previously, Phase 05 hardcoded `["PC1"]`, causing sign-flipped ideal points in 3 sessions (79th, 84th, 91st Senate) where PC1 was not the ideology axis or the sampler found a flipped mode.

### Init strategy override (`--init-strategy`)

**Decision:** Accept `--init-strategy {auto,irt-informed,pca-informed,2d-dim1}` to override the default PCA-informed initialization. Uses the shared `analysis/init_strategy.py` module (ADR-0107).

**Why:** For sessions where the 1D model collapses ideology and establishment into one dimension (e.g., 79th Kansas Senate), the 2D IRT model's Dim 1 correctly isolates ideology. Re-running the 1D model with `--init-strategy 2d-dim1` nudges chains toward the ideology-only solution, producing meaningful ideal points where PCA-informed init fails.

**Workflow:** Run the pipeline normally → inspect 2D results → if 1D results look collapsed, re-run with `just irt --init-strategy 2d-dim1`. Auto never selects `2d-dim1` — it's an explicit user choice for iterative refinement.

### Native missing data handling

**Decision:** Absences are handled by simply excluding those (legislator, vote) pairs from the likelihood. No imputation.

**Why:** This is a key advantage of IRT over PCA. The Bernoulli likelihood is defined only over observed cells. A legislator who was absent for 70% of votes still contributes information from their 30%, and the model correctly widens their credible interval to reflect the uncertainty.

**Impact:** Legislators with few votes (e.g., Miller with 30/194 Senate votes) will have wide HDIs but unbiased point estimates. No imputation artifacts.

### Two chains (not four)

**Decision:** Default to 2 MCMC chains instead of the textbook 4. Chains run in parallel (`cores=n_chains`).

**Why:** Runtime. Each chain takes ~5-10 minutes per chamber. With parallel chain sampling, 2 chains complete in roughly the time of 1 (~5-10 min) instead of running sequentially (~15-20 min). The model is well-identified (anchored, PCA-initialized), so 2 chains are typically sufficient.

**Parallel safety:** PyMC uses multiprocessing (not threading) for parallel chains. Each chain gets its own process, its own memory, and a deterministic per-chain seed derived from `random_seed`. Results are mathematically identical to sequential execution.

**Trade-off:** Less power to detect multi-modal posteriors. Memory usage doubles with parallel chains (~2x model size). If R-hat > 1.01 or ESS < 400, re-run with `--n-chains 4`.

### In-sample holdout (not true out-of-sample)

**Decision:** Use posterior means from the full model to predict a random 20% of observed cells. This is documented as in-sample prediction.

**Why:** A true out-of-sample holdout requires masking cells before fitting and running a second MCMC — doubling runtime. The posterior predictive check (PPC) provides the proper Bayesian validation: it samples from the full posterior and compares replicated data to observed.

**Impact:** The holdout accuracy overstates predictive performance (model saw all data during fitting). The PPC Bayesian p-value is the more reliable validation metric.

### Per-chamber models (not joint)

**Decision:** House and Senate are fitted as completely independent IRT models.

**Why:** Standard approach in the literature (NOMINATE, Clinton et al. 2004). Avoids the complexity of cross-chamber identification.

**Trade-off:** Cannot leverage bridging legislators (e.g., Miller, who served in both chambers). A joint model could use Miller's ~300+ House votes to tightly constrain his Senate ideal point. This is deferred to a future enhancement and documented in `docs/analytic-flags.md`.

## Downstream Implications

### For Clustering (Phase 5)
- **Use IRT ideal points (xi_mean) as the primary clustering input**, not PCA scores. IRT accounts for vote difficulty and discrimination; PCA does not.
- **HDI width is a confidence measure.** Consider weighting legislators by 1/xi_sd or excluding those with xi_sd above a threshold.
- Sen. Miller's wide HDI means his cluster assignment is low-confidence. Flag in cluster interpretation.

### For Network Analysis (Phase 6)
- Bill discrimination parameters (beta) identify the most ideologically informative votes. Consider building networks from high-discrimination votes only (beta > 1.5) for a cleaner signal.
- The difficulty parameter (alpha) identifies where on the spectrum each bill "cuts." This can annotate network edges with substantive meaning.

### For Prediction (Phase 7)
- The IRT posterior can predict future votes: P(Yea) = logit^-1(beta * xi - alpha). For a new bill, estimate alpha and beta from similar past bills; for a new legislator, use party-average xi as a prior.

### For interpretation
- **Ideal points are on an arbitrary scale** (anchored at +1/-1, but the units are not "degrees of conservatism"). Compare legislators to each other, not to absolute values.
- **HDI overlap = indistinguishable.** Two legislators whose 95% HDIs overlap cannot be reliably ranked. The forest plot visualizes this.
- **Discrimination sign indicates direction.** Positive β = conservative position is Yea. Negative β = liberal position is Yea. |β| measures discriminating power. A bill with |β| > 1.5 is highly discriminating; |β| < 0.5 is weakly discriminating.
- **The 1D model is a simplification.** Tyson's contrarianism on routine bills, Thompson's mild version of the same, and any other multi-dimensional pattern will be compressed into a single number. If a legislator's ideal point seems surprising, check whether they have unusual PC2 behavior (see `docs/analytic-flags.md`).

## Cross-Chamber Equating

### Motivation

Per-chamber IRT models estimate ideal points on separate, incomparable scales. A House xi=+2.0 and a Senate xi=+2.0 do not mean the same thing because the models are fitted independently with different bill sets, different anchors, and different posterior geometries. To place all legislators on a common scale for cross-chamber comparisons, test equating is needed.

### Why Not a Hierarchical Joint MCMC Model?

A hierarchical joint IRT model was attempted (Phase 07, `build_joint_graph()`): a 3-level hierarchy (global → chamber → party → legislator) with ~1,000 parameters. This failed convergence in all 8 bienniums (R-hat up to 2.56, ESS as low as 5). Root causes: Neal's funnel geometry in the 3-level hierarchy, partial reflections in `sigma_chamber`, and insufficient shared bills to identify the chamber-level hyperparameters. See ADR-0074 and `docs/joint-model-deep-dive.md`.

### Flat Pooled Joint IRT (Experimental)

A **flat** (non-hierarchical) joint IRT model succeeds where the hierarchical model fails. The experiment (`analysis/experimental/joint_irt_experiment.py`) pools all legislators from both chambers into a single 1D 2PL IRT model on the full union of votes. For the 79th biennium (2001-02):

- **168 legislators** on **888 votes** (451 House-only, 267 Senate-only, 170 shared)
- Block-sparse missing data: Senate members have NaN on House-only votes and vice versa; IRT handles this natively by excluding missing observations from the likelihood
- **Perfect convergence**: R-hat 1.003, ESS 1012, 0 divergences
- The 128 House legislators anchor the ideology scale; 170 shared bills bridge the chambers
- House ideal points virtually unchanged vs per-chamber model (r = 0.998)
- Senate ideal points gain meaningful identification from the larger bill set

The key insight: the hierarchical model's failure was about the 3-level hierarchy's funnel geometry, not about pooling chambers per se. A flat model sidesteps the funnel entirely. Anchors are selected from House PCA extremes (the larger, better-identified chamber). This is particularly valuable for supermajority chambers (e.g., 79th Senate: 30R/10D) where per-chamber models fail to converge.

### Test Equating Approach

Instead, we use **classical test equating** — a well-established psychometric method that transforms one scale to match another using shared items and/or common examinees.

**A (scale factor)** from shared bill discrimination parameters:
- For each of the ~71 shared bills, compare the beta (discrimination) from the House IRT model vs the Senate IRT model
- Use only concordant bills (same sign of beta in both chambers)
- A = SD(beta_senate) / SD(beta_house)

**B (location shift)** from bridging legislators:
- 3 legislators served in both chambers: Thompson, Hill, Miller
- Each has a per-chamber ideal point from the separate IRT models
- B = mean(xi_house) - A * mean(xi_senate) across bridging legislators

**Transformation**: xi_equated = A × xi_senate + B (House scale is the reference; House ideal points are unchanged)

### Bill Matching Algorithm

For each bill_number appearing in both chambers' filtered matrices:
1. Find all vote_ids in each chamber for that bill
2. Prefer the vote_id with "Final Action" or "Emergency Final Action" motion
3. If multiple, pick the latest chronologically (vote_id encodes timestamp)

### 2025-26 Results

- **A = 1.136**: Senate discrimination parameters have ~14% more spread than House
- **B = -0.305**: Small leftward shift (Senate center is slightly more conservative than House center)
- **51 concordant / 71 shared bills** used for A estimation
- **3 bridging legislators** used for B estimation

### Assumptions

1. **Shared bills have the same ideological content in both chambers.** If a bill is substantially amended between chambers, the equating is weakened for that bill.
2. **Bridging legislators have stable ideology across chambers.** A legislator who shifts position when moving from House to Senate would bias B.
3. **Linear relationship between scales.** The transformation assumes a linear mapping; nonlinear distortions are not captured.

### Limitations

- Senate uncertainty (HDI width) is scaled by |A| but the correlation structure between legislators is not preserved — these are transformed marginals, not a joint posterior.
- With only 3 bridging legislators, B has limited precision. A single outlier bridging legislator can shift the location substantially.
- Per-chamber models remain primary for within-chamber analyses. Equated scores are for cross-chamber comparison only.

### Validation

- House equated vs per-chamber: r = 1.0 (unchanged by construction)
- Senate equated vs per-chamber: r = 1.0 (linear transformation preserves rank order)
- Bridging legislator positions should be consistent across chambers after equating

## Horseshoe Effect and Supermajority Diagnostics

See `docs/horseshoe-effect-and-solutions.md` for a general-audience explanation of the horseshoe effect in supermajority chambers and six approaches to addressing it.

See `docs/79th-horseshoe-robustness-analysis.md` for empirical validation of the robustness flags system (ADR-0104) on the 79th biennium, which exhibits clear horseshoe distortion.

The identification strategy system (ADR-0103) auto-selects strategies designed to mitigate horseshoe effects in supermajority chambers via `anchor-agreement` or `sort-constraint` approaches.

When identification strategies alone are insufficient (e.g., the 79th Senate where PCA dimensions are swapped), `--horseshoe-remediate` auto-refits using PC2-filtered votes and a PC2 informative prior. This redirects the 1D model from the establishment-loyalty axis (PC1) to the ideology axis (PC2). The remediation is gated on `detect_horseshoe()` — only chambers that fail the diagnostic are refitted. See `results/experimental_lab/2026-03-09_pc2-targeted-irt/experiment.md` for the validation experiment.

At the report level, 8 report builders accept `horseshoe_status` from `phase_utils.load_horseshoe_status()` and inject styled warning banners via `horseshoe_warning_html()` when distortion is detected (ADR-0114). The IRT report also adds a key finding when Republican mean < Democrat mean (party mean inversion), and uses data-driven IRT-PCA correlation captions instead of hardcoded text.

### Party Separation Quality Gate (R2, ADR-0118)

After extraction, the 1D IRT computes Cohen's d between Republican and Democrat mean ideal points and writes it to `convergence_summary.json` alongside standard convergence diagnostics:

- `party_separation_d`: Cohen's d between party means
- `axis_uncertain`: true when d < 1.5

When `axis_uncertain` is true, the 1D IRT is likely measuring intra-party factionalism rather than ideology. Downstream consumers (canonical routing, synthesis, profiles) should treat such results with caution. The Tier 2 quality gate in canonical routing uses `party_separation_d` instead of PCA rank correlation to avoid circular dependency (R3). See `docs/pca-ideology-axis-instability.md`.
