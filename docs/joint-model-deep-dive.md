# Joint Hierarchical IRT: Why Cross-Chamber Scaling Fails and How to Fix It

*February 2026*

## The Problem in Plain English

Tallgrass estimates where each Kansas legislator falls on a liberal-to-conservative spectrum using Bayesian Item Response Theory (IRT). The per-chamber models — one for the House, one for the Senate — work well: they produce reliable estimates that correlate strongly with the Shor-McCarty academic benchmark (House r = 0.97, Senate r = 0.95). But the moment we try to put both chambers on a **common scale** using a single "joint" model, everything falls apart.

The joint model fails convergence diagnostics in **all eight bienniums** (2011-2026). The R-hat statistic — which should be below 1.01 for trustworthy results — ranges from 1.007 to 2.42. The effective sample size (ESS) collapses to 5-7 in seven of eight sessions, meaning the model's 8,000 posterior draws contain roughly the same information as 7 independent samples. The error bars on individual legislators' ideal points explode to absurd widths: a legislator whose per-chamber model places them at 0.5 +/- 0.3 might show up in the joint model at 0.5 +/- 3.0.

This matters because the joint model is the only way to directly compare legislators across chambers — to answer questions like "Is the Senate more conservative than the House?" or "Is Senator X further right than Representative Y?" Without a working joint model, we rely on the flat IRT's post-hoc test equating, which makes strong assumptions about the equivalence of shared bills.

This article examines the joint model's architecture, diagnoses the root causes of failure, reports our experimental results, surveys the political science and psychometrics literature for solutions, and proposes a revised improvement plan.

## The Joint Model's Architecture

The joint model (`build_joint_graph()` in `analysis/07_hierarchical/hierarchical.py`, line 562) is a three-level Bayesian hierarchy:

```
Level 3 (global):    mu_global ~ Normal(0, 2)
                          |
Level 2 (chamber):   mu_chamber = mu_global + sigma_chamber * chamber_offset
                          |
Level 1 (group):     mu_group = mu_chamber[chamber] + sigma_party * group_offset
                          |
Level 0 (legislator): xi = mu_group[group] + sigma_within[group] * xi_offset
```

There are four groups (House Democrats, House Republicans, Senate Democrats, Senate Republicans), each with its own mean and within-group variance. Identification is achieved by sorting each chamber's party offsets so that Democrats are to the left of Republicans (`pt.sort(group_offset_raw[:2])` for House, `pt.sort(group_offset_raw[2:])` for Senate).

The vote model is a standard two-parameter logistic (2PL) IRT:

```
P(Yea | xi_i, alpha_j, beta_j) = logistic(beta_j * xi_i - alpha_j)
```

where `alpha_j` is the bill's difficulty (how likely a Yea vote overall) and `beta_j` is the bill's discrimination (how strongly it separates liberals from conservatives).

### Parameter Count

For the 91st Legislature (2025-26), the joint model has approximately **1,042 free parameters**:

| Component | Count | Notes |
|-----------|-------|-------|
| `mu_global` | 1 | Global mean |
| `sigma_chamber` | 1 | Chamber-level spread |
| `chamber_offset` | 2 | House, Senate |
| `sigma_party` | 1 | Party-level spread |
| `group_offset_raw` | 4 | HD, HR, SD, SR (2 sorted pairs) |
| `sigma_within` | 4 | Per-group within-party spread |
| `xi_offset` | 172 | One per legislator (130 House + 42 Senate) |
| `alpha` | 420 | Bill difficulty (71 shared + 179 House-only + 170 Senate-only) |
| `beta` | 420 | Bill discrimination |
| **Total** | **~1,025** | |

This is a large model by MCMC standards. The NUTS sampler's mixing time scales as O(d^{1/4}) with dimension d, meaning a 1,000-parameter model mixes roughly 5.6x slower than a 100-parameter model *even in the best case* — and IRT posteriors are far from the best case.

### The Bill-Matching Bridge

The joint model's primary innovation (ADR-0043, implemented February 2026) is **concurrent calibration**: bills that pass through both chambers share a single set of alpha/beta parameters. When the House votes on HB 2001 and the Senate later votes on the same bill, those observations constrain the same `beta_j`. This creates a mathematical bridge — the shared bill's discrimination parameter forces the two chambers' ideal point scales to be comparable.

In the 91st Legislature, 174 bill numbers appear in both chambers' roll call records. After the EDA near-unanimous filter, 71 survive as shared bridge items. These 71 bills constitute 17% of the total vote set (71 / 420), meaning 83% of the bill parameters have no cross-chamber bridging function. They provide within-chamber information only.

Before the bill-matching fix, the joint model used `vote_id` deduplication, which treated every roll call as unique — meaning **zero shared items** across chambers. The two chambers' likelihoods were completely separable, and the hierarchy alone had to link their scales. The fix improved runtime from 93 minutes to 32 minutes but did not achieve convergence.

### Sign Identification: Necessary but Not Sufficient

The model constrains `mu_group` ordering within each chamber (D < R), which breaks the global reflection symmetry at the *group level*. But the individual bill discrimination parameters (`beta`) can each independently flip sign: if `beta_j` flips to `-beta_j` and the corresponding `xi` interactions adjust, the likelihood is unchanged. With 420 bills, this creates 420 independent axes along which partial reflection can occur.

The production code includes a post-hoc sign correction (`fix_joint_sign_convention()`, line 851) that compares joint ideal points to per-chamber estimates and negates the posterior if they're anti-correlated. But this is a band-aid: if the sampler got stuck exploring reflection modes, the posterior is bimodal and averaging across modes is not meaningful.

## Experiment: PCA Init + LogNormal Beta (84th Biennium)

### What We Changed

Based on the initial analysis in this article, we implemented two changes:

1. **PCA initialization for the joint model.** Concatenated per-chamber PCA PC1 scores (standardized to N(0,1) scale) as starting values for `xi_offset`, matching the order of `build_joint_graph()` (House first, then Senate). Added `JOINT_BETA = BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})` to `model_spec.py`.

2. **LogNormal(0, 0.5) beta prior** for the joint model only. Per-chamber models continue to use Normal(0, 1). The `JOINT_BETA` constant is passed only to `build_joint_model()`.

We ran the full 14-phase pipeline on the 84th biennium (2011-12, run ID `84-260228.3`).

### Results

**Per-chamber models: flawless convergence (unchanged).**

| Metric | House | Senate |
|--------|-------|--------|
| R-hat(xi) max | 1.0021 | 1.0025 |
| ESS(xi) min | 1,026 | 869 |
| Divergences | 0 | 0 |
| Sampling time | 253.9s | — |
| Verdict | **PASS** | **PASS** |

**Joint model: improved R-hat, catastrophic divergences.**

| Metric | Baseline (Normal, no PCA) | PCA + LogNormal |
|--------|--------------------------|-----------------|
| R-hat(xi) max | 1.5362 | **1.2225** |
| ESS(xi) min | 7 | 14 |
| Divergences | 10 | **2,041** |
| Sampling time | ~30 min | 142.5s |
| Sign correction needed | Yes | **No** (r=0.96 House, r=0.82 Senate) |

### What Happened

The PCA initialization worked as intended — the sign is correct without post-hoc correction, and R-hat improved from 1.54 to 1.22. But the LogNormal prior created a **geometric catastrophe**: divergences exploded from 10 to 2,041.

This was the opposite of what we expected based on the 91st biennium positive-beta experiment, where LogNormal produced only 25 divergences. The 84th has different data characteristics (more ODT-derived votes, smaller vote set, more near-unanimous bills), but the root cause is deeper.

### Why LogNormal Causes Divergences

When PyMC encounters `beta ~ LogNormal(0, 0.5)`, it applies a log transform internally to sample in unconstrained space. The sampler works with `log(beta)`, and adds a Jacobian correction. This creates **severe curvature variation** near the zero boundary:

- The distance between beta = 0.01 and beta = 0.001 in unconstrained space is `log(0.01) - log(0.001) = 2.3` nats
- The distance between beta = 1.0 and beta = 0.99 is only 0.01 nats

With ~365 bill parameters, even a small fraction with posterior mass near zero creates local curvature pathologies. The step size adapts globally to the worst case, slowing the entire model. Betancourt's case study on divergences identifies this as the primary cause: "highly varying posterior curvature, for which small step sizes are too inefficient in some regions and diverge in other regions."

The Stan documentation, PyMC discourse, and Betancourt's diagnostic case studies all converge on the same point: **hard boundaries on constrained parameters are a leading cause of NUTS divergences**, and the LogNormal's `1/beta` density term makes it worse than HalfNormal near zero.

## Root Cause Diagnosis: Three Reinforcing Problems

The joint model's convergence failure has three distinct root causes. Our experiment confirmed the first two and revealed the third.

### Problem 1: Reflection Mode Multimodality

With `beta ~ Normal(0, 1)`, each of the 420 bill discrimination parameters can be positive or negative. The likelihood is invariant under `(beta_j, xi) -> (-beta_j, -xi)` for each bill independently. The `pt.sort` constraint on group offsets breaks the *global* reflection (all betas flip simultaneously), but it cannot prevent *partial* reflections where a subset of betas flip.

This creates a posterior landscape with ridges and saddle points that slow NUTS exploration. The sampler doesn't diverge (the geometry is smooth), but takes many steps to traverse between regions, inflating autocorrelation and depressing ESS.

**Evidence**: The 91st positive-beta experiment (LogNormal prior) reduced joint R-hat from 1.53 to 1.024 and ESS from 7 to 243. PCA init on the 84th eliminated the need for post-hoc sign correction (r=0.96 House, r=0.82 Senate).

### Problem 2: High-Dimensional Slow Mixing

Even after removing reflection modes, the joint model has ~1,000 free parameters. NUTS mixing time scales as O(d^{1/4}) with dimension, meaning the 91st joint model mixes ~3.2x slower than the per-chamber House. The block-diagonal vote matrix (~83% chamber-specific bills), three-level hierarchy, and ~50% structural missingness compound this.

**Evidence**: Even with LogNormal beta, the 91st joint model's ESS was 243 — below the 400 threshold. Step size collapsed to 0.007-0.009 (vs. 0.06 per-chamber), and draw speed was ~1/second (vs. ~9/second).

### Problem 3: LogNormal Boundary Geometry (Newly Identified)

The LogNormal prior solves Problem 1 (reflection modes) but introduces Problem 3: a curvature catastrophe near beta = 0. Many bills in a legislative session have genuinely low discrimination — bipartisan votes, procedural motions, near-unanimous confirmations. The LogNormal(0, 0.5) prior, with 95% of its mass in [0.37, 2.69], fights to keep these bills' betas away from zero. This prior-data conflict creates high-curvature regions that NUTS cannot navigate at any reasonable step size.

**Evidence**: 84th joint model divergences went from 10 (Normal beta) to 2,041 (LogNormal beta). The 84th biennium's ODT-derived data likely has more near-zero-discrimination bills than the 91st, explaining why the divergence explosion was worse than the 91st experiment (25 divergences).

## Literature Survey: How Does the Field Handle This?

The problem of placing legislators from different chambers on a common ideological scale has been studied extensively in political science and psychometrics. The approaches fall into two broad categories.

### Category 1: Concurrent Calibration (One Big Model)

This is what our joint model attempts: estimate all parameters in a single model where shared items (bills voted on by both chambers) provide the cross-chamber link.

**DW-NOMINATE Common Space** (Poole and Rosenthal 1991, updated through present). The original and still most widely used method. Instead of shared *bills*, it uses **bridge legislators** — members who serve in both chambers across their careers — to anchor the common scale. Kansas has very few senators who previously served in the House within a single biennium, making this approach impractical for within-biennium analysis. DW-NOMINATE also uses a deterministic optimization (not MCMC), sidestepping the convergence problem entirely.

**Shor and McCarty (2011)**. The gold standard for state-level cross-chamber scaling. They use *survey bridging*: Project Vote Smart interest group scores and NPAT questionnaire responses serve as common items across chambers. This is exogenous data — not roll call votes — so the bridging is independent of the legislative process. Their dataset covers all 50 states from 1993-2020 and is externally validated against congressional scores. Our flat IRT correlates strongly with Shor-McCarty (House r = 0.975, Senate r = 0.950 pooled), confirming our per-chamber estimates are sound.

**Bailey (2007)**. Uses "bridge observations" — presidential position-taking on Supreme Court cases and congressional votes — to place presidents, Congress members, and Supreme Court justices on a common scale. Clever bridging design, but not applicable to state legislatures.

**Hierarchical Multi-Group IRT** (Fox 2010, De Boeck et al. 2011). The theoretical basis for our model: treat groups (chambers, parties) as levels in a hierarchy, with partial pooling. Fox (2010) demonstrates this for educational testing where multiple schools take partially overlapping exams. The key difference from our setting: educational tests typically have *many* shared items (50-100% overlap) and relatively few dimensions of variation. Our 17% shared-bill rate may be too low for effective concurrent calibration.

### Category 2: Separate-Then-Link (Test Equating)

Estimate each chamber's ideal points independently (which we know works well), then use a post-hoc transformation to align the scales.

**Test Equating Methods** (Kolen and Brennan 2014). The psychometrics field has developed sophisticated methods for placing scores from different test forms on a common scale. Four methods are relevant, in order of sophistication:

- **Mean-Mean**: Uses the ratio of mean discrimination and difference of mean difficulty across anchor items. Simplest; sensitive to outlier items.
- **Mean-Sigma**: Uses the ratio of difficulty standard deviations across anchor items. Still closed-form. This is essentially what our flat IRT's test equating does.
- **Haebara**: Minimizes squared differences between individual item characteristic curves (ICCs), integrated over the ability distribution. More robust because item-level discrepancies have bounded influence.
- **Stocking-Lord**: Minimizes squared differences between test characteristic curves (TCCs — the sum of all anchor ICCs). Most widely used in operational testing. A single badly-fitting anchor item can have unbounded influence, but with 67 anchors this is manageable.

Simulation studies (Kim and Kolen 2006) consistently find that **Haebara and Stocking-Lord yield more accurate linking** than Mean-Mean and Mean-Sigma. With 67 anchor items, all four methods should produce similar results, but Stocking-Lord is the standard choice.

All four methods are computationally cheap — they're optimization problems on 2 parameters (slope and intercept) — and don't require MCMC. The trade-off is that they assume the shared items measure the same construct in both chambers.

**Differential Item Functioning (DIF)** (Holland and Wainer 1993). Before using shared bills as anchors, one should test whether each bill functions equivalently across chambers. A bill that passes routinely in one chamber but provokes a party-line fight in the other violates the linking assumption. Methods available in Python include Mantel-Haenszel (via `CMH` package or `scipy`), logistic regression DIF (via `statsmodels`), and direct posterior comparison of per-chamber item parameter estimates.

With 67 shared bills, we can afford to be conservative — dropping 10-15 items with DIF would still leave 50+ anchors, more than the 20-30 the literature recommends.

### Category 3: Alternative Estimation Strategies

**emIRT** (Imai, Lo, and Olmsted 2016). Variational EM algorithm for IRT. Trades MCMC for a fast deterministic approximation. Implemented in R only. Could handle our dimensionality but requires an R dependency we've explicitly avoided.

**Pathfinder** (Zhang et al. 2022). L-BFGS-based variational inference that finds multiple approximate posterior modes. Available in `pymc-extras`. Not a full posterior estimate — provides starting points for MCMC or a standalone approximate posterior.

**Normalizing Flow Adaptation** (nutpie experimental). nutpie's NF feature learns a nonlinear transformation that makes the posterior approximately Gaussian. Designed for models with ~1,000 parameters. On a 100-dimensional funnel model, it eliminated all divergences and improved minimum ESS from 31 to 1,836. Requires JAX backend and scales poorly beyond ~1,000 parameters (our model is right at the boundary). GPU helps significantly.

### Political Science Precedent

The field's most successful cross-chamber methods do not use concurrent calibration:

- **DW-NOMINATE** uses bridge *legislators*, not shared items, and deterministic optimization, not MCMC.
- **Shor-McCarty** uses *survey bridging* — exogenous data independent of the legislative process.
- **Clinton, Jackman, and Rivers (2004)** use anchor constraints (fixing two legislators), not positive-beta constraints.

Notably, the legislative ideal point literature (Jackman/Arnold) uses **unconstrained beta** with skew-normal distributions rather than hard positive constraints — because in political science, unlike educational testing, negative discrimination has interpretive meaning (a bill where voting Yea indicates liberalism).

Erosheva and Curtis (2017) explicitly warn that "ensuring unique identification of a CFA model by requiring selected loadings to be positive may not lead to a satisfactory solution" due to convergence failures. Our 84th biennium experiment confirms this warning.

## Experiment 2: Reparameterized LogNormal + Tighter Alpha (84th Biennium)

### What We Changed

Based on Experiment 1's diagnosis, we implemented three changes:

1. **exp(Normal) reparameterization.** Instead of `pm.LogNormal("beta", ...)`, we use `log_beta ~ Normal(0, 1)` and `beta = exp(log_beta)`. Mathematically identical distribution on beta, but the sampler works with smooth Gaussian geometry — no boundary curvature, no Jacobian correction. This is the Stan 2PL IRT tutorial approach. Added as `lognormal_reparam` case in `BetaPriorSpec.build()`.

2. **Wider prior (sigma = 1.0).** `exp(Normal(0, 1))` has prior median = 1.0 and 95% interval [0.14, 7.39] — wide enough for bills with near-zero discrimination. The Stan User's Guide IRT section uses `lognormal(0.5, 1)` with sigma = 1.0.

3. **Tighter alpha prior (sigma = 2.0).** Changed from `Normal(0, 5)` to `Normal(0, 2)`. Difficulty parameters are typically in [-3, 3] for legislative votes; the wider prior added unnecessary posterior volume.

### Results

**Joint model: dramatic improvement, still not converged.**

| Metric | Baseline | Exp 1 (LogNormal) | Exp 2 (Reparam) |
|--------|----------|-------------------|-----------------|
| R-hat(xi) max | 1.5362 | 1.2225 | **1.0098** |
| ESS(xi) min | 7 | 14 | **378** |
| Divergences | 10 | 2,041 | **828** |
| mu_chamber R-hat | — | — | 1.10 |
| sigma_chamber R-hat | — | — | 1.08 |
| sigma_party R-hat | — | — | 1.03 |
| Sign correction | Yes | No | **No** |

The reparameterization eliminated the boundary geometry catastrophe (2,041 → 828 divergences) while preserving the reflection-mode fix. R-hat(xi) dropped to 1.0098 — just barely passing the 1.01 threshold. But ESS(xi) = 378 is below the 400 minimum, and the hyperparameters (`mu_chamber`, `sigma_chamber`) are clearly not converged.

### Interpretation

The exp(Normal) reparameterization was a major win: it delivered the positive-beta constraint without the sampling pathology. The remaining 828 divergences and poor hyperparameter mixing are caused by the **3-level hierarchy itself** — the chamber-level funnel geometry. With only 2 chamber offsets and 4 group offsets, `sigma_chamber` (ESS = 39) and `sigma_party` (ESS = 140) are poorly informed by data. The sampler oscillates between exploring the outer hierarchy (requiring wide step sizes) and the 365-dimensional bill parameter space (requiring narrow step sizes).

This is Neal's funnel problem applied to a three-level model: the top-level variance parameters create extreme correlations with the parameters they govern. Non-centered parameterization (which we already use for xi) mitigates this for the legislator level but doesn't fully address the chamber and party levels.

## Experiment 3: Stocking-Lord Linking (84th Biennium)

### Implementation

We implemented all four IRT linking methods in `analysis/07_hierarchical/irt_linking.py`:

- **Stocking-Lord**: Minimizes squared difference between test characteristic curves (TCCs) over a standard-normal-weighted theta grid.
- **Haebara**: Minimizes squared ICC differences per item — more robust to individual outlier items.
- **Mean-Sigma**: Closed-form using SD ratio of difficulty parameters.
- **Mean-Mean**: Closed-form using mean ratio of discrimination parameters.

The linking pipeline: extract posterior mean alpha/beta for the 67 shared bills from each chamber's well-converged per-chamber InferenceData, compute linking coefficients (A, B), and transform Senate ideal points to the House scale: `xi_linked = A * xi_senate + B`.

### Initial Results (Sign-Naive) — Degenerate

The first attempt used raw posterior mean betas without sign correction:

| Method | A | B | Notes |
|--------|---|---|-------|
| Stocking-Lord | 703.44 | -15.46 | **Degenerate** |
| Haebara | 0.71 | -1.79 | Reasonable |
| Mean-Sigma | 0.65 | -0.08 | Reasonable |
| Mean-Mean | -0.41 | -3.35 | Negative A — sign issue |

The Stocking-Lord optimizer found a degenerate minimum, and Mean-Mean produced a negative A. The root cause: per-chamber models use `beta ~ Normal(0, 1)` which allows negative discrimination. When an anchor item has `a_house = 0.5` but `a_senate = -0.3` (the same bill discriminating in opposite directions due to sign indeterminacy), the TCC criterion is ill-conditioned and the optimizer exploits the curvature.

### The Fix: Sign-Aware Anchor Extraction

In standard IRT test equating, discrimination is always positive (constrained by design). Our unconstrained per-chamber betas introduce sign ambiguity. The fix is two-step:

1. **Filter**: Exclude anchor items where the two chambers disagree on sign direction (a bill with positive beta in one chamber and negative in the other violates the linking assumption — it may represent genuine differential item functioning, or just the sign indeterminacy of unconstrained IRT).

2. **Normalize**: For retained items, take `|beta|` as discrimination and flip alpha accordingly (the ICC `a*theta - b` is invariant under `(a, b) → (-a, -b)`).

This is implemented in `extract_anchor_params()` which now returns a `n_dropped` count for diagnostic transparency.

### Final Results (Sign-Filtered) — All Methods Agree

After sign-aware extraction, 40 of 67 anchor items are usable (27 dropped for cross-chamber sign disagreement):

| Method | A | B |
|--------|---|---|
| Stocking-Lord | 1.03 | 0.69 |
| Haebara | 0.76 | 0.58 |
| Mean-Sigma | 0.67 | 0.63 |
| Mean-Mean | 0.86 | 0.58 |

All four methods produce positive, reasonable A values (0.67-1.03). Rank-order correlation between all method pairs is r = 1.000 — the ideal point rankings are robust to method choice. The methods disagree on the *magnitude* of the scale stretch (A ranges from 0.67 to 1.03), but the relative ordering of legislators is identical.

Stocking-Lord A = 1.03 means the Senate scale is almost identical to the House scale, with a B = 0.69 rightward shift — consistent with the 84th Senate being slightly more conservative on average.

**Validation against joint model**: S-L linked ideal points correlate with joint model estimates at r = 0.967 (House) and r = 0.894 (Senate). Given that the joint model itself has questionable convergence (828 divergences), this agreement is encouraging — the linking recovers essentially the same information without MCMC risks.

**40% sign disagreement is notable.** Of 67 shared bills, 27 (40%) have opposite sign discrimination across chambers. This could reflect genuine DIF (the bill discriminates differently depending on chamber dynamics) or it could be an artifact of sign indeterminacy in unconstrained IRT. A proper DIF analysis would distinguish between these cases. Even after excluding 27 items, the remaining 40 anchors is well above the 20-30 minimum the literature recommends.

## Current Status and Remaining Work

### What Works

- **Per-chamber models**: Perfect convergence across all 8 bienniums. Rock-solid ideal points validated against Shor-McCarty (House r = 0.975, Senate r = 0.950).
- **exp(Normal) reparameterization**: Eliminates the LogNormal boundary catastrophe. The `lognormal_reparam` case in `BetaPriorSpec` provides positive beta without sampling pathology.
- **PCA initialization**: Eliminates the need for post-hoc sign correction.
- **IRT linking infrastructure**: All four methods implemented, sign-aware anchor extraction, integrated into the hierarchical pipeline. All methods agree on rank order (r = 1.000 pairwise). S-L linked vs joint: r = 0.97 (House), r = 0.89 (Senate).

### What Doesn't Work Yet

- **Joint model convergence**: R-hat(xi) = 1.0098 (borderline), ESS(xi) = 378 (below threshold), 828 divergences, hyperparameters failing. The 3-level hierarchy's funnel geometry is the remaining obstacle.

### Improvement Plan

The priorities are now:

1. **Validate sign-aware Stocking-Lord linking** on the 84th biennium. If Haebara and Stocking-Lord agree on A and B after sign filtering, the linking is robust.

2. **1PL joint model** as a simplified concurrent calibration. Drop beta entirely (fix beta = 1), reducing the model from ~1,042 to ~622 parameters and eliminating all sign-flip axes. The per-chamber 2PL models provide precise estimates; the 1PL joint model only needs to provide the cross-chamber scale alignment.

3. **Collapse the hierarchy** to two levels. Replace the chamber → party → legislator hierarchy with a simpler party → legislator hierarchy (4 groups), eliminating the poorly-identified `sigma_chamber` and `mu_chamber` parameters. The chamber distinction is already captured by the 4-group structure (HD, HR, SD, SR).

4. **DIF screening** for anchor items. Before linking, test each shared bill for differential item functioning. Bills that discriminate differently across chambers should be excluded. With 67 anchors, we can drop 10-15 and still have plenty.

5. **Normalizing flow adaptation** (nutpie experimental). If the joint model is worth pursuing, nutpie's NF feature was designed for exactly this class of problem (~1,000 parameter models with funnel geometry). Requires JAX backend.

### Recommendation

**Stocking-Lord linking is the production path.** The per-chamber models are excellent, the anchor items are plentiful, and the method is what the field actually uses. The joint model should continue as an experimental benchmark — if it eventually converges, it provides a theoretical gold standard — but the pipeline should not depend on it.

The key insight from these experiments: concurrent calibration (one big MCMC) is theoretically superior but practically fragile for legislative IRT. Separate-then-link (per-chamber MCMC + 2-parameter optimization) is theoretically inferior but robust, transparent, and well-validated by decades of psychometric practice.

---

**Related documents:**
- Joint model diagnosis: `docs/joint-hierarchical-irt-diagnosis.md` (bill-matching bug and fix)
- Convergence improvement plan: `docs/hierarchical-convergence-improvement.md` (per-chamber + joint theory)
- Positive beta experiment: `results/experimental_lab/2026-02-27_positive-beta/experiment.md`
- nutpie hierarchical experiment: `results/experimental_lab/2026-02-27_nutpie-hierarchical/experiment.md`
- External validation: `docs/external-validation-results.md` (per-chamber models validated)
- Hierarchical shrinkage: `docs/hierarchical-shrinkage-deep-dive.md`
- PCA init experiment: `docs/hierarchical-pca-init-experiment.md`
- Model specification: `analysis/07_hierarchical/model_spec.py`
- IRT linking: `analysis/07_hierarchical/irt_linking.py` (Stocking-Lord, Haebara, Mean-Sigma, Mean-Mean)
