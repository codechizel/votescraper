# Chapter 2: Dynamic Ideal Points: The Martin-Quinn Model

> *A legislator's ideology isn't frozen. People change. Can we track how?*

---

## The Limitation of Static IRT

Volume 4 estimated a single ideology score per legislator per biennium. That score was the best summary of their voting pattern across 500+ roll calls — but it was an average. A legislator who was moderate in their first biennium and conservative in their fifth got separate scores for each. The scores couldn't speak to each other because each biennium's IRT model was fit independently.

That independence is a problem. If Senator Smith scores +0.8 (moderately conservative) in the 88th Legislature and +1.2 in the 89th, did she genuinely shift rightward? Or did the 89th Legislature's scale just happen to stretch further, making the same voting pattern look more extreme? Without a shared scale, you can't answer the question.

**Dynamic IRT** solves this by fitting a single model across all eight bienniums (84th through 91st, 2011-2026), linking the scales through legislators who serve in multiple sessions. The result: a trajectory for every legislator who served in the Kansas Legislature over 15 years, measured on a single consistent scale.

## The State-Space Model

### The Core Idea

The model has one simple assumption: **a legislator's ideology today is their ideology yesterday, plus a small random step.**

Think of a random walk. Imagine standing at position 0 on a number line. Each year, you flip a coin: heads, you step one unit to the right; tails, one unit to the left. After 8 flips, you might be at +2, or -3, or back at 0. The path is unpredictable, but it's *continuous* — you can't teleport. Each step starts from where the last one ended.

Dynamic IRT treats ideology the same way. A legislator's ideal point in the 85th Legislature starts from where it was in the 84th, then drifts by a small random amount. The drift could be toward the party base, toward the center, or nowhere at all. The data (their votes) determines which direction.

### The Equations

The model has two parts:

**The evolution equation** (how ideology drifts):

```
xi[t] = xi[t-1] + tau * innovation[t-1]
innovation ~ Normal(0, 1)
```

**Plain English:** "A legislator's ideology at time *t* equals their ideology at time *t-1*, plus a random step whose size is controlled by tau."

Let's walk through each piece:

- **xi[t]**: The legislator's ideal point in biennium *t* (e.g., the 88th Legislature)
- **xi[t-1]**: Their ideal point in the previous biennium (the 87th Legislature)
- **tau**: The **evolution variance** — a single number that controls how big the steps can be. Small tau means ideology changes slowly; large tau means it can shift dramatically.
- **innovation**: A random draw from a standard normal distribution (mean 0, standard deviation 1). This is the "coin flip" — the direction and relative size of the step. In time series statistics, the random adjustment at each step is called an "innovation" — not because it's creative, but because it's the new, unpredictable part of the change. Think of it as the surprise component: after accounting for where a legislator was last session, the innovation is how much they unexpectedly shifted.

**A worked example:** Suppose Representative Jones has xi = +1.0 (moderately conservative) in the 87th Legislature. Tau for Republicans is 0.2. The model draws innovation = -0.5 (a half-step toward the center). Jones's expected ideal point in the 88th Legislature is:

```
xi[88th] = 1.0 + 0.2 × (-0.5) = 1.0 - 0.1 = 0.9
```

Jones drifted 0.1 units toward the center. If the innovation had been +1.5 (a large step rightward), the new position would be 1.0 + 0.2 × 1.5 = 1.3. Tau scales the step — a small tau of 0.1 would have made the same innovation produce only a 0.15-unit shift.

**The observation equation** (how votes reveal ideology):

```
P(vote = Yea) = logistic(beta * xi - alpha)
```

This is the same 2PL IRT from Volume 4. Each vote is a noisy observation of the legislator's current ideal point, filtered through the bill's difficulty (alpha) and discrimination (beta). The model sees thousands of votes across all bienniums and infers the trajectory that best explains the voting pattern.

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt.py` (`build_dynamic_irt_graph()`)

### Tau: How Fast Can Ideology Change?

Tau is arguably the most important parameter in the model. It's not assumed — it's **estimated from the data**.

- **Small tau (0.05-0.10):** Ideology barely budges between bienniums. The random walk is nearly frozen. This would describe a legislature where members vote the same way year after year.
- **Moderate tau (0.15-0.30):** Ideology drifts gradually. Over 8 bienniums, a legislator might shift by 1-2 units on the ideal point scale. This is the typical range for Kansas.
- **Large tau (> 0.50):** Ideology is volatile — legislators can shift dramatically between sessions. This would be unusual and might indicate model misspecification rather than genuine instability.

Tallgrass estimates tau separately for each party (in large chambers) or globally (in small chambers). This allows the model to detect that, for example, Republicans experienced more ideological churning than Democrats over the study period — or vice versa.

**The adaptive prior:** For the Kansas House (80+ legislators), tau gets a HalfNormal(0.5) prior per party — relatively permissive. For the Senate (40 legislators), the prior tightens to HalfNormal(0.15) with a single global tau. This prevents the sampler from finding spurious per-party volatility differences in a chamber with too few members to reliably estimate them.

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt.py` (`DEFAULT_TAU_SIGMA = 0.5`, `SMALL_CHAMBER_TAU_SIGMA = 0.15`, `SMALL_CHAMBER_THRESHOLD = 80`)

## Bridge Legislators: Anchoring the Scale

### Why Bridges Matter

If every legislator served for exactly one biennium, the model would have no way to link the scales. The 84th Legislature's ideal points and the 85th Legislature's ideal points would be estimated from entirely different groups of people, with no common reference.

**Bridge legislators** — lawmakers who serve in adjacent bienniums — solve this. A legislator who votes in both the 88th and 89th Legislatures provides a direct connection: the model observes their votes in both periods and estimates a trajectory that must be consistent across the boundary. With dozens of bridges per transition, the scales are firmly linked.

The analogy: imagine measuring the heights of students in two different classrooms, using two different rulers. If no student appears in both classrooms, you can't compare the measurements — maybe one ruler measures in centimeters and the other in inches. But if 50 students appear in both classrooms, you can use their measurements to calibrate the rulers against each other. The more shared students, the better the calibration.

### Kansas Bridge Coverage

In Kansas, bridge coverage is excellent. Adjacent bienniums typically share 70-80% of legislators:

| Transition | Shared Legislators | Overlap |
|-----------|-------------------|---------|
| 84th → 85th (post-redistricting) | ~60-70% | Weakest link (redistricting reshuffled districts) |
| Most other transitions | ~70-80% | Strong — the standard two-year election cycle |

The weakest link is the 84th → 85th transition, which followed the 2012 redistricting. Even there, the overlap far exceeds the minimum threshold of 5 shared legislators. The model doesn't need explicit anchor constraints (like the hard anchors in static IRT) because the bridges embed naturally in the likelihood: the model must explain the same legislator's votes in both periods with a trajectory that starts from one ideal point and random-walks to the next.

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt_data.py` (`compute_adjacent_bridges()`, `MIN_BRIDGE_OVERLAP = 5`)

## Identification: Keeping the Sign Straight

### The Three-Layer Strategy

Dynamic IRT has the same identification problem as static IRT (Volume 4, Chapter 3): PCA and IRT don't know which direction is "conservative." But the problem is harder because the sign could flip independently at each biennium. A model that thinks +2.0 is "conservative" in the 87th Legislature but "liberal" in the 88th would produce nonsensical trajectories.

Tallgrass uses a three-layer identification strategy:

**Layer 1: Positive beta constraint.** All bill discrimination parameters are constrained to be positive (beta ~ HalfNormal(2.5)). This fixes one degree of freedom: a bill that separates legislators must separate them in a consistent direction.

**Layer 2: Informative prior from static IRT.** The initial ideal points (xi at biennium 1) are given a prior centered on the static IRT posterior means from Volume 4. Since static IRT uses hard anchor constraints and has been thoroughly validated (Volume 5), its sign convention is trustworthy. The dynamic model inherits that convention through the prior, and the random walk propagates it forward through all subsequent bienniums.

The prior is intentionally loose (sigma = 1.5) — tight enough to transfer the sign convention, loose enough that the dynamic model can find its own scale:

```
xi_init ~ Normal(mu = static_IRT_mean, sigma = 1.5)
```

**Layer 3: Post-hoc safety net.** After sampling, Tallgrass correlates each biennium's dynamic ideal points with the corresponding static IRT. If any biennium shows a negative correlation (the sign flipped despite the informative prior), the model negates that biennium's posterior and documents the correction in the report. This layer should rarely fire — the informative prior handles identification in practice — but it provides a transparent safety mechanism.

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt.py` (`fix_period_sign_flips()`, `XI_INIT_PRIOR_SIGMA = 1.5`)

## The Sampling Budget

Dynamic IRT is a large model. The Kansas House version has roughly 10,000 parameters: ~250 legislators × 8 time periods, plus thousands of bill parameters and evolution variances. The sampler needs a generous budget to explore this space.

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Chains** | 4 | Standard for convergence diagnostics (R-hat requires ≥ 2) |
| **Tuning steps** | 2,000 per chain | Warm-up for the sampler to find the typical set |
| **Posterior draws** | 2,000 per chain | 8,000 total draws (4 chains × 2,000) for reliable estimates |
| **Sampler** | nutpie (Rust NUTS) | Same as all IRT models in the pipeline |

Convergence thresholds are relaxed relative to static IRT:

| Diagnostic | Static IRT (Vol. 4) | Dynamic IRT |
|-----------|---------------------|-------------|
| **R-hat** | < 1.01 | < 1.05 |
| **ESS bulk** | > 400 | > 400 |
| **Max divergences** | 0 | 50 |

The relaxation is principled. With ~10,000 parameters, demanding R-hat < 1.01 for every parameter would reject models that are substantively well-estimated but have a few slow-mixing bill parameters in low-information bienniums. The 84th Legislature (2011-2012) has ~30% committee-of-the-whole votes with no individual records, producing wider posteriors for legislators who served only that biennium — and wider posteriors are harder to push below R-hat 1.01.

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt.py` (`DEFAULT_N_SAMPLES = 2000`, `DEFAULT_N_TUNE = 2000`, `DEFAULT_N_CHAINS = 4`, `RHAT_THRESHOLD = 1.05`, `ESS_THRESHOLD = 400`, `MAX_DIVERGENCES = 50`)

## What the Results Show

### Trajectory Plots

The signature output is a **spaghetti plot** of individual trajectories: one line per legislator, colored by party, spanning the bienniums they served. Most lines are nearly flat — ideology is stable for most legislators across their careers. But a few show clear movement: a moderate who drifts toward the base, a conservative who mellows, an outlier who swings back and forth.

### Top Movers

For each legislator, Tallgrass computes two measures of movement:

- **Total movement:** The sum of absolute shifts between consecutive bienniums. A legislator who goes +1.0 → +0.5 → +1.2 has total movement of |−0.5| + |+0.7| = 1.2.
- **Net movement:** The difference between the first and last observed ideal point. The same legislator has net movement of +1.2 − 1.0 = +0.2 (slightly rightward overall).

A legislator with high total movement but low net movement oscillated — they moved a lot but ended up near where they started. A legislator with high total and high net movement drifted consistently in one direction.

### Polarization Trend

The **polarization trend** plot shows the gap between Republican and Democrat mean ideal points across all 8 bienniums, with 95% credible bands. A widening gap means the parties are growing further apart. A narrowing gap means convergence. The credible bands capture uncertainty from the MCMC posterior — if the bands overlap zero change, the trend isn't statistically meaningful.

### Ridgeline Plot

A **ridgeline** (or "joy plot") stacks the ideal point distributions for each biennium vertically, with party-colored density curves. This shows not just the party means but the full shape of each party's ideology distribution: is it concentrated (most members agree) or spread out (internal factions)? Did the distribution shift, or did it just widen?

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt.py` (`identify_top_movers()`, `TOP_MOVERS_N = 20`)

## Validation: Do the Dynamic Scores Agree with Static IRT?

Dynamic and static IRT should agree within each biennium — they're estimating the same thing (legislator ideology) from the same data (votes), just with different model structures. Tallgrass validates this by correlating each biennium's dynamic ideal points with the corresponding static IRT posterior means.

Target: Pearson r > 0.90 per biennium. This is a high bar, and meeting it means the dynamic model is capturing the same ideological structure as the static model while adding the cross-biennium linkage.

If a biennium shows r < 0.90, it's typically the 84th (data quality issues) or a Senate session affected by the horseshoe artifact. These are documented in the report rather than treated as failures.

**Codebase:** `analysis/27_dynamic_irt/dynamic_irt_data.py` (`correlate_with_static()`)

## What Can Go Wrong

### 84th Legislature Data Quality

The 84th Legislature (2011-2012) has approximately 30% committee-of-the-whole votes — floor votes where individual legislator positions aren't recorded. This means fewer data points per legislator in that biennium, producing wider posterior credible intervals. Legislators who served only the 84th have noisier estimates than those who served later sessions. The model handles this correctly (wider uncertainty when data is sparse), but readers should interpret 84th-specific results with appropriate caution.

### Senate Mode-Splitting

With only ~40 legislators per biennium, the Kansas Senate is a small sample for a model with per-party evolution variance. The sampler can "mode-split" — finding two posterior modes (one where Republican tau is high and Democrat tau is low, and vice versa) and failing to mix between them. Tallgrass mitigates this by switching to a single global tau for chambers below 80 legislators, with a tighter prior (HalfNormal(0.15) instead of HalfNormal(0.5)).

### The 84th → 85th Bridge

The post-2012 redistricting reduced the overlap between the 84th and 85th Legislatures. While still above the minimum threshold, this is the weakest link in the 8-biennium chain. The model's uncertainty about the 84th-to-85th scale calibration is slightly higher than for other transitions, reflected in wider credible bands on the polarization trend during that period.

### Non-Centered Parameterization: A Technical Necessity

The random walk produces a statistical phenomenon called the **funnel geometry**: when tau is small, the innovations must also be small (otherwise the trajectory would drift too fast), creating a narrow region of high probability that standard samplers struggle to explore. Tallgrass uses **non-centered parameterization** — sampling the raw innovations (always standard normal) and multiplying by tau after the fact — which eliminates the funnel and allows the sampler to explore efficiently. This is a standard technique in Bayesian hierarchical modeling, and its absence would cause convergence failures.

---

## Key Takeaway

Dynamic IRT extends Volume 4's static model across time by introducing a random walk: each legislator's ideology drifts between bienniums, with the speed of drift (tau) estimated from data. Bridge legislators who serve in multiple sessions anchor the cross-biennium scale, while a three-layer identification strategy (positive beta, informative prior, post-hoc check) keeps the sign convention consistent. The result is a 15-year trajectory for every Kansas legislator — revealing who moved, how fast, and in which direction.

---

*Terms introduced: dynamic ideal point, state-space model, random walk, evolution equation, observation equation, evolution variance (tau), bridge legislator, bridge coverage, informative prior (from static IRT), post-hoc sign correction, non-centered parameterization, funnel geometry, total movement, net movement, polarization trend, ridgeline plot, mode-splitting*

*Previous: [Time Series: Detecting When the Legislature Shifts](ch01-time-series.md)*

*Next: [Cross-Session Alignment: Putting Sessions on the Same Scale](ch03-cross-session-alignment.md)*
