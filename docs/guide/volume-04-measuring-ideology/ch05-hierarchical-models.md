# Chapter 5: Partial Pooling: Hierarchical Models and Party Structure

> *If you know a legislator is a Republican, that tells you something about where they'll fall on the ideology spectrum — but it doesn't tell you everything. Hierarchical models use party information as a starting point, then let the data pull legislators away from their party average.*

---

## The Information You're Leaving on the Table

The flat IRT models from Chapters 2-4 treat every legislator as completely independent. They estimate each ideal point from scratch, using only that legislator's own voting record. This is the statistical equivalent of estimating each student's test score without considering which school they attend, which class they're in, or what curriculum they followed.

But we *know* something before looking at a single vote. We know party membership. And party membership is enormously informative in American legislatures — it predicts about 70-80% of vote choices in Kansas. Ignoring this information is like knowing a patient is 6'5" and refusing to account for it when estimating their weight.

Hierarchical models fix this by encoding a simple idea: **legislators within the same party tend to be similar, but they aren't identical.** Each legislator's ideal point starts at their party's average, then adjusts based on their individual voting record. The more votes a legislator has, the more the data overrides the party average. The fewer votes they have, the more they're pulled toward the party center.

This behavior — borrowing strength from the group while preserving individual variation — is called **partial pooling**.

## The Pooling Spectrum

To understand partial pooling, consider two extreme alternatives:

### Complete Pooling (Everyone Is the Same)

Give every Republican the same ideal point. Give every Democrat the same ideal point. Ignore individual differences entirely.

This is obviously too simple. Susan Wagle and Sandy Praeger were both Kansas Senate Republicans, but their voting records looked very different. Complete pooling erases this distinction.

### No Pooling (Everyone Is Independent)

Estimate each legislator's ideal point using only their own votes. Ignore party membership entirely. This is what the flat IRT model does.

This works well when every legislator has hundreds of votes. But for a legislator who joined the chamber mid-session and cast only 30 votes, the estimate is noisy — the credible interval is wide, and the point estimate bounces around based on which 30 votes they happened to cast. Meanwhile, the model is ignoring the highly relevant fact that this legislator is a Republican in a chamber where Republicans vote together 85% of the time.

### Partial Pooling (The Best of Both)

Start with the party average, then adjust based on individual data. Legislators with many votes pull away from the party mean (their individual record dominates). Legislators with few votes stay close to the party mean (the group provides stability).

**The analogy: predicting a person's height.**

If you know nothing about someone, your best guess is the population average: about 5'7". If you learn they're a professional basketball player, you update: NBA players average about 6'6". But not all NBA players are 6'6" — some are 6'0" point guards, others are 7'1" centers. Knowing the team average gives you a much better starting point than knowing nothing, but you'll still adjust based on the individual.

Partial pooling is the formalization of this common sense. The population average (party mean) provides a starting point. Individual data (votes) refines it. The balance between the two depends on how much data you have and how much individuals vary within the group.

| Approach | Ideal Point Source | Strength | Weakness |
|----------|-------------------|----------|----------|
| Complete pooling | Party average only | Stable | Ignores individual variation |
| No pooling | Individual votes only | Captures individuals | Noisy for sparse data |
| **Partial pooling** | **Party average + individual votes** | **Stable and individualized** | **Requires a model of group structure** |

## The Hierarchical IRT Equation

The Tallgrass hierarchical model adds a **party layer** above the individual legislators.

**Plain English:**

> "Each party has an average ideology. Each legislator's ideology starts at their party's average, then shifts based on their personal voting record. How far a legislator can stray from their party depends on how much within-party variation the data supports."

**Equations (the party level):**

```
μ_party  ~  Normal(0, 2), ordered so that Democrat < Republican
σ_within  ~  HalfNormal(σ_scale)
```

- **μ_party:** The average ideology for each party. The ordering constraint (`Democrat < Republican`) provides identification — it ensures the model always places Democrats to the left of Republicans.

- **σ_within:** How much individual legislators vary around their party's average. A small σ_within means the party is ideologically tight (all members vote alike). A large σ_within means the party is a broad tent with diverse views.

**Equations (the legislator level):**

```
offset_i  ~  Normal(0, 1)
ξ_i  =  μ_party[p_i]  +  σ_within[p_i]  ·  offset_i
```

- **offset_i:** How many standard deviations this legislator is from their party's average. An offset of +1.5 means they're 1.5 standard deviations more conservative than their party mean. An offset of −0.8 means they're 0.8 standard deviations more liberal.

- **ξ_i:** The final ideal point, computed as the party mean plus the individual offset scaled by the within-party spread.

**The bill-level parameters** (α and β) remain the same as in the flat model:

```
α  ~  Normal(0, 5)      — bill difficulty
β  ~  Normal(0, 1)      — bill discrimination
```

**Worked example:**

Suppose the model estimates:
- Republican party mean: μ_R = +0.9
- Democratic party mean: μ_D = −1.1
- Republican within-party SD: σ_R = 0.6
- Democratic within-party SD: σ_D = 0.4

A moderate Republican with an offset of −0.5 has:
```
ξ = μ_R + σ_R · offset = 0.9 + 0.6 × (−0.5) = 0.9 − 0.3 = +0.6
```

A progressive Democrat with an offset of −1.2 has:
```
ξ = μ_D + σ_D · offset = −1.1 + 0.4 × (−1.2) = −1.1 − 0.48 = −1.58
```

Both legislators' ideal points are informed by their party's average but adjusted by their individual offset. The Republican starts at +0.9 and is pulled toward the center (−0.3). The Democrat starts at −1.1 and is pulled further left (−0.48).

## Non-Centered Parameterization: A Technical Necessity

The equations above use a technique called **non-centered parameterization** that's worth understanding, because it's the reason the hierarchical model can converge at all.

### The Problem: The Funnel of Hell

In the centered version of the model, you'd write the legislator ideal point directly:

```
ξ_i  ~  Normal(μ_party[p_i],  σ_within[p_i])
```

This looks simpler, but it creates a geometric trap for the MCMC sampler. When σ_within is small (tight party discipline), the ideal points must cluster tightly around the party mean. The sampler needs to take tiny steps. But when σ_within is large (loose party discipline), the ideal points can range widely. The sampler needs to take large steps.

The problem: the sampler has to explore regions of the posterior where σ_within is both small and large, and it can't simultaneously use tiny steps and large steps. The result is a **funnel-shaped posterior** — a narrow neck (small σ_within, clustered ideal points) that the sampler struggles to navigate.

### The Analogy: Navigating a Funnel

Imagine hiking through a canyon that narrows into a bottleneck. In the wide part, you can stride freely. In the narrow part, you need to shuffle sideways. A standard hiker can't do both efficiently — they either overshoot in the narrow part (crashing into walls) or undershuffle in the wide part (wasting steps).

### The Solution: Change Coordinates

Non-centered parameterization sidesteps the funnel by changing variables. Instead of sampling the ideal point directly, we sample a **standardized offset** (always between roughly −3 and +3, regardless of σ_within) and multiply by σ_within afterward:

```
offset_i  ~  Normal(0, 1)                             ← always the same scale
ξ_i  =  μ_party[p_i]  +  σ_within[p_i]  ·  offset_i  ← combine after sampling
```

Now the sampler only needs to navigate a standard Normal distribution for each offset — a well-behaved landscape with no funnels. The funnel geometry is algebraically equivalent but numerically tractable.

This is one of those cases where mathematically identical formulations have profoundly different computational properties. The centered and non-centered parameterizations define the same statistical model, but only the non-centered version lets the MCMC sampler do its job.

**Codebase:** `analysis/07_hierarchical/hierarchical.py` — the non-centered parameterization is implemented with `xi_offset ~ Normal(0, 1)` and `xi = mu_party[party_idx] + sigma_within[party_idx] * xi_offset`

## Shrinkage: The Signature of Partial Pooling

The most visible effect of partial pooling is **shrinkage** — the tendency for estimates with less data to be pulled toward the group average.

### How Shrinkage Works

Consider two Kansas House Democrats:

- **Veteran:** 15-term representative, 480 contested votes in the current session
- **Freshman:** Appointed mid-session, only 40 contested votes

The flat (no pooling) model estimates each independently:
- Veteran ideal point: −1.35, 95% HDI [−1.52, −1.18]
- Freshman ideal point: −0.62, 95% HDI [−1.31, +0.08]

The freshman's estimate is noisy (wide HDI, including positive values that would make them look Republican) and based on a small sample. They might genuinely be moderate, or they might just have happened to cast their 40 votes on bills that don't distinguish ideology well.

The hierarchical model applies shrinkage:
- Veteran ideal point: −1.33, 95% HDI [−1.49, −1.17] (barely changed — lots of data)
- Freshman ideal point: −0.91, 95% HDI [−1.38, −0.45] (pulled toward party mean, narrower HDI)

The veteran's estimate barely budged — 480 votes speak for themselves, so the party average has little influence. But the freshman's estimate was pulled from −0.62 toward the Democratic party mean (about −1.1), settling at −0.91. And the HDI narrowed, excluding the implausible positive values.

### The Analogy: The Baseball Batting Average

This is exactly the logic behind **empirical Bayes** in baseball statistics. A player who goes 3-for-5 in their first week has a 0.600 batting average. Nobody expects them to hit 0.600 for the full season. Their "true" ability is pulled toward the league average (~0.260), resulting in a "regressed" estimate of maybe 0.340.

A player who is 150-for-500 at mid-season (0.300) barely gets adjusted — 500 at-bats provide strong evidence. But the 5-at-bat player gets pulled heavily toward the population average because the individual data is too thin to trust on its own.

Shrinkage is not a flaw — it's a feature. It reflects the perfectly rational belief that an extreme estimate based on little data is more likely to be noise than signal.

### Measuring Shrinkage

The amount of shrinkage for a given legislator depends on two things:

1. **How many votes they cast** — more votes = less shrinkage
2. **How variable the party is** — more within-party variation = less shrinkage (the model is less confident about the party mean)

The **Intraclass Correlation Coefficient (ICC)** captures how much of the variation in ideal points is between parties versus within parties:

```
ICC = σ²_between / (σ²_between + σ²_within)
```

**Plain English:** "What fraction of ideological variation is explained by party membership?"

For the Kansas Senate, ICC is typically around **0.70** — party membership explains about 70% of ideological variation. The remaining 30% is individual. This means the hierarchical model provides substantial shrinkage: party is a strong signal.

For the Kansas House, ICC is somewhat lower (~0.55-0.65) because the larger chamber allows for more within-party diversity.

## The Small-Caucus Problem

Partial pooling works best when each group has enough members for the group average to be estimated precisely. But what happens when a group is very small?

The Kansas Senate typically has 10-15 Democrats — a small caucus. With only 10-15 members, the within-party standard deviation (σ_within for Democrats) is estimated from a tiny sample. This makes σ_within itself uncertain, which in turn makes the shrinkage strength uncertain.

In extreme cases, the sampler can get stuck: σ_within drifts toward zero (all Democrats identical) or toward infinity (no shrinkage at all), bouncing between these extremes without settling.

### Adaptive Priors

Tallgrass addresses this with **adaptive priors** for small groups. When a party has fewer than 20 members, the prior on σ_within is tightened from HalfNormal(1.0) to HalfNormal(0.5). This gently nudges the model toward moderate within-party variation, preventing the sampler from drifting to extreme values.

```
Standard prior:   σ_within ~ HalfNormal(1.0)    — for groups ≥ 20 members
Adaptive prior:   σ_within ~ HalfNormal(0.5)    — for groups < 20 members
```

The tighter prior doesn't force anything — it just says "within-party variation is probably moderate" rather than "it could be anything." With 10 Democrats, this modest assumption provides enough stability for the sampler to converge.

**Codebase:** `analysis/07_hierarchical/hierarchical.py` (`sigma_scale` adapts based on group size; the adaptive prior follows Gelman 2015)

## Identification in the Hierarchical Model

The hierarchical model uses a different identification strategy than the flat model. Instead of anchoring specific legislators, it uses an **ordering constraint on party means**:

```
μ_party = sort(μ_party_raw)     — forces Democrat mean < Republican mean
```

This single constraint accomplishes three things:

1. **Breaks reflection invariance:** If you flip all ideal points, the ordering constraint is violated (Republican mean would become smaller than Democratic mean). Only one of the two mirror-image solutions satisfies the constraint.

2. **Sets the direction:** The constraint establishes the convention that positive = conservative, negative = liberal.

3. **Is soft enough to sample:** Unlike hard anchors (which fix specific ideal points), the ordering constraint allows the party means to move freely as long as they maintain their ordering. This gives the sampler more room to maneuver.

An additional **soft minimum-separation penalty** ensures the party means don't collapse to the same value:

```
Penalty = switch(μ_R − μ_D > 0.5, 0.0, −100.0)
```

**Plain English:** "If the Republican mean is at least 0.5 points above the Democratic mean, no penalty. If the gap shrinks below 0.5, a severe penalty discourages this." This prevents the degenerate solution where both parties have the same mean (complete pooling with no party distinction).

## Hierarchical 2D: The Full Model

Tallgrass also fits a **hierarchical 2D** model (Phase 07b) that combines the 2D structure from Chapter 4 with the party-level pooling from this chapter. This is the most complex model in the pipeline.

The hierarchical 2D model has party-level parameters for *both* dimensions:

```
Party level (Dimension 1 — ideology):
  μ_party_dim1 = sort(μ_party_raw_dim1)    — D < R identification
  σ_party_dim1 ~ HalfNormal(σ_scale)

Party level (Dimension 2 — establishment):
  μ_party_dim2 ~ Normal(dim2_avg, 2.0)     — no ordering (not a party axis)
  σ_party_dim2 ~ HalfNormal(σ_scale)

Legislator level:
  ξ_dim1 = μ_party_dim1[party] + σ_party_dim1[party] · offset_dim1
  ξ_dim2 = μ_party_dim2[party] + σ_party_dim2[party] · offset_dim2
```

Notice that Dimension 1 gets the ordering constraint (D < R) while Dimension 2 does not. This makes sense: Dimension 1 is ideology, which aligns with party. Dimension 2 is establishment loyalty, which doesn't.

The hierarchical 2D model is the **preferred source** for canonical ideal points when it converges well. It combines the horseshoe resolution of 2D models with the shrinkage benefits of hierarchical models — the best of both worlds.

**Codebase:** `analysis/07b_hierarchical_2d/hierarchical_2d.py`

## What Partial Pooling Tells Us About Kansas

The hierarchical model's group-level parameters are interesting in their own right. They answer questions about party structure:

### How Far Apart Are the Parties?

The gap between μ_R and μ_D (the party means) is a direct measure of **between-party polarization**. In the 91st House, this gap is about 2.1 points — meaning the average Republican and average Democrat are 2.1 standard deviations apart on the ideology scale.

### How Unified Is Each Party?

The within-party SD (σ_within) measures **intra-party diversity**. A small σ_within means the party votes as a bloc; a large one means there's significant internal disagreement.

In Kansas:
- Republicans: σ_within ≈ 0.60 (moderate diversity — the caucus has moderates and conservatives)
- Democrats: σ_within ≈ 0.40 (tighter — the smaller caucus is more ideologically cohesive)

The asymmetry makes intuitive sense. The Republican caucus is larger and includes members from both moderate suburban districts and conservative rural ones. The Democratic caucus is smaller and more ideologically concentrated, drawn largely from urban areas (Kansas City, Lawrence, Wichita).

### How Much Does Party Explain?

The ICC of ~0.70 for the Senate tells us that knowing someone's party tells you about 70% of what you need to know about their ideology. The remaining 30% is the individual variation that makes politics interesting — the moderate Republicans, the maverick Democrats, the freshman who surprises everyone.

---

## Key Takeaway

Hierarchical models improve on flat IRT by sharing information across party members. Legislators with many votes are estimated primarily from their individual record; legislators with few votes are pulled toward their party's average. This shrinkage is automatic and proportional to data strength. Non-centered parameterization ensures the sampler can navigate the hierarchical geometry, and adaptive priors handle small groups like the Kansas Senate Democrats. The hierarchical 2D model combines pooling with multidimensional structure, providing the most complete picture of legislative ideology.

---

*Terms introduced: partial pooling, complete pooling, no pooling, hierarchical model, party-level mean (μ_party), within-party standard deviation (σ_within), non-centered parameterization, funnel geometry, shrinkage, Intraclass Correlation Coefficient (ICC), adaptive prior, ordering constraint, minimum-separation penalty, hierarchical 2D IRT*

*Next: [The Identification Zoo: Seven Strategies](ch06-identification-zoo.md)*
