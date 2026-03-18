# Chapter 5: Quality Gates: Automatic Trust Levels

> *A model's job isn't done when it finishes running. It still has to prove it's trustworthy enough to publish. The quality gate system is the pipeline's last line of defense — an automated inspector that decides whether each model's output meets the bar.*

---

## Why Automation Matters

Everything in the previous four chapters requires human judgment to interpret. A Pearson correlation of 0.93 — is that good enough? A Q3 violation rate of 8% — should we worry? A Pareto k value of 0.72 — does that invalidate the LOO comparison?

For a research project analyzing one or two sessions, human judgment works fine. But Tallgrass covers **fourteen bienniums** across two chambers — 28 chamber-sessions in total. Each chamber-session runs up to four IRT models. That's potentially 112 model fits to evaluate. Doing this by hand for every pipeline run would be slow, inconsistent, and error-prone.

The quality gate system automates these decisions. It doesn't replace human judgment — the PPC and external validation reports are still there for humans to review. But it provides a **default decision** that the pipeline uses to choose which model's ideal points to publish as canonical.

Think of it like the automated quality checks at a factory. Skilled inspectors still review the final product, but the automated system catches the obvious failures before a human ever needs to look at them.

## The Routing Problem, Revisited

As explained in Volume 4, Chapter 7, the pipeline produces up to four ideal point estimates for each chamber-session:

| Model | What It Captures | When It's Best |
|-------|-----------------|---------------|
| 1D Flat IRT | Single ideology axis | Balanced chambers with a clean party divide |
| 2D Flat IRT, Dim 1 | Ideology separated from a second axis | Supermajority chambers where 1D conflates ideology with establishment loyalty |
| Hierarchical 1D IRT | Ideology with party-based shrinkage | Balanced chambers where party provides useful prior info |
| Hierarchical 2D IRT, Dim 1 | Both dimensions + party shrinkage | Supermajority chambers with clear party structure |

The question: **which estimate should downstream phases (clustering, network analysis, temporal trends, legislator profiles) use?**

The routing preference is:

```
1st choice: Hierarchical 2D, Dim 1  (most structure)
2nd choice: Flat 2D, Dim 1          (2D without pooling)
3rd choice: 1D IRT                  (simplest, most robust)
```

Prefer complexity — but only when it converges. A complex model that didn't converge is worse than a simple model that did. This is codified in the three-tier quality gate.

## The Three Tiers

The quality gate evaluates each model against two diagnostics: **R-hat** (convergence) and **party separation** (substantive validity). Let's unpack each tier.

### Tier 1: Fully Converged — Full Trust

| Diagnostic | Threshold |
|-----------|-----------|
| R-hat (maximum across all ideal point parameters) | < 1.10 |
| Effective Sample Size (minimum across all ideal point parameters) | > 100 |

**What these mean, in plain language:**

R-hat measures whether the MCMC chains agree with each other. Recall from Volume 4 that MCMC works by running multiple independent "random walkers" through the parameter space. If they all end up in the same neighborhood, R-hat is close to 1.0 — the chains converged. If they're scattered across different regions, R-hat is large — the chains haven't found the answer yet.

An analogy: imagine sending four friends to explore a new city and find the best restaurant. If they all independently discover the same restaurant (R-hat ≈ 1.0), it's probably genuinely the best. If they each recommend a different restaurant (R-hat >> 1), maybe they didn't explore enough, or maybe there's no single "best."

The threshold of 1.10 is a standard convention in Bayesian statistics — originally proposed by Andrew Gelman and Donald Rubin in 1992. It says: the between-chain variance should be no more than 10% larger than the within-chain variance. More conservative practitioners use 1.05 or even 1.01, but 1.10 has stood the test of time as a practical cutoff.

**Effective Sample Size (ESS)** measures how many independent pieces of information the chains provide. MCMC chains are autocorrelated — each sample is similar to the previous one. ESS discounts for this autocorrelation to estimate the true information content. An ESS of 100 means the chains provide as much information as 100 completely independent samples, which is enough for reliable point estimates and credible intervals.

**Tier 1 implication:** Use the model's ideal points with full confidence. Both point estimates (the means) and uncertainty (the credible intervals) are trustworthy.

In the code (`analysis/canonical_ideal_points.py`):

```python
TIER1_RHAT_THRESHOLD = 1.10
TIER1_ESS_THRESHOLD = 100

if xi_rhat < TIER1_RHAT_THRESHOLD and xi_ess > TIER1_ESS_THRESHOLD:
    result["tier"] = 1
    result["usable"] = True
```

### Tier 2: Point Estimates Credible — Use With Caution

| Diagnostic | Threshold |
|-----------|-----------|
| R-hat | < 2.50 |
| Party separation (Cohen's d) | > 1.5 |

Tier 2 handles a common scenario: **the model didn't fully converge, but the point estimates are still usable.**

This happens frequently with 2D models. The second dimension is often weakly identified (there's less signal to estimate it), which means the MCMC chains for Dimension 2 parameters may not converge perfectly. But Dimension 1 — the ideology axis we actually care about — may still be well-estimated, even if the overall R-hat (computed across all parameters, including Dimension 2) exceeds 1.10.

The key additional check: **party separation**. This measures the standardized distance between the average Republican ideal point and the average Democrat ideal point, using Cohen's d:

```
Cohen's d = |mean(R) - mean(D)| / pooled_SD
```

**Why party separation, not just R-hat?**

R-hat tells you whether the chains converged. It doesn't tell you whether they converged to *the right answer*. A model could converge beautifully to a set of ideal points that don't separate the parties — meaning it captured something other than ideology (perhaps seniority, or regional affiliation, or random noise).

The party separation check is a **substantive validity gate**: it asks whether the model's output makes political sense. In any state legislature, the parties should be separated on the ideology axis. If Democrats and Republicans overlap heavily (Cohen's d < 1.5), the model is either capturing the wrong dimension or is too noisy to be useful.

This check was specifically designed to catch a subtle problem documented in ADR-0118: **PCA axis instability.** In 7 of 14 Senate sessions, PCA's first component captures intra-Republican factionalism rather than the party divide. If a 2D model is initialized from this misaligned PCA, its Dimension 1 may also be misaligned. The party separation check catches this — if Dimension 1 doesn't separate parties, it's not measuring ideology, regardless of what R-hat says.

**Tier 2 implication:** The ideal point rankings (who is more conservative than whom) are trustworthy, but the credible intervals should be treated with skepticism. The model knows the *order* but not the *precision*.

```python
TIER2_RHAT_THRESHOLD = 2.50
TIER2_PARTY_D_THRESHOLD = 1.5

if xi_rhat < TIER2_RHAT_THRESHOLD:
    party_d = _compute_party_separation(ip_2d)
    if party_d is not None and party_d > TIER2_PARTY_D_THRESHOLD:
        result["tier"] = 2
        result["usable"] = True
```

### Tier 3: Failed — Fall Back

Anything that doesn't meet Tier 2 criteria falls to Tier 3: the model's output is not trustworthy, and the pipeline falls back to the next-simplest model.

Common Tier 3 scenarios:

- **R-hat > 2.50:** The chains are in fundamentally different regions of parameter space. The model didn't converge at all.
- **Party separation d < 1.5:** The model converged to something, but it's not ideology. Dimension 1 may be capturing seniority, geographic region, or noise.
- **No convergence data available:** The model crashed or was skipped.

**Tier 3 implication:** Don't use this model. Fall back to the next-simpler option in the routing hierarchy.

## The Routing Decision in Practice

Here's how the quality gate plays out across the 28 chamber-sessions in the Tallgrass pipeline:

### Balanced Chambers (Typical Case)

For a balanced chamber like the Kansas House (where Democrats hold about 30% of seats), the routing often looks like:

1. **Check Hierarchical 2D (Phase 07b):** R-hat = 1.32, party d = 5.8 → **Tier 2**, usable
2. **Since Tier 2 is usable, use it as canonical**

The 2D hierarchical model's R-hat exceeds the Tier 1 cutoff (1.10) but passes Tier 2 (R-hat < 2.50, party separation excellent). The point estimates are used; the credible intervals are flagged as wider than ideal.

### Supermajority Chambers (Where Routing Matters Most)

For a supermajority chamber like the Kansas Senate (where Republicans hold 70%+ of seats), the 1D model faces the horseshoe problem (Volume 3, Chapter 5). The routing:

1. **Check Hierarchical 2D (Phase 07b):** R-hat = 1.08, party d = 6.2 → **Tier 1**, fully converged
2. **Use Hierarchical 2D Dim 1 as canonical** — it correctly separates ideology from the establishment axis

Without the quality gate, the pipeline might default to 1D IRT, which in a supermajority chamber conflates ideology with party loyalty — placing moderate Republicans next to Democrats in a horseshoe pattern that doesn't reflect genuine ideological proximity.

### The Worst Case: Everything Fails

In rare cases (typically very small chambers or sessions with unusual voting patterns), all complex models fail:

1. **Check Hierarchical 2D:** R-hat = 4.2 → Tier 3, skip
2. **Check Flat 2D:** R-hat = 3.1 → Tier 3, skip
3. **Use 1D IRT as canonical** (it almost always converges, being the simplest model)

The 1D model is the safety net. It may not capture subtle multidimensional structure, but it reliably produces a meaningful left-to-right ordering.

## Horseshoe Detection: The Upstream Guard

The quality gate system operates *after* models are fit. But for the horseshoe problem, there's also an *upstream* detector that runs before the complex models, alerting the pipeline to expect trouble.

### What the Horseshoe Is (Recap)

In a supermajority chamber, 1D ideal points can produce a **horseshoe pattern**: the minority party (Democrats) gets folded back toward the center or even the conservative end of the scale. This happens because the 1D model can't distinguish "votes with the majority on everything" (ideological agreement) from "votes with the majority on everything because there's no viable opposition" (strategic acquiescence).

The result: some Democrats get ideal points that look moderate or even conservative — not because they are, but because their voting pattern (always on the losing side of lopsided votes) is hard for a 1D model to separate from a moderate Republican's pattern (sometimes on the losing side).

### The Three Detection Checks

Tallgrass detects the horseshoe using three metrics computed from the 1D ideal points:

**1. Democrat wrong-side fraction (> 20%)**

What fraction of Democrats have positive (conservative) ideal points? In a healthy model, Democrats should be on the negative (liberal) side. If more than 20% are on the conservative side, the model has likely conflated the parties.

**2. Party overlap fraction (> 30%)**

How much do the two parties' ideal point distributions overlap? Some overlap is normal (there are moderate legislators in both parties), but more than 30% suggests the model isn't cleanly separating the ideological groups.

**3. Most extreme Republican check**

Is the most negative Republican ideal point more liberal than the Democrat mean? If a Republican is being placed further left than the average Democrat, something has gone wrong with the ideological axis.

From the code:

```python
HORSESHOE_DEM_WRONG_SIDE_FRAC = 0.20
HORSESHOE_OVERLAP_FRAC = 0.30

detected = (
    d_wrong_side > HORSESHOE_DEM_WRONG_SIDE_FRAC
    or r_more_liberal_than_d_mean
    or overlap_frac > HORSESHOE_OVERLAP_FRAC
)
```

If any of these checks fire, the session is flagged as horseshoe-affected, and the 2D models become the preferred source for canonical ideal points (if they converge).

## The Seven Safety Checks (R1–R7)

The quality gate and horseshoe detection are part of a broader system of **seven party-separation quality gates** documented in ADR-0118. These checks operate at different stages of the pipeline, forming a layered defense against axis confusion:

| Gate | Where | What It Checks |
|------|-------|---------------|
| R1 | PCA initialization | Party-aware sign convention: PC1 should separate parties |
| R2 | 1D IRT | Party Cohen's d after estimation: adequate separation? |
| R3 | Tier 2 quality gate | Party d replaces PCA rank correlation (avoids circular dependency) |
| R4 | Hierarchical IRT | Minimum party separation guard before shrinkage |
| R5 | 2D IRT | Dimension swap detection: is Dim 1 really ideology? |
| R6 | Dynamic IRT | Canonical reference alignment + per-period party d |
| R7 | Cross-session | Per-period sign correction using party separation |

The key insight behind this system: **no single check is sufficient.** The horseshoe problem can manifest at any stage of the pipeline. A session might pass the PCA check (R1) because PC1 happens to separate parties, but fail the 1D IRT check (R2) because the MCMC sampler found a different local mode. Or it might pass R2 but fail R5 because the 2D model swapped its dimensions during estimation.

By layering seven checks across the pipeline, Tallgrass ensures that axis confusion is caught *somewhere* — even if one or two individual checks happen to miss it.

### The PCA Axis Instability Story

This seven-gate system was developed in response to a specific discovery: in 7 of 14 Kansas Senate sessions (the 78th through 83rd and the 88th), PCA's first principal component captures **intra-Republican factionalism** rather than the party divide.

In these sessions, the biggest source of voting variation isn't "Republican vs. Democrat" (because Republicans win almost every vote) but "moderate Republican vs. conservative Republican" (because the party's internal disagreements drive the close votes). PCA, being a purely mathematical technique that finds the axis of maximum variance, correctly identifies this as the dominant pattern — but it's the wrong pattern for ideology measurement.

Before the seven-gate system was built, this misalignment could propagate through the pipeline: PCA misaligns → IRT is initialized from misaligned PCA → hierarchical model inherits the misalignment → canonical ideal points are wrong. The gates catch and correct this at every stage.

The full technical analysis is documented in `docs/pca-ideology-axis-instability.md`. The lesson for the general reader: **statistical correctness and substantive correctness aren't the same thing.** PCA is mathematically correct in these sessions — it really does find the axis of maximum variance. But maximum variance isn't the same as maximum ideological relevance. The quality gate system bridges this gap by adding domain knowledge (parties should be separated on the ideology axis) as a validation check on the purely mathematical output.

## The Routing Manifest

When the canonical routing module finishes, it writes a **routing manifest** — a JSON file that records every decision:

```json
{
  "House": {
    "source": "hierarchical_2d_dim1",
    "tier": 1,
    "rhat": 1.08,
    "ess": 487,
    "party_d": 6.2,
    "horseshoe_detected": false
  },
  "Senate": {
    "source": "flat_2d_dim1",
    "tier": 2,
    "rhat": 1.45,
    "ess": 89,
    "party_d": 4.8,
    "horseshoe_detected": true
  }
}
```

This manifest serves two purposes:

1. **Downstream phases read it** to know which ideal points to load. Instead of hardcoding "use Phase 05" or "use Phase 07b," downstream phases call `load_horseshoe_status()` to read the manifest and load the appropriate source.

2. **Human auditors read it** to understand *why* the pipeline chose a particular model. If the final profiles for the 88th Senate look unusual, the manifest explains: "horseshoe detected, routed to 2D Dim 1, Tier 2 convergence with party d = 4.8."

Transparency is the goal. The pipeline makes the decision, but it shows its work.

## Validation as a System, Not a Step

This chapter — and this volume — has described validation as a collection of techniques: PPCs, external benchmarks, convergence diagnostics, quality gates. But the deeper lesson is that validation isn't a single step you do after fitting a model. It's a **system** that pervades the entire pipeline.

The scraper validates data at ingest (Volume 2). The EDA validates data distributions (Volume 3). PCA validates dimensionality before IRT runs. IRT validates convergence during estimation. PPCs validate model fit after estimation. External validation benchmarks against the field. Quality gates validate trustworthiness for downstream consumption. And every downstream phase inherits these validations — trusting the canonical ideal points because the quality gate system has already certified them.

This layered approach means that a bug, a convergence failure, or a data quality issue has to slip past *multiple independent checks* to affect the final output. No single check is foolproof. But the system, taken as a whole, provides a level of assurance that no single check could achieve on its own.

That's what "checking our work" really means: not a one-time audit, but a culture of continuous verification baked into every step of the pipeline.

## Key Takeaway

Validation isn't a step — it's a system. The quality gate automates the routing decision (which model's ideal points to publish) using a three-tier convergence check and a party separation guard. Seven layered safety checks (R1–R7) catch axis confusion at every stage of the pipeline, from PCA initialization through dynamic IRT. The routing manifest records every decision for human review. The result: downstream phases can trust the canonical ideal points because the pipeline has already certified them through convergence diagnostics, substantive validity checks, and multiple independent detection gates. A bug or convergence failure would have to slip past all seven checks to affect the final output.

*Terms introduced: quality gate, three-tier system, Tier 1 (fully converged), Tier 2 (point estimates credible), Tier 3 (failed/fallback), R-hat, effective sample size (ESS), Cohen's d (party separation), horseshoe detection, Democrat wrong-side fraction, party overlap fraction, R1–R7 safety gates, PCA axis instability, dimension swap detection, routing manifest, canonical ideal points, graceful degradation*

---

*Previous: [Chapter 4 — W-NOMINATE: Comparing to the Field Standard](ch04-w-nominate.md)*

*Next: [Volume 6 — Finding Patterns](../volume-06-finding-patterns/)*
