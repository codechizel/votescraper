# Chapter 7: Canonical Ideal Points: Choosing the Best Score

> *The pipeline fits four different models for each chamber-session. Each produces its own ideal point estimate. Which one do you use? This chapter explains how Tallgrass picks the winner — and why the answer isn't always the most complex model.*

---

## The Routing Problem

By the time the pipeline reaches this point, it has produced up to four ideal point estimates for every legislator:

| Model | Phase | What It Captures |
|-------|-------|-----------------|
| 1D Flat IRT | Phase 05 | Single ideology axis, no party structure |
| 2D Flat IRT, Dim 1 | Phase 06 | Ideology axis separated from establishment axis |
| Hierarchical 1D IRT | Phase 07 | Ideology with party-based shrinkage |
| Hierarchical 2D IRT, Dim 1 | Phase 07b | Ideology + establishment, with party shrinkage |

For a well-behaved session like the 91st House, all four estimates agree closely — correlations above 0.95. It barely matters which one you use.

But for a horseshoe-affected session like the 79th Senate, the estimates diverge sharply. The 1D flat model captures the wrong axis (party separation d = 0.28). The 2D flat model's Dimension 1 captures ideology correctly (d = 6.17). The hierarchical models add shrinkage on top of either the correct or incorrect axis.

Choosing the wrong model means publishing the wrong ideal points. Choosing the right model means every downstream analysis — clustering, network analysis, temporal trends, legislator profiles — starts from a solid foundation.

This is the **canonical routing problem**: given multiple model outputs with different quality characteristics, how do you automatically select the best one?

## The Solution: Prefer Complexity When It Converges

The routing system follows a simple principle: **prefer the model that captures the most structure, as long as it converged well enough to trust.**

The preference order is:

```
1st choice:  Hierarchical 2D, Dim 1  (most structure: 2D + party pooling)
2nd choice:  Flat 2D, Dim 1          (2D structure without pooling)
3rd choice:  1D IRT                  (simplest model)
```

At each step, the system checks whether the preferred model passed its **quality gate**. If it did, use it. If it didn't, fall back to the next option.

This is the same logic a doctor might use when choosing a diagnostic test: prefer the MRI (most detail), but fall back to a CT scan (less detail, more reliable) if the MRI is unavailable or produced artifacts. And if the CT scan also failed, fall back to an X-ray. The goal is the best information available, not the fanciest technology.

### Why Not Always Use the Most Complex Model?

Two-dimensional models have more parameters and weaker signals on Dimension 2. In balanced chambers where the ideology axis is clean and one-dimensional, the 2D model may produce Dimension 1 estimates that are noisier than the 1D estimates — the extra parameters add uncertainty without capturing meaningful structure.

For these sessions, 1D is both simpler and better. Complexity should serve the data, not the other way around.

## The Three-Tier Quality Gate

The quality gate system determines whether a model's output is trustworthy. It has three tiers, each with progressively looser requirements:

### Tier 1: Fully Converged

| Diagnostic | Threshold |
|-----------|-----------|
| R-hat | < 1.10 |
| Effective Sample Size (ESS) | > 100 |

**What it means:** The MCMC chains agree closely, and we have enough independent samples to trust both the point estimates and the credible intervals. The ideal points can be used with full confidence, including uncertainty quantification.

**The analogy:** Four independent witnesses give nearly identical accounts of an event. You trust not just their story but their level of detail.

### Tier 2: Point Estimates Credible

| Diagnostic | Threshold |
|-----------|-----------|
| R-hat | < 2.50 |
| Party separation (Cohen's d) | > 1.5 |

**What it means:** The chains haven't fully converged (R-hat is above the strict threshold), but the *ranking* of legislators is consistent across chains and the axis clearly separates parties. The point estimates (means) are trustworthy, but the credible intervals may be too wide to use.

**Why party separation matters:** This is the check that breaks the circular PCA dependency. Earlier versions of the quality gate compared 2D ideal points against PCA PC1 — but in horseshoe-affected sessions, PCA PC1 is wrong. Comparing the 2D result (which correctly captures ideology) against a wrong PCA axis would *reject the correct result*.

The party separation check avoids this trap entirely. It asks: "Do Republicans and Democrats end up on opposite sides of this axis, with a gap of at least 1.5 standard deviations?" This is a standalone quality measure that doesn't depend on PCA.

**The analogy:** Four witnesses give accounts that differ in some details (exact times, minor descriptions) but agree on the main facts (who did what). You trust the big picture but not every detail.

### Tier 3: Failed

| Diagnostic | Threshold |
|-----------|-----------|
| R-hat | ≥ 2.50 |

**What it means:** The chains fundamentally disagree. The model may have gotten stuck in different modes, or the data simply doesn't support the model's complexity. Fall back to a simpler model.

**The analogy:** Four witnesses give contradictory accounts. You can't trust any of them and need a different source of information.

### Constants

```
TIER1_RHAT_THRESHOLD  = 1.10
TIER1_ESS_THRESHOLD   = 100
TIER2_RHAT_THRESHOLD  = 2.50
TIER2_PARTY_D_THRESHOLD = 1.5
```

**Codebase:** `analysis/canonical_ideal_points.py`

## Horseshoe Detection

Before consulting the quality gate, the routing system first checks whether the 1D model even captured the right axis. The horseshoe detection uses three independent signals:

1. **Democrat wrong-side fraction:** What fraction of Democrats have ideal points on the Republican side of the scale (positive ξ)? If more than 20% of Democrats appear "conservative," the axis is likely wrong.

2. **Party overlap fraction:** What fraction of all legislators fall in the overlap zone between party distributions? If more than 30% of legislators can't be reliably assigned to a party based on their ideal point, the separation is too weak.

3. **Individual crossover:** Is any Republican more liberal than the Democratic mean? This catches cases where the scale is reversed for a subset of legislators.

If any of these signals triggers, the session is flagged as horseshoe-affected, and the routing system preferentially selects a 2D model (where Dimension 1 is ideology, freed from the establishment-vs-rebel contamination).

```
HORSESHOE_DEM_WRONG_SIDE_FRAC = 0.20
HORSESHOE_OVERLAP_FRAC = 0.30
```

## The Routing Decision Tree

Putting it all together, the canonical routing follows this logic for each chamber:

```
Step 1: Load 1D IRT ideal points from Phase 05
Step 2: Check for horseshoe
        │
        ├─ No horseshoe detected:
        │   └─ Use 1D IRT (simplest sufficient model)
        │
        └─ Horseshoe detected:
            │
            ├─ Step 3a: Check Hierarchical 2D (Phase 07b)
            │   ├─ Available + Tier 1 or Tier 2: Use H2D Dim 1  ← best option
            │   └─ Not available or Tier 3: Continue to 3b
            │
            ├─ Step 3b: Check Flat 2D (Phase 06)
            │   ├─ Available + Tier 1 or Tier 2: Use 2D Dim 1  ← second best
            │   └─ Not available or Tier 3: Continue to 3c
            │
            └─ Step 3c: Fall back to 1D IRT (with horseshoe warning)
```

Notice the asymmetry: for non-horseshoe sessions, 1D is used directly (no point checking 2D when 1D works fine). For horseshoe sessions, the system cascades through 2D options before reluctantly falling back to 1D.

## The Routing Manifest

Every routing decision is recorded in a **routing manifest** — a JSON file that documents exactly what was chosen and why:

```json
{
  "sources": {
    "House": "1d_irt",
    "Senate": "2d_dim1"
  },
  "metadata": {
    "House": {
      "horseshoe": {"detected": false},
      "reason": "no_horseshoe"
    },
    "Senate": {
      "horseshoe": {
        "detected": true,
        "dem_wrong_side_frac": 0.40,
        "overlap_frac": 0.35
      },
      "convergence_tier": {
        "tier": 2,
        "rhat_max": 1.69,
        "party_d": 6.17
      },
      "reason": "horseshoe_detected_2d_tier2"
    }
  },
  "thresholds": {
    "horseshoe_dem_wrong_side_frac": 0.20,
    "horseshoe_overlap_frac": 0.30,
    "tier1_rhat_threshold": 1.10,
    "tier1_ess_threshold": 100,
    "tier2_rhat_threshold": 2.50
  }
}
```

The manifest serves three purposes:

1. **Reproducibility:** Anyone can see exactly which model was used and verify the decision by re-checking the thresholds.
2. **Debugging:** If a downstream analysis looks wrong, the manifest is the first place to look — was the routing decision correct?
3. **Auditing:** A reviewer can check whether the thresholds are reasonable and whether the decision logic makes sense for each session.

**Codebase:** `analysis/canonical_ideal_points.py` (`route_canonical_ideal_points()`, `write_canonical_ideal_points()`)

## The Canonical Output

Regardless of which model was selected, the canonical output has a **uniform schema** — every downstream phase sees the same column structure:

| Column | Type | Description |
|--------|------|-------------|
| legislator_slug | string | Unique legislator identifier |
| full_name | string | Display name |
| party | string | R or D |
| xi_mean | float | Posterior mean ideal point |
| xi_sd | float | Posterior standard deviation |
| xi_hdi_2.5 | float | Lower bound of 95% HDI |
| xi_hdi_97.5 | float | Upper bound of 95% HDI |
| source | string | Which model produced this: "1d_irt", "2d_dim1", or "hierarchical_2d_dim1" |

The `source` column is the key transparency mechanism. Any downstream analysis that reads canonical ideal points can see — for every single legislator — which model produced their score. If a researcher wants to exclude Tier 2 results and use only Tier 1, they can filter on the source and cross-reference the routing manifest.

## Kansas in Practice: How Each Session Routes

Here's a representative view of how the routing system handles different Kansas sessions:

| Session | Chamber | Horseshoe? | Source | Tier | Party d |
|---------|---------|-----------|--------|------|---------|
| 91st (2025-26) | House | No | 1d_irt | — | 5.2 |
| 91st (2025-26) | Senate | No | 1d_irt | — | 7.5 |
| 89th (2021-22) | House | No | 1d_irt | — | 4.8 |
| 89th (2021-22) | Senate | No | 1d_irt | — | 7.2 |
| 88th (2019-20) | House | No | 1d_irt | — | 4.5 |
| **88th (2019-20)** | **Senate** | **Yes** | **2d_dim1** | **Tier 2** | **3.1** |
| 85th (2013-14) | House | No | 1d_irt | — | 5.0 |
| 85th (2013-14) | Senate | No | 1d_irt | — | 6.8 |
| **79th (2001-02)** | House | No | 1d_irt | — | 4.3 |
| **79th (2001-02)** | **Senate** | **Yes** | **2d_dim1** | **Tier 2** | **6.2** |

The pattern is clear:
- **All House sessions** route to 1D. The 125-member chamber is large enough that even a 72% majority doesn't produce horseshoe distortion.
- **Most Senate sessions** also route to 1D. After the 2012 "purge" primaries, the Republican caucus became more homogeneous, and the party divide reasserted itself on PC1.
- **Horseshoe Senate sessions** (78th-83rd, 88th) route to 2D Dim 1. These are the sessions where intra-Republican factionalism dominated PCA, and only a 2D model can separate ideology from establishment dynamics.

## The DW-NOMINATE Precedent

The idea of using 2D Dimension 1 as the canonical ideology score is not new. The political science field's standard tool — **DW-NOMINATE**, developed by Keith Poole and Howard Rosenthal in the 1980s — has been doing exactly this for over 40 years.

DW-NOMINATE fits a 2D spatial voting model to the U.S. Congress and reports Dimension 1 as the primary ideology axis. The widely cited "Poole-Rosenthal scores" that political journalists reference are DW-NOMINATE Dimension 1 values. The second dimension (historically capturing civil rights/slavery, now less clear) is reported but rarely used as the primary measure.

Tallgrass's canonical routing follows the same logic, adapted for state legislatures with supermajority complications that the balanced U.S. Congress doesn't face. The routing system automates the judgment that DW-NOMINATE users have applied manually for decades: when one dimension captures ideology cleanly, use it; when it doesn't, use a 2D model and take Dimension 1.

## What Downstream Phases See

Every phase after canonical routing — clustering, network analysis, classical indices, synthesis, legislator profiles — reads from the canonical ideal points file. They don't know or care whether the score came from a 1D model, a 2D model, or a hierarchical model. They just see:

> "This legislator has an ideal point of +1.23 with a 95% HDI of [+0.98, +1.48], source: 2d_dim1."

This abstraction means downstream phases never need to handle the horseshoe, the identification strategy, or the quality gate logic. All of that complexity is resolved once, here, and the result is a clean, uniform interface.

```bash
# Run the full pipeline — canonical routing happens automatically
just pipeline 2025-26
```

---

## Key Takeaway

Canonical routing solves the "which model?" problem by preferring the most structured model that converges well enough to trust. A three-tier quality gate (Tier 1: fully converged, Tier 2: point estimates credible, Tier 3: failed) determines trustworthiness. Horseshoe detection flags sessions where 1D ideal points capture the wrong axis, triggering a cascade to 2D models. The routing manifest documents every decision for reproducibility. Downstream phases receive a uniform output regardless of which model produced it — the complexity of model selection is contained in one place.

---

*Terms introduced: canonical routing, routing manifest, quality gate tier, horseshoe detection, party overlap, Democrat wrong-side fraction, DW-NOMINATE, Poole-Rosenthal scores, uniform schema, model fallback cascade*

*Previous: [The Identification Zoo: Seven Strategies](ch06-identification-zoo.md)*

*Back to: [Volume 4 — Measuring Ideology](README.md)*
