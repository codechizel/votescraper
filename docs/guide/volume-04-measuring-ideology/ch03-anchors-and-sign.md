# Chapter 3: Anchors and Sign: Telling Left from Right

> *If you spin a compass, north is wherever you say it is. IRT has the same problem — without an anchor, "liberal" and "conservative" are interchangeable. This chapter explains the identification problem and how Tallgrass solves it.*

---

## The Problem No One Warned You About

In Chapter 2, we built a model that estimates each legislator's ideal point from their voting record. The model works beautifully — it converges, the diagnostics look clean, and the ideal points form a nice spread from one end to the other.

But there's a catch. Look at the equation again:

```
P(Yea) = logistic(β · ξ − α)
```

Now watch what happens if we negate both ξ and β — replace every ideal point with its negative, and every discrimination with its negative:

```
P(Yea) = logistic((−β) · (−ξ) − α)
       = logistic(β · ξ − α)
```

It's the same equation. **The predictions don't change at all.** A model where conservative Republicans are at +2 and liberal Democrats are at −2 makes exactly the same predictions as a model where conservative Republicans are at −2 and liberal Democrats are at +2 — as long as the discrimination signs flip too.

This is called **reflection invariance**, and it means the model literally cannot tell left from right. Both solutions are equally valid mathematically. Without some external help, asking the model "is this legislator liberal or conservative?" is like asking a compass that's free to spin "which way is north?"

## Why This Matters

Reflection invariance isn't just a theoretical curiosity — it creates a practical problem for the MCMC sampler.

Remember the mountain-hiking analogy from Chapter 2? Imagine the hiker is dropped into a landscape with **two identical mountain ranges** — mirror images of each other. Both are equally tall, equally valid. The hiker will wander up one range, but occasionally stumble across the valley to the other range and wander up that one. The hiker's path alternates between two different regions, never settling down.

In MCMC terms, this means:

- **R-hat spikes** because the chains disagree (one chain is exploring the "left" solution, another the "right" solution)
- **Effective sample size crashes** because the samples are a mixture of two mirror-image solutions
- **The posterior mean is useless** — averaging +2 and −2 gives 0, which describes nobody

This is why every IRT model needs an **identification strategy** — a way to break the symmetry and pin down which direction is "conservative."

## The Compass Analogy

Think of the ideal point axis as a compass needle. The needle spins freely — it can point in any direction. To make the compass useful, you need two things:

1. **A magnetic field** — something that pulls the needle toward a specific orientation. In IRT, this is the identification constraint: some piece of external information that tells the model which direction is "conservative."

2. **A label on the needle** — an agreement about which end is "N" and which is "S." In IRT, this is the sign convention: we arbitrarily (but consistently) label positive values as conservative and negative values as liberal.

The identification constraint is not a preference or a bias — it's a **calibration step**, like setting a scale to zero before weighing something. Without it, the scale gives you a number, but you don't know where zero is.

## How Anchors Work

The most common identification strategy is **anchoring**: we pick two legislators whose ideology we're confident about and fix their ideal points to known values. This breaks the reflection symmetry, because only one of the two mirror-image solutions is consistent with the anchors.

### Step by Step

1. **Choose a conservative anchor:** Pick a Republican who is clearly on the right end of the spectrum — say, the legislator with the highest PC1 score from the PCA analysis in Volume 3. Fix their ideal point at ξ = +1.

2. **Choose a liberal anchor:** Pick a Democrat who is clearly on the left end. Fix their ideal point at ξ = −1.

3. **Run the model:** All other legislators' ideal points are estimated freely, but they're now on a scale where +1 is "about as conservative as our conservative anchor" and −1 is "about as liberal as our liberal anchor."

Think of it like surveying a landscape. Before you can draw a map, you need to fix at least two reference points with known coordinates. Once you know "Point A is here and Point B is there," you can locate everything else relative to them. Anchoring works the same way — two fixed ideal points establish the scale.

### Why Two Anchors?

One anchor would break the reflection symmetry (it pins the direction), but it wouldn't fully set the scale. With one anchor at +1, a liberal legislator could plausibly be at −1 or at −3 — you've fixed one end of the ruler but not the other.

Two anchors fix both the **direction** (which end is conservative) and the **scale** (how far apart the endpoints are). The result is a well-calibrated axis where the numbers have consistent meaning.

### What PCA Tells Us

Where do we get the "known" ideology of the anchor legislators? We don't have gold-standard ideology scores for Kansas legislators. But we do have PCA from Volume 3, which gave every legislator a PC1 score that — in most sessions — cleanly separates parties.

Tallgrass's default strategy (**anchor-pca**) works like this:

1. Compute PCA on the vote matrix
2. Among Republicans, find the one with the **highest** PC1 score (most extremely conservative by PCA)
3. Among Democrats, find the one with the **lowest** PC1 score (most extremely liberal by PCA)
4. Fix the Republican anchor at ξ = +1 and the Democrat anchor at ξ = −1

This works well for **balanced chambers** — sessions where the majority party holds 50-65% of seats. In these sessions, PCA's PC1 reliably captures the party divide, so the anchors are genuinely extreme liberals and conservatives.

**Codebase:** `analysis/05_irt/irt.py` (`select_anchors()` function, using `IdentificationStrategy.ANCHOR_PCA`)

## When Anchors Go Wrong

But what about those 7 out of 14 Kansas Senate sessions where PCA's PC1 doesn't capture ideology at all?

Recall from Volume 3, Chapter 5: in the **79th Kansas Senate** (2001-2002), the Republican supermajority (75%) caused PCA to assign PC1 to intra-Republican factionalism rather than the party divide. The legislator with the highest PC1 score isn't the most conservative Republican — it's the most *establishment* Republican. The legislator with the lowest PC1 score isn't the most liberal Democrat — it might be a contrarian Republican rebel.

If you anchor on these legislators, you get a model where the ideology axis runs from "party rebel" to "party loyalist" instead of from "liberal" to "conservative." The ideal points are internally consistent but **substantively wrong** — they capture a real dimension of disagreement, just not the one we want.

This is why Tallgrass has seven identification strategies instead of just one. When anchor-pca fails, the pipeline automatically detects the problem and switches to a strategy that's robust to the horseshoe effect.

## Post-Hoc Sign Validation

Even with good anchors, Tallgrass runs a safety check after the model finishes. The check uses an independent signal — **cross-party agreement on contested votes** — to verify that the sign of the ideal points is correct.

Here's the logic:

1. For each Republican, compute their agreement rate with Democrats on contested votes (votes where neither party was unanimous)
2. Correlate these agreement rates with the estimated ideal points
3. If the correlation is **negative** (Republicans with higher ideal points agree *less* with Democrats), the sign is correct — more conservative = more disagreement with the other party
4. If the correlation is **positive** (higher ideal points = more agreement with Democrats), the sign is flipped — the model has liberal and conservative backwards

When a sign flip is detected, the fix is simple: negate all ideal points and all discrimination parameters. The predictions stay identical (that's the whole point of reflection invariance), but now the labels are correct.

```
Before correction:           After correction:
  Conservative R → -2.0       Conservative R → +2.0
  Moderate R     → -0.5       Moderate R     → +0.5
  Moderate D     → +0.5       Moderate D     → -0.5
  Liberal D      → +2.0       Liberal D      → -2.0
```

**The analogy:** It's like looking at a map and realizing you've been holding it upside down. The roads are all in the right places relative to each other — you just need to flip the map over. The sign validation detects the upside-down map and flips it.

**Codebase:** `analysis/05_irt/irt.py` (`validate_sign()` — computes Spearman correlation between Republican ideal points and Democrat agreement rate; triggers sign flip if correlation is positive with p < 0.10)

## A Kansas Example: The 79th Senate

The 79th Kansas Senate (2001-2002) is the hardest case in the dataset — 30 Republicans, 10 Democrats, a 75% Republican supermajority, and a vicious factional fight within the Republican caucus.

### What Anchor-PCA Does

PCA's PC1 captures intra-Republican factionalism. The highest-PC1 legislator is a moderate Republican establishment figure. The lowest-PC1 legislator is also a Republican — a conservative rebel. The Democrat who should be the liberal anchor is somewhere in the middle of PC1, mixed in with moderate Republicans.

Anchoring on these legislators produces an axis that separates establishment Republicans from rebel Republicans. Democrats end up scattered across the middle — ideologically indistinguishable from moderate Rs. The party separation (Cohen's d) on the resulting ideal points is only **0.86** — barely above zero. For comparison, well-behaved sessions like the 91st Senate show d > 7.0.

### What Anchor-Agreement Does

Instead of relying on PCA, anchor-agreement looks at contested votes — the ~50 votes where neither party was unanimous. For each pair of legislators from different parties, it computes how often they agreed on these contested votes.

The **most partisan Republican** is the one who agrees *least* with Democrats on contested votes. This legislator — the one who votes against Democrats even when some of their Republican colleagues don't — is a genuine ideological conservative, not just an establishment figure.

The **most partisan Democrat** is the one who agrees least with Republicans on contested votes. This legislator is a genuine ideological liberal.

Anchoring on these two produces an axis where the party separation is **d = 6.17** on 2D Dim 1 — a clean ideology spectrum. The conservative Republican rebels are correctly placed at the conservative extreme, and Democrats are correctly placed at the liberal end.

The auto-detection system detects the 75% Republican supermajority and selects anchor-agreement automatically, without human intervention.

### The Quality Gates

Even after switching strategies, Tallgrass double-checks the result:

1. **R-hat check:** Are the chains converging? (In the 79th Senate, 1D convergence remains difficult — R-hat ≈ 1.53 — pushing us toward 2D, which is Chapter 4's story.)
2. **Party separation check:** Is Cohen's d > 1.5 between party means? (Passes with anchor-agreement; fails with anchor-pca.)
3. **Sign validation:** Does the post-hoc agreement-correlation check confirm the sign? (Passes.)

The quality gate system means that even if the auto-detection fails — say, a new edge case not covered by the current logic — the downstream checks will catch the error before it propagates.

## The Bigger Picture: Why Identification Is Hard

The identification problem is not unique to IRT. It appears whenever you try to measure something that isn't directly observable:

- **Factor analysis** (psychology): The "Big Five" personality traits can be rotated into different but equally valid trait systems.
- **Topic modeling** (NLP): Topics can be relabeled or merged without changing the model fit.
- **Structural equation modeling** (sociology): Latent variables are only identified up to their relationship with observed indicators.

What makes IRT's identification problem particularly tricky for legislative data is the **interaction with chamber composition**. In a balanced chamber (50/50 party split), nearly any identification strategy works — the party divide is the dominant signal, and it's hard to go wrong. In a supermajority chamber (75/25), the dominant signal changes, and strategies that work in balanced chambers can point the model in completely the wrong direction.

This is why Tallgrass doesn't have one identification strategy but seven — and why the pipeline auto-detects which one to use based on the chamber's composition. Chapter 6 will explore all seven in detail.

---

## Key Takeaway

IRT models have a fundamental symmetry: negating all ideal points and discriminations gives identical predictions. This means the model can't distinguish liberal from conservative without external help. Anchoring — fixing the ideal points of two known legislators — breaks this symmetry and establishes a meaningful scale. But in supermajority chambers, PCA-based anchors can point the model at the wrong dimension. Tallgrass solves this with automatic strategy selection (switching to agreement-based anchors for supermajority chambers) and post-hoc sign validation (correlating ideal points with cross-party agreement to verify the correct orientation).

---

*Terms introduced: reflection invariance, identification problem, anchor, anchor-pca, anchor-agreement, sign validation, sign flip, partisan void, cross-party agreement, supermajority detection*

*Next: [When One Dimension Isn't Enough: 2D IRT](ch04-2d-irt.md)*
