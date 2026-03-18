# Chapter 6: The Identification Zoo: Seven Strategies

> *There isn't one right way to tell the model which direction is "conservative." Tallgrass implements seven strategies, each with different strengths, different assumptions, and different failure modes. This chapter is a field guide to all of them.*

---

## Why Seven Strategies?

Chapter 3 introduced the identification problem — IRT models can't tell left from right without external help — and showed two strategies: anchor-pca (fix anchors based on PCA extremes) and anchor-agreement (fix anchors based on contested-vote agreement patterns). If those two work, why does Tallgrass need five more?

The answer is that no single strategy works for every session in the dataset. Kansas has 28 chamber-sessions (14 bienniums × 2 chambers), and they vary enormously:

- **Balanced chambers** where PC1 cleanly separates parties (the Kansas House in every session)
- **Supermajority chambers** where PC1 captures intra-Republican factionalism instead of ideology (7 of 14 Senate sessions)
- **Sessions with external benchmark data** (Shor-McCarty scores covering 2011–2020)
- **Sessions with very small minority caucuses** (some Senate sessions with fewer than 10 Democrats)

Each scenario calls for a different approach. Rather than hard-coding one strategy, Tallgrass implements seven and uses an automatic detection system to choose the best one for each chamber-session. Let's walk through all seven.

## Strategy 1: Anchor-PCA

**Mechanism:** Fix two legislators' ideal points at ξ = +1 (conservative anchor) and ξ = −1 (liberal anchor), selected by their PCA PC1 extremes.

**How it works:**
1. Run PCA on the vote matrix (Volume 3)
2. Among Republicans, find the one with the highest PC1 score → conservative anchor
3. Among Democrats, find the one with the lowest PC1 score → liberal anchor
4. In the IRT model, these two legislators' ideal points are fixed constants, not estimated parameters. Everyone else is estimated freely.

**When it's auto-selected:** Balanced chambers where no party holds ≥70% of seats.

**Strengths:**
- Simple and well-understood
- The field standard — this is essentially what Clinton, Jackman, and Rivers (2004) used in their foundational IRT paper on the U.S. Congress
- Works reliably when PC1 captures the party divide

**Weaknesses:**
- Fails in supermajority chambers where PC1 captures intra-party factionalism (the horseshoe effect)
- Pins two specific legislators, which means their credible intervals are zero by construction — we're assuming perfect knowledge of their ideology

**The analogy:** Anchor-pca is like calibrating a thermometer by sticking it in ice water (0°C) and boiling water (100°C). It works perfectly when you have genuine ice and genuine boiling water. But if someone secretly switched your ice water for cold tap water, your whole scale is off.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.ANCHOR_PCA`)

## Strategy 2: Anchor-Agreement

**Mechanism:** Fix two legislators' ideal points, but select anchors using **cross-party contested-vote agreement** rather than PCA scores.

**How it works:**
1. Identify **contested votes** — roll calls where neither party voted unanimously (at least 10% of each party on each side)
2. For each Republican, compute their agreement rate with Democrats on these contested votes
3. The Republican with the **lowest** Democrat-agreement rate is the most partisan Republican → conservative anchor at ξ = +1
4. The Democrat with the **lowest** Republican-agreement rate is the most partisan Democrat → liberal anchor at ξ = −1

**When it's auto-selected:** Supermajority chambers (≥70% majority) with at least 10 contested votes and 6 eligible legislators with agreement data.

**Strengths:**
- Immune to the horseshoe effect — agreement rates are computed from actual voting behavior, not from PCA's potentially distorted axis
- Selects genuinely ideologically extreme legislators (ones who vote against the other party most consistently), not just legislators extreme on whatever PCA happened to capture
- Developed for this project to handle the Kansas Senate supermajority pattern

**Weaknesses:**
- Requires enough contested votes (≥10) to compute meaningful agreement rates
- Still pins two specific legislators

**The analogy:** If anchor-pca calibrates a thermometer by sticking it in what you *hope* is ice water, anchor-agreement calibrates it by finding the coldest and hottest things in the room through an independent temperature measurement. Even if someone mislabeled the containers, the agreement-based selection finds the real extremes.

**How it handles the 79th Senate:** The 75% Republican supermajority makes PCA unreliable, so the pipeline auto-detects the supermajority and switches to anchor-agreement. It selects the most partisan Republican (the one who broke with Democrats on the most contested votes) and the most partisan Democrat. The resulting ideal points show party separation d > 6.0, confirming the axis captures ideology.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.ANCHOR_AGREEMENT`, `compute_cross_party_agreement()`)

## Strategy 3: Sort-Constraint

**Mechanism:** No individual anchors at all. Instead, apply a **soft ordering constraint** that forces the Democratic party mean to be less than the Republican party mean.

**How it works:**
1. Estimate all legislator ideal points freely (no fixed values)
2. Compute the mean ideal point for each party
3. Add a penalty to the model's likelihood: if the Republican mean exceeds the Democratic mean by at least 0.5 points, no penalty. If the gap is smaller, a large penalty (−100 in log-probability) discourages that region of the posterior.

**Mathematically:**

```
Penalty = switch(mean(ξ_R) − mean(ξ_D) > 0.5,  0.0,  −100.0)
```

**When it's auto-selected:** Supermajority chambers where anchor-agreement can't run (not enough contested votes or eligible legislators), but both parties are present.

**Strengths:**
- No legislator is pinned — everyone gets a full posterior distribution
- Naturally extends to chambers where individual anchors are hard to identify
- The ordering constraint is mild: it only requires Democrats to be more liberal *on average* than Republicans, not that every Democrat is more liberal than every Republican

**Weaknesses:**
- Identifies only the **sign** (direction), not the **scale**. The distance between party means is determined by the data and the Normal(0,1) prior, but there's no external calibration point.
- The sharp penalty (0 vs. −100) can cause sampling difficulties at the boundary

**The analogy:** Instead of placing two surveying stakes in the ground (anchoring specific points), you're putting up a single sign that says "north is this way." It gives you direction but not distance.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.SORT_CONSTRAINT`)

## Strategy 4: Positive-Beta

**Mechanism:** Force all bill discrimination parameters to be positive (β > 0), eliminating reflection invariance through a constraint on the bill side rather than the legislator side.

**How it works:**
1. Replace the standard Normal(0, 1) prior on β with **HalfNormal(1)** — a distribution that only produces positive values
2. With all β > 0, the model is forced into a single mode: if conservative legislators vote Yea more often on bills with high β, they get positive ideal points. If liberal legislators vote Nay more often, they get negative ideal points.

**When it's auto-selected:** Never — manual override only.

**Strengths:**
- Eliminates reflection invariance completely — there's only one solution, not two
- No anchors needed; every legislator is estimated freely
- A common approach in educational testing (where "getting the answer right" has a natural positive direction)

**Weaknesses:**
- **Silences "D-Yea" bills.** About 12.5% of Kansas roll call votes are bills where the liberal position is Yea (e.g., expanding social services, increasing education funding). For these bills, the "correct" discrimination should be negative (liberal legislators vote Yea, conservative legislators vote Nay). Forcing β > 0 compresses these bills' discrimination toward zero, effectively telling the model to ignore them.
- Assumes all bills have the same ideological polarity, which is empirically false

**The analogy:** Positive-beta is like requiring all thermometers to read higher numbers for "hotter." It works for most thermometers, but it would break a wind chill index, where lower numbers mean colder but the measurement is inverted. Forcing all β positive breaks the model's ability to represent bills where the liberal-conservative polarity is reversed.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.POSITIVE_BETA`)

## Strategy 5: Hierarchical-Prior

**Mechanism:** Use **party-informed priors** — give Republicans a prior centered at +0.5 and Democrats a prior centered at −0.5.

**How it works:**
```
Republican ideal points:  ξ ~ Normal(+0.5, 1)
Democratic ideal points:  ξ ~ Normal(−0.5, 1)
```

The priors are soft — they say "Republicans are probably conservative and Democrats are probably liberal, but the data can override this." With enough votes, a moderate Republican will be pulled toward zero despite their +0.5 prior starting point. With few votes, a Republican will stay near +0.5.

**When it's auto-selected:** Never — manual override only.

**Strengths:**
- Soft identification: the data can overwhelm the prior for any individual legislator
- Handles cross-party outliers gracefully (a conservative Democrat will end up with a positive ideal point if the data supports it)
- Incorporates genuine prior knowledge (party membership really does predict ideology)

**Weaknesses:**
- The prior dominates for legislators with very few votes — they'll be pulled toward their party's center regardless of their actual voting record
- Requires both parties to be present
- The choice of ±0.5 is somewhat arbitrary

**The analogy:** Hierarchical-prior is like a teacher who starts each student's grade at the class average, then adjusts based on their actual performance. Students who took many tests get grades based mostly on their own work. Students who took only one test get grades pulled toward the class average — the teacher assumes they're probably similar to their classmates until proven otherwise.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.HIERARCHICAL_PRIOR`)

## Strategy 6: Unconstrained

**Mechanism:** No identification at all during MCMC. Run the model with symmetric priors and correct the sign afterward using `validate_sign()`.

**How it works:**
1. Fit the model with no constraints: ξ ~ Normal(0, 1), β ~ Normal(0, 1)
2. The posterior is bimodal (two mirror-image solutions)
3. After sampling, use the post-hoc sign validation from Chapter 3 to determine which mode is "correct"
4. If the sign is flipped, negate all ξ and β values

**When it's auto-selected:** Never — diagnostic use only.

**Strengths:**
- The purest approach: no assumptions baked into the model
- Useful for checking whether the data itself suggests a clear liberal-conservative axis (if both modes look messy, the data may not support a clean 1D interpretation)

**Weaknesses:**
- The bimodal posterior causes severe sampling problems: chains often get stuck in different modes, R-hat spikes, and effective sample sizes crash
- The post-hoc correction can fail if chains are poorly mixed
- Not suitable for production use

**The analogy:** Running an unconstrained model is like asking four hikers to each independently find the tallest mountain in a range that's perfectly symmetric — two identical peaks on opposite sides. Two hikers will end up on the left peak and two on the right. Averaging their reports gives you the valley in between, which is the *lowest* point — exactly wrong.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.UNCONSTRAINED`)

## Strategy 7: External-Prior

**Mechanism:** Use **ideology scores from an external dataset** (such as Shor-McCarty state legislator scores) as informative priors.

**How it works:**
```
ξ_i ~ Normal(external_score_i, 0.5)
```

Each legislator's ideal point prior is centered on their externally-estimated ideology score, with a standard deviation of 0.5 that allows the data to pull them away if the evidence is strong enough.

**When it's auto-selected:** When external scores are available for the chamber-session being analyzed.

**Strengths:**
- Solves sign, location, *and* scale simultaneously — the model inherits the external dataset's calibration
- Enables cross-state and cross-time comparisons (if the external scores are on a common scale)
- Maximum information incorporation: we're not ignoring relevant prior knowledge

**Weaknesses:**
- Requires external scores, which may not be available for recent sessions
- A tight prior (σ = 0.5) may suppress genuine variation if the external scores are out of date or measured at a different level (Shor-McCarty scores are career-averaged, not session-specific)
- Creates a dependency on an external dataset's methodology

**The analogy:** External-prior is like calibrating your bathroom scale by first weighing yourself at the doctor's office (the "gold standard"). Your home scale might be slightly off, but starting from the doctor's measurement gives you a much better baseline than starting from scratch.

**Codebase:** `analysis/05_irt/irt.py` (`IdentificationStrategy.EXTERNAL_PRIOR`)

## The Auto-Detection System

Tallgrass doesn't ask the user to choose a strategy. The pipeline examines each chamber-session and selects automatically:

```
Step 1: Are external scores available?
        → Yes: Use EXTERNAL_PRIOR
        → No:  Continue to Step 2

Step 2: Is this a supermajority chamber? (majority party ≥ 70%)
        → No:  Use ANCHOR_PCA (the simple default)
        → Yes: Continue to Step 3

Step 3: Are there enough contested votes for agreement-based anchoring?
        (≥ 10 contested votes, ≥ 6 eligible legislators with agreement data)
        → Yes: Use ANCHOR_AGREEMENT
        → No:  Continue to Step 4

Step 4: Are both parties present?
        → Yes: Use SORT_CONSTRAINT
        → No:  Use ANCHOR_PCA as last resort (warn about potential issues)
```

The supermajority threshold of **70%** was determined empirically from the Kansas data. Below 70%, PCA's PC1 consistently captures the party divide. Above 70%, the horseshoe effect becomes increasingly likely.

**Codebase:** `analysis/05_irt/irt.py` (`select_identification_strategy()`, `SUPERMAJORITY_THRESHOLD = 0.70`)

### The Strategy Rationale Table

Every IRT run produces a **strategy rationale table** in its report, documenting why each strategy was or wasn't selected. Here's an example for the 79th Senate (75% Republican supermajority):

| Strategy | Selected? | Reason |
|----------|-----------|--------|
| external-prior | No | No external scores available for 79th Senate |
| anchor-pca | No | Supermajority detected (R = 75%, above 70% threshold) |
| **anchor-agreement** | **Yes** | Supermajority detected; 47 contested votes; R and D anchors identified |
| sort-constraint | No | anchor-agreement succeeded (not needed as fallback) |
| positive-beta | No | Not auto-selected (manual override only) |
| hierarchical-prior | No | Not auto-selected (manual override only) |
| unconstrained | No | Not auto-selected (diagnostic only) |

This transparency ensures that any user — or any future researcher auditing the results — can see exactly what decision was made and why.

## Comparing Strategies on the 79th Senate

The 79th Kansas Senate (2001-2002) is the ideal test case: a 75% Republican supermajority with strong intra-party factionalism. Here's what different strategies produce:

| Strategy | Party Separation (d) | Converged? | Correct Axis? |
|----------|---------------------|------------|---------------|
| anchor-pca | 0.28 | Poorly (R-hat > 1.5) | No — captures factionalism |
| anchor-agreement | 0.86 (1D) / 6.17 (2D Dim 1) | 1D: poorly; 2D: Tier 2 | 1D: marginal; 2D: yes |
| sort-constraint | ~1.2 | Moderately | Marginal — sign correct but weak separation |
| positive-beta | ~0.9 | Moderately | Marginal — D-Yea bills suppressed |
| hierarchical-prior | ~1.5 | Yes | Yes — party priors guide the model |
| unconstrained | N/A | No (bimodal) | Can't determine |
| external-prior | N/A | Not tested | External scores not available for this session |

The clear lesson: **anchor-agreement combined with 2D IRT** produces the best result for this supermajority session. The 1D model struggles regardless of identification strategy — the factional structure genuinely requires two dimensions. This is why the canonical routing system (Chapter 7) prefers 2D Dim 1 over any 1D result for horseshoe-affected chambers.

## A Summary of the Zoo

| # | Strategy | Anchors? | When Used | Key Assumption |
|---|----------|----------|-----------|----------------|
| 1 | anchor-pca | 2 hard anchors from PCA | Balanced chambers | PCA PC1 = ideology |
| 2 | anchor-agreement | 2 hard anchors from agreement | Supermajority chambers | Contested-vote agreement = partisanship |
| 3 | sort-constraint | None (ordering penalty) | Supermajority, few contested votes | D mean < R mean |
| 4 | positive-beta | None (β > 0) | Manual only | All bills have the same polarity |
| 5 | hierarchical-prior | None (party priors) | Manual only | Party predicts ideology |
| 6 | unconstrained | None | Diagnostics only | Post-hoc correction works |
| 7 | external-prior | None (informative prior) | External scores available | External scores are accurate |

No single strategy is "best." Each encodes different assumptions, and those assumptions have different failure modes. The auto-detection system matches strategies to chamber characteristics, and the post-hoc sign validation (Chapter 3) provides a final safety net regardless of which strategy was used.

---

## Key Takeaway

Tallgrass implements seven identification strategies because the Kansas Legislature spans balanced and supermajority chambers, sessions with and without external benchmarks, and chambers of different sizes. The auto-detection system examines each chamber's party composition and data availability to choose the most appropriate strategy. Transparency is maintained through a rationale table that documents every decision. No strategy is universal — the right choice depends on the data.

---

*Terms introduced: auto-detection, supermajority threshold, contested-vote agreement, strategy rationale table, D-Yea bill, informative prior, bimodal posterior*

*Next: [Canonical Ideal Points: Choosing the Best Score](ch07-canonical-ideal-points.md)*
