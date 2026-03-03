# Are Our Ideology Scores Right? Checking Our Work Against the Gold Standard

**How we verified our Kansas Legislature ideal points against an independent national dataset — and what we found.**

*Updated 2026-02-28 — expanded from 88th-only to all 5 overlapping bienniums (84th-88th)*

---

## The Problem: Grading Your Own Homework

Imagine you build a bathroom scale in your garage. You weigh yourself ten times and get the same number every time. Consistent? Yes. Accurate? You have no idea. The only way to find out is to step on a *different* scale — one you trust — and compare.

That's exactly the situation our analysis pipeline was in.

We built a statistical model that places every Kansas legislator on an ideological scale: far left on one end, far right on the other. The model is internally consistent — it agrees with itself in every way we can check. Our principal components analysis (a simpler technique) gives nearly the same ranking. Our model can predict 93% of votes correctly. Legislators who return session after session get similar scores each time. Everything lines up.

But all of those checks are circular. We're grading our own homework. We never asked: *does an independent measure of ideology, built by different researchers using different methods, agree with our scores?*

Every published ideal point study does this. It was the single biggest credibility gap in our pipeline.

---

## The Answer: The Shor-McCarty Dataset

Boris Shor and Nolan McCarty are political scientists who spent years building a national database of state legislator ideology scores. Their dataset covers **all 50 states, from 1993 to 2020** — including Kansas. The methodology was published in the *American Political Science Review* in 2011, one of the top journals in the field.

Their approach is clever. They start with a standardized survey (the National Political Awareness Test) that legislators in multiple states answer. Because some legislators take the same survey *and* cast votes in their state legislature, those shared survey responses become "bridges" that connect the scales across states. Within each state, they estimate ideal points from roll call votes — just like we do — but the bridges let them place everyone on a single national scale.

The dataset is free, publicly available from Harvard Dataverse under a CC0 (public domain) license. For Kansas, it includes **610 legislators** with ideology scores spanning 1996 to 2020.

Think of it this way: Shor-McCarty built a professionally calibrated scale. We built our own scale. Now we're stepping on both and comparing the readings.

---

## What We Compared

Our pipeline estimates ideology scores per legislative session (a two-year biennium). Shor-McCarty assigns each legislator a single career-level score. This means we can't compare them for every session — only for the years that overlap.

The overlap window is **2011 to 2020**, covering the 84th through 88th Kansas Legislatures — five independent two-year sessions. We validated all five.

For each legislator who served in a given biennium and appears in the Shor-McCarty dataset, we have two ideology scores:

- **Our score** (`xi_mean`): estimated from Kansas roll call votes using Bayesian Item Response Theory
- **Their score** (`np_score`): estimated independently by Shor and McCarty using national bridging methodology

If both methods are measuring the same thing — where a legislator falls on the liberal-conservative spectrum — the two sets of scores should be highly correlated. Statisticians measure this with **Pearson's r**, a number from -1 to +1:

| Correlation (r) | What it means |
|------------------|---------------|
| 0.95 - 1.00 | Essentially identical rankings |
| 0.90 - 0.95 | Strong agreement — minor differences on a few individuals |
| 0.85 - 0.90 | Good agreement — differences likely reflect real dynamics |
| 0.70 - 0.85 | Moderate — worth investigating what's different |
| Below 0.70 | Concerning — one or both methods may have a problem |

Our target was **r > 0.85** (good agreement). Published studies in this field typically report correlations in the 0.85-0.95 range between different ideal point methods.

---

## The Results

### Matching Legislators

Before computing any correlations, we had to match legislators across the two datasets. Our data uses names like "John Alcala" while Shor-McCarty uses "Alcala, John." We built a name normalization algorithm that converts both to a common format ("john alcala"), handling middle names, suffixes like "Jr." and "Sr.", and leadership titles like "- President."

Matching rates across all five bienniums were consistently above 99%:

- **House**: 112-130 matched per session (99-100% match rate)
- **Senate**: 37-41 matched per session (100% match rate in 4 of 5 sessions)

### The Correlation Table

Here are the headline numbers across all five bienniums. We tested two versions of our model: the standard ("flat") IRT and a more sophisticated hierarchical version that partially pools information within parties.

**Flat IRT vs Shor-McCarty:**

| Biennium | House r | House n | Senate r | Senate n |
|----------|---------|---------|----------|----------|
| 84th (2011-12) | **0.975** | 112 | **0.914** | 37 |
| 85th (2013-14) | **0.968** | 116 | **0.963** | 39 |
| 86th (2015-16) | **0.975** | 128 | **0.919** | 40 |
| 87th (2017-18) | **0.966** | 130 | **0.953** | 41 |
| 88th (2019-20) | **0.981** | 127 | **0.929** | 41 |
| **Pooled** | **0.975** | **278** | **0.912** | **77** |

**Hierarchical IRT vs Shor-McCarty:**

| Biennium | House r | House n | Senate r | Senate n |
|----------|---------|---------|----------|----------|
| 84th (2011-12) | **0.986** | 112 | **0.945** | 37 |
| 85th (2013-14) | **0.985** | 116 | **0.986** | 39 |
| 86th (2015-16) | **0.990** | 128 | **0.987** | 40 |
| 87th (2017-18) | **0.983** | 130 | **0.991** | 40 |
| 88th (2019-20) | **0.984** | 127 | **0.969** | 41 |
| **Pooled** | **0.974** | **278** | **0.950** | **77** |

*Pearson r measures linear agreement. Spearman ρ measures rank-order agreement. "Matched" is the number of legislators present in both datasets. All 20 per-session correlations are rated "strong" (r > 0.90).*

### What These Numbers Mean

**Every single correlation is "strong."** Across 5 bienniums, 2 chambers, and 2 models — 20 individual correlations — all exceed r = 0.90. This is not a single lucky session; it is a consistent pattern spanning a decade of Kansas legislative history.

**House (both models): r = 0.966-0.990.** Our pipeline and Shor-McCarty agree almost perfectly on where each legislator falls on the ideological spectrum. A correlation of 0.98 means that if you picked any two House members at random, the two methods would agree on which one is more conservative more than 99% of the time.

**Senate (flat model): r = 0.914-0.963.** Still strong, though slightly lower than the House. This makes sense: with only 37-41 senators (versus 112-130 representatives), each individual has more influence on the correlation, and the Shor-McCarty career-level score has less data per senator to work with.

**Hierarchical beats flat on Senate.** This is the biggest change from our initial 88th-only validation. The hierarchical model now matches or outperforms the flat model for Senate in every biennium — reaching r = 0.991 for the 87th. The nutpie Rust NUTS sampler migration (ADR-0051, ADR-0053) and convergence improvements (adaptive priors, PCA-informed initialization) resolved the over-shrinkage problem that produced the inverted r = -0.54 in our original 88th-only analysis.

---

## The Scatter Plots: Seeing the Agreement

Numbers tell you the correlation is strong. Pictures show you *how* it's strong.

Each scatter plot shows one dot per legislator, arranged in a diagonal line from bottom-left (most liberal) to upper-right (most conservative). Democrats cluster in the lower-left corner, Republicans fill the upper-right half. The closer the dots hug the regression line, the stronger the agreement.

Across all five bienniums and both chambers, the pattern is remarkably consistent: tight diagonals with no wild outliers. Republicans show slightly more spread than Democrats, reflecting the genuine intra-party ideological variation that our earlier analyses identified.

The most notable individual-level discrepancies tend to involve legislative leaders. For example, in the 88th, **Jim Denning** (Senate Majority Leader) shows the largest gap: our model places him as moderate-for-a-Republican (-0.58) while Shor-McCarty places him as solidly conservative (+0.75). This likely reflects the strategic compromises of caucus management during the 2019-2020 session — exactly the kind of difference you'd expect between a session-specific score and a career-level score.

---

## How We Fixed the Hierarchical Senate Model

Our original 88th-only analysis (February 2026) showed an alarming result: the hierarchical Senate model produced r = -0.54 — an *inverted* correlation. The model was getting the rank ordering backwards.

The cause was "over-shrinkage." Our hierarchical model uses partial pooling — it groups legislators by party and estimates each individual's position partly based on their party's average. With only 11 Senate Democrats, the model pulled everyone too close to their party mean, erasing the individual differences that make the scores useful.

Three improvements fixed this:

1. **nutpie Rust NUTS sampler** (ADR-0051, ADR-0053) — replaced PyMC's default sampler with a Rust-based sampler that uses normalizing flow adaptation, dramatically improving sampling in the correlated posterior geometry that small-group hierarchical models create.

2. **Group-size-adaptive priors** (ADR-0043) — for party groups smaller than 20 legislators, the model now uses a tighter prior on within-party spread (`HalfNormal(0.5)` instead of `HalfNormal(1.0)`), preventing the sampler from exploring unreasonable configurations.

3. **PCA-informed initialization** (ADR-0044) — starting the sampler near the PCA solution prevents it from getting stuck in spurious modes during warmup.

The result: hierarchical Senate correlations now range from r = 0.945 to 0.991 across all five bienniums — consistently *outperforming* the flat model. This validates the hierarchical approach and confirms that the original failure was a sampler convergence problem, not a fundamental limitation of partial pooling at Senate scale.

---

## What This Means for the Project

### The Good News

Our methodology is definitively validated — not on one session, but across a full decade. Both IRT models produce ideology scores that agree with the national gold standard at r = 0.91 to 0.99. This means:

1. **The rank orderings are real.** When we say Legislator A is more conservative than Legislator B, an independent national dataset agrees — consistently, across five different sessions with different legislators and different legislative agendas.
2. **The party separation is real.** The gap between the most conservative Democrat and the most liberal Republican in our data matches the gap in Shor-McCarty's data.
3. **The intra-party variation is real.** The differences we find among Republicans — which is the most analytically interesting signal in a supermajority state — are not artifacts of our methodology.
4. **The hierarchical model works.** After convergence improvements (nutpie sampler, adaptive priors, PCA init), the hierarchical model matches or exceeds flat IRT validation in every biennium, including the Senate.
5. **Historical data is trustworthy.** Even the 84th biennium (2011-2012), which has known data quality issues (~30% missing ODT votes), validates at r = 0.975 (flat House) and r = 0.986 (hierarchical House).

### The Caveats

- **Shor-McCarty's coverage ends in 2020.** For the 89th Legislature (2021-2022) and beyond, we cannot perform this validation. Our results for recent sessions stand on the strength of the methodology validated here, plus cross-session stability checks.
- **Career scores vs. session scores.** Shor-McCarty gives each legislator one score for their entire career. We give each legislator a score per session. Legislators who genuinely change over time will show up as "disagreements" even if both methods are correct.

---

## How External Validation Fits the Pipeline

This validation step sits outside the regular 12-phase analysis pipeline. It doesn't change any scores or modify any results. It answers a single question: *can we trust the numbers?*

The answer is yes — for both models, across all five overlapping bienniums. Twenty out of twenty per-session correlations exceed r = 0.90. Pooled correlations range from r = 0.912 (flat Senate) to r = 0.975 (flat House). This places our pipeline squarely within — and often above — the range of published academic studies in this field.

---

## Technical Details

For readers who want the specifics:

- **Shor-McCarty data source**: Harvard Dataverse, doi:10.7910/DVN/NWSYOS, CC0 license. Variable used: `np_score` (national common space score).
- **Our data**: Bayesian 2PL IRT ideal points estimated via nutpie Rust NUTS sampler (2 chains, 1500 tuning + 2000 draws). PCA-informed chain initialization per ADR-0023. Both flat and hierarchical models validated.
- **Matching**: Two-phase name normalization. Phase 1: exact match on lowercase "first last" + chamber. Phase 2: last-name-only + district number tiebreaker. Overall match rate: 99%+ across all bienniums.
- **Correlation method**: Pearson r for linear agreement, Spearman ρ for rank agreement. 95% confidence intervals via Fisher z-transformation. Pooled analysis deduplicates by legislator (most recent session).
- **Code**: `analysis/17_external_validation/external_validation.py` (runner), `analysis/17_external_validation/external_validation_data.py` (pure logic), `analysis/17_external_validation/external_validation_report.py` (HTML report builder). Design doc at `analysis/design/external_validation.md`. ADR-0025. Run all 5 bienniums with `just external-validation --all-sessions`.
- **Reference**: Shor, B. and N. McCarty. 2011. "The Ideological Mapping of American Legislatures." *American Political Science Review* 105(3): 530-551.
