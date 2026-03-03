# Method Evaluation: What the Field Does and What We Should Consider

**An honest assessment of methods from the legislative analysis literature, evaluated against our data and pipeline.**

*2026-02-24*

---

## Context

After surveying the landscape of legislative vote analysis (see `docs/landscape-legislative-vote-analysis.md`), we evaluated every major method and tool for potential adoption. This document records what we considered, what we recommend, and — just as importantly — what we rejected and why.

The question is not "what sounds interesting?" but "what would actually improve our results, given our data, our pipeline, and our goals?"

---

## The Recommendation: External Validation with Shor-McCarty Scores

**Status:** Implemented and validated (2026-02-24). See `analysis/17_external_validation/external_validation.py`, ADR-0025, and `analysis/design/external_validation.md`. Results: flat IRT House r=0.981, flat IRT Senate r=0.929, hierarchical House r=0.984 (all "Strong"). Hierarchical Senate r=-0.541 due to J=2 over-shrinkage with 11 Democrats — documented in `docs/hierarchical-shrinkage-deep-dive.md`. General-audience writeup in `docs/external-validation-results.md`.

### The Gap

Every validation in our pipeline is internal:

- PCA correlates with IRT at r > 0.98
- Holdout vote prediction accuracy is 93%, AUC 0.97
- Cross-session alignment shows returning members are stable
- Hierarchical IRT correlates with flat IRT at r > 0.97

All of this tells us our model is *internally consistent*. None of it tells us our ideal points are *correct*. We have never compared our results to an independent external measure of ideology. Every published ideal point paper does this. It is the single biggest credibility gap in the pipeline.

### The Data

Boris Shor and Nolan McCarty's state legislature ideology project provides exactly what we need:

- **610 Kansas legislators** with ideology scores (`np_score`)
- **Coverage: 1996-2020**, overlapping our 84th-88th bienniums (2011-2020)
- **Free, CC0 license**, 6 MB download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NWSYOS)
- **Roll-call-based methodology** (bridged via NPAT survey responses), measuring the same construct as our IRT
- **Clean name data** for 2011-2020: "Last, First" format, matchable to our `legislator_slug` via name normalization
- District numbers available per year for secondary matching

The methodology is described in:

> Shor, B. and N. McCarty. 2011. "The Ideological Mapping of American Legislatures." *American Political Science Review* 105(3): 530-551.

Their approach uses Project Vote Smart's National Political Awareness Test (NPAT) — a standardized survey answered by legislators in multiple states — as "bridge observations" that place all state legislators on a common ideological scale. Within each state, they estimate Bayesian IRT ideal points from roll call votes, then link them via the NPAT bridges to produce a national common space.

### The Comparison

Shor-McCarty `np_score` is **fixed per legislator's career** — a single score across all sessions they served. Our IRT `xi_mean` varies by biennium. This is a limitation: a legislator who drifted ideologically between 2011 and 2020 would have one Shor-McCarty score but five different scores in our pipeline.

But the rank ordering should be highly correlated. If our IRT says Legislator A is more conservative than Legislator B, Shor-McCarty should agree. Pearson and Spearman correlations above 0.85 would constitute strong external validation. Below 0.85 would signal a problem worth investigating.

The comparison also tests our PCA-informed initialization and non-centered hierarchical parameterization against an established methodology. If we converge on the same answers despite different MCMC implementations (PyMC vs. pscl/R) and different identification strategies (PCA-init + anchors vs. informative priors), that's evidence both approaches are recovering the true latent structure.

### Proposed Implementation

A lightweight new phase: `analysis/17_external_validation/external_validation.py`. No MCMC, no heavy computation — just data download, name matching, correlation analysis, and a few plots:

1. **Download** Shor-McCarty `.tab` file from Harvard Dataverse (one-time, cached)
2. **Filter** to Kansas (`st == "KS"`) and relevant bienniums
3. **Match** legislators by normalized name + chamber + district
4. **Compute** Pearson and Spearman correlations between `np_score` and our `xi_mean` per biennium
5. **Plot** scatter plots (one per biennium, one pooled) with regression lines and confidence bands
6. **Report** match rates, correlation coefficients, outliers (legislators where the two methods disagree most)
7. **Optional**: Compare party-separated correlations (intra-Republican and intra-Democrat rank agreement)

The five overlapping bienniums (84th-88th) give us five independent correlation estimates, plus a pooled estimate. This is enough to assess whether our ideal points are externally valid and whether the validation is stable across sessions.

### What It Would Tell Us

- **r > 0.90**: Our IRT ideal points are essentially interchangeable with the field standard. Strong validation.
- **r = 0.85-0.90**: Good agreement. Differences likely reflect session-specific dynamics our biennium-specific model captures but Shor-McCarty's career-fixed score does not.
- **r = 0.70-0.85**: Moderate agreement. Investigate outliers — are they legislators who shifted ideology mid-career, or is our model producing different rank orderings?
- **r < 0.70**: Concerning. Would require investigating whether the issue is our model, data quality, or systematic differences in methodology.

We would also check whether the five biennium-specific correlations are stable. If r = 0.92 for the 85th but r = 0.75 for the 84th (which has known convergence failures), that confirms our analytic flags about historical session quality.

---

## What We Considered and Rejected

### Vote-Type-Stratified IRT (Issue-Specific Ideal Points)

**Source**: Shin (2024), "Measuring Issue Specific Ideal Points from Roll Call Votes" (IssueIRT). Lipman, Moser, and Rodriguez (2025), "Explaining Differences in Voting Patterns Across Voting Domains Using Hierarchical Bayesian Models" (*Political Analysis*).

**The idea**: Estimate separate ideal points for different vote types — do legislators' positions on veto override votes differ from their positions on final action votes?

**Why we rejected it**: Our data doesn't support it. The clustering phase already showed that veto override voting in Kansas is strictly party-line: Republicans vote Yea at 98%, Democrats at 1%, Rice cohesion above 0.96 for both parties. There is no ideological variation to model. The override subnetwork has near-zero edges because nearly everyone votes the same way within their party.

Other vote types have sample size problems: Committee of the Whole (56 votes), Consent Calendar (22 votes), Procedural Motions (5 votes). These are too few for reliable per-legislator IRT estimation.

IssueIRT is designed for settings with policy domain labels on bills and enough bills per domain to estimate domain-specific ideal points. Kansas's vote data doesn't have policy domain labels, and the natural categorization (vote type) doesn't produce interesting subgroup variation. If we obtained policy area labels from bill text classification, this could become viable — but that's a substantial data engineering project, not a method adoption.

### DIME/CFscores as Second External Validation

**Source**: Bonica, A. 2014. "Mapping the Ideological Marketplace." *AJPS* 58(2). Data at [data.stanford.edu/dime](https://data.stanford.edu/dime).

**The idea**: Campaign-finance-based ideology scores provide a completely independent measure — they capture who donors think you are, not how you vote.

**Why we deferred it**: Within-party correlation between CFscores and Shor-McCarty drops to 0.65-0.67. For our Kansas analysis, where intra-Republican variation is the most interesting signal, CFscores may not have enough resolution to validate our within-party rank orderings. The data is also large (~2 GB compressed), requires filtering, and matching is harder than Shor-McCarty.

Worth doing eventually as triangulation if the Shor-McCarty validation goes well. Not the first priority.

### W-NOMINATE Benchmarking

**Source**: Poole, Lewis, Lo, and Carroll. `wnominate` R package. The field standard for ideal point estimation.

**The idea**: Run wnominate in R on our vote matrix and compare the resulting ideal points to our IRT scores. This would directly test our claim that PCA + Bayesian IRT covers the same ground as NOMINATE.

**Why we deferred it**: Published studies already show PCA and NOMINATE first dimensions correlate at r > 0.95. Our IRT and PCA correlate at r > 0.98. By transitivity, IRT ≈ NOMINATE. A direct comparison would confirm this but is unlikely to surprise.

More practically, wnominate requires R, which violates our technology preferences. A one-time cross-validation exercise in R is defensible for a methods paper, but it's not a pipeline addition.

### Strategic Absence Modeling

**Source**: Kubinec, R. 2024. "Generalized Ideal Point Models for Robust Measurement with Dirty Data." idealstan R/Stan package.

**The idea**: Treat legislative absences as informative rather than missing at random. A hurdle model separates the "decide to participate" stage from the "vote Yea or Nay" stage, allowing ideal points to be influenced by strategic absence patterns.

**Why we rejected it**: Kansas's absence rate is 2.6%, and "Present and Passing" (the clearest strategic non-vote) accounts for only 22 instances out of 77,865 individual votes. The practical impact on ideal points is negligible.

However, a quick empirical check is worthwhile: are absence rates correlated with ideology? If moderate Republicans are disproportionately absent on party-line votes, that would be evidence our missing-at-random assumption is somewhat violated. This belongs in `docs/analytic-flags.md` as a one-time investigation, not a new model.

### Dynamic IRT Within Biennium

**Source**: Martin and Quinn (2002), dynamic Bayesian IRT with random walk prior. Imai et al. (2016), emIRT `dynIRT` for fast dynamic estimation.

**The idea**: Allow ideal points to change over time within a single biennium. Track whether legislators shift during the session.

**Why we rejected it**: Kansas bienniums span ~2 years, with active voting concentrated in two ~90-day sessions. Within-biennium ideological drift is implausible for most legislators — they don't fundamentally change their views in 90 days. The cross-session validation module already handles between-biennium comparison, where genuine shifts are more plausible (redistricting, changed circumstances, new committee assignments).

Dynamic IRT within a biennium would also require substantially more tuning samples and longer convergence times for minimal interpretive gain.

### Cosponsorship Networks

**Source**: Fowler, J. 2006. "Connecting the Congress: A Study of Cosponsorship Networks." *Political Analysis* 14(4). Showed cosponsorship network centrality predicts amendment passage and vote choice beyond ideology.

**The idea**: Build a second network from bill cosponsorship data, complementing our co-voting network. Cosponsorship reveals proactive collaboration, while co-voting reveals passive agreement.

**Why we deferred it**: We only have primary sponsor data from the KLISS API (the `sponsor` field in `rollcalls.csv`). Cosponsor data would require new scraping infrastructure — parsing additional bill detail pages for cosponsor lists. This is a substantial engineering investment for incremental analytical insight.

If the scraper is ever extended to capture cosponsor data, this becomes a natural addition to the network phase.

### Generalized Graded Unfolding Model (GGUM)

**Source**: Duck-Mayr and Montgomery. `bggum` R package. Handles non-monotonic item response functions where legislators at both ideological extremes vote together against the center.

**The idea**: Standard IRT assumes that the probability of voting Yea increases monotonically with ideology (or decreases monotonically). GGUM handles single-peaked preferences where both extremes oppose a centrist bill.

**Why we rejected it**: This pattern — far-left and far-right voting together against the middle — doesn't occur in Kansas. Veto override votes (the most plausible candidate) are strictly party-line, not strange-bedfellows coalitions. Our standard 2PL IRT achieves 93% holdout accuracy, suggesting the monotonic assumption is adequate.

### LLM Agents for Legislative Simulation

**Source**: Li, Gong, and Jiang (AAAI 2025). "Political Actor Agent: Simulating Legislative System for Roll Call Votes Prediction with Large Language Models."

**Why we rejected it**: Too experimental. Our XGBoost already achieves 0.98 AUC on vote prediction. Simulating legislators with LLMs would add computational cost and interpretive complexity for no measurable improvement in our core task of ideology estimation.

### Text-Based Ideal Points (TBIP)

**Source**: Vafa, Naidu, and Blei (ACL 2020). Unsupervised probabilistic topic model that estimates ideal points from text alone.

**Why we rejected it**: Requires individual authorship. Kansas bills are ~92% committee-sponsored — only ~27 individual sponsors across ~38 bills in the 91st Legislature, insufficient for stable TBIP estimates. **Replaced by Phase 21** (ADR-0086): embedding-vote approach multiplies vote matrix by Phase 18 bill embeddings, extracts PC1 via PCA as text-derived ideal point. Validates against IRT (flat + hierarchical).

### emIRT (Fast EM-Based Ideal Points)

**Source**: Imai, Lo, and Olmsted (2016). `emIRT` R package.

**Why we rejected it**: R-only. Speed is not our bottleneck — the full 91st Legislature pipeline completes in under 2 hours including the joint cross-chamber model, and the IRT phase itself takes under 6 minutes. PyMC gives us full posterior distributions (credible intervals for every ideal point), which emIRT's EM approach does not. For a single-state analysis where uncertainty quantification matters, MCMC is the right tool.

---

## Summary

| Method | Verdict | Rationale |
|--------|---------|-----------|
| **Shor-McCarty external validation** | **Implemented** | Addresses biggest credibility gap; low effort, high value (ADR-0025) |
| DIME/CFscores validation | Deferred | Lower within-party resolution; do after Shor-McCarty |
| Vote-type-stratified IRT | Rejected | Data doesn't support it (party-line overrides, small N) |
| W-NOMINATE benchmarking | Deferred | Published studies already show PCA ≈ NOMINATE; requires R |
| Strategic absence modeling | Not needed | 2.6% absence rate; investigate empirically, don't model |
| Dynamic IRT within biennium | Rejected | 2-year window too short; cross-session handles between-biennium |
| Cosponsorship networks | Deferred | Requires new scraping infrastructure |
| GGUM unfolding | Rejected | No extreme-alliance voting pattern in Kansas |
| LLM agents | Rejected | Too experimental; XGBoost already at 0.98 AUC |
| TBIP text-based scaling | Replaced | Phase 21 uses embedding-vote approach (ADR-0086) |
| emIRT | Rejected | R-only; PyMC gives posteriors; speed not a bottleneck |

The pipeline's methodology is sound. The gap is not in our methods — it's in proving our results are valid against independent measurement.
