# Common Space Ideal Points: A 28-Year Scale for the Kansas Legislature

**Date:** 2026-03-24
**Scope:** 78th-91st Legislatures (1999-2026), House and Senate
**Related:** `docs/dynamic-ideal-points-deep-dive.md`, `docs/pca-ideology-axis-instability.md`, ADR-0058, ADR-0070, ADR-0118

---

## The Problem

The Tallgrass pipeline produces ideal point estimates for every Kansas legislator, but each biennium's scores live on an independent scale. A score of +1.5 in the 79th Legislature (2001-2002) and +1.5 in the 91st Legislature (2025-2026) don't mean the same thing — the identification constraints, anchors, and sign conventions are session-specific. There is no way to directly compare Tim Huelskamp's conservatism in 2001 with any current legislator's, or to ask whether the Kansas Senate has polarized over the last quarter-century.

Creating a single cross-temporal scale — a "common space" — is one of the most studied problems in quantitative political science. This article surveys the approaches, evaluates them against the Kansas data, and proposes a solution built on the infrastructure that already exists in the pipeline.

---

## What the Literature Offers

### Bridge Legislators: The Natural Anchor

The fundamental insight, common to every cross-temporal method, is that legislators who serve in multiple sessions are natural anchors. If Senator X served in both the 79th and 80th, and her true ideology didn't change dramatically between sessions, then the difference between her two scores tells us how the two scales relate. With enough such bridges, we can estimate the affine transformation (slope A, intercept B) that maps one session's scale onto another's.

Kansas has exceptional bridge coverage. No term limits and ~170 seats per chamber produce 62-85% overlap between adjacent bienniums — far above the psychometric minimum of 20-25 bridge persons. Three legislators (Barbara Ballard, David Haley, Henry Helgerson) span the entire 28-year period from 1999 to 2026.

### Shor & McCarty (2011): The Field Standard

Boris Shor and Nolan McCarty created the most widely used cross-state, cross-temporal ideal point dataset for American state legislatures. Their method uses a two-step bridge: first, they pool state legislators' and members of Congress' responses to the Project Vote Smart National Political Awareness Test (NPAT) to create a shared survey-based space, then they project roll call ideal points into that space.

The key assumption is that legislators use the same ideal point when answering surveys as when casting votes. The method produces career-fixed scores — each legislator gets a single number across their entire career, with no within-career dynamics. Coverage spans the mid-1990s to the present across all 50 states.

Shor-McCarty scores are already used as external validation in Phase 14 of the Tallgrass pipeline, covering the 84th-88th Kansas bienniums. They provide an independent check on absolute scale positioning.

### Martin & Quinn (2002): Dynamic Ideal Points

Originally developed for the U.S. Supreme Court, the Martin-Quinn state-space model treats ideal points as evolving over time via a Gaussian random walk:

```
theta[j,t] = theta[j,t-1] + epsilon[j,t]    where epsilon ~ N(0, Delta_j)
```

The random walk prior *is* the bridging mechanism — temporal dependence connects the scales automatically. No explicit linking step is needed. The researcher chooses the evolution variance Delta, which controls smoothing: too small freezes ideal points artificially, too large makes each period independent.

Phase 27 (Dynamic IRT) of the Tallgrass pipeline already implements this model for 8 bienniums (84th-91st), with innovations including per-party evolution variance, informative priors from static IRT, and three-layer sign identification (ADR-0070). It produces smooth trajectories for long-serving legislators and decomposes polarization into conversion (returning members moving) and replacement (turnover) components.

### Groseclose, Levitt & Snyder (1999): Adjusted Scores

GLS proposed the simplest bridge-based approach: estimate affine transformations for each Congress that map raw interest group ratings (ADA scores) onto a common scale. Bridge legislators who serve in overlapping periods identify the transformation parameters. Their key finding was that naive cross-temporal comparison of raw scores is seriously misleading — adjusted scores revealed a strong liberal trend in Congress during 1947-94 that was invisible in raw data.

### DW-NOMINATE: Linear Career Trends

The "D" in DW-NOMINATE constrains each legislator to a linear trajectory over their career: `x[i,t] = x_start + (x_end - x_start) * (t - t_start) / (t_end - t_start)`. This is computationally efficient but cannot capture nonlinear change — if a legislator moderated after a primary scare and then re-radicalized, the linear trend misses it entirely. Bateman & Lapinski (2016) argue this is a fundamental limitation for studying periods of rapid political realignment.

### Bailey (2007): Bridge Observations

Michael Bailey's method is the most sophisticated. He identifies four types of bridging information: cross-institutional positions (members of Congress taking positions on Supreme Court cases), temporal citations (justices ruling on prior cases), ideological ordering constraints (expert coding), and within-institution votes on shared topics. This produces a unified scale spanning presidents, senators, representatives, and Supreme Court justices from 1950-2020.

Bailey's approach demonstrates that bridges need not be roll call votes — any observable behavior with ideological content can serve. For Kansas, this suggests that external data (campaign donations, survey responses, bill sponsorship patterns) could supplement roll call bridges.

---

## Kansas Bridge Coverage

Adjacent biennium overlap, measured by name-matched legislators:

| Pair | Bridges | Overlap |
|------|---------|---------|
| 78th → 79th | 133 | 79.6% |
| 79th → 80th | 134 | 79.8% |
| 80th → 81st | 122 | 71.8% |
| 81st → 82nd | 140 | 83.3% |
| 82nd → 83rd | 140 | 83.3% |
| 83rd → 84th | 127 | 75.1% |
| **84th → 85th** | **106** | **62.7%** |
| 85th → 86th | 145 | 85.3% |
| 86th → 87th | 109 | 64.1% |
| 87th → 88th | 109 | 64.1% |
| 88th → 89th | 112 | 65.9% |
| 89th → 90th | 125 | 74.9% |
| 90th → 91st | 132 | 79.0% |

The weakest link is the 84th→85th transition (62.7%), reflecting the 2012 Republican primary "purge" that eliminated nine moderate senators and triggered substantial House turnover. Even this weakest link has 106 bridges — more than four times the psychometric minimum.

Three legislators span the entire 28-year period: Barbara Ballard (D-House), David Haley (D-Senate), and Henry Helgerson (D-House). All three are Democrats, which limits their utility as sole anchors but provides valuable end-to-end constraints in a simultaneous alignment.

---

## The Recommended Approach: Simultaneous Common Space Alignment

### Why Not Sequential Chaining

The naive approach — link 78th to 79th, 79th to 80th, and so on — accumulates error with each link. The 78th's scores would pass through 13 affine transformations before reaching the 91st's scale, with each transformation adding noise. By the early bienniums, the uncertainty would be substantial.

### Simultaneous Alignment

Instead, estimate all 14 sessions' transformations at once in a single optimization. Fix the 91st Legislature as the reference scale (A = 1, B = 0). For each other biennium t, find the affine parameters (A_t, B_t) that minimize the discrepancy across *all* bridge pairs — not just adjacent ones.

The objective function:

```
minimize sum over all bridge pairs (i, s, t):
    (A_s * xi_s[i] + B_s  -  A_t * xi_t[i] - B_t)^2

subject to: A_91st = 1, B_91st = 0
```

A legislator who served in the 78th, 80th, and 84th contributes six equations (three pairwise comparisons). The three 28-year veterans contribute direct constraints between the earliest and latest sessions, bypassing all intermediate links. The system has 26 unknowns (13 sessions × 2 parameters) with thousands of observations — massively over-determined and very stable.

This is essentially the Groseclose, Levitt & Snyder (1999) approach applied to Bayesian IRT ideal points rather than ADA ratings.

### Input: Canonical Ideal Points

The alignment operates on *canonical* ideal points — the horseshoe-corrected scores from the pipeline's routing system (ADR-0109). For Senate sessions with the horseshoe effect (primarily 78th-83rd and 88th), the canonical score is Hierarchical 2D Dim 1 rather than flat 1D IRT. For clean sessions, it's flat 1D. The hard work of per-biennium identification is already done; the common space phase just links the corrected scales.

### Uncertainty: Bootstrap

Resample bridge legislators 1,000 times. For each resample, re-estimate all (A, B) pairs simultaneously. Each legislator's common-space score gets a 95% confidence interval that honestly reflects the chain distance from the reference. Legislators in the 78th will have wider intervals than those in the 91st — this is the correct behavior.

### Cross-Chamber Unification

House and Senate alignments run separately (different ideal point scales, different bill sets). Legislators who switched chambers during their career (e.g., served in the House in the 82nd and the Senate in the 84th) provide natural cross-chamber bridges. These are sparse but valuable — a supplementary alignment step can unify the two scales.

---

## Validation Strategy

Three independent external checks, all already integrated into the pipeline:

1. **Shor-McCarty scores** (84th-88th coverage): Career-averaged common-space scores should correlate strongly with SM career-fixed scores. This tests whether the alignment preserves the well-established cross-state comparable scale.

2. **DIME CFscores** (84th-89th coverage): Campaign-finance-derived ideology is entirely independent of voting behavior. Strong correlation validates that the common space is measuring something real, not just a roll call artifact.

3. **Dynamic IRT** (84th-91st coverage): The state-space model produces independently estimated cross-temporal trajectories. Agreement between the chain-linked point estimates and the Bayesian trajectories validates the linking methodology.

Additionally, the common-space scores should pass internal consistency checks:
- Party separation should be stable or slowly trending (no abrupt sign flips)
- Bridge legislators' aligned scores should be highly correlated across sessions (r > 0.85)
- The polarization trend should match known Kansas political history (moderate era → Brownback polarization → partial re-moderation)

---

## Known Limitations

**Absolute drift is undetectable from votes alone.** If the entire Kansas legislature shifted right by 0.5 units between 2001 and 2025, bridge-based methods cannot detect it. Every bridge legislator moved by the same amount, so the relative positioning looks unchanged. Only external anchors (SM, DIME) can detect absolute drift, and their coverage is incomplete.

**Bridge legislator selection bias.** Returning legislators are a non-random sample. They tend to be electorally safer (surviving both primaries and generals) and more senior. If survivors are systematically more moderate or more extreme than departing legislators, the linkage may be biased. The trimmed regression (excluding the 10% most extreme residuals) partially mitigates this, and with 100+ bridges per link, individual bias washes out.

**Dimensionality change across eras.** The Kansas Senate had a genuinely two-dimensional voting structure before ~2013 (the moderate-conservative Republican factional split) and a one-dimensional structure after (post-purge). The canonical routing corrects this per-biennium, but the *meaning* of the primary dimension may have shifted. The 7 quality gates (ADR-0118) provide defense, but this remains the deepest conceptual challenge.

**The 84th-85th gap.** The 2012 redistricting and Republican primary purge produced the weakest bridge (62.7%). The simultaneous alignment mitigates this because non-adjacent bridges bypass the gap, but it remains the most fragile link in the chain.

**KanFocus-only bienniums.** The 78th-83rd have KanFocus data only (votes, rollcalls, legislators — no bill texts, no bill actions). The pipeline needs to run flat IRT on these sessions before they can be included. The canonical routing system will need to assess horseshoe status and route accordingly.

---

## Relationship to Existing Pipeline Phases

| Phase | Role in Common Space |
|-------|---------------------|
| Phase 05 (Flat IRT) | Provides per-biennium ideal points — the raw input |
| Phase 06 (2D IRT) | Provides Dim 1 for horseshoe sessions |
| Phase 07b (Hierarchical 2D) | Provides H2D Dim 1 (preferred for horseshoe sessions) |
| Canonical routing | Selects best available score per chamber-session |
| Phase 26 (Cross-Session) | Provides pairwise affine alignment — the building block |
| Phase 27 (Dynamic IRT) | Independent cross-temporal trajectories — validation |
| Phase 14 (Shor-McCarty) | External validation anchor |
| Phase 18 (DIME) | External validation anchor |
| **Phase 28 (Common Space)** | **New: simultaneous alignment across all 14 bienniums** |

The common space phase sits downstream of all per-biennium analysis and upstream of any cross-temporal reporting. It consumes canonical ideal points and produces a universal score file.

---

## What This Enables

With a validated common space scale, the platform can answer questions that no single-biennium analysis can:

- **Cross-era comparison:** "Was Tim Huelskamp in 2001 more conservative than [any current legislator]?"
- **Polarization trajectory:** "How has the distance between party means changed from 1999 to 2026?"
- **Replacement vs. conversion:** "Did the Kansas Senate move right because existing members changed their votes, or because moderate members were replaced by conservatives?"
- **Career trajectories:** "Which legislators moved the most over their careers, and in which direction?"
- **Institutional memory:** "How does today's legislature compare to the pre-Brownback era?"

These are the questions that journalists, policymakers, and political scientists ask — and they require a cross-temporal scale to answer.

---

## References

- Bailey, M. A. (2007). Comparable preference estimates across time and institutions for the Court, Congress, and Presidency. *American Journal of Political Science*, 51(3), 433-448.
- Bateman, D. A., & Lapinski, J. S. (2016). Ideal points and American political development. *Studies in American Political Development*, 30(2), 147-171.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *American Political Science Review*, 98(2), 355-370.
- Groseclose, T., Levitt, S. D., & Snyder, J. M. (1999). Comparing interest group scores across time and chambers: Adjusted ADA scores for the U.S. Congress. *American Political Science Review*, 93(1), 33-50.
- Martin, A. D., & Quinn, K. M. (2002). Dynamic ideal point estimation via Markov chain Monte Carlo for the U.S. Supreme Court, 1953-1999. *Political Analysis*, 10(2), 134-153.
- Shor, B., & McCarty, N. (2011). The ideological mapping of American legislatures. *American Political Science Review*, 105(3), 530-551.
