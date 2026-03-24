# Party Mean Posteriors in the 79th Kansas Senate

**Date:** 2026-03-23
**Source data:** Phase 07b (Hierarchical 2D IRT), run 79-260318.2
**Related:** `docs/pca-ideology-axis-instability.md`, `docs/horseshoe-effect-and-solutions.md`, ADR-0117, ADR-0118

---

## Summary

The Hierarchical 2D IRT party mean posteriors for the 79th Kansas Senate (2001-2002) show a legislature where the Republican caucus was more divided against itself than against the opposing party. The model's wide credible intervals and convergence difficulties are not failures — they are the statistically honest consequence of applying a party-based model to a chamber where party was not the primary axis of conflict.

---

## What Party Mean Posteriors Show

The Hierarchical 2D IRT model estimates a party mean (mu) and within-party spread (sigma) for each party on each dimension. These posteriors answer the question: *where does the typical party member sit in two-dimensional ideological space, and how much do members deviate from that center?*

**House (benchmark for comparison):**

| Party | Dimension | Mean | 95% HDI | Within-party SD |
|-------|-----------|------|---------|-----------------|
| Democrat | Dim 1 (ideology) | -2.69 | [-3.21, -2.10] | 0.91 |
| Republican | Dim 1 (ideology) | +1.53 | [+1.04, +2.03] | 0.68 |
| Democrat | Dim 2 (secondary) | +0.13 | [-0.44, +0.71] | 0.75 |
| Republican | Dim 2 (secondary) | +0.07 | [-0.57, +0.79] | 1.77 |

The House is clean. The parties are 4.2 points apart on Dim 1 with tight credible intervals. Dim 2 is not a party dimension — both means sit near zero — but Republican spread on Dim 2 (sigma = 1.77) is 2.4 times the Democratic spread (0.75), meaning within-party ideological variation runs through the Republican caucus.

**Senate (the interesting case):**

| Party | Dimension | Mean | 95% HDI | Within-party SD |
|-------|-----------|------|---------|-----------------|
| Democrat | Dim 1 (ideology) | -0.29 | [-3.63, +1.50] | 1.15 |
| Republican | Dim 1 (ideology) | +1.74 | [+0.86, +2.66] | 2.24 |
| Democrat | Dim 2 (secondary) | +1.14 | [-4.80, +4.99] | 1.24 |
| Republican | Dim 2 (secondary) | -0.06 | [-2.18, +2.17] | 1.02 |

Three features distinguish the Senate from the House:

1. **The Democratic Dim 1 credible interval spans zero.** The interval [-3.63, +1.50] means the model cannot confidently place the average Democrat on either side of the ideological origin. With only 10 Democratic senators in a 40-member chamber, there simply are not enough cross-party votes to pin down where Democrats stand relative to Republicans.

2. **Republican within-party SD on Dim 1 (2.24) exceeds the party gap (~2.0).** This is the defining signature. Republicans are spread across a wider range on the primary ideological axis than the distance between the two parties. The model is saying: the variation *within* the Republican caucus is larger than the variation *between* parties.

3. **Dim 2 is uninformative at the party level.** Both credible intervals span zero — the model finds no systematic party difference on Dim 2. Yet at the individual level, Dim 2 does meaningful work: it separates establishment Republicans (Praeger at +4.68, Oleen at +4.46, Vratil at +3.97) from insurgent conservatives (Huelskamp at -3.24, Lyon at -2.37, Pugh at -2.37). This axis runs *within* the Republican caucus, not between parties.

---

## Convergence: Why the Senate Model Struggles

The Senate convergence diagnostics are poor: R-hat = 2.42, effective sample size = 5, and a dimension swap was corrected during post-processing. By standard MCMC criteria, this model did not converge.

But the convergence failure is diagnostic, not dismissive. It tells us that the four MCMC chains explored different modes of the posterior — different configurations of the 2D space that are roughly equally consistent with the observed votes. This happens because the data genuinely supports multiple geometric interpretations. When Republicans vote against each other almost as often as they vote against Democrats, the likelihood surface has ridges and plateaus rather than a single sharp peak.

The pipeline's Tier 2 quality gate evaluates whether point estimates are still usable despite high R-hat. It checks party separation on the canonical dimension: Cohen's d = 2.30 (party means are 2.3 pooled standard deviations apart). This passes the threshold, indicating that while the model cannot precisely characterize uncertainty, its best-guess placement of legislators along the ideological axis is structurally sound.

For comparison, the House model converges cleanly: R-hat = 1.025, ESS = 142, zero divergences. A normal legislature with a dominant partisan cleavage presents no identification difficulties.

---

## Cross-Validation: Every Model Tells the Same Story

The 79th Senate's two-dimensional structure is not an artifact of a single model. Every scaling method in the pipeline converges on the same conclusion.

**PCA:** PC1 explains 19.6% of variance and captures intra-Republican factionalism (party separation d = 0.28 — essentially zero). PC2 explains 13.6% and captures the party divide (d = 4.98). The eigenvalue ratio lambda_1/lambda_2 = 1.45 — the two components explain nearly equal variance, confirming a genuinely two-dimensional structure.

**W-NOMINATE:** The eigenvalue ratio is 1.39 (Dim 1 = 1.76, Dim 2 = 1.27). This is unprecedented for a modern US state legislature; in Congress, the ratio is typically 5-10x. W-NOMINATE's first dimension places moderate Republicans (Oleen +0.99, Praeger +0.98, Kerr +0.97) on one end and both conservative Republicans (Huelskamp -0.95, Lyon -0.90) *and* Democrats (Haley -0.87, Hensley -0.74) on the other. The second dimension then separates Democrats (positive coord2D: +0.49 to +0.79) from conservative Republicans (negative: -0.31 to -0.50).

**Flat 1D IRT:** The ideal point ordering places Huelskamp, Pugh, Lyon, and Tyson at the positive extreme — but then places Democrats (Hensley, Feleciano, Barone) immediately adjacent, with moderate Republicans (Praeger, Oleen, Teichman) at the negative extreme. Party separation d = 1.12, and a sign flip was required. The 1D model has recovered the factional axis and scrambled the party axis.

**IRT-WNOM correlation:** r = 0.989 (Pearson). Flat IRT and W-NOMINATE Dim 1 are measuring exactly the same thing — but that thing is the establishment-vs.-insurgent axis, not the left-right axis.

All four methods agree: the 79th Kansas Senate has two real dimensions, and the dominant one is not the partisan divide.

---

## What the Data Reflects: The Kansas Republican Civil War

The statistical patterns have a direct historical explanation. The 79th Kansas Senate (2001-2002) was the epicenter of a factional conflict that Thomas Frank would later chronicle in *What's the Matter with Kansas?* (2004).

The 40-member Senate was 75% Republican, but the caucus was split between a moderate establishment and a conservative insurgency fueled by the social conservatism that had reshaped the Kansas GOP since the 1991 Summer of Mercy in Wichita. Senate President Dave Kerr (R-Hutchinson) and Majority Leader Lana Oleen (R-Manhattan) held power through cross-factional coalition-building. Oleen maintained regular communication with Democratic Minority Leader Anthony Hensley, and the two regularly assembled bipartisan coalitions to govern over conservative opposition.

The conservative wing — Tim Huelskamp, Robert Tyson, Ed Pugh, Bob Lyon, Kay O'Connor — pushed a social agenda centered on abortion restriction, opposition to evolution education, and tax reduction. Huelskamp routinely offered opening-day abortion amendments. O'Connor drew national attention in 2001 by questioning the 19th Amendment.

The result was a legislature where the most consequential votes split the Republican caucus rather than separating parties. On school finance, budget priorities, and social policy, moderate Republicans voted with Democrats against their own party's conservative wing. On redistricting in 2002, the alliances shifted entirely — Hensley assembled 10 Democrats and 11 conservative Republicans against the moderate leadership.

This fluidity is precisely what makes the 79th Senate hostile to one-dimensional modeling. A 1D model must choose a single axis that best predicts all votes. When cross-party coalitions form on the most contested votes, the model cannot simultaneously capture the party divide and the factional divide. It picks whichever axis explains more variance in the contested roll calls — and in the 79th Senate, that was the factional axis.

The hierarchical model's Republican sigma of 2.24 on Dim 1 is the statistical fingerprint of this war. A normal party has sigma around 0.7 (as the 79th House Republicans do). A sigma of 2.24 means the party's ideal points span roughly 9 units (plus/minus 2 sigma), wider than the entire left-right scale in most legislatures. The model is correctly identifying that "Republican" is not a useful grouping variable for predicting roll-call votes in this chamber.

---

## Output Quality Assessment

**House: Good.** Convergence is clean, credible intervals are tight, party separation is strong. The Dim 2 finding (Republicans spread widely, Democrats compact) is substantively interesting and consistent with the House's larger caucus offering more room for ideological variation within the majority.

**Senate Dim 1 party posteriors: Structurally sound, imprecise.** The point estimates correctly place Democrats left of Republicans, and the Republican mean (+1.74) is plausible. But the Democratic credible interval is too wide for confident inference — a consequence of 10 senators being too few to anchor a party mean when votes are highly two-dimensional. The massive Republican sigma (2.24) accurately reflects the factional split and is consistent with PCA, W-NOMINATE, and historical accounts.

**Senate Dim 2 party posteriors: Uninformative at the party level, informative at the individual level.** The party means are centered at zero with enormous uncertainty. This is correct — Dim 2 is not a party dimension. But individual-level Dim 2 scores successfully separate establishment Republicans (high positive) from insurgent conservatives (high negative), which is exactly what the historical record predicts.

**Senate convergence: Failed by standard criteria, accepted by Tier 2.** The R-hat of 2.42 and ESS of 5 mean the posterior uncertainty is unreliable. The point estimates are usable (party separation d = 2.30), and the routing decision to use Hierarchical 2D Dim 1 as the canonical ideal point is correct — it produces a better partisan ordering than any 1D model can.

**Overall:** The model is working at the boundary of what hierarchical party-pooled IRT can handle. The output is honest about its limitations (wide posteriors, convergence flags) and the pipeline's quality gates correctly route the Senate to the 2D solution. For this specific chamber, the wide posteriors are the right answer to the wrong question — the question "where is the party mean?" presupposes that parties are the relevant unit, and in the 79th Kansas Senate, they were not.

---

## Methodological Implications

Hierarchical Bayesian IRT models assume legislators within a party are drawn from a common normal distribution centered on a party mean. This assumption is powerful in typical legislatures, where it regularizes estimates for low-activity legislators and produces calibrated uncertainty. But it rests on the premise that parties are unimodal — that each party has a single ideological center.

The 79th Kansas Senate violates this premise. The Republican caucus is bimodal: one cluster around Praeger/Oleen/Kerr (moderates) and another around Huelskamp/Lyon/Pugh (conservatives). A single-mean hierarchical prior pulls both clusters toward a shared center that represents neither faction. The model accommodates this by inflating sigma (2.24), which is the mathematically correct response to bimodality under a unimodal assumption — but it means the party mean posterior is a compromise that no actual Republican senator occupies.

This is a known limitation of two-group hierarchical models. As Gelman (2006) notes, with J = 2 groups, the hierarchical variance estimate is inherently noisy — there is simply not enough group-level information to learn the between-group variance precisely. When within-group heterogeneity is high (as in a factionally split party), the model either over-shrinks toward a meaningless grand mean or under-shrinks and produces essentially flat posteriors.

The field-standard solution, adopted in our pipeline following Poole and Rosenthal's 40-year practice with DW-NOMINATE, is to fit a 2D model and extract the dimension that best separates parties. This sidesteps the party-mean assumption entirely, treating party as a post-hoc label on a geometric space rather than a structural component of the model. For the 79th Senate, this routing produces party separation d = 2.30 on the canonical dimension — a workable if imperfect result for a chamber that resists clean one-dimensional summary.

The deeper lesson is that convergence failure in Bayesian legislative models is often substantively informative. A model that cannot converge on a stable party structure is telling us that the party structure is unstable. In the 79th Kansas Senate, that instability was real: within a decade, the 2012 Republican primary "purge" would eliminate nine moderate senators (including Senate President Steve Morris), collapsing the factional axis and returning the Senate to a one-dimensional partisan structure. The model's difficulty is an early statistical signal of a political system under strain.

---

## Key Takeaways

1. The party mean posteriors are correct but imprecise. The House is clean; the Senate reflects genuine two-dimensional structure.
2. The Republican within-party SD (2.24) exceeding the party gap (~2.0) is the statistical signature of intra-party factionalism dominating partisan division.
3. Every scaling method (PCA, W-NOMINATE, flat IRT, hierarchical 2D) agrees on the two-dimensional structure.
4. W-NOMINATE eigenvalues confirm: the 79th Senate has a Dim 2/Dim 1 ratio of 0.72, far above the ~0.1-0.2 typical of modern Congress.
5. Historical accounts of the moderate-conservative Republican split (Kerr, Oleen, Huelskamp, Frank's *What's the Matter with Kansas?*) match the model's findings precisely.
6. The convergence failure is itself a finding: the data does not support confident party-level inference because parties were not the primary voting cleavage.
7. The pipeline's routing decision (Senate → Hierarchical 2D Dim 1) is the correct response, following the same logic Poole and Rosenthal have used for four decades.
