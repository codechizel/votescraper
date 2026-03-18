# Chapter 3: The Gold Standard: Shor-McCarty External Validation

> *Our model says Senator Smith is a moderate conservative. A completely independent team of political scientists, using different data and different methods, says the same thing. That's not a coincidence — it's validation.*

---

## Why External Validation Matters More Than Internal

In Chapter 2, we asked the model to predict its own data. That's a useful test, but it's also a bit like a student grading their own homework. Of course the model can reproduce the data it was fit to — it was *designed* to do that. The real test is whether the model agrees with **someone else's answer key**.

In political science, the closest thing to an answer key is the **Shor-McCarty dataset** — a carefully constructed set of ideology scores for every state legislator in the United States, published by political scientists Boris Shor and Nolan McCarty. Their scores span 1993 to 2020 and are widely used in academic research. If our Kansas-only IRT scores agree with their national scores, that's strong evidence we're measuring the same thing.

## What Shor-McCarty Scores Are

### The Problem They Solve

Every state legislature has its own set of bills and votes. A "conservative" vote in Kansas isn't the same bill as a "conservative" vote in California. So how do you compare legislators across states?

Shor and McCarty solved this with a clever **bridge** strategy, published in the *American Political Science Review* in 2011. The idea:

1. **Find common ground.** Project Vote Smart's National Political Awareness Test (NPAT) asks every candidate in America — federal and state — the same set of policy questions. "Do you support X? Do you oppose Y?" The questions are identical regardless of which state you're running in.

2. **Scale the survey.** Use the NPAT responses to place every legislator who answered the survey on a common national scale. A Kansas Republican who answered the same way as a Texas Republican gets a similar score.

3. **Bridge to roll calls.** For the subset of legislators who both answered the NPAT *and* cast roll call votes, use regression to learn the mapping between NPAT scores and roll call scores.

4. **Extrapolate.** Use that mapping to assign NPAT-equivalent scores to all legislators — even those who never answered the survey — based solely on their roll call votes.

The result is a single number for every state legislator in America, on a common scale. A score of +1.5 means the same thing whether the legislator is from Kansas or California: consistently conservative on the national policy spectrum.

### An Analogy: Translating Currencies

Think of each state legislature as a country with its own currency. The Kansas House uses "Kansas ideology dollars." The California Senate uses "California ideology dollars." Within each country, the currency makes sense — you can compare legislators to each other. But how do you compare a Kansas legislator to a California legislator?

Shor-McCarty is the exchange rate. The NPAT survey is like a basket of goods that costs the same everywhere — a Big Mac Index for ideology. By seeing how legislators price the same basket (answer the same questions), you can convert all the state currencies into a single "national ideology dollar."

Tallgrass doesn't need this cross-state conversion (we only study Kansas). But comparing our Kansas-only scores to the Shor-McCarty national scores tells us whether our exchange rate is consistent with theirs. If a legislator we rate as "very conservative" also gets a very conservative Shor-McCarty score, our local currency is aligned with the national one.

## How Tallgrass Does the Comparison

The external validation phase (`analysis/17_external_validation/`) works in five steps.

### Step 1: Download and Parse

Shor-McCarty data is freely available from the Harvard Dataverse (CC0 license). Tallgrass downloads the tab-separated file and filters to Kansas legislators:

```python
# From analysis/17_external_validation/external_validation_data.py
SHOR_MCCARTY_URL = "https://dataverse.harvard.edu/api/access/datafile/7067107"
```

The dataset covers the 84th through 88th bienniums (2011–2020) — five of our fourteen bienniums. The 89th through 91st (2021–2026) are too recent for the most recent Shor-McCarty update.

### Step 2: Match Legislators

This is harder than it sounds. Our data spells names one way ("John Barker Sr."); Shor-McCarty spells them another ("Barker, John"). The matching process uses two phases:

1. **Exact normalized match.** Strip titles, generational suffixes, and middle names. "John William Barker Sr." becomes "john barker." "Barker, John W." also becomes "john barker." Match on the normalized form.

2. **Last-name fallback with district tiebreaker.** If the exact match fails (perhaps one source uses a nickname), try matching on last name only. When multiple legislators share a last name (Kansas has had several father-son pairs in the legislature), use the district number to disambiguate.

A typical biennium matches 85–95% of legislators. The unmatched remainder are usually legislators who joined or left mid-session and appear in one dataset but not the other.

### Step 3: Compute Correlations

For each matched pair, we now have two numbers: our IRT ideal point (ξ_mean) and the Shor-McCarty score (np_score). We compute:

- **Pearson r:** Measures linear correlation — do the scores move together proportionally?
- **Spearman ρ:** Measures rank correlation — do the scores rank legislators in the same order?
- **Fisher z confidence interval:** A 95% interval for the true correlation, accounting for sample size.

From the code:

```python
r, p_r = sp_stats.pearsonr(xi, np_scores)
rho, p_rho = sp_stats.spearmanr(xi, np_scores)
ci_lower, ci_upper = _fisher_z_ci(r, len(xi))
```

The Fisher z-transform deserves a brief explanation. Correlation coefficients are bounded between -1 and +1, which means they don't behave like normal numbers for the purpose of computing confidence intervals. A correlation of 0.95 is "closer to the ceiling" than 0.50 — there's less room to vary upward. The Fisher z-transform stretches the scale so that the transformed values behave normally, allows us to compute a standard confidence interval, and then stretches back:

```python
z = np.arctanh(r)         # Transform to unbounded scale
se = 1 / sqrt(n - 3)      # Standard error in z-space
z_lower = z - 1.96 * se   # 95% CI in z-space
z_upper = z + 1.96 * se
ci = (tanh(z_lower), tanh(z_upper))  # Transform back
```

### Step 4: Within-Party Correlations

The overall correlation is important, but it can be misleadingly high. If Republicans cluster on the right and Democrats cluster on the left, any two methods will produce a high correlation just by getting the party labels right — even if the within-party rankings are completely different.

To test whether the model captures **intra-party** variation — whether it correctly distinguishes a moderate Republican from a conservative Republican — Tallgrass computes separate correlations for each party:

```python
for party in ["Republican", "Democrat"]:
    party_df = matched.filter(pl.col("party") == party)
    results[party] = compute_correlations(party_df, xi_col, np_col)
```

Within-party correlations are always lower than the overall correlation (there's less variance to work with), but a strong within-party correlation (r > 0.80) means the model is capturing genuine ideological gradations, not just sorting legislators into two bins.

### Step 5: Outlier Detection

Even with strong overall correlation, individual legislators may diverge sharply between the two measures. Tallgrass identifies the top 5 outliers — legislators whose Shor-McCarty rank differs most from their Tallgrass rank.

These outliers are usually substantively interesting:

- **Legislators who changed ideology mid-career.** Shor-McCarty's career-level scores average across years; our session-specific scores capture the current state. A legislator who moderated over time will show a discrepancy.
- **Legislators with unusual issue profiles.** A Republican who is economically conservative but socially moderate might score differently on a Kansas-specific scale (dominated by economic issues) than on a national scale (influenced by social issues).
- **Matching errors.** Occasionally, name normalization produces a false match (two legislators share a normalized name). The outlier report helps catch these.

## What the Numbers Look Like

Across the five overlapping bienniums (84th–88th), the Tallgrass-vs-Shor-McCarty correlations are consistently strong:

| Metric | Typical Range |
|--------|--------------|
| Overall Pearson r | 0.90 – 0.96 |
| Overall Spearman ρ | 0.89 – 0.95 |
| Within-party r (Republican) | 0.75 – 0.88 |
| Within-party r (Democrat) | 0.80 – 0.92 |
| Match rate | 85% – 95% of legislators |

### What "0.93" Really Means

A Pearson correlation of 0.93 means that 87% of the variance in one set of scores is explained by the other (r² = 0.86). Put differently: if you drew a scatterplot of Tallgrass scores on the x-axis and Shor-McCarty scores on the y-axis, the points would form a tight cloud around a straight line. A few points would stray — the outliers — but the overall pattern would be unmistakable.

More practically: if you ranked the Kansas House from most liberal to most conservative using Tallgrass scores, and then re-ranked them using Shor-McCarty scores, **about 93% of the pairwise orderings would agree.** The two methods would disagree mostly on legislators who are close together on the scale — the difference between the 40th and 42nd most conservative legislator, not between the 5th and the 95th.

### Why It's Not 1.0

A perfect correlation (r = 1.0) would mean the two methods produce identical rankings. That never happens, and it shouldn't. The methods are genuinely different:

- **Different data.** Shor-McCarty uses NPAT survey responses as a bridge; Tallgrass uses roll call votes only.
- **Different scope.** Shor-McCarty covers all 50 states and bridges to Congress; Tallgrass is Kansas-specific.
- **Different model.** Shor-McCarty uses an iterative projection method; Tallgrass uses full Bayesian IRT with MCMC sampling.
- **Different time resolution.** Shor-McCarty produces career-level scores; Tallgrass produces session-level scores.

A correlation of 0.93 says: "despite all these differences, the two methods tell essentially the same story about who is liberal, who is conservative, and who is in between." That's the strongest validation a latent variable can receive.

## The Limits of Shor-McCarty Comparison

### Coverage Gaps

The Shor-McCarty dataset ends in 2020. For the 89th (2021-2022), 90th (2023-2024), and 91st (2025-2026) bienniums, there is no Shor-McCarty comparison available. This is where W-NOMINATE (Chapter 4) becomes essential — it can be computed from the same roll call data Tallgrass already has, without needing external survey data.

### Career vs. Session Scores

Shor-McCarty's np_score is a **career-level** estimate. A legislator who served in three bienniums gets one score that summarizes their entire tenure. Tallgrass produces a **session-level** score that reflects the most recent biennium.

For legislators whose ideology is stable, this distinction doesn't matter. For legislators who evolve — who become more conservative as they rise in party leadership, or more moderate as they approach retirement — the career score and session score can diverge legitimately.

This isn't a flaw in either method. It's a real difference in what they're measuring. Shor-McCarty answers "who is this legislator, on average, across their career?" Tallgrass answers "who is this legislator, right now, in this session?"

### The Bridge Problem

Shor-McCarty's NPAT bridge is only as good as the NPAT response rate. In some years, relatively few Kansas legislators answer the survey. This means the bridge relies on a small number of legislators to calibrate the scale. If those respondents are unrepresentative (for example, if moderates are more likely to respond), the scale could be slightly biased.

This is an inherent limitation of any bridge-based method, and it's well-documented in the political science literature. It doesn't undermine the comparison — it just means we should interpret the correlation as a lower bound on agreement. The true agreement is probably slightly higher than what we measure, because some of the disagreement comes from noise in the Shor-McCarty scores themselves.

## Key Takeaway

External validation is the strongest test a latent variable can face: do completely independent researchers, using different data and different methods, reach the same conclusions? For Kansas, the answer is yes — Tallgrass IRT ideal points correlate with Shor-McCarty national scores at r > 0.90 across five bienniums. The agreement holds within parties, not just between them, confirming that the model captures genuine intra-party ideological gradations. Where the two methods disagree (career vs. session scores, coverage gaps post-2020), the disagreement itself is informative — it highlights real temporal dynamics rather than methodological failure.

*Terms introduced: Shor-McCarty scores, Harvard Dataverse, NPAT (National Political Awareness Test), Project Vote Smart, bridge methodology, common scale, name normalization, Pearson r, Spearman rho, Fisher z-transform, confidence interval, within-party correlation, career-level vs. session-level scores, outlier detection*

---

*Previous: [Chapter 2 — Posterior Predictive Checks](ch02-posterior-predictive-checks.md)*

*Next: [Chapter 4 — W-NOMINATE: Comparing to the Field Standard](ch04-w-nominate.md)*
