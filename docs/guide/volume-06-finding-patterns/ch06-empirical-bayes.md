# Chapter 6: Empirical Bayes: Shrinkage Estimates of Party Loyalty

> *A legislator voted with their party on 11 out of 12 party votes. Their unity score is 91.7%. But should you trust a percentage based on 12 data points as much as one based on 200? Empirical Bayes says no — and it shows you exactly how much to adjust.*

---

## The Problem with Small Samples

Chapter 5 computed party unity scores for every legislator: the fraction of party votes where they voted with their party's majority. This is a clean, interpretable metric. But it has a problem that the raw number hides.

Consider two Kansas House Republicans:

- **Rep. Veteran:** Voted with party on 176 out of 195 party votes. Unity = 90.3%.
- **Rep. Freshman:** Appointed mid-session, voted with party on 11 out of 12 party votes. Unity = 91.7%.

By the raw numbers, the freshman looks *more* loyal than the veteran. But does anyone really believe that? The freshman's score is based on just 12 votes. If they'd happened to defect on one more, their score would drop to 83.3%. If the veteran had defected one more time, theirs would barely budge — from 90.3% to 89.7%.

The raw score treats 12 votes and 195 votes with equal confidence. That's misleading. The veteran's score is a reliable reading; the freshman's is a noisy estimate that could easily be 10 points off in either direction.

This is the same problem that Volume 4, Chapter 5 solved for IRT ideal points with hierarchical models: legislators with little data get noisy estimates that benefit from borrowing strength from their party. This chapter applies the same logic to party loyalty scores, using a technique called **empirical Bayes**.

## The Baseball Connection

In 1956, mathematician **Herbert Robbins** published a paper at the Third Berkeley Symposium proposing a radical idea: instead of specifying a prior distribution subjectively (as a traditional Bayesian would), you could **estimate the prior from the data itself**. He called this "empirical Bayes."

The most famous application is in baseball statistics. In April, a player who goes 10-for-25 has a batting average of .400 — which would be historically elite if sustained. But nobody expects it to last. Why? Because we know, from looking at *all* hitters, that the true batting averages cluster around .260 with a certain spread. That knowledge — the distribution of talent across the league — is the prior. A .400 April average, filtered through this prior, gets "regressed" to something like .320.

The logic is identical for legislative loyalty. We know, from looking at *all* Republican legislators in the Kansas House, that loyalty rates cluster around some party average with some spread. A 91.7% loyalty rate based on 12 votes, filtered through this knowledge, gets pulled toward the party average — not all the way (the 12 votes are real data), but enough to reflect the uncertainty.

## The Beta-Binomial Model

### Why This Model?

The mathematical framework that makes empirical Bayes work for loyalty scores is the **Beta-Binomial** model. Let's build it from the ground up.

Each legislator's loyalty data consists of two numbers:
- **y** = votes with party (e.g., 11)
- **n** = total party votes where present (e.g., 12)

The question: what is the legislator's *true* loyalty rate θ (theta)?

The raw estimate is y/n = 11/12 = 0.917. But we want an estimate that reflects our uncertainty — something that accounts for the smallness of the sample.

### The Binomial Likelihood

Given a true loyalty rate θ, the probability of observing y successes out of n trials follows the **Binomial distribution**:

```
P(y | θ, n) = C(n,y) · θ^y · (1−θ)^(n−y)
```

This is the same math behind coin flipping. If a coin has probability θ of landing heads, the probability of getting y heads in n flips is the binomial formula.

### The Beta Prior

We need a prior distribution on θ — our belief about where loyalty rates fall *before* looking at this specific legislator. The **Beta distribution** is the natural choice because:

1. It's defined on [0, 1], which is exactly the range of loyalty rates
2. It's flexible enough to represent many different prior beliefs
3. It's the **conjugate prior** for the Binomial — meaning the posterior is also a Beta distribution, giving us closed-form answers with no need for MCMC

The Beta distribution has two parameters, α (alpha) and β (beta), that control its shape:

| Alpha, Beta | Shape | What it represents |
|-------------|-------|-------------------|
| α = 1, β = 1 | Flat (uniform) | "I know nothing — any loyalty rate is equally plausible" |
| α = 10, β = 2 | Right-skewed peak near 0.8 | "Most legislators are fairly loyal" |
| α = 50, β = 5 | Tight peak near 0.9 | "Almost all legislators are very loyal" |

The mean of a Beta(α, β) distribution is α / (α + β), and the sum α + β controls how concentrated the distribution is — a larger sum means a tighter peak (more prior confidence).

### Estimating the Prior from the Data

Here's the empirical Bayes twist: instead of choosing α and β subjectively, we estimate them from the observed loyalty rates of *all* legislators in the same party and chamber.

The technique is **method of moments** (Karl Pearson, 1894): set the theoretical mean and variance of the Beta distribution equal to the observed mean and variance of the loyalty rates.

If the observed loyalty rates across all Republican House members have mean μ and variance v:

```
α + β = μ(1 − μ) / v − 1
α = μ · (α + β)
β = (1 − μ) · (α + β)
```

**Step-by-step example:**

Suppose the 85 Republican House members have a mean loyalty rate of μ = 0.88 and a variance of v = 0.004:

```
α + β = 0.88 × 0.12 / 0.004 − 1 = 0.1056 / 0.004 − 1 = 26.4 − 1 = 25.4
α = 0.88 × 25.4 = 22.4
β = 0.12 × 25.4 = 3.0
```

So the estimated prior is Beta(22.4, 3.0), which peaks near 0.88 with moderate concentration. This says: "Based on all Republicans in this chamber, loyalty rates tend to be around 88%, with some spread."

**Codebase:** `analysis/14_beta_binomial/beta_binomial.py` (`estimate_beta_params()`)

### The Posterior: Combining Prior and Data

Now we combine the prior with each legislator's individual data. The posterior distribution for legislator *i* with y_i votes-with-party out of n_i party votes is:

```
Posterior = Beta(α + y_i, β + (n_i − y_i))
```

That's it — no MCMC, no sampling, no convergence diagnostics. The Beta-Binomial conjugacy gives us the exact posterior in closed form.

**For Rep. Veteran** (y = 176, n = 195):

```
Posterior = Beta(22.4 + 176, 3.0 + 19) = Beta(198.4, 22.0)
Posterior mean = 198.4 / (198.4 + 22.0) = 0.900
```

Almost identical to the raw rate of 0.903. With 195 votes, the data overwhelms the prior.

**For Rep. Freshman** (y = 11, n = 12):

```
Posterior = Beta(22.4 + 11, 3.0 + 1) = Beta(33.4, 4.0)
Posterior mean = 33.4 / (33.4 + 4.0) = 0.893
```

The raw rate was 0.917 — now it's been pulled down to 0.893, toward the party average of 0.88. With only 12 votes, the prior matters more.

**Codebase:** `analysis/14_beta_binomial/beta_binomial.py` (`compute_bayesian_loyalty()`)

## Shrinkage: The Invisible Hand

The posterior mean is a weighted average of the raw rate and the prior mean:

```
Posterior mean = (1 − shrinkage) × (raw rate) + shrinkage × (prior mean)
```

Where:

```
shrinkage = (α + β) / (α + β + n)
```

**For Rep. Veteran:** shrinkage = 25.4 / (25.4 + 195) = 0.115 — the prior gets 11.5% of the weight. Their estimate barely moves.

**For Rep. Freshman:** shrinkage = 25.4 / (25.4 + 12) = 0.679 — the prior gets 67.9% of the weight. Their estimate moves substantially.

This is the "invisible hand" of empirical Bayes: legislators with fewer votes are pulled harder toward the party average. The less data you have, the more you rely on the group.

| Scenario | n (party votes) | Shrinkage | Effect |
|----------|----------------|-----------|--------|
| Veteran (195 votes) | 195 | 0.12 | Barely moves — data dominant |
| Regular (80 votes) | 80 | 0.24 | Modest pull — mostly data |
| Mid-session joiner (30 votes) | 30 | 0.46 | Substantial pull — prior matters |
| Appointed replacement (12 votes) | 12 | 0.68 | Heavy pull — prior dominant |

### The Shrinkage Arrow Plot

The signature visualization of this chapter is the **shrinkage arrow plot**. Each legislator gets a dot at their raw loyalty rate and an arrow pointing to their posterior mean:

- **Y-axis:** Number of party votes (sample size)
- **X-axis:** Loyalty rate
- **Faded dot:** Raw rate (before shrinkage)
- **Bright dot:** Posterior mean (after shrinkage)
- **Arrow:** The distance and direction of the adjustment

The visual pattern is unmistakable. Legislators near the top of the chart (many votes) have tiny arrows — their estimates barely change. Legislators near the bottom (few votes) have long arrows, all pointing toward the party average line. The visual tells the whole story at a glance: more data, less adjustment.

**Codebase:** `analysis/14_beta_binomial/beta_binomial.py` (`plot_shrinkage_arrows()`)

## Credible Intervals: Honest Uncertainty

Unlike the raw unity score (a single number with no error bars), the Beta posterior provides a full probability distribution. From it, Tallgrass extracts a **95% credible interval** — the range that contains the true loyalty rate with 95% probability.

```
CI = [Beta.ppf(0.025, α_post, β_post), Beta.ppf(0.975, α_post, β_post)]
```

**For Rep. Veteran:** CI = [0.855, 0.937] — tight, 8 percentage points wide. We're confident.

**For Rep. Freshman:** CI = [0.771, 0.966] — wide, 19.5 percentage points. We're uncertain.

The forest plot of credible intervals, sorted by posterior mean, shows at a glance which legislators have precisely estimated loyalty and which are still ambiguous. Wide intervals don't mean the legislator is disloyal — they mean we don't have enough data to say.

**Codebase:** `analysis/14_beta_binomial/beta_binomial.py` (`plot_credible_intervals()`, `CI_LEVEL = 0.95`)

## Tarone's Overdispersion Test

Before fitting the Beta-Binomial, Tallgrass runs **Tarone's score test** (Tarone, 1979) to check whether the Beta-Binomial is actually needed. The test asks: is there more variation in loyalty rates across legislators than the Binomial alone would predict?

If all legislators had the same true loyalty rate θ, the variation in observed rates would be explained entirely by sampling noise (some had 100 party votes, others had 30). Tarone's test checks whether the observed variation exceeds what sampling noise alone would produce.

A significant Tarone test (p < 0.05) means there's genuine overdispersion — legislators truly differ in their loyalty rates, and the Beta-Binomial is justified. In Kansas, the test is always significant: legislators genuinely differ, and the empirical Bayes adjustment is warranted.

**Codebase:** `analysis/14_beta_binomial/beta_binomial.py` (`tarone_test()`)

## Four Groups, Four Priors

The prior distribution is estimated separately for each party-chamber combination:

| Group | Typical size | Typical prior mean | Typical prior concentration (α + β) |
|-------|-------------|-------------------|--------------------------------------|
| House Republicans | ~85 | 0.88 | 25–30 |
| House Democrats | ~40 | 0.91 | 20–25 |
| Senate Republicans | ~30 | 0.85 | 15–20 |
| Senate Democrats | ~10 | 0.93 | 10–15 |

The Democratic caucuses tend to show higher loyalty (smaller caucuses in the minority vote together more tightly — they have little room for defection). The prior concentration is lower for smaller groups because there's less data to estimate it from, which appropriately leads to weaker priors and less shrinkage.

## Connection to Hierarchical IRT

If this sounds familiar, it should. Volume 4, Chapter 5 described hierarchical IRT models that apply partial pooling to ideal points: each legislator starts at their party's mean and adjusts based on individual data.

The Beta-Binomial model is a simpler version of the same idea:

| Feature | Hierarchical IRT | Beta-Binomial |
|---------|-----------------|---------------|
| **What it estimates** | Ideology (continuous, latent) | Loyalty rate (proportion, observed) |
| **Prior** | Normal distribution on ideal points | Beta distribution on loyalty rates |
| **Estimation** | MCMC (nutpie, thousands of samples) | Closed-form (no sampling needed) |
| **Shrinkage** | Toward party mean ideology | Toward party mean loyalty |
| **Computation time** | Minutes to hours | Under a second |

The Beta-Binomial gives you the *shrinkage benefits* of hierarchical Bayes without the *computational cost* of MCMC. It's a complement to IRT, not a replacement: IRT measures ideology (a latent trait), while Beta-Binomial measures loyalty (an observed proportion). A legislator can be ideologically extreme but highly loyal, or ideologically moderate but frequently defecting.

## Three Exemplar Legislators

To make the posteriors concrete, Tallgrass plots the full Beta posterior density for three legislators chosen to illustrate different extremes:

### The Most-Shrunk Legislator

A recently appointed representative with only 8 party votes, raw loyalty of 87.5% (7/8). The posterior is a wide, gentle hill centered near the party average. The message: we know almost nothing about this person's true loyalty — they could be anywhere from 70% to 98%.

### The Least-Shrunk Legislator

A 10-term veteran with 195 party votes and 90.3% raw loyalty. The posterior is a tall, narrow spike at 0.90. The data speaks for itself — the prior barely matters.

### The Lowest-Loyalty Legislator

The most maverick member of the caucus, with 68% raw loyalty on 150 votes. The posterior is moderately narrow (lots of data) but centered well below the party average. Shrinkage pulls them up slightly (from 68% to maybe 70%), but the data clearly identifies them as an outlier.

**Codebase:** `analysis/14_beta_binomial/beta_binomial.py` (`plot_posterior_distributions()`)

## What Empirical Bayes Tells Us About Kansas

### Shrinkage Is Small for Most Legislators

Most Kansas legislators serve full sessions and cast 100+ party votes. For them, shrinkage is modest — the posterior mean moves 1-2 percentage points from the raw rate. The primary value isn't the point estimate (which barely changes) but the **credible interval** (which quantifies how much we should trust the score).

### The Real Value Is at the Margins

Empirical Bayes matters most for the legislators who need it most: appointees, mid-session replacements, and anyone who missed a substantial portion of the session. For these legislators, the raw unity score is unreliable noise. The posterior mean provides a better estimate, and the credible interval honestly communicates the uncertainty.

### Party Differences in Loyalty Dispersion

The empirical Bayes priors reveal something about party structure. The prior concentration (α + β) is higher for the House Republican caucus (~25) than for the Senate Democrats (~12). Higher concentration means less within-party variation in loyalty rates — the House Republicans are more homogeneous in their party discipline than the small Senate Democratic caucus.

This aligns with the IRT finding (Volume 4): larger caucuses tend to have more within-party variation, but the variation is spread across more members, producing a tighter *distribution* of loyalty rates even if the *range* of ideology is wider.

---

## Key Takeaway

Empirical Bayes applies the logic of hierarchical models to party loyalty scores: legislators with more data keep their raw estimates nearly unchanged, while legislators with less data are pulled toward the party average. The Beta-Binomial conjugate model provides this adjustment in closed form — no MCMC needed. The key outputs are credible intervals (how uncertain is each loyalty estimate?) and shrinkage arrows (how much did the adjustment move each legislator?). The technique doesn't change the story for veteran legislators, but it prevents us from over-interpreting the noisy scores of newcomers and absentees.

---

*Terms introduced: empirical Bayes, Beta-Binomial model, conjugate prior, method of moments, prior concentration (α + β), posterior mean, shrinkage factor, credible interval, Tarone's overdispersion test, shrinkage arrow plot, per-party-per-chamber prior*

*Previous: [Classical Indices: Rice, Party Unity, and the Maverick Score](ch05-classical-indices.md)*

*Next: [Volume 7 — Prediction and Text](../volume-07-prediction-and-text/)*
