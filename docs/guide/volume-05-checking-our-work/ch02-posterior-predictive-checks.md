# Chapter 2: Posterior Predictive Checks: Can the Model Predict Its Own Data?

> *Imagine a chef who claims to have the perfect recipe for chocolate cake. The simplest test: bake the cake and taste it. Posterior predictive checks are the statistical equivalent — we ask the model to bake us some data, and we taste it.*

---

## The Core Idea

A posterior predictive check (PPC) works like this:

1. **Fit the model.** Estimate all the parameters (ideal points, bill difficulties, bill discriminations) from the observed votes.
2. **Simulate.** Using those estimated parameters, generate a brand-new set of votes — a "fake legislature" that should, if the model is correct, look like the real one.
3. **Compare.** Check whether the simulated votes match the observed votes on key summary statistics.

If the model is well-calibrated, the simulated data should be essentially indistinguishable from the real data on the metrics we care about. If it's not — if the fake legislature looks systematically different from the real one — then the model is missing something.

### The Chef Analogy, Extended

Think of it this way. The model is like a recipe that claims to describe how legislators vote. The ingredients are the estimated parameters: each legislator's ideal point, each bill's difficulty and discrimination. The recipe is the IRT equation from Volume 4:

```
P(Yea) = logistic(β · ξ − α)
```

A posterior predictive check says: "Okay, recipe, prove it. Use your ingredients (the estimated parameters) and your instructions (the IRT equation) to bake a new batch of votes. If your batch tastes nothing like what the legislature actually produced, your recipe is wrong."

But here's the crucial twist: we don't generate just *one* batch. We generate **hundreds** — typically 500 in Tallgrass. Why? Because the model is Bayesian, which means each parameter isn't a single number but a **distribution** of plausible values. Each time we simulate, we randomly pick one set of parameter values from their posterior distributions, generating a slightly different fake legislature each time.

This gives us not just a single comparison but a *distribution* of possible outcomes. The observed data should fall comfortably within that distribution. If it's an outlier — if the real legislature looks nothing like any of the 500 fake legislatures — something is wrong.

## What Tallgrass Checks

The PPC phase (`analysis/08_ppc/ppc.py`) runs four types of checks on each fitted model.

### Check 1: The Yea Rate (Basic Calibration)

The simplest possible test: **does the model get the overall Yea fraction right?**

In the Kansas Legislature, about 82% of all recorded votes are Yea votes. This is common in state legislatures — most bills that reach a floor vote are bipartisan or uncontroversial. The hard-fought, party-line votes that dominate the news are actually a minority of the total.

If the model produces simulated legislatures where the Yea rate is consistently 75% or 90%, it's miscalibrated at the most basic level. Something about how it models bill difficulty (α) is systematically off.

**What the numbers look like:**

| Statistic | Typical Value |
|-----------|--------------|
| Observed Yea rate | 0.82 |
| Replicated Yea rate (mean across 500 draws) | 0.82 ± 0.004 |
| Bayesian p-value | 0.45 – 0.55 |

The Bayesian p-value here measures what fraction of the 500 simulated legislatures had a Yea rate at least as extreme as the observed one. A value near 0.5 means the observed data is right in the middle of the simulated distribution — exactly what we want. A value near 0 or 1 would signal a systematic misfit.

Think of it like this: if you claim a coin is fair, flip it 100 times, and get 80 heads, the Bayesian p-value asks "how many of my 500 simulated fair coins also got 80 or more heads?" If the answer is "almost none," the coin probably isn't fair.

### Check 2: Classification Accuracy (Can It Predict Individual Votes?)

The Yea rate tells us about the *average*. Classification accuracy tells us about the *individual votes*.

For each simulated legislature, the model predicts the most likely outcome for every vote (Yea if the predicted probability > 0.5, Nay otherwise) and counts how many it gets right. Across 500 simulations, we get a distribution of accuracy scores.

**Why accuracy alone is tricky:**

Here's the catch: with an 82% Yea base rate, a "model" that simply predicts Yea for every single vote would be 82% accurate. That's not skill — that's just following the crowd.

This is why Tallgrass reports two additional metrics that correct for the base rate.

### Check 3: GMP — Geometric Mean Probability

GMP stands for Geometric Mean Probability. It's a fit metric that answers: **how confident was the model, and was that confidence justified?**

Here's the intuition. Imagine two weather forecasters:

- **Forecaster A** says there's a 90% chance of rain every day. On the 80% of days when it does rain, she looks great. On the 20% when it doesn't, she was very wrong. But she never said "I'm not sure."
- **Forecaster B** says there's a 75% chance of rain on days that look rainy and a 30% chance on days that look dry. She's less boldly confident, but she's more nuanced. When she says 30%, it really does only rain 30% of the time.

Accuracy would rate Forecaster A highly (she gets the majority class right). GMP would rate Forecaster B more highly because GMP **punishes confident wrong predictions harshly**.

The formula:

```
GMP = exp(mean(log(p_correct)))
```

where `p_correct` is the model's predicted probability for the outcome that *actually occurred*. The logarithm means that a single vote where the model said "99% Yea" but the legislator voted Nay has a devastating effect on the score — much more than a vote where the model said "55% Yea" and got it right boosts the score.

In code (`analysis/08_ppc/ppc_data.py`), this computation looks like:

```python
p_correct = np.where(y_obs == 1, p, 1.0 - p)
p_correct = np.clip(p_correct, 1e-15, 1.0)
gmp = float(np.exp(np.mean(np.log(p_correct))))
```

The `np.clip` prevents taking the log of zero — a safeguard for votes where the model is essentially certain of one outcome but the other occurs.

**What the numbers mean:**

| GMP Value | Interpretation |
|-----------|---------------|
| > 0.85 | Excellent — the model is confident and correct |
| 0.80 – 0.85 | Good — minor calibration issues |
| 0.75 – 0.80 | Adequate — some votes poorly predicted |
| < 0.75 | Concern — the model is regularly surprised |

### Check 4: APRE — The "Better Than Guessing" Test

APRE (Aggregate Proportional Reduction in Error) asks the toughest question: **how much does the model improve over the stupidest possible baseline?**

The stupidest baseline for the Kansas Legislature is: "predict Yea for every vote." That baseline is 82% accurate (because 82% of votes are Yea). Any useful model should do better. APRE measures *how much* better.

```
APRE = (model_accuracy - baseline_accuracy) / (1 - baseline_accuracy)
```

Let's walk through the arithmetic:

- Baseline accuracy: **0.82** (always predict Yea)
- Model accuracy: **0.91** (a typical IRT model)
- APRE: **(0.91 - 0.82) / (1 - 0.82) = 0.09 / 0.18 = 0.50**

An APRE of 0.50 means the model eliminates **50% of the errors** that the dumb baseline makes. Of the votes the baseline gets wrong (all the Nay votes), the model correctly identifies half of them.

**Why APRE matters more than accuracy:**

| Model | Accuracy | APRE | Interpretation |
|-------|----------|------|---------------|
| Always guess Yea | 82% | 0% | No skill at all |
| Random + bias | 85% | 17% | Barely better than guessing |
| 1D IRT | 91% | 50% | Captures half the ideology structure |
| 2D IRT | 92% | 56% | The second dimension adds a bit more |

APRE strips away the flattery of the base rate and shows the model's actual contribution.

## Item and Person Fit: Going Deeper

The four global metrics tell us how the model does overall. But a model can perform well on average while badly mishandling specific bills or legislators. Item and person fit checks catch these cases.

### Item Fit: Which Bills Does the Model Get Wrong?

For every bill in the dataset, Tallgrass computes the **observed Yea rate** (what fraction of legislators actually voted Yea?) and the **replicated Yea rate** (what fraction voted Yea in the 500 simulated legislatures?).

If the observed rate falls outside the 95% replicated interval — meaning the bill's actual outcome is more extreme than 97.5% of the simulations — the bill is flagged as **misfitting**.

Think of it like quality control on a factory assembly line. Each bill is a product rolling off the line. The model's prediction is the specification. If a product is out of spec — the real outcome is too far from the predicted one — it gets flagged for inspection.

What makes a bill misfit? Common causes:

- **Logrolling or vote trading:** A legislator votes against their ideology to secure support for a different bill. The model, which only knows about ideology, can't predict this.
- **Constituency pressure:** A legislator breaks from their ideology because the bill directly affects their district. A farm-state conservative might break party ranks on an agriculture subsidy.
- **Genuine bipartisanship on an issue the model classifies as partisan:** The model assumes bill discrimination is constant, but sometimes a bill that *looks* ideological (based on its parameters) actually generates bipartisan support for specific reasons.

### Person Fit: Which Legislators Does the Model Get Wrong?

The same logic, but for legislators instead of bills. For each legislator, Tallgrass computes their **observed total Yea count** and their **replicated total Yea count** across 500 simulations.

A misfitting legislator is one whose total Yea count is consistently too high or too low compared to what the model predicts. This usually indicates a legislator whose voting pattern doesn't conform to the one-dimensional ideological model — either because they have genuinely multidimensional preferences or because external factors (leadership pressure, term-limit calculations, impending retirement) are shaping their votes.

In the code (`analysis/08_ppc/ppc_data.py`), misfitting items and persons are identified with a simple interval check:

```python
lo = np.percentile(replicated_rates, 2.5, axis=0)
hi = np.percentile(replicated_rates, 97.5, axis=0)
misfitting = np.where((observed_rates < lo) | (observed_rates > hi))[0]
```

## Yen's Q3: Testing Whether One Dimension Is Enough

The checks above tell us whether the model fits the data well. Yen's Q3 tells us something subtler: **whether the model is capturing all the structure, or whether there's a hidden pattern it's missing.**

### The Idea

In a well-fitting 1D IRT model, once you account for each legislator's ideology and each bill's difficulty and discrimination, the votes should be **independent** — knowing how a legislator voted on Bill A should tell you nothing extra about how they voted on Bill B, beyond what their ideology score already predicts.

If two bills are still correlated *after* controlling for ideology, that's called **local dependence** — there's a relationship between those items that the single ideology dimension doesn't explain. Maybe both bills are about agriculture, and there's a farm-versus-urban dimension that ideology doesn't capture. Maybe both are gun bills, and Second Amendment attitudes cut across the party line in ways that a single left-right axis can't represent.

### The Analogy: Residuals in a Science Experiment

Imagine you're studying the relationship between studying hours and test scores. You fit a straight line (linear regression) and look at the residuals — the gaps between the line and the actual data points.

If the residuals are random noise, your line captured the relationship. But if you notice that the residuals for Monday exams are all positive and the residuals for Friday exams are all negative, there's a **pattern in the noise** — day of the week matters, and your model missed it.

Q3 does the same thing for IRT. It computes the **residual** for each vote (actual outcome minus the model's predicted probability), then checks whether the residuals are correlated across pairs of bills. If they are, the model is missing something.

### How It's Computed

For each posterior draw:

1. **Compute the predicted probability** for every vote using the model's parameters.
2. **Compute the residual:** actual vote (0 or 1) minus the predicted probability.
3. **Arrange residuals into a matrix** (legislators × bills).
4. **Correlate the columns:** for each pair of bills, compute the Pearson correlation of their residual columns across legislators.

Average across draws to get a stable Q3 matrix. The convention (from Yen, 1993): **|Q3| > 0.2 indicates local dependence.**

In Tallgrass (`analysis/08_ppc/ppc_data.py`):

```python
residual = y_obs - p  # actual vote minus predicted probability
resid_matrix[leg_idx, vote_idx] = residual

# Correlate columns (bills) across rows (legislators)
r = np.corrcoef(col1[valid], col2[valid])[0, 1]
```

### What Q3 Violations Tell You

If the 1D model shows a high violation rate (say, 15% of bill pairs have |Q3| > 0.2) but the 2D model drops it to 3%, that's empirical evidence that the second dimension is doing real work — it's capturing a genuine pattern that the 1D model missed.

This is one of the ways Tallgrass decides whether a 2D model is justified for a given session: not by assumption, but by evidence. If the first dimension already captures everything, the 2D model's Q3 numbers won't look any better than the 1D model's.

## LOO-CV: Which Model Fits Best?

The PPC battery tells us whether *each individual model* fits the data. But when we have multiple models (1D flat, 2D flat, hierarchical 1D, hierarchical 2D), we also want to know which one is *best*.

### The Holdout Principle

The gold standard for model comparison is **holdout testing**: fit the model on most of the data, then see how well it predicts the data you held out. The model that predicts the held-out data best is the winner.

The problem: with MCMC models that take 10-20 minutes to fit, refitting once for each of 50,000+ observations is computationally absurd.

### PSIS-LOO: The Shortcut

PSIS-LOO (Pareto Smoothed Importance Sampling Leave-One-Out) is a mathematical trick that *approximates* what would happen if you left out each observation and refit. It was developed by Aki Vehtari, Andrew Gelman, and Jonah Gabry in 2017, and it has become the standard model comparison tool in Bayesian statistics.

The full math is beyond this guide, but the intuition is accessible. When you leave out one observation and refit the model, the new posterior is similar to the old one — most observations contribute a tiny amount to the overall fit. PSIS-LOO exploits this by *reweighting* the existing posterior samples rather than resampling from scratch. It's like asking "how would the model change if this one vote were missing?" without actually removing it.

The result is an estimated **ELPD** (Expected Log Pointwise Predictive Density) for each model — a single number that captures out-of-sample predictive accuracy. Higher ELPD means better out-of-sample prediction.

### Pareto k: When the Shortcut Fails

PSIS-LOO works because most observations have small influence on the posterior. But some observations are **highly influential** — removing them would change the model substantially. For these, the reweighting trick fails.

The **Pareto k diagnostic** flags these cases:

| Pareto k | Interpretation |
|----------|---------------|
| k < 0.5 | Safe — the LOO approximation is reliable |
| 0.5 ≤ k < 0.7 | Marginal — the estimate is still usable but less precise |
| 0.7 ≤ k < 1.0 | Problematic — the estimate for this observation is unreliable |
| k ≥ 1.0 | Failed — the LOO approximation has broken down |

Think of Pareto k as a warning light on your car's dashboard. A few yellow lights (k between 0.5 and 0.7) are normal — some observations are naturally more influential than others. A cluster of red lights (k > 1.0) means you shouldn't trust the LOO comparison for this model.

In practice, the 1D IRT model for the Kansas Legislature typically has very few high-k observations. The 2D model, with more parameters and weaker identification on the second dimension, tends to have more. This is expected — more complex models are more sensitive to individual observations.

### Model Comparison in Practice

ArviZ's `compare()` function ranks the models by ELPD and assigns **stacking weights** — the proportion of a hypothetical mixture model that should come from each component. A model with a stacking weight of 0.95 dominates the comparison; a model with weight 0.01 adds almost nothing.

Tallgrass logs these comparisons but does *not* use them to make routing decisions (that's the quality gate system in Chapter 5). LOO-CV is a diagnostic for humans to review, not an automated switch.

## Putting It All Together

A complete PPC report for one chamber-session looks like this:

| Metric | 1D Flat | Hierarchical 1D | 2D Flat | Hierarchical 2D |
|--------|---------|-----------------|---------|-----------------|
| Yea rate p-value | 0.48 | 0.50 | 0.47 | 0.49 |
| Accuracy | 91.2% | 91.5% | 92.1% | 92.4% |
| GMP | 0.838 | 0.841 | 0.847 | 0.850 |
| APRE | 0.51 | 0.53 | 0.56 | 0.58 |
| Misfitting items | 12/487 | 10/487 | 7/487 | 6/487 |
| Misfitting persons | 3/125 | 2/125 | 2/125 | 1/125 |
| Q3 violation rate | 8.2% | 7.8% | 2.1% | 1.9% |

Reading this table:

- **Yea rate p-values** are all near 0.5 — every model is well-calibrated at the aggregate level.
- **Accuracy** improves modestly from 1D to 2D, and from flat to hierarchical. The gains are small because most votes are easy to predict — the hard ones are the close votes, and those are a minority.
- **GMP** shows slightly larger improvements, because GMP penalizes confident wrong predictions — exactly the kind of errors that a more nuanced model fixes.
- **APRE** shows the most dramatic difference, because it strips away the base rate. The 2D hierarchical model eliminates 58% of the errors the dumb baseline makes, versus 51% for the simplest model.
- **Misfitting items** decrease as models grow more complex — expected, since more flexible models can capture more patterns.
- **Q3 violation rate** drops dramatically from 1D to 2D, confirming that the second dimension captures real structure (likely the moderate-versus-establishment split in the Republican supermajority).

These numbers aren't dramatic. Nobody looks at a table like this and gasps. But that's the point: validation is about building quiet confidence, not delivering drama. When every metric tells the same story — each model works, the more complex ones work a bit better, and nothing is badly broken — that's what good science looks like.

## Key Takeaway

Posterior predictive checks ask the simplest possible question: can the model explain the data it was fit to? Tallgrass generates 500 fake legislatures from the fitted model and compares them to the real one on four global metrics (Yea rate, accuracy, GMP, APRE), item-level fit, person-level fit, and local dependence (Q3). LOO-CV then compares competing models on their ability to predict held-out votes. None of these checks are dramatic — a well-fitting model quietly passes them all. The value is in the rare cases where something fails, which points directly to what the model is missing.

*Terms introduced: posterior predictive check (PPC), Bayesian p-value, classification accuracy, GMP (Geometric Mean Probability), APRE (Aggregate Proportional Reduction in Error), base rate, item fit, person fit, misfitting item, misfitting person, Yen's Q3, local dependence, LOO-CV (Leave-One-Out Cross-Validation), PSIS (Pareto Smoothed Importance Sampling), ELPD (Expected Log Pointwise Predictive Density), Pareto k diagnostic, stacking weights*

---

*Previous: [Chapter 1 — What Does Validation Mean?](ch01-what-does-validation-mean.md)*

*Next: [Chapter 3 — The Gold Standard: Shor-McCarty External Validation](ch03-shor-mccarty.md)*
