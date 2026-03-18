# Chapter 1: What Does Validation Mean?

> *You can't look up a legislator's "true" ideology score the way you can look up the temperature outside. So how do you know if your model got the right answer?*

---

## The Thermometer Problem

When you check a thermometer, you have a clear standard. If it reads 72°F and you step outside and it feels like 72°F, the thermometer is working. If it reads 72°F and you see snow falling, something is wrong. The true temperature exists independently of the thermometer. The thermometer is just measuring it.

Ideology scores aren't like that.

There is no "true" ideology score for a legislator. You can't open someone's head and read "conservative = +1.47." Ideology is a **latent variable** — a quantity that we believe exists but can never observe directly. What we observe is *behavior*: hundreds of Yea and Nay votes. The ideology score is our best summary of the pattern behind that behavior.

This creates a philosophical puzzle. If we can't know the true value, how do we know if our estimate is any good?

The answer is that we test the *consequences* of the estimate, not the estimate itself. A good ideology model should do several things:

1. **Reproduce what happened.** If we plug the estimated ideal points back into the model, the predicted votes should look like the votes that actually occurred.
2. **Agree with other methods.** If a completely different statistical technique produces similar rankings, that's reassuring — two different roads led to the same destination.
3. **Make substantive sense.** Democrats should generally score differently than Republicans. A legislator known as a maverick should land in an unusual spot. The most extreme scores should belong to legislators whom observers would call the most ideological.
4. **Be internally consistent.** The model shouldn't contradict itself — claiming high confidence in an estimate that is actually unstable.

None of these tests prove the model is "correct" in the way that a thermometer can be proven correct. But together, they build a case — a **preponderance of evidence** — that the model is capturing something real.

## An Analogy: Validating a Map

Think of an ideology model as a **map**. The terrain — the actual ideological landscape of the Kansas Legislature — is the thing that exists. The map is our representation of it. Maps are always simplifications: a road map ignores elevation, a topographic map ignores building names. The question isn't whether the map is a perfect replica of the terrain, but whether it's **useful for what you need it for**.

How do you validate a map? Four ways:

1. **Internal consistency.** Do the distances on the map make sense relative to each other? If city A is shown closer to city B than to city C, is that actually true? In our context: if the model says Legislator A is more similar to B than to C in ideology, do their voting records bear that out?

2. **Navigation test.** Can you use the map to get somewhere? If you follow its roads, do you arrive at real destinations? In our context: can the model predict how a legislator would vote on a bill they haven't seen? If we hold out some votes and ask the model to fill them in, does it get them right?

3. **Comparison with other maps.** Does this map agree with other independently made maps of the same area? If your map shows Main Street running east-west but Google Maps shows it running north-south, someone is wrong. In our context: do other researchers' ideology estimates agree with ours?

4. **Ground truth checks.** For a few key landmarks, can you verify the map is right? You know where your own house is — is it in the right spot? In our context: do a few well-known legislators land where political observers would expect?

Tallgrass implements all four kinds of tests. This volume explains each one in detail.

## A Taxonomy of Validation

Before diving into specifics, it helps to distinguish two categories of validation that we use throughout this volume:

### Internal Validation: Testing Against Yourself

Internal validation asks: **does the model describe the data it was fit to?** This is the minimum bar. If a model can't even reproduce the patterns it was trained on, it certainly can't be trusted for anything else.

The primary tool here is the **posterior predictive check** (PPC), which we'll cover in Chapter 2. The idea is simple: ask the model to generate fake votes using its estimated parameters, then compare the fake votes to the real ones. If they match, the model is at least internally consistent. If they don't, something is wrong with the model's assumptions.

Internal validation also includes **model comparison** — asking which of several competing models best describes the data. The pipeline fits four different IRT models (1D flat, 2D flat, hierarchical 1D, hierarchical 2D), and we want to know which one is the best representation for each legislative session. Leave-one-out cross-validation (LOO-CV) provides a principled way to compare them.

Internal validation is necessary but not sufficient. A model can perfectly reproduce its training data and still be wrong — the phenomenon machine learning engineers call "overfitting." If you build a model with enough parameters, it can memorize every vote without learning anything about ideology at all. That's why we also need external validation.

### External Validation: Testing Against the World

External validation asks: **does the model agree with independent evidence?**

The most powerful form of external validation is comparison with results from other researchers using different methods and sometimes different data. If our Bayesian IRT ideal points correlate strongly with:

- **Shor-McCarty scores** (a political science gold standard that bridges state legislators into a common national scale using survey data we never touch)
- **W-NOMINATE scores** (the dominant method in the field, using maximum likelihood instead of Bayesian estimation and a different probability model)
- **Optimal Classification** (a completely nonparametric approach that makes no distributional assumptions at all)

...then we can be confident that our estimates are capturing something real about the ideological structure of the legislature, not just an artifact of our particular modeling choices.

Each of these comparisons has a different strength:

| External Benchmark | What Makes It Independent | What Agreement Means |
|-------------------|--------------------------|---------------------|
| Shor-McCarty | Different data (survey bridge), different method, national scale | Our Kansas-only scores align with national ideology measures |
| W-NOMINATE | Different probability model (Gaussian kernel vs. logistic), MLE not Bayesian | Our results aren't sensitive to the choice of estimation technique |
| Optimal Classification | No probability model at all — purely geometric | Even without distributional assumptions, the same structure emerges |

## What "Good" Looks Like

When we compute a correlation between our scores and an external benchmark, what counts as strong? Here are the rough standards in the field:

| Correlation (|r|) | Interpretation | What It Signals |
|-------------------|---------------|-----------------|
| > 0.95 | Excellent | Methods are essentially interchangeable for this legislature |
| 0.90 – 0.95 | Strong | Methods agree on the structure; minor differences in edge cases |
| 0.85 – 0.90 | Good | Methods agree on the broad picture; some individual legislators differ |
| 0.70 – 0.85 | Concern | Methods disagree enough to investigate — possibly different dimensions captured |
| < 0.70 | Problem | Something is structurally wrong — different data, axis confusion, or a bug |

For Tallgrass, across all sessions where Shor-McCarty data overlaps (the 84th through 88th bienniums, 2011–2020), the typical correlation exceeds 0.90. This puts our estimates firmly in the "strong agreement" range — different enough to be scientifically interesting (they're not identical), but similar enough to confirm that both methods are measuring the same underlying construct.

## The Validation Pipeline

In the Tallgrass pipeline, validation spans several phases:

| Phase | Name | Type | What It Tests |
|-------|------|------|--------------|
| 08 | Posterior Predictive Checks | Internal | Can the model reproduce its own data? (Chapter 2) |
| 17 | External Validation | External | Do our scores match Shor-McCarty? (Chapter 3) |
| 16 | W-NOMINATE | External | Do our scores match the field standard? (Chapter 4) |
| Canonical routing | Internal | Which model is trustworthy enough to publish? (Chapter 5) |

Phase 08 is a standalone validation step — it doesn't feed into any downstream analysis. Its sole purpose is diagnostic: does the model work? Phases 16 and 17 are similarly validation-only. The canonical routing system (not a numbered phase, but a decision module that runs after Phase 07b) is the one place where validation directly affects the pipeline's output.

The rest of this volume walks through each of these in turn.

## Key Takeaway

Ideology scores are latent variables — you can never observe the "true" value. Validation therefore isn't about checking against ground truth but about building a preponderance of evidence: does the model reproduce what happened? Does it agree with independent methods? Does it make substantive sense? Tallgrass implements four kinds of validation — internal consistency, navigation (prediction), comparison with other methods, and ground truth checks — across multiple pipeline phases. No single test proves the model is correct, but together they establish quiet, cumulative confidence.

*Terms introduced: latent variable, internal validation, external validation, posterior predictive check (PPC), model comparison, holdout testing, Shor-McCarty scores, W-NOMINATE, Optimal Classification, canonical routing*

---

*Next: [Chapter 2 — Posterior Predictive Checks](ch02-posterior-predictive-checks.md)*
