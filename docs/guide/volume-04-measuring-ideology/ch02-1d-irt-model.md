# Chapter 2: The 1D IRT Model — Step by Step

> *"The probability that a legislator votes Yea depends on how close their ideology is to what the bill requires." That's the whole model. Everything else is details.*

---

## The Core Equation

At the heart of every IRT model is a single equation that predicts how likely a legislator is to vote Yea on a particular bill. Let's build it in three layers.

**Plain English:**

> "Take the legislator's ideology score, multiply it by how much this bill cares about ideology, then subtract how hard the bill is to pass. Convert the result to a probability."

**Mathematical notation:**

```
P(Yea) = logistic(β · ξ − α)
```

where:
- ξ (xi) = the legislator's ideal point (their position on the ideology spectrum)
- β (beta) = the bill's discrimination (how sharply it separates liberals from conservatives)
- α (alpha) = the bill's difficulty (how hard it is to pass)
- logistic(x) = 1 / (1 + e^(−x)) — the function that converts any number into a probability between 0 and 1

**Worked example:**

Suppose Senator Miller is a moderate conservative with an ideal point of ξ = +0.8. A tax reform bill comes up for a vote with discrimination β = 1.2 (fairly partisan) and difficulty α = −0.3 (relatively easy to pass). What's the probability Senator Miller votes Yea?

1. Start with the ideal point: **ξ = +0.8**
2. Multiply by discrimination: **1.2 × 0.8 = 0.96**
3. Subtract difficulty: **0.96 − (−0.3) = 0.96 + 0.3 = 1.26**
4. Apply the logistic function: **1 / (1 + e^(−1.26)) = 1 / (1 + 0.284) = 0.78**
5. **Result: 78% probability of a Yea vote**

That number — 78% — is the model's prediction. Not a certainty, but a probability. Senator Miller will probably vote Yea on this bill, but there's a 22% chance they won't. Maybe they have a constituency concern. Maybe they're trading votes on a different bill. The model doesn't claim to know everything — just the base rate implied by ideology.

## Understanding the Logistic Function

The logistic function is the engine that makes IRT work. It's worth understanding intuitively before we go further.

### The S-Curve

Plot the logistic function and you get an **S-shaped curve** (technically called a sigmoid):

```
P(Yea)
  1.0 ┤                          ─────────
      │                       ╱
  0.8 ┤                     ╱
      │                   ╱
  0.6 ┤                 ╱
      │               ╱
  0.4 ┤             ╱
      │           ╱
  0.2 ┤         ╱
      │  ──────
  0.0 ┤
      └─────────────────────────────────────
     -4   -3   -2   -1    0   +1   +2   +3
                   β·ξ − α
```

The three key features of this curve:

1. **It's bounded between 0 and 1.** No matter what numbers you plug in, the output is always a valid probability. You can never get a prediction of "120% likely" or "−30% likely."

2. **It's steepest in the middle.** Near the tipping point (where β·ξ − α = 0), small changes in ideology make a big difference in vote probability. Far from the tipping point, the curve flattens — the outcome is nearly certain either way.

3. **It's symmetric.** The curve rises from 0 toward 1 in the same shape it would fall from 1 toward 0 if flipped.

### The Analogy: A Light Dimmer

Think of the logistic function as a **light dimmer switch** — but one with a particular behavior. When the input (β·ξ − α) is very negative, the light is nearly off (probability near 0). When it's very positive, the light is nearly full brightness (probability near 1). In between, there's a smooth transition zone where the light gradually brightens.

The key: the transition zone is where all the action is. A legislator whose ideology puts them right at the tipping point of a bill could go either way — that's where the model is most uncertain, and that's also where the most information lies. A legislator far from the tipping point is almost certain to vote one way, so their vote is informative but unsurprising.

## What Each Parameter Does

Now let's see how each of the three parameters shapes the model's predictions.

### The Ideal Point (ξ): Where the Legislator Stands

The ideal point is the number that we're ultimately trying to estimate. Negative values mean more liberal, positive values mean more conservative (by Tallgrass's convention — we'll see in Chapter 3 why a convention is necessary).

For the 91st Kansas House, typical ideal points range from about −2.0 (the most liberal Democrat) to about +2.5 (the most conservative Republican), with the center of the scale at 0.

```
Most Liberal  ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → Most Conservative
  -2.0      -1.0       0       +1.0      +2.5
   ↑                    ↑                   ↑
 Liberal D         Moderate R         Conservative R
```

### The Difficulty (α): The Tipping Point

When a bill's discrimination (β) is positive, the difficulty (α) determines where on the ideological spectrum the 50/50 point falls. Specifically, the tipping point is at ξ = α/β.

**Low α** (negative, like α = −2): the bill is easy to pass. Even fairly liberal legislators are likely to vote Yea. Think of a noncontroversial transportation bill.

**High α** (positive, like α = +3): the bill is hard to pass. Only the most conservative legislators are likely to vote Yea. Think of a bill to eliminate a state agency.

**α near 0:** the bill splits the chamber roughly at the ideological midpoint.

### The Discrimination (β): How Sharply the Bill Divides

Discrimination controls the **steepness** of the S-curve. This is where the real power of IRT lies.

**High |β|** (like β = 2.0): The S-curve is steep. There's a narrow zone where the outcome is uncertain; on either side, the vote is nearly guaranteed. This bill is a precise scalpel — it cleanly separates legislators by ideology.

**Low |β|** (like β = 0.3): The S-curve is nearly flat. The probability of a Yea vote barely changes as you move across the ideology spectrum. This bill tells you almost nothing about ideology — everyone is roughly equally likely to vote Yea regardless of where they stand.

**β near 0:** The bill has no ideological content at all. IRT effectively ignores it.

Here's what different discrimination values look like in practice:

```
P(Yea)                               P(Yea)
  1.0 │   ___________                  1.0 │           _____
      │  │                                 │         ╱
  0.5 │  │                             0.5 │    ───
      │  │                                 │  ╱
  0.0 │___│                             0.0 │___
      └───────────────                      └───────────────
      Low ideal  High ideal             Low ideal  High ideal
      HIGH discrimination (β = 2.5)     LOW discrimination (β = 0.3)
      Sharp cutoff — precise scalpel    Gradual slope — blunt instrument
```

### The Sign of β

One subtle but important point: β can be **negative**. A negative discrimination means the bill's ideological polarity is *reversed* — liberals are more likely to vote Yea, not conservatives. About 12.5% of Kansas roll call votes are "D-Yea" bills — votes where the Democratic position is Yea (for example, a bill to expand social services). For these bills, β is negative, reflecting the fact that ideology works in the opposite direction.

## The Item Characteristic Curve

When you plot P(Yea) against ξ for a single bill, you get that bill's **Item Characteristic Curve** (ICC) — the signature S-shape that defines how the bill interacts with ideology.

Let's draw the ICCs for three real types of Kansas bills:

```
P(Yea)
  1.0 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      │    ╱╱╱╱╱╱╱╱╱╱
  0.8 ┤  ╱╱         _____________________________ Routine bill (β=0.3, α=-1)
      │╱╱         ╱╱
  0.6 ┤         ╱╱
      │       ╱╱
  0.4 ┤     ╱╱     Partisan bill (β=1.5, α=0.5)
      │    ╱
  0.2 ┤  ╱                     ╱╱╱╱╱╱ Wedge issue (β=2.5, α=1.5)
      │╱                    ╱╱╱
  0.0 ┤─ ─ ─ ─ ─ ─ ─ ─ ╱╱╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      └─────────────────────────────────────────────
     -3   -2   -1    0    +1   +2   +3
              Legislator ideal point (ξ)
```

- **Routine bill** (flat curve, shifted left): Nearly everyone votes Yea. The curve barely rises — it's already high at the liberal end. Low discrimination means this vote provides almost no ideological information.

- **Partisan bill** (moderate slope, centered): The classic S-curve. Liberals vote Nay, conservatives vote Yea, and there's a meaningful transition zone around ξ = 0.33 (the tipping point at α/β = 0.5/1.5).

- **Wedge issue** (steep slope, shifted right): A sharp divide at ξ = 0.6 (the tipping point at 1.5/2.5). Only strongly conservative legislators vote Yea. The steepness means there's very little gray area — you're either on one side or the other.

## Bayesian Estimation: Learning from the Data

We've described what the model predicts *given* the parameters. But in practice, we don't know the parameters — the whole point is to estimate them from the observed votes. This is where **Bayesian inference** enters the picture.

### The Key Idea

Bayesian statistics works by combining two things:

1. **Prior beliefs** — what we think is plausible before looking at the data
2. **The data** — the actual votes we observed

The combination produces **posterior beliefs** — updated knowledge about the parameters after accounting for the data.

**The analogy: estimating someone's height.** Before you meet someone, you have a prior belief: most adults are between 5'0" and 6'5", with the average around 5'7". If you then learn they're a professional basketball player (data), you update toward the tall end — maybe 6'2" to 6'10". But if you learn they're a jockey, you update toward the short end. The prior gives you a starting range; the data narrows it down.

For IRT, the process works the same way:

- **Before seeing votes:** We believe most legislators are near the ideological center (prior for ξ: Normal(0, 1), meaning most values between −2 and +2).
- **After seeing votes:** A legislator who voted Yea on every conservative bill gets pulled toward the positive end. A legislator who voted Nay on all of them gets pulled toward the negative end. A legislator with mixed votes stays near the center.

The crucial output is not a single number but a **distribution** — a range of plausible values with associated probabilities. This is what makes Bayesian IRT honest about uncertainty.

### Prior Distributions

The Tallgrass 1D IRT model uses the following priors:

```
ξ  ~  Normal(0, 1)     — ideal points: most legislators near center
α  ~  Normal(0, 5)     — difficulty: wide range allowed
β  ~  Normal(0, 1)     — discrimination: moderate values expected
```

**What these mean in plain English:**

- **ξ ~ Normal(0, 1):** "Before seeing any votes, we expect a legislator's ideal point to be somewhere between −2 and +2, with most values near 0." The standard deviation of 1 means about 68% of legislators should fall between −1 and +1, and about 95% between −2 and +2. This is a gentle assumption — it doesn't force anything, just says extreme values (like ±5) are unusual.

- **α ~ Normal(0, 5):** "Bill difficulty could be anywhere from very easy (−10) to very hard (+10)." The wide standard deviation of 5 makes this prior nearly uninformative — we're letting the data decide.

- **β ~ Normal(0, 1):** "Discrimination is usually moderate." Most bills will have |β| between 0 and 2. This gently discourages extreme discrimination values, which can cause numerical instability.

### The Posterior: What We Learn

After fitting the model to the observed votes, each parameter gets a **posterior distribution** — a probability curve showing which values are consistent with the data.

For a well-estimated legislator (one with hundreds of votes), the posterior is a narrow bell curve:

```
Probability
      │       ╱╲
      │      ╱  ╲
      │     ╱    ╲
      │    ╱      ╲
      │   ╱        ╲
      │──╱──────────╲──
      └─────────────────
         +0.6  +0.8  +1.0
          ξ (ideal point)

    "We're confident this legislator is around +0.8"
```

For a legislator with fewer votes (say, 30 out of 500), the posterior is wider:

```
Probability
      │     ╱────╲
      │    ╱      ╲
      │   ╱        ╲
      │  ╱          ╲
      │ ╱            ╲
      │╱──────────────╲
      └─────────────────
       -0.5   +0.5  +1.5
           ξ (ideal point)

    "This legislator is probably moderate-to-conservative, but we're less sure"
```

The width of the posterior is the honest answer to "how much does the data tell us?" Bayesian IRT never hides its uncertainty behind a single number.

### Credible Intervals

The standard summary of a posterior is the **95% Highest Density Interval (HDI)** — the narrowest range that contains 95% of the posterior probability. If a legislator's 95% HDI for ξ is [+0.52, +1.08], that means:

> "Given the observed votes, we believe there is a 95% probability that this legislator's true ideal point lies between +0.52 and +1.08."

This is the Bayesian equivalent of a confidence interval, but with a more intuitive interpretation. (In frequentist statistics, a 95% confidence interval has a more convoluted meaning that most people get wrong. The Bayesian HDI means exactly what it sounds like.)

## How the Sampler Works (Simplified)

We don't solve the Bayesian equations analytically — the math is too complex for that. Instead, we use a **Markov Chain Monte Carlo (MCMC)** algorithm: a computer program that takes millions of guided random walks through the space of possible parameter values, spending more time in regions that are consistent with the data.

### The Analogy: Mountain Hiking in Fog

Imagine you're dropped into a mountain range in dense fog. You can't see the peaks, but you can feel the slope under your feet. Your goal is to find the highest peak.

MCMC works like a hiker who:
1. Takes a step in a random direction
2. If the new position is higher (more consistent with the data), accepts the step
3. If the new position is lower, sometimes accepts anyway (to avoid getting trapped on a foothill), but usually rejects
4. Repeats millions of times

After enough steps, the hiker's path traces out the shape of the mountain range. The places they visited most often are the peaks (the most likely parameter values). The width of their wanderings around each peak reveals its breadth (the uncertainty in the estimate).

Tallgrass uses **nutpie**, a state-of-the-art sampler written in Rust that implements the **NUTS** (No-U-Turn Sampler) algorithm. NUTS is like giving the hiker a pair of rocket boots — instead of taking one step at a time, it simulates the physics of a ball rolling on the landscape, following the curvature of the terrain to take large, efficient leaps. The "no U-turn" part stops the simulation before the ball turns back on itself, avoiding wasted computation.

**Codebase:** `analysis/05_irt/irt.py` (the `build_irt_graph()` function constructs the PyMC model; nutpie compiles and samples it)

### Chains and Convergence

To be confident we've found the right answer, we run **multiple independent chains** — separate hikers starting from different locations. If they all end up wandering around the same peaks, we trust the result. If they end up on different peaks, something went wrong.

Tallgrass runs **2 chains** for the 1D model (4 for 2D and hierarchical models). The key convergence diagnostic is **R-hat** (pronounced "R-hat"):

- **R-hat ≈ 1.00:** The chains agree perfectly. We trust the result.
- **R-hat < 1.01:** Excellent convergence. The standard threshold for production use.
- **R-hat > 1.10:** Concerning. The chains are exploring different regions.
- **R-hat > 2.00:** Failed. The chains fundamentally disagree.

**The analogy:** R-hat is like asking four hikers to independently find the tallest mountain. If they all report the same peak (R-hat ≈ 1.0), you're confident it's the real summit. If two report one peak and two report another (R-hat > 1.5), there might be two mountains of similar height, and you can't be sure which is taller.

For the 91st Kansas House (a well-behaved session), the 1D IRT model converges beautifully: R-hat = 1.001, with each chain producing over 1,600 effective samples in about 42 seconds. For the 79th Kansas Senate (a horseshoe-affected session), convergence is much harder — R-hat can spike to 1.53, with effective sample sizes in the single digits. That's the model telling you something is wrong, and we'll explore what in Chapters 3 and 4.

## A Worked Kansas Example

Let's trace through the entire 1D IRT estimation for a real Kansas session: the **91st House (2025-2026)**.

### The Input

After filtering (removing near-unanimous votes and legislators with fewer than 20 votes), we have:

- **120 legislators** (84 Republicans, 36 Democrats)
- **~500 contested roll calls**
- **~48,000 individual votes** (the non-missing cells in the 120 × 500 matrix)

### The Model

The model has **120 ideal points** (one per legislator), **~500 difficulties** (one per bill), and **~500 discriminations** (one per bill) — roughly **1,120 parameters** to estimate from 48,000 data points. That's about 43 votes per parameter, which is more than enough for reliable estimation.

### The Computation

nutpie compiles the model to native Rust code, then runs 2 chains of 1,000 tuning steps (discarded) plus 2,000 sampling steps each. Total: 4,000 posterior samples.

### The Output

After sampling, each legislator has a posterior distribution for their ideal point. Here's what the results look like for a few notable legislators:

| Legislator | Party | ξ (mean) | 95% HDI | Interpretation |
|-----------|-------|---------|---------|----------------|
| Most liberal D | D | −2.05 | [−2.41, −1.72] | Farthest left in the chamber |
| Median Democrat | D | −1.22 | [−1.48, −0.97] | Typical Democrat |
| Most moderate D | D | −0.45 | [−0.78, −0.14] | Closest Democrat to the center |
| Most moderate R | R | +0.31 | [+0.04, +0.58] | Closest Republican to the center |
| Median Republican | R | +0.89 | [+0.68, +1.11] | Typical Republican |
| Most conservative R | R | +2.42 | [+2.08, +2.79] | Farthest right in the chamber |

Notice the gap between the most moderate Democrat (−0.45) and the most moderate Republican (+0.31). That 0.76-point gap represents the **partisan void** — the ideological space where almost no one stands. It's the mathematical signature of party sorting: legislators cluster within their party, with empty space in between.

### Convergence Diagnostics

For this session:

| Diagnostic | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| R-hat (max across all parameters) | 1.001 | < 1.01 | Pass |
| Bulk ESS (minimum) | 1,658 | > 400 | Pass |
| Tail ESS (minimum) | 1,312 | > 400 | Pass |
| Divergent transitions | 0 | < 10 | Pass |

All green. The model converged cleanly, and we can trust both the point estimates and the credible intervals.

**Codebase:** `analysis/05_irt/irt.py` — the full 1D IRT phase, including model construction, sampling, convergence checks, and output generation.

## What Can Go Wrong

IRT is powerful, but it's not magic. Several things can cause problems:

### Separation

When a legislator always votes one way (100% Yea or 100% Nay), the data provides no upper or lower bound on their ideal point. They could be at +3 or +30 — the data can't tell. The prior keeps the estimate from drifting to infinity, but the posterior will be wide. This is rare for legislators with many votes but can happen for those who voted on only a few bills.

### The Horseshoe Effect (Preview)

As we saw in Volume 3, Chapter 5, PCA can capture the wrong axis in supermajority chambers. This affects IRT too, because IRT's initial values come from PCA. If PC1 points in the wrong direction, the IRT model may converge to a solution that captures intra-party factionalism instead of ideology. We'll address this fully in Chapter 3 (sign correction) and Chapter 4 (2D IRT).

### Multimodality

The IRT model's likelihood has a fundamental symmetry: if you negate all ideal points *and* all discriminations, the predictions are identical. This creates a **bimodal posterior** — two equally valid solutions that are mirror images of each other. Without identification constraints (Chapter 3), the MCMC sampler may get stuck bouncing between these two modes, failing to converge. This is the identification problem, and it's the subject of the next chapter.

---

## Key Takeaway

The 1D IRT model predicts each vote as a function of three things: how far the legislator is from the bill's tipping point (ξ − α/β), how sharply the bill discriminates (β), and random noise. Bayesian estimation produces not just a best guess for each legislator's ideology but a full distribution that honestly represents uncertainty. The model self-checks through convergence diagnostics: if the chains agree (R-hat ≈ 1.0), we trust the answer; if they disagree, we investigate.

---

*Terms introduced: logistic function (sigmoid), S-curve, Item Characteristic Curve (ICC), tipping point (α/β), Bayesian inference, prior distribution, posterior distribution, credible interval, Highest Density Interval (HDI), MCMC (Markov Chain Monte Carlo), NUTS (No-U-Turn Sampler), nutpie, chain, convergence, R-hat, effective sample size (ESS), divergent transitions, separation, bimodal posterior*

*Next: [Anchors and Sign: Telling Left from Right](ch03-anchors-and-sign.md)*
