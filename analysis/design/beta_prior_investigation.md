# When Your Model Can Only Hear One Side of the Argument

**A case study in how a "standard" statistical choice silenced 12% of our data — and how we found and fixed it.**

## The Setup

We're building a statistical model of the Kansas Legislature. The goal: estimate where each legislator falls on an ideological spectrum, from most liberal to most conservative, using nothing but their roll-call votes (Yea or Nay on each bill).

The method is called **Item Response Theory (IRT)**, borrowed from educational testing. In a classroom, IRT figures out how skilled each student is based on which test questions they got right. In a legislature, it figures out how conservative each legislator is based on which bills they voted for. Each legislator gets an "ideal point" (their position on the spectrum), and each bill gets two numbers: how "hard" it is (what fraction of legislators vote Yea) and how "discriminating" it is (how well it separates liberals from conservatives).

A highly discriminating bill is one where knowing a legislator's ideology almost perfectly predicts their vote — a classic party-line fight. A low-discrimination bill is one where ideology tells you nothing — maybe it's bipartisan, maybe it's about renaming a bridge.

## The Model, Simply

The core equation is:

```
P(Yea) = logistic(discrimination × ideology - difficulty)
```

For any given bill, the probability that a legislator votes Yea depends on:
- Their **ideology** (higher = more conservative)
- The bill's **discrimination** (how much ideology matters for this vote)
- The bill's **difficulty** (the overall threshold — is this an easy Yea or a hard one?)

The logistic function squishes the result into a probability between 0% and 100%. If `discrimination × ideology - difficulty` is a big positive number, the legislator almost certainly votes Yea. If it's a big negative number, they almost certainly vote Nay.

## The Choice That Seemed Safe

Here's where it gets interesting. The discrimination parameter can theoretically be positive or negative:

- **Positive discrimination:** More conservative legislators are more likely to vote Yea. (A typical Republican-sponsored bill.)
- **Negative discrimination:** More *liberal* legislators are more likely to vote Yea. (A typical Democrat-sponsored bill.)

When fitting the model, the computer explores millions of possible combinations of parameters. There's a well-known problem: if you flip the sign of *both* the discrimination *and* every legislator's ideology score, the math works out identically. The model can't tell the difference. This is called the **sign identification problem**, and it causes the fitting algorithm to wander between two mirror-image solutions, producing garbage.

The standard fix from the academic literature: **force discrimination to be positive.** If discrimination can only be positive, the mirror-image solution is impossible, and the model behaves. We used a LogNormal prior — a statistical constraint that ensures discrimination is always greater than zero.

This is textbook. It's what the papers recommend. It seemed safe.

## The Problem We Didn't See

Forcing discrimination to be positive means the model can only represent one direction: more conservative = more likely to vote Yea. But legislatures vote on bills from both sides of the aisle. When Democrats propose a bill and vote Yea while Republicans vote Nay, the "correct" discrimination is *negative* — more liberal legislators are more likely to vote Yea.

With positive-only discrimination, the model has no way to express this. Its only option is to set discrimination to nearly zero, which is the model's way of saying "this bill tells me nothing about ideology." But that's wrong — a clean party-line vote where Democrats vote Yea and Republicans vote Nay is *extremely* informative about ideology. The model just can't hear it.

We didn't notice this during development because the model's overall accuracy was good (91%) and all the standard diagnostic checks passed. The problem only became visible when we audited the bill-level parameters and asked: "Why do some highly partisan votes have near-zero discrimination?"

## What the Data Showed

We classified every contested House bill by which party formed the Yea majority:

| Bill type | Count | Model's discrimination |
|-----------|-------|----------------------|
| Republican-Yea bills | 259 (87%) | Average: 3.46 (high — model sees these) |
| Democrat-Yea bills | 37 (13%) | Average: 0.19 (near zero — model is blind) |

**Every single Democrat-Yea bill** had a discrimination below 0.5. The model treated all 37 of them as pure noise. On a bill where all 37 Democrats voted Yea and all 84 Republicans voted Nay — a perfect ideological split — the model's prediction was "everyone has about a 28% chance of voting Yea, regardless of their ideology."

That's like a teacher grading a test and deciding that the questions where the smart kids got them right and the struggling kids got them wrong are... uninformative. Only the reverse pattern counts.

## Why We Missed It (And Why You Might Too)

Three things conspired to hide the problem:

**1. Kansas has a Republican supermajority.** With 92 Republicans and 38 Democrats in the House, most contested bills pass with Republican support. Only 13% of contested bills had Democrat-Yea majorities. In a more evenly divided legislature, this number could be 40-50%, making the problem devastating.

**2. The information loss was partially redundant.** A legislator who consistently votes Nay on Republican-Yea bills will typically vote Yea on Democrat-Yea bills. So the Republican-Yea votes already capture *most* of the ranking information. The Democrat-Yea votes add precision — they help separate legislators who are similar — but they don't fundamentally reorder everyone.

**3. The standard diagnostics didn't flag it.** Holdout accuracy was 91%. The posterior predictive checks passed. The sensitivity analysis showed robust results. Everything looked fine because the model was getting the *direction* right for most legislators — it just wasn't using all available information to get the *spacing* right.

## The Key Insight: We Already Solved the Problem Another Way

Remember the sign identification problem — the reason we constrained discrimination to be positive in the first place? We had actually already solved it using a different technique: **anchoring**.

Before fitting the model, we permanently fixed two legislators' ideology scores: the most conservative Republican at +1 and the most liberal Democrat at -1. These "anchors" break the mirror-image symmetry. Once the model knows that Legislator A is at +1 and Legislator B is at -1, there's only one consistent solution for everyone else.

And here's the key: **once the ideology scores have a fixed direction (positive = conservative), the discrimination parameter's sign is determined by the data.** If conservatives tend to vote Yea on a bill, discrimination comes out positive. If liberals tend to vote Yea, it comes out negative. No ambiguity. No sign-switching.

We were wearing both a belt and suspenders. The anchors (belt) were doing the job. The positive constraint (suspenders) was redundant — and it was so tight it was cutting off circulation to our legs.

## The Fix

One line of code. Change:

```python
# Before: constrained positive (LogNormal)
beta = pm.LogNormal("beta", mu=0.5, sigma=0.5, ...)

# After: unconstrained (Normal, centered at zero)
beta = pm.Normal("beta", mu=0, sigma=1, ...)
```

This lets discrimination be positive (Republican-Yea bills) or negative (Democrat-Yea bills), with the anchors ensuring the sign is always meaningful.

## The Experiment

We ran both versions on the same data (House chamber, 130 legislators, 297 bills) and compared everything:

| What we measured | Constrained (LogNormal) | Unconstrained (Normal) | Change |
|---|---|---|---|
| Holdout accuracy | 90.8% | 94.3% | **+3.5%** |
| Holdout AUC-ROC | 0.954 | 0.979 | **+0.025** |
| D-Yea bill discrimination | 0.19 (blind) | 2.38 (sees them) | **12× better** |
| Effective sample size | 21 (struggling) | 203 (healthy) | **10× better** |
| Sampling speed | 62 seconds | 51 seconds | **18% faster** |
| Sign-switching problems | N/A | None (0 divergences) | No issues |
| Correlation with PCA | 0.950 | 0.972 | **+0.022** |

The unconstrained model is better on every single metric. It's more accurate, converges faster, runs faster, and uses all the data. The feared sign-switching problem — the entire reason for the constraint — simply didn't happen.

## What Changed in the Results

The most visible difference is in the **beta (discrimination) distribution**:

- **Before:** A bimodal pile-up. Republican-Yea bills clustered around +7 (hitting a ceiling), Democrat-Yea bills crushed near zero. A gap in between.
- **After:** A clean, symmetric spread. Republican-Yea bills are positive, Democrat-Yea bills are negative, and the magnitude reflects how partisan each vote was.

The ideal point estimates (the legislator scores) shifted modestly. The overall ranking barely changed — the correlation between old and new scores is r = 0.983. But the *spacing* changed meaningfully. Democrats became more spread out, because the model now has 37 additional discriminating votes to tell them apart. The most extreme conservatives have tighter credible intervals, because those same 37 votes (on which they uniformly voted Nay) provide additional confirming evidence.

## The Broader Lesson

**Standard recommendations exist for a reason — but they encode assumptions. When your model doesn't share those assumptions, the recommendation can quietly work against you.**

The LogNormal prior for IRT discrimination is standard advice in political science. It works well when the model uses "soft identification" — relying entirely on priors to keep the parameters from wandering. But our model uses "hard identification" — two legislators are pinned to fixed values. The advice was designed for a different situation, and applying it uncritically cost us 3.5 percentage points of accuracy and 10× convergence efficiency.

The fix wasn't complicated. The investigation took longer than the repair. The real cost was the time between "this looks fine" and "wait, why are all these partisan votes showing zero discrimination?" — a question that only arose because we audited the bill-level parameters instead of just checking the headline accuracy number.

Statistical models can pass every standard diagnostic while still having a structural blind spot. The diagnostics test whether the model is *internally consistent*. They don't test whether it's *using all your data*.

---

## Technical Reference

**Date:** 2026-02-20
**Status:** Resolved. Normal(0,1) prior adopted.
**Script:** `analysis/irt_beta_experiment.py`
**Output:** `results/2025-2026/irt/beta_experiment/`
**Related files:** `analysis/design/irt.md`, `docs/adr/0006-irt-implementation-choices.md`, `docs/lessons-learned.md` (Lesson 6)

### Experiment details

- **Data:** Kansas House, 2025-26 session (130 legislators × 297 contested votes)
- **MCMC:** 500 draws, 300 tune, 2 chains (reduced for speed; full production run uses 2000/1000/2)
- **Bill classification:** R-Yea = majority of Republicans vote Yea (259 bills); D-Yea = majority of Democrats vote Yea (37 bills)
- **Variants tested:** LogNormal(0.5, 0.5), Normal(0, 2.5), Normal(0, 1)
- **Winner:** Normal(0, 1) — best convergence, fastest sampling, highest PCA correlation, holdout accuracy within 0.1% of wider Normal(0, 2.5)

### Full metrics

| Metric | LogNormal(0.5,0.5) | Normal(0,2.5) | Normal(0,1) |
|---|---|---|---|
| Sampling time (s) | 62 | 79 | **51** |
| Divergences | 0 | 0 | 0 |
| ξ R-hat max | 1.070 | 1.014 | **1.013** |
| ξ ESS min | 21 | 123 | **203** |
| β R-hat max | 1.017 | 1.018 | **1.014** |
| PCA Pearson r | 0.950 | 0.963 | **0.972** |
| PCA Spearman ρ | **0.916** | 0.905 | 0.915 |
| Holdout accuracy | 0.908 | **0.944** | 0.943 |
| Holdout AUC-ROC | 0.954 | **0.980** | 0.979 |
| D-Yea \|β\| mean | 0.186 | **4.767** | 2.384 |
| R-Yea β mean | 3.45 | 2.68 | 1.37 |
| Bills with β < 0 | 0 | 63 | 67 |

### The math in detail

The IRT model: `P(Yea) = logit⁻¹(β_j · ξ_i - α_j)`

- `ξ_i` = legislator i's ideal point (higher = more conservative)
- `β_j` = bill j's discrimination (how much ideology predicts the vote)
- `α_j` = bill j's difficulty (overall Yea threshold)

With `β > 0`: `∂P/∂ξ > 0` always. More conservative legislators always have higher P(Yea). No value of α changes this — α shifts the logistic curve left/right but never flips it. For a D-Yea bill (where `∂P/∂ξ < 0` is the truth), the model must set β ≈ 0, collapsing to `P(Yea) ≈ logit⁻¹(-α)` for everyone.

With unconstrained β: D-Yea bills get `β < 0`, giving `∂P/∂ξ < 0` — more conservative legislators have *lower* P(Yea), matching reality. The sign of β now encodes the direction of the vote, and the magnitude encodes how partisan it was.

### Why anchors provide sufficient identification

The sign ambiguity in IRT arises because `(β, ξ)` and `(-β, -ξ)` produce identical likelihoods. Hard anchors break this symmetry by fixing specific ξ values:

- Anchor conservative at ξ = +1
- Anchor liberal at ξ = -1

These constraints eliminate the `(-β, -ξ)` solution entirely. For any bill, the data uniquely determines whether β is positive (conservatives favor Yea) or negative (liberals favor Yea). No additional constraint on β is needed.

This is in contrast to soft identification (Normal(0,1) prior on ξ, no anchors), where the posterior is symmetric and β can wander. In that setting, positive-constrained β is genuinely helpful. But it's solving a problem that doesn't exist when you have anchors.
