# Chapter 4: W-NOMINATE: Comparing to the Field Standard

> *In political science, W-NOMINATE is the 800-pound gorilla of ideal point estimation. It's been the default method for decades, used in thousands of studies. If our Bayesian IRT scores disagree with W-NOMINATE, we'd better have a good explanation.*

---

## What W-NOMINATE Is

W-NOMINATE (Weighted NOMINAl Three-step ESTimation) is a method for estimating legislators' ideal points from roll call votes, developed by Keith Poole and Howard Rosenthal in the 1980s. For forty years, it has been the dominant approach in American political science. The DW-NOMINATE scores produced by Poole's Voteview project are the standard reference for every claim about Congressional polarization, party sorting, or ideological alignment you've ever read in a newspaper.

### A Different Approach to the Same Problem

Both W-NOMINATE and Bayesian IRT solve the same problem: given a matrix of votes (legislators × bills), estimate where each legislator falls on an ideological spectrum. But they solve it differently.

Think of it this way. Bayesian IRT is like a **detective building a probabilistic case**. The detective starts with prior beliefs about each legislator (probably near the middle of the spectrum), examines each vote as evidence, updates their beliefs, and ends with a posterior distribution — a range of plausible positions, not just a single number.

W-NOMINATE is like a **cartographer drawing a map**. The cartographer places each legislator as a point in ideological space and each bill as a pair of points (the "Yea" and "Nay" outcomes). A legislator votes for whichever outcome is closer to their position on the map. The cartographer adjusts all the positions until the map explains as many votes as possible.

Both approaches produce a ranking of legislators from liberal to conservative. Both produce similar results. But the underlying machinery is different.

### The Spatial Model: Distances, Not Probabilities

The conceptual difference between W-NOMINATE and IRT comes down to how they think about voting.

**IRT says:** "The probability of a Yea vote is a function of the legislator's ideology and the bill's characteristics." The function is the logistic curve — the S-shaped function from Volume 4. The model is inherently probabilistic: even a very conservative legislator has *some* probability of voting against a conservative bill.

**W-NOMINATE says:** "Each bill has two outcome points in ideological space — where you'd end up if the bill passes (Yea outcome) and where you'd end up if it fails (Nay outcome). A legislator votes for the closer outcome, with some random noise."

This is called a **spatial voting model**, and it has a beautiful geometric intuition. Imagine a map:

```
Liberal ←──────────────────────────────→ Conservative

     Nay outcome         Yea outcome
         ●──────────────────●
                  ↑
            Tipping point
            (equal distance)
```

Any legislator to the left of the tipping point is closer to the Nay outcome and votes Nay. Any legislator to the right is closer to the Yea outcome and votes Yea. The tipping point is where the Yea and Nay outcomes are equidistant — a legislator here could go either way.

In IRT, the tipping point is governed by the difficulty parameter (α). In W-NOMINATE, it's governed by the geometry of the outcome points. The two approaches formalize the same intuition differently.

### The Gaussian Kernel

Where IRT uses the logistic function to convert ideology into a vote probability, W-NOMINATE uses a **Gaussian (normal distribution) kernel**. The probability of voting Yea decreases with the square of the distance from the Yea outcome point, following a bell curve.

In practice, the logistic and Gaussian kernels produce very similar results. The S-curve (logistic) and the bell curve (Gaussian) have slightly different shapes in the tails, but in the range where most of the action happens — near the tipping point — they're nearly identical. This is why IRT and W-NOMINATE tend to agree so strongly.

### MLE vs. MCMC

W-NOMINATE uses **Maximum Likelihood Estimation (MLE)** — it finds the single set of parameters that makes the observed votes most probable. Think of it as finding the peak of a mountain. There's one answer: the summit.

Bayesian IRT uses **MCMC** — it explores the entire mountain range, recording not just the peak but the shape of every ridge and valley. The result isn't a single point but a distribution: "the summit is probably around here, but it could be a little to the east or west."

The peak (MLE estimate) and the posterior mean (Bayesian estimate) usually agree closely. The difference is in what you get alongside the estimate: MLE gives you a standard error (how precise the estimate is, assuming the model is exactly right), while MCMC gives you a credible interval (how uncertain you are about the true value, accounting for both estimation noise and model uncertainty).

For ranking legislators — the primary use case — the distinction rarely matters. The moderate Republican ranks 40th by both methods. But for quantifying uncertainty — "how confident are we that Legislator A is more conservative than Legislator B?" — the Bayesian approach is more natural.

## How Tallgrass Runs W-NOMINATE

W-NOMINATE is implemented in R, not Python. The `wnominate` R package wraps Fortran code that has been refined over decades. Rather than reimplementing this in Python (and risking subtle numerical differences), Tallgrass calls R as a subprocess:

```
Python (Tallgrass) → prepares vote matrix as CSV
    → calls Rscript wnominate.R
        → R loads pscl, builds rollcall object
        → R runs wnominate() (MLE optimization, 3 trials)
        → R runs oc() (Optimal Classification)
        → R writes results as CSV + JSON
Python ← reads results, computes correlations, builds report
```

This pattern (Python orchestration with R computation) is common in data science. R's `wnominate` and `oc` packages have decades of testing and validation behind them. Tallgrass defers to them for the estimation, then brings the results back into Python for comparison.

### Vote Matrix Encoding

W-NOMINATE uses a specific coding convention (from the `pscl` package):

| Code | Meaning |
|------|---------|
| 1 | Yea |
| 6 | Nay |
| 9 | Missing / Not Voting |
| NA | Not in legislature |

Tallgrass converts its binary vote matrix (1 = Yea, 0 = Nay, NaN = absent) to this format before passing it to R. The lopsided-vote filter (minority < 2.5%) and minimum-vote requirement (20 votes per legislator) are applied identically to both the IRT and W-NOMINATE analyses, ensuring an apples-to-apples comparison.

### Polarity: Which End Is Conservative?

Like IRT, W-NOMINATE faces the reflection problem — the model can't inherently distinguish left from right (see Volume 4, Chapter 3). W-NOMINATE requires a **polarity legislator**: one legislator who is specified as being on the positive end of the first dimension. The algorithm then orients the entire scale so that this legislator is positive.

Tallgrass selects the polarity legislator automatically using PCA PC1: the legislator with the highest PC1 score (among those with ≥50% participation) is designated as the polarity anchor. This is the same data-driven convention used by `pscl::ideal` (Jackman 2001) and matches Tallgrass's approach across all other phases.

After estimation, Tallgrass performs a **sign alignment check** — computing the Pearson correlation between W-NOMINATE's Dimension 1 and the IRT ideal points. If the correlation is negative (indicating the scales are flipped), all W-NOMINATE scores are multiplied by -1.

## What Gets Compared

The W-NOMINATE phase computes several types of comparisons.

### Score-Level Correlation

The primary test: **do the rankings agree?**

For each chamber, Tallgrass computes:
- Pearson r between IRT ideal points and W-NOMINATE Dimension 1 coordinates
- Spearman rank correlation (less sensitive to nonlinear relationships)
- Within-party correlations (separately for Republicans and Democrats)

Typical results for Kansas chambers:

| Comparison | Typical |r| |
|-----------|---------|
| IRT vs. W-NOMINATE Dim 1 (overall) | 0.96 – 0.99 |
| IRT vs. W-NOMINATE (within Republican) | 0.90 – 0.96 |
| IRT vs. W-NOMINATE (within Democrat) | 0.88 – 0.95 |
| IRT vs. OC (overall) | 0.93 – 0.97 |

The IRT-vs-WNOM correlation is typically higher than the IRT-vs-Shor-McCarty correlation. This makes sense: W-NOMINATE and IRT use the same input data (Kansas roll call votes), just with different statistical machinery. Shor-McCarty uses different data (a national survey bridge), so more divergence is expected.

### Fit Statistics: How Each Method Describes the Data

W-NOMINATE and Optimal Classification each produce global fit statistics that Tallgrass reports alongside IRT metrics:

| Metric | What It Measures | IRT Equivalent |
|--------|-----------------|----------------|
| CC (Correct Classification) | Fraction of votes predicted correctly | Classification accuracy |
| APRE | Improvement over always-majority baseline | Same concept, same formula |
| GMP | Geometric mean probability of correct predictions | Same concept, same formula |

Because these metrics use the same formulas, they provide a direct apples-to-apples comparison. A typical result:

| Method | Accuracy | APRE | GMP |
|--------|----------|------|-----|
| 1D IRT | 91.2% | 0.51 | 0.838 |
| W-NOMINATE 2D | 91.8% | 0.55 | 0.845 |
| Optimal Classification | 93.5% | 0.64 | — |

Optimal Classification typically achieves the highest accuracy because it has no distributional assumptions — it simply finds the cut points that minimize errors, without worrying about probability. But this flexibility comes at a cost: OC doesn't produce probabilities, so GMP can't be computed for it.

### Scree Plots: How Many Dimensions?

W-NOMINATE reports **eigenvalues** that indicate how much of the voting variance each dimension captures. Tallgrass plots these as a scree plot — a chart that shows the explained variance dropping off with each additional dimension.

For the Kansas Legislature, the scree plot typically shows:

- **Dimension 1** captures 60–75% of the variance (the party divide)
- **Dimension 2** captures 5–10% of the variance (moderate-vs-establishment within the Republican majority)
- **Dimensions 3+** capture diminishing amounts, usually below the noise floor

This is consistent with the IRT finding that 2D is sometimes justified but 1D captures the dominant structure. The scree plot provides a visual sanity check: if W-NOMINATE said Kansas needs five dimensions, we'd question our 2D IRT model. Instead, both methods agree on the dimensionality.

## Optimal Classification: The Nonparametric Benchmark

Alongside W-NOMINATE, Tallgrass runs **Optimal Classification (OC)**, developed by Keith Poole in 2000. Where W-NOMINATE and IRT both assume a parametric probability model (Gaussian kernel and logistic function, respectively), OC makes **no distributional assumptions at all**.

### How OC Works

OC asks a purely geometric question: is there a set of ideal points and a set of cutting planes such that every legislator votes Yea on the side of the plane where their ideal point lies, and Nay on the other side?

In one dimension, a "cutting plane" is just a point on the line:

```
Nay ← ────────|──────── → Yea
               cut point
```

OC finds the ideal points and cut points that **minimize the number of classification errors** — votes where a legislator ended up on the "wrong" side of the cut. It uses a combinatorial optimization algorithm rather than a likelihood function.

### Why OC Matters for Validation

OC provides the most method-independent comparison possible. If OC, W-NOMINATE, and Bayesian IRT all produce similar legislator rankings, the rankings are **robust to the choice of method** — they're not an artifact of the logistic function, the Gaussian kernel, or the Bayesian framework. They reflect genuine structure in the voting data.

When all three methods agree (which they typically do in Kansas, with pairwise correlations above 0.93), it's strong evidence that there really is a one-dimensional ideological spectrum along which Kansas legislators are arrayed, and our estimates of where each legislator falls on that spectrum are reliable.

### When OC Disagrees

OC's nonparametric flexibility means it can sometimes capture structure that parametric models miss — or be misled by patterns that parametric models correctly smooth over. A common scenario:

- A small cluster of legislators has a highly unusual voting pattern (e.g., three moderate Republicans who break from the party on a specific set of issues).
- OC, being nonparametric, can perfectly classify these legislators by placing them in an unusual position.
- W-NOMINATE and IRT, constrained by their parametric assumptions, place these legislators closer to the party mainstream and accept a few classification errors.

Neither answer is "wrong." OC describes the data more precisely. The parametric models describe the *pattern* more robustly — they're less likely to chase noise. This is the classic bias-variance tradeoff in statistics: more flexibility means better fit to the current data but potentially worse generalization to new data.

## What the Three-Way Comparison Tells Us

When Tallgrass reports the correlation matrix between IRT, W-NOMINATE, and OC, it's testing the claim that our ideology estimates are **method-invariant** — that they reflect real structure rather than an artifact of our particular modeling choices.

```
              IRT    WNOM    OC
IRT           1.00   0.97   0.95
W-NOMINATE    0.97   1.00   0.96
OC            0.95   0.96   1.00
```

In a matrix like this, every method agrees with every other method at r > 0.95. The small differences are explained by the different assumptions: the parametric methods (IRT and W-NOMINATE) agree most closely with each other, while OC — unconstrained by distributional assumptions — shows slightly more divergence.

This is exactly the pattern a political methodologist would hope to see: **the ideological structure is robust, and the methods are interchangeable for practical purposes.** The choice to use Bayesian IRT (with its credible intervals, hierarchical extensions, and model comparison framework) is a choice of convenience and additional capability, not a choice that alters the substantive conclusions.

## Why Tallgrass Uses IRT, Not W-NOMINATE

If W-NOMINATE has been the standard for decades, why does Tallgrass use Bayesian IRT as its primary method? Several reasons:

1. **Uncertainty quantification.** IRT produces full posterior distributions. W-NOMINATE produces point estimates and bootstrap standard errors. For downstream analyses (clustering, profiles, temporal trends), having posterior distributions enables propagation of uncertainty — knowing not just "where is this legislator?" but "how confident are we?"

2. **Hierarchical extensions.** Bayesian IRT naturally extends to hierarchical models (Volume 4, Chapter 5) that incorporate party structure. W-NOMINATE has no equivalent hierarchical framework.

3. **Model comparison.** The Bayesian framework allows principled comparison between 1D and 2D models via LOO-CV (Chapter 2). W-NOMINATE's comparison requires separate model fits and ad hoc criteria.

4. **Flexibility.** Bayesian IRT can incorporate prior information (from PCA, from external scores, from domain knowledge) into the estimation. W-NOMINATE's MLE framework has no mechanism for priors.

5. **Modern ecosystem.** IRT integrates with PyMC, nutpie, and ArviZ — a mature Python-native toolchain. W-NOMINATE requires R.

None of these reasons mean W-NOMINATE is inferior. For its intended purpose — ranking legislators on an ideological scale — it works excellently and has been validated far more extensively than any other method. Tallgrass uses it as a benchmark precisely because of its track record and widespread acceptance.

## Key Takeaway

W-NOMINATE and Optimal Classification provide the most powerful form of methods-based validation: they use the same data as our IRT model but different statistical machinery. When all three methods agree — Bayesian IRT (logistic, MCMC), W-NOMINATE (Gaussian, MLE), and Optimal Classification (nonparametric) — the legislator rankings are robust to the choice of method. They reflect genuine structure in the voting data, not an artifact of any particular modeling decision. For Kansas, the typical three-way agreement exceeds r = 0.95, confirming that the choice to use Bayesian IRT is a choice of additional capability (posteriors, hierarchical extensions, model comparison), not a choice that changes the substantive conclusions.

*Terms introduced: W-NOMINATE (Weighted NOMINAl Three-step ESTimation), DW-NOMINATE, Keith Poole, Howard Rosenthal, spatial voting model, Gaussian kernel, outcome points, tipping point, Maximum Likelihood Estimation (MLE), polarity legislator, sign alignment, Optimal Classification (OC), nonparametric, classification error, bias-variance tradeoff, scree plot, eigenvalues, method-invariance, convergent validity*

---

*Previous: [Chapter 3 — The Gold Standard: Shor-McCarty External Validation](ch03-shor-mccarty.md)*

*Next: [Chapter 5 — Quality Gates: Automatic Trust Levels](ch05-quality-gates.md)*
