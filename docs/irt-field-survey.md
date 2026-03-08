# The IRT Identification Problem and Why Python Has No Good Answer

How a one-line prior change revealed a gap between textbook recommendations and empirical reality — and why there's no production Python package for legislative ideal points.

**Date:** 2026-02-25

---

## The Problem Every Implementation Must Solve

Item Response Theory models for legislative voting have a fundamental mathematical ambiguity. The core equation:

```
P(Yea) = logit⁻¹(β_j · ξ_i - α_j)
```

produces identical predictions if you negate both the discrimination (β) and all ideal points (ξ). Flip every legislator from liberal-negative to liberal-positive, and flip every bill's discrimination sign to match — the probabilities don't change. The math can't tell the difference.

This is the **reflection invariance** problem, and it's only one of several identification failures. Farouni's [taxonomy](https://rfarouni.github.io/assets/projects/Identification/Identification.html) documents five:

1. **Additive aliasing** — shifting all abilities and difficulties by a constant preserves the likelihood
2. **Multiplicative aliasing** — rescaling discrimination and ability-difficulty differences cancels out
3. **Reflection invariance** — negating all β and ξ simultaneously produces identical predictions
4. **Label switching** — in D dimensions, D! permutations of dimension labels are equivalent
5. **Rotation invariance** — combined, these produce 2^D × D! equivalent posterior modes

In practice, the first two are resolved by fixing the location and scale of the ideal point distribution (via anchors or standardization). The last two matter only in multi-dimensional models. But reflection invariance affects every 1D IRT implementation.

In MCMC sampling, reflection invariance manifests as **mode-splitting**: two chains exploring mirror-image solutions, producing bimodal posteriors, R-hat values near 2.0, and effective sample sizes near 1. As Erosheva and Bhattacharyya (2023) document in [Dealing with Reflection Invariance in Bayesian Factor Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758924/): "When MCMC moves between 2^q equivalent reflection modes, simple summaries of parameters such as the posterior mean or posterior standard deviation will be misleading." If chains explore both (+β, +ξ) and (-β, -ξ) solutions, the posterior mean collapses to zero — pure noise.

Every implementation of legislative IRT must solve this problem. How they solve it determines what else they get right — and wrong.

---

## How the Field Solved It: A Brief History

### The Early Answer: Constrain Discrimination Positive

The first widely-adopted solution came from educational testing, where IRT originated. If you force discrimination to be positive (using a LogNormal or Half-Normal prior on β), the mirror-image solution is impossible. Only one mode exists.

This is clean, simple, and it works — for educational testing. When every test question has a "right" answer, discrimination is naturally positive: more skilled students are more likely to get it right. Constraining β > 0 merely encodes a truth about the domain.

Political science adopted this convention. The Stan User's Guide [recommends LogNormal priors for 2PL discrimination](https://mc-stan.org/docs/2_20/stan-users-guide/item-response-models-section.html), stating the constraint "prevents questions from being easier for lower-ability students." Gelman's [2018 blog post on IRT priors](https://statmodeling.stat.columbia.edu/2018/03/01/prior-use-item-response-parameters/) partially dissented — arguing that "in real life, items can have zero or negative discrimination" and preferring a Normal prior — but this was in an educational testing context where negative discrimination indicates a badly constructed item, not a systematic directional pattern. The default in most implementations — R's `pscl::ideal()`, early Stan examples, textbook code — remains positive-constrained discrimination.

### Clinton-Jackman-Rivers (2004): Anchor Constraints

The canonical framework for legislative IRT came from Clinton, Jackman, and Rivers. Their key insight: instead of constraining bill parameters, **fix two legislators at known positions**. Pin the most conservative legislator at ξ = +1 and the most liberal at ξ = -1.

This simultaneously resolves:
- **Additive aliasing**: ξ + c, α + c·β produce the same likelihood. Fixed points eliminate c.
- **Multiplicative aliasing**: c·ξ, c·β, c·α are indistinguishable. Fixed scale eliminates c.
- **Sign ambiguity**: -ξ, -β give the same likelihood. Fixed direction eliminates the flip.

Anchor constraints are "hard identification" — they eliminate the ambiguity structurally rather than through the prior distribution.

### Jackman (2001): PCA Starting Values

Simon Jackman's implementation in R's `pscl::ideal()` added a practical innovation: initialize MCMC chains with eigendecomposition of the vote matrix (equivalent to PCA PC1 scores). This has been the default (`startvals="eigen"`) since 2001.

The rationale is straightforward: PCA and IRT with a logistic link agree to first order. Starting chains near the PCA solution puts them in the correct mode's basin of attraction, preventing mode-splitting even when the prior alone doesn't fully resolve the ambiguity.

### The Modern Toolkit

By the mid-2010s, the field had settled on a standard approach:

1. **Anchors** for identification (fix 2 legislators)
2. **PCA initialization** for convergence (avoid mode-splitting)
3. **LogNormal or positive-constrained β** as a "belt-and-suspenders" safety measure
4. **R-hat < 1.01, ESS > 400** for convergence assessment (Vehtari et al., 2021)

This combination works reliably. But item 3 — the positive constraint on β — interacts with item 1 in a way the literature rarely discusses.

---

## The Tension: Constrained β + Hard Anchors

Here's the key insight, discovered through our [beta prior investigation](analysis/design/beta_prior_investigation.md):

**When you use hard anchors, the positive constraint on β is not just redundant — it's actively harmful.**

The argument:

1. Hard anchors fix ξ_conservative = +1 and ξ_liberal = -1. This eliminates the mirror-image solution entirely.
2. With ξ direction established, β's sign is determined by the data. A bill where conservatives vote Yea gets β > 0. A bill where liberals vote Yea gets β < 0. No ambiguity.
3. Constraining β > 0 forces Democrat-Yea bills to β ≈ 0, treating them as uninformative.
4. In a Republican-dominated legislature like Kansas (72% R), this silences ~13% of contested bills.

The model can't represent a bill where liberals are more likely to vote Yea. Its only option is to set β near zero — the mathematical equivalent of saying "this bill tells us nothing about ideology." But a clean party-line vote where all Democrats vote Yea and all Republicans vote Nay is *extremely* informative. The model just can't hear it.

### Why Nobody Noticed

Three factors conspire to hide this:

**The data distribution protects you.** In a Republican supermajority, most contested bills pass with Republican support. Only 13% of contested bills had Democrat-Yea majorities. In an evenly divided legislature, the problem would affect ~40-50% of bills.

**The information loss is partially redundant.** A legislator who consistently votes Nay on R-Yea bills will typically vote Yea on D-Yea bills. The R-Yea votes already capture most ranking information. The D-Yea votes add precision — they help separate similar legislators — but don't fundamentally reorder anyone.

**Standard diagnostics don't flag it.** Holdout accuracy was 91%. Posterior predictive checks passed. Everything looked fine because the model got the direction right for most legislators — it just wasn't using all available information to get the spacing right.

### The Experiment

We ran three β prior variants on the same data (Kansas House, 130 legislators, 297 contested bills):

| Metric | LogNormal(0.5, 0.5) | Normal(0, 2.5) | Normal(0, 1) |
|---|---|---|---|
| Holdout accuracy | 90.8% | 94.4% | **94.3%** |
| Holdout AUC-ROC | 0.954 | 0.980 | **0.979** |
| D-Yea bill \|β\| mean | 0.19 | 4.77 | **2.38** |
| Min ESS (ξ) | 21 | 123 | **203** |
| Sampling time | 62s | 79s | **51s** |
| PCA correlation | 0.950 | 0.963 | **0.972** |

The unconstrained Normal(0, 1) is better on every metric. It's more accurate, converges faster, runs faster, and uses all the data.

### Why This Isn't Widely Documented

The literature on IRT identification mostly assumes one of two setups:

1. **Soft identification** (Normal priors on ξ, no anchors) — where positive β constraint genuinely helps by eliminating one mode
2. **Hard identification** (anchored ξ) — where anchors do the work

Most textbooks and tutorials discuss these separately. They rarely address what happens when you combine hard anchors with a positive β constraint inherited from the soft-identification tradition. The combination is what most practitioners actually use, because they follow two pieces of standard advice simultaneously without realizing they conflict.

Bob Carpenter, writing on the [PyMC Discourse in April 2025](https://discourse.pymc.io/t/how-to-parameterize-and-identify-an-irt-2-pl-model-pymc-bot-test/16922), noted that "soft identification is usually insufficient for these models" and recommended sum-to-zero constraints. But even that thread doesn't explicitly flag the interaction between hard anchors and positive-constrained discrimination.

---

## The Python Ecosystem Gap

### What Exists

A survey of every Python IRT implementation we could find:

| Package | Framework | Method | Suitable for Legislative IRT? |
|---------|-----------|--------|-------------------------------|
| [py-irt](https://github.com/nd-ball/py-irt) | Pyro/PyTorch | Variational inference | No — no anchors, positive β constraint, NLP/educational focus, Python <3.12 |
| [girth](https://pypi.org/project/girth/) | NumPy/SciPy | MML (MLE) | No — psychometric tool, MLE only, no Bayesian, unmaintained since 2021 |
| [girth-mcmc](https://pypi.org/project/girth-mcmc/) | PyMC3 | MCMC | No — Bayesian extension of girth, unmaintained since 2021 |
| [deepirtools](https://pypi.org/project/deepirtools/) | PyTorch | Deep learning | No — exploratory/confirmatory factor models, not legislative IRT |
| [tbip](https://github.com/keyonvafa/tbip) | TF 1.14 / NumPyro | VI / MCMC | Partially — text-focused, outdated TF dependencies |
| [Jnotype](https://github.com/cbg-ethz/Jnotype) | JAX | Various | No — WIP, unstable API, 3 stars |
| [jamalex/bayesian-irt](https://github.com/jamalex/bayesian-irt) | PyMC | MCMC | No — educational example only |
| PyMC Discourse examples | PyMC | MCMC | Fragments — no production-grade implementation |

The [ideal-point-estimation](https://github.com/topics/ideal-point-estimation) GitHub topic lists 5 repositories total, only one in Python (Jnotype, WIP).

### What Political Scientists Actually Use

The production tools are all in R or R+Stan:

| Package | Method | Key Feature | Status |
|---------|--------|-------------|--------|
| `pscl::ideal()` | Gibbs MCMC | Canonical CJR; `startvals="eigen"` default | Stable, mature |
| `MCMCpack::MCMCirt1d()` | Gibbs MCMC | Martin-Quinn dynamic model | Stable, mature |
| `emIRT` | EM algorithm | 100-1000x faster; Imai et al. (2016) | Active |
| `idealstan` | Stan HMC | Most comprehensive: time-varying, informative missingness, auto-ID via `pathfinder()` | Active |
| `wnominate` | MLE | Poole-Rosenthal NOMINATE | Stable |
| `brms` | Stan HMC | General-purpose; supports IRT via formula interface | Active |
| `hbamr` | Stan HMC | Hierarchical Bayesian A-M scaling | Active (Feb 2026) |
| `pgIRT` | Polya-Gamma EM | Data augmentation for fast IRT | Active (Oct 2024) |

There is no PyPI package, no maintained GitHub repository, and no tutorial that provides a production-ready PyMC implementation of 2PL IRT with proper identification for roll-call data. This is remarkable given Python's dominance in data science and Bayesian modeling.

### Why the Gap Exists

Three factors:

1. **Path dependence.** The canonical implementations (CJR, Martin-Quinn, NOMINATE) were built in R, Fortran, and BUGS. Political science methodology programs teach R. The toolchain is mature and there's no pressure to rewrite in Python.

2. **Identification is hard to get right.** The April 2025 PyMC Discourse thread where Bob Carpenter evaluated AI-generated IRT code found that every attempt had critical identification failures. Getting IRT right requires deep domain knowledge that pure Python/stats users may lack.

3. **The audience is small.** Legislative roll-call analysis is a niche within political science. The user base for a Python IRT package would be tiny compared to general ML tools.

---

## Convergence Diagnostics: The Modern Standard

Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021) established the [modern standard for MCMC convergence assessment](https://arxiv.org/abs/1903.08008). Their key contributions:

### Rank-Normalized R-hat

Traditional R-hat can fail to detect convergence problems when chains have heavy tails or different variances. The rank-normalized version (now the default in ArviZ and Stan) fixes this by comparing the ranks of draws rather than the raw values.

**Threshold:** R-hat < 1.01 for all parameters. This is stricter than the traditional 1.1 threshold but better calibrated for modern MCMC.

### Bulk-ESS and Tail-ESS

Traditional ESS measures mixing efficiency in the bulk of the posterior. But the tails — where credible intervals are computed — can mix poorly even when the bulk looks fine.

**Bulk-ESS** (default `az.ess()`) diagnoses sampling efficiency for central tendency estimates (posterior means, medians).

**Tail-ESS** (`az.ess(method="tail")`) diagnoses sampling efficiency for quantile estimates (HDI bounds, credible intervals). Uses the 5th and 95th percentiles.

**Threshold:** Both > 400 (100 per chain for 4 chains, 200 per chain for 2 chains).

Our implementation now checks both. Tail-ESS catches a class of problems that bulk-ESS alone misses — particularly relevant for IRT where credible interval width is a key output.

### E-BFMI

Energy Bayesian Fraction of Missing Information measures how efficiently the sampler explores the posterior's energy distribution. Low E-BFMI (< 0.3) indicates the sampler is struggling with the posterior geometry.

For IRT models, low E-BFMI typically indicates funnel-like geometry in the hierarchical structure or insufficient tuning steps.

### What We Check

| Diagnostic | Threshold | What It Catches |
|------------|-----------|-----------------|
| R-hat (rank-normalized) | < 1.01 | Mode-splitting, poor convergence |
| Bulk-ESS | > 400 | Poor mixing in posterior center |
| Tail-ESS | > 400 | Poor mixing in posterior tails |
| Divergences | < 10 | Posterior geometry problems |
| E-BFMI | > 0.3 | Energy transition inefficiency |

---

## PCA Initialization: More Than a Heuristic

PCA initialization for IRT chains is often presented as a practical convenience. It's actually a well-justified technique with theoretical backing.

### The Eigendecomposition Connection

For binary data with a logistic link, the leading eigenvector of the double-centered vote matrix approximates the maximum likelihood ideal point estimates. This is because the logistic function is approximately linear near p = 0.5, making the IRT model locally equivalent to a factor analysis — which PCA solves exactly.

Jackman (2001) used this insight to implement `startvals="eigen"` in `pscl::ideal()`. It's been the default for 25 years.

### Our Experimental Evidence

We observed that 5 of 16 chamber-sessions (31%) failed to converge with random initialization:
- 84th House, 85th Senate, 86th House, 87th Senate, 89th House
- R-hat ~ 1.83, ESS ~ 3 (catastrophic mode-splitting)

With PCA initialization: 0 of 16 failures. The fix is free (PCA is already computed in an earlier phase) and actually speeds up sampling by 10-15% because chains start closer to the posterior mode.

### When It Can Go Wrong

PCA initialization is fragile for one scenario: when PCA produces a nonsensical PC1 (e.g., if the vote matrix is too sparse or has degenerate structure). In that case, biased initialization could push chains toward a local mode rather than the global mode.

Our escape hatch: `--no-pca-init` disables initialization. The PCA-IRT correlation check (expected r > 0.95) validates that the initialization didn't distort the result.

---

## Our Implementation in Context

Our PyMC-based IRT implementation for Kansas legislative analysis makes the following design choices, each grounded in field best practices:

| Decision | Our Choice | Standard Practice | Justification |
|----------|-----------|-------------------|---------------|
| Identification | 7 strategies, auto-selected (anchor-pca, anchor-agreement, sort-constraint, etc.; ADR-0103) | Hard anchors or soft priors | Fully automated; adapts to chamber composition |
| β prior | Normal(0, 1) unconstrained | LogNormal (most) or Normal (some) | Anchors provide sign ID; unconstrained uses all data |
| Initialization | PCA PC1 scores | Eigendecomposition (pscl) or random | Standard; prevents 31% failure rate |
| Missing data | Excluded from likelihood (MAR) | Same (universal) | Only alternative is informative missingness (not needed) |
| Convergence | R-hat, bulk-ESS, tail-ESS, divergences, E-BFMI | R-hat, ESS (most don't check tail-ESS or E-BFMI) | Comprehensive; follows Vehtari et al. 2021 |
| Parameterization | Centered (flat), non-centered (hierarchical) | Same | Non-centered essential for hierarchical |
| Hierarchical ID | Ordering constraint pt.sort() | Various | Clean; avoids anchoring individual legislators |
| Cross-chamber | Test equating (discrimination ratio + bridging) | Rarely implemented | Novel addition; most implementations are single-chamber |

### Where We Exceed the Field

1. **Automated anchor selection** via PCA. Most implementations require the user to specify which legislators to anchor. We automate this with a participation guard (>50% votes).

2. **Unconstrained β with hard anchors.** The beta prior investigation documented that positive-constrained β silences 13% of contested bills. This finding is empirically validated (+3.5% accuracy, 10x ESS) and theoretically grounded, but rarely discussed in the literature.

3. **Tail-ESS in convergence checking.** Most IRT implementations check only R-hat and bulk-ESS. We now check tail-ESS per Vehtari et al. (2021), catching poor tail mixing.

4. **Cross-chamber test equating.** Placing House and Senate legislators on a common scale via discrimination ratios and bridging legislators. This is standard in educational measurement but rarely implemented for legislative analysis.

5. **External validation.** Correlation with Shor-McCarty ideology scores (r = 0.93-0.98) provides ground truth that most IRT implementations lack.

### Where the Field Has Tools We Don't

1. **Time-varying ideal points** (Martin-Quinn dynamic model). On our roadmap but not yet implemented.
2. **Informative missingness** (idealstan's hurdle model). Not needed for Kansas (2.6% absence rate).
3. **EM-based fast estimation** (emIRT). Could be useful for exploratory analysis; we use MCMC for full posteriors.
4. **Multi-dimensional IRT** (2D models). Integrated as pipeline phase 04b (ADR-0054). Custom PyMC graph with nutpie sampling, M2PL with PLT identification, both chambers. Confirms Tyson paradox is a real 2D pattern (Dim 2 vs PC2 r=0.81). No existing Python package was suitable — py-irt, girth, girth-mcmc, deepirtools all lack proper 2D identification for legislative data. See `docs/2d-irt-deep-dive.md`.

---

## Summary

The IRT identification problem has been solved multiple times by the field, but the solutions were developed for different contexts (educational testing, soft-identified models, R-based workflows) and don't always compose well. Our experience — discovering that the standard positive-constraint on β was silencing 13% of our data — illustrates a broader pattern: standard recommendations encode assumptions, and when your model doesn't share those assumptions, following the advice uncritically can quietly degrade results.

The Python ecosystem's gap in legislative IRT is real and unlikely to be filled soon. The user base is small, the identification problem is tricky, and the existing R tools are mature. Our PyMC implementation fills this gap for our specific use case and contributes at least one finding (unconstrained β with hard anchors) that the field hasn't well-documented.

---

## References

### Canonical Methodology
- Clinton, Jackman, Rivers (2004). "The Statistical Analysis of Roll Call Data." *APSR* 98: 355-370. [PDF](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf)
- Jackman (2001). "Multidimensional Analysis of Roll Call Data via Bayesian Simulation." *Political Analysis* 9(3): 227-242.
- Martin & Quinn (2002). "Dynamic Ideal Point Estimation via MCMC." *Political Analysis* 10(2): 134-153.
- Bafumi, Gelman, Park & Kaplan (2005). "Practical Issues in Implementing and Understanding Bayesian Ideal Point Estimation." *Political Analysis* 13: 171-187. [PDF](https://sites.stat.columbia.edu/gelman/research/published/171.pdf)
- Carroll et al. (2013). "Comparing NOMINATE and IDEAL." [PDF](https://scholar.princeton.edu/sites/default/files/jameslo/files/lsq_nomvsideal.pdf)
- Imai, Lo & Olmsted (2016). "Fast Estimation of Ideal Points with Massive Data." *APSR* 110: 631-656. [PDF](https://imai.fas.harvard.edu/research/files/fastideal.pdf)

### Identification
- Farouni. "Model Identification in IRT and Factor Analysis Models." [Link](https://rfarouni.github.io/assets/projects/Identification/Identification.html)
- Erosheva & Bhattacharyya (2023). "Dealing with Reflection Invariance in Bayesian Factor Analysis." [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6758924/)

### Convergence
- Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021). "Rank-Normalization, Folding, and Localization: An Improved R-hat." *Bayesian Analysis* 16(2): 667-718. [arXiv](https://arxiv.org/abs/1903.08008)
- Betancourt (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." [arXiv](https://arxiv.org/abs/1701.02434)

### Priors
- Gelman (2018). "What Prior to Use for Item-Response Parameters?" [Blog](https://statmodeling.stat.columbia.edu/2018/03/01/prior-use-item-response-parameters/)
- Stan User's Guide, IRT Models. [Link](https://mc-stan.org/docs/2_20/stan-users-guide/item-response-models-section.html)

### Implementations
- Jeffrey Arnold's "BUGS Examples in Stan" — Legislators. [Link](https://jrnold.github.io/bugs-examples-in-stan/legislators.html)
- Bob Carpenter on PyMC Discourse. [Thread](https://discourse.pymc.io/t/how-to-parameterize-and-identify-an-irt-2-pl-model-pymc-bot-test/16922)
- py-irt: [GitHub](https://github.com/nd-ball/py-irt)
- idealstan: [GitHub](https://github.com/saudiwin/idealstan)
- emIRT: [GitHub](https://github.com/kosukeimai/emIRT), [Paper](https://imai.fas.harvard.edu/research/files/fastideal.pdf)

### Interpretability
- White (2019). "Interpretational Challenges with Ideal Point Models." [Blog](https://www.johnmyleswhite.com/notebook/2019/01/20/interpretational-challenges-with-ideal-point-models/)

### Our Prior Work
- Beta prior investigation: `analysis/design/beta_prior_investigation.md`
- IRT deep dive: `docs/irt-deep-dive.md`
- Hierarchical shrinkage deep dive: `docs/hierarchical-shrinkage-deep-dive.md`
- ADR-0006: IRT implementation choices
- ADR-0023: PCA-informed chain initialization
