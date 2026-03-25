# W-NOMINATE Deep Dive

**A literature survey, ecosystem comparison, and integration analysis for NOMINATE-family legislative scaling methods.**

---

## Executive Summary

W-NOMINATE (Weighted Nominal Three-Step Estimation) is the dominant scaling method in U.S. congressional research. Developed by Poole and Rosenthal beginning in 1982, it recovers legislator ideal points from roll call votes via maximum likelihood estimation of a spatial voting model with Gaussian utility functions. This deep dive surveys the NOMINATE family's history, mathematical foundations, and full open-source ecosystem; compares NOMINATE to Bayesian IRT (which Tallgrass already implements); and evaluates integration strategies for our Kansas Legislature pipeline.

**Key findings:**

- W-NOMINATE and Bayesian IRT scores correlate at **r = 0.99** on the first dimension (Carroll et al. 2009, Clinton et al. 2004). The rank ordering of legislators is essentially identical. The methods differ meaningfully only in uncertainty quantification, model extensibility, and behavioral assumptions — not in point estimation.
- **No production-quality Python W-NOMINATE implementation exists.** The ecosystem is entirely R-based. Integration requires either `rpy2` bridging or an R subprocess call.
- The R `wnominate` package (v1.5, CRAN, actively maintained) is the canonical implementation. It consumes `pscl::rollcall()` objects and produces coordinates, standard errors (via parametric bootstrap), and fit statistics (correct classification, APRE, GMP).
- For our use case — a **single state legislature** analyzed per-biennium — W-NOMINATE's value is primarily as a **validation benchmark** and **lingua franca credential**: "our Bayesian IRT scores correlate at r = X with W-NOMINATE." The Bayesian IRT approach we already use is methodologically stronger for small chambers (Peress 2009), provides full posterior uncertainty, and extends naturally to hierarchical and dynamic variants.
- **Alpha-NOMINATE** (Carroll et al. 2013) empirically demonstrated that legislators' utility functions are nearly Gaussian (alpha ≈ 0.99), validating NOMINATE's original functional form over IRT's implied quadratic utility. This is a genuine theoretical advantage for NOMINATE, though it has negligible practical impact on point estimates.
- The R `oc` package (Optimal Classification) provides a useful **nonparametric robustness check** — if W-NOMINATE, OC, and IRT all produce similar orderings, the result is not an artifact of any method's parametric assumptions.

---

## Part 1: History and Evolution

### 1.1 The Problem NOMINATE Solved

Before NOMINATE, political scientists had limited tools for recovering metric spatial positions from binary (yea/nay) voting data. Guttman scaling and factor analysis were available but couldn't handle the inherent nonlinearity of the spatial voting model — the fact that a legislator's probability of voting Yea is a nonlinear function of their distance from the bill's policy location.

Keith T. Poole and Howard Rosenthal, then at Carnegie Mellon University, developed NOMINATE beginning in 1982. They created a database of every recorded vote in the history of the U.S. Congress — over 16 million individual roll call votes from 1789 to the present — and developed the scaling procedures to analyze it. In 2009, Poole and Rosenthal received the Society for Political Methodology's inaugural Best Statistical Software Award. Poole's 2016 Career Achievement Award citation stated that "the modern study of the U.S. Congress would be simply unthinkable without NOMINATE."

### 1.2 The NOMINATE Family

| Variant | Year | Key Innovation | Temporal | Error |
|---------|------|----------------|----------|-------|
| **NOMINATE** | 1985 | Original 1D scaling with logit errors | Static | Logistic |
| **D-NOMINATE** | 1989 | Dynamic: linear ideal point drift across Congresses | Dynamic | Logistic |
| **W-NOMINATE** | 1991 | Weighted: per-dimension salience parameters allowing elliptical indifference curves | Static | Logistic |
| **DW-NOMINATE** | 1997 | Dynamic Weighted: synthesized D- and W- capabilities; switched to normal errors | Dynamic | Normal |
| **Alpha-NOMINATE** | 2013 | Bayesian MCMC; alpha parameter mixing Gaussian (α=1) and quadratic (α=0) utility | Static | Normal (Bayesian) |

**W-NOMINATE** is the single-session variant most relevant to Tallgrass. It was written by Nolan McCarty and Keith Poole in 1991, essentially unchanged since 1997, and constrains legislators and roll call midpoints to lie within a unit hypersphere.

**DW-NOMINATE** is the cross-session variant used for temporal comparisons at the congressional level. It uses legislators who served in multiple Congresses as "bridge" observations to link coordinate systems across time. DW-NOMINATE scores are the lingua franca of congressional research.

For **state legislatures**, neither W-NOMINATE nor DW-NOMINATE solves the cross-state comparability problem — each state votes on different bills. Shor & McCarty (2011) addressed this by bridging via NPAT survey responses, using IRT rather than NOMINATE for the underlying estimation. For cross-temporal comparison within Kansas, Phase 30 applies pairwise chain linking (the same approach as Phase 28's IRT common space) to per-session W-NOMINATE Dim 1 scores, producing field-standard W-NOMINATE-scaled career scores. See `docs/wnominate-common-space.md`.

---

## Part 2: Mathematical Model

### 2.1 The Spatial Voting Framework

Each legislator *i* has an ideal point **x**_i in an *s*-dimensional Euclidean space (typically s = 1 or 2). Each roll call *j* is represented by two outcome points — a Yea outcome **z**_jy and a Nay outcome **z**_jn. Legislators vote probabilistically for the closer outcome point.

### 2.2 Utility Function

W-NOMINATE uses a Gaussian (bell-shaped) deterministic utility function:

```
U_ijy = β · exp(-w² · d²_ijy / 2) + ε_ijy
```

where:
- **d_ijy** = Euclidean distance between legislator *i*'s ideal point and the Yea outcome of roll call *j*
- **β** = signal-to-noise ratio (default: 15), weighting spatial utility against random error
- **w** (omega) = shape parameter (default: 0.5), controlling utility curvature
- **ε** = random error (logistic in W-NOMINATE, normal in DW-NOMINATE)

The Gaussian utility function is bell-shaped, symmetric, single-peaked at the ideal point, and asymptotically approaches (but never reaches) zero. This distinguishes it from the **quadratic utility** implied by standard IRT models, where utility goes to negative infinity as distance increases. Alpha-NOMINATE (Carroll et al. 2013) empirically tested this: the estimated alpha parameter across the 114th Senate averaged **0.9916**, strongly favoring Gaussian over quadratic utility.

### 2.3 Multi-Dimensional Distance

In multiple dimensions, the weighted squared distance is:

```
d²_ijk = Σ_{k=1}^{s} w_k² · (x_ik - z_jk)²
```

The dimension-specific weights **w_k** allow elliptical indifference curves. The first dimension's weight is normalized to 1.0.

### 2.4 Probability and Likelihood

The probability that legislator *i* votes Yea on roll call *j*:

```
Pr(Yea) = Pr(U_ijy > U_ijn)
```

Using the logistic link function:

```
Pr(Yea) = Λ(β · [exp(-w² · d²_ijy / 2) - exp(-w² · d²_ijn / 2)])
```

The full log-likelihood is maximized over all observed votes:

```
ℓ = Σ_{i,j} [y_ij · log P_ij + (1 - y_ij) · log(1 - P_ij)]
```

### 2.5 Three-Step Alternating Estimation

The "Nominal Three-Step Estimation" procedure:

1. **Initialize**: Eigenvalue-eigenvector decomposition of a double-centered legislator agreement score matrix provides starting coordinates (essentially PCA initialization — paralleling our `--pca-init` approach for IRT).
2. **Fix ideal points, estimate roll calls**: Hold legislator coordinates fixed. Estimate each roll call's Yea/Nay outcome parameters via weighted logistic regression.
3. **Fix roll calls, estimate ideal points**: Hold roll call parameters fixed. Nonlinear optimization (Newton-Raphson) maximizes per-legislator likelihood for each **x**_i independently.

Steps 2 and 3 alternate until convergence (typically when consecutive iterations correlate at 0.99+ or coordinate changes fall below tolerance). Convergence typically requires 10–20 iterations.

### 2.6 Identification

Spatial voting models have four identification problems: location, scale, rotation, and reflection. W-NOMINATE resolves them via:

1. **Hypersphere constraint**: All coordinates constrained to the unit hypersphere (radius = 1), resolving location and scale.
2. **Polarity fixing**: User designates reference legislators to establish sign convention (which end is "conservative"). This is analogous to our IRT anchor selection, though NOMINATE uses a single polarity reference rather than fixing two anchor legislators at ±1.
3. **Lopsided vote exclusion**: Roll calls where the minority side falls below a threshold (default 2.5%) are excluded.
4. **Minimum vote threshold**: Legislators with fewer than 20 votes are excluded.

The hypersphere constraint is a hard boundary that can cause problems at the extremes — legislators estimated at exactly ±1 are "hitting the wall." This is a known limitation that IRT's unbounded parameter space avoids.

### 2.7 Fit Statistics

W-NOMINATE produces three primary fit statistics:

- **Correct classification (CC)**: Percentage of votes correctly predicted by assigning each legislator to the closer outcome point. Typically ~83% for 1D, ~85% for 2D in modern Congresses.
- **APRE** (Aggregate Proportional Reduction in Error): Improvement over predicting every legislator votes with the majority. Accounts for the base rate.
- **GMP** (Geometric Mean Probability): The geometric mean of the predicted probabilities for the observed votes. A probabilistic measure of fit that penalizes confident wrong predictions.

---

## Part 3: Relationship to Bayesian IRT

### 3.1 The Fundamental Equivalence

The spatial voting model and IRT are deeply connected. The 1D spatial model with **quadratic utilities** maps directly onto the **two-parameter IRT model** (2PL):

- **IRT 2PL**: Pr(y=1) = Logistic(α_j + β_j · θ_i)
- **Spatial model (quadratic utility)**: algebraically reduces to the same form

where θ_i = ideal point, α_j = bill difficulty/location, β_j = bill discrimination/salience. Alpha-NOMINATE made this explicit: with α=0 (quadratic utility), the model is **identical** to standard IRT; with α=1 (Gaussian utility), it is classic NOMINATE.

### 3.2 Key Differences

| Feature | W-NOMINATE | Bayesian IRT (Tallgrass) |
|---------|-----------|--------------------------|
| **Utility function** | Gaussian (bell-shaped) | Quadratic (implied by 2PL logistic) |
| **Estimation** | Maximum likelihood (alternating) | MCMC (nutpie Rust NUTS sampler) |
| **Uncertainty** | Parametric bootstrap (post-hoc, slow) | Full posterior distributions (inherent) |
| **Identification** | Hypersphere + polarity legislator | Hard anchors at ±1 (PCA-selected extremes) |
| **Multi-dimensional** | Explicit 2D parameterization | Rotation-invariant; requires PLT identification |
| **Covariates** | Not easily incorporated | Natural via priors (hierarchical by party) |
| **Initialization** | PCA of agreement matrix | PCA PC1 scores (same idea, our `--pca-init`) |
| **Model comparison** | Classification accuracy, APRE | Posterior predictive checks, WAIC/LOO |

### 3.3 Empirical Correlations

The literature consistently finds near-perfect first-dimension agreement:

| Study | Data | Correlation |
|-------|------|-------------|
| Clinton, Jackman & Rivers (2004) | 106th U.S. House | r ≈ 0.99 |
| Carroll et al. (2009) | 109th Senate, Monte Carlo | r ≈ 0.99 |
| Carroll et al. (2009) | IDEAL vs MCMC-NOMINATE | r ≈ 0.99 |

The near-perfect correlation means the choice between methods is driven by what you need **beyond** point estimates — uncertainty quantification, model extension, computational convenience — not by accuracy differences.

### 3.4 Where They Diverge

Meaningful differences arise in specific circumstances:

- **Extreme legislators**: NOMINATE's Gaussian utility penalizes large "voting errors" less harshly than IRT's quadratic utility. Legislators at the ideological extremes may receive different placements.
- **Uncertainty at the poles**: NOMINATE reports greater certainty about extremists (they're pinned to the hypersphere boundary). IRT may report greater certainty about centrists (more information from cross-cutting votes). These differences are artifacts of identification strategies, not fundamental model properties (Carroll et al. 2009).
- **Small chambers**: With fewer legislators and roll calls, parametric assumptions matter more. Peress (2009) found "no clear advantage of one method over the other" in Monte Carlo simulations for small chambers, but Bayesian regularization via priors provides natural shrinkage that prevents implausible estimates.

### 3.5 The Alpha-NOMINATE Verdict

Carroll et al. (2013) tested whether real legislators have Gaussian or quadratic utility. Their alpha-NOMINATE model estimates a per-legislator mixing parameter: α = 1 is Gaussian (NOMINATE's assumption), α = 0 is quadratic (IRT's assumption). For the 114th Senate, the mean alpha was **0.9916** — overwhelmingly Gaussian. This is a genuine theoretical point in NOMINATE's favor, though the practical impact on ideal point estimation is negligible given the r = 0.99 correlation.

---

## Part 4: The Software Ecosystem

### 4.1 Roll Call Data Infrastructure

The `pscl` R package (Simon Jackman, v1.5.9, actively maintained on CRAN) provides the `rollcall()` function that creates the data structure consumed by every major scaling package. This is the lingua franca format:

```r
rc <- rollcall(
  data = vote_matrix,      # legislators × votes matrix
  yea = c(1, 2, 3),        # codes meaning "yea"
  nay = c(4, 5, 6),        # codes meaning "nay"
  missing = c(7, 8, 9),    # codes meaning missing/abstain
  notInLegis = 0,
  legis.names = names,
  legis.data = metadata_df
)
```

### 4.2 The R Ecosystem

| Package | Version | Method | Speed | Uncertainty | CRAN | Maintained |
|---------|---------|--------|-------|-------------|------|-----------|
| `wnominate` | 1.5 | ML (W-NOMINATE) | Fast | Bootstrap SE | Yes | Yes |
| `oc` | 1.2.1 | Nonparametric (OC) | Fast | None | Yes | Yes |
| `anominate` | 0.7 | Bayesian (Alpha-NOM) | Slow | Posterior | Yes | Yes |
| `pscl` | 1.5.9 | Bayesian IRT (Gibbs) | Slow | Posterior | Yes | Yes |
| `MCMCpack` | 1.7-1 | Bayesian IRT (Gibbs) | Slow | Posterior | Yes | Yes |
| `emIRT` | 0.0.13 | EM algorithm (IRT) | Very fast | Point only | Yes | Yes |
| `idealstan` | 0.7.2 | Bayesian (Stan HMC) | Moderate | Posterior | Yes | Yes |
| `dwnominate` | 1.2 | ML (DW-NOMINATE) | Fast | Bootstrap SE | GitHub | Semi |
| `basicspace` | 0.25 | SVD/ALS (survey data) | Fast | None | Yes | Yes |
| `pgIRT` | — | Polya-Gamma EM | Fast | Bootstrap | GitHub | Semi |

**W-NOMINATE** (`wnominate`): The canonical implementation. Key function:

```r
result <- wnominate(rc, dims = 2, polarity = c(1, 5), minvotes = 20, lop = 0.025)
```

Returns `result$legislators` (coordinates, SEs, fit) and `result$rollcalls` (midpoints, spreads, classification). The `polarity` parameter is **critical** — it specifies which legislator index anchors the sign convention per dimension. Without it, the orientation is arbitrary.

**Optimal Classification** (`oc`): Poole's (2000) nonparametric alternative. Makes no distributional assumptions — simply finds the legislator arrangement that maximizes correct classification. Same `rollcall` input, same `polarity` requirement. Useful as a robustness check: if OC and NOMINATE produce the same ordering, the result is not an artifact of parametric assumptions.

**Alpha-NOMINATE** (`anominate`): Bayesian MCMC version with the Gaussian-vs-quadratic utility test. Computationally expensive but answers the theoretical question about utility function shape.

### 4.3 Python Implementations

**There is no production-quality Python W-NOMINATE implementation.** The ecosystem:

| Project | Status | What It Actually Does |
|---------|--------|----------------------|
| `pypscl` (twneale) | Dead (2013-era pandas) | rpy2 wrapper around pscl/wnominate |
| `dw-nominate` (jeremyjbowers) | Parser only | Reads pre-computed Voteview scores |
| `tbip` (keyonvafa) | Academic code | Text-based ideal points (not roll call) |
| `Jnotype` (cbg-ethz) | Biological focus | JAX-based binary IRT (genotyping, not legislative) |

The practical options for Python integration are:

1. **`rpy2` bridge**: Call R's `wnominate` from Python. Technically feasible; adds R as a runtime dependency. No published examples of this specific integration exist.
2. **R subprocess**: Write CSV → call `Rscript` → read CSV. Simpler, looser coupling. Our dynamic IRT phase already has an R script (`analysis/27_dynamic_irt/dynamic_irt_emirt.R`) using this pattern.
3. **Native Python implementation**: Would require porting the alternating optimization algorithm. Significant effort with diminishing returns given r = 0.99 agreement with our existing IRT.

### 4.4 Voteview.com

Poole's website (maintained by UCLA) provides pre-computed DW-NOMINATE scores for all U.S. Congresses:

- `HSall_members.csv` — NOMINATE scores, congress, chamber, party, `nominate_dim1`, `nominate_dim2`
- `HSall_rollcalls.csv` — Roll call metadata
- `HSall_votes.csv` — Individual vote records

**Important**: Voteview covers U.S. Congress only, not state legislatures. Not directly useful for Kansas data, but relevant if we ever want to validate against congressional scores or bridge state-federal comparisons.

### 4.5 Helper Packages

- **`Rvoteview`** (GitHub: voteview/Rvoteview): Queries Voteview API; returns `rollcall` objects compatible with `wnominate`.
- **`wnomadds`** (GitHub: jaytimm/wnomadds): ggplot2 visualization helpers for wnominate results (cutlines, angles).
- **`politicaldata`** (CRAN): Convenience wrappers for downloading DW-NOMINATE scores.

---

## Part 5: State Legislature Considerations

### 5.1 The Cross-State Problem

Each state legislature votes on different bills, making W-NOMINATE scores incomparable across states. Shor & McCarty (2011) solved this by using NPAT survey responses as bridging observations. Their methodology uses IRT (not NOMINATE) for the within-state step, partly because IRT's Bayesian framework more naturally accommodates the hierarchical structure and incomplete data inherent in state-level applications.

### 5.2 Small Chamber Issues

Kansas has 40 senators and 125 representatives. Small chambers create specific challenges:

- **The incidental parameters problem**: With many bill parameters estimated as fixed effects relative to few legislators, ideal point estimates can be biased. Bayesian regularization via priors (which Tallgrass already uses) provides natural shrinkage.
- **W-NOMINATE constraints**: The constraint that extreme legislators be no more than 0.1 units from their nearest neighbor is not applied for chambers with fewer than 20 members. The Kansas Senate (40 members) is above this threshold but still small enough that parametric assumptions have outsized influence.
- **Peress (2009)**: Monte Carlo simulations found "no clear advantage of one method over the other" for small chambers, but noted that hierarchical Bayesian IRT models offer the most principled approach to regularization.

### 5.3 High Polarization and Supermajorities

Kansas features high party polarization and Republican supermajorities (often 72%+ of seats). This creates:

- **Separation/perfect prediction**: When parties vote as blocks, discrimination parameters diverge toward infinity in MLE. Bayesian priors on β prevent this; W-NOMINATE's bounded parameter space handles it through the hypersphere constraint.
- **Within-party resolution**: The substantively interesting variation is within the Republican majority. Both methods can recover within-party ordering, but IRT with hierarchical party priors (our Phase 10) does so more stably.

### 5.4 What We Already Have

Tallgrass already implements the Bayesian IRT approach that Shor & McCarty themselves use for the within-state step. Our external validation (Phase 14) correlates our scores against Shor-McCarty data. Adding W-NOMINATE provides a second methodological validation point — convergent validity from a fundamentally different estimation approach.

---

## Part 6: Criticisms and Limitations of NOMINATE

### 6.1 The Mismeasurement Critique

Judge Glock's "The Mismeasurement of Polarization" (National Affairs) raises fundamental objections to DW-NOMINATE's cross-temporal claims:

- **Shifting baseline**: DW-NOMINATE treats the ideological center as static, but the policy landscape has shifted dramatically. Federal spending was ~2% of GDP in 1900 vs. ~20% by 2007, yet NOMINATE scores for "conservative" legislators look similar across eras.
- **Historical anachronism**: Late-19th-century Democrats scored as "liberal" despite supporting low taxes and opposing civil rights.
- **Patronage vs. ideology**: Frances Lee's research demonstrates that extreme 19th-century polarization reflected patronage voting, not ideological division.

**Relevance to Tallgrass**: These criticisms target cross-era comparability, which we don't attempt. Within a single Kansas biennium, they are largely moot.

### 6.2 Party Discipline vs. Ideology

NOMINATE cannot distinguish genuine ideological conviction from party whipping, strategic behavior, or agenda control. High party-line voting could reflect any of these. This limitation applies equally to all roll-call-based methods including IRT.

### 6.3 The ~15% Misclassification Floor

Even with two dimensions, NOMINATE misclassifies approximately 15% of votes. This residual is attributed to measurement error, non-policy factors (logrolling, constituency service), or violations of the spatial assumption. Some scholars argue this residual contains substantive information about party pressure that the model treats as noise.

### 6.4 Dimensional Collapse

Since 1987, the second dimension in U.S. Congressional data has become negligible (~2% additional variance). Post-polarization, 1D fits better than 2D. For Kansas, our PCA already shows PC1 captures ~57% of variance with PC2 at a much lower level. The empirical question is whether a second W-NOMINATE dimension adds anything beyond what our 2D IRT (Phase 06) already captures.

---

## Part 7: How We Compare to Best Practices

### 7.1 Ideal Point Estimation

**Assessment: Strong.** Our Bayesian 2PL IRT with PCA-informed initialization, hard anchors, nutpie NUTS sampling, and comprehensive convergence diagnostics (R-hat < 1.01, ESS > 400, tail-ESS > 400) follows best practices from Clinton, Jackman & Rivers (2004) and the Stan IRT literature. The unconstrained beta prior is more principled than NOMINATE's bounded parameter space. PCA initialization mirrors NOMINATE's own eigendecomposition starting values.

### 7.2 Uncertainty Quantification

**Assessment: Superior to NOMINATE.** Full posterior distributions with credible intervals, posterior predictive checks, and uncertainty propagation to downstream phases. NOMINATE's parametric bootstrap is adequate but computationally expensive and provides only frequentist confidence intervals.

### 7.3 Hierarchical Structure

**Assessment: Beyond NOMINATE's capability.** Our hierarchical IRT (Phase 10) with party-level partial pooling has no NOMINATE analogue. This is a genuine advantage for state legislatures with small party caucuses.

### 7.4 Multi-Dimensionality

**Assessment: Comparable.** Our 2D IRT (Phase 06) with PLT identification is methodologically more principled than NOMINATE's polarity-based 2D identification, though our convergence thresholds are relaxed (R-hat < 1.05, ESS > 200) reflecting the experimental status of this phase.

### 7.5 External Validation

**Assessment: Could be stronger.** We validate against Shor-McCarty (r > 0.93 for flat IRT). Adding W-NOMINATE as a second benchmark would strengthen the convergent validity argument: "three independent methods — Bayesian IRT, W-NOMINATE, and Shor-McCarty — produce consistent legislator orderings."

---

## Part 8: Integration Analysis for Tallgrass

### 8.1 What W-NOMINATE Would Add

1. **Convergent validity**: r = 0.99 expected correlation with our IRT, confirming our scores against the field standard.
2. **Publication credibility**: "Our IRT scores correlate at r = X with W-NOMINATE" is a sentence political scientists trust immediately.
3. **Fit statistics**: Correct classification, APRE, and GMP provide model fit metrics distinct from IRT's posterior predictive checks.
4. **Scree plot**: NOMINATE's eigenvalue decomposition provides a principled dimensionality assessment independent of our PCA.
5. **Nonparametric validation**: If we also run OC, three-way agreement (IRT/NOMINATE/OC) provides very strong evidence for the recovered ideological dimension.

### 8.2 What It Would Not Add

1. **Better point estimates**: r = 0.99 correlation means negligible improvement.
2. **Better uncertainty**: Parametric bootstrap is inferior to our full posteriors.
3. **New substantive findings**: The same data, similar model, near-identical results.
4. **Cross-session comparability**: DW-NOMINATE solves this for Congress but not for state legislatures (different bills each session).

### 8.3 Implementation Options

**Option A: rpy2 Bridge (Recommended)**

Call R's `wnominate` directly from Python via `rpy2`. Tighter integration, same process, programmatic access to all result objects.

```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

pscl = importr('pscl')
wnom = importr('wnominate')

rc = pscl.rollcall(vote_matrix_r, yea=1, nay=0, missing=ro.NA_Integer)
result = wnom.wnominate(rc, dims=2, polarity=ro.IntVector([polarity_idx]))
coords = result.rx2('legislators')
```

Pros: Full access to R object structure, in-process, no intermediate files.
Cons: R dependency, rpy2 complexity, error messages span two runtimes.

**Option B: R Subprocess**

Write vote matrix to CSV, call `Rscript`, read results back. Matches our existing `emIRT` R script pattern in Phase 16.

```bash
Rscript analysis/XX_wnominate/wnominate.R input.csv output.csv house 2
```

Pros: Simple, debuggable, no rpy2 dependency, precedent in codebase.
Cons: I/O overhead (negligible for our data sizes), less programmatic control.

**Option C: Both via Shared R Infrastructure**

Establish `rpy2` as the standard R bridge (needed eventually for Bai-Perron confidence intervals in TSA hardening). Use it for W-NOMINATE, OC, and any future R-dependent methods.

### 8.4 Pipeline Integration

W-NOMINATE would slot as a new pipeline phase alongside Phase 04 (IRT):

- **Input**: EDA-filtered vote matrices (same as IRT) — `vote_matrix_{chamber}_filtered.parquet`
- **Output**: `wnominate_points_{chamber}.parquet` with columns matching IRT format: `legislator_slug`, `xi_mean` (coord1D), `xi_sd` (se1D), `full_name`, `party`, `chamber`, `district`
- **Additional output**: Fit statistics (CC, APRE, GMP), scree plot, cutline visualization, coordinate scatter
- **Downstream**: Synthesis (Phase 11) can consume W-NOMINATE as an alternative or supplementary set of ideal points. External validation (Phase 14) can correlate W-NOMINATE against Shor-McCarty alongside IRT. Cross-session (Phase 13) can test whether W-NOMINATE vs IRT stability differs.

### 8.5 OC as a Bonus

The `oc` package uses the same `rollcall` input format and `polarity` interface. If we establish R infrastructure for W-NOMINATE, adding Optimal Classification is trivial — one additional function call per chamber. The three-way comparison (IRT/W-NOMINATE/OC) provides the strongest possible convergent validity evidence.

---

## Part 9: The Broader Ecosystem at a Glance

### 9.1 Method Comparison Matrix

| Method | Approach | Speed | Uncertainty | Dimensions | Dynamic | Our Status |
|--------|----------|-------|-------------|------------|---------|------------|
| **W-NOMINATE** | ML, Gaussian utility | Seconds | Bootstrap SE | 1–10 | No | Planned |
| **DW-NOMINATE** | ML, dynamic Gaussian | Minutes | Bootstrap SE | 1–10 | Yes (linear) | Not needed (single state) |
| **OC** | Nonparametric | Seconds | None | 1–10 | No | Planned (with W-NOMINATE) |
| **Bayesian IRT (2PL)** | MCMC, quadratic utility | Minutes | Full posterior | 1–2 | No | **Implemented** (Phase 04) |
| **Hierarchical IRT** | MCMC, party priors | Minutes | Full posterior | 1 | No | **Implemented** (Phase 10) |
| **2D IRT (M2PL)** | MCMC, PLT identification | Minutes | Full posterior | 2 | No | **Implemented** (Phase 06) |
| **Dynamic IRT** | MCMC, random walk | Hours | Full posterior | 1 | Yes (random walk) | **Implemented** (Phase 16) |
| **emIRT** | EM algorithm | Seconds | Point only | 1 | Yes (variational) | Available via R script |
| **Alpha-NOMINATE** | Bayesian MCMC | Hours | Posterior | 1–2 | No | Lower priority |
| **idealstan** | Stan HMC, multiple models | Hours | Posterior | K | Yes (RW/AR/GP) | Not needed |

### 9.2 What the Canonical References Say

**Clinton, Jackman & Rivers (2004)**: "When N and J are both reasonably large and a low-dimensional model fits the data well, there is extremely little difference in the ideal point estimates." Their argument for Bayesian IRT is about flexibility and uncertainty, not accuracy.

**Carroll et al. (2009)**: "With the development of MCMC versions of NOMINATE, the differences between estimators are likely to get smaller and the choice between estimators will become one of convenience and taste."

**McCarty (2010)**: The Bayesian approach is "more appropriate for smaller legislatures." Hierarchical priors provide regularization that prevents the incidental parameters problem.

**Peress (2009)**: "Although most existing ideal point estimators perform well when N and J are both large," small chambers require specialized attention. No method has a clear advantage for point estimation, but Bayesian regularization is the most principled approach.

**Oxford Bibliographies (Ideal Point Estimation)**: "Studies weighing up NOMINATE and Bayesian IRT estimation have typically preferred the latter."

### 9.3 The Consensus View

For a single state legislature analyzed per-biennium:

1. **Bayesian IRT is the methodologically stronger choice** — full posteriors, hierarchical priors for small caucuses, natural extensibility.
2. **W-NOMINATE is the credibility benchmark** — the sentence "our scores correlate at r = X with W-NOMINATE" carries immediate weight in political science.
3. **The two methods are complements, not competitors** — different estimation philosophies recovering the same underlying structure.

---

## Part 10: Recommendations for Tallgrass

### 10.1 Implement W-NOMINATE as a Validation Phase

Add W-NOMINATE as a new pipeline phase (not a replacement for IRT). Primary purpose: convergent validity and publication credibility. Secondary purpose: fit statistics and dimensionality assessment.

### 10.2 Use R Subprocess (Initially), rpy2 (Eventually)

Start with the R subprocess pattern already established in Phase 16 (`dynamic_irt_emirt.R`). This minimizes integration risk. Migrate to rpy2 when the Bai-Perron TSA hardening work (roadmap item #7) justifies establishing a proper R bridge.

### 10.3 Bundle OC with W-NOMINATE

Since `oc` uses the same `rollcall` input and `polarity` interface, add it in the same phase. The three-way IRT/W-NOMINATE/OC comparison table is a powerful validation artifact.

### 10.4 Automate Polarity Selection

The `polarity` parameter requires specifying which legislator anchors the conservative end. Automate this by using the same PCA-based anchor selection we already use for IRT: the legislator with the highest PC1 score (most conservative by PCA) becomes the polarity reference.

### 10.5 Output Format Compatibility

Produce `wnominate_points_{chamber}.parquet` with the same column schema as IRT (`legislator_slug`, `xi_mean`, `xi_sd`, etc.) so downstream phases can consume W-NOMINATE scores without modification.

### 10.6 Report the Three-Way Correlation

The headline number from this phase should be a 3×3 correlation matrix: IRT vs W-NOMINATE vs OC (and vs Shor-McCarty for overlapping bienniums). This is the strongest convergent validity evidence available.

---

## References

- Carroll, R., Lewis, J.B., Lo, J., Poole, K.T., & Rosenthal, H. (2009). Comparing NOMINATE and IDEAL: Points of Difference and Monte Carlo Tests. *Legislative Studies Quarterly*, 34(4), 555–591.
- Carroll, R., Lewis, J.B., Lo, J., Poole, K.T., & Rosenthal, H. (2013). The Structure of Utility in Spatial Models of Voting. *American Journal of Political Science*, 57(4), 1008–1028.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The Statistical Analysis of Roll Call Data. *American Political Science Review*, 98(2), 355–370.
- Glock, J. (2020). The Mismeasurement of Polarization. *National Affairs*.
- Imai, K., Lo, J., & Olmsted, J. (2016). Fast Estimation of Ideal Points with Massive Data. *American Political Science Review*, 110(4), 631–656.
- Lewis, J.B. & Poole, K.T. (2004). Measuring Bias and Uncertainty in Ideal Point Estimates via the Parametric Bootstrap. *Political Analysis*, 12(2), 105–127.
- Lo, J., Poole, K., Lewis, J., & Carroll, R. (2011). Scaling Roll Call Votes with wnominate in R. *Journal of Statistical Software*, 42(14), 1–21.
- McCarty, N. (2010). Measuring Legislative Preferences. Princeton manuscript.
- McCarty, N., Poole, K.T., & Rosenthal, H. (2006). *Polarized America: The Dance of Ideology and Unequal Riches*. MIT Press.
- Peress, M. (2009). Small Chamber Ideal Point Estimation. *Political Analysis*, 17(1), 276–290.
- Poole, K.T. (2000). Non-parametric Unfolding of Binary Choice Data. *Political Analysis*, 8(3), 211–237.
- Poole, K.T. (2008). NOMINATE: A Short Intellectual History. Working paper.
- Poole, K.T. & Rosenthal, H. (1985). A Spatial Model for Legislative Roll Call Analysis. *American Journal of Political Science*, 29(2), 357–384.
- Poole, K.T. & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Remmel, M.L. & Mondak, J.J. (2020). Three Validation Tests of the Shor-McCarty State Legislator Ideology Data. *American Politics Research*, 48(5), 577–588.
- Shin, S. (2024). L1-based Bayesian Ideal Point Model for Multidimensional Politics. *Journal of the American Statistical Association*, 120(550).
- Shor, B. & McCarty, N. (2011). The Ideological Mapping of American Legislatures. *American Political Science Review*, 105(3), 530–551.
