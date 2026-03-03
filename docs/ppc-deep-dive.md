# Posterior Predictive Checks Deep Dive

**A literature survey, ecosystem comparison, code audit, and implementation recommendations for Bayesian model validation and comparison in the Tallgrass IRT pipeline.**

---

## Executive Summary

Phase 4c (Standalone Posterior Predictive Checks) adds unified model validation and comparison across the three IRT variants in the Tallgrass pipeline: flat 1D IRT, hierarchical IRT, and 2D IRT. This deep dive surveys the academic literature on posterior predictive model checking (PPMC) for IRT models, evaluates the software ecosystem across Python and R, audits our existing partial implementation, and proposes a concrete implementation plan.

**Key findings:**

- Our Phase 04 (flat IRT) already implements a **basic PPC** — overall Yea rate and classification accuracy with 500 replications. Phases 04b (2D IRT) and 10 (hierarchical) have **no PPC code at all**.
- The existing `run_ppc()` function in Phase 04 uses manual posterior sampling rather than `pm.sample_posterior_predictive()`. This is actually more flexible but misses **ArviZ's information-criterion tools** (`az.loo()`, `az.compare()`), which require the `log_likelihood` group in InferenceData.
- **nutpie does not compute log-likelihood** during sampling. A post-hoc `pm.compute_log_likelihood()` call is required — this is the primary integration hurdle.
- The literature strongly recommends **item-pair odds ratios** (Sinharay et al. 2006) as the most powerful discrepancy measure for detecting local dependence and multidimensionality in IRT — precisely the question we need answered for flat vs. 2D comparison.
- **PSIS-LOO** (Vehtari et al. 2017, updated 2024) is preferred over WAIC for model comparison because it provides Pareto k diagnostics. However, for IRT models the **conditional vs. marginal likelihood distinction** (Merkle et al. 2019) matters: standard PSIS-LOO answers "how well does the model predict one vote given we know the legislator's ability?" rather than the full marginal question. This is still useful for comparing model parameterizations (flat vs. hierarchical, 1D vs. 2D).
- **PPP-values are conservative** (cluster around 0.5 under the true model) and should not be interpreted as frequentist p-values. Graphical displays are more informative than numeric summaries.
- There is **no Python package** that computes IRT-specific fit statistics (S-X2, Q3, person-fit). These exist only in R (mirt, sirt). We can compute the most useful ones (Q3, ICC residuals) directly from posterior samples with minimal code.

---

## Part 1: The Problem

The Tallgrass pipeline now has three IRT model variants, each making different assumptions:

| Model | Phase | Parameterization | Assumptions |
|-------|-------|-----------------|-------------|
| **Flat 1D IRT** | 04 | 2PL with hard anchors | One dimension, exchangeable priors |
| **Hierarchical 1D IRT** | 10 | 2PL with party-level pooling | One dimension, party structure informs priors |
| **2D IRT** | 04b | M2PL with PLT identification | Two dimensions, exchangeable priors |

All three produce ideal point estimates. External validation (Shor-McCarty correlations in Phase 14, W-NOMINATE/OC comparison in Phase 17) confirms they agree with independent data sources. But we lack **internal validation** — does each model reproduce the patterns in its own data? And we lack **formal model comparison** — which model fits best after accounting for complexity?

Phase 04 has a basic PPC (Yea rate + accuracy), but it answers only the coarsest question. A complete PPC battery would tell us:

1. **Calibration**: Does the model reproduce item-level and person-level patterns, not just the global mean?
2. **Local independence**: Are there item pairs whose association the model cannot explain (suggesting a missing dimension)?
3. **Person fit**: Are there legislators whose voting patterns the model systematically mispredicts?
4. **Model selection**: Which model provides the best predictive accuracy per parameter?

---

## Part 2: Academic Foundations

### 2.1 Posterior Predictive Model Checking (PPMC)

**Gelman, Meng & Stern (1996).** "Posterior Predictive Assessment of Model Fitness via Realized Discrepancies." *Statistica Sinica*, 6(4), 733-807. The foundational paper. The key idea: draw parameter values from the posterior, simulate replicated datasets, compute a test quantity T(y) on both observed and replicated data, and compare. The posterior predictive p-value (PPP-value) is P(T(y_rep) >= T(y_obs) | y). Graphical checks are emphasized over numeric summaries.

**Sinharay, Johnson & Stern (2006).** "Posterior Predictive Assessment of Item Response Theory Models." *Applied Psychological Measurement*, 30(4), 298-321. The definitive paper on PPMC for IRT. After systematic simulation studies, they recommend:
- **Item-pair odds ratios** as the most powerful discrepancy measure across all types of model misfit (local dependence, multidimensionality, wrong number of parameters).
- **Observed score distributions** for global model fit.
- **Biserial correlation coefficients** for item discrimination assessment.

**Levy & Mislevy (2016).** *Bayesian Psychometric Modeling.* Chapman & Hall/CRC. Chapter 11 covers PPMC for IRT in detail, using Bayesian chi-square as the primary discrepancy measure. Provides a unified treatment across IRT, CFA, and latent class models.

**Fox (2010).** *Bayesian Item Response Modeling.* Springer. Covers log odds ratios and Bayesian latent residuals. Notes that prior predictive checks may be more sensitive than posterior predictive checks for detecting local independence violations.

**Beguin & Glas (2001).** "MCMC Estimation and Some Model-Fit Analysis of Multidimensional IRT Models." *Psychometrika*, 66(4), 541-561. Early PPMC application: chi-square statistics and test score distribution plots with 95% credible intervals.

### 2.2 Standard PPC Test Statistics for IRT

#### Item-Level Statistics

| Statistic | What It Checks | Implementation Complexity |
|-----------|---------------|--------------------------|
| **Item endorsement rates** | Model reproduces per-item pass rates | Low — direct comparison of observed vs. replicated means per roll call |
| **Biserial/point-biserial correlations** | Item discrimination | Medium — compute item-total correlations on replicated data |
| **Item-pair odds ratios** | Local dependence / multidimensionality | Medium — 2x2 contingency tables for all item pairs |
| **ICC residuals** | Item fit across ability range | Medium — bin legislators by ability, compare observed vs. predicted proportions |

#### Person-Level Statistics

| Statistic | What It Checks | Implementation Complexity |
|-----------|---------------|--------------------------|
| **Total score distribution** | Model reproduces individual Yea counts | Low — histogram comparison |
| **Per-legislator Yea rates** | Individual calibration | Low — compare observed vs. replicated legislator means |
| **lz person-fit** (Drasgow 1985) | Aberrant response patterns | Medium — standardized log-likelihood per response vector |
| **lz* corrected** (Snijders 2001) | Corrected person fit | High — requires estimated-parameter correction |

#### Global Statistics

| Statistic | What It Checks | Implementation Complexity |
|-----------|---------------|--------------------------|
| **Overall Yea rate** | Base rate calibration | Low — **already implemented** |
| **Vote margin distribution** | Contestedness patterns | Low — compare margin histograms |
| **Classification accuracy** | Predictive power | Low — **already implemented** |
| **APRE** | Proportional reduction in error | Low — controls for lopsided margins |
| **GMP** | Geometric mean probability | Low — more robust than classification accuracy |

#### Pairwise / Local Dependence Statistics

| Statistic | What It Checks | Notes |
|-----------|---------------|-------|
| **Yen's Q3** (1984) | Local independence via residual correlations | Most common; no single critical value — Christensen et al. (2017) recommend bootstrapping |
| **Chen-Thissen X2/G2** | Local dependence from item-pair contingency tables | More sensitive than Q3 for surface-level LD |
| **Item-pair odds ratios** | Association beyond what model explains | **Most powerful** per Sinharay et al. (2006) |

### 2.3 Information Criteria for Model Comparison

**PSIS-LOO** (Vehtari, Gelman & Gabry 2017; updated Vehtari et al. 2024, JMLR 25). Leave-one-out cross-validation approximated via Pareto-smoothed importance sampling. Preferred over WAIC because it provides the Pareto k diagnostic indicating when the approximation fails.

**Pareto k thresholds** (updated 2024, sample-size dependent):

| Condition | Meaning | Action |
|-----------|---------|--------|
| k < min(1 - 1/log10(S), 0.7) | Reliable | None needed |
| 0.7 <= k < 1 | Large bias | Consider K-fold CV or model change |
| k >= 1 | Non-finite mean | Likely model misspecification |

With S=4000 posterior draws (our typical setup), the effective threshold is 0.7.

**The conditional vs. marginal problem** (Merkle, Furr & Rabe-Hesketh 2019, *Psychometrika*). Standard PSIS-LOO from PyMC conditions on latent abilities — it answers "how well does the model predict a single vote, given we know the legislator's ability?" Marginal LOO (integrating out abilities) answers the arguably more relevant question "how well does the model predict all votes for a new legislator?" but requires analytic integration or quadrature that is often intractable. In practice, conditional LOO is standard and still informative for comparing model parameterizations.

**WAIC** is asymptotically equivalent to PSIS-LOO but lacks the Pareto k diagnostic. Not recommended as the primary tool.

### 2.4 PPP-Value Calibration Issues

PPP-values are **not uniformly distributed under the true model** — they concentrate around 0.5, making them conservative (low power). This was noted by Gelman et al. (1996) and extensively studied since.

Practical implications:
- A PPP-value of 0.05 does **not** correspond to frequentist alpha=0.05 rejection.
- Values between 0.10 and 0.90 are routinely interpreted as "acceptable" but this is convention.
- Graphical displays are more informative than numeric summaries.
- **LOO-PIT** (LOO Probability Integral Transform) directly addresses the "double use of data" concern by approximating leave-one-out cross-validation.

Gelman's recommendation: use graphical displays and interpret PPP-values as rough summaries, not formal tests. "The goal is to find *where* the model fails, not to compute a rejection threshold."

### 2.5 PPC in Political Science IRT

**Clinton, Jackman & Rivers (2004).** "The Statistical Analysis of Roll Call Data." *APSR*, 98(2), 355-370. The foundational Bayesian IRT paper for legislative ideal points. Model fit assessed via classification rates (percent correctly predicted votes) and posterior predictive score distributions.

**Jackman (2001).** "Multidimensional Analysis of Roll Call Data via Bayesian Simulation." *Political Analysis*, 9(3), 227-241. Model checking section includes proportion correctly predicted and geometric mean probability (GMP). GMP is more robust than classification rate because it penalizes confident wrong predictions.

**Joo, Lee, Park & Stark (2023).** "Assessing Dimensionality of the Ideal Point IRT Model Using Posterior Predictive Model Checking." *Organizational Research Methods*. Uses PPMC to evaluate whether a unidimensional ideal point model is adequate — directly relevant to our flat vs. 2D comparison.

**What political scientists actually report:**

| Metric | Description | Typical Values |
|--------|-------------|----------------|
| PCP | Percent correctly predicted | 85-95% for legislative data |
| APRE | Aggregate proportional reduction in error (controls for margins) | 0.5-0.7 for NOMINATE |
| GMP | Geometric mean probability | 0.7-0.8 typical |
| PPP-value | Proportion of replications exceeding observed statistic | Near 0.5 = good |

The Stan 2PL tutorial uses a "mixed" PPMC approach: item parameters from posterior, but ability parameters from the *prior* (not posterior), reducing the conservatism of standard PPP-values. This is an important practical innovation that addresses double-use concerns.

---

## Part 3: Software Ecosystem

### 3.1 ArviZ (Python)

ArviZ (current stable: 0.23.x) provides the core PPC toolkit:

| Function | Purpose | Binary IRT Notes |
|----------|---------|-----------------|
| `az.plot_ppc()` | KDE/histogram overlay of observed vs. replicated | Degenerate on raw binary data — must aggregate first |
| `az.loo()` | PSIS-LOO-CV with Pareto k diagnostics | Requires `log_likelihood` group in InferenceData |
| `az.waic()` | WAIC computation | Same requirement; less robust than LOO |
| `az.compare()` | Model comparison table (ELPD, weights) | Takes dict of InferenceData objects |
| `az.plot_compare()` | Visualize model comparison with error bars | Built on `az.compare()` output |
| `az.plot_loo_pit()` | LOO Probability Integral Transform | Addresses double-use concern |
| `az.plot_khat()` | Pareto k diagnostic plot | Identifies influential observations |

**Key limitation for binary data**: `az.plot_ppc()` with KDE on binary {0,1} data produces degenerate density plots. For binary responses, aggregation is required first (e.g., plot distributions of total scores, item proportions, or summary statistics across replications).

### 3.2 PyMC + nutpie Integration

**The nutpie log-likelihood gap**: nutpie does not compute or store element-wise log-likelihood during sampling, even when `idata_kwargs={"log_likelihood": True}` is passed. The workaround:

```python
# After nutpie sampling
pm.compute_log_likelihood(idata, model=model)
```

This is a post-hoc computation that adds overhead but produces the `log_likelihood` group needed by `az.loo()` and `az.waic()`.

**Posterior predictive sampling**: nutpie does not provide forward sampling. After sampling with nutpie, use the original PyMC model context:

```python
# After nutpie sampling
with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
```

**Practical workflow for Phase 4c**:
1. Load saved InferenceData from upstream phases (04, 04b, 10)
2. Rebuild the PyMC model graph (using existing `build_irt_graph()`, `build_per_chamber_graph()`)
3. Compute log-likelihood: `pm.compute_log_likelihood(idata, model=model)`
4. Generate PPC samples: `pm.sample_posterior_predictive(idata, model=model)`
5. Run diagnostics: `az.loo(idata)`, custom PPC statistics, `az.compare()`

**Alternative**: The existing `run_ppc()` in Phase 04 bypasses PyMC entirely — it manually samples from the posterior and computes the Bernoulli likelihood with numpy. This is faster for custom statistics but cannot produce the `log_likelihood` group needed for LOO/WAIC.

### 3.3 R Ecosystem (for reference)

| Package | Relevant Functions | Notes |
|---------|-------------------|-------|
| **loo** (Vehtari) | `loo()`, `loo_compare()`, `waic()`, `kfold()` | Reference PSIS-LOO implementation |
| **bayesplot** | `ppc_bars()`, `ppc_stat()`, `ppc_loo_pit_ecdf()`, `ppc_error_binned()` | Comprehensive PPC visualization |
| **idealstan** (Kubinec) | `id_post_pred()`, `id_plot_ppc()`, `id_sim_resid()` | PPC specifically for ideal point models |
| **pscl** | `predict.ideal()` | Classification rate, PCP per legislator |
| **mirt** | `itemfit()`, `personfit()`, `M2()` | IRT-specific fit statistics (S-X2, Zh, infit/outfit) |

We do not plan to call R for PPC (unlike Phase 17 for W-NOMINATE), but the R ecosystem shows what statistics the field considers standard.

### 3.4 Python IRT Packages — PPC Gap

| Package | Estimation | Model Fit/PPC |
|---------|-----------|---------------|
| **girth** (v0.8) | MML for 1PL/2PL/3PL, GRM | None |
| **py-irt** (v0.6.6) | Bayesian (Pyro) 1PL/2PL | None |
| **deepirtools** | Deep learning IRT | None |

There is no Python package that computes IRT-specific fit statistics. All exist only in R. The most useful ones (Q3, ICC residuals) are straightforward to implement from posterior samples — approximately 10-20 lines of numpy each.

---

## Part 4: Code Audit

### 4.1 Existing PPC Implementation (Phase 04)

**Location**: `analysis/05_irt/irt.py`, lines 2085-2232

The existing `run_ppc()` function:
- Extracts posterior arrays (`xi`, `alpha`, `beta`) directly from InferenceData
- Generates 500 replications by random sampling from (chain, draw) indices
- Computes two test statistics: overall Yea rate and classification accuracy
- Returns a dict with observed/replicated means, SD, and Bayesian p-value
- Companion `plot_ppc_yea_rate()` produces a histogram with observed value overlaid

**Report integration**: `analysis/05_irt/irt_report.py` has three PPC-related functions:
- `_add_ppc_figure()` (lines 466-479) — loads the Yea rate histogram
- `_add_ppc_summary_table()` (lines 593-641) — great_tables summary with p-values
- `_add_ppc_interpretation()` (lines 835-863) — plain-English explanation

**What works well**:
- Manual posterior sampling is flexible and fast (no PyMC compilation overhead)
- The 2PL likelihood formula `eta = beta * xi - alpha` is correct and shared across all model variants
- Report integration follows the established Tallgrass pattern

**What's missing**:
- Only 2 test statistics (Yea rate, accuracy) out of the ~10 recommended by the literature
- No per-item or per-person checks
- No local dependence assessment (Q3, odds ratios)
- No information criteria (LOO, WAIC) — no `log_likelihood` group computed
- No model comparison capability
- Zero test coverage for PPC functions

### 4.2 Phases With No PPC

| Phase | File | Status | Difficulty to Add |
|-------|------|--------|------------------|
| **04b: 2D IRT** | `analysis/06_irt_2d/irt_2d.py` | No PPC | Low — same Bernoulli likelihood, just xi is 2D |
| **10: Hierarchical** | `analysis/07_hierarchical/hierarchical.py` | No PPC | Low — posterior variables named identically to flat IRT |
| **16: Dynamic IRT** | `analysis/27_dynamic_irt/dynamic_irt.py` | No PPC | Medium — time-varying abilities need adapted statistics |

### 4.3 Code Reuse Potential

The 2PL likelihood computation is identical across all three IRT variants:

```
eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
p = 1 / (1 + exp(-eta))
y_rep ~ Bernoulli(p)
```

For 2D IRT, `xi` is a vector and `beta` is a matrix, but the dot product replaces the scalar multiplication. A shared `run_ppc()` could handle all variants with a small adapter for the linear predictor.

---

## Part 5: Recommendations

### Recommendation 1: Extract Shared PPC Infrastructure

Create `analysis/ppc_utils.py` with:
- A generic replicated-data generator that accepts a linear predictor function
- Standard test statistics (Yea rate, accuracy, APRE, GMP, per-item endorsement rates, per-person total scores, vote margin distribution)
- Standard plot functions (histogram overlay, LOO-PIT, Pareto k)
- Report sections (reusable across all IRT report builders)

### Recommendation 2: Add LOO-CV via ArviZ

For each IRT variant, add post-hoc log-likelihood computation and PSIS-LOO:

```python
pm.compute_log_likelihood(idata, model=model)
loo_result = az.loo(idata, pointwise=True)
```

This enables `az.compare()` across all three models — the single most informative model comparison tool. Report the ELPD table, Pareto k distribution, and stacking weights.

**Computational cost**: For N=50,000 observations and S=4,000 posterior draws, the log-likelihood matrix is 200M entries. Computation takes minutes, not hours. Memory is manageable (~1.5 GB at float64).

### Recommendation 3: Add Item-Pair Local Dependence (Yen's Q3)

Compute residual correlations between roll calls after conditioning on estimated ability. This is the key diagnostic for dimensionality assessment — if the flat 1D model shows strong Q3 violations, the 2D model is justified.

Implementation is ~15 lines of numpy:
1. For each observation, compute `residual = y_obs - P(y=1 | xi_mean, alpha_mean, beta_mean)`
2. Reshape residuals into a legislators x roll-calls matrix
3. Compute pairwise Pearson correlations
4. Flag pairs with |Q3| > 0.20

### Recommendation 4: Add Per-Item and Per-Person PPC

Extend the test statistic battery beyond global Yea rate:
- **Per-item endorsement rates**: For each roll call, compare observed Yea proportion to distribution of replicated proportions. Items where the 95% interval excludes the observed rate are misfitting.
- **Per-person total scores**: For each legislator, compare observed total Yea count to replicated distribution. Legislators outside the 95% interval have response patterns the model cannot reproduce.
- **Vote margin distribution**: Compare the histogram of vote margins (|Yea - Nay| / Total) to replicated distributions. A common failure mode of 1D models is under-predicting the frequency of close votes.

### Recommendation 5: Add GMP and APRE

**Geometric Mean Probability** (Jackman 2001): GMP = exp(mean(log(p_correct))), where p_correct is the model's predicted probability of the observed vote for each observation. More robust than classification accuracy because it penalizes confident wrong predictions. A model that gives 0.51 probability to every vote gets 100% classification accuracy but low GMP.

**APRE** (Aggregate Proportional Reduction in Error): APRE = (errors_null - errors_model) / errors_null, where errors_null counts minority votes as errors. Controls for the 82% Yea base rate that inflates raw accuracy.

### Recommendation 6: Build Phase 4c as a Standalone Comparison Phase

Phase 4c should be a **standalone phase** that:
1. Loads InferenceData from Phases 04, 04b, and 10 (via `resolve_upstream_dir()`)
2. Rebuilds model graphs to compute log-likelihood and posterior predictive samples
3. Runs the full PPC battery on each model
4. Runs LOO-CV comparison across all models
5. Produces a single HTML report with side-by-side comparison tables and plots

This keeps the individual phases lean (they already do convergence diagnostics and holdout validation) while centralizing the model comparison logic.

### Recommendation 7: Do Not Call R for PPC

Unlike Phase 17 (W-NOMINATE/OC), where the R packages implement algorithms with no Python equivalent, the PPC ecosystem in Python is mature. ArviZ provides LOO/WAIC/compare, and the custom statistics (Q3, ICC residuals, GMP, APRE) are trivial to implement from posterior samples. Adding an R dependency would increase complexity without proportional benefit.

---

## Part 6: What We Learn from Each Check

| Check | If It Passes | If It Fails |
|-------|-------------|-------------|
| **Overall Yea rate** | Model is calibrated at the aggregate level | Intercept/difficulty parameters are systematically biased |
| **Per-item endorsement rates** | Individual roll call difficulties well-estimated | Specific bills have dynamics the model misses |
| **Per-person total scores** | Individual ideal points are well-estimated | Specific legislators have response patterns the model cannot reproduce |
| **Vote margin distribution** | Model captures contestedness patterns | Missing dimension or non-spatial dynamics (e.g., logrolling) |
| **Q3 local dependence** | 1D model adequate | Bills cluster into issue dimensions — 2D model justified |
| **GMP** | Model gives appropriate confidence | Model is overconfident or underconfident |
| **APRE** | Model improves over null (majority prediction) | Model adds little beyond base rates |
| **LOO-CV comparison** | Preferred model identified with uncertainty | Models are equivalent (prefer simpler) or LOO is unreliable (high Pareto k) |

---

## Part 7: Expected Results for Kansas

Based on our existing external validation and the literature:

- **Flat 1D IRT**: Should achieve PCP 85-90%, GMP 0.7-0.8. May show Q3 violations for related bills (e.g., tax votes correlating beyond what ideology explains). This would empirically justify the 2D model in Phase 06.
- **Hierarchical 1D IRT**: Should have better LOO than flat IRT (partial pooling improves prediction for legislators with few votes). Per-person PPC should show fewer outliers than flat IRT.
- **2D IRT**: Should have better LOO than flat 1D if the second dimension captures real variation. Q3 violations should be reduced. But if the second dimension is noise, LOO will penalize the extra complexity.
- **Republican intra-party variation**: PPCs focused on within-Republican prediction should be the most informative diagnostic, since that is where the interesting variation lives.
- **Veto override votes**: Likely model outliers — cross-party coalitions on these 34 votes may resist ideological explanation.

---

## Part 8: Key References

| Reference | Year | Contribution |
|-----------|------|-------------|
| Gelman, Meng & Stern | 1996 | Realized discrepancies, PPP-values, PPMC framework |
| Beguin & Glas | 2001 | PPMC for multidimensional IRT, score distribution checks |
| Jackman | 2001 | GMP for ideal point model checking |
| Sinharay | 2003 | ETS simulation studies of PPMC discrepancy measures |
| Clinton, Jackman & Rivers | 2004 | Bayesian ideal points, classification rate PPC |
| Sinharay, Johnson & Stern | 2006 | Systematic evaluation; odds ratios most powerful |
| Fox | 2010 | Odds ratios, latent residuals, prior predictive sensitivity |
| Yen | 1984 | Q3 statistic for local independence |
| Christensen, Makransky & Horton | 2017 | Critical values for Q3 via parametric bootstrapping |
| Levy & Mislevy | 2016 | Unified Bayesian psychometric text with PPMC |
| Vehtari, Gelman & Gabry | 2017 | PSIS-LOO, WAIC, practical model evaluation |
| Merkle, Furr & Rabe-Hesketh | 2019 | Conditional vs. marginal LOO/WAIC for latent variable models |
| Joo, Lee, Park & Stark | 2023 | PPMC for ideal point dimensionality assessment |
| Vehtari, Simpson, Gelman, Yao & Gabry | 2024 | Updated PSIS thresholds (sample-size dependent) |
| Luo & Al-Harbi | 2017 | LOO/WAIC performance comparison for IRT model selection |
| Kubinec | 2024 | idealstan: PPC for ideal point IRT models |

---

## Part 9: Implementation Impact Assessment

**Effort**: Medium. The PPC statistics are computationally simple — the hard part is the integration with nutpie/PyMC for log-likelihood computation and the report assembly.

**Runtime**: Log-likelihood computation adds 2-5 minutes per model. PPC replications (500) add ~1 minute per model. LOO computation is negligible. Total Phase 4c runtime estimate: 15-25 minutes (dominated by model graph rebuilding and log-likelihood computation for 3 models x 2 chambers).

**Dependencies**: No new dependencies. ArviZ, PyMC, numpy, matplotlib, great_tables all already in the environment.

**Test coverage**: Currently zero PPC tests. Phase 4c should ship with tests for:
- PPC statistic computation correctness (synthetic data with known properties)
- LOO/WAIC computation (at least smoke tests)
- Report generation
- Upstream resolution (finding Phase 04/04b/10 results)
