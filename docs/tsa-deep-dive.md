# Time Series Analysis Deep Dive

**A literature survey, ecosystem comparison, code audit, and implementation recommendations for temporal analysis of legislative voting data.**

---

## Executive Summary

Phase 15 (TSA) adds temporal analysis to the Tallgrass pipeline via two complementary methods: rolling-window PCA for ideological drift detection and PELT changepoint detection on weekly Rice Index time series. This deep dive surveys the academic literature, evaluates the full open-source ecosystem across Python, R, Julia, Rust, and Go, audits our implementation against best practices, and identifies seven concrete recommendations.

**Key findings:**

- Our choice of **ruptures + PELT with RBF kernel** is the canonical Python choice for offline changepoint detection. No alternative library offers a compelling reason to switch.
- **Rolling PCA** is a pragmatic choice over dynamic IRT (Martin-Quinn), trading statistical rigor for speed (seconds vs. hours). The correlation between PC1 and IRT ideal points (r > 0.95 in our data) validates this trade-off for within-session drift detection.
- The **Rice Index** is the correct cohesion metric for Kansas data (two-option voting, no strategic abstention culture). Carey's UNITY and Hix's Agreement Index solve problems we don't have.
- Our implementation is clean and well-tested (85 tests), with **all seven improvements resolved**: Desposato small-group correction, explicit short-session validation, finer penalty grid, imputation sensitivity check, variance-change detection test, CROPS exact penalty selection (via R `changepoint` subprocess), and Bai-Perron confidence intervals (via R `strucchange` subprocess). R enrichment is optional — Python-only PELT always runs.
- The R ecosystem remains dominant for dynamic ideal points (`idealstan`, `emIRT`, `MCMCpack`). If dynamic ideal points become a priority, `idealstan` (Stan-based, supports random walk/AR(1)/GP priors) is the strongest single tool. A PyMC native implementation using `GaussianRandomWalk` priors is also viable given our existing infrastructure.

---

## Part 1: The Problem

Legislative voting analysis traditionally treats each session as a static snapshot — one ideology score per legislator, one cohesion score per party. But a two-year Kansas biennium contains approximately 882 roll calls spanning roughly 18 months of active voting. Several things can change during that span:

1. **Individual drift**: A legislator may shift positions, whether from genuine ideological evolution, changed issue priorities, constituent pressure, or strategic repositioning.
2. **Party cohesion breaks**: A party's internal unity may change abruptly around events like veto overrides, leadership changes, or end-of-session deal-making.
3. **Polarization dynamics**: The gap between parties may widen or narrow over the course of a session.

Static analyses (Phases 1–14) capture none of these dynamics. Phase 15 fills the gap with two questions:

- **Drift**: Did anyone move? (Rolling-window PCA)
- **Breaks**: Did party cohesion change abruptly? (PELT changepoint detection)

---

## Part 2: Academic Foundations

### 2.1 Measuring Ideology Over Time

The dominant methods for tracking legislator ideology across time, ranked by usage in the field:

**DW-NOMINATE** (McCarty, Poole & Rosenthal 1997). The workhorse of congressional ideal point estimation. Extends W-NOMINATE by allowing legislators' ideal points to drift *linearly* between congresses. Legislators serving in multiple congresses provide overlap that anchors the ideological space across time. The linear drift constraint is both its strength (maintains comparability) and limitation (cannot capture nonlinear shifts). **Nokken-Poole scores** (2004) relax this by holding bill parameters fixed and re-estimating a separate ideal point per legislator per congress, allowing maximum flexibility. **Penalized Spline DW-NOMINATE** (Lewis & Sonnet) replaces the linear constraint with B-splines and a roughness penalty, allowing flexible nonlinear movement.

**Martin-Quinn Dynamic IRT** (Martin & Quinn 2002). The first explicitly Bayesian dynamic ideal point model. Originally developed for the U.S. Supreme Court (1953–1999), it treats each justice's ideal point as a random walk: `θ_{i,t} = θ_{i,t-1} + ε_{i,t}`, where `Δ_i` (the evolutionary variance) is estimated from data. Unlike DW-NOMINATE's linear drift, the random walk allows *any* trajectory shape. The model uses 2PL IRT (probit link) and Gibbs sampling via data augmentation. Martin-Quinn scores for SCOTUS are updated annually at mqscores.wustl.edu.

**Clinton-Jackman-Rivers (CJR) Model** (2004, *APSR*). The foundational Bayesian IRT paper for ideal points. The base model is static, but naturally extends to dynamics by placing time-series priors on latent ideal points — precisely what `MCMCdynamicIRT1d` in MCMCpack implements.

**Variational EM for Dynamic IRT** (Imai, Lo & Olmsted 2016, *APSR*). Orders of magnitude faster than MCMC: estimates half a million ideal points in about two hours for massive datasets (N=500, J=1000, T=100). Implemented in the `emIRT` R package as `dynIRT()`. Produces essentially identical results to MCMC.

**Rolling-window PCA**. The simplest approach: apply PCA to a sliding window of roll calls, producing a sequence of PC1 scores. Completely nonparametric and computationally trivial, but suffers from the Procrustes problem — each window produces its own coordinate system, requiring post-hoc sign alignment (e.g., enforce Republicans = positive). Window width is a critical tuning parameter. No principled uncertainty quantification. **This is what Tallgrass Phase 15 implements.**

**Functional Data Analysis** (Chen et al. 2022, *PLOS ONE*). Treats each justice's ideal-point trajectory as a *function* of time, then decomposes via functional principal components (FPCs). Applied to Martin-Quinn scores (1791–2020): two FPCs suffice to characterize any justice's career trajectory. FPC1 = static position, FPC2 = drift direction/magnitude.

### 2.2 Measuring Party Cohesion

**Rice Index** (Rice 1925, 1928). The foundational metric: `Rice = |Yea - Nay| / (Yea + Nay)`. Range 0 (50-50 split) to 1 (unanimous). Abstentions excluded from the denominator. Simple, interpretable, but inflated for small parties (see Desposato below) and silent on strategic abstention.

**UNITY Score** (Carey 2007, *AJPS*). Accounts for nonvoting as a distinct behavioral category. A weighted variant discounts lopsided votes and emphasizes closely contested ones. Carey's comparison across 10 countries showed that UNITY, RICE, and WELDON reveal different aspects of party behavior: the U.S. Congress shows high UNITY but moderate RICE (legislators show up but dissent), while some Latin American legislatures show the reverse.

**Agreement Index** (Hix, Noury & Roland 2005, *BJPS*). Designed for three-option voting (Yes/No/Abstain) in the European Parliament. Rescales Attina's (1990) index to [0, 1]. Not needed for Kansas (two-option voting: Yea/Nay).

**Desposato Correction** (2005, *BJPS*). Small-party bias: Rice is systematically inflated for small groups even when underlying individual behavior is identical. Desposato's nonparametric correction subsamples from the larger party to match the smaller party's size, eliminating the artifact. **Relevant for Kansas**: the Senate Democrat caucus has as few as 8 members in recent bienniums; uncorrected Rice comparisons with the 32-member Republican caucus are biased.

**Adjusted Party Unity** (Crespin, Rohde & Vander Wielen 2013, *Party Politics*). Controls for agenda effects — conventional measures confound genuine changes in party discipline with changes in which bills reach the floor. Their correction accounts for agenda composition.

**CQ Party Unity Score**. The Congressional Quarterly standard since 1953. Defines a "party vote" as one where ≥50% of one party opposes ≥50% of the other. Per-legislator: fraction of party votes where the legislator sided with their party majority. This is a per-legislator metric (unlike Rice, which is per-vote).

### 2.3 Changepoint Detection in Political Science

Published applications of changepoint detection to political data are surprisingly sparse:

- **Spirling (2007, *The American Statistician*)**: Reversible-jump MCMC to detect changepoints in Iraq War casualty time series. Found strongest evidence for k=4 breaks.
- **Park & Yamauchi (2022, *Political Analysis*)**: Changepoint detection and regularization in time-series cross-sectional (TSCS) data — directly addressing the panel structure common in legislative research.
- **Hidden Markov / regime-switching models**: Used for partisan regime analysis in Congress. Captures discrete regime dynamics rather than continuous evolution.

The specific application of changepoint detection to legislative cohesion time series (weekly Rice) appears to be uncommon in published political science. Our approach — PELT with RBF kernel, weekly aggregation, penalty sensitivity — is pragmatic and defensible, if not drawn from a specific methodological precedent.

### 2.4 Event Studies in Legislative Analysis

Political scientists study the effect of specific events on voting patterns using:

- **Interrupted Time Series (ITS)**: Pre/post comparison with controls for pre-intervention trends. The most common quasi-experimental design for legislative events.
- **Bayesian Structural Time Series (BSTS) / CausalImpact** (Brodersen et al. 2015): Builds a counterfactual prediction of what *would have happened* without the intervention. Google's CausalImpact R package implements this. Could formalize our veto override cross-referencing.
- **Difference-in-Differences (DiD)**: Within-individual comparisons across periods.
- **Bai-Perron structural break tests** (1998, 2003): Multiple breakpoints with formal confidence intervals. The econometric gold standard for structural breaks. Implemented in R's `strucchange` and `mbreaks` packages.

---

## Part 3: Open-Source Ecosystem

### 3.1 Changepoint Detection Libraries

#### Python

| Library | GitHub Stars | Algorithms | Maintained | Verdict |
|---------|-------------|------------|------------|---------|
| **[ruptures](https://github.com/deepcharles/ruptures)** | ~2,000 | PELT, BinSeg, BottomUp, Window, Dynp, KernelCPD | Yes (v1.1.10, Sep 2025) | **Best-in-class offline CPD. Our choice.** |
| **[stumpy](https://github.com/stumpy-dev/stumpy)** | ~4,100 | Matrix Profile (FLUSS/FLOSS semantic segmentation) | Yes (v1.14.x) | Regime change via matrix profile. Different problem framing. |
| **[Kats](https://github.com/facebookresearch/Kats)** | ~6,300 | CUSUM, BOCPD, StatSig | Community-maintained | Full toolkit but heavy deps. Changepoint detection is a small part. |
| **[river](https://github.com/online-ml/river)** | ~5,700 | ADWIN, Page-Hinkley, DDM, EDDM, KSWIN | Yes | Online drift detection for ML monitoring. Different problem. |
| **[skchange](https://github.com/NorskRegnesentral/skchange)** | ~40 | PELT, BinSeg, BOCPD, AMOC, SeededBS, MOSUM | Yes (active dev) | sktime-compatible. Growing but young. |
| **[changepoint-online](https://github.com/grosed/changepoint_online)** | ~8 | FOCuS, NUNC, NPFocus | Yes (Lancaster group) | State-of-the-art online/sequential CPD. Academic cutting edge. |
| **[bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection)** | ~750 | BOCPD (online), offline Bayesian | Maintained | Clean BOCPD. PyTorch backend, GPU support. |

**Assessment**: `ruptures` is the clear winner for offline changepoint detection in Python. It is the reference implementation of PELT (Killick et al. 2012), published in the Signal Processing journal (Truong, Oudre & Vayatis 2020), with 6 search methods × 10+ cost functions and C-optimized kernels. No other Python library comes close for our use case (offline, bounded time series, need for penalty sensitivity).

#### R

| Package | Algorithms | Key Strength |
|---------|------------|--------------|
| **[changepoint](https://cran.r-project.org/package=changepoint)** | PELT, BinSeg, SegNeigh, AMOC, CROPS | Reference R implementation. CROPS for penalty selection. |
| **[ecp](https://cran.r-project.org/package=ecp)** | E-Divisive, E-Agglomerative, KCP | Distribution-free multivariate CPD. Energy statistics. |
| **[mcp](https://lindeloev.github.io/mcp/)** | Bayesian regression changepoints (JAGS) | Most flexible for regression-based CPD. Full posterior. |
| **[bcp](https://cran.r-project.org/package=bcp)** | Barry-Hartigan Bayesian | Posterior probability at each index. |
| **[strucchange](https://cran.r-project.org/package=strucchange)** | CUSUM, MOSUM, Bai-Perron F-test | Structural break testing with formal hypothesis tests and confidence intervals. |
| **[breakfast](https://cran.r-project.org/package=breakfast)** | WBS, WBS2, NOT, SeedBS | Unified interface for Fryzlewicz-group algorithms. |
| **[mosum](https://cran.r-project.org/package=mosum)** | MOSUM (single and multiscale) | Bootstrap confidence intervals for changepoint locations. |
| **[EnvCpt](https://cran.r-project.org/package=EnvCpt)** | Auto-selection across 12 models | Compares trend, AR, changepoint combinations. |
| **[robseg](https://github.com/guillemr/robust-fpop)** | Robust FPOP (biweight, Huber, L1) | Robust to outliers. |
| **[cpop](https://cran.r-project.org/package=cpop)** | CPOP (change-in-slope) | Piecewise-linear signals. Published JSS 2024. |
| **[InspectChangepoint](https://cran.r-project.org/package=InspectChangepoint)** | INSPECT (sparse projection + WBS) | High-dimensional multivariate CPD. |

**Assessment**: R's ecosystem is substantially deeper than Python's. The key tool we lack a Python equivalent for is **CROPS** (Changepoints for a Range of Penalties) — a meta-algorithm that efficiently finds *all* optimal segmentations across a continuous range of penalties. Our manual penalty sensitivity sweep (`SENSITIVITY_PENALTIES = [3, 5, 10, 15, 25, 50]`) is a crude approximation of what CROPS does exactly and efficiently. The R `changepoint` package includes CROPS natively.

Additionally, **strucchange** (Bai-Perron) provides formal confidence intervals on changepoint locations — something PELT does not offer. Our penalty sensitivity analysis provides an indirect robustness check, but confidence intervals would be a stronger statistical statement.

#### Julia

| Package | Algorithms | Status |
|---------|------------|--------|
| **[Changepoints.jl](https://github.com/STOR-i/Changepoints.jl)** | PELT, BinSeg, WBS, SeedBS, MOSUM | Unmaintained (last commit 2022). Lancaster group. |

Julia's changepoint ecosystem is thin. Changepoints.jl has the right design but is dormant.

#### Rust / Go

| Tool | Language | Algorithms | Notes |
|------|----------|------------|-------|
| **[augurs](https://github.com/grafana/augurs)** | Rust | Wraps `changepoint` crate | Grafana's time series toolkit. ~560 stars, very active. JS/WASM bindings. |
| **[changepoint](https://github.com/promised-ai/changepoint)** (Redpoll) | Rust | BOCPD, ArgpCPD | Bayesian online focus. Python bindings available via pip. |
| **[changepoint](https://github.com/flyingmutant/changepoint)** | Go | ED-PELT (nonparametric) | O(n log n). Simple API. |

**Assessment**: Rust is emerging for production CPD (Grafana's augurs is notable), but these tools target operational monitoring, not statistical analysis.

### 3.2 Dynamic Ideal Point Tools

#### R (dominant ecosystem)

| Package | Dynamic | Algorithm | Key Function |
|---------|---------|-----------|-------------|
| **[idealstan](https://github.com/saudiwin/idealstan)** | Yes (3 types) | Stan (full Bayes) | 16 model types. Random walk, AR(1), and GP time-series priors. Most flexible. |
| **[MCMCpack](https://cran.r-project.org/package=MCMCpack)** | Yes | MCMC (Gibbs) | `MCMCdynamicIRT1d()` — canonical Martin-Quinn implementation. |
| **[emIRT](https://cran.r-project.org/package=emIRT)** | Yes | Variational EM | `dynIRT()` — orders of magnitude faster than MCMC. |
| **[pscl](https://cran.r-project.org/package=pscl)** | No | MCMC | `ideal()` — static CJR model. |
| **[wnominate](https://cran.r-project.org/package=wnominate)** | No | EM-like | Static W-NOMINATE. |
| **[dwnominate](https://github.com/wmay/dwnominate)** | Yes | Three-step EM | DW-NOMINATE with linear drift. Requires Fortran compiler. |

#### Python

| Package | Dynamic | Notes |
|---------|---------|-------|
| **py-irt** | No | Pyro/PyTorch. GPU-accelerated 1PL/2PL/4PL. No temporal. |
| **PyMC (custom)** | Build-your-own | `GaussianRandomWalk` + 2PL IRT likelihood. Infrastructure exists. |
| **idealstan via rpy2** | Yes | Call R's idealstan from Python. |

**Assessment**: There is **no turnkey Python package for dynamic ideal points**. The path forward is either (a) build a custom dynamic IRT in PyMC/nutpie using `GaussianRandomWalk` priors (our infrastructure supports this), or (b) call R packages via `rpy2`. Given that our roadmap already includes W-NOMINATE (R), the same tooling path applies.

For the specific problem Phase 15 solves — within-session drift detection — rolling PCA is the right tool. Dynamic IRT is the answer to a different question: tracking ideology *across* congresses/sessions with proper measurement uncertainty.

### 3.3 Political Science Time Series Tools

| Tool | Language | Purpose |
|------|----------|---------|
| **[Rvoteview](https://github.com/voteview/Rvoteview)** | R | Query VoteView (DW-NOMINATE scores, roll-call data) |
| **[politicsR](https://cran.r-project.org/package=politicsR)** | R | Rice Index, ENP, HH concentration |
| **[legislatoR](https://github.com/saschagobel/legislatoR)** | R | Comparative Legislators Database (67,000+ politicians, 16 countries) |
| **[CausalImpact](https://cran.r-project.org/package=CausalImpact)** | R | Bayesian causal inference for event studies |
| **[mbreaks](https://cran.r-project.org/package=mbreaks)** | R | Bai-Perron (1998) multiple structural breaks with confidence intervals |

---

## Part 4: Method Evaluation for Kansas Data

### 4.1 Why Rolling PCA Over Dynamic IRT

| Criterion | Rolling PCA | Dynamic IRT (Martin-Quinn) |
|-----------|------------|---------------------------|
| **Runtime** | Seconds | Hours (MCMC) or minutes (variational EM) |
| **Correlation with IRT** | r > 0.95 (our data, Phases 2 vs. 4) | Reference standard |
| **Uncertainty** | None (point estimates only) | Full posterior |
| **Missing data** | Mean imputation | Proper likelihood |
| **Identification** | Sign alignment heuristic | Anchored priors or constraints |
| **Cross-session comparability** | Requires Procrustes | Built-in via random walk |
| **Implementation complexity** | ~50 lines of core logic | Hundreds of lines + MCMC diagnostics |
| **Appropriate for** | Within-session screening | Cross-session tracking, publication-quality |

**Verdict**: For within-session drift screening, rolling PCA is the right choice. It answers "did anyone move?" quickly and cheaply. Dynamic IRT answers "by exactly how much, with what uncertainty?" — a question we don't need answered within a single biennium, especially given our cross-session validation (Phase 13) showing ideology is remarkably stable (r = 0.94–0.98 across bienniums).

### 4.2 Why PELT Over Alternatives

| Criterion | PELT | BOCPD | Bai-Perron | Binary Segmentation |
|-----------|------|-------|------------|---------------------|
| **Optimality** | Exact (penalized) | Bayesian posterior | Exact (regression) | Approximate (greedy) |
| **Speed** | O(n) typical | O(n) per step | O(n²) | O(n log n) |
| **Online/offline** | Offline | Online | Offline | Offline |
| **Penalty selection** | Manual or CROPS | Hazard rate prior | BIC/LWZ/sequential | Information criterion |
| **Confidence intervals** | No (use sensitivity) | Posterior credible intervals | Yes (formal) | No |
| **Python library** | ruptures (excellent) | Kats, bayesian_cpd | statsmodels (limited) | ruptures |
| **Multivariate** | Yes (kernel-based) | Yes (conjugate families) | Limited | Yes |

**Verdict**: PELT is the canonical choice for offline changepoint detection. Our data is offline (full session scraped), the ruptures library is mature, and the RBF kernel handles bounded [0, 1] Rice data without normality assumptions. The main thing we're missing is **CROPS** for principled penalty selection and **Bai-Perron confidence intervals** for formal statistical inference on changepoint locations.

### 4.3 Why Rice Index Over Alternatives

| Metric | Handles Abstention | Handles Small Groups | Three-Option | Implementation |
|--------|-------------------|---------------------|--------------|----------------|
| **Rice** | No (excluded) | No (biased) | No | Phase 7 + Phase 15 |
| **UNITY (Carey)** | Yes | No | No | Not implemented |
| **Agreement Index (Hix)** | Yes (third option) | No | Yes | Not needed (2-option) |
| **Desposato-corrected Rice** | No | Yes | No | Not implemented |

**Verdict**: Rice is the correct base metric for Kansas. Our data has two-option voting (Yea/Nay); absent legislators are not engaging in strategic abstention (Kansas has no formal "Abstain" option — legislators are simply absent). The **Desposato correction** is the most relevant enhancement, given the Kansas Senate's small Democrat caucus (8–11 members in recent bienniums vs. 29–32 Republicans). Without correction, Senate Democrat Rice is inflated relative to Republican Rice simply due to group size.

---

## Part 5: Code Audit

### 5.1 Architecture

Phase 15 follows the established pattern from Phase 7 (Indices): `parse_args()` → `RunContext` → per-chamber analysis → report. Code is organized into clean, pure, testable functions.

**File inventory:**

| File | Lines | Purpose |
|------|-------|---------|
| `analysis/19_tsa/tsa.py` | ~1,560 | Main script: data loading, 14 core functions, 10 plotting functions, R integration, main() |
| `analysis/19_tsa/tsa_r_data.py` | ~190 | Pure R result parsing: CROPS, Bai-Perron, elbow detection, PELT/BP merge |
| `analysis/19_tsa/tsa_strucchange.R` | ~105 | R script: CROPS (`changepoint`) + Bai-Perron (`strucchange`) |
| `analysis/19_tsa/tsa_report.py` | ~680 | HTML report builder: 16 section builders (4 new for R enrichment) |
| `analysis/design/tsa.md` | ~140 | Design document |
| `tests/test_tsa.py` | ~1,560 | 85 tests across 22 test classes |

### 5.2 Constants

| Constant | Value | Justification | Sensitivity |
|----------|-------|---------------|-------------|
| `WINDOW_SIZE` | 75 | ~2–3 weeks of roll calls | 50–150 reasonable; <50 → noisy PC1 |
| `STEP_SIZE` | 15 | 75% overlap (60–75% common in literature) | 10–20 reasonable |
| `MIN_WINDOW_VOTES` | 10 | Per-legislator minimum within window | 5–20 reasonable |
| `MIN_WINDOW_LEGISLATORS` | 20 | Cross-section minimum (matches Phase 2) | 15–25 reasonable |
| `PELT_PENALTY_DEFAULT` | 10.0 | Moderate (sensitivity range: 3–50) | Empirical; CROPS would be better |
| `PELT_MIN_SIZE` | 5 | Minimum 5 weeks per segment | 3–7 reasonable |
| `WEEKLY_AGG_DAYS` | 7 | Calendar week aggregation | Fixed; natural legislative boundary |
| `SENSITIVITY_PENALTIES` | `np.linspace(1,50,25)` | Explores parameter space | 25-point grid; CROPS provides exact solution path when R available |
| `TOP_MOVERS_N` | 10 | Headline count | Arbitrary; 5–15 reasonable |
| `MIN_TOTAL_VOTES` | 20 | Session-wide filter (matches Phase 1 EDA) | Established |

### 5.3 Function Coverage

**Core functions (12)**: All pure, all testable, all tested.

| Function | Tests | Coverage | Notes |
|----------|-------|----------|-------|
| `load_data()` | 0 | 0% | Integration-tested via main(); no unit tests |
| `build_vote_matrix()` | 3 | ~50% | Missing: chronological ordering, filter edge cases |
| `rolling_window_pca()` | 8 | ~80% | Missing: imputation correctness validation |
| `align_pc_signs()` | 5 | ~83% | Good edge case coverage |
| `compute_party_trajectories()` | 4 | ~60% | Missing: gap with missing party data |
| `compute_early_vs_late()` | 5 | ~60% | Missing: legislator dropout between halves |
| `find_top_movers()` | 3 | 100% | Straightforward function; adequate |
| `build_rice_timeseries()` | 5 | ~80% | Missing: fill_null path for missing columns |
| `aggregate_weekly()` | 4 | ~60% | Missing: partial-week behavior |
| `detect_changepoints_pelt()` | 5 | ~85% | Missing: variance-change detection test |
| `detect_changepoints_joint()` | 2 | ~70% | Missing: divergent signal directions |
| `run_penalty_sensitivity()` | 3 | ~66% | Adequate |
| `cross_reference_veto_overrides()` | 3 | ~50% | Missing: unparseable date paths |
| Plotting functions (8) | 0 | 0% | No unit tests (smoke-tested in production) |

### 5.4 Design Decisions — Audit

All decisions documented in `analysis/design/tsa.md` are correctly implemented:

- Rolling PCA over dynamic IRT — **correct for use case**
- PELT over BOCPD — **canonical choice**
- RBF kernel — **appropriate for bounded [0,1] data**
- Weekly aggregation — **justified by voting frequency**
- Self-contained Rice computation — **good isolation from Phase 7**
- Per-window sign alignment — **essential for temporal coherence**
- 75% window overlap — **standard in rolling analysis literature**

### 5.5 Strengths

1. **Fast**: Under 1 minute for a full biennium (both chambers, all plots, full report).
2. **Two complementary analyses**: Drift (individual-level) and cohesion breaks (party-level) answer different questions.
3. **Robust to missing data**: Column-mean imputation + per-window filtering handles irregular participation gracefully.
4. **Penalty sensitivity**: The sweep across 25 evenly-spaced penalties (np.linspace(1, 50, 25)) provides indirect robustness assessment.
5. **Clean separation of concerns**: Rice, PCA, and PELT are independent subsystems.
6. **Veto override cross-referencing**: Automatic contextual annotation of detected changepoints.
7. **Good test coverage**: 64 tests covering all core computation functions, including Desposato correction, imputation sensitivity, short-session warnings, and variance-change detection.
8. **Self-documenting report**: The HTML report includes "How to Read" sections and interpretation guidance for non-technical audiences.

### 5.6 Weaknesses

1. ~~**Silent failure on short sessions**~~: **RESOLVED.** `warnings.warn()` now fires in `rolling_window_pca()` (n_votes < window_size), `compute_early_vs_late()` (n_votes < 2 * MIN_WINDOW_VOTES), and `main()` (too few weekly obs for changepoints).
2. ~~**No formal uncertainty**~~: **PARTIALLY RESOLVED.** Rolling PCA still produces point estimates only, but PELT changepoint locations now have Bai-Perron 95% confidence intervals (when R is available). The penalty sensitivity sweep remains as an additional robustness check.
3. ~~**Mean imputation bias**~~: **RESOLVED.** `compute_imputation_sensitivity()` runs both column-mean and listwise deletion, reports correlation. Integrated into `main()` and filtering manifest.
4. ~~**Desposato bias unaddressed**~~: **RESOLVED.** `desposato_corrected_rice()` implements Monte Carlo correction (Desposato 2005). Applied by default via `correct_size_bias=True` in `build_rice_timeseries()`. 6 new tests.
5. ~~**Discrete penalty search**~~: **RESOLVED.** Replaced 6-point grid with `np.linspace(1, 50, 25)` (25 evenly-spaced penalties). Smoother elbow plots at negligible computational cost.
6. ~~**No Bai-Perron inference**~~: **RESOLVED.** Bai-Perron 95% CIs via R `strucchange` subprocess (ADR-0061). PELT breaks cross-referenced with BP breaks for dual-method confirmation.
7. **Polars deprecation**: `is_in` with a Series of the same dtype triggers a deprecation warning in Polars ≥0.20. One instance was fixed (`build_vote_matrix`, line 183); patterns elsewhere in the codebase may recur.

---

## Part 6: Recommendations

### Recommendation 1: Add CROPS for Penalty Selection — RESOLVED

**Status**: Implemented (2026-02-28)

CROPS implemented via R's `changepoint::cpt.mean(penalty="CROPS")` subprocess. The R script (`tsa_strucchange.R`) finds the exact penalty thresholds where the optimal segmentation changes. `tsa_r_data.py` parses the result and identifies the elbow (largest marginal jump). Solution path plotted as a step function with elbow marker. R is optional — Python-only PELT with 25-point `np.linspace` grid always runs as baseline. ADR-0061.

### Recommendation 2: Desposato Small-Group Correction — RESOLVED

**Status**: Implemented (2026-02-28)

`desposato_corrected_rice()` uses Monte Carlo simulation (10K draws, seed=42) to compute expected Rice under random voting for the group size, then subtracts it from raw Rice (floored at 0). Applied by default via `correct_size_bias=True` parameter on `build_rice_timeseries()`. Existing tests updated to pass `correct_size_bias=False` for raw-formula validation. 6 new tests cover perfect cohesion, random→0, small>large correction, zero votes, deterministic seed, and floor at 0.

### Recommendation 3: Explicit Short-Session Validation — RESOLVED

**Status**: Implemented (2026-02-28)

`warnings.warn()` added in three locations: `rolling_window_pca()` when n_votes < window_size, `compute_early_vs_late()` when n_votes < 2 * MIN_WINDOW_VOTES, and `main()` when a party has too few weekly observations for changepoint detection. Two new tests verify with `pytest.warns()`.

### Recommendation 4: Imputation Sensitivity Check — RESOLVED

**Status**: Implemented (2026-02-28)

`compute_imputation_sensitivity()` runs `compute_early_vs_late()` twice (column-mean vs listwise deletion), correlates drift scores. Integrated into `main()` automatically — no flag needed. Result stored in `chamber_results["imputation_correlation"]` and written to the filtering manifest. Displayed in the HTML report's analysis parameters table. Two new tests.

### Recommendation 5: Finer Penalty Grid — RESOLVED

**Status**: Implemented (2026-02-28)

Replaced `SENSITIVITY_PENALTIES = [3, 5, 10, 15, 25, 50]` with `np.linspace(1, 50, 25).tolist()` — 25 evenly-spaced penalties from 1 to 50. Produces smoother elbow plots at negligible computational cost. True CROPS would require extending `ruptures`; the finer grid is sufficient for our data scale.

### Recommendation 6: Variance-Change Detection Test — RESOLVED

**Status**: Implemented (2026-02-28)

`TestVarianceChangeDetection::test_detects_variance_change` verifies RBF kernel detects a variance change with constant mean (Normal(0.7, 0.02) → Normal(0.7, 0.15)). Validates the specific advantage of RBF over L1/L2 cost functions.

### Recommendation 7: Consider Bai-Perron for Formal Inference — RESOLVED

**Status**: Implemented (2026-02-28)

Bai-Perron implemented via R's `strucchange::breakpoints()` + `confint()` subprocess. Provides formal 95% confidence intervals on break date locations. `tsa_r_data.py` parses results, maps R 1-based indices to dates, and cross-references with PELT breaks. PELT breaks within a Bai-Perron CI are "confirmed" by two independent methods. Plotted as Rice line + break lines + shaded CI bands. ADR-0061.

---

## Part 7: What the Literature Says We Got Right

1. **PELT + RBF for bounded time series**: The RBF kernel is distribution-free and detects changes in both mean and variance — exactly what's needed for Rice Index data on [0, 1]. The TCPDBench evaluation (van den Burg & Williams 2020) found PELT-family methods among the best performers across 37 benchmark time series.

2. **Weekly aggregation**: With ~2 roll calls per day, daily Rice is noisy. Weekly aggregation at 7-day intervals is the natural legislative boundary and produces ~14 observations per data point — sufficient for stable estimates. This matches the approach used in operational political science (CQ reports aggregate by congress, we aggregate by week within a session).

3. **Self-contained Rice computation**: Not depending on Phase 7 means TSA can run standalone and is insulated from upstream changes. The Rice formula is trivial enough that duplication is cheaper than versioning.

4. **Penalty sensitivity analysis**: While not as principled as CROPS, sweeping across multiple penalties and looking for flat regions is a widely-used robustness check in the changepoint literature. Truong et al. (2020) recommend exactly this approach when CROPS is unavailable.

5. **Per-window sign alignment**: The Procrustes problem (PCA sign indeterminacy) is the most common pitfall of rolling PCA. Enforcing Republicans = positive per window, consistent with Phases 2 and 4, is the standard solution.

6. **Joint multivariate detection**: Stacking both parties' Rice series for 2D PELT is a smart addition that finds session-wide events (affecting both parties simultaneously) rather than party-specific dynamics.

7. **Veto override cross-referencing**: Contextual annotation of detected changepoints with known legislative events is exactly what political scientists recommend (face validity checking). Our 14-day window is a reasonable proximity threshold.

---

## Part 8: What the Field Does Differently

1. **Dynamic IRT over rolling PCA** for tracking ideology. The political science gold standard is Martin-Quinn (random walk IRT) or DW-NOMINATE (linear drift). Rolling PCA is a screening tool, not a measurement model. For within-session drift (our use case), the distinction matters less — but for cross-session tracking or publication, dynamic IRT is expected.

2. **CROPS over manual penalty selection**. The changepoint detection literature (Haynes et al. 2017) considers manual penalty selection a methodological weakness. CROPS finds the exact solution path efficiently and identifies the "elbow" where adding changepoints yields diminishing returns.

3. **Bai-Perron with confidence intervals** is the econometric standard for structural breaks. Reporting a changepoint date without a confidence interval is like reporting a mean without a standard error.

4. **Desposato correction** is standard when comparing cohesion across groups of different sizes. Not applying it is a known source of bias.

5. **Agenda-adjusted cohesion** (Crespin et al. 2013) controls for the fact that different bills reaching the floor can drive apparent changes in cohesion even when underlying preferences are constant. This is a more sophisticated issue that we do not currently address.

6. **CausalImpact / BSTS** for event studies. Our veto override cross-referencing identifies proximity but does not estimate causal effects. A Bayesian structural time series model could estimate the counterfactual cohesion trajectory in the absence of the event.

---

## References

### Foundational Methods

- Poole, K.T. & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting*. Oxford University Press.
- Clinton, J.D., Jackman, S. & Rivers, D. (2004). "The Statistical Analysis of Roll Call Data." *APSR* 98(2): 355–370.
- Martin, A.D. & Quinn, K.M. (2002). "Dynamic Ideal Point Estimation via MCMC for the U.S. Supreme Court, 1953–1999." *Political Analysis* 10(2): 134–153.
- Imai, K., Lo, J. & Olmsted, J. (2016). "Fast Estimation of Ideal Points with Massive Data." *APSR* 110(4): 631–656.
- Nokken, T.P. & Poole, K.T. (2004). "Congressional Party Defection in American History." *Legislative Studies Quarterly* 29(4): 545–568.

### Cohesion Metrics

- Rice, S.A. (1928). *Quantitative Methods in Politics*. New York: Alfred A. Knopf.
- Carey, J.M. (2007). "Competing Principals, Political Institutions, and Party Unity." *AJPS* 51(1): 92–107.
- Hix, S., Noury, A. & Roland, G. (2005). "Power to the Parties." *BJPS* 35(2): 209–234.
- Desposato, S. (2005). "Correcting for Small Group Inflation of Roll-Call Cohesion Scores." *BJPS* 35(4): 731–744.
- Crespin, M.H., Rohde, D.W. & Vander Wielen, R.J. (2013). "Measuring Variations in Party Unity Voting." *Party Politics* 19(3): 432–457.

### Changepoint Detection

- Killick, R., Fearnhead, P. & Eckley, I.A. (2012). "Optimal Detection of Changepoints with a Linear Computational Cost." *JASA* 107(500): 1590–1598.
- Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection." arXiv:0710.3742.
- Haynes, K., Eckley, I.A. & Fearnhead, P. (2017). "Computationally Efficient Changepoint Detection for a Range of Penalties." *JCGS* 26(1): 134–143.
- Truong, C., Oudre, L. & Vayatis, N. (2020). "Selective Review of Offline Change Point Detection Methods." *Signal Processing* 167: 107299.
- Bai, J. & Perron, P. (2003). "Computation and Analysis of Multiple Structural Change Models." *Journal of Applied Econometrics* 18(1): 1–22.
- Van den Burg, G.J.J. & Williams, C.K.I. (2020). "An Evaluation of Change Point Detection Algorithms." arXiv:2003.06222.

### Political Time Series

- Spirling, A. (2007). "Turning Points in the Iraq Conflict." *The American Statistician* 61(4): 315–320.
- Park, J.H. & Yamauchi, C. (2022). "Change-Point Detection and Regularization in TSCS Data Analysis." *Political Analysis*.
- Shor, B. & McCarty, N. (2011). "The Ideological Mapping of American Legislatures." *APSR* 105(3): 530–551.
- Chen, X. et al. (2022). "The Dynamics of Ideology Drift Among U.S. Supreme Court Justices." *PLOS ONE* 17(5): e0269598.

### Software

- ruptures: [github.com/deepcharles/ruptures](https://github.com/deepcharles/ruptures)
- idealstan: [github.com/saudiwin/idealstan](https://github.com/saudiwin/idealstan)
- emIRT: [CRAN](https://cran.r-project.org/package=emIRT)
- MCMCpack: [CRAN](https://cran.r-project.org/package=MCMCpack)
- changepoint (R): [CRAN](https://cran.r-project.org/package=changepoint)
- strucchange: [CRAN](https://cran.r-project.org/package=strucchange)
- mcp: [lindeloev.github.io/mcp](https://lindeloev.github.io/mcp/)
- CausalImpact: [CRAN](https://cran.r-project.org/package=CausalImpact)
- TCPDBench: [github.com/alan-turing-institute/TCPDBench](https://github.com/alan-turing-institute/TCPDBench)
- skchange: [github.com/NorskRegnesentral/skchange](https://github.com/NorskRegnesentral/skchange)
- augurs (Rust): [github.com/grafana/augurs](https://github.com/grafana/augurs)
