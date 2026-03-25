# Network Psychometrics & Exploratory Graph Analysis: Deep Dive for Tallgrass

**Date:** 2026-03-25
**Status:** Research note
**Scope:** Hudson Golino's network psychometrics framework (EGA, bootEGA, hierEGA, dynEGA, UVA, TEFI) and its applicability to the tallgrass Kansas Legislature voting analysis pipeline.

---

## Executive Summary

Hudson Golino (University of Virginia) has developed a suite of network psychometrics methods — collectively called **Exploratory Graph Analysis (EGA)** — that estimate dimensionality by building sparse networks from data and detecting communities. His methods offer three things our pipeline currently lacks:

1. **A principled, non-parametric dimensionality estimate** that doesn't assume a factor model — potentially resolving our 1D-vs-2D routing question without the Tier 1/2/3 quality-gate cascade.
2. **Bootstrap stability assessment** (bootEGA) for dimensional structures — directly applicable to the 7/14 Senate sessions where PCA axis instability plagues our pipeline.
3. **Hierarchical dimensionality** (hierEGA) that discovers whether lower-order policy domains organize into higher-order ideological axes — the structural question our 2D IRT only partially answers.

This document evaluates each method, maps it to our pipeline, assesses feasibility, and recommends concrete integration points.

---

## Table of Contents

1. [Background: Who is Hudson Golino?](#1-background)
2. [Exploratory Graph Analysis (EGA)](#2-ega)
3. [Bootstrap EGA (bootEGA)](#3-bootega)
4. [Unique Variable Analysis (UVA)](#4-uva)
5. [Hierarchical EGA (hierEGA)](#5-hierega)
6. [Dynamic EGA (dynEGA)](#6-dynega)
7. [Total Entropy Fit Index (TEFI)](#7-tefi)
8. [Network Loadings & the Latent Variable Debate](#8-network-loadings)
9. [Current Pipeline: What We Already Have](#9-current-pipeline)
10. [Gap Analysis: Where EGA Methods Add Value](#10-gap-analysis)
11. [Implementation Feasibility](#11-feasibility)
12. [Recommendations](#12-recommendations)
13. [References](#13-references)

---

## 1. Background

Hudson F. Golino is an Associate Professor of Quantitative Methods in the Department of Psychology at the University of Virginia (ORCID: 0000-0002-1601-1447). His primary contributions are in network psychometrics — a framework that models psychological constructs as networks of causally coupled variables rather than effects of latent causes. His signature methodological contribution is **Exploratory Graph Analysis (EGA)**, implemented in the **EGAnet** R package (v2.4.0, CRAN, AGPL-3), co-maintained with Alexander P. Christensen. Key collaborators include Luis Eduardo Garrido, Dingjing Shi, and Steven M. Boker. His work is partly funded by UVA's Democracy Initiative, signaling interest in political science applications.

Golino has not directly analyzed roll-call voting data. His closest political work is a dynEGA analysis of Russian troll tweets during the 2016 US election (Psychometrika, 2022). However, his methods are domain-general — they operate on any item-response matrix, which is exactly what roll-call votes are.

---

## 2. Exploratory Graph Analysis (EGA)

### What It Is

EGA estimates the number of dimensions in a dataset using a three-step procedure:

1. **Correlation estimation.** Auto-detects the appropriate correlation type: Pearson for continuous, polychoric for ordinal, **tetrachoric for binary** (our case — Yea/Nay votes). This is a meaningful advantage over PCA, which treats binary data as continuous.

2. **Sparse network estimation.** Applies the **Graphical LASSO** (L1-penalized inverse covariance estimation) with model selection via the **Extended Bayesian Information Criterion (EBIC)**. Sweeps 100 lambda values (logarithmically spaced from lambda_max to lambda_max/100) and selects the network with lowest EBIC (gamma=0.5, controlling sparsity). The result is a **partial correlation network** — edges represent conditional dependencies between items after controlling for all other items. Alternative: **TMFG** (Triangulated Maximally Filtered Graph), a non-regularized planar graph with exactly 3n-6 edges. No tuning parameter needed, but denser.

3. **Community detection.** Applies a clustering algorithm to the estimated network. Each community = one latent dimension. The number of communities = the estimated dimensionality. Algorithms: **Walktrap** (default, random walks of step length 4), **Louvain** (modularity optimization with consensus clustering over 1000 iterations), or **Leiden** (refined modularity, guaranteed connected communities). A **unidimensional check** tests whether a 1D solution fits better.

### Key Result: Performance vs. Traditional Methods

In a 2020 simulation study (Psychological Methods, 32,000 datasets, 64 conditions):

- EGA was the **only method achieving 100% accuracy** for 4-factor structures with r=0.70 inter-factor correlations at N=5000.
- EGA was "the technique whose accuracy was least affected by the conditions investigated."
- EGA outperformed parallel analysis (PA), Minimum Average Partial (MAP), BIC, EBIC, Kaiser-Guttman, and Very Simple Structure across conditions.
- **EGA's advantage is most pronounced when factors are highly correlated** — exactly the condition in legislative data where ideology and partisanship are near-collinear.

### Why This Matters for Tallgrass

Our pipeline's central dimensionality question — is the voting space 1D (ideology) or 2D (ideology + establishment/contrarian)? — is currently answered by a cascade of quality gates (ADR-0110):

| Current method | What it does | Limitation |
|---|---|---|
| PCA scree plot | Visual eigenvalue inspection | Subjective; PC1 axis flips in 7/14 Senate sessions (ADR-0118) |
| 2D IRT convergence | R-hat thresholds (Tier 1 < 1.10, Tier 2 < 2.50) | Tests model *convergence*, not whether 2D is *correct* |
| W-NOMINATE cross-validation | Correlation gate against unsupervised ranking (ADR-0123) | External dependency; catches errors but doesn't *determine* dimensionality |
| Q3 residual analysis (Phase 08) | Local dependence in 1D vs 2D | Diagnostic, not prescriptive — flags violations but doesn't choose K |

EGA would provide a **direct, data-driven dimensionality estimate** before we commit to a model. Run EGA on the tetrachoric correlation matrix of roll-call votes; if it finds 1 community, the chamber is unidimensional and we can skip 2D IRT entirely. If it finds 2+, we have principled evidence for multidimensionality — and the community assignments tell us which bills load on which dimension.

### Caveats for Legislative Data

- **Items = bills, not psychometric items.** In psychometrics, items are carefully designed with known factor loadings. Bills are messy: procedural votes, unanimous consent, and party-line votes all coexist. EGA may find communities that reflect bill *type* (procedural vs. contested) rather than latent ideology. This is why our `--contested-only` flag exists — filtering to contested votes before running EGA would produce cleaner signal.

- **High base rate.** Kansas legislatures pass ~82% of votes Yea. This extreme marginal creates near-zero variance for many items, which can produce degenerate tetrachoric correlations. Row/column filtering (legislators with < MIN_VOTES contested votes, bills with < 10% dissent) is essential preprocessing.

- **N/p ratio.** Typical psychometric datasets have N >> p (many respondents, few items). Legislative data inverts this: ~165 House members, ~40 Senators, but potentially 500+ contested bills. EGA handles p > N poorly (GLASSO regularization helps, but community detection on very sparse networks is unreliable). The Senate's 40-member chamber may be borderline. Subsetting to high-discrimination bills (|beta| > threshold from Phase 05) could help.

---

## 3. Bootstrap EGA (bootEGA)

### What It Is

bootEGA runs EGA on B bootstrap replicates (default 500) using either parametric (generate from the empirical correlation matrix) or non-parametric (resample rows with replacement) bootstrapping. It produces:

1. **Dimension frequency:** How often each K was recovered (e.g., "K=2 in 487/500 replicates").
2. **Structural consistency:** How often each dimension is replicated exactly (same items in same community).
3. **Item stability:** Per-item, the proportion of replicates assigning it to its empirical community. Items below 0.70 are flagged as unstable.
4. **Median network:** Median-aggregated across replicates.

### Why This Matters for Tallgrass

The 7/14 Senate PCA axis instability problem (ADR-0118) is fundamentally a **stability problem**. In the 78th-83rd and 88th Senate sessions, the dominant dimension flips between party-divide and intra-Republican factionalism depending on which bills are included. bootEGA would:

- **Quantify instability directly.** If K=1 in 60% of replicates and K=2 in 40%, that's a principled "unstable" flag — no need for the 7-gate quality system to detect it post hoc.
- **Identify unstable bills.** Bills with item stability < 0.70 are the ones causing dimensional instability. This is a more targeted filter than `--contested-only`, which uses a blunt dissent threshold.
- **Validate our horseshoe detection.** If bootEGA finds the same sessions unstable that our horseshoe detector flags, we have convergent evidence. If it finds *different* sessions, we're missing something.

The parametric bootstrap (generating from the tetrachoric correlation matrix) is preferred for binary data, since resampling rows can produce degenerate columns (bills with 100% Yea in a resample).

---

## 4. Unique Variable Analysis (UVA)

### What It Is

UVA detects **locally dependent (redundant) variable pairs** — variables that share variance beyond what their common latent factor explains. It works by:

1. Estimating a network via EBICglasso.
2. Computing **weighted topological overlap (wTO)** for all variable pairs. wTO measures whether two variables share the same neighbors with similar edge weights — i.e., they occupy the same structural position.
3. Pairs with wTO > 0.25 are flagged as redundant.

Reduction strategies: remove all but one, average scores, sum scores, or create latent variables via CFA.

### Why This Matters for Tallgrass

Legislative voting has massive local dependence:

- **Procedural vote sequences.** A bill's second reading, third reading, and final action votes are near-identical. These aren't independent items — they're the same vote measured three times.
- **Amendment cascades.** Votes on amendments to the same bill are conditionally dependent given the bill's ideology, not just the legislator's ideology.
- **Party-line votes.** In supermajority chambers, many bills produce identical voting patterns (all R Yea, all D Nay) — these are effectively duplicates.

Our pipeline partially addresses this: `--contested-only` filters low-dissent votes, and Phase 08 (PPC) computes Q3 residual correlations to detect local dependence globally. But UVA would give us **per-bill-pair redundancy scores**, enabling:

- **Principled bill filtering** before IRT, replacing the blunt 10% dissent threshold with a topological overlap criterion.
- **Identifying amendment cascades** automatically (bills with wTO > 0.25 that share the same bill number prefix).
- **Improving IRT convergence** by removing redundant items that inflate the effective number of parameters without adding information.

### Caveat

With 500+ bills per session, the wTO matrix is 500x500. Computing it is fast (it's just network algebra), but the number of flagged pairs could be enormous in sessions with many procedural votes. A two-stage approach — first filter to contested votes, then run UVA on the survivors — would be more practical.

---

## 5. Hierarchical EGA (hierEGA)

### What It Is

hierEGA discovers **multi-level dimensional structures** via a two-stage procedure:

**Lower-order stage:**
1. Estimate the network (GLASSO, TMFG, or BGGM).
2. Apply Louvain with consensus clustering (1000 iterations) to find lower-order dimensions.
3. Compute network loadings for each variable on its assigned community.

**Higher-order stage:**
1. Compute scores for each lower-order dimension (network scores or factor scores).
2. Run a second-level EGA on these dimension scores.
3. Community detection identifies higher-order dimensions that organize the lower-order factors.

Fit evaluation uses **genTEFI** (Generalized Total Entropy Fit Index) to compare hierarchical vs. correlated factor structures.

### Why This Matters for Tallgrass

This is the **most directly relevant** of Golino's methods to our pipeline. The central question in our 2D IRT work is: what does Dimension 2 represent? We've identified it as "establishment vs. contrarian" (the Tyson paradox — conservative legislators who vote Nay on routine party-line bills). But this is an *interpretation* imposed after fitting, not a structural discovery.

hierEGA could answer: **do roll-call votes naturally organize into lower-order policy domains (fiscal, social, procedural, institutional) that then organize into higher-order ideological dimensions?**

Concrete scenario for the 91st Kansas House:

1. **Lower-order EGA** might find 4-5 communities: fiscal policy votes, social/cultural votes, education policy votes, procedural/institutional votes, and veto overrides.
2. **Higher-order EGA** on these 4-5 dimension scores might find 2 higher-order dimensions: ideology (fiscal + social load together) and institutional loyalty (procedural + veto override load together).

This would structurally validate our 2D IRT finding while providing **interpretable dimension labels** rather than post-hoc "Dim 1 is ideology because it correlates with party."

### Comparison to Current Approach

| Aspect | Current (2D IRT + PCA) | hierEGA |
|---|---|---|
| Dimensionality estimate | Assumed (1D or 2D), validated by convergence | Discovered from data |
| Dimension interpretation | Post-hoc (correlate with party, PCA) | Structural (which bills cluster together) |
| Hierarchy | Flat (Dim 1, Dim 2 are unordered) | Explicit (lower-order → higher-order) |
| Stability assessment | 7-gate quality system | bootEGA on both levels |
| Model comparison | Tier cascade (convergence-based) | genTEFI (information-theoretic) |
| Computational cost | High (MCMC sampling) | Low (network estimation + community detection) |

### Potential Integration

hierEGA wouldn't *replace* our IRT models — it would serve as a **pre-IRT dimensionality oracle**. Run hierEGA before Phase 05 (or between Phase 02 PCA and Phase 05 IRT) to determine:

- How many dimensions to fit in IRT (currently hardcoded as 1 or 2).
- Which bills load on which dimension (informing beta initialization for 2D IRT).
- Whether the hierarchical structure suggests a bifactor model (one general factor + specific factors) rather than a correlated factors model.

This directly addresses a known limitation: our 2D IRT uses PLT identification (positive lower triangular constraint on the loading matrix), which imposes a specific rotation. If hierEGA's lower-order communities don't align with PLT's rotation, we may be fitting the wrong 2D structure.

---

## 6. Dynamic EGA (dynEGA)

### What It Is

dynEGA extends EGA to multivariate time series and longitudinal data. Instead of modeling how variables covary (static), dynEGA models how variables **change together** over time.

Method:
1. **Derivative estimation via GLLA** (Generalized Local Linear Approximation): time-delay embedding + Savitzky-Golay smoothing to estimate 0th (level), 1st (velocity), and 2nd (acceleration) order derivatives.
2. Standard EGA (GLASSO/TMFG + community detection) applied to the derivative matrix.
3. Three levels: individual (per-person), group, population.

### Why This Matters for Tallgrass

Our pipeline covers 8 bienniums (84th-91st, 2011-2026). We already have Phase 22 (time-series analysis) and Phase 24 (dynamic IRT), but neither uses network-based dimensionality tracking.

dynEGA could model **how the dimensional structure of Kansas voting evolves across sessions**:

- Did a 2nd dimension (establishment vs. contrarian) emerge only after the Republican supermajority solidified?
- Does dimensional instability correlate with session-specific events (e.g., Brownback tax cuts in the 87th, Medicaid expansion debates in the 90th)?
- Are there periods where the structure is transiently 3-dimensional (e.g., when a cross-party coalition forms on a specific issue)?

### Feasibility Concern

dynEGA requires time-series data — repeated measurements of the same variables over time. Roll-call votes don't repeat across sessions (different bills each biennium). Two approaches:

1. **Legislator-level time series.** Track each legislator's ideal point across sessions they serve in. This works for the ~30% of legislators who serve multiple terms, but produces sparse, short time series.
2. **Session-level structural summary.** Treat each session as a time point and track *structural* properties (number of dimensions, community composition, TEFI) over time. This is more like a descriptive time series of 16 data points (8 bienniums × 2 chambers) — too short for GLLA derivative estimation.

**Verdict:** dynEGA is less immediately applicable than static EGA/hierEGA. Our existing dynamic IRT approach (Phase 24) is better suited to the longitudinal structure of legislative data, where the latent trait (ideology) drifts continuously rather than the item set changing discretely.

---

## 7. Total Entropy Fit Index (TEFI)

### What It Is

TEFI applies **Von Neumann entropy** (from quantum information theory) to the correlation matrix, evaluated relative to a proposed dimensional structure. Von Neumann entropy generalizes Shannon entropy to density matrices (positive semidefinite, trace-1 matrices), treating the normalized correlation matrix as a quantum state.

Three variants:
- **EFI** — basic entropy fit
- **EFI.vn** — Von Neumann entropy version
- **TEFI.vn** — Total EFI.vn (recommended, penalizes over-extraction)

Lower TEFI = better fit. **Generalized TEFI (genTEFI)** extends to hierarchical structures, achieving .94-.95 balanced accuracy for distinguishing correct from incorrect hierarchical models.

### Critical Property

When more factors are extracted than truly exist, **only TEFI.vn maintains high accuracy** — traditional fit measures (RMSEA, CFI, TLI) fail to penalize over-extraction. This is directly relevant to our pipeline, where the question is often "is Dim 2 real or noise?"

### Why This Matters for Tallgrass

TEFI could serve as an **objective model comparison metric** between our dimensional structures:

| Comparison | Current method | TEFI alternative |
|---|---|---|
| 1D vs 2D IRT | Tier cascade (convergence-based) | TEFI(1D structure) vs TEFI(2D structure) — lower wins |
| Hierarchical vs flat IRT | Not compared directly | genTEFI(hierarchical) vs TEFI(correlated factors) |
| Which IRT dimension is ideology? | W-NOMINATE cross-validation (ADR-0123) | Compare TEFI of structures with Dim 1 vs Dim 2 as the party-separating axis |
| Optimal K | Not estimated (assumed 1 or 2) | TEFI across K=1,2,3,...; minimum identifies optimal |

### Implementation

TEFI is computationally simple — it requires only the correlation matrix and a dimension assignment vector. The formula:

```
S(rho) = -Tr(rho * log2(rho))
```

where `rho = R / Tr(R)` (normalized correlation matrix). In practice, compute from eigenvalues: `S = -sum(lambda_i * log2(lambda_i))` where `lambda_i` are eigenvalues of `rho`. TEFI decomposes this by dimension and sums. Total computation: one eigendecomposition of a p×p matrix — negligible compared to MCMC.

---

## 8. Network Loadings & the Latent Variable Debate

### The Theoretical Question

Golino positions network psychometrics not as replacing latent variable models (IRT, factor analysis) but as a **complementary framework with different causal assumptions**:

- **Latent variable (IRT):** Voting patterns arise from a **common latent cause** — ideology. A legislator's conservatism *causes* their Yea on gun bills, Nay on tax increases, Yea on abortion restrictions. All correlations between votes are explained by the latent trait.

- **Network model:** Voting positions are **causally coupled**. A legislator's position on gun control is connected to their position on crime policy, which connects to criminal justice reform — not because "conservatism" causes all three, but because positions on these issues *directly influence each other* through constituency pressure, logrolling, and ideological coherence.

For legislative voting, the network interpretation is arguably more realistic: legislators don't have an abstract "ideology score" that independently generates each vote. They have positions on specific issues that constrain and influence each other, mediated by party discipline, constituency pressure, and strategic coalition-building.

### Statistical Equivalence

For binary data, the class of factor models (IRT) is **statistically equivalent** to the class of network models (Ising models). The same data can be generated by either. The choice is theoretical/interpretive, not empirical. This means EGA and IRT will generally agree on dimensionality when both are correctly specified — disagreement signals model misspecification in one or both.

### Network Loadings

Christensen & Golino developed **network loadings** — per-item, per-community decompositions of node strength that are analogous to factor loadings. Revised in 2025 to correct for the reduction in partial correlation magnitude as variable count increases. Network loadings can be computed without rotation (a free parameter in factor analysis), eliminating one analytic degree of freedom.

**For our pipeline:** Network loadings on the EGA communities would provide an alternative bill-dimension assignment to our IRT discrimination parameters (beta). Comparing the two — which bills load most strongly on Dim 1 in IRT vs. which have the highest network loading on Community 1 in EGA — would validate or challenge our IRT structure.

---

## 9. Current Pipeline: What We Already Have

Before proposing additions, it's important to inventory what the pipeline already does that overlaps with or substitutes for EGA methods.

### Dimensionality Assessment (Current)

| Phase | Method | Output | Role |
|---|---|---|---|
| 02 (PCA) | Principal components on standardized vote matrix | Scree plot, PC1-5 scores, party separation d | Initial dimensionality screen |
| 03 (MCA) | Multiple correspondence analysis | Greenacre-corrected eigenvalues, horseshoe detection | Categorical validation of PCA |
| 05 (1D IRT) | Bayesian 2PL via nutpie | Ideal points, convergence diagnostics | Primary ideology scores |
| 06 (2D IRT) | Bayesian M2PL with PLT | 2D ideal points, Dim 1 vs Dim 2 separation | Secondary dimension discovery |
| 08 (PPC) | Q3 residual correlations, GMP, APRE | Local dependence flags | Diagnostic for 1D vs 2D |
| Canonical routing | Tier 1/2/3 quality gates + W-NOMINATE cross-validation | Final ideal point selection | Dimension choice arbiter |

### Network Analysis (Current)

| Phase | Method | Output | Role |
|---|---|---|---|
| 11 (Network) | Kappa agreement network, Leiden/CPM community detection, 6 centrality measures, disparity filter backbone | Community assignments, centrality scores, polarization metric | Relational structure discovery |
| 12 (Bipartite) | Bill-legislator bipartite network, BiCM backbone, Newman projection | Bill communities, bridge bills, hidden alliances | Bill-centric structure |

### Clustering & Faction Discovery (Current)

| Phase | Method | Output | Role |
|---|---|---|---|
| 09 (Clustering) | 5 methods (hierarchical, K-means, GMM, spectral, HDBSCAN) | Optimal K, cluster assignments, party loyalty | Faction existence test |
| 10 (LCA) | Bernoulli mixture model (EM), BIC selection | K, class profiles, Salsa test | Statistically principled faction confirmation |

### What's Notable

- We already use **Leiden** community detection (Phase 11) — the same algorithm available in EGA.
- We already compute **Kappa agreement networks** — but these are *observed* networks (direct vote agreement), not *estimated* networks (conditional dependencies via GLASSO). This is a crucial distinction. Kappa networks conflate direct and indirect associations; GLASSO networks isolate conditional dependencies.
- Phase 08 (PPC) computes Q3 (local dependence) but only globally — not per bill-pair or per legislator-pair.
- Phase 11 operates **independently** from IRT. There is no feedback from network structure to IRT dimensionality, and no comparison of network communities to IRT dimensions.

---

## 10. Gap Analysis: Where EGA Methods Add Value

### Gap 1: No Conditional Dependency Network

**Current:** Phase 11 builds networks from Cohen's Kappa (observed pairwise agreement). Two legislators who both vote conservative on fiscal issues will have a strong Kappa edge even if they disagree on social issues — Kappa doesn't condition on the shared latent trait.

**EGA's approach:** GLASSO estimates a **partial correlation network** — edge between legislators i and j represents their association *after controlling for all other legislators*. For bills, GLASSO on the tetrachoric correlation matrix produces edges that represent bill-bill associations conditional on all other bills. This is the conditional dependence structure that IRT's local independence assumption says should be empty (all edges zero) if the model is correct.

**Value:** A GLASSO network on IRT residuals would directly visualize local dependence violations — non-zero edges are associations unexplained by the fitted IRT model. This is strictly more informative than Phase 08's aggregate Q3 statistic.

### Gap 2: No Pre-IRT Dimensionality Estimate

**Current:** We fit 1D IRT (Phase 05) and 2D IRT (Phase 06) and then use convergence diagnostics + external validation to choose. This is model-comparison, not dimensionality *estimation*.

**EGA's approach:** Estimate K directly from the data before fitting any model. EGA on the tetrachoric bill correlation matrix produces K and the bill-dimension assignments in seconds (no MCMC needed).

**Value:** If EGA says K=1 for a chamber-session, we can skip Phase 06 entirely (saving 10-30 minutes of MCMC per chamber). If EGA says K=3, we're missing a dimension. Either way, it informs rather than replaces the IRT workflow.

### Gap 3: No Per-Item Stability Assessment

**Current:** The 7-gate quality system (ADR-0118) detects axis instability at the *session* level — the entire session is flagged. We don't know which specific bills are causing the instability.

**bootEGA's approach:** Per-item stability scores identify which bills are assigned to different dimensions across bootstrap replicates. Bills with stability < 0.70 are dimensionally ambiguous — they might load on the ideology axis in one resample and the institutional axis in another.

**Value:** Targeted bill removal/flagging. Instead of `--contested-only` (blunt threshold), remove only bills that bootEGA identifies as dimensionally unstable. This preserves more data while improving dimensional clarity.

### Gap 4: No Hierarchical Dimensionality Discovery

**Current:** Our hierarchy is *imposed*: party → legislator (hierarchical IRT). The dimensional structure itself is flat (Dim 1, Dim 2).

**hierEGA's approach:** Discovers hierarchy from the data — lower-order bill communities organize into higher-order dimensions.

**Value:** Structural evidence for or against the 2D model. If hierEGA finds 4 lower-order bill communities that collapse into 1 higher-order dimension, our 2D IRT is likely overfitting. If they collapse into 2, we have structural validation. If the hierarchy is more complex (bifactor), we may need a different IRT model entirely.

### Gap 5: No Information-Theoretic Model Comparison

**Current:** Model comparison uses convergence diagnostics (R-hat, ESS), rank correlations (Spearman with PCA/W-NOMINATE), and Cohen's d (party separation). These are *necessary conditions* for a good model, not measures of structural fit.

**TEFI's approach:** A single number that quantifies how well a proposed dimensional structure fits the correlation matrix, with proper penalization for over-extraction.

**Value:** TEFI(1D) vs TEFI(2D) provides a direct, interpretable comparison that doesn't depend on MCMC convergence. It's fast (one eigendecomposition) and could be computed in Phase 02 alongside PCA.

### Gap 6: Network Analysis Disconnected from IRT

**Current:** Phase 11 (network) and Phase 05/06 (IRT) operate independently. Network communities are compared to party labels (modularity) but not to IRT dimensions.

**Potential bridge:** After fitting IRT, compute predicted Kappa from ideal points and compare to observed. Build a network on the residuals. Non-zero residual edges identify co-voting patterns unexplained by ideology — geographic coalitions, committee effects, logrolling.

---

## 11. Implementation Feasibility

### Python Availability

There is **no official Python port of EGAnet**. The package is R-only. However, all building blocks exist in Python:

| Component | Python library | Notes |
|---|---|---|
| Tetrachoric correlations | `semopy`, `factor_analyzer`, or custom (2x2 tables → `scipy.optimize`) | `factor_analyzer` only handles binary; `semopy` handles mixed |
| GLASSO + EBIC | `sklearn.covariance.GraphicalLasso`, `sklearn.covariance.GraphicalLassoCV` | scikit-learn's implementation uses coordinate descent; EBIC model selection needs a custom loop over lambda values |
| TMFG | No standard package | Algorithm is simple to implement (~100 lines): greedy planar graph construction |
| Community detection | `igraph` (Walktrap, Louvain, Leiden), `leidenalg` | We already use `igraph` + `leidenalg` in Phase 11 |
| Von Neumann entropy (TEFI) | `numpy`/`scipy` eigendecomposition | Trivial: ~10 lines of code |
| Bootstrap | `numpy` resampling | Standard |
| Network loadings | Custom (sum of absolute edge weights within community) | ~20 lines using `igraph` edge weights |

### Implementation Effort Estimates

| Method | Effort | Dependencies | Risk |
|---|---|---|---|
| EGA (static) | Medium | `sklearn`, `igraph`, tetrachoric correlation | Tetrachoric estimation at scale (500+ bills) needs validation |
| bootEGA | Low (once EGA exists) | Same + parallel loop | Parametric bootstrap needs multivariate normal assumption check |
| TEFI | Low | `numpy` only | Well-defined formula; hard to get wrong |
| UVA | Medium | EGA + weighted topological overlap | wTO computation on large networks needs efficient implementation |
| hierEGA | High | EGA + network scores + second-level EGA | Two-stage procedure with multiple algorithmic choices; debugging is harder |
| dynEGA | High | EGA + GLLA derivative estimation | GLLA is non-trivial; applicability to legislative data is uncertain |

### Recommended Implementation Strategy

Build a `analysis/ega/` module (not a numbered phase yet) with:

1. `tetrachoric.py` — Tetrachoric correlation matrix from binary vote data.
2. `glasso_ebic.py` — GLASSO network estimation with EBIC model selection.
3. `ega.py` — Core EGA: network estimation + community detection + unidimensional check.
4. `boot_ega.py` — Bootstrap wrapper producing stability metrics.
5. `tefi.py` — TEFI computation from correlation matrix + dimension assignments.
6. `uva.py` — Weighted topological overlap for redundancy detection.

This modular approach lets us validate each component independently before assembling hierEGA (which chains them).

### R Bridge Alternative

If Python implementation proves too costly, we could call EGAnet directly via `rpy2`:

```python
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
eganet = importr("EGAnet")
result = eganet.EGA(data_matrix, model="glasso", algorithm="walktrap")
```

This adds an R dependency but gives us the full, tested EGAnet implementation. The Justfile would need `R` and `EGAnet` as prerequisites. Given our "check for existing open source solutions first" philosophy, this may be the pragmatic first step.

---

## 12. Recommendations

### Tier 1: High Value, Low Risk (Implement Soon)

**1. TEFI as Phase 02 Supplement**
Add TEFI computation to Phase 02 (PCA) or as a standalone step between PCA and IRT. Compare TEFI(K=1) vs TEFI(K=2) vs TEFI(K=3) on the tetrachoric correlation matrix. Use as an advisory signal alongside the scree plot. ~50 lines of code, `numpy` only, no new dependencies.

**2. Residual Network Analysis in Phase 11**
After IRT (Phase 05/06), compute predicted pairwise agreement from ideal points. Subtract from observed Kappa. Build a network on the residuals and run Leiden community detection. Report which legislator pairs have unexplained co-voting. This bridges the IRT/network gap using infrastructure that already exists.

**3. Q3 Per-Pair Heatmap in Phase 08**
Extend Phase 08 to report Q3 not just as a distribution but as a per-bill-pair and per-legislator-pair matrix. Visualize as a heatmap with bills sorted by IRT dimension loading. Clusters of high Q3 that don't align with the IRT structure indicate missing dimensions.

### Tier 2: High Value, Medium Risk (Implement After Validation)

**4. Static EGA as Pre-IRT Dimensionality Oracle**
Implement core EGA (GLASSO + EBIC + Walktrap/Leiden) in Python. Run on each chamber-session's tetrachoric correlation matrix. Compare EGA's K estimate to the canonical routing decision for all 28 chamber-sessions. If they agree in 25+/28 cases, EGA is a reliable pre-IRT screen. If they disagree, investigate — EGA may be catching errors our quality gates miss, or vice versa.

**5. bootEGA for Axis Instability Diagnosis**
Run bootEGA on the 7 Senate sessions flagged by ADR-0118 (78th-83rd, 88th). Compare per-bill stability scores to the bills that drive the PCA axis flip. If the same bills are unstable in bootEGA, we have convergent validation. If different bills are flagged, bootEGA is finding instability our current methods miss.

**6. UVA for Principled Bill Filtering**
Run UVA on the full bill set (before contested-only filtering). Compare the wTO-flagged redundant bill pairs to our current contested-only filter. If UVA removes a similar set but with better justification (conditional dependence vs. marginal dissent rate), adopt it as the primary filter.

### Tier 3: Potentially High Value, High Risk (Research First)

**7. hierEGA as Structural Validation**
Implement the full two-stage hierEGA procedure. Run on 2-3 well-understood sessions (e.g., 91st House, 88th Senate, 84th Senate) and compare:
- Lower-order communities vs. bill topic labels (if available).
- Higher-order structure vs. our 1D/2D IRT result.
- genTEFI(hierarchical) vs. TEFI(correlated 2-factor) vs. TEFI(unidimensional).

This is the most ambitious integration and the most likely to produce novel insights — but also the most likely to require iteration and debugging.

**8. Network Loadings as IRT Beta Validation**
Compute network loadings from the EGA network and compare to IRT discrimination parameters (beta). Bills with high network loading on Community 1 should have high |beta| in 1D IRT. Discrepancies identify bills where IRT and network methods disagree about dimensional assignment — these are the bills worth investigating.

### Not Recommended (For Now)

**dynEGA:** The longitudinal structure of legislative data (different bills each session, discrete biennium boundaries) is a poor fit for GLLA derivative estimation, which assumes continuous time series of the same variables. Our existing dynamic IRT (Phase 24) handles the legislator-level temporal dynamics better.

**Full Pipeline Replacement:** EGA/network psychometrics should *complement* IRT, not replace it. IRT provides individual-level scores with uncertainty quantification (posterior distributions), which EGA does not. Network loadings approximate factor scores but lack the probabilistic interpretation that makes IRT ideal points useful for downstream analysis (prediction, temporal tracking, cross-chamber comparison).

---

## 13. Implementation Status (2026-03-25)

The following recommendations from this deep dive have been implemented:

| Recommendation | Tier | Status | Location |
|---|---|---|---|
| TEFI as Phase 02 supplement | Tier 1 | **Implemented** | `analysis/02_pca/pca.py` (TEFI computation + curve plot) |
| Residual network in Phase 11 | Tier 1 | **Implemented** | `analysis/11_network/network.py` (IRT-adjusted Kappa residuals) |
| Q3 per-pair heatmap in Phase 08 | Tier 1 | **Implemented** | `analysis/08_ppc/ppc.py` (heatmap + top violations table) |
| Static EGA as pre-IRT oracle | Tier 2 | **Implemented** | `analysis/02b_ega/` (Phase 02b, full EGA pipeline) |
| bootEGA for stability | Tier 2 | **Implemented** | `analysis/ega/boot_ega.py` (500 replicates, item stability) |
| UVA for bill filtering | Tier 2 | **Implemented** | `analysis/ega/uva.py` (wTO redundancy detection) |
| hierEGA structural validation | Tier 3 | Not yet | Requires two-stage procedure |
| Network loadings vs IRT beta | Tier 3 | Not yet | Available in `analysis/ega/ega.py` (network_loadings field) |
| dynEGA for cross-session | — | Not recommended | Poor fit for legislative data structure |

**ADRs:** ADR-0124 (EGA Phase 02b), ADR-0125 (TEFI/residual network/Q3 integration).

---

## 14. References

### Core EGA Papers

- Golino, H., & Epskamp, S. (2017). Exploratory graph analysis: A new approach for estimating the number of dimensions in psychological research. *PLoS ONE*, 12(6), e0174035.
- Golino, H., Shi, D., Christensen, A. P., Garrido, L. E., et al. (2020). Investigating the performance of exploratory graph analysis and traditional techniques to identify the number of latent factors: A simulation and tutorial. *Psychological Methods*, 25(3), 292-320.

### bootEGA & Stability

- Christensen, A. P., & Golino, H. (2021). Estimating the stability of psychological dimensions via bootstrap exploratory graph analysis: A Monte Carlo simulation and tutorial. *Psych*, 3(3), 479-500.

### UVA

- Christensen, A. P., Garrido, L. E., & Golino, H. (2023). Unique variable analysis: A network psychometrics method to detect local dependence. *Multivariate Behavioral Research*, 58(6).

### hierEGA

- Jimenez, M., Abad, F. J., Garcia-Garzon, E., Golino, H., Christensen, A. P., & Garrido, L. E. (2023). Dimensionality assessment in bifactor structures with multiple general factors: A network psychometrics approach. *Psychological Methods*.

### dynEGA

- Golino, H., Christensen, A. P., Moulder, R., Kim, S., & Boker, S. M. (2022). Modeling latent topics in social media using Dynamic Exploratory Graph Analysis: The case of the right-wing and left-wing trolls in the 2016 US elections. *Psychometrika*, 87(1), 156-187.

### TEFI

- Golino, H., Moulder, R., Shi, D., et al. (2021). Entropy fit indices: New fit measures for assessing the structure and dimensionality of multiple latent variables. *Multivariate Behavioral Research*, 56(6), 874-902.
- Golino, H., Jimenez, M., Garrido, L. E., & Christensen, A. P. (2024). Generalized Total Entropy Fit Index. *PsyArXiv*.

### Network Loadings

- Christensen, A. P., & Golino, H. (2021). On the equivalency of factor and network loadings. *Behavior Research Methods*, 53(4), 1563-1580.
- Christensen, A. P., et al. (2025). Revised network loadings. *Behavior Research Methods*.

### EGAnet Package

- Golino, H., & Christensen, A. P. (2025). EGAnet: Exploratory Graph Analysis — A Framework for Estimating the Number of Dimensions in Multivariate Data Using Network Psychometrics. R package version 2.4.0. CRAN. https://r-ega.net/

### Background: Network Psychometrics

- Epskamp, S., & Fried, E. I. (2018). A tutorial on regularized partial correlation networks. *Psychological Methods*, 23(4), 617-634.
- Borsboom, D. (2017). A network theory of mental disorders. *World Psychiatry*, 16(1), 5-13.

### Legislative Network Analysis (Non-Golino)

- Waugh, A. S., Pei, L., Fowler, J. H., Mucha, P. J., & Porter, M. A. (2009). Party polarization in Congress: A network science approach. *arXiv:0907.3509*.
- Serrano, M. A., Boguna, M., & Vespignani, A. (2009). Extracting the multiscale backbone of complex weighted networks. *PNAS*, 106(16), 6483-6488.
