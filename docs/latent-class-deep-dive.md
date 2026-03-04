# Latent Class Analysis Deep Dive

A literature survey, ecosystem evaluation, and integration design for Phase 10 of the Kansas Legislature vote analysis pipeline.

**Date:** 2026-02-28
**Scope:** Latent Class Mixture Models for legislative roll-call voting data
**Status:** Research complete, implementation planned

---

## Executive Summary

Latent Class Analysis (LCA) is a finite mixture model for categorical data that identifies discrete, unobserved subgroups within a population. Unlike IRT (which places legislators on a continuous ideological spectrum), LCA asks: *how many qualitatively distinct types of legislators are there, and who belongs to each?*

For the Tallgrass pipeline, LCA serves as the probabilistic complement to Phase 5's distance-based clustering. Phase 5 already found that k=2 is optimal (the party split) and that within-party variation is continuous, not factional. LCA provides the statistically principled test of that finding: if a Bernoulli mixture model with BIC-selected K recovers only the party split, we have strong model-based confirmation that discrete factions do not exist beneath the party boundary. If it finds K>2, we need to determine whether the extra classes represent qualitatively distinct voting patterns or simply discretize the IRT continuum (the "Salsa effect").

**Recommended library:** StepMix 2.2.3 (Python, MIT, JSS 2025). The only dedicated, peer-reviewed Python LCA package. scikit-learn API, native binary data support, FIML missing data handling, built-in BIC/AIC for class enumeration. Stays in the Python ecosystem; no R subprocess needed.

**Key risk:** The Lubke & Neale (2006) impossibility result — you cannot empirically distinguish categorical from continuous latent structure using model fit alone. Our interpretation must rest on whether class profiles are qualitatively distinct (crossover voting patterns) or merely quantitatively graded (same pattern, different intensity).

---

## 1. What LCA Is

### 1.1 Origin

LCA originates with Paul Lazarsfeld (1950), who introduced the latent class model as part of his "latent structure analysis" framework for building typologies from dichotomous survey variables. The model was computationally intractable until Leo Goodman (1974) operationalized it using maximum likelihood via an early application of the EM algorithm — three years before Dempster, Laird, and Rubin's canonical 1977 paper. McCutcheon's 1987 Sage monograph (*Latent Class Analysis*, QASS series) became the standard introductory text. Hagenaars and McCutcheon's 2002 edited volume (*Applied Latent Class Analysis*, Cambridge UP) remains the definitive reference.

### 1.2 The Generative Model

LCA assumes the population consists of C unobserved subgroups. The observed response pattern for individual i across J binary indicators arises from a mixture:

```
P(Y_i) = SUM_{c=1}^{C} pi_c * PROD_{j=1}^{J} p_jc^{y_ij} * (1 - p_jc)^{1 - y_ij}
```

Where:
- `pi_c` = prevalence of latent class c (SUM pi_c = 1)
- `p_jc` = probability of a Yea vote on bill j for a member of class c
- C = number of latent classes
- J = number of roll-call votes (binary indicators)

The model parameters are estimated via EM:
- **E-step:** Compute posterior class membership for each legislator given current parameters (Bayes' theorem)
- **M-step:** Update class prevalences and item-response probabilities using the posterior weights

### 1.3 The Local Independence Assumption

The core structural assumption: conditional on class membership, all votes are mutually independent. The observed correlations among roll calls are *entirely* explained by the latent class variable. This is the categorical-data analog of the conditional independence assumption in factor analysis.

When local independence is violated (items remain correlated within classes), the model may overestimate the number of classes to absorb the residual dependence (Magidson & Vermunt, 2004). Bivariate residuals > 4.0 indicate significant violations (Oberski, 2016).

### 1.4 Identifiability

For J binary indicators and C classes, identifiability requires:
- **Necessary:** C(J+1) - 1 <= 2^J - 1 (free parameters must not exceed unique response patterns)
- **Sufficient:** The information matrix must be non-singular (checked empirically)
- **Minimum:** 3 binary indicators for a 2-class model (Goodman, 1974)

With legislative vote matrices (J >> 100), identifiability is never a concern. The challenge is the opposite: overparameterization with sparse response patterns.

**Label invariance:** The likelihood is symmetric under permutations of class labels — any reordering of (pi_c, p_jc) pairs yields identical fit. This is a structural symmetry of all mixture models. In EM estimation, random restarts resolve it (each run converges to a particular labeling). In Bayesian estimation, it causes the label switching problem (Section 2.4).

---

## 2. LCA vs. IRT: The Deep Structural Comparison

### 2.1 Continuous vs. Discrete Latent Variables

Both LCA and IRT are latent variable models for binary observed data. The key structural difference:

| Feature | IRT (Phase 4) | LCA (Phase 10) |
|---------|---------------|-----------------|
| Latent variable | Continuous (xi on the real line) | Categorical (C discrete classes) |
| What it estimates | Ideal points + bill parameters | Class memberships + class-specific vote profiles |
| Assumption | Monotonic P(Yea) as a function of ideology | Local independence within classes |
| Output | Legislator ranking/scaling | Legislator typology/clustering |
| Dimensionality | 1-2 continuous dimensions | K unordered classes |
| Missing data | Excluded from likelihood | FIML or pairwise deletion |
| Uncertainty | Posterior intervals on xi | P(class c \| votes) for each legislator |

IRT answers: *where is each legislator on the ideological spectrum?*
LCA answers: *how many distinct types of legislators exist, and who belongs to each?*

### 2.2 The Lubke & Neale Impossibility Result

Lubke and Neale (2006) proved that a K-class latent class model and a (K-1)-factor model can fit equally well, regardless of which is the true data-generating process. There is no purely empirical way to distinguish categorical from continuous latent structure. The choice must rest on theoretical considerations.

This means: if our IRT model (continuous) fits well, an LCA model (discrete) with the right number of classes will also fit well — and vice versa. The question is not "which fits better?" but "which framing is more useful for understanding Kansas legislative politics?"

### 2.3 When LCA Reveals Something IRT Misses

LCA adds genuine value when:

1. **Issue-specific crossover patterns exist** — a subset of Republicans who vote with Democrats on education but not taxes, producing a qualitatively distinct voting profile (not just "more moderate")
2. **Multi-dimensional structure collapses differently** — if the underlying space is 2D (economic + social), LCA might find 4 quadrant-types that a 1D IRT model misses
3. **A genuine faction operates** — a Freedom Caucus subgroup with categorically different voting behavior (defecting on procedural votes, leadership bills) rather than just being "further right"

### 2.4 The "Salsa Effect" Risk

The Practitioner's Guide (Nylund-Gibson & Choi, 2018) warns about the "Salsa effect": when class profiles show parallel lines (same voting pattern at different intensities), the model has simply discretized a continuum. In a legislative context, if Class 1 votes Republican 90% of the time, Class 2 votes Republican 70%, and Class 3 votes Republican 40%, you have just binned a continuous liberal-conservative dimension into three arbitrary slices.

**This is the primary risk for our data.** Phase 5 already found that within-party variation is continuous. LCA may simply confirm that finding by recovering K=2 (the party split) and failing to justify K>2 with qualitatively distinct profiles.

**But confirming the null is itself valuable.** A formal Bernoulli mixture model with BIC-based class selection is a stronger statement than k-means silhouette analysis. If BIC selects K=2, we have model-based evidence — not just heuristic evidence — that the party split is the only discrete structure in the data.

---

## 3. LCA in Political Science: A Sparse Literature

### 3.1 LCA Is Rare in Legislative Roll-Call Analysis

This is the most important finding from the literature search. **LCA is not commonly used for analyzing legislative roll-call votes.** The field is overwhelmingly dominated by IRT, NOMINATE, and W-NOMINATE. LCA has been applied extensively in other political science domains — survey analysis, political participation typologies, voter behavior classification — but its direct application to roll-call voting is sparse.

The absence is theoretically motivated: legislative ideology is generally understood as continuous (not discrete), and the dominant framework (spatial voting theory, Poole & Rosenthal) is explicitly spatial/continuous. LCA's discrete framing doesn't align with the field's theoretical priors.

### 3.2 What Does Exist: Discrete/Coalition-Based Approaches

Several researchers have explored mixture approaches to voting data, though often not under the "LCA" label:

**Choi & Goldsmith (2022):** Dynamic Dirichlet Process Mixture Models on UN General Assembly human rights roll-call votes (1992-2017). Bayesian nonparametric approach identifies 3-5 voting coalitions per year without pre-specifying K. Key result: DDPM achieved F1=0.87 vs IRT's F1=0.85, suggesting coalition-based models can match or outperform continuous ideal-point models for international voting where coalitions (not ideology) are the natural unit.

**Vermunt (2010):** Published in *Political Analysis* — the methodology journal for the field. Introduced the bias-adjusted 3-step approach for LCA with covariates. While the paper focuses on methodology rather than legislative applications, its publication venue signals the field considers LCA a legitimate tool.

**Interval/niche models (Pettit et al., 2020):** Replace point ideal points with interval orders. Found that one-dimensional interval models suffice for recent U.S. Senate voting, but low dimensionality was historically the exception. Conceptually closer to discrete typology than continuous scaling.

### 3.3 Why LCA Still Has Value Here

Despite its rarity in the legislative literature, LCA serves three purposes in our pipeline:

1. **Formal null hypothesis test.** Phase 5's finding that "within-party variation is continuous" was based on k-means silhouette analysis and GMM BIC selection. LCA with BIC provides a more principled test using the correct generative model for binary data (Bernoulli mixture, not Gaussian mixture applied to IRT scores).

2. **Soft classification.** LCA returns P(class | votes) for each legislator — a probability vector, not a hard assignment. This naturally identifies legislators who straddle class boundaries (potential swing voters, coalition-builders).

3. **Methodological completeness.** The 29-method analytic framework includes LCA (#28). Implementing it with the canonical approach and documenting the (likely null) result is more valuable than leaving a gap.

---

## 4. Model Selection: Choosing the Number of Classes

### 4.1 The Nylund et al. (2007) Consensus

The definitive simulation study, cited >3,000 times. Key findings for class enumeration:

| Criterion | Performance | Role |
|-----------|-------------|------|
| **BIC** | Best overall; penalizes complexity proportional to log(N) | **Primary criterion** — select lowest BIC |
| **SABIC** | Sometimes outperforms BIC; less severe penalty | Secondary confirmation |
| **AIC** | Too liberal; tends to overfit (too many classes) | Supplement only |
| **Bootstrap LRT (BLRT)** | Most powerful hypothesis test | **Strongest confirmation** — tests K vs K-1 |
| **Lo-Mendell-Rubin LRT** | Elevated Type I error in some conditions | Secondary |
| **Entropy** | Measures classification quality, NOT model selection | **Descriptive only** — report but don't use to select K |
| **ICL** (Integrated Completed Likelihood) | BIC with entropy penalty; favors well-separated classes | Useful when separation matters |

### 4.2 Practical Strategy

Following the Practitioner's Guide (Nylund-Gibson & Choi, 2018) and Weller, Bowen, & Faubert (2020):

1. **Always test K=1** as baseline. If it fits well, LCA may not be appropriate.
2. Fit models from K=1 to K=K_max (typically 6-8).
3. Examine BIC (primary) and SABIC together — look for the elbow.
4. Check entropy for classification quality (>0.8 acceptable, >0.6 moderate).
5. Evaluate substantive interpretability: each class must have a meaningful profile.
6. Minimum class size >5% of sample (prevent overfitting to outliers).
7. Use at least 50 random starts, replicate best log-likelihood at least 20 times (guard against local maxima).

### 4.3 Thresholds and Rules of Thumb

| Metric | Threshold | Source |
|--------|-----------|--------|
| Entropy | >0.8 acceptable; >0.6 moderate | Clark & Muthén (2009) |
| Average posterior probability per class | >0.7 minimum; >0.8 preferred | Nagin (2005) |
| Minimum N | >=300 preferred | Nylund-Gibson & Choi (2018) |
| Random starts | >=50; replicate best LL >=20 times | Nylund-Gibson & Choi (2018) |
| Minimum class size | >5% of sample | Substantive convention |
| Bivariate residual threshold | >4.0 indicates local dependence | Oberski (2016) |

Our legislative data (N~130-170 per chamber, J~200-400 bills) is at the lower end of the recommended sample size range. This is mitigated by the large number of indicators, but we should interpret 4+ class solutions with caution.

---

## 5. Open-Source Ecosystem

### 5.1 Python

| Package | Algorithm | Binary Native | Missing Data | Status |
|---------|-----------|---------------|--------------|--------|
| **StepMix** | EM (1/2/3-step) | Yes | FIML | **Recommended** — JSS 2025, MIT, scikit-learn API |
| pomegranate | EM | Yes (Categorical) | No | General-purpose; no LCA workflow |
| PyMC (custom) | MCMC/NUTS | Yes (manual) | Yes | Full Bayesian; ~50-100 lines model code |
| scikit-learn GMM | EM/VB | **No** (Gaussian) | No | **Wrong model** for binary data |

**StepMix** is the clear choice for Python LCA:
- Published in Journal of Statistical Software (Vol. 113, Issue 8, 2025)
- scikit-learn-compatible API (`fit` / `predict` / `predict_proba`)
- Native binary measurement model (`measurement="binary"`)
- FIML for missing data (`measurement="binary_nan"`)
- Built-in BIC/AIC for class enumeration
- Supports 1/2/3-step estimation with covariates and distal outcomes
- Bias-adjusted 3-step (BCH/ML corrections for classification error)
- MIT license, actively maintained (latest: v2.2.3, July 2025)

### 5.2 R (Reference Only)

| Package | Algorithm | Notes |
|---------|-----------|-------|
| **poLCA** | EM+NR | Canonical LCA (~1,200 citations). Co-authored by Jeffrey Lewis (of NOMINATE). JSS 2011, updated Feb 2026 |
| BayesLCA | EM/Gibbs/VB | Bayesian binary LCA. JSS 2014 |
| glca | EM+NR | Multi-group LCA with measurement invariance |
| flexmix | EM (pluggable) | General mixture framework |

poLCA is the field standard, but StepMix replicates its core functionality in Python. No R subprocess needed for Phase 10 — unlike Phase 17 (W-NOMINATE) where there is no Python equivalent.

### 5.3 Why Not Bayesian LCA?

A PyMC implementation would integrate naturally with our existing MCMC infrastructure and provide full posterior uncertainty. However:

1. **Label switching** is a severe problem for Bayesian mixture models — the posterior is symmetric under permutations of class labels, causing inflated R-hat and meaningless posterior means. Solutions (ordering constraints, post-hoc relabeling) add complexity.
2. **Discrete latent variables** cannot be sampled by NUTS/HMC. They must be marginalized out, which is straightforward for LCA but adds model-writing overhead.
3. **Model comparison** (selecting K) requires fitting multiple models and computing WAIC/LOO across them — more complex than EM + BIC.
4. **The question we're asking** (how many classes? which class does each legislator belong to?) is well-served by EM + BIC. We don't need full posterior distributions on class membership probabilities for this phase.

StepMix's EM approach is simpler, faster, and sufficient for our needs. If the EM results reveal something surprising (e.g., a genuine 3-class structure), we could follow up with a Bayesian model for uncertainty quantification.

---

## 6. Diagnostics and Interpretation

### 6.1 Class Profile Inspection

The most important diagnostic is visual: plot each class's item-response probabilities (P(Yea | class c) for each bill) and check whether profiles are qualitatively different or merely quantitatively graded.

**Qualitatively distinct** (genuine factions): Class A votes Yea on bills {1,3,7} and Nay on {2,4,6}; Class B has a different — not simply scaled — pattern. This would indicate LCA has found structure IRT misses.

**Quantitatively graded** (Salsa effect): Class A votes Yea on most bills at rate 0.9; Class B at rate 0.7; Class C at rate 0.4. Same pattern, different intensity. The model has discretized a continuum.

### 6.2 Cross-Validation Against IRT

For each LCA class, compute the mean IRT ideal point of its members. If classes line up monotonically along the IRT dimension (class 1 = most liberal, class K = most conservative), LCA is likely redundant with IRT. If classes overlap in IRT space or cross party lines, LCA has found genuinely different structure.

### 6.3 Local Independence Check

Compute bivariate residuals for each pair of high-loading bills within each class. Residuals > 4.0 indicate that the class structure doesn't fully explain vote correlations — either more classes are needed, or the local independence assumption is violated for those bill pairs.

### 6.4 Classification Table

The C x C matrix of average posterior probabilities (assigned-class by true-class) should have high diagonal entries and low off-diagonal entries. This measures how cleanly legislators separate into distinct types.

---

## 7. Integration with the Tallgrass Pipeline

### 7.1 Relationship to Phase 5 (Clustering)

Phase 10 is a natural extension of Phase 5, applying model-based clustering (LCA) where Phase 5 used distance-based (hierarchical), centroid-based (k-means), density-based (HDBSCAN), and Gaussian model-based (GMM) methods.

**Input:** Binary vote matrix from EDA (same as Phase 5's hierarchical clustering input), not IRT scores (which Phase 5 uses for k-means/GMM). This is the key difference — LCA operates on the raw binary data with the correct generative model, rather than on a continuous projection of it.

**Output:** Class assignments, class membership probabilities, class-specific vote profiles, BIC/entropy model selection results.

### 7.2 Upstream Dependencies

- **Phase 1 (EDA):** Vote matrix, legislator metadata, filtering parameters
- **Phase 4 (IRT):** Ideal points for cross-validation (are classes monotonic in IRT space?)
- **Phase 5 (Clustering):** ARI comparison (do LCA classes agree with k-means/hierarchical clusters?)

### 7.3 Downstream Consumers

- **Phase 11 (Synthesis):** LCA class assignments as an additional clustering perspective
- **Phase 12 (Profiles):** LCA membership probabilities as a per-legislator feature

### 7.4 What LCA Adds Beyond Phase 5

| Feature | Phase 5 (Clustering) | Phase 10 (LCA) |
|---------|---------------------|----------------|
| Input data | Kappa distances or IRT scores | Raw binary vote matrix |
| Generative model | GMM (Gaussian, wrong for binary) | Bernoulli mixture (correct for binary) |
| K selection | Silhouette, elbow, BIC on GMM | BIC on Bernoulli mixture (principled) |
| Output | Hard assignments | Soft assignments (P(class \| votes)) |
| Missing data | Pairwise deletion (Kappa) | FIML (StepMix native) |
| Profile interpretation | Cluster centroids in IRT space | Class-specific P(Yea) for each bill |

---

## 8. Implementation Design

### 8.1 StepMix Integration

```python
from stepmix.stepmix import StepMix
import numpy as np

# Binary vote matrix: legislators x bills, values 0/1/NaN
# StepMix FIML handles NaN natively with measurement="binary_nan"
model = StepMix(
    n_components=k,
    measurement="binary_nan",
    n_init=50,          # 50 random starts (Nylund-Gibson recommendation)
    max_iter=1000,
    abs_tol=1e-10,
    rel_tol=1e-10,
    random_state=42,
)
model.fit(vote_matrix)

# Soft assignments
probs = model.predict_proba(vote_matrix)   # N x K matrix
# Hard assignments
labels = model.predict(vote_matrix)         # N-length array
# Model selection
bic = model.bic(vote_matrix)
aic = model.aic(vote_matrix)
```

### 8.2 Class Enumeration Pipeline

```python
def enumerate_classes(vote_matrix, k_max=8):
    """Fit LCA models from K=1 to K=k_max, return BIC/AIC/entropy."""
    results = []
    for k in range(1, k_max + 1):
        model = StepMix(
            n_components=k,
            measurement="binary_nan",
            n_init=50,
            max_iter=1000,
            random_state=42,
        )
        model.fit(vote_matrix)
        probs = model.predict_proba(vote_matrix)
        entropy = _compute_entropy(probs, k)
        results.append({
            "k": k,
            "bic": model.bic(vote_matrix),
            "aic": model.aic(vote_matrix),
            "log_likelihood": model.score(vote_matrix) * len(vote_matrix),
            "entropy": entropy,
            "n_params": _count_params(k, vote_matrix.shape[1]),
        })
    return results
```

### 8.3 Salsa Effect Detection

After selecting the BIC-optimal K, programmatically test whether class profiles are qualitatively distinct or merely graded:

1. For each class, compute the mean P(Yea) profile across all bills
2. Compute pairwise Spearman correlations between class profiles
3. If all pairwise correlations > 0.8, flag as "Salsa effect" — classes are quantitatively graded, not qualitatively distinct
4. If any pairwise correlation < 0.5, the classes represent genuinely different voting patterns

### 8.4 Report Sections

Following the Phase 5 report pattern:

1. **Class Enumeration** — BIC/AIC elbow plot, entropy by K, selected K
2. **Class Profiles** — Heatmap of P(Yea | class) for top discriminating bills
3. **Class Composition** — Party breakdown within each class, class sizes
4. **Membership Probabilities** — Distribution of max P(class), identification of "straddlers"
5. **IRT Cross-Validation** — Boxplot of IRT ideal points by LCA class; monotonicity test
6. **Clustering Agreement** — ARI between LCA classes and Phase 5 clusters
7. **Salsa Effect Assessment** — Profile correlation matrix, qualitative vs quantitative distinction
8. **Within-Party Classes** — Separate LCA on each party to test for intra-party structure

---

## 9. Expected Outcomes

Based on our existing findings (k=2 optimal, continuous within-party variation, 82% Yea base rate, 72% R supermajority):

**Most likely (80%):** BIC selects K=2, perfectly recovering the party split. Entropy > 0.9 (near-perfect classification). ARI with Phase 5 clusters ≈ 1.0. Within-party LCA finds K=1 (no internal structure) or K=2 with the Salsa effect. **This is the scientifically valuable null result** — model-based confirmation that discrete factions don't exist.

**Possible (15%):** BIC selects K=3, with a small "moderate crossover" class bridging the parties. Schreiber, Dietrich, or other identified mavericks may anchor this class. If the class has a qualitatively different profile (not just intermediate P(Yea) rates), this would be a genuine finding. Cross-validate with IRT to distinguish from Salsa effect.

**Unlikely but interesting (5%):** BIC selects K=4+, revealing issue-specific voting blocs. Would suggest multi-dimensional structure that 1D IRT misses. Cross-validate with Phase 4b (2D IRT) to determine if the second IRT dimension captures the same structure.

---

## 10. References

### Foundational

- Lazarsfeld, P.F. (1950). The logical and mathematical foundation of latent structure analysis. In S.A. Stouffer et al. (Eds.), *Measurement and Prediction*.
- Goodman, L.A. (1974). Exploratory latent structure analysis using both identifiable and unidentifiable models. *Biometrika*, 61(2), 215-231.
- McCutcheon, A.L. (1987). *Latent Class Analysis*. Sage QASS #64.
- Hagenaars, J.A. & McCutcheon, A.L. (Eds.) (2002). *Applied Latent Class Analysis*. Cambridge UP.
- Clogg, C.C. (1995). Latent class models. In G. Arminger et al. (Eds.), *Handbook of Statistical Modeling for the Social and Behavioral Sciences*.

### Estimation and Model Selection

- Vermunt, J.K. & Magidson, J. (2002). Latent class cluster analysis. In Hagenaars & McCutcheon (Eds.), *Applied Latent Class Analysis*.
- [Nylund, K.L., Asparouhov, T., & Muthén, B.O. (2007). Deciding on the number of classes in LCA and growth mixture modeling. *Structural Equation Modeling*, 14(4), 535-569.](https://www.statmodel.com/download/LCA_tech11_nylund_v83.pdf)
- [Nylund-Gibson, K. & Choi, A.Y. (2018). Ten frequently asked questions about LCA. *Translational Issues in Psychological Science*, 4(4), 440-461.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7746621/)
- [Weller, B.E., Bowen, N.K., & Faubert, S.J. (2020). LCA: A guide to best practice. *Journal of Black Psychology*, 46(4), 287-311.](https://journals.sagepub.com/doi/full/10.1177/0095798420930932)
- [Biernacki, C., Celeux, G., & Govaert, G. (2000). Assessing a mixture model for clustering with the ICL. *IEEE TPAMI*, 22(7), 719-725.](https://www.researchgate.net/publication/3193130_Assessing_a_Mixture_Model_for_Clustering_with_the_Integrated_Completed_Likelihood)

### Categorical vs. Continuous Latent Structure

- [Lubke, G.H. & Neale, M.C. (2006). Distinguishing between latent classes and continuous factors: Resolution by maximum likelihood? *Multivariate Behavioral Research*, 41(4), 499-532.](https://pubmed.ncbi.nlm.nih.gov/26794916/)
- [Muthén, B. (2006). Should substance use disorders be considered as categorical or dimensional? *Addiction*, 101(Suppl 1), 6-16.](https://www.statmodel.com/download/Muthen_tobacco_2006.pdf)

### Diagnostics

- [Oberski, D.L. (2016). Measuring local dependence in LCA models. *Structural Equation Modeling*.](https://www.tandfonline.com/doi/abs/10.1080/10705511.2022.2033622)
- [van Lissa, C.J. (2023). Recommended practices in LCA using tidySEM. *Structural Equation Modeling*.](https://www.tandfonline.com/doi/full/10.1080/10705511.2023.2250920)

### Legislative Voting Applications

- [Choi, S. & Goldsmith, J. (2022). Dynamic Dirichlet process mixture models for voting coalitions. *PLOS ONE*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9415553/)
- [Vermunt, J.K. (2010). Latent class modeling with covariates: Two improved three-step approaches. *Political Analysis*, 18(4), 450-469.](https://www.cambridge.org/core/journals/political-analysis/article/abs/latent-class-modeling-with-covariates-two-improved-threestep-approaches/7DEF387D6ED4CF0A26A2FA06F9470D02)
- [Pettit, C. et al. (2020). Legislators' roll-call voting behavior increasingly corresponds to intervals. *Scientific Reports*, 10, 17369.](https://www.nature.com/articles/s41598-020-74175-w)

### Bayesian LCA

- [van Havre, Z. et al. (2015). Overfitting Bayesian mixture models with an unknown number of components. *PLOS ONE*.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131739)
- [White, N. et al. (2022). A tutorial on Bayesian LCA using JAGS.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6364555/)
- [Stan case study: Bayesian latent class models.](https://mc-stan.org/learn-stan/case-studies/Latent_class_case_study.html)

### Binary Clustering Comparison

- [Foss, A.H. et al. (2018). A comparison of K-means, LCA, and K-median for binary data. *PMC*.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5982597/)
- [Magidson, J. & Vermunt, J.K. (2002). Latent class models for clustering: A comparison with K-means. *Canadian Journal of Marketing Research*, 20, 36-43.](https://jeroenvermunt.nl/cjmr2002.pdf)

### Software

- [Linzer, D.A. & Lewis, J.B. (2011). poLCA: An R package for polytomous variable LCA. *Journal of Statistical Software*, 42(10), 1-29.](https://www.jstatsoft.org/v42/i10/)
- [Morin, S. et al. (2025). StepMix: A Python package for stepwise estimation of LCA with measurement and structural models. *Journal of Statistical Software*, 113(8).](https://www.jstatsoft.org/article/view/v113i08)
- [StepMix GitHub repository.](https://github.com/Labo-Lacourse/stepmix)
