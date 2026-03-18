# Multiple Correspondence Analysis Deep Dive

**Date:** 2026-02-25
**Scope:** Theory survey, Python ecosystem evaluation, Tallgrass integration plan
**Status:** Research complete, implementation planned

---

## Executive Summary

MCA (Multiple Correspondence Analysis) is the categorical-data analogue of PCA. It decomposes chi-square associations among categorical variables rather than Euclidean variance among continuous ones. For legislative roll call data, MCA's value depends critically on *how votes are encoded*:

- **Binary encoding (Yea=1, Nay=0, Absent=NaN):** MCA produces results mathematically equivalent to PCA, up to a scaling factor. No new information.
- **Categorical encoding (Yea/Nay/Absent as distinct categories):** MCA reveals absence patterns as a genuine dimension, weights rare vote patterns appropriately via chi-square distance, and maps legislators and vote categories into a shared space.

The second encoding is where MCA earns its place in the pipeline. The Tallgrass project currently uses binary encoding for PCA (Phase 02). Adding MCA with the full categorical vote matrix — treating Yea, Nay, and Absent as first-class categories — provides a complementary view that PCA structurally cannot offer.

**Recommended library:** `prince` 0.16.5 (MIT, validated against R's FactoMineR, Benzécri and Greenacre corrections, supplementary variable support). The pandas dependency is acceptable at our matrix size (~170×500); the conversion boundary is a single line.

---

## 1. What MCA Is

### Origin

MCA emerged from the French school of data analysis (*analyse des données*) founded by Jean-Paul Benzécri in the 1960s-70s. It extends simple Correspondence Analysis (CA) — which analyzes a single two-way contingency table — to the case of multiple categorical variables analyzed simultaneously. Pierre Bourdieu adopted MCA as the primary statistical method for *Distinction* (1979), arguing there was an "internal link" between his relational vision of social space and MCA's geometric properties.

### The Core Computation

Given *n* observations (legislators) and *Q* categorical variables (roll call votes), where variable *q* has *J_q* categories:

1. **Construct the indicator matrix Z** — an *n* × *J* binary matrix where *J* = Σ*J_q*. Each row has exactly *Q* ones (one per variable). For 3-category votes (Yea/Nay/Absent), each roll call becomes 3 columns.

2. **Compute the correspondence matrix** — P = Z / sum(Z).

3. **Compute standardized residuals** — S = D_r^{-1/2} (P − rc^T) D_c^{-1/2}, where r and c are row/column marginal vectors and D_r, D_c are diagonal matrices of row/column masses.

4. **SVD** — S = UΣV^T. The squared singular values (σ_k²) are the principal inertias (MCA eigenvalues).

5. **Extract coordinates:**
   - Row (legislator) coordinates: F = D_r^{-1/2} U Σ
   - Column (category) coordinates: G = D_c^{-1/2} V Σ

The **Burt matrix** B = Z^T Z is the categorical analogue of the correlation matrix. Its diagonal blocks are univariate marginal tables; off-diagonal blocks are two-way contingency tables between every pair of variables.

### How MCA Differs from PCA

| Property | PCA | MCA |
|----------|-----|-----|
| Input | Continuous variables | Categorical variables (indicator matrix) |
| Distance metric | Euclidean | Chi-square |
| Decomposed matrix | Correlation/covariance | Standardized residuals of indicator matrix |
| Variable representation | Arrows (loadings) | Points (category centroids) |
| Joint mapping | Observations only | Observations AND categories in same space |
| Rare-category weighting | Equal | Upweighted (inverse frequency) |
| Total variance | Data-dependent | Structure-dependent: (J−Q)/Q |

Both are eigendecomposition methods that find the principal axes of variation. The fundamental difference is the distance metric: PCA maximizes Euclidean variance explained, MCA maximizes chi-square association explained.

### Chi-Square Distance

The defining geometric property of MCA:

d²_χ²(i, j) = Σ_k (p_ik − p_jk)² / p_k

Chi-square distance weights differences by the inverse of the marginal frequency. Rare categories contribute more to distance than common ones. For legislative data, this means a Yea vote on a nearly unanimous bill contributes less to distinguishing legislators than a Yea vote on a closely contested bill — exactly the right behavior, and equivalent to what our 2.5% contested threshold filter achieves with a hard threshold, but continuous rather than binary.

---

## 2. MCA vs PCA on Legislative Vote Data

### The Binary Equivalence Result

Lebart, Morineau, and Tabard (1977) proved: **MCA on the indicator matrix of binary variables produces results equivalent to PCA on the centered binary matrix, up to a scaling factor.** The two methods yield identical orderings of observations and identical relative positions — only the scale of coordinates differs.

For our current binary vote matrix (Yea=1, Nay=0, Absent=NaN with row-mean imputation), MCA adds nothing beyond what PCA already provides. The ideological ordering would be identical. This is why MCA must use a different encoding to justify its existence in the pipeline.

### Where MCA Adds Genuine Value

**1. Multi-category structure.** When Absent is treated as a third category rather than as missing data, MCA positions it in the ideological space. "Absent" may appear:
- Between Yea and Nay (random absence — no signal)
- Closer to one side (partisan absence — strong signal)
- In its own distinct region (systematic pattern different from both Yea and Nay)

This directly addresses a question PCA cannot answer: *are absences random, partisan, or their own phenomenon?*

**2. Chi-square weighting.** MCA naturally downweights near-unanimous votes without requiring a hard contested threshold. A vote with 95% Yea contributes less to the analysis than a vote with 55% Yea. This is more principled than PCA's approach of either filtering (our 2.5% threshold) or treating all votes equally.

**3. Category-level coordinates.** MCA provides coordinates for each category of each variable. You can see where "Yea on HB2001" sits relative to "Nay on HB2001" and relative to "Yea on SB150." PCA loadings give per-variable information but not per-category. This enables the **biplot** — the defining MCA visualization — where legislators and vote categories appear in the same space.

**4. Supplementary variables.** MCA has a natural framework for projecting supplementary (passive) variables into the space without influencing axis construction. Party affiliation, committee membership, or district characteristics can be projected as supplementary categories — appearing in the map without driving the axes.

### When They Agree vs. Disagree

For strongly unidimensional data (most legislatures), PCA and MCA will produce very similar first dimensions. The Kansas Legislature, dominated by a single partisan axis that explains ~57% of PCA variance, will show MCA Dimension 1 highly correlated with PC1 (expected r > 0.95 on the binary encoding).

Disagreements emerge when:
- **Absence patterns are systematic.** MCA with 3 categories captures this; PCA with imputed NaN cannot.
- **Marginal distributions are highly unequal.** Kansas's 72% Republican supermajority means many near-unanimous bills. MCA's chi-square weighting handles this more gracefully.
- **The horseshoe effect appears.** MCA on gradient-structured data produces a characteristic arch where Dimension 2 is a quadratic function of Dimension 1. This is not a bug — it confirms unidimensionality rather than indicating a genuine second dimension.

---

## 3. The Pessimistic Eigenvalue Problem

### The Problem

When you run MCA and look at "percentage of inertia explained," the numbers look shockingly low compared to PCA. A PCA showing PC1 at 57% variance might correspond to an MCA showing Dimension 1 at ~12% inertia. This is an artifact of the coding scheme, not a reflection of fit quality.

### Why It Happens

The indicator matrix codes each *Q*-category variable with *Q* binary columns (subject to a sum-to-1 constraint). This creates artificial dimensions — *Q*−1 per variable — that are purely structural. The total inertia includes variance from these artificial dimensions, diluting the apparent contribution of meaningful ones.

For *Q* = 300 roll calls with 3 categories each:
- Total inertia = (900 − 300) / 300 = 2.0
- Trivial eigenvalue threshold = 1/Q = 1/300 = 0.0033
- A first eigenvalue of 0.12 is actually excellent

### Benzécri Correction (1979)

Drop all eigenvalues ≤ 1/Q and rescale:

λ*_k = [Q/(Q−1) × (λ_k − 1/Q)]² for λ_k > 1/Q

This can be dramatic: 12% raw inertia → ~85% Benzécri-corrected inertia. However, Benzécri's correction tends to overestimate quality because the denominator only includes corrected eigenvalues.

### Greenacre Correction (1984, 2006)

Divides by average off-diagonal inertia rather than the sum of corrected eigenvalues. Produces lower but more honest percentages — they represent the proportion of actual *association* (not structural noise) captured by each dimension.

**Recommendation:** Always report Greenacre-corrected inertia alongside raw. Greenacre is more conservative and generally preferred in the literature. Both corrections are available in `prince` via the `correction` parameter.

---

## 4. Interpretation

### Category Points

Each category of each variable receives coordinates. The category point is the **centroid** (weighted average) of all individuals who gave that response.

- Categories close together: chosen by the same individuals
- Categories far from origin: rare or distinctive — contribute more to discrimination
- Categories near origin: common (high marginal frequency) — contribute less
- Distance between categories of *same* variable: how differently those responses divide the population
- Distance between categories of *different* variables: strength of co-occurrence

For legislative data: if "Yea on HB2001" is close to "Yea on SB150," those bills attract the same coalition.

### Contributions (CTR)

ctr_ks = f_k × (y_ks)² / λ_s

Contributions sum to 1 across all categories for each axis. They identify which categories define each dimension. Rule of thumb: categories with contributions above 1/J (where J = total categories) are important for that axis.

For legislative analysis, high-contribution categories on Dimension 1 are the **most partisan votes** — the bills that best separate the parties.

### Squared Cosines (cos²)

cos²_ks = (y_ks)² / d_k²

Measures **quality of representation** — how well a category is captured by a given axis. cos² close to 1 on axes 1-2 means well-represented in the 2D map. cos² close to 0 means the position on the map should not be over-interpreted.

### The Horseshoe Effect

MCA on gradient-structured data produces a parabolic arch: Dimension 2 is a quadratic function of Dimension 1. This is mathematically inevitable — the kth eigenvector of a gradient-structured matrix is the kth-degree orthogonal polynomial of the first. Legislative voting data, which is strongly unidimensional, is particularly susceptible. When you see it, it confirms unidimensionality rather than revealing a genuine second dimension.

### Supplementary Variables

Variables projected into the MCA space without influencing axis construction. They receive coordinates based on the profiles of individuals who belong to each supplementary category. Extremely useful for legislative analysis:
- **Party** as supplementary: see where R/D centroids fall without party driving the axes
- **Committee membership** as supplementary: which committees cluster in ideology space
- **District characteristics** as supplementary: urban/rural, competitiveness

---

## 5. Best Practices for Legislative Vote Data

### Missing Data Strategy

MCA offers three approaches, each appropriate for different questions:

| Approach | When to use | Effect |
|----------|-------------|--------|
| Absent as **active** category | Studying strategic absence | Absence influences axis construction |
| Absent as **passive** category | Studying ideology only | Absent receives coordinates but doesn't drive axes |
| Drop absent, Yea/Nay only | Direct PCA comparison | Equivalent to binary PCA (validation use only) |

**Recommendation for Tallgrass:** Run with Absent as active category (the default). This is MCA's value proposition — if we wanted passive treatment, we'd just use PCA.

### Rare Categories

Categories with very low frequency (< 5% of observations) can distort MCA axes because chi-square weighting inflates their influence. Options:
- **Specific MCA (speMCA):** Make rare categories passive (Le Roux & Rouanet's variant)
- **Merge** rare categories with substantively similar ones
- **Remove** variables with very rare categories

For Kansas data: "Present and Passing" (22 instances in the 91st, ~0.05% of all votes) and some "Not Voting" categories are rare enough to warrant passive treatment. Use specific MCA to make these passive while keeping Yea/Nay/Absent active.

### Choosing Dimensions

1. Benzécri/Greenacre corrected eigenvalues > 0 (raw eigenvalues > 1/Q)
2. Scree plot on corrected eigenvalues — look for the elbow
3. Cumulative corrected inertia ≥ 70-90%
4. Parallel analysis (permutation-based — less common in MCA but theoretically sound)
5. Interpretability — each retained dimension should have a substantive interpretation

For legislative data: 2-3 dimensions are almost always sufficient. Dimension 1 = partisan divide, Dimension 2 = either horseshoe artifact or cross-cutting issue dimension.

### Filtering

Apply the same contested-vote filter as PCA (contested threshold < 2.5%) even though MCA's chi-square weighting partially handles this. Rationale: near-unanimous votes still add columns to the indicator matrix and increase the trivial-eigenvalue noise floor. Filtering them reduces *J* and improves corrected inertia estimates.

---

## 6. Python Ecosystem

### prince (Recommended)

| Property | Value |
|----------|-------|
| Version | 0.16.5 (January 9, 2026) |
| License | MIT |
| GitHub | github.com/MaxHalford/prince — 1.4k stars, 0 open issues |
| Dependencies | scikit-learn ≥ 1.5.1, pandas ≥ 2.2.0, altair ≥ 5.0 |
| Corrections | Benzécri and Greenacre via `correction` parameter |
| Validation | Cross-validated against FactoMineR via rpy2 |
| API | scikit-learn-compatible (fit/transform) |

Key features:
- `MCA(n_components, correction, one_hot, random_state)`
- `row_coordinates()` / `column_coordinates()` — legislator and category positions
- `row_contributions_` / `column_contributions_` — what defines each axis
- `row_cosine_similarities()` / `column_cosine_similarities()` — representation quality
- `eigenvalues_summary` — corrected and uncorrected inertia
- Supplementary row/column support

**Input requirement:** pandas DataFrames only (`@check_is_dataframe_input` decorator). Conversion path: Polars → pandas at the MCA boundary, results back to Polars immediately after. At our matrix size (~170×500), the conversion cost is negligible.

### Alternatives Evaluated

| Library | Verdict | Why not |
|---------|---------|---------|
| **mca** (esafak) | Decent | Less tested, no FactoMineR validation, also requires pandas |
| **scikit-learn** | N/A | No MCA implementation (despite some web claims) |
| **fanalysis** | Dead | 5 commits total since 2018, unmaintained |
| **scientisttools** | Immature | 97% notebooks, no PyPI releases |
| **statsmodels** | N/A | No CA/MCA support |
| **Custom numpy** | Viable | ~100-150 lines, avoids pandas, but must validate manually |

A custom numpy implementation is feasible and would avoid the pandas dependency. However, prince is already FactoMineR-validated and actively maintained. The pandas conversion at our scale is a non-issue. **Use prince; don't reinvent it.**

### Contrastive MCA (cMCA)

Worth noting: Fujiwara et al. (2023, PLOS ONE) published contrastive MCA specifically for identifying latent subgroups within political parties. It finds dimensions where a target group (e.g., moderate Republicans) has more variance than a background group (e.g., all legislators). This directly addresses our core analytical interest — intra-Republican variation. Not for the initial MCA phase, but a candidate for future exploration.

---

## 7. Integration Design

### Phase Placement

**Phase 03 (between PCA and UMAP).** Following the precedent of UMAP as "Phase 2b," MCA becomes "Phase 2c" — a third dimensionality reduction view on the same data. Directory: `analysis/03_mca/`.

This placement reflects MCA's role: an alternative geometric view of the vote matrix, consuming the same EDA output as PCA and UMAP, but with a different encoding and distance metric. It does not depend on PCA or UMAP results and can run independently.

### Data Flow

```
Raw votes CSV (data/kansas/{session}/)
  ↓
Build categorical vote matrix (Yea / Nay / Absent)
  ↓
Apply minority filter (< 2.5%) + min-vote filter (< 20)
  ↓
Mark rare categories as passive (Present and Passing, Not Voting)
  ↓
Polars → pandas conversion (MCA boundary)
  ↓
prince.MCA.fit() with Greenacre correction
  ↓
Extract: coordinates, contributions, cos², eigenvalues
  ↓
pandas → Polars conversion (back to project standard)
  ↓
Orient Dim1 (Republicans positive, matching PCA sign convention)
  ↓
Validate: Spearman correlation of Dim1 vs PCA PC1
  ↓
Save parquet files + plots + HTML report
```

**Critical difference from PCA:** MCA builds its own categorical vote matrix from the raw votes CSV, preserving Yea/Nay/Absent as distinct categories. It does NOT consume the binary vote matrices from EDA. This is the entire point — if it used binary matrices, it would reproduce PCA.

### Outputs

**Data files** (parquet, in `results/{session}/03_mca/latest/data/`):
- `mca_scores_{chamber}.parquet` — legislator coordinates on MCA dimensions + metadata
- `mca_category_coords_{chamber}.parquet` — category coordinates (vote × category)
- `mca_contributions_{chamber}.parquet` — category contributions to each dimension
- `mca_eigenvalues_{chamber}.parquet` — raw and corrected inertia per dimension
- `mca_cos2_{chamber}.parquet` — squared cosines for representation quality
- `filtering_manifest.json` — parameters, dimensions retained, votes/legislators filtered

**Plots** (PNG, in `results/{session}/03_mca/latest/plots/`):
- `mca_ideological_map_{chamber}.png` — legislator scatter on Dims 1-2, colored by party
- `mca_biplot_{chamber}.png` — legislators + top-contributing category points
- `mca_category_map_{chamber}.png` — category points only, colored by vote type (Yea/Nay/Absent)
- `mca_inertia_{chamber}.png` — scree plot with raw and corrected inertia
- `mca_absence_map_{chamber}.png` — legislators colored by absence rate, showing where Absent categories cluster
- `mca_dim1_distribution_{chamber}.png` — Dim1 scores ranked by party (comparable to PCA PC1 distribution)
- `mca_pca_correlation_{chamber}.png` — Dim1 vs PC1 scatter with Spearman r annotation

**HTML report:** Self-contained, following the ReportBuilder pattern. Sections: inertia summary, ideological map, biplot, absence analysis, PCA validation, top contributions, interpretation guide.

### Validation Strategy

1. **PCA correlation.** Spearman r between MCA Dim1 and PCA PC1. Expected: r > 0.95. If much lower, either the categorical encoding reveals genuine new structure or there's a bug.

2. **Sensitivity analysis.** Re-run with:
   - Binary encoding (Yea/Nay only) — should reproduce PCA exactly (up to scale)
   - Different contested thresholds (2.5% vs 10%)
   - Absent as passive vs. active

3. **Holdout validation.** Mask 20% of votes, project masked observations into the fitted MCA space, measure reconstruction quality.

4. **Horseshoe detection.** Compute polynomial regression of Dim2 on Dim1. If R² > 0.80, flag as horseshoe artifact and note that Dim2 is not a genuine second dimension.

### Files to Create

| File | Purpose |
|------|---------|
| `analysis/03_mca/__init__.py` | Empty package marker |
| `analysis/03_mca/mca.py` | Main phase script (~600-800 lines) |
| `analysis/03_mca/mca_report.py` | HTML report builder (~400 lines) |
| `analysis/design/mca.md` | Design choices document |
| `tests/test_mca.py` | Tests (~30-40 tests) |

### Files to Modify

| File | Change |
|------|--------|
| `analysis/__init__.py` | Add `"mca": "03_mca"` and `"mca_report": "03_mca"` to `_MODULE_MAP` |
| `pyproject.toml` | Add `prince>=0.13` dependency; add `"prince.**"` to ty `replace-imports-with-any` |
| `Justfile` | Add `mca *args:` recipe |
| `CLAUDE.md` | Add MCA to analysis recipes list, update architecture description |
| `docs/roadmap.md` | Move MCA from "Next Up" to "Completed" (after implementation) |

### Constants

```python
DEFAULT_N_COMPONENTS = 5
CONTESTED_THRESHOLD = 0.025
MIN_VOTES = 20
CORRECTION = "greenacre"                    # Greenacre > Benzécri (more conservative)
PASSIVE_CATEGORIES = {"Present and Passing", "Not Voting"}
ACTIVE_CATEGORIES = {"Yea", "Nay", "Absent and Not Voting"}
ABSENT_LABEL = "Absent"                     # Canonical label for absent categories
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
CATEGORY_COLORS = {"Yea": "#2ca02c", "Nay": "#d62728", "Absent": "#7f7f7f"}
TOP_CONTRIBUTIONS_N = 20                    # Number of top-contributing categories to label in biplot
HORSESHOE_R2_THRESHOLD = 0.80              # Polynomial R² above this flags horseshoe artifact
PCA_VALIDATION_MIN_R = 0.90                 # Minimum expected Spearman r between Dim1 and PC1
```

### Test Plan

| Test Group | Count | Coverage |
|------------|-------|----------|
| Categorical vote matrix construction | 5 | Yea/Nay/Absent encoding, null handling, shape |
| Minority filtering on categorical data | 4 | Threshold, rare category detection, chamber split |
| MCA fitting (synthetic data) | 5 | Component capping, output shapes, correction modes |
| Orientation (Dim1 sign convention) | 3 | Flip, no-flip, unknown party |
| Contribution/cos² computation | 4 | Shape, sums, non-negative, threshold filtering |
| Horseshoe detection | 3 | Clear horseshoe, no horseshoe, borderline |
| PCA validation correlation | 3 | High r (expected), low r (warning), missing PCA |
| Sensitivity (binary encoding) | 3 | Matches PCA, correct scaling, no categories lost |
| Passive category handling | 3 | Categories excluded from fit, still receive coordinates |
| Report builder | 2 | All sections present, HTML valid |

**Total: ~35 tests**

---

## 8. Key References

### Foundational

- Benzécri, J.-P. (1973). *L'Analyse des Données.* 2 vols. Paris: Dunod.
- Benzécri, J.-P. (1979). "Sur le calcul des taux d'inertie dans l'analyse d'un questionnaire." *Cahiers de l'Analyse des Données*, 4, 377-378.
- Greenacre, M.J. (1984). *Theory and Applications of Correspondence Analysis.* London: Academic Press.
- Greenacre, M.J. (2006). "From Simple to Multiple Correspondence Analysis." In *Multiple Correspondence Analysis and Related Methods* (ed. Greenacre and Blasius). Chapman and Hall/CRC.
- Greenacre, M.J. (2017). *Correspondence Analysis in Practice,* 3rd ed. CRC Press.
- Lebart, L., Morineau, A., and Tabard, N. (1977). *Techniques de la Description Statistique.* Paris: Dunod.

### Geometric Data Analysis

- Le Roux, B. and Rouanet, H. (2004). *Geometric Data Analysis: From Correspondence Analysis to Structured Data Analysis.* Dordrecht: Kluwer.
- Le Roux, B. and Rouanet, H. (2010). *Multiple Correspondence Analysis.* SAGE, QASS Vol. 163.
- Hjellbrekke, J. (2018). *Multiple Correspondence Analysis for the Social Sciences.* London: Routledge.

### Technical

- Abdi, H. and Valentin, D. (2007). "Multiple Correspondence Analysis." In *Encyclopedia of Measurement and Statistics.* Sage.
- Husson, F., Le, S., and Pagès, J. (2010). *Exploratory Multivariate Analysis by Example Using R.* Chapman and Hall/CRC.

### Political Science Applications

- Bourdieu, P. (1979/1984). *Distinction: A Social Critique of the Judgement of Taste.* Translated by Richard Nice. Harvard University Press.
- Fujiwara, T. et al. (2023). "Contrastive Multiple Correspondence Analysis (cMCA): Applying the contrastive learning method to identify political subgroups." *PLOS ONE*.

### Software

- `prince` Python library: github.com/MaxHalford/prince (v0.16.5, MIT, FactoMineR-validated)
- FactoMineR R package: CRAN (the gold standard implementation)

---

## 9. Honest Assessment

**What MCA will likely confirm:** The partisan divide is the dominant axis. MCA Dimension 1 will correlate r > 0.95 with PCA PC1. The same mavericks (Schreiber, Dietrich) will appear as outliers. The horseshoe effect will appear on Dimensions 1-2 for at least one chamber.

**What MCA might uniquely reveal:**
- Whether absence patterns are random or partisan (the absence dimension)
- Which specific vote categories (not just variables) define the partisan axis
- Whether chi-square weighting identifies different "most partisan" votes than PCA's linear loadings
- How supplementary party labels position relative to the data-driven axes

**What MCA will NOT do:** Produce a fundamentally different ideological ordering. For binary data, MCA and PCA are mathematically equivalent. The categorical encoding adds nuance, not revolution.

**Cost-benefit:** ~2-3 days of implementation for a phase that will likely confirm PCA results while adding the absence dimension and category-level insights. This is a reasonable investment — MCA is a standard tool in the dimensionality reduction toolkit, and the categorical encoding is the methodologically correct approach for vote data. The pipeline's credibility benefits from showing that PCA's simplifying assumptions don't distort results, and the absence analysis may reveal something genuinely new.
