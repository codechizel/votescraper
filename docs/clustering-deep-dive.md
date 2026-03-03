# Clustering Deep Dive

A literature survey, code audit, and set of recommendations for the clustering phase of the Kansas Legislature vote analysis pipeline.

**Date:** 2026-02-24
**Scope:** `analysis/09_clustering/clustering.py` (2527 lines), `analysis/09_clustering/clustering_report.py` (1031 lines), `tests/test_clustering.py` (413 lines), `analysis/design/clustering.md`, ADR-0007, ADR-0014

---

## 1. Literature Survey: Clustering Legislative Voting Data

### 1.1 The Landscape of Algorithms

Legislative roll-call data presents a specific challenge: a binary matrix (legislators x roll calls) where the dominant structure is almost always the party split, and the interesting question is whether finer structure exists beneath it. The field has converged on several families of approaches.

**Distance-based methods** construct a pairwise similarity or distance matrix between legislators, then cluster over it. The choice of distance metric is the single most consequential preprocessing decision. For binary voting data, the key question is whether mutual "Nay" votes (both legislators voting against a bill) carry the same information as mutual "Yea" votes. In practice they usually do — voting against a bill together is as politically meaningful as voting for it together. This favors **symmetric metrics** (Hamming, Simple Matching, Cohen's Kappa) over asymmetric ones (Jaccard, Dice). Jaccard ignores matched zeros, which means it discounts co-Nay agreement. Our pipeline uses Kappa, which corrects for base-rate agreement — a good choice given Kansas's 82% Yea base rate. See [Choi et al. (2010)](https://www.iiisci.org/journal/pdv/sci/pdfs/GS315JG.pdf) for a survey of 76 binary similarity/distance measures.

**Centroid-based methods** (K-Means) operate in a continuous feature space. Applied directly to binary vote matrices, K-Means uses Euclidean distance, which treats vote positions as continuous — theoretically questionable but often adequate in practice, especially after dimensionality reduction via PCA. [Kozlowski & Dukes (2025)](https://royalsocietypublishing.org/rsos/article/13/2/251428/479919/A-new-measure-of-issue-polarization-using-k-means) use K-Means on PCA-reduced survey data to measure political polarization across 105 countries.

**Model-based methods** (Gaussian Mixture Models, Bernoulli Mixture Models, Latent Class Analysis) posit a generative model. GMMs assume continuous Gaussian features — inappropriate for raw binary data, but defensible when applied to IRT ideal points or PCA scores. The theoretically correct analog for binary data is the **Bernoulli Mixture Model**, where each cluster is characterized by a vector of Bernoulli probabilities. [Scrucca & Raftery (2018)](https://ar5iv.labs.arxiv.org/html/1805.04203) applied model-based clustering with BIC selection to US Congressional voting records. **Latent Class Analysis** (LCA) via [StepMix](https://github.com/Labo-Lacourse/stepmix) (JSS 2025) is the most statistically principled Python approach for binary data, with native missing-data handling via Full Information Maximum Likelihood.

**Hierarchical methods** produce dendrograms that naturally visualize intra-party factions. **Ward linkage** minimizes within-cluster variance and produces balanced, compact clusters, but has a critical constraint: it requires Euclidean distance (discussed in Section 3). Research by Hands & Everitt on multivariate binary data found Ward best overall, complete linkage second-best, and average linkage best when k is overspecified.

**Density-based methods** (HDBSCAN, DBSCAN, OPTICS) identify clusters as dense regions separated by sparse areas. [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html) is particularly attractive because it designates noise points rather than forcing every legislator into a cluster, doesn't require specifying k, and handles varying-density clusters. It works best on PCA- or UMAP-reduced inputs; the [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/clustering.html) reports HDBSCAN cluster assignment jumping from 17% to 99% when preprocessing with UMAP. HDBSCAN is now in scikit-learn (since v1.3), though there is a known issue where the scikit-learn implementation [modifies precomputed distance matrices in-place](https://github.com/scikit-learn/scikit-learn/issues/31907).

**Network community detection** treats the agreement matrix as a weighted graph. The [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z) (Traag et al. 2019) has superseded Louvain, providing guaranteed well-connected communities and supporting the Constant Potts Model (CPM) resolution parameter for multi-scale analysis. [Spieksma et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7566643/) used the Stochastic Degree Sequence Model on US Senate co-voting networks to discover that since the 104th Congress, voting behavior collapses to a single ideological dimension. Our pipeline already runs a separate network analysis phase (Phase 6); the question is whether community detection results should cross-validate clustering results more formally.

**Spectral clustering** transforms data into a graph Laplacian eigenspace before applying K-Means. It accepts precomputed affinity matrices, making it natural for agreement-based input. The `assign_labels='cluster_qr'` option in scikit-learn produces deterministic, visually clean partitions.

**Consensus clustering** aggregates multiple clustering runs into a single robust partition by building a co-association matrix (Monti et al. 2003). Python implementations include [consensusclustering](https://github.com/burtonrj/consensusclustering) and [Cluster_Ensembles](https://pypi.org/project/Cluster_Ensembles/). For voting data, the consensus matrix itself is interpretable as a "probability of co-membership" heatmap.

### 1.2 Validation Metrics

| Metric | Type | Range | Best For |
|--------|------|-------|----------|
| Silhouette | Internal | [-1, 1] | Most informative for convex clusters ([PeerJ 2025](https://peerj.com/articles/cs-3309/)) |
| Calinski-Harabasz | Internal | [0, inf) | Fast but less informative than Silhouette |
| Davies-Bouldin | Internal | [0, inf) | Ratio of within- to between-cluster distance |
| ARI | External | [-0.5, 1] | Chance-adjusted partition comparison |
| AMI | External | [-1, 1] | Chance-adjusted mutual information |
| NMI | External | [0, 1] | **Not chance-adjusted** — biased toward larger k |
| Gap Statistic | Model selection | N/A | Compares dispersion against bootstrap null |
| Bootstrap Instability | Stability | [0, 1] | Cluster reproducibility under resampling |

**Key insight from the literature:** Always prefer chance-adjusted metrics (ARI, AMI) over non-adjusted ones (NMI). NMI is [biased in favor of larger k](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf) — the expected NMI for random partitions rises as the number of clusters increases. Our pipeline correctly uses ARI.

**Stability-based validation** ([Liu et al. 2022, WIREs](https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.1575)) is arguably the most rigorous approach to selecting k: if a clustering is real, it should be reproducible under resampling. Key methods include bootstrap instability (Fang & Wang 2012), prediction strength (Tibshirani & Walther), and Clest (Dudoit & Fridlyand). Values above 0.80-0.90 indicate robust clusters. Our pipeline does not currently implement stability-based validation.

### 1.3 Handling Missing Votes

Legislative vote matrices are inherently incomplete. The missingness mechanism matters: "Absent/Not Voting" is usually MNAR (strategic avoidance), "Excused" is closer to MCAR. Approaches from simplest to most principled:

1. **Drop low-participation legislators** (NOMINATE drops <20 votes — we do this)
2. **Drop high-missingness roll calls** (>20% abstentions)
3. **Pairwise deletion** (compute distances only over shared votes — Kappa does this naturally)
4. **Multiple imputation + consensus clustering** ([MICA, PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10291928/))
5. **StepMix FIML** (handles missing data natively via `measurement="binary_nan"`)

Our pipeline uses approaches 1 and 3 (minimum vote threshold + Kappa's natural pairwise computation). This is reasonable and well-justified.

### 1.4 The k=2 Question

In US legislative bodies, k=2 (Democrat/Republican) is the expected baseline. The ARI between a k=2 clustering and party labels should be near 1.0 for any reasonable method. The scientific interest lies in finer structure: k=3+ for cross-party moderates or factional splits, soft clustering probabilities revealing gradients, and hierarchical subclusters within parties.

Pure statistical optimization (silhouette maximization) will almost always select k=2 because the party signal dominates. This is exactly what we observe. The within-party analysis is the right approach to look for finer structure, though the finding that intra-party variation is continuous rather than discrete is itself scientifically valuable.

---

## 2. Our Implementation: What We Do Well

### 2.1 Three-Method Robustness Design

The architecture of running hierarchical (on Kappa distance), K-Means (on IRT), and GMM (on IRT) independently with ARI cross-validation is methodologically sound. Each method has different assumptions, and agreement across all three (mean ARI > 0.93) is strong evidence that the structure is real. This multi-method approach aligns with best practices in the field.

### 2.2 Kappa Distance for Hierarchical Clustering

Using Cohen's Kappa rather than raw agreement is a critical correction for Kansas's 82% Yea base rate. Raw agreement would inflate similarity between all legislators (everyone agrees on unanimous votes). Kappa corrects for chance agreement, and the NaN-to-max-distance fill for insufficient shared votes is appropriately conservative.

### 2.3 Party Loyalty Metric

The "contested vote" approach (threshold at 10% party dissent) elegantly solves the Tyson paradox — distinguishing ideologically extreme legislators from unreliable caucus members. This is a genuinely useful second dimension that standard clustering approaches miss.

### 2.4 GMM Uncertainty Weighting

Down-weighting uncertain legislators (wide posterior intervals) via observation replication is a pragmatic workaround for sklearn's lack of sample weights in GMM. The design doc correctly acknowledges this is approximate. The 3x effective sample scaling prevents the replicated dataset from being too small or too large.

### 2.5 Within-Party Clustering

Removing the party boundary to test for finer structure is the right follow-up to k=2. The flat silhouette profile (0.57-0.60 across k=2 to k=7) is itself an important finding: intra-party variation is continuous, not discrete.

### 2.6 Visualization Suite

The four visualization types (dendrogram, voting blocs, polar dendrogram, icicle) serve different audiences well. The `_build_display_labels()` helper centralizing name extraction prevents the `.split()[-1]` bug.

### 2.7 Test Coverage

22 tests covering party loyalty, ARI comparison, within-party clustering, constants, and the mirrored constant sync. The synthetic fixtures are well-designed with clear expected behaviors.

---

## 3. Issues Found: Correctness

### 3.1 Ward Linkage on Non-Euclidean Distance (Methodological)

**Severity: Medium — results are valid but methodologically impure.**

Ward linkage requires Euclidean distance. The [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) states: *"Methods 'centroid', 'median', and 'ward' are correctly defined only if Euclidean pairwise metric is used."* Our hierarchical clustering uses Ward linkage on a Kappa-derived distance matrix (1 - Kappa), which is not Euclidean.

In practice, this often works acceptably — Ward on non-Euclidean distances produces a valid dendrogram and the cophenetic correlation validates it (0.70+ threshold). But the merge heights lose their interpretation as within-cluster variance, and in pathological cases, scipy can produce negative branch heights (the Lance-Williams update formula for Ward assumes the triangle inequality, which non-Euclidean metrics can violate).

**Why this hasn't caused visible problems:** The Kappa distance matrix for this data is well-behaved (positive semi-definite after symmetrization, low NaN count). The k=2 structure is so dominant that minor metric distortions don't change the optimal partition.

**Options:**
1. **Switch to average or complete linkage** for the Kappa distance matrix (these are valid with any metric). This is the most defensible fix.
2. **Keep Ward but compute silhouette on the Kappa distance matrix** (already done correctly with `metric="precomputed"`). The silhouette validation operates on the actual distances, so the validation is correct even if the linkage is impure.
3. **Add a second hierarchical run on Euclidean distance of PCA scores**, where Ward is fully valid. This would give a methodologically pure Ward comparison point.

**Recommendation:** Option 1 (switch to average linkage for the Kappa-distance hierarchical clustering) is the cleanest fix. Average linkage is the standard choice for precomputed non-Euclidean distance matrices and performs well when the number of clusters is potentially overspecified (which is the case here — we evaluate k=2 through k=7). The design doc should be updated to document why average linkage was chosen. If you want to preserve Ward's compact-cluster behavior, option 3 (add a Ward-on-PCA run) provides it with correct assumptions.

### 3.2 Distance Matrix Recomputed Redundantly

**Severity: Low — performance only.**

`find_optimal_k_hierarchical()` (lines 506-543) recomputes the entire Kappa-to-distance conversion (1 - kappa, symmetrize, NaN fill, clip) that `run_hierarchical()` already computed (lines 470-487). For 130 legislators this takes negligible time, but it's unnecessary duplication. The distance matrix should be computed once and passed to both functions.

### 3.3 Silhouette Computed on Euclidean for K-Means 1D

**Severity: Low — arguably correct for the input space.**

K-Means operates on IRT `xi_mean` (1D continuous). The silhouette score at line 1056 uses the default metric (Euclidean), which is correct for this input space. However, the hierarchical silhouette (line 534) uses `metric="precomputed"` on the Kappa distance matrix. This means the two silhouette scores are not directly comparable — they measure separation in different spaces. The code correctly reports them separately, but the interpretation could be clearer.

---

## 4. Issues Found: Code Quality

### 4.1 Mirrored Constant (`CONTESTED_PARTY_THRESHOLD`)

**Severity: Low — mitigated by test.**

`clustering.py` defines `CONTESTED_PARTY_THRESHOLD = 0.10` and `clustering_report.py` defines `_CONTESTED_PARTY_THRESHOLD = 0.10` separately due to a circular import. A test (line 409-413) verifies they stay in sync. This works but is fragile.

**Fix:** Extract shared constants to a module that neither `clustering.py` nor `clustering_report.py` imports from each other. A `clustering_constants.py` or a shared section in an existing module like `run_context.py` would eliminate the duplication.

### 4.2 `DEFAULT_K = 3` Is Stale

**Severity: Cosmetic — no behavioral impact.**

`DEFAULT_K = 3` was the pre-analysis hypothesis (conservative R, moderate R, Democrat). The analysis conclusively showed k=2 is optimal. `DEFAULT_K` is used in the hierarchical assignment output (`cluster_k{DEFAULT_K}`) to provide a k=3 cut for comparison. This is intentional, but the variable name and comment ("Expected optimal k") are misleading now that k=2 is confirmed. Consider renaming to `COMPARISON_K = 3` with a comment explaining it provides a forced k=3 cut for downstream comparison.

### 4.3 Figure Sizing Magic Numbers

**Severity: Low — aesthetic.**

Multiple figure sizing heuristics are scattered through the code:
- Line 570: `fig_height = 10 if truncate else max(14, len(slugs) * 0.25)`
- Line 666: `figsize=(10, max(8, n * 0.12))`
- Line 829: `gap < even_sep * 0.8`
- Line 693: `np.percentile(np.abs(xi_arr), 95)`
- Lines 1285-1300: `loy_vals[i] < 0.7 or abs(xi_vals[i]) > 3.5`

These are empirical and work for the current data, but they're undocumented. Extracting them as named constants (e.g., `DENDROGRAM_HEIGHT_PER_LEGISLATOR = 0.25`, `NOTABLE_LOYALTY_THRESHOLD = 0.7`, `NOTABLE_XI_THRESHOLD = 3.5`) would make tuning easier.

### 4.4 `tempfile` Imported Inside Test Methods

**Severity: Cosmetic.**

`test_clustering.py` imports `tempfile` inside three individual test methods (lines 275, 310, 348) instead of at the top of the file. This works but violates Python import conventions.

### 4.5 Independent Party Not Handled in Cluster Characterization

**Severity: Low — affects edge cases.**

`characterize_clusters()` (line 1537-1538) counts `n_republican` and `n_democrat` but doesn't count Independents. For the 89th biennium (which has an Independent legislator), the party composition won't sum correctly. The fix is to add `n_other = n - n_r - n_d` (similar to the fix already applied in `analyze_community_composition()` per the results audit).

---

## 5. Missing Test Coverage

### 5.1 Hierarchical Clustering Functions

`run_hierarchical()` and `find_optimal_k_hierarchical()` have zero unit tests. These are the most complex functions in the module (NaN handling, distance conversion, linkage, cophenetic validation, silhouette model selection). Tests needed:

- **NaN distance fill**: Verify that NaN entries become `max_finite`
- **Symmetry enforcement**: Asymmetric input should be symmetrized
- **Cophenetic threshold**: Verify the OK/WARNING logic
- **Optimal k selection**: With a known well-separated distance matrix, verify k recovery
- **Negative distance clipping**: Verify kappa > 1 (shouldn't happen) is clipped to 0

### 5.2 GMM Functions

`run_gmm_irt()` has no unit tests. The weighting-by-replication logic is the most novel code in the module and most likely to harbor subtle bugs. Tests needed:

- **Weight computation**: Verify that high-uncertainty legislators get fewer repeats
- **BIC model selection**: With well-separated Gaussian data, verify correct k
- **Prediction on original data**: Verify that labels are for the original (unweighted) observations, not the replicated ones

### 5.3 Veto Override Analysis

`analyze_veto_overrides()` has no unit tests. The function has several join operations and a `replace_strict()` with `default=-1` that could silently produce incorrect results. Tests needed:

- **Override ID filtering**: Verify correct motion string matching
- **Empty overrides**: Verify graceful handling when < 2 override votes exist
- **Cluster label mapping**: Verify the slug-to-label dictionary produces correct assignments

### 5.4 Characterize Clusters

`characterize_clusters()` has no unit tests. Tests needed:

- **Party composition**: Verify counts sum to cluster size
- **Cluster labels**: Verify "Conservative R" / "Democrat" / etc. assignment logic
- **Missing loyalty**: Verify behavior when `loyalty` is None

### 5.5 Plot Functions

No plot functions are tested. While full visual regression testing is overkill, smoke tests (function runs without error, output file exists) would catch import errors and data shape mismatches.

---

## 6. Refactoring Opportunities

### 6.1 Extract Distance Matrix Computation

The Kappa-to-distance conversion (1 - kappa, symmetrize, NaN fill, clip) is duplicated between `run_hierarchical()` and `find_optimal_k_hierarchical()`. Extract to a helper:

```python
def _kappa_to_distance(kappa_matrix: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert Kappa agreement matrix to distance matrix.

    Returns (distance_array, slug_list).
    NaN entries (insufficient shared votes) filled with max finite distance.
    """
```

### 6.2 Standardization Helper

Z-score standardization is duplicated between `run_kmeans_irt()` (lines 1073-1075) and `run_within_party_clustering()` (lines 1698-1700):

```python
xi_std = (xi_2d - xi_2d.mean()) / (xi_2d.std() + 1e-10)
loy_std = (loy_2d - loy_2d.mean()) / (loy_2d.std() + 1e-10)
X_2d = np.column_stack([xi_std, loy_std])
```

Extract to `_standardize_2d(xi, loyalty)` or use `sklearn.preprocessing.StandardScaler` for consistency.

### 6.3 Constants Module

The mirrored `CONTESTED_PARTY_THRESHOLD` between `clustering.py` and `clustering_report.py` could be eliminated by extracting clustering constants to a shared module. This also enables other phases to import constants without pulling in the full clustering module.

---

## 7. Methods We Don't Use (And Whether We Should)

### 7.1 Spectral Clustering — Worth Adding

Spectral clustering on a precomputed agreement matrix would be a natural fourth method for cross-validation. It handles non-linear cluster boundaries, is deterministic with `assign_labels='cluster_qr'`, and operates in the same input space as our hierarchical clustering (agreement matrix). Adding it would strengthen the multi-method robustness story.

**Effort:** Low. A `run_spectral()` function would be ~30 lines using `sklearn.cluster.SpectralClustering(affinity='precomputed')`. The agreement matrix is already computed.

**Value:** Provides a fundamentally different algorithmic family (graph-based) alongside agglomerative, centroid-based, and probabilistic methods.

### 7.2 HDBSCAN on PCA Embeddings — Worth Considering

HDBSCAN would provide density-based clustering without requiring k specification, and would designate outlier legislators (noise points) — potentially useful for identifying legislators like Miller and Hill who have uncertain IRT estimates. Applied to PCA scores (which we already compute), it would offer a complementary perspective.

**Effort:** Medium. Requires tuning `min_cluster_size` and `min_samples`. PCA scores are available from the upstream PCA phase.

**Value:** Noise detection is the unique capability. No other method in our pipeline explicitly identifies clustering outliers.

### 7.3 Latent Class Analysis (StepMix) — Worth Evaluating

LCA is the most statistically principled approach for binary data. StepMix (published JSS 2025) provides a scikit-learn-compatible API with native missing-data support (`measurement="binary_nan"`). It would replace the theoretically impure GMM-on-IRT approach with a model that directly fits the binary vote matrix.

**Effort:** Medium-high. New dependency, different input space (raw vote matrix vs. IRT), and new model selection procedure (BIC over latent classes).

**Value:** Provides soft cluster assignments (like GMM) with a theoretically correct generative model for binary data. Would be the "gold standard" clustering method for this data type.

**Risk:** May simply recover k=2 (party split) from the same dominant signal. The added complexity may not reveal structure beyond what we already find. Recommend a prototype evaluation before committing.

### 7.4 Consensus Clustering — Not Recommended

We already accomplish the goal of consensus clustering (multi-method agreement) via ARI cross-validation. Formal consensus clustering (Monti et al.) would add complexity without clear additional insight. The co-association matrix would likely mirror our ARI findings.

### 7.5 Biclustering — Not Recommended

SpectralCoclustering and SpectralBiclustering are designed for continuous matrices (gene expression, document-term). Their assumptions (high values indicate patterns, blockwise-constant structure) are questionable for binary vote data where 0 and 1 are equally meaningful. Not worth the interpretive burden.

### 7.6 Leiden Community Detection — Not Recommended as Clustering

Our network analysis phase (Phase 6) already runs community detection. Adding Leiden to the clustering phase would create redundancy. The better approach is to cross-reference Phase 6 community assignments with Phase 5 clustering assignments, which the synthesis phase (Phase 11) already does implicitly.

---

## 8. Recommendations Summary

> **Implementation status as of 2026-02-24:** All correctness fixes, code quality improvements, and high-priority tests implemented. Spectral clustering and HDBSCAN added. LCA deferred for future evaluation. See ADR-0028.

### Correctness Fixes (Done)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Switch hierarchical linkage from Ward to average for Kappa distances | Medium | **Done** — ADR-0028 |
| 2 | Add `n_other` count for Independents in `characterize_clusters()` | Low | **Done** |
| 3 | Extract distance matrix computation to avoid duplication | Low | **Done** — `_kappa_to_distance()` helper |

### Code Quality (Done)

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 4 | Rename `DEFAULT_K` to `COMPARISON_K` | Cosmetic | **Done** |
| 5 | Extract figure sizing magic numbers as named constants | Low | **Done** — 8 constants |
| 6 | Move `tempfile` import to module level in tests | Cosmetic | **Done** |
| 7 | Extract shared constants to eliminate mirrored constant | Low | **Partial** — report still uses local copy due to circular import |
| 8 | Extract standardization helper for 2D clustering | Low | **Done** — `_standardize_2d()` helper |

### Testing (Done)

| # | Test Gap | Status |
|---|----------|--------|
| 9 | `run_hierarchical()` — NaN fill, symmetry, cophenetic | **Done** — 4 tests |
| 10 | `find_optimal_k_hierarchical()` — k recovery on known data | **Done** — 4 tests |
| 11 | `run_gmm_irt()` — weighting, BIC selection, unweighted prediction | Deferred |
| 12 | `analyze_veto_overrides()` — ID filtering, empty case, label mapping | Deferred |
| 13 | `characterize_clusters()` — party counts, label logic | **Done** — 6 tests |
| 14 | Plot function smoke tests | Deferred |

### New Methods

| # | Method | Value | Status |
|---|--------|-------|--------|
| 15 | Spectral clustering on agreement matrix | High (new algorithmic family) | **Done** — `run_spectral_clustering()` |
| 16 | HDBSCAN on PCA embeddings | Medium (outlier detection) | **Done** — `run_hdbscan_pca()` |
| 17 | Latent Class Analysis (StepMix) | High (principled binary model) | Deferred — evaluate in future session |

### Not Recommended

| Method | Reason |
|--------|--------|
| Consensus clustering | Redundant with our ARI cross-validation approach |
| Biclustering | Assumptions don't fit binary vote data |
| Leiden in clustering phase | Already in network phase; would create redundancy |
| RAPIDS cuML | Dataset too small (~165 legislators) to benefit from GPU acceleration |

---

## 9. References

### Algorithms and Methods
- [Choi et al. (2010) — Survey of 76 binary distance measures](https://www.iiisci.org/journal/pdv/sci/pdfs/GS315JG.pdf)
- [Scrucca & Raftery (2018) — Robust model-based clustering of voting records](https://ar5iv.labs.arxiv.org/html/1805.04203)
- [Kozlowski & Dukes (2025) — K-Means polarization measurement](https://royalsocietypublishing.org/rsos/article/13/2/251428/479919/A-new-measure-of-issue-polarization-using-k-means)
- [Spieksma et al. (2020) — Political niches and interval graphs](https://pmc.ncbi.nlm.nih.gov/articles/PMC7566643/)
- [Traag et al. (2019) — From Louvain to Leiden](https://www.nature.com/articles/s41598-019-41695-z)

### Validation
- [Liu et al. (2022) — Stability estimation for clustering: a review](https://wires.onlinelibrary.wiley.com/doi/10.1002/wics.1575)
- [Vinh et al. (2010) — NMI chance adjustment](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)
- [PeerJ (2025) — Silhouette vs Calinski-Harabasz comparison](https://peerj.com/articles/cs-3309/)

### Python Libraries
- [scikit-learn clustering module](https://scikit-learn.org/stable/modules/clustering.html)
- [HDBSCAN documentation](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html)
- [UMAP clustering guidance](https://umap-learn.readthedocs.io/en/latest/clustering.html)
- [StepMix — Latent Class Analysis](https://github.com/Labo-Lacourse/stepmix) (JSS 2025)
- [leidenalg — Leiden community detection](https://leidenalg.readthedocs.io/en/stable/intro.html)
- [consensusclustering — Monti et al. method](https://github.com/burtonrj/consensusclustering)
- [gap-stat — Gap statistic](https://pypi.org/project/gap-stat/)

### Missing Data
- [MICA — Multiple Imputation Clustering Analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC10291928/)
- [Roll-call vote selection (APSR)](https://www.cambridge.org/core/journals/american-political-science-review/article/rollcall-vote-selection-implications-for-the-study-of-legislative-politics/FFAD60FB9CA9BBD54F02DE44A1FF0264)

### Political Science
- [Voteview — DW-NOMINATE ideal points](https://voteview.com/data)
- [Nature Human Behaviour (2025) — Multidimensional ideological polarization](https://www.nature.com/articles/s41562-025-02251-0)
