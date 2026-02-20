# Clustering Design Choices

**Script:** `analysis/clustering.py`
**Constants defined at:** `analysis/clustering.py:48-62`

## Assumptions

1. **Discrete faction structure exists.** The clustering hypothesis is that Kansas legislators form recognizable voting blocs — not a continuous spectrum — that can be recovered from IRT ideal points and voting agreement. The IRT design doc expected three clusters: conservative Republicans, moderate Republicans, and Democrats. **Actual finding (2026-02-20): k=2 is optimal for both chambers — the moderate/conservative R distinction is continuous, not discrete. See "Key Findings" below.**

2. **IRT ideal points capture the primary ideological dimension.** Clustering operates on IRT xi_mean (1D) rather than raw vote matrices or PCA scores. IRT accounts for bill discrimination, handles missing data natively, and provides uncertainty estimates — all improvements over PCA for clustering input.

3. **Party loyalty captures a complementary dimension.** The "Tyson paradox" (see `analysis/design/tyson_paradox.md`) shows that IRT ideal points alone cannot distinguish "ideologically extreme" from "unreliable caucus member." A party loyalty metric — fraction of contested votes where a legislator agrees with their party median — provides a second axis orthogonal to ideology.

4. **Chambers are independent.** House and Senate are clustered separately. Cross-chamber clustering is not attempted (different bill sets, different dynamics). This is consistent with all upstream phases.

5. **Ward linkage produces the most interpretable dendrograms for legislative data.** Ward minimizes within-cluster variance, producing balanced, compact clusters. This is preferred over single-linkage (chaining artifacts) or complete-linkage (unbalanced clusters) for ~40-130 legislators.

## Parameters & Constants

| Constant | Value | Justification | Code location |
|----------|-------|---------------|---------------|
| `RANDOM_SEED` | 42 | Reproducibility; consistent with EDA/PCA/IRT | `clustering.py:48` |
| `K_RANGE` | range(2, 8) | Evaluate k=2 through k=7; covers expected k=3 with margin | `clustering.py:49` |
| `DEFAULT_K` | 3 | Expected optimal k (conservative R, moderate R, Democrat) | `clustering.py:50` |
| `LINKAGE_METHOD` | "ward" | Ward minimizes within-cluster variance; produces balanced clusters | `clustering.py:51` |
| `COPHENETIC_THRESHOLD` | 0.70 | Minimum acceptable cophenetic correlation for dendrogram validity | `clustering.py:52` |
| `SILHOUETTE_GOOD` | 0.50 | Silhouette > 0.5 indicates "good" cluster structure | `clustering.py:53` |
| `GMM_COVARIANCE` | "full" | Full covariance for GMM (only ~130 points; diag insufficient) | `clustering.py:54` |
| `GMM_N_INIT` | 20 | Multiple random restarts for GMM stability | `clustering.py:55` |
| `MINORITY_THRESHOLD` | 0.025 | Inherited from EDA; for sensitivity re-filtering | `clustering.py:58` |
| `SENSITIVITY_THRESHOLD` | 0.10 | Alternative minority threshold for sensitivity analysis | `clustering.py:59` |
| `MIN_VOTES` | 20 | Inherited from EDA; minimum substantive votes per legislator | `clustering.py:60` |
| `CONTESTED_PARTY_THRESHOLD` | 0.10 | A vote is "contested" for a party if >= 10% of the party dissents | `clustering.py:61` |

## Methodological Choices

### Primary input: IRT ideal points (not PCA)

**Decision:** Use IRT xi_mean as the primary clustering input, supplemented by party loyalty.

**Why:** IRT accounts for bill discrimination and handles missing data natively. PCA requires imputation for absent legislators (producing artifacts like Miller's PC2 extreme). IRT also provides xi_sd, which enables confidence-weighted clustering via GMM.

**Alternatives considered:**
- PCA scores (PC1 + PC2) — rejected: imputation artifacts for sparse legislators; no native uncertainty
- Raw vote matrix — rejected: high-dimensional, sparse, no discrimination weighting
- Agreement Kappa matrix — used for hierarchical clustering as a complementary method, not primary

### Party loyalty metric

**Decision:** For each legislator, compute the fraction of "contested" votes where they agree with their party's median position. A vote is "contested" for a party if >= 10% of the party's voting members dissent.

**Why:** Addresses the Tyson paradox directly. Tyson has an extreme IRT position (+4.17) because she's perfectly conservative on discriminating bills, but she dissents from her party on many routine bills. Party loyalty captures this second dimension: Tyson will have a low loyalty score despite her extreme ideological position.

**Impact:** The 2D space (IRT ideal point, party loyalty) should separate three types:
- Core party members: extreme ideology + high loyalty
- Mavericks: moderate-to-extreme ideology + low loyalty (Tyson, Thompson)
- Cross-pressured moderates: moderate ideology + moderate loyalty

### Three methods for robustness

**Decision:** Run hierarchical clustering (on Kappa distance), k-means (on IRT), and GMM (on IRT) independently. A finding is robust only if all three methods agree.

**Why:** Each method has different assumptions:
- Hierarchical/Ward: agglomerative, based on pairwise agreement distance
- K-means: centroid-based, assumes spherical clusters of equal size
- GMM: probabilistic, allows elliptical clusters and soft assignments

Cross-method agreement (measured by Adjusted Rand Index) validates that the cluster structure is real and not an artifact of a particular algorithm.

### GMM weighting by 1/xi_sd

**Decision:** Weight GMM observations by 1/xi_sd — legislators with tighter posterior intervals contribute more to cluster estimation.

**Why:** Down-weights uncertain legislators (Miller: 0.90 HDI width; Hill: 2.03 HDI width) whose ideal-point estimates are noisy. Prevents these uncertain observations from distorting cluster boundaries.

**Implementation:** Not a built-in sklearn feature. Implemented by repeating observations proportional to 1/xi_sd (normalized), scaled to an effective sample count of ~3x the original sample size for numerical stability. This approximates importance weighting.

### Ward linkage on Kappa distance

**Decision:** Hierarchical clustering uses distance = 1 - Kappa (from EDA agreement matrices), with Ward linkage.

**Why:** Kappa corrects for the 82% Yea base rate (raw agreement is inflated by unanimous votes). Ward produces balanced, compact clusters suited to legislative groupings. The cophenetic correlation coefficient validates that the dendrogram faithfully represents the distance matrix.

**NaN handling:** Some legislator pairs have insufficient shared votes to compute Kappa, producing NaN entries (60 in House, 8 in Senate). These are filled with the maximum finite distance, treating unknown pairs as maximally dissimilar. This is conservative — it prevents unknown pairs from being grouped together.

### k selection via silhouette

**Decision:** For k-means and hierarchical, select k by maximizing silhouette score across K_RANGE. For GMM, use BIC. Expected k=3 for both chambers.

**Why:** Silhouette balances cluster cohesion and separation. BIC penalizes model complexity, naturally selecting the most parsimonious number of components. Both are standard model selection criteria.

**Fallback:** If automated selection disagrees with k=3, report both the optimal k and the k=3 assignments for comparison. Use `--k 3` to force k=3 for comparison.

**Actual finding:** Both chambers selected k=2 by silhouette (hierarchical and k-means). GMM selected k=4 by BIC. The silhouette/BIC disagreement reflects their different objectives: silhouette measures cluster separation in the observation space, while BIC measures generative model fit. The GMM at k=4 likely captures finer distributional structure (e.g., the long right tail of Republican ideal points) that doesn't correspond to discrete factions.

### Veto override subgroup analysis

**Decision:** Separately analyze the ~34 veto override votes, which require a 2/3 supermajority.

**Why:** Veto overrides reveal cross-party coalitions (some Rs join Ds, or vice versa) that are invisible in the full-dataset clusters. The higher threshold creates different alignment incentives than simple-majority votes.

## Key Findings (2026-02-20)

The initial hypothesis predicted k=3 (conservative Rs, moderate Rs, Democrats). **All silhouette-based methods found k=2 as optimal for both chambers.** The two clusters correspond exactly to the party split: all Republicans in one cluster, all Democrats in the other.

| Chamber | Method | Optimal k | Silhouette at k=2 | Silhouette at k=3 |
|---------|--------|-----------|--------------------|--------------------|
| House | Hierarchical | 2 | 0.746 | 0.624 |
| House | K-Means (1D) | 2 | 0.822 | 0.636 |
| Senate | Hierarchical | 2 | 0.709 | 0.512 |
| Senate | K-Means (1D) | 2 | 0.792 | 0.573 |

**Interpretation:** The moderate-vs-conservative Republican distinction is continuous, not discrete. There is no clean separation point within the Republican caucus. The party boundary is the dominant structure. This is consistent with Kansas's Republican supermajority: with ~72% Rs, intra-party variation is spread across a wide range of the ideal-point spectrum without forming distinct subclusters.

**Cross-method agreement:** Mean ARI = 0.958 (House), 0.935 (Senate) — very strong, confirming the 2-cluster structure is robust across methods.

**Party loyalty adds a second dimension but does not change k.** The 2D (IRT + loyalty) silhouette scores are similar to 1D, meaning loyalty does not create new clusters. However, the 2D view is valuable for identifying mavericks: Tyson (loyalty=0.417) and Thompson (0.472) are visually separated from core party members despite being in the same cluster.

**Veto overrides show no cross-party coalition.** Override Yea rates are 98% (R cluster) vs 1% (D cluster). The 2/3 threshold did not produce detectable bipartisan voting blocs — overrides were strictly party-line.

### Within-Party Clustering

**Decision:** Cluster each party caucus separately using k-means on 1D (IRT ideal point) and 2D (IRT + party loyalty rate).

**Why:** Whole-chamber k=2 is dominated by the D-R party boundary. Removing it allows finer structure (e.g., moderate vs. conservative Republicans) to emerge if it exists as discrete subclusters.

**Minimum caucus size:** `WITHIN_PARTY_MIN_SIZE = 15`. The Senate Democratic caucus (10 members) is too small for meaningful k-means model selection.

**Interpretation threshold:** If within-party silhouette < `SILHOUETTE_GOOD` (0.50) for all k > 1, the variation is continuous, not discrete. This is a valid and important finding — it means legislators are spread across a spectrum rather than forming distinct factions.

**Findings (2026-02-20):**

| Chamber | Party | N | Optimal k (1D) | Silhouette (1D) | Note |
|---------|-------|---|-----------------|-----------------|------|
| House | Republican | 92 | 6 | 0.605 | Flat profile (0.57-0.60 across k); 1D structure but not strongly peaked |
| House | Democrat | 38 | 7 | 0.615 | Similar flat profile; 1D granularity but not strongly peaked |
| Senate | Republican | 32 | 3 | 0.606 | 2D (k=4, sil=0.612) slightly better; modest structure |
| Senate | Democrat | 10 | N/A | N/A | Skipped (below 15-member minimum) |

**Interpretation:** Within-party silhouette scores exceed 0.50 but are not strongly peaked at any particular k — the silhouette curve is essentially flat from k=2 to k=7. This means:
1. There *is* more structure than a single blob (silhouette > 0.50), but...
2. The "optimal" k is somewhat arbitrary — the data doesn't clearly prefer k=3 over k=5 or k=6.
3. The within-party variation is better characterized as **weakly structured continuous variation** than as discrete factions.

For downstream analysis, continuous features (IRT ideal points, party loyalty rates) will be more informative than within-party cluster labels. Network analysis may reveal community structure through pairwise agreement patterns rather than centroid-based clustering.

## Downstream Implications

### For Network Analysis (Phase 6)
- Cluster labels annotate network nodes, enabling visual and quantitative assessment of within-cluster vs. between-cluster connectivity
- Within-cluster density should be high; between-cluster density should be low
- If community detection recovers different groups than clustering, investigate whether the network captures dynamics that static clustering misses

### For Prediction (Phase 7)
- With k=2, cluster membership is equivalent to party — unlikely to add predictive power beyond what party already provides
- Party loyalty as a continuous feature may be more useful than cluster labels
- Consider running prediction with both k=2 and forced k=3 to test whether the finer granularity helps

### For Interpretation
- Cluster labels are heuristic ("Conservative R" = highest mean xi, majority R; "Democrat" = lowest mean xi, majority D). They capture the dominant party composition but individual members may be surprising.
- Tyson and Thompson are expected to cluster with conservative Rs (extreme IRT positions) despite low party loyalty. The 2D plot (IRT x loyalty) will make this visually clear.
- Miller's cluster assignment is low-confidence due to sparse data. Hill's assignment is the least certain in the Senate due to widest HDI.
