# ADR-0007: Clustering Implementation Choices

**Date:** 2026-02-20
**Status:** Accepted

## Context

Clustering (Phase 5) identifies discrete voting blocs using IRT ideal points from Phase 4. Several implementation choices needed to be resolved:

1. **Primary input space.** Clustering can operate on the raw vote matrix, PCA scores, IRT ideal points, or pairwise agreement distances. Each has different handling of missing data, dimensionality, and bill discrimination weighting.

2. **Number of methods.** A single clustering algorithm's result may be an artifact of its assumptions (e.g., k-means assumes spherical clusters). Multiple methods with cross-validation needed.

3. **The Tyson paradox.** IRT ideal points alone cannot distinguish "ideologically extreme" from "unreliable caucus member." Sen. Tyson ranks 1st (most conservative) by IRT but has 42% party loyalty — a complementary metric is needed.

4. **Within-party structure.** The party boundary dominates whole-chamber clustering (k=2 optimal). Detecting finer intra-party structure (moderate vs. conservative Republicans) requires removing the cross-party gap.

5. **Confidence weighting.** IRT provides posterior uncertainty (xi_sd). Legislators with sparse data (e.g., Miller with 30/194 votes) have wider intervals and should contribute less to cluster estimation.

6. **NaN handling in Kappa distance.** Some legislator pairs have insufficient shared votes to compute Cohen's Kappa, producing NaN entries (60 in House, 8 in Senate).

## Decision

1. **IRT ideal points as primary input, supplemented by party loyalty.** IRT accounts for bill discrimination and handles missing data natively. PCA requires imputation (producing artifacts like Miller's PC2 extreme). The Kappa distance matrix is used as a complementary input for hierarchical clustering only.

2. **Three methods for robustness.** Run hierarchical (Ward on Kappa distance), k-means (on IRT), and GMM (on IRT with confidence weighting) independently. Cross-method agreement is measured by Adjusted Rand Index (ARI). A finding is robust only if all three methods agree.

3. **Party loyalty metric.** For each legislator, compute the fraction of "contested" votes (>= 10% party dissent) where they agree with their party's median position. This provides a second axis orthogonal to ideology: the 2D space (IRT, loyalty) separates core party members from mavericks like Tyson and Thompson.

4. **Within-party clustering.** Cluster each party caucus separately using k-means on 1D (IRT ideal point) and 2D (IRT + party loyalty). Minimum caucus size of 15 members (the Senate Democratic caucus at 10 is too small). If best within-party silhouette < 0.50 for all k > 1, the variation is continuous, not discrete.

5. **GMM weighting by 1/xi_sd.** Legislators with tighter posterior intervals contribute more to cluster estimation. Implemented by observation replication proportional to 1/xi_sd, scaled to ~3x the original sample size.

6. **NaN Kappa distances filled with maximum finite distance.** Treats unknown pairs as maximally dissimilar — a conservative choice that prevents unknown pairs from being grouped together.

## Consequences

**Benefits:**
- Three independent methods with ARI cross-validation provide high confidence that k=2 is real structure, not algorithmic artifact (ARI = 0.96 House, 0.94 Senate).
- Party loyalty successfully separates Tyson (loyalty=0.42, xi=+4.17) and Thompson (loyalty=0.47, xi=+3.44) from core party members in the 2D view, despite same cluster assignment at k=2.
- Within-party analysis confirms intra-R variation is continuous (flat silhouette profile from k=2 to k=7), preventing over-interpretation of arbitrary k values.
- Confidence weighting down-weights noisy ideal points (Miller, Hill) without excluding them entirely.

**Trade-offs:**
- The `_CONTESTED_PARTY_THRESHOLD` constant in `clustering_report.py` must be kept in sync with `CONTESTED_PARTY_THRESHOLD` in `clustering.py` (circular import prevents direct import). A test validates this.
- Within-party silhouette > 0.50 but flat profile means "weakly structured continuous variation" — this is hard to communicate to non-technical audiences who expect discrete faction labels.
- GMM's BIC-optimal k=4 disagrees with silhouette-optimal k=2. The difference reflects their objectives (generative model fit vs. cluster separation), not a contradiction. Both are reported.

**Key finding (2026-02-20):** The initial hypothesis of k=3 (conservative Rs, moderate Rs, Democrats) was rejected. k=2 is optimal for both chambers — the party boundary is the dominant clustering structure. The moderate/conservative Republican distinction is continuous, not discrete.
