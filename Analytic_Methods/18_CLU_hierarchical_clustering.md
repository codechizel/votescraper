# Hierarchical Clustering of Legislators

**Category:** Clustering & Classification
**Prerequisites:** `04_EDA_agreement_matrix_heatmap`
**Complexity:** Low-Medium
**Related:** `19_CLU_kmeans_voting_patterns`, `23_NET_community_detection`

## What It Measures

Hierarchical clustering builds a tree (dendrogram) of legislators based on voting similarity. It starts with each legislator as their own cluster and progressively merges the most similar pairs until everyone is in one cluster. The resulting tree shows the full hierarchical structure of legislative alliances — not just "two parties" but the nested factions within each party.

The dendrogram is one of the most informative single visualizations for legislative data: you can read off the party structure, identify factions, spot mavericks, and determine the natural number of clusters all from one plot.

## Questions It Answers

- What is the hierarchical structure of voting alliances?
- How many natural clusters exist (beyond the two-party split)?
- Within the Republican majority, what sub-factions emerge?
- Which legislators are the most "distant" from their party peers?
- At what level of similarity do the parties merge (how polarized are they)?

## Mathematical Foundation

### Distance Metric

Start with the agreement matrix $A$ (from `04_EDA_agreement_matrix_heatmap`). Convert to distance:

$$d(i, j) = 1 - A(i, j)$$

Alternatively, use Hamming distance on the binary vote vectors:

$$d_H(i, j) = \frac{|\{v : \text{vote}_i(v) \neq \text{vote}_j(v), \text{ both present}\}|}{|\{v : \text{both present}\}|}$$

### Linkage Methods

| Method | Merging Rule | Best For |
|--------|-------------|----------|
| **Ward** | Minimize within-cluster variance at each merge | Tends to produce balanced, equal-sized clusters. Best default. |
| **Complete** | Use maximum distance between cluster members | Produces compact, tight clusters. Conservative. |
| **Average** (UPGMA) | Use average distance between all cluster member pairs | Balanced between Ward and single linkage. |
| **Single** | Use minimum distance between cluster members | Finds elongated "chains." Can produce straggly clusters. Not recommended for legislative data. |

**Recommendation: Use Ward linkage** for legislative data. It produces the most interpretable dendrograms.

### Optimal Number of Clusters

Several criteria for choosing $k$:

1. **Visual inspection**: Cut the dendrogram at the level that produces substantively meaningful groups.
2. **Silhouette score**: Measures how similar each point is to its own cluster vs. other clusters. Higher is better.
3. **Gap statistic**: Compares within-cluster dispersion to a null reference distribution.
4. **Cophenetic correlation**: Measures how well the dendrogram preserves the original pairwise distances. Higher is better.

## Python Implementation

```python
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def perform_hierarchical_clustering(
    agreement_matrix: pd.DataFrame,
    method: str = "ward",
) -> np.ndarray:
    """Compute hierarchical clustering from agreement matrix.

    Args:
        agreement_matrix: Square DataFrame of pairwise agreement rates.
        method: Linkage method ('ward', 'complete', 'average', 'single').

    Returns:
        Linkage matrix (scipy format).
    """
    # Convert agreement to distance
    distance_matrix = 1 - agreement_matrix.fillna(0.5)
    np.fill_diagonal(distance_matrix.values, 0)

    # Ensure symmetry (numerical precision)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix.values)

    # Compute linkage
    Z = linkage(condensed, method=method)

    return Z


def plot_dendrogram(
    Z: np.ndarray,
    labels: list[str],
    legislator_meta: pd.DataFrame,
    chamber: str | None = None,
    save_path: str | None = None,
):
    """Plot a dendrogram colored by party."""
    meta = legislator_meta.loc[labels]
    party_colors = {
        "Republican": "#E81B23",
        "Democrat": "#0015BC",
    }

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 0.15), 8))

    # Plot dendrogram
    dend = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=6 if len(labels) <= 50 else 4,
        ax=ax,
        color_threshold=0,  # Color all links the same
        above_threshold_color="gray",
    )

    # Color leaf labels by party
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        slug = lbl.get_text()
        if slug in meta.index:
            party = meta.loc[slug, "party"]
            lbl.set_color(party_colors.get(party, "black"))

    title = f"Legislator Dendrogram — {chamber or 'All'} (Ward linkage)"
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Distance (1 - Agreement Rate)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def find_optimal_k(
    Z: np.ndarray,
    agreement_matrix: pd.DataFrame,
    k_range: range = range(2, 10),
) -> dict:
    """Find optimal number of clusters using silhouette analysis."""
    distance_matrix = 1 - agreement_matrix.fillna(0.5)
    np.fill_diagonal(distance_matrix.values, 0)

    results = {}
    for k in k_range:
        cluster_labels = fcluster(Z, k, criterion="maxclust")
        score = silhouette_score(distance_matrix.values, cluster_labels, metric="precomputed")
        results[k] = score

    return results


def plot_silhouette_analysis(
    silhouette_scores: dict,
    save_path: str | None = None,
):
    """Plot silhouette scores vs. number of clusters."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())

    ax.plot(ks, scores, "bo-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Optimal Number of Clusters")

    best_k = ks[np.argmax(scores)]
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def assign_clusters(
    Z: np.ndarray,
    labels: list[str],
    legislator_meta: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """Assign legislators to k clusters and analyze composition."""
    cluster_labels = fcluster(Z, k, criterion="maxclust")

    result = pd.DataFrame({
        "legislator_slug": labels,
        "cluster": cluster_labels,
    })
    result = result.merge(
        legislator_meta.reset_index()[["slug", "party", "chamber", "full_name"]],
        left_on="legislator_slug",
        right_on="slug",
    )

    # Cluster composition summary
    composition = result.groupby(["cluster", "party"]).size().unstack(fill_value=0)
    print("Cluster composition:")
    print(composition)

    return result
```

## Interpretation Guide

### Reading the Dendrogram

- **The first major split** (highest branch point) typically separates Democrats from Republicans.
- **Subsequent splits within a party** reveal factions: moderate vs. conservative Republicans, etc.
- **Branch height** indicates how different the groups are. A tall branch between clusters = very different voting patterns. Short branch = similar.
- **Leaf color** (party) that doesn't match the cluster assignment identifies mavericks: legislators grouped with the other party.
- **Singletons** (legislators who merge late) have unique voting patterns — investigate why.

### Expected Kansas Structure

With $k = 2$: Two clusters aligning with party. **This is the optimal k** (2026-02-20 finding). Silhouette at k=2 = 0.75 (House) / 0.71 (Senate) for hierarchical clustering on Kappa distance.
With $k = 3$: Previously hypothesized (Democrats, moderate Rs, conservative Rs) but **rejected** — silhouette drops substantially (0.62/0.51). The moderate/conservative R distinction is continuous, not discrete.
With $k = 4+$: GMM/BIC selects k=4, but this captures distributional shape (e.g., the long right tail of Republican ideal points), not genuine factions.

### Cophenetic Correlation

```python
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

c, coph_dists = cophenet(Z, squareform(distance_matrix.values))
print(f"Cophenetic correlation: {c:.3f}")
```

Values above 0.7 indicate the dendrogram is a good representation of the original distances.

## Kansas-Specific Considerations

- **Run per chamber.** House (130 members) and Senate (42 members) should be analyzed separately.
- **Senate dendrogram is the more readable one.** With 42 members, every label is visible and the structure is clear.
- **For the House**, consider showing a truncated dendrogram (only the top $k$ levels) since 130 leaf labels are hard to read.
- **k=2 is optimal** (2026-02-20 finding). The initial k=3 hypothesis was rejected. See `analysis/design/clustering.md` for full rationale.
- **Within-party analysis**: After establishing k=2 at whole-chamber level, cluster each party caucus separately to look for finer structure. Intra-party variation is weakly structured and continuous.
- **Filter to contested votes** before computing the agreement matrix. On near-unanimous votes, everyone clusters together.
- **NaN handling**: Some legislator pairs lack sufficient shared votes for Kappa — fill with maximum finite distance (conservative: unknown pairs treated as maximally dissimilar).

## Feasibility Assessment

- **Data size**: 170 x 170 distance matrix = trivial
- **Compute time**: Sub-second
- **Libraries**: `scipy.cluster.hierarchy`, `seaborn`, `sklearn.metrics`
- **Difficulty**: Low (computation), Medium (interpretation)

## Key References

- Andris, Clio, et al. "The Rise of Partisanship and Super-Cooperators in the US House of Representatives." *PLOS ONE* 10(4), 2015.
- Murtagh, Fionn, and Pierre Legendre. "Ward's Hierarchical Agglomerative Clustering Method: Which Algorithms Implement Ward's Criterion?" *Journal of Classification* 31, 2014.
