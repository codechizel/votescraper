# K-Means Clustering on Voting Patterns

**Category:** Clustering & Classification
**Prerequisites:** `01_DATA_vote_matrix_construction`, ideally `09_DIM_principal_component_analysis`
**Complexity:** Low
**Related:** `18_CLU_hierarchical_clustering`, `23_NET_community_detection`

## What It Measures

K-Means partitions legislators into $k$ groups that minimize within-cluster variance — each legislator is assigned to the cluster whose centroid (average voting profile) they most closely resemble. Unlike hierarchical clustering (which reveals tree structure), K-Means produces a flat partition. It is best used after PCA to cluster legislators in the reduced ideological space.

## Questions It Answers

- How many distinct voting blocs does the legislature effectively have?
- Do the K-Means clusters align with party labels, or do they reveal cross-cutting coalitions?
- What is the "average voting profile" of each cluster (the cluster centroids)?
- Which legislators are on the boundary between clusters (ambiguous bloc membership)?

## Mathematical Foundation

K-Means minimizes the within-cluster sum of squared distances:

$$\min_{C_1, \ldots, C_k} \sum_{j=1}^{k} \sum_{i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$

Where $\mathbf{x}_i$ is legislator $i$'s vote vector (or PC scores) and $\boldsymbol{\mu}_j$ is the centroid of cluster $j$.

### Choosing $k$

- **Elbow method**: Plot within-cluster sum of squares (inertia) vs. $k$. The "elbow" point is where adding more clusters gives diminishing returns.
- **Silhouette score**: Average silhouette coefficient across all points. Peaks at optimal $k$.
- **Gap statistic**: Compares observed inertia to expected inertia under a null (uniform) distribution.
- **Domain knowledge**: In a two-party legislature, start with $k = 2$ and increase. $k = 3$ (party split + intra-majority faction) is often the sweet spot.

## Python Implementation

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def find_optimal_k_kmeans(
    X: np.ndarray,
    k_range: range = range(2, 10),
) -> dict:
    """Find optimal k using elbow and silhouette methods."""
    results = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        results["k"].append(k)
        results["inertia"].append(kmeans.inertia_)
        results["silhouette"].append(silhouette_score(X, labels))

    return results


def run_kmeans(
    pca_scores: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    k: int = 3,
    n_components: int = 5,
) -> pd.DataFrame:
    """Run K-Means on PCA-reduced vote data.

    Args:
        pca_scores: PC scores from PCA analysis.
        legislator_meta: Legislator metadata.
        k: Number of clusters.
        n_components: Number of PCA components to use.

    Returns:
        DataFrame with cluster assignments and distances to centroids.
    """
    X = pca_scores.iloc[:, :n_components].values
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    distances = kmeans.transform(X)  # Distance to each centroid

    result = pd.DataFrame({
        "legislator_slug": pca_scores.index,
        "cluster": labels,
    })

    # Add distance to assigned centroid (uncertainty of assignment)
    result["dist_to_centroid"] = [distances[i, labels[i]] for i in range(len(labels))]

    # Merge with metadata
    meta = legislator_meta.loc[pca_scores.index]
    result["party"] = meta["party"].values
    result["chamber"] = meta["chamber"].values
    result["full_name"] = meta["full_name"].values

    return result, kmeans


def plot_kmeans_on_pca(
    pca_scores: pd.DataFrame,
    cluster_result: pd.DataFrame,
    kmeans: KMeans,
    save_path: str | None = None,
):
    """Visualize K-Means clusters in PC1-PC2 space."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by party (ground truth)
    party_colors = cluster_result["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})
    axes[0].scatter(pca_scores["PC1"], pca_scores["PC2"], c=party_colors, s=30, alpha=0.6)
    axes[0].set_title("Colored by Party")

    # Color by cluster
    cluster_cmap = plt.cm.Set2
    cluster_colors = [cluster_cmap(c) for c in cluster_result["cluster"]]
    axes[1].scatter(pca_scores["PC1"], pca_scores["PC2"], c=cluster_colors, s=30, alpha=0.6)

    # Plot centroids
    centroids = kmeans.cluster_centers_
    axes[1].scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=200,
                   edgecolors="white", linewidth=2, zorder=10, label="Centroids")
    axes[1].set_title(f"K-Means Clusters (k={kmeans.n_clusters})")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("PC1 (primary ideological dimension)")
        ax.set_ylabel("PC2 (secondary dimension)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def analyze_cluster_composition(cluster_result: pd.DataFrame) -> pd.DataFrame:
    """Analyze the party composition of each cluster."""
    composition = cluster_result.groupby(["cluster", "party"]).size().unstack(fill_value=0)
    composition["total"] = composition.sum(axis=1)
    for party in composition.columns[:-1]:
        composition[f"{party}_pct"] = composition[party] / composition["total"] * 100
    return composition
```

## Interpretation Guide

- **$k = 2$ aligns perfectly with parties**: Partisanship explains everything. The legislature is one-dimensional.
- **$k = 2$ but some misassignment**: A few moderates cluster with the opposing party. These are the most interesting legislators.
- **$k = 3$ is optimal**: The third cluster typically contains moderate Republicans (or occasionally, conservative Democrats). This is the common finding in supermajority legislatures.
- **$k = 4+$ produces small, interpretable clusters**: These might correspond to geographic caucuses, issue-specific coalitions, or freshman legislators.
- **Boundary legislators** (high dist_to_centroid) are the swing voters who don't fit neatly into any bloc.

## Kansas-Specific Considerations

- **Whole-chamber k=2 is optimal** (2026-02-20 finding). The initial k=3 hypothesis (conservative Rs, moderate Rs, Democrats) was rejected — silhouette at k=2 = 0.82 (House) / 0.79 (Senate) vs k=3 at 0.64/0.57. The party boundary is the dominant structure.
- **Within-party k-means** is needed to detect finer intra-party structure. Cluster each party caucus separately on IRT ideal points (1D) and IRT + party loyalty (2D). Within-party silhouette is > 0.50 but flat across k=2-7 — the variation is continuous, not discrete. See `analysis/design/clustering.md`.
- **Run on IRT ideal points** (not PCA scores or raw vote matrix). IRT handles missing data natively and weights discriminating bills. Supplement with party loyalty as a second dimension.
- **Cluster centroids** (the "average voter" in each cluster) can be compared to identify which issues split the majority party.
- **Analyze per-chamber**: The House and Senate may have different cluster structures.
- **Minimum caucus size of 15** for within-party clustering — the Senate Democratic caucus (10 members) is too small for reliable model selection.

## Feasibility Assessment

- **Data size**: 170 points in 2-5D = trivially fast
- **Compute time**: Sub-second
- **Libraries**: `scikit-learn`
- **Difficulty**: Low

## Key References

- Hartigan, J. A., and M. A. Wong. "Algorithm AS 136: A K-Means Clustering Algorithm." *Journal of the Royal Statistical Society. Series C* 28(1), 1979.
- Rousseeuw, Peter J. "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis." *Journal of Computational and Applied Mathematics* 20, 1987.
