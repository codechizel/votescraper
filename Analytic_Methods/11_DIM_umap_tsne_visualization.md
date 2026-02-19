# UMAP and t-SNE Visualization

**Category:** Dimensionality Reduction / Visualization
**Prerequisites:** `01_DATA_vote_matrix_construction`, ideally after `09_DIM_principal_component_analysis`
**Complexity:** Low-Medium

## What It Measures

UMAP (Uniform Manifold Approximation and Projection) and t-SNE (t-distributed Stochastic Neighbor Embedding) are non-linear dimensionality reduction techniques that preserve *local neighborhood structure* in the data. While PCA finds the best linear projection, UMAP and t-SNE can reveal clusters, sub-clusters, and non-linear relationships that PCA misses.

For legislative data, these methods excel at revealing *factions within parties* — moderates, hardliners, and swing voters that blur together in a linear PCA projection can separate clearly in UMAP/t-SNE space.

## Questions It Answers

- Are there clear visual clusters beyond the two-party split?
- How many distinct voting blocs exist within each party?
- Which legislators sit at the boundaries between clusters?
- Do the clusters align with any known caucuses or geographic patterns?

## Mathematical Foundation

### t-SNE

t-SNE converts high-dimensional distances into probabilities using Gaussian kernels, then finds a low-dimensional embedding where the same distances are represented by Student-t distributions (which have heavier tails, avoiding the "crowding problem"):

1. Compute pairwise similarities in high-D: $p_{ij} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq l} \exp(-\|x_k - x_l\|^2 / 2\sigma_k^2)}$
2. Define similarities in low-D using Student-t: $q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$
3. Minimize KL divergence: $\text{KL}(P \| Q) = \sum_{ij} p_{ij} \log \frac{p_{ij}}{q_{ij}}$

### UMAP

UMAP is mathematically grounded in Riemannian geometry and algebraic topology. It constructs a fuzzy simplicial complex (a weighted graph) in high-D and finds a low-D embedding that preserves the topological structure. In practice:

1. Construct a weighted k-nearest-neighbor graph in high-D
2. Optimize a low-D layout that minimizes cross-entropy between high-D and low-D edge weights

**Key difference from t-SNE:** UMAP better preserves *global structure* (the distance between clusters is meaningful), while t-SNE tends to equalize inter-cluster distances. For legislative data, UMAP is generally preferred because the distance between, say, moderate Republicans and conservative Republicans should be smaller than the distance between moderate Republicans and Democrats.

### Hyperparameters

| Parameter | UMAP | t-SNE | Effect |
|-----------|------|-------|--------|
| n_neighbors / perplexity | `n_neighbors` (default: 15) | `perplexity` (default: 30) | Controls local vs. global focus. Higher = more global structure. |
| min_dist | `min_dist` (default: 0.1) | N/A | Minimum distance between points. Lower = tighter clusters. |
| metric | Any sklearn metric | Euclidean or precomputed | Distance function in high-D. |
| n_components | 2 or 3 | 2 or 3 | Output dimensionality. |

## Python Implementation

```python
import pandas as pd
import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def compute_umap(
    vote_matrix: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute UMAP embedding of legislators from vote matrix.

    Args:
        vote_matrix: Binary vote matrix (imputed, no NaN).
        n_neighbors: Number of neighbors (higher = more global structure).
        min_dist: Minimum distance between points in embedding.
        metric: Distance metric ('cosine', 'euclidean', 'hamming').
        n_components: Output dimensions (2 for plotting).

    Returns:
        DataFrame with UMAP1, UMAP2 columns indexed by legislator_slug.
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(vote_matrix.values)

    return pd.DataFrame(
        embedding,
        index=vote_matrix.index,
        columns=[f"UMAP{i+1}" for i in range(n_components)],
    )


def compute_tsne(
    vote_matrix: pd.DataFrame,
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute t-SNE embedding."""
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(vote_matrix) - 1),
        random_state=random_state,
        n_iter=1000,
    )
    embedding = tsne.fit_transform(vote_matrix.values)

    return pd.DataFrame(
        embedding,
        index=vote_matrix.index,
        columns=[f"tSNE{i+1}" for i in range(n_components)],
    )
```

### Visualization with Multiple Colorings

```python
def plot_embedding(
    embedding: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    color_by: str = "party",
    method_name: str = "UMAP",
    save_path: str | None = None,
):
    """Plot 2D embedding with various coloring schemes."""
    meta = legislator_meta.loc[embedding.index]
    dim1, dim2 = embedding.columns[0], embedding.columns[1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Color by party
    party_colors = meta["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})
    axes[0].scatter(embedding[dim1], embedding[dim2], c=party_colors, s=40, alpha=0.7,
                    edgecolors="black", linewidth=0.3)
    axes[0].set_title(f"{method_name} — Colored by Party")

    # Color by chamber
    chamber_colors = meta["chamber"].map({"House": "#2ca02c", "Senate": "#9467bd"})
    axes[1].scatter(embedding[dim1], embedding[dim2], c=chamber_colors, s=40, alpha=0.7,
                    edgecolors="black", linewidth=0.3)
    axes[1].set_title(f"{method_name} — Colored by Chamber")

    # Color by district (numeric, gradient)
    districts = pd.to_numeric(meta["district"], errors="coerce")
    scatter = axes[2].scatter(embedding[dim1], embedding[dim2], c=districts, s=40, alpha=0.7,
                              cmap="viridis", edgecolors="black", linewidth=0.3)
    plt.colorbar(scatter, ax=axes[2], label="District Number")
    axes[2].set_title(f"{method_name} — Colored by District")

    for ax in axes:
        ax.set_xlabel(dim1)
        ax.set_ylabel(dim2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_umap_with_ideal_points(
    embedding: pd.DataFrame,
    pca_scores: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    save_path: str | None = None,
):
    """Color UMAP embedding by PCA-derived ideal point (PC1 score)."""
    meta = legislator_meta.loc[embedding.index]
    pc1 = pca_scores.loc[embedding.index, "PC1"]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embedding.iloc[:, 0],
        embedding.iloc[:, 1],
        c=pc1,
        cmap="RdBu_r",
        s=50,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="PC1 Score (ideal point proxy)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP Embedding Colored by Ideal Point")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

### Hyperparameter Sensitivity

```python
def plot_umap_sensitivity(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    n_neighbors_values: list[int] = [5, 15, 30, 50],
    save_path: str | None = None,
):
    """Show how UMAP embedding changes with n_neighbors."""
    fig, axes = plt.subplots(1, len(n_neighbors_values), figsize=(5 * len(n_neighbors_values), 5))

    for i, nn in enumerate(n_neighbors_values):
        embedding = compute_umap(vote_matrix, n_neighbors=nn)
        meta = legislator_meta.loc[embedding.index]
        colors = meta["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

        axes[i].scatter(embedding["UMAP1"], embedding["UMAP2"], c=colors, s=20, alpha=0.7)
        axes[i].set_title(f"n_neighbors = {nn}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
```

## Interpretation Guide

### Reading UMAP/t-SNE Plots

- **Tight clusters**: Groups of legislators who almost always vote together. In a partisan legislature, expect at least two clusters (one per party).
- **Sub-clusters within a party cluster**: Factions. If the Republican cluster splits into two sub-clusters, it suggests a moderate-conservative divide.
- **Bridge legislators** (between clusters): Swing voters or moderates who sometimes vote with the other party.
- **Isolated points**: Legislators with unique voting patterns. Could be extreme ideologues, chronic absentees, or new members with limited voting records.
- **Cluster distance (UMAP only)**: In UMAP, the distance between clusters is meaningful. Closer clusters are more similar. In t-SNE, inter-cluster distances are not reliable.

### What NOT to Interpret

- **Absolute positions** (left/right/up/down) are arbitrary. Only relative positions matter.
- **Cluster shape** (round vs. elongated) depends on hyperparameters, not the data.
- **Tight vs. diffuse clusters** depend on `min_dist` (UMAP) and `perplexity` (t-SNE), not just the data.

### UMAP vs t-SNE for Legislative Data

| Feature | UMAP (preferred) | t-SNE |
|---------|-------------------|-------|
| Global structure | Preserved | Not preserved |
| Speed | Faster | Slower |
| Determinism | More reproducible | Varies across runs |
| Inter-cluster distance | Meaningful | Not meaningful |
| Continuous gradients | Yes | Tends to form discrete blobs |

**Recommendation: Use UMAP** for legislative data. The continuous gradient from liberal to conservative is better represented by UMAP than by t-SNE's tendency to create discrete blobs.

## Kansas-Specific Considerations

- **Use cosine metric** rather than Euclidean for the vote matrix. Cosine similarity handles the binary vote data better and is less affected by the number of votes cast.
- **Analyze chambers separately** or use chamber as a color annotation to avoid conflating cross-chamber differences with ideological differences.
- **Run with multiple n_neighbors values** (5, 15, 30, 50) to assess robustness. The Republican factional structure should be visible across a range of settings.
- **Combine with PCA**: Color the UMAP embedding by PC1 scores to validate that the clusters correspond to the ideological dimension.

## Feasibility Assessment

- **Data size**: 170 legislators = tiny for UMAP/t-SNE (designed for millions of points)
- **Compute time**: Sub-second for UMAP, 1-3 seconds for t-SNE
- **Libraries**: `umap-learn`, `scikit-learn`
- **Difficulty**: Easy to run, moderate to interpret. Hyperparameter sensitivity is the main challenge.

## Key References

- McInnes, Leland, John Healy, and James Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." 2018. https://arxiv.org/abs/1802.03426
- Van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing Data using t-SNE." *Journal of Machine Learning Research* 9, 2008.
- `umap-learn` documentation: https://umap-learn.readthedocs.io/
