# Community Detection (Louvain/Leiden)

**Category:** Network Analysis
**Prerequisites:** `20_NET_covoting_network`
**Complexity:** Medium
**Related:** `18_CLU_hierarchical_clustering`, `19_CLU_kmeans_voting_patterns`

## What It Measures

Community detection algorithms find groups of densely interconnected nodes in a network — legislators who vote together more often than expected by chance. Unlike K-Means (which requires specifying $k$ in advance), algorithms like Louvain and Leiden automatically determine the optimal number of communities by maximizing network modularity.

Community detection answers the question: *if we only knew the voting patterns (not the party labels), how would we divide the legislature into groups?*

## Questions It Answers

- How many natural voting communities exist in the legislature?
- Do the communities align with party labels, or reveal unexpected coalitions?
- What is the modularity score (how strong is the community structure)?
- Are there sub-communities within each party?
- Which legislators sit at community boundaries?

## Mathematical Foundation

### Modularity

Modularity $Q$ measures how much more dense within-community edges are compared to a random graph with the same degree sequence:

$$Q = \frac{1}{2m} \sum_{ij} \left(A_{ij} - \frac{k_i k_j}{2m}\right) \delta(c_i, c_j)$$

Where:
- $A_{ij}$ = edge weight between $i$ and $j$
- $k_i$ = degree of node $i$ (sum of edge weights)
- $m$ = total edge weight
- $c_i$ = community of node $i$
- $\delta(c_i, c_j) = 1$ if $c_i = c_j$ (same community)

**Louvain algorithm**: Greedy modularity maximization. Fast ($O(n \log n)$). May produce poorly connected communities.

**Leiden algorithm**: Improved version of Louvain. Guaranteed to produce connected communities. Slightly slower but more reliable.

### Resolution Parameter

Both Louvain and Leiden accept a resolution parameter $\gamma$:
- $\gamma = 1$: Standard modularity (default)
- $\gamma < 1$: Favors fewer, larger communities
- $\gamma > 1$: Favors more, smaller communities

This allows exploring community structure at different granularities.

## Python Implementation

```python
import networkx as nx
import pandas as pd
import numpy as np
from community import community_louvain  # python-louvain package
import matplotlib.pyplot as plt

def detect_communities(
    G: nx.Graph,
    method: str = "louvain",
    resolution: float = 1.0,
) -> dict:
    """Detect communities in the co-voting network.

    Args:
        G: Co-voting network.
        method: 'louvain' or 'leiden'.
        resolution: Resolution parameter (higher = more communities).

    Returns:
        Dict mapping node -> community_id.
    """
    if method == "louvain":
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=42,
        )
    elif method == "leiden":
        try:
            import igraph as ig
            import leidenalg

            # Convert networkx to igraph
            ig_graph = ig.Graph.from_networkx(G)
            result = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=42,
            )
            partition = {G.nodes()[i]: result.membership[i] for i in range(len(G.nodes()))}
        except ImportError:
            print("leidenalg not installed, falling back to Louvain")
            partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
    else:
        raise ValueError(f"Unknown method: {method}")

    return partition


def analyze_communities(
    partition: dict,
    G: nx.Graph,
    legislator_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze the composition of detected communities."""
    # Compute modularity
    communities_list = {}
    for node, comm in partition.items():
        if comm not in communities_list:
            communities_list[comm] = set()
        communities_list[comm].add(node)

    modularity = community_louvain.modularity(partition, G, weight="weight")
    print(f"Modularity: {modularity:.4f}")
    print(f"Number of communities: {len(communities_list)}")

    # Community composition
    results = []
    for comm_id, members in sorted(communities_list.items()):
        for slug in members:
            party = G.nodes[slug].get("party", "Unknown") if slug in G.nodes else "Unknown"
            chamber = G.nodes[slug].get("chamber", "Unknown") if slug in G.nodes else "Unknown"
            results.append({
                "legislator_slug": slug,
                "community": comm_id,
                "party": party,
                "chamber": chamber,
            })

    df = pd.DataFrame(results)

    # Summary table
    composition = df.groupby(["community", "party"]).size().unstack(fill_value=0)
    composition["total"] = composition.sum(axis=1)
    print("\nCommunity composition:")
    print(composition)

    return df


def plot_communities_on_network(
    G: nx.Graph,
    partition: dict,
    save_path: str | None = None,
):
    """Visualize detected communities on the network layout."""
    pos = nx.spring_layout(G, weight="weight", k=1/np.sqrt(G.number_of_nodes()),
                           iterations=50, seed=42)

    n_communities = len(set(partition.values()))
    cmap = plt.cm.get_cmap("Set2", n_communities)
    node_colors = [cmap(partition[node]) for node in G.nodes()]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Left: colored by community
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=node_colors,
                          node_size=50, alpha=0.8, edgecolors="black", linewidths=0.3)
    nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.03, width=0.3)
    axes[0].set_title(f"Communities Detected ({n_communities} groups)")
    axes[0].axis("off")

    # Right: colored by party (for comparison)
    party_colors = [
        "#E81B23" if G.nodes[n].get("party") == "Republican" else "#0015BC"
        for n in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=party_colors,
                          node_size=50, alpha=0.8, edgecolors="black", linewidths=0.3)
    nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.03, width=0.3)
    axes[1].set_title("Party Labels (ground truth)")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")


def multi_resolution_analysis(
    G: nx.Graph,
    resolutions: list[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
) -> pd.DataFrame:
    """Run community detection at multiple resolutions."""
    results = []
    for gamma in resolutions:
        partition = detect_communities(G, resolution=gamma)
        n_communities = len(set(partition.values()))
        modularity = community_louvain.modularity(partition, G, weight="weight")
        results.append({
            "resolution": gamma,
            "n_communities": n_communities,
            "modularity": modularity,
        })

    return pd.DataFrame(results)
```

## Interpretation Guide

### Community-Party Alignment

| Scenario | N Communities | Party Alignment | Interpretation |
|----------|-------------|-----------------|----------------|
| 2 communities, perfect party match | 2 | 100% | Pure partisanship — party explains all voting |
| 2 communities, some cross-party | 2 | 90%+ | Strong partisanship with a few moderates "misclassified" |
| 3+ communities, party splits | 3+ | Partial | Intra-party factions detected |
| Many communities, no party pattern | 5+ | Weak | Ideological dimensions beyond partisanship |

### Multi-Resolution Insights

- **$\gamma = 0.5$**: Likely merges the two parties into one community (the "everyone agrees on most votes" effect). Useful baseline.
- **$\gamma = 1.0$**: Standard resolution. Usually finds 2-4 communities.
- **$\gamma = 2.0$**: Finds finer-grained sub-communities. May reveal moderate caucus, freshman class, geographic blocs.
- **Modularity peaking at $\gamma = 1.0$** suggests the standard resolution captures the "natural" structure best.

### Normalized Mutual Information (NMI)

Compare detected communities to party labels using NMI:

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(
    [partition[n] for n in G.nodes()],
    [G.nodes[n]["party"] for n in G.nodes()],
)
print(f"NMI (communities vs. party): {nmi:.3f}")
```

NMI = 1.0: Communities perfectly match parties. NMI = 0: No relationship.

## Kansas-Specific Considerations

- **At resolution 1.0, expect 2-3 communities**: one Democratic, one or two Republican.
- **At resolution 1.5-2.0, the Republican community should split** into moderate and conservative factions. This is the most informative resolution for Kansas analysis.
- **The Democratic community may remain unified** even at higher resolutions (small group, more cohesive).
- **Compare community membership with maverick scores** from `08_IDX_loyalty_and_maverick_scores`. Legislators detected as mavericks should appear at community boundaries or in unexpected communities.

## Feasibility Assessment

- **Data size**: 170 nodes = near-instantaneous for Louvain/Leiden
- **Compute time**: Sub-second
- **Libraries**: `python-louvain` (Louvain), optionally `leidenalg` + `igraph` (Leiden)
- **Difficulty**: Low (running), Medium (interpretation and resolution selection)

## Key References

- Blondel, Vincent D., et al. "Fast Unfolding of Communities in Large Networks." *Journal of Statistical Mechanics: Theory and Experiment*, 2008.
- Traag, V. A., L. Waltman, and N. J. van Eck. "From Louvain to Leiden: Guaranteeing Well-Connected Communities." *Scientific Reports* 9, 2019.
- Fortunato, Santo. "Community Detection in Graphs." *Physics Reports* 486(3-5), 2010.
