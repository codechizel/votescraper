# Co-Voting Network Analysis

**Category:** Network Analysis
**Prerequisites:** `04_EDA_agreement_matrix_heatmap`
**Complexity:** Medium
**Related:** `21_NET_bipartite_bill_legislator`, `22_NET_centrality_measures`, `23_NET_community_detection`

## What It Measures

A co-voting network represents the legislature as a graph where legislators are nodes and edges represent voting similarity. This transforms the legislative analysis from a matrix problem to a network problem, unlocking a powerful toolkit: centrality measures identify influential legislators, community detection reveals factions, and network visualization provides an intuitive map of the political landscape.

The co-voting network captures relationships that neither party labels nor simple clustering can express — transitive alliances, bridging behavior, isolation, and the topology of political coalitions.

## Questions It Answers

- What does the "social structure" of the legislature look like as a network?
- Which legislators are the most connected (vote similarly to many peers)?
- Which legislators bridge the partisan divide?
- Are there isolated legislators who vote differently from everyone?
- How dense and clustered is each party's voting network?
- Does the network have small-world properties (highly clustered but short path lengths)?

## Mathematical Foundation

### Network Construction

Given the agreement matrix $A$, construct a weighted undirected graph $G = (V, E, W)$:

- **Nodes** $V$: Legislators ($n = 172$)
- **Edges** $E$: Pairs of legislators with agreement above a threshold
- **Weights** $W$: Agreement rates

**Edge thresholding**: Include edge $(i, j)$ only if $A(i, j) > \tau$. Common choices:
- $\tau = 0$: Include all edges (fully connected). Good for weighted analysis.
- $\tau = 0.5$: Include only above-chance agreement. Removes random connections.
- $\tau = 0.7$: Include only substantial agreement. Produces cleaner network structure.

### Key Network Metrics

| Metric | Formula | What It Reveals |
|--------|---------|-----------------|
| **Density** | $\frac{2|E|}{n(n-1)}$ | Fraction of possible edges present |
| **Clustering coefficient** | $\frac{\text{triangles}_i}{\text{possible triangles}_i}$ | How much neighbors of $i$ also agree with each other |
| **Average path length** | $\frac{1}{n(n-1)} \sum_{i \neq j} d(i,j)$ | Average steps to get from any legislator to any other |
| **Modularity** | $Q = \frac{1}{2m} \sum_{ij} (A_{ij} - \frac{k_ik_j}{2m}) \delta(c_i, c_j)$ | Quality of the partition into communities |
| **Assortativity** | Pearson correlation of degree across edges | Do high-degree nodes connect to other high-degree nodes? |

## Python Implementation

### Building the Network

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def build_covoting_network(
    agreement_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    threshold: float = 0.5,
    use_weights: bool = True,
) -> nx.Graph:
    """Build co-voting network from agreement matrix.

    Args:
        agreement_matrix: Pairwise agreement rates.
        legislator_meta: Metadata for node attributes.
        threshold: Minimum agreement to create an edge.
        use_weights: Include agreement rates as edge weights.

    Returns:
        NetworkX Graph with legislator nodes and co-voting edges.
    """
    G = nx.Graph()

    # Add nodes with attributes
    for slug in agreement_matrix.index:
        if slug in legislator_meta.index:
            meta = legislator_meta.loc[slug]
            G.add_node(
                slug,
                party=meta["party"],
                chamber=meta["chamber"],
                full_name=meta.get("full_name", slug),
                district=meta.get("district", ""),
            )

    # Add edges
    slugs = agreement_matrix.index.tolist()
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            agreement = agreement_matrix.iloc[i, j]
            if pd.notna(agreement) and agreement > threshold:
                weight = agreement if use_weights else 1
                G.add_edge(slugs[i], slugs[j], weight=weight, agreement=agreement)

    return G


def network_summary(G: nx.Graph) -> dict:
    """Compute basic network statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G, weight="weight"),
        "transitivity": nx.transitivity(G),
        "is_connected": nx.is_connected(G),
        "n_components": nx.number_connected_components(G),
    }
```

### Network Visualization

```python
def plot_covoting_network(
    G: nx.Graph,
    layout: str = "spring",
    save_path: str | None = None,
):
    """Visualize the co-voting network.

    Args:
        G: Co-voting network.
        layout: Layout algorithm ('spring', 'kamada_kawai', 'spectral').
    """
    # Compute layout
    if layout == "spring":
        pos = nx.spring_layout(G, weight="weight", k=1/np.sqrt(G.number_of_nodes()),
                               iterations=50, seed=42)
    elif layout == "kamada_kawai":
        # Use distance = 1 - agreement for layout
        pos = nx.kamada_kawai_layout(G, weight="weight")
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors by party
    node_colors = []
    for node in G.nodes():
        party = G.nodes[node].get("party", "Unknown")
        node_colors.append("#E81B23" if party == "Republican" else "#0015BC")

    # Node sizes by degree
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 300 * degrees[n] / max_deg for n in G.nodes()]

    # Edge weights for transparency
    edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.02 + 0.15 * w / max_w for w in edge_weights]

    fig, ax = plt.subplots(figsize=(16, 14))

    # Draw edges (light, in background)
    for (u, v), alpha in zip(G.edges(), edge_alphas):
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color="gray", alpha=alpha, linewidth=0.3,
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    # Label key nodes (highest betweenness centrality)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
    labels = {n: n.split("_")[1].title() for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    ax.set_title("Co-Voting Network", fontsize=14)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E81B23", label="Republican"),
        Patch(facecolor="#0015BC", label="Democrat"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
```

### Cross-Party Edge Analysis

```python
def analyze_cross_party_edges(G: nx.Graph) -> pd.DataFrame:
    """Analyze edges that cross party boundaries."""
    cross_party = []
    within_party = []

    for u, v, data in G.edges(data=True):
        party_u = G.nodes[u].get("party")
        party_v = G.nodes[v].get("party")
        agreement = data.get("agreement", data.get("weight", 0))

        if party_u != party_v:
            cross_party.append({"from": u, "to": v, "agreement": agreement})
        else:
            within_party.append({"from": u, "to": v, "agreement": agreement,
                               "party": party_u})

    print(f"Within-party edges: {len(within_party)}")
    print(f"Cross-party edges: {len(cross_party)}")
    print(f"Cross-party ratio: {len(cross_party) / (len(cross_party) + len(within_party)):.2%}")

    cross_df = pd.DataFrame(cross_party).sort_values("agreement", ascending=False)
    return cross_df
```

## Interpretation Guide

### Network Topology

- **Two dense clusters connected by sparse bridges**: Strong partisanship with a few bipartisan moderates.
- **One dense cluster with outliers**: One dominant party (the supermajority) forms a dense core; the minority party is on the periphery.
- **Multiple clusters**: Multi-factional legislature.
- **Hub-and-spoke pattern**: Central leadership with peripheral backbenchers.

### Key Visual Cues

- **Node size** (degree): Larger nodes agree with more legislators. Party leaders and moderates tend to have high within-party degree.
- **Position in layout**: Nodes between clusters (spring layout) are bridge legislators. Nodes at the periphery are mavericks or ideologues.
- **Edge density**: Dense clusters of edges indicate highly cohesive voting blocs.

### Modularity

- $Q > 0.3$: Meaningful community structure (strong partisanship)
- $Q > 0.5$: Very strong community structure
- $Q > 0.7$: Exceptional (rare in real networks)

## Kansas-Specific Considerations

- **Threshold selection matters.** With a Republican supermajority, most legislators agree on most votes. Use a high threshold ($\tau \geq 0.7$) to reveal the interesting structure.
- **The most informative edges are cross-party.** Which Republicans have high agreement with Democrats? These are the moderate bridge-builders.
- **Network on contested votes only** will be more informative than on all votes.
- **Spring layout with weight=agreement** will naturally push parties apart and keep party members together.

## Feasibility Assessment

- **Data size**: 170 nodes = small for network analysis
- **Compute time**: Sub-second for all metrics
- **Libraries**: `networkx`, `matplotlib`
- **Difficulty**: Medium (construction is easy; meaningful visualization takes iteration)

## Key References

- Andris, Clio, et al. "The Rise of Partisanship and Super-Cooperators in the US House of Representatives." *PLOS ONE* 10(4), 2015.
- Newman, Mark. *Networks: An Introduction*. Oxford University Press, 2010.
- Waugh, Andrew S., et al. "Party Polarization in Congress: A Network Science Approach." 2009.
