# Network Centrality Measures

**Category:** Network Analysis
**Prerequisites:** `20_NET_covoting_network`
**Complexity:** Medium

## What It Measures

Network centrality quantifies the importance or influence of each legislator within the co-voting network. Different centrality measures capture different types of "importance":

- **Degree centrality**: Who agrees with the most colleagues? (Popularity/consensus-building)
- **Betweenness centrality**: Who bridges between groups? (Brokerage/mediation)
- **Eigenvector centrality**: Who is connected to other well-connected legislators? (Elite status)
- **Closeness centrality**: Who is "closest" to everyone else? (Information access)

These measures identify legislators who play structurally important roles that party labels and ideology scores miss.

## Questions It Answers

- Who are the most connected legislators in the voting network?
- Who bridges the partisan divide (high betweenness)?
- Which legislators are peripheral (low centrality of all types)?
- Do party leaders have distinctive centrality profiles?
- Is there a correlation between centrality and other measures (ideal points, party unity)?

## Mathematical Foundation

### Degree Centrality

$$C_D(i) = \frac{\sum_{j \neq i} w_{ij}}{(n - 1) \cdot \max(w)}$$

For weighted networks, this is the sum of edge weights connected to node $i$, normalized by the maximum possible. High degree = agrees substantially with many colleagues.

### Betweenness Centrality

$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

Where $\sigma_{st}$ is the number of shortest paths from $s$ to $t$, and $\sigma_{st}(i)$ is the number of those paths passing through $i$.

**Political interpretation**: Legislators with high betweenness are gatekeepers between voting blocs. They sit at the "bridge" between parties or factions. Removing them would increase the "distance" between groups.

### Eigenvector Centrality

$$C_E(i) = \frac{1}{\lambda} \sum_{j} A_{ij} C_E(j)$$

A legislator is central if they are connected to other central legislators. This captures "prestige" — being well-connected to the party elite, not just to random colleagues.

### Closeness Centrality

$$C_C(i) = \frac{n - 1}{\sum_{j \neq i} d(i, j)}$$

Where $d(i, j)$ is the shortest path length. High closeness = can "reach" any other legislator quickly through the voting similarity network.

## Python Implementation

```python
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_all_centralities(G: nx.Graph) -> pd.DataFrame:
    """Compute all centrality measures for the co-voting network."""
    centralities = pd.DataFrame(index=G.nodes())

    centralities["degree"] = pd.Series(nx.degree_centrality(G))
    centralities["weighted_degree"] = pd.Series(
        dict(G.degree(weight="weight"))
    )
    centralities["betweenness"] = pd.Series(
        nx.betweenness_centrality(G, weight="weight")
    )
    centralities["eigenvector"] = pd.Series(
        nx.eigenvector_centrality_numpy(G, weight="weight")
    )

    if nx.is_connected(G):
        centralities["closeness"] = pd.Series(
            nx.closeness_centrality(G, distance="weight")
        )
    else:
        # Compute per component
        closeness = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 1:
                c = nx.closeness_centrality(subgraph, distance="weight")
                closeness.update(c)
        centralities["closeness"] = pd.Series(closeness)

    # Add metadata
    for node in G.nodes():
        centralities.loc[node, "party"] = G.nodes[node].get("party", "Unknown")
        centralities.loc[node, "chamber"] = G.nodes[node].get("chamber", "Unknown")

    return centralities


def plot_centrality_comparison(
    centralities: pd.DataFrame,
    x_metric: str = "degree",
    y_metric: str = "betweenness",
    save_path: str | None = None,
):
    """Scatter plot comparing two centrality measures."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = centralities["party"].map({"Republican": "#E81B23", "Democrat": "#0015BC"})

    ax.scatter(
        centralities[x_metric],
        centralities[y_metric],
        c=colors,
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.3,
    )

    # Label high-centrality legislators
    for metric in [x_metric, y_metric]:
        top = centralities[metric].nlargest(5).index
        for slug in top:
            name = slug.split("_")[1].title()
            ax.annotate(
                name,
                (centralities.loc[slug, x_metric], centralities.loc[slug, y_metric]),
                fontsize=7, ha="left",
            )

    ax.set_xlabel(f"{x_metric.replace('_', ' ').title()} Centrality")
    ax.set_ylabel(f"{y_metric.replace('_', ' ').title()} Centrality")
    ax.set_title(f"Centrality: {x_metric} vs. {y_metric}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)


def centrality_by_party(centralities: pd.DataFrame) -> pd.DataFrame:
    """Compare centrality measures across parties."""
    numeric_cols = centralities.select_dtypes(include=[np.number]).columns
    return centralities.groupby("party")[numeric_cols].agg(["mean", "median", "std"])


def identify_bridge_legislators(
    centralities: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find legislators with high betweenness (bridging role)."""
    return centralities.nlargest(top_n, "betweenness")[
        ["party", "chamber", "degree", "betweenness", "eigenvector"]
    ]
```

## Interpretation Guide

### Centrality Profiles by Role

| Profile | Degree | Betweenness | Eigenvector | Likely Role |
|---------|--------|-------------|-------------|-------------|
| High | High | Low | High | Party loyalist connected to party core |
| Low | High | Low | High | Bridge/moderate connecting factions |
| High | Low | High | High | Elite party leader (connected to other leaders) |
| Low | Low | Low | Low | Backbencher or chronic absentee |
| Moderate | High | Moderate | Moderate | Cross-party dealmaker |

### Party-Level Patterns

- **Higher average degree for majority party**: Expected — they vote together more often and have more members.
- **Higher betweenness for minority party members**: Possible — minority members who occasionally vote with the majority have bridging positions.
- **Eigenvector centrality concentrated in majority party**: The "elite" cluster is likely within the majority party.

## Kansas-Specific Considerations

- **Republican legislators will dominate degree centrality** simply because there are more of them and they form the majority coalition. Normalize by within-party rank for fairer comparison.
- **The most interesting betweenness scores belong to moderate Republicans** who bridge between the conservative Republican core and the Democratic minority.
- **Senate vs. House centrality distributions will differ** — the Senate's smaller size means centrality is more evenly distributed.
- **Combine with ideal points**: Plot centrality vs. ideal point to see if moderates (center of ideological spectrum) are also bridges (high betweenness). This is the political science hypothesis of "the median voter as broker."

## Feasibility Assessment

- **Data size**: 170 nodes = all centrality measures compute in milliseconds
- **Compute time**: Sub-second
- **Libraries**: `networkx`
- **Difficulty**: Low (computation), Medium (interpretation)

## Key References

- Freeman, Linton C. "Centrality in Social Networks: Conceptual Clarification." *Social Networks* 1(3), 1979.
- Waugh, Andrew S., et al. "Party Polarization in Congress: A Network Science Approach." arXiv:0907.3509, 2009.
- Fowler, James H. "Connecting the Congress: A Study of Cosponsorship Networks." *Political Analysis* 14(4), 2006.
