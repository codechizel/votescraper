# Bipartite Bill-Legislator Network

**Category:** Network Analysis
**Prerequisites:** `01_DATA_vote_matrix_construction`
**Complexity:** Medium
**Related:** `20_NET_covoting_network`

## What It Measures

A bipartite network has two types of nodes: legislators and bills (roll calls). Edges connect legislators to the bills they voted Yea on. This two-mode representation preserves the full structure of legislative voting — rather than collapsing into a legislator-legislator agreement matrix (which loses information about which specific bills drove the agreement), the bipartite network shows the complete picture.

Bipartite projection to a one-mode network yields either a legislator-legislator co-voting network (weighted by shared Yea votes) or a bill-bill co-support network (weighted by shared supporters). The latter is particularly useful for identifying clusters of related legislation.

## Questions It Answers

- Which bills are associated with which legislators (and vice versa)?
- Are there clusters of bills that attract similar coalitions?
- Which bills are "bridging" — supported by diverse coalitions?
- What is the bill-bill similarity network? (Do fiscal bills, social bills, etc. form clusters?)
- How does the bipartite structure differ from the simple co-voting network?

## Mathematical Foundation

### Bipartite Incidence Matrix

The vote matrix $\mathbf{X}$ (legislators x roll calls, binary) IS the biadjacency matrix of the bipartite graph:

$$B_{ij} = \begin{cases} 1 & \text{if legislator } i \text{ voted Yea on roll call } j \\ 0 & \text{otherwise} \end{cases}$$

### One-Mode Projections

**Legislator projection**: $\mathbf{L} = \mathbf{B} \mathbf{B}^T$ — entry $L_{ik}$ counts the number of bills both legislators $i$ and $k$ voted Yea on.

**Bill projection**: $\mathbf{R} = \mathbf{B}^T \mathbf{B}$ — entry $R_{jl}$ counts the number of legislators who voted Yea on both bills $j$ and $l$.

### Information Loss in Projection

One-mode projections lose information. Two legislator pairs can have the same co-vote count but for completely different bills. The bipartite structure preserves which specific bills create each connection.

## Python Implementation

```python
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

def build_bipartite_network(
    vote_matrix: pd.DataFrame,
    legislator_meta: pd.DataFrame,
    rollcalls: pd.DataFrame,
) -> nx.Graph:
    """Build bipartite legislator-bill network from vote matrix."""
    B = nx.Graph()

    # Add legislator nodes
    for slug in vote_matrix.index:
        if slug in legislator_meta.index:
            meta = legislator_meta.loc[slug]
            B.add_node(slug, bipartite=0, node_type="legislator",
                      party=meta["party"], chamber=meta["chamber"])

    # Add bill nodes
    rc_meta = rollcalls.set_index("vote_id")
    for vote_id in vote_matrix.columns:
        attrs = {"bipartite": 1, "node_type": "bill"}
        if vote_id in rc_meta.index:
            rc = rc_meta.loc[vote_id]
            attrs["bill_number"] = rc.get("bill_number", "")
            attrs["vote_type"] = rc.get("vote_type", "")
            attrs["chamber"] = rc.get("chamber", "")
        B.add_node(vote_id, **attrs)

    # Add edges (Yea votes)
    for slug in vote_matrix.index:
        for vote_id in vote_matrix.columns:
            if vote_matrix.loc[slug, vote_id] == 1:
                B.add_edge(slug, vote_id)

    return B


def project_to_legislators(B: nx.Graph) -> nx.Graph:
    """Project bipartite network to legislator-legislator co-voting network."""
    legislator_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}
    G = bipartite.weighted_projected_graph(B, legislator_nodes)
    return G


def project_to_bills(B: nx.Graph) -> nx.Graph:
    """Project bipartite network to bill-bill co-support network."""
    bill_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}
    G = bipartite.weighted_projected_graph(B, bill_nodes)
    return G


def analyze_bill_clusters(
    bill_network: nx.Graph,
    rollcalls: pd.DataFrame,
    n_clusters: int = 5,
) -> pd.DataFrame:
    """Find clusters of bills with similar support coalitions."""
    from community import community_louvain  # python-louvain package

    # Detect communities
    partition = community_louvain.best_partition(bill_network, weight="weight")

    # Analyze each community
    rc_meta = rollcalls.set_index("vote_id")
    results = []
    for vote_id, community_id in partition.items():
        attrs = {"vote_id": vote_id, "community": community_id}
        if vote_id in rc_meta.index:
            rc = rc_meta.loc[vote_id]
            attrs["bill_number"] = rc.get("bill_number", "")
            attrs["vote_type"] = rc.get("vote_type", "")
            attrs["bill_title"] = rc.get("bill_title", "")[:80]
        results.append(attrs)

    df = pd.DataFrame(results)

    # Community summary
    for comm_id in sorted(df["community"].unique()):
        comm_bills = df[df["community"] == comm_id]
        print(f"\nCommunity {comm_id} ({len(comm_bills)} bills):")
        if "vote_type" in comm_bills.columns:
            print(f"  Vote types: {comm_bills['vote_type'].value_counts().head(3).to_dict()}")
        print(f"  Sample bills: {comm_bills['bill_number'].head(5).tolist()}")

    return df
```

## Interpretation Guide

- **Legislator projection** closely resembles the co-voting network but weighted by absolute co-vote count rather than rate. Normalize by shared opportunities for a fairer comparison.
- **Bill projection clusters** reveal latent policy dimensions. Bills in the same cluster attracted similar voting coalitions, even if their subject matter differs.
- **High-degree bills** (in the bipartite graph) are bills with many Yea votes (near-unanimous). Low-degree bills are contested.
- **High-degree legislators** voted Yea most often. Consider filtering to contested bills for more informative analysis.

## Kansas-Specific Considerations

- **Filter to contested bills** before building the bipartite network. Near-unanimous bills (which dominate) create a nearly fully-connected bipartite graph that masks the interesting structure.
- **Bill clusters may reveal policy dimensions** that party labels don't capture: fiscal/tax bills, education bills, criminal justice bills may each attract distinct coalitions.
- **The bill-bill network is a novel contribution.** Most analyses focus on legislator-legislator networks. The bill-bill perspective can reveal which policy areas are linked by shared coalitions.

## Feasibility Assessment

- **Data size**: ~170 + ~400 nodes with ~30K edges = small for network analysis
- **Compute time**: Seconds for construction and projection, minutes for community detection
- **Libraries**: `networkx` (bipartite module), `python-louvain`
- **Difficulty**: Medium

## Key References

- Briatte, François. "Network Patterns of Legislative Collaboration in Twenty Parliaments." *Network Science* 4(2), 2016.
- Neal, Zachary P. "A Statistical Model of Bipartite Networks: Application to Cosponsorship in the United States Senate." *Political Analysis* 22(3), 2014.
