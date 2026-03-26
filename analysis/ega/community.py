"""Community detection for EGA networks.

Wraps igraph's Walktrap and Leiden algorithms for identifying
communities (dimensions) in the GLASSO partial correlation network.

Includes Golino's unidimensional check: after community detection
finds K >= 2, test whether Louvain on the zero-order correlation
matrix finds K=1. If so, the data may be unidimensional despite
apparent multidimensionality in the partial correlation network.

Fragmentation guard: when Walktrap/Leiden produces K > p/4 (e.g.,
196 communities from 226 bills), the network is too fragmented for
meaningful community detection — fall back to connected-component
analysis on the largest component, or report unidimensional if even
that is fragmented. See ADR-0126.

References:
    Pons, P., & Latapy, M. (2006). Computing communities in large
    networks using random walks. JGAA, 10(2), 191-218.

    Golino et al. (2020). Investigating the performance of EGA and
    traditional techniques. Psychological Methods, 25(3), 292-320.
"""

from dataclasses import dataclass

import igraph as ig
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CommunityResult:
    """Result of community detection on an EGA network.

    Attributes:
        assignments: Length-p array of community labels (0-indexed).
        n_communities: Number of detected communities (K).
        modularity: Modularity of the partition.
        unidimensional: True if the unidimensional check overrode
            the community detection result.
        fragmented: True if the initial detection was too fragmented
            and the result was corrected by the fragmentation guard.
        algorithm: Algorithm used ("walktrap" or "leiden").
    """

    assignments: NDArray[np.int64]
    n_communities: int
    modularity: float
    unidimensional: bool
    fragmented: bool
    algorithm: str


def _partial_corr_to_graph(partial_corr: NDArray[np.float64]) -> ig.Graph:
    """Build an igraph Graph from a partial correlation matrix.

    Only non-zero edges (|partial_corr| > 1e-10) are included.
    Edge weights are absolute partial correlations.
    """
    p = partial_corr.shape[0]
    g = ig.Graph(n=p)

    edges = []
    weights = []
    for i in range(p):
        for j in range(i + 1, p):
            w = abs(partial_corr[i, j])
            if w > 1e-10:
                edges.append((i, j))
                weights.append(w)

    g.add_edges(edges)
    g.es["weight"] = weights
    return g


def _walktrap(graph: ig.Graph, steps: int = 4) -> ig.VertexClustering:
    """Run Walktrap community detection (Golino's default)."""
    dendro = graph.community_walktrap(weights="weight", steps=steps)
    return dendro.as_clustering()


def _leiden(graph: ig.Graph) -> ig.VertexClustering:
    """Run Leiden community detection (modularity optimization)."""
    import leidenalg

    return leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        seed=42,
    )


def _unidimensional_check(
    corr_matrix: NDArray[np.float64],
) -> bool:
    """Golino's unidimensional check: Louvain on the zero-order correlation matrix.

    If Louvain finds K=1 on the full (non-regularized) correlation matrix,
    the data is likely unidimensional even if the GLASSO network has K >= 2.

    Returns True if the data appears unidimensional.
    """
    p = corr_matrix.shape[0]
    g = ig.Graph(n=p)

    edges = []
    weights = []
    for i in range(p):
        for j in range(i + 1, p):
            w = abs(corr_matrix[i, j])
            if w > 0.01:  # Small threshold to exclude near-zero correlations
                edges.append((i, j))
                weights.append(w)

    if not edges:
        return True  # No edges → unidimensional

    g.add_edges(edges)
    g.es["weight"] = weights

    # Louvain community detection
    clustering = g.community_multilevel(weights="weight")
    return len(clustering) == 1


def _fragmentation_guard(
    graph: ig.Graph,
    assignments: NDArray[np.int64],
    k: int,
    algorithm: str,
) -> tuple[NDArray[np.int64], int, float, bool]:
    """Detect and correct network fragmentation.

    When Walktrap/Leiden produces K > p/4, the network is too sparse for
    meaningful community detection (most bills are in singleton communities).
    Fall back to community detection on only the largest connected component,
    assigning all other nodes to a catch-all community.

    Returns (assignments, k, modularity, was_fragmented).
    """
    p = graph.vcount()
    threshold = max(p // 4, 10)

    if k <= threshold:
        return assignments, k, 0.0, False

    # Network is fragmented — retry on the largest connected component
    components = graph.connected_components()
    if len(components) <= 1:
        # Single component but still fragmented — algorithm issue, not topology
        return assignments, k, 0.0, True

    # Find largest component
    largest_idx = max(range(len(components)), key=lambda i: len(components[i]))
    largest_members = components[largest_idx]

    if len(largest_members) < 5:
        # Even the largest component is tiny — network is too sparse
        return np.zeros(p, dtype=np.int64), 1, 0.0, True

    # Build subgraph of largest component and re-run community detection
    subgraph = graph.induced_subgraph(largest_members)
    if algorithm == "walktrap":
        sub_clustering = _walktrap(subgraph)
    else:
        sub_clustering = _leiden(subgraph)

    sub_k = len(set(sub_clustering.membership))
    sub_mod = float(sub_clustering.modularity)

    # Map back to full node set: non-largest-component nodes get community label sub_k
    # (a catch-all "unassigned" community)
    new_assignments = np.full(p, sub_k, dtype=np.int64)
    for local_idx, global_idx in enumerate(largest_members):
        new_assignments[global_idx] = sub_clustering.membership[local_idx]

    # If the largest component is also fragmented, fall back to unidimensional
    sub_threshold = max(len(largest_members) // 4, 10)
    if sub_k > sub_threshold:
        return np.zeros(p, dtype=np.int64), 1, 0.0, True

    # +1 for the catch-all community of isolated nodes
    return new_assignments, sub_k + 1, sub_mod, True


def detect_communities(
    partial_corr: NDArray[np.float64],
    corr_matrix: NDArray[np.float64] | None = None,
    algorithm: str = "walktrap",
    check_unidimensional: bool = True,
) -> CommunityResult:
    """Detect communities (dimensions) in a GLASSO partial correlation network.

    Parameters:
        partial_corr: p × p sparse partial correlation matrix from GLASSO.
        corr_matrix: p × p zero-order correlation matrix for unidimensional check.
            Required if check_unidimensional=True.
        algorithm: "walktrap" (default, Golino's recommendation) or "leiden".
        check_unidimensional: If True and K >= 2, run the unidimensional check.

    Returns:
        CommunityResult with community assignments and metadata.
    """
    graph = _partial_corr_to_graph(partial_corr)
    p = partial_corr.shape[0]

    # Handle disconnected graph (isolated nodes with no edges)
    if graph.ecount() == 0:
        return CommunityResult(
            assignments=np.zeros(p, dtype=np.int64),
            n_communities=1,
            modularity=0.0,
            unidimensional=True,
            fragmented=False,
            algorithm=algorithm,
        )

    if algorithm == "walktrap":
        clustering = _walktrap(graph)
    elif algorithm == "leiden":
        clustering = _leiden(graph)
    else:
        msg = f"Unknown algorithm: {algorithm!r}. Use 'walktrap' or 'leiden'."
        raise ValueError(msg)

    assignments = np.array(clustering.membership, dtype=np.int64)
    k = len(set(clustering.membership))
    modularity = float(clustering.modularity)

    # Fragmentation guard: catch pathological K (e.g., K=196 from 226 items)
    assignments, k, modularity, fragmented = _fragmentation_guard(
        graph, assignments, k, algorithm
    )

    # Unidimensional check
    uni = False
    if check_unidimensional and k >= 2 and corr_matrix is not None:
        uni = _unidimensional_check(corr_matrix)
        if uni:
            assignments = np.zeros(p, dtype=np.int64)
            k = 1

    return CommunityResult(
        assignments=assignments,
        n_communities=k,
        modularity=modularity,
        unidimensional=uni,
        fragmented=fragmented,
        algorithm=algorithm,
    )
