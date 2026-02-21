"""
Kansas Legislature — Network Analysis (Phase 6)

Transforms Cohen's Kappa agreement matrices into weighted graphs, computes
centrality measures to identify structurally important legislators, and runs
Louvain community detection at multiple resolutions to test whether network-based
grouping finds finer structure than the k=2 party split from clustering.

Usage:
  uv run python analysis/network.py [--session 2025-26] [--kappa-threshold 0.40]
      [--skip-cross-chamber] [--skip-high-disc]

Outputs (in results/<session>/network/<date>/):
  - data/:   Parquet files (centrality, communities, threshold sweep)
  - plots/:  PNG visualizations (layouts, centrality, communities)
  - filtering_manifest.json, run_info.json, run_log.txt
  - network_report.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import community as community_louvain  # python-louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.network_report import build_network_report
except ModuleNotFoundError:
    from network_report import build_network_report  # type: ignore[no-redef]


# ── Constants ────────────────────────────────────────────────────────────────

KAPPA_THRESHOLD_DEFAULT = 0.40
KAPPA_THRESHOLD_SENSITIVITY = [0.30, 0.40, 0.50, 0.60]
LOUVAIN_RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
HIGH_DISC_THRESHOLD = 1.5
TOP_BRIDGE_N = 15
TOP_LABEL_N = 10
RANDOM_SEED = 42
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}

# Plain-English labels for centrality measures and IRT columns (nontechnical audience)
PLAIN_LABELS: dict[str, str] = {
    "betweenness": "Connections to Other Blocs (Betweenness)",
    "eigenvector": "Connections to Influential Legislators (Eigenvector)",
    "xi_mean": "Ideology (Liberal \u2190 \u2192 Conservative)",
    "degree": "Share of Possible Connections (Degree)",
    "weighted_degree": "Total Connection Strength",
    "closeness": "Closeness to All Legislators",
    "pagerank": "Random-Walk Importance (PageRank)",
}

PLAIN_TITLES: dict[tuple[str, str], str] = {
    ("betweenness", "eigenvector"): "Who Holds the Most Influence?",
    ("xi_mean", "betweenness"): "Ideology vs Network Influence",
}

NETWORK_PRIMER = """\
# Network Analysis

## Purpose

Transforms pairwise Cohen's Kappa agreement matrices into weighted legislator
graphs and applies graph-theoretic methods to identify structurally important
legislators and voting communities. This supplements IRT ideal points (which
measure ideological position) and clustering (which found k=2 party split)
with relational structure that captures who votes with whom.

## Method

### Network Construction
- **Nodes:** Legislators, with attributes from IRT (ideal point, uncertainty),
  clustering (cluster assignment, party loyalty), and metadata (party, chamber).
- **Edges:** Kappa > threshold (default 0.40, "substantial" on Landis-Koch scale).
  Weight = Kappa value. NaN Kappa = no edge (unknown = no observed connection).

### Centrality Measures
Five centrality measures identify structurally important legislators:
- **Degree centrality:** Fraction of possible connections realized
- **Weighted degree:** Sum of edge weights (total agreement strength)
- **Betweenness centrality:** Frequency on shortest paths between other pairs
- **Eigenvector centrality:** Importance via connections to other important nodes
- **Closeness centrality:** Inverse mean distance to all reachable nodes
- **PageRank:** Random-walk importance (robust to disconnected graphs)

### Community Detection
Louvain algorithm at 8 resolution parameters (0.5 to 3.0). Lower resolutions
produce fewer, larger communities (party-level); higher resolutions split into
potential subcaucuses. Communities compared to party labels and clustering
assignments via NMI and ARI.

### Subnetwork Analyses
- **Within-party:** Separate networks per party caucus
- **High-discrimination bills:** Network built from |beta| > 1.5 bills only
- **Veto overrides:** Network built from override votes only
- **Cross-chamber:** Union of per-chamber networks with equated ideal points

## Inputs

| File | Source | Contents |
|------|--------|----------|
| `agreement_kappa_{chamber}.parquet` | EDA | Pairwise Kappa matrices |
| `vote_matrix_{chamber}_filtered.parquet` | EDA | Filtered binary vote matrices |
| `ideal_points_{chamber}.parquet` | IRT | Per-legislator ideal points |
| `ideal_points_joint_equated.parquet` | IRT | Cross-chamber equated ideal points |
| `bill_params_{chamber}.parquet` | IRT | Bill discrimination parameters |
| `kmeans_assignments_{chamber}.parquet` | Clustering | Cluster assignments |
| `party_loyalty_{chamber}.parquet` | Clustering | Party loyalty scores |

## Outputs

| File | Contents |
|------|----------|
| `centrality_{chamber}.parquet` | All centrality measures per legislator |
| `communities_{chamber}.parquet` | Community assignments at best resolution |
| `community_composition_{chamber}.parquet` | Community demographics |
| `community_resolution_{chamber}.parquet` | Modularity at each resolution |
| `bridge_legislators_{chamber}.parquet` | Top betweenness bridge legislators |
| `threshold_sweep_{chamber}.parquet` | Network stats at each Kappa threshold |
| `within_party_{party}_{chamber}.parquet` | Within-party community results |
| `network_summary.parquet` | Summary statistics per chamber |
| `network_report.html` | Self-contained HTML report |

## Interpretation Guide

- **High betweenness + moderate ideology** = genuine bridge legislator
- **High betweenness + extreme ideology** = unusual; investigate (may be unique voter)
- **Community ≠ party** = network finds structure clustering missed
- **Community = party at all resolutions** = party is the dominant structure
- **Within-party modularity > 0.3** = meaningful subcaucus structure

## Caveats

- Network topology is threshold-dependent; always check sensitivity sweep
- NaN Kappa entries (60 House, 8 Senate) reduce graph completeness
- Louvain is stochastic; seed is fixed but results may vary across versions
- ~170 nodes is small for network analysis; interpret cautiously
"""


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Network Analysis")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument(
        "--clustering-dir", default=None, help="Override clustering results directory"
    )
    parser.add_argument(
        "--kappa-threshold",
        type=float,
        default=KAPPA_THRESHOLD_DEFAULT,
        help=f"Kappa threshold for edge creation (default: {KAPPA_THRESHOLD_DEFAULT})",
    )
    parser.add_argument(
        "--skip-cross-chamber",
        action="store_true",
        help="Skip cross-chamber network analysis",
    )
    parser.add_argument(
        "--skip-high-disc",
        action="store_true",
        help="Skip high-discrimination subnetwork",
    )
    return parser.parse_args()


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _build_ip_lookup(ideal_points: pl.DataFrame) -> dict[str, dict]:
    """Build a slug → row dict from an ideal points DataFrame."""
    return {row["legislator_slug"]: row for row in ideal_points.iter_rows(named=True)}


def _build_network_from_vote_subset(
    vote_matrix: pl.DataFrame,
    vote_ids: set[str],
    ideal_points: pl.DataFrame,
    kappa_threshold: float = KAPPA_THRESHOLD_DEFAULT,
    min_shared: int = 5,
) -> nx.Graph | None:
    """Build a network from pairwise Kappa computed on a subset of votes.

    Shared by high-discrimination and veto override subnetwork builders.
    Returns None if fewer than 5 matching votes.
    """
    vm_cols = vote_matrix.columns
    slug_col = vm_cols[0]
    bill_cols = [c for c in vm_cols[1:] if c in vote_ids]

    if len(bill_cols) < 5:
        return None

    vm_filtered = vote_matrix.select([slug_col] + bill_cols)
    slugs = vm_filtered[slug_col].to_list()
    mat = vm_filtered.drop(slug_col).to_numpy().astype(float)
    n = len(slugs)

    # Compute pairwise Kappa
    kappa_mat = np.full((n, n), np.nan)
    for i in range(n):
        kappa_mat[i, i] = 1.0
        for j in range(i + 1, n):
            vi = mat[i, :]
            vj = mat[j, :]
            mask = ~np.isnan(vi) & ~np.isnan(vj)
            if mask.sum() < min_shared:
                continue
            try:
                k = cohen_kappa_score(vi[mask], vj[mask])
                kappa_mat[i, j] = k
                kappa_mat[j, i] = k
            except ValueError, ZeroDivisionError:
                pass

    # Build graph
    ip_dict = _build_ip_lookup(ideal_points)
    G = nx.Graph()
    for slug in slugs:
        ip = ip_dict.get(slug, {})
        G.add_node(
            slug,
            party=ip.get("party", "Unknown"),
            xi_mean=ip.get("xi_mean", 0.0),
            full_name=ip.get("full_name", slug),
        )

    for i in range(n):
        for j in range(i + 1, n):
            k = kappa_mat[i, j]
            if np.isnan(k) or k <= kappa_threshold:
                continue
            G.add_edge(slugs[i], slugs[j], weight=float(k), distance=1.0 / float(k))

    return G


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_kappa_matrices(eda_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load pairwise Kappa agreement matrices from EDA."""
    house = pl.read_parquet(eda_dir / "data" / "agreement_kappa_house.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "agreement_kappa_senate.parquet")
    return house, senate


def load_ideal_points(irt_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load IRT ideal points for both chambers."""
    house = pl.read_parquet(irt_dir / "data" / "ideal_points_house.parquet")
    senate = pl.read_parquet(irt_dir / "data" / "ideal_points_senate.parquet")
    return house, senate


def load_equated_ideal_points(irt_dir: Path) -> pl.DataFrame:
    """Load cross-chamber equated ideal points."""
    return pl.read_parquet(irt_dir / "data" / "ideal_points_joint_equated.parquet")


def load_bill_params(irt_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load bill discrimination parameters for high-disc subnetwork."""
    house = pl.read_parquet(irt_dir / "data" / "bill_params_house.parquet")
    senate = pl.read_parquet(irt_dir / "data" / "bill_params_senate.parquet")
    return house, senate


def load_cluster_assignments(
    clustering_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load k-means cluster assignments from clustering phase."""
    house = pl.read_parquet(clustering_dir / "data" / "kmeans_assignments_house.parquet")
    senate = pl.read_parquet(clustering_dir / "data" / "kmeans_assignments_senate.parquet")
    return house, senate


def load_party_loyalty(clustering_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load party loyalty scores from clustering phase."""
    house = pl.read_parquet(clustering_dir / "data" / "party_loyalty_house.parquet")
    senate = pl.read_parquet(clustering_dir / "data" / "party_loyalty_senate.parquet")
    return house, senate


def load_vote_matrices(eda_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load filtered binary vote matrices from EDA."""
    house = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
    return house, senate


def load_metadata(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load rollcall and legislator CSVs for metadata enrichment."""
    prefix = data_dir.name
    rollcalls = pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    return rollcalls, legislators


# ── Phase 2: Network Construction ───────────────────────────────────────────


def _kappa_matrix_to_numpy(kappa_df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Convert a polars Kappa DataFrame to a numpy array and list of slugs.

    The first column is 'legislator_slug', remaining columns are legislator slugs.
    """
    slugs = kappa_df["legislator_slug"].to_list()
    mat = kappa_df.drop("legislator_slug").to_numpy().astype(float)
    return mat, slugs


def build_kappa_network(
    kappa_df: pl.DataFrame,
    ideal_points: pl.DataFrame,
    cluster_assignments: pl.DataFrame | None = None,
    loyalty: pl.DataFrame | None = None,
    threshold: float = KAPPA_THRESHOLD_DEFAULT,
) -> nx.Graph:
    """Build a weighted undirected graph from a Kappa agreement matrix.

    Nodes: legislators with attributes (party, chamber, xi_mean, xi_sd, cluster,
    loyalty_rate, full_name).
    Edges: Kappa > threshold; weight = Kappa value; NaN = no edge.
    Edge attr 'distance' = 1/weight for path-based centrality.
    """
    mat, slugs = _kappa_matrix_to_numpy(kappa_df)
    n = len(slugs)

    # Build lookup dicts from upstream data
    ip_dict = _build_ip_lookup(ideal_points)

    cluster_dict: dict[str, int] = {}
    if cluster_assignments is not None:
        # Find the cluster column (cluster_k2, cluster_k3, etc.)
        cluster_cols = [c for c in cluster_assignments.columns if c.startswith("cluster_k")]
        cluster_col = cluster_cols[0] if cluster_cols else None
        if cluster_col:
            for row in cluster_assignments.iter_rows(named=True):
                cluster_dict[row["legislator_slug"]] = row[cluster_col]

    loyalty_dict: dict[str, float] = {}
    if loyalty is not None:
        for row in loyalty.iter_rows(named=True):
            loyalty_dict[row["legislator_slug"]] = row["loyalty_rate"]

    G = nx.Graph()

    # Add nodes with attributes
    for slug in slugs:
        ip = ip_dict.get(slug, {})
        attrs = {
            "party": ip.get("party", "Unknown"),
            "chamber": ip.get("chamber", "Unknown"),
            "xi_mean": ip.get("xi_mean", 0.0),
            "xi_sd": ip.get("xi_sd", 1.0),
            "full_name": ip.get("full_name", slug),
            "cluster": cluster_dict.get(slug, -1),
            "loyalty_rate": loyalty_dict.get(slug, None),
        }
        G.add_node(slug, **attrs)

    # Add edges above threshold
    for i in range(n):
        for j in range(i + 1, n):
            kappa = mat[i, j]
            if np.isnan(kappa) or kappa <= threshold:
                continue
            G.add_edge(
                slugs[i],
                slugs[j],
                weight=float(kappa),
                distance=1.0 / float(kappa),
            )

    return G


def compute_network_summary(G: nx.Graph) -> dict:
    """Compute summary statistics for a network graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)

    components = list(nx.connected_components(G))
    n_components = len(components)

    # Clustering coefficient and transitivity
    avg_clustering = nx.average_clustering(G, weight="weight") if n_edges > 0 else 0.0
    transitivity = nx.transitivity(G) if n_edges > 0 else 0.0

    # Party assortativity
    try:
        assortativity_party = nx.attribute_assortativity_coefficient(G, "party")
    except ValueError, ZeroDivisionError:
        assortativity_party = None

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(density, 4),
        "avg_clustering": round(avg_clustering, 4),
        "transitivity": round(transitivity, 4),
        "n_components": n_components,
        "assortativity_party": (
            round(assortativity_party, 4) if assortativity_party is not None else None
        ),
    }


# ── Phase 3: Centrality ─────────────────────────────────────────────────────


def compute_centralities(G: nx.Graph) -> pl.DataFrame:
    """Compute centrality measures for all nodes.

    Returns DataFrame with: legislator_slug, full_name, party, xi_mean,
    degree, weighted_degree, betweenness, eigenvector, closeness, pagerank.
    """
    if G.number_of_nodes() == 0:
        return pl.DataFrame()

    nodes = list(G.nodes())

    # Degree centrality (unweighted fraction)
    degree = nx.degree_centrality(G)

    # Weighted degree (sum of edge weights)
    weighted_degree = {n: sum(d["weight"] for _, _, d in G.edges(n, data=True)) for n in nodes}

    # Betweenness centrality (distance = 1/kappa)
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    # Eigenvector centrality (weighted; per component for disconnected graphs)
    eigenvector: dict = {}
    try:
        if nx.is_connected(G):
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        else:
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                if len(component) > 2:
                    ec = nx.eigenvector_centrality_numpy(subgraph, weight="weight")
                    eigenvector.update(ec)
                else:
                    for n in component:
                        eigenvector[n] = 0.0
    except nx.NetworkXError, nx.AmbiguousSolution, np.linalg.LinAlgError:
        eigenvector = {n: 0.0 for n in nodes}

    # Closeness centrality (distance = 1/kappa; per component for disconnected graphs)
    closeness = {}
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(component) > 1:
            cc = nx.closeness_centrality(subgraph, distance="distance")
            closeness.update(cc)
        else:
            for n in component:
                closeness[n] = 0.0

    # PageRank (weighted)
    pagerank = nx.pagerank(G, weight="weight")

    # Build DataFrame
    rows = []
    for n in nodes:
        attrs = G.nodes[n]
        rows.append(
            {
                "legislator_slug": n,
                "full_name": attrs.get("full_name", n),
                "party": attrs.get("party", "Unknown"),
                "xi_mean": attrs.get("xi_mean", 0.0),
                "degree": degree[n],
                "weighted_degree": weighted_degree[n],
                "betweenness": betweenness[n],
                "eigenvector": eigenvector.get(n, 0.0),
                "closeness": closeness.get(n, 0.0),
                "pagerank": pagerank.get(n, 0.0),
            }
        )

    return pl.DataFrame(rows)


def identify_bridge_legislators(
    centralities: pl.DataFrame,
    G: nx.Graph,
    top_n: int = TOP_BRIDGE_N,
) -> pl.DataFrame:
    """Identify top betweenness legislators with cross-party edge count."""
    top = centralities.sort("betweenness", descending=True).head(top_n)

    cross_party_counts = []
    for slug in top["legislator_slug"].to_list():
        node_party = G.nodes[slug].get("party", "Unknown")
        cross = sum(
            1
            for neighbor in G.neighbors(slug)
            if G.nodes[neighbor].get("party", "Unknown") != node_party
        )
        total = G.degree(slug)
        cross_party_counts.append(
            {"legislator_slug": slug, "cross_party_edges": cross, "total_edges": total}
        )

    cross_df = pl.DataFrame(cross_party_counts)
    return top.join(cross_df, on="legislator_slug", how="left")


def compute_party_centrality_summary(centralities: pl.DataFrame) -> pl.DataFrame:
    """Compute mean/median/std per party per centrality measure."""
    metrics = ["degree", "weighted_degree", "betweenness", "eigenvector", "closeness", "pagerank"]
    rows = []
    for party in centralities["party"].unique().to_list():
        pf = centralities.filter(pl.col("party") == party)
        row: dict = {"party": party, "n": pf.height}
        for m in metrics:
            vals = pf[m]
            row[f"{m}_mean"] = float(vals.mean()) if vals.len() > 0 else 0.0
            row[f"{m}_median"] = float(vals.median()) if vals.len() > 0 else 0.0
            row[f"{m}_std"] = float(vals.std()) if vals.len() > 1 else 0.0
        rows.append(row)

    return pl.DataFrame(rows)


# ── Phase 4: Community Detection ─────────────────────────────────────────────


def detect_communities_multi_resolution(
    G: nx.Graph,
    resolutions: list[float] | None = None,
) -> tuple[dict[float, dict[str, int]], pl.DataFrame]:
    """Run Louvain community detection at multiple resolution parameters.

    Returns:
        partitions_dict: {resolution: {node: community_id}}
        resolution_df: DataFrame with (resolution, n_communities, modularity)
    """
    if resolutions is None:
        resolutions = LOUVAIN_RESOLUTIONS

    partitions_dict: dict[float, dict[str, int]] = {}
    resolution_rows = []

    for res in resolutions:
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=res, random_state=RANDOM_SEED
        )
        partitions_dict[res] = partition

        n_communities = len(set(partition.values()))
        modularity = community_louvain.modularity(partition, G, weight="weight")

        resolution_rows.append(
            {
                "resolution": res,
                "n_communities": n_communities,
                "modularity": round(modularity, 4),
            }
        )
        print(f"    Resolution {res:.2f}: {n_communities} communities, modularity={modularity:.4f}")

    resolution_df = pl.DataFrame(resolution_rows)
    return partitions_dict, resolution_df


def analyze_community_composition(
    partition: dict[str, int],
    G: nx.Graph,
    chamber: str,
) -> pl.DataFrame:
    """Analyze party composition and IRT summary of each community."""
    communities = set(partition.values())
    rows = []
    for comm_id in sorted(communities):
        members = [n for n, c in partition.items() if c == comm_id]
        n_total = len(members)

        parties: dict[str, int] = {}
        xi_values = []
        loyalty_values = []
        for m in members:
            attrs = G.nodes[m]
            party = attrs.get("party", "Unknown")
            parties[party] = parties.get(party, 0) + 1
            xi_values.append(attrs.get("xi_mean", 0.0))
            loy = attrs.get("loyalty_rate")
            if loy is not None:
                loyalty_values.append(loy)

        row: dict = {
            "chamber": chamber,
            "community": comm_id,
            "n_legislators": n_total,
            "n_republican": parties.get("Republican", 0),
            "n_democrat": parties.get("Democrat", 0),
            "pct_republican": round(parties.get("Republican", 0) / n_total * 100, 1),
            "mean_xi": round(float(np.mean(xi_values)), 3) if xi_values else None,
            "std_xi": round(float(np.std(xi_values)), 3) if len(xi_values) > 1 else None,
        }
        if loyalty_values:
            row["mean_loyalty"] = round(float(np.mean(loyalty_values)), 3)
        else:
            row["mean_loyalty"] = None
        rows.append(row)

    return pl.DataFrame(rows)


def compare_communities_to_party(
    partition: dict[str, int],
    G: nx.Graph,
) -> dict:
    """Compare community partition to party labels via NMI and ARI."""
    nodes = list(partition.keys())
    comm_labels = [partition[n] for n in nodes]
    party_labels = [G.nodes[n].get("party", "Unknown") for n in nodes]

    nmi = normalized_mutual_info_score(party_labels, comm_labels)
    ari = adjusted_rand_score(party_labels, comm_labels)

    # Find misclassified (community majority party != legislator party)
    comm_majority_party: dict[int, str] = {}
    for comm_id in set(comm_labels):
        members = [n for n, c in partition.items() if c == comm_id]
        party_counts: dict[str, int] = {}
        for m in members:
            p = G.nodes[m].get("party", "Unknown")
            party_counts[p] = party_counts.get(p, 0) + 1
        comm_majority_party[comm_id] = max(party_counts, key=party_counts.get)  # type: ignore[arg-type]

    misclassified = []
    for n in nodes:
        majority = comm_majority_party[partition[n]]
        actual = G.nodes[n].get("party", "Unknown")
        if majority != actual:
            misclassified.append(
                {
                    "legislator_slug": n,
                    "full_name": G.nodes[n].get("full_name", n),
                    "party": actual,
                    "community": partition[n],
                    "community_majority": majority,
                }
            )

    return {
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "misclassified": misclassified,
    }


def compare_communities_to_clusters(
    partition: dict[str, int],
    cluster_assignments: pl.DataFrame,
    G: nx.Graph,
) -> dict:
    """Compare community partition to clustering assignments via NMI and ARI."""
    # Find the cluster column
    cluster_cols = [c for c in cluster_assignments.columns if c.startswith("cluster_k")]
    if not cluster_cols:
        return {"nmi": None, "ari": None}
    cluster_col = cluster_cols[0]

    cluster_dict = {
        row["legislator_slug"]: row[cluster_col]
        for row in cluster_assignments.iter_rows(named=True)
    }

    # Only compare nodes present in both
    common = [n for n in partition if n in cluster_dict]
    if len(common) < 5:
        return {"nmi": None, "ari": None}

    comm_labels = [partition[n] for n in common]
    clust_labels = [cluster_dict[n] for n in common]

    nmi = normalized_mutual_info_score(clust_labels, comm_labels)
    ari = adjusted_rand_score(clust_labels, comm_labels)

    return {"nmi": round(nmi, 4), "ari": round(ari, 4)}


# ── Phase 5: Within-Party Subnetworks ────────────────────────────────────────


def build_within_party_network(
    kappa_df: pl.DataFrame,
    ideal_points: pl.DataFrame,
    party: str,
    threshold: float = KAPPA_THRESHOLD_DEFAULT,
) -> nx.Graph:
    """Build a network containing only legislators of the specified party."""
    # Get party slugs from ideal points
    party_slugs = set(ideal_points.filter(pl.col("party") == party)["legislator_slug"].to_list())

    # Filter Kappa matrix
    mat, slugs = _kappa_matrix_to_numpy(kappa_df)
    party_indices = [i for i, s in enumerate(slugs) if s in party_slugs]
    party_slugs_ordered = [slugs[i] for i in party_indices]

    # Build lookup
    ip_dict = _build_ip_lookup(ideal_points)

    G = nx.Graph()
    for slug in party_slugs_ordered:
        ip = ip_dict.get(slug, {})
        G.add_node(
            slug,
            party=party,
            chamber=ip.get("chamber", "Unknown"),
            xi_mean=ip.get("xi_mean", 0.0),
            xi_sd=ip.get("xi_sd", 1.0),
            full_name=ip.get("full_name", slug),
        )

    for ii, i in enumerate(party_indices):
        for jj in range(ii + 1, len(party_indices)):
            j = party_indices[jj]
            kappa = mat[i, j]
            if np.isnan(kappa) or kappa <= threshold:
                continue
            G.add_edge(
                party_slugs_ordered[ii],
                party_slugs_ordered[jj],
                weight=float(kappa),
                distance=1.0 / float(kappa),
            )

    return G


def analyze_within_party_communities(
    G_party: nx.Graph,
    resolutions: list[float] | None = None,
    party: str = "",
    chamber: str = "",
) -> dict:
    """Run community detection on a within-party subnetwork."""
    if resolutions is None:
        resolutions = LOUVAIN_RESOLUTIONS

    if G_party.number_of_edges() == 0:
        return {"skipped": True, "reason": "No edges in party subnetwork"}

    best_mod = -1.0
    best_res = resolutions[0]
    best_partition: dict[str, int] = {}
    all_results = []

    for res in resolutions:
        partition = community_louvain.best_partition(
            G_party, weight="weight", resolution=res, random_state=RANDOM_SEED
        )
        mod = community_louvain.modularity(partition, G_party, weight="weight")
        n_comm = len(set(partition.values()))
        all_results.append(
            {"resolution": res, "n_communities": n_comm, "modularity": round(mod, 4)}
        )
        if mod > best_mod:
            best_mod = mod
            best_res = res
            best_partition = partition

    n_best = len(set(best_partition.values()))
    print(
        f"    {party} ({chamber}): best resolution={best_res:.2f}, "
        f"{n_best} communities, modularity={best_mod:.4f}"
    )

    return {
        "skipped": False,
        "party": party,
        "chamber": chamber,
        "best_resolution": best_res,
        "best_modularity": round(best_mod, 4),
        "n_communities": n_best,
        "partition": best_partition,
        "resolution_results": all_results,
    }


def check_tyson_thompson_edge_weights(
    G: nx.Graph,
    chamber: str,
) -> dict | None:
    """Check edge weights for Tyson and Thompson vs Republican median (Senate only)."""
    if chamber != "Senate":
        return None

    targets = {
        "sen_tyson_caryn_1": "Caryn Tyson",
        "sen_thompson_mike_1": "Mike Thompson",
    }

    # Get all R-R edge weights
    r_nodes = {n for n in G.nodes() if G.nodes[n].get("party") == "Republican"}
    r_edges = []
    for u, v, d in G.edges(data=True):
        if u in r_nodes and v in r_nodes:
            r_edges.append(d["weight"])

    if not r_edges:
        return None

    r_median = float(np.median(r_edges))
    r_mean = float(np.mean(r_edges))

    results = {
        "r_median_edge_weight": round(r_median, 4),
        "r_mean_edge_weight": round(r_mean, 4),
        "r_n_edges": len(r_edges),
        "legislators": [],
    }

    for slug, name in targets.items():
        if slug not in G.nodes():
            continue
        edges = [(v, d["weight"]) for _, v, d in G.edges(slug, data=True)]
        r_only = [(v, w) for v, w in edges if G.nodes[v].get("party") == "Republican"]
        if r_only:
            weights = [w for _, w in r_only]
            results["legislators"].append(
                {
                    "slug": slug,
                    "name": name,
                    "n_r_edges": len(r_only),
                    "mean_r_weight": round(float(np.mean(weights)), 4),
                    "median_r_weight": round(float(np.median(weights)), 4),
                    "min_r_weight": round(float(np.min(weights)), 4),
                    "vs_r_median": round(float(np.mean(weights)) - r_median, 4),
                }
            )

    return results


# ── Phase 6: Threshold Sensitivity ──────────────────────────────────────────


def run_threshold_sweep(
    kappa_df: pl.DataFrame,
    ideal_points: pl.DataFrame,
    thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Sweep Kappa thresholds and report network statistics at each."""
    if thresholds is None:
        thresholds = KAPPA_THRESHOLD_SENSITIVITY

    rows = []
    for t in thresholds:
        G = build_kappa_network(kappa_df, ideal_points, threshold=t)
        summary = compute_network_summary(G)

        # Compute modularity at default resolution
        modularity = 0.0
        if G.number_of_edges() > 0:
            partition = community_louvain.best_partition(
                G, weight="weight", resolution=1.0, random_state=RANDOM_SEED
            )
            modularity = community_louvain.modularity(partition, G, weight="weight")

        rows.append(
            {
                "threshold": t,
                "n_edges": summary["n_edges"],
                "density": summary["density"],
                "n_components": summary["n_components"],
                "avg_clustering": summary["avg_clustering"],
                "modularity": round(modularity, 4),
            }
        )
        print(
            f"    Threshold {t:.2f}: {summary['n_edges']} edges, "
            f"density={summary['density']:.4f}, "
            f"components={summary['n_components']}, "
            f"modularity={modularity:.4f}"
        )

    return pl.DataFrame(rows)


# ── Phase 7: High-Discrimination Subnetwork ──────────────────────────────────


def build_high_disc_network(
    vote_matrix: pl.DataFrame,
    bill_params: pl.DataFrame,
    ideal_points: pl.DataFrame,
    beta_threshold: float = HIGH_DISC_THRESHOLD,
    kappa_threshold: float = KAPPA_THRESHOLD_DEFAULT,
) -> nx.Graph | None:
    """Build a network from pairwise Kappa computed on high-discrimination bills only.

    High-discrimination: bills with |beta_mean| > beta_threshold.
    """
    beta_col = "beta_mean" if "beta_mean" in bill_params.columns else "discrimination_mean"
    if beta_col not in bill_params.columns:
        print("    Warning: No discrimination column found in bill_params")
        return None

    high_disc = bill_params.filter(pl.col(beta_col).abs() > beta_threshold)
    high_disc_ids = set(high_disc["vote_id"].to_list())

    if len(high_disc_ids) < 5:
        print(f"    Only {len(high_disc_ids)} high-disc bills — skipping subnetwork")
        return None

    print(f"    High-disc bills: {len(high_disc_ids)} (|beta| > {beta_threshold})")
    G = _build_network_from_vote_subset(
        vote_matrix, high_disc_ids, ideal_points, kappa_threshold, min_shared=5
    )
    if G is None:
        print("    Too few high-disc bills in vote matrix — skipping")
        return None

    print(f"    High-disc network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Phase 9: Cross-Chamber Network ──────────────────────────────────────────


def build_cross_chamber_network(
    equated_ideal_points: pl.DataFrame,
    G_house: nx.Graph,
    G_senate: nx.Graph,
) -> nx.Graph:
    """Build a combined cross-chamber network.

    Union of per-chamber networks with equated xi as node attribute for
    cross-chamber positioning. No cross-chamber edges (different bill sets).
    """
    G = nx.Graph()

    # Build equated lookup
    eq_dict: dict[str, dict] = {}
    for row in equated_ideal_points.iter_rows(named=True):
        eq_dict[row["legislator_slug"]] = row

    # Add House nodes and edges
    for n, attrs in G_house.nodes(data=True):
        eq = eq_dict.get(n, {})
        G.add_node(
            n,
            **attrs,
            xi_equated=eq.get("xi_mean", attrs.get("xi_mean", 0.0)),
        )
    for u, v, d in G_house.edges(data=True):
        G.add_edge(u, v, **d)

    # Add Senate nodes and edges
    for n, attrs in G_senate.nodes(data=True):
        eq = eq_dict.get(n, {})
        G.add_node(
            n,
            **attrs,
            xi_equated=eq.get("xi_mean", attrs.get("xi_mean", 0.0)),
        )
    for u, v, d in G_senate.edges(data=True):
        G.add_edge(u, v, **d)

    return G


# ── Phase 8: Veto Override Subnetwork ────────────────────────────────────────


def build_veto_override_network(
    vote_matrix: pl.DataFrame,
    rollcalls: pl.DataFrame,
    ideal_points: pl.DataFrame,
    chamber: str,
    kappa_threshold: float = KAPPA_THRESHOLD_DEFAULT,
) -> nx.Graph | None:
    """Build a network from veto override votes only.

    Returns None if fewer than 5 override votes in this chamber.
    """
    override_motions = rollcalls.filter(
        pl.col("motion").str.to_lowercase().str.contains("override|veto")
        & (pl.col("chamber") == chamber)
    )
    override_ids = set(override_motions["vote_id"].to_list())

    if len(override_ids) < 5:
        print(f"    Only {len(override_ids)} override votes in {chamber} — skipping")
        return None

    print(f"    Override votes: {len(override_ids)} in {chamber}")
    G = _build_network_from_vote_subset(
        vote_matrix, override_ids, ideal_points, kappa_threshold, min_shared=3
    )
    if G is None:
        print(f"    Too few override votes in {chamber} vote matrix — skipping")
        return None

    print(f"    Override network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Phase 10: Plots ─────────────────────────────────────────────────────────


def _compute_layout(
    G: nx.Graph,
    xi_attr: str = "xi_mean",
) -> dict[str, tuple[float, float]]:
    """Compute spring layout seeded by IRT ideal points for interpretability.

    X-axis is anchored to ideal points; Y-axis is determined by spring layout.
    """
    if G.number_of_nodes() == 0:
        return {}

    # Initialize positions: x = xi_mean (scaled), y = random
    rng = np.random.default_rng(RANDOM_SEED)
    init_pos = {}
    for n in G.nodes():
        xi = G.nodes[n].get(xi_attr, 0.0)
        init_pos[n] = (xi, rng.uniform(-1, 1))

    # Spring layout with initial positions
    pos = nx.spring_layout(
        G,
        pos=init_pos,
        weight="weight",
        seed=RANDOM_SEED,
        k=2.0 / np.sqrt(G.number_of_nodes()),
        iterations=100,
    )
    return pos


def plot_network_layout(
    G: nx.Graph,
    color_by: str,
    chamber: str,
    title: str,
    out_path: Path,
    pos: dict | None = None,
    label_top_n: int = TOP_LABEL_N,
) -> dict:
    """Plot network with nodes colored by party or IRT gradient.

    Returns the position dict for reuse.
    """
    if G.number_of_nodes() == 0:
        return {}

    if pos is None:
        pos = _compute_layout(G)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    nodes = list(G.nodes())

    if color_by == "party":
        node_colors = [PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in nodes]
        legend_elements = [
            Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
            Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
        ]
    elif color_by == "xi_mean":
        xi_vals = [G.nodes[n].get("xi_mean", 0.0) for n in nodes]
        norm = Normalize(vmin=min(xi_vals), vmax=max(xi_vals))
        cmap = plt.cm.RdBu_r  # type: ignore[attr-defined]
        node_colors = [cmap(norm(x)) for x in xi_vals]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="IRT Ideal Point (xi)", shrink=0.8)
        legend_elements = []
    else:
        node_colors = "#999999"
        legend_elements = []

    # Node sizes proportional to degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 300 * degrees[n] / max_deg for n in nodes]

    # Draw edges first (light gray)
    edge_weights = [d.get("weight", 0.5) for _, _, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.3 + 1.5 * w / max_w for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=edge_widths, edge_color="#888888")

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )

    # Label top-N by betweenness
    if label_top_n > 0:
        betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
        top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:label_top_n]  # type: ignore[arg-type]
        labels = {n: G.nodes[n].get("full_name", n).split()[-1] for n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(f"{chamber} — {title}", fontsize=14, fontweight="bold")
    ax.axis("off")

    save_fig(fig, out_path)
    return pos


def plot_centrality_scatter(
    centralities: pl.DataFrame,
    x_col: str,
    y_col: str,
    chamber: str,
    out_path: Path,
) -> None:
    """Scatter plot of two centrality measures, colored by party."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for party, color in PARTY_COLORS.items():
        pf = centralities.filter(pl.col("party") == party)
        if pf.height == 0:
            continue
        ax.scatter(
            pf[x_col].to_list(),
            pf[y_col].to_list(),
            c=color,
            label=party,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.5,
        )

    # Label top-5 by betweenness with callout boxes
    top = centralities.sort("betweenness", descending=True).head(5)
    for row in top.iter_rows(named=True):
        last_name = row["full_name"].split()[-1]
        ax.annotate(
            last_name,
            (row[x_col], row[y_col]),
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            xytext=(8, 6),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.3", "fc": "wheat", "alpha": 0.7},
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        )

    x_label = PLAIN_LABELS.get(x_col, x_col.replace("_", " ").title())
    y_label = PLAIN_LABELS.get(y_col, y_col.replace("_", " ").title())
    title = PLAIN_TITLES.get((x_col, y_col), f"{x_label} vs {y_label}")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"{chamber} \u2014 {title}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_fig(fig, out_path)


def plot_centrality_distributions(
    centralities: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """4-panel histogram of key centrality measures."""
    metrics = ["degree", "betweenness", "eigenvector", "pagerank"]
    panel_titles = {
        "degree": "How Connected Is Each Legislator?",
        "betweenness": "Who Bridges Different Groups?",
        "eigenvector": "Who Is Connected to the Most Influential?",
        "pagerank": "Overall Importance (PageRank)",
    }
    panel_subtitles = {
        "degree": "Higher = more co-voting connections",
        "betweenness": "Higher = links otherwise-separate blocs",
        "eigenvector": "Higher = connected to other well-connected legislators",
        "pagerank": "Higher = more central in the voting network",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for party, color in PARTY_COLORS.items():
            vals = centralities.filter(pl.col("party") == party)[metric].to_list()
            if vals:
                ax.hist(vals, bins=15, alpha=0.6, color=color, label=party, edgecolor="white")
        ax.set_xlabel(PLAIN_LABELS.get(metric, metric.replace("_", " ").title()), fontsize=9)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(
            f"{panel_titles[metric]}\n",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            0.5,
            1.0,
            panel_subtitles[metric],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontstyle="italic",
            color="#555555",
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{chamber} \u2014 How Important Is Each Legislator in the Voting Network?",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    save_fig(fig, out_dir / f"centrality_distributions_{chamber.lower()}.png")


def plot_centrality_vs_irt(
    centralities: pl.DataFrame,
    chamber: str,
    out_path: Path,
) -> None:
    """Scatter: betweenness centrality vs IRT ideal point."""
    plot_centrality_scatter(centralities, "xi_mean", "betweenness", chamber, out_path)


def plot_community_network(
    G: nx.Graph,
    partition: dict[str, int],
    chamber: str,
    resolution: float,
    out_path: Path,
    pos: dict | None = None,
) -> None:
    """Side-by-side: party colors vs community colors, with bridge highlights."""
    if G.number_of_nodes() == 0:
        return

    if pos is None:
        pos = _compute_layout(G)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    nodes = list(G.nodes())

    # Compute betweenness for bridge highlighting
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    top_bridge_slugs = sorted(betweenness, key=betweenness.get, reverse=True)[:3]  # type: ignore[arg-type]

    # Left: party colors
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.1, edge_color="#888888")

    # Draw non-bridge nodes first
    non_bridge = [n for n in nodes if n not in top_bridge_slugs]
    bridge_nodes = [n for n in nodes if n in top_bridge_slugs]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=non_bridge,
        ax=ax1,
        node_color=[PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in non_bridge],
        node_size=80,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )
    # Draw bridge nodes with red halo
    if bridge_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bridge_nodes,
            ax=ax1,
            node_color=[
                PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in bridge_nodes
            ],
            node_size=140,
            alpha=0.9,
            edgecolors="red",
            linewidths=2.5,
        )
        # Label bridge legislators
        for slug in bridge_nodes:
            name = G.nodes[slug].get("full_name", slug).split()[-1]
            ax1.annotate(
                name,
                pos[slug],
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                xytext=(0, 10),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.2", "fc": "lightyellow", "alpha": 0.8},
            )

    ax1.set_title(f"{chamber} \u2014 By Party", fontsize=13, fontweight="bold")
    ax1.axis("off")

    # Right: community colors
    n_communities = len(set(partition.values()))
    cmap = plt.cm.Set2  # type: ignore[attr-defined]
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.1, edge_color="#888888")

    # Non-bridge nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=non_bridge,
        ax=ax2,
        node_color=[cmap(partition.get(n, 0) / max(n_communities - 1, 1)) for n in non_bridge],
        node_size=80,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )
    # Bridge nodes with red halo
    if bridge_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bridge_nodes,
            ax=ax2,
            node_color=[
                cmap(partition.get(n, 0) / max(n_communities - 1, 1)) for n in bridge_nodes
            ],
            node_size=140,
            alpha=0.9,
            edgecolors="red",
            linewidths=2.5,
        )
        for slug in bridge_nodes:
            name = G.nodes[slug].get("full_name", slug).split()[-1]
            ax2.annotate(
                name,
                pos[slug],
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="bottom",
                xytext=(0, 10),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.2", "fc": "lightyellow", "alpha": 0.8},
            )

    legend_elements = [
        Patch(facecolor=cmap(i / max(n_communities - 1, 1)), label=f"Community {i}")
        for i in range(n_communities)
    ]
    # Add red halo legend entry
    legend_elements.append(
        Patch(facecolor="white", edgecolor="red", linewidth=2, label="Bridge legislator"),
    )
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax2.set_title(
        f"{chamber} \u2014 Detected Communities (res={resolution:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.axis("off")

    fig.suptitle(
        f"{chamber} \u2014 Party Alignment and Bridge Legislators",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Red-ringed nodes are legislators who connect otherwise-separate voting blocs"
        " (highest betweenness centrality)",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    save_fig(fig, out_path)


def plot_multi_resolution(
    resolution_df: pl.DataFrame,
    chamber: str,
    out_path: Path,
) -> None:
    """Plot n_communities and modularity vs resolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    res = resolution_df["resolution"].to_list()
    n_comm = resolution_df["n_communities"].to_list()
    mod = resolution_df["modularity"].to_list()

    ax1.plot(res, n_comm, "o-", color="#333333", linewidth=2, markersize=6)
    ax1.set_xlabel("Resolution", fontsize=11)
    ax1.set_ylabel("N Communities", fontsize=11)
    ax1.set_title("Communities vs Resolution", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(res, mod, "s-", color="#E81B23", linewidth=2, markersize=6)
    ax2.set_xlabel("Resolution", fontsize=11)
    ax2.set_ylabel("Modularity", fontsize=11)
    ax2.set_title("Modularity vs Resolution", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{chamber} — Multi-Resolution Louvain", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_fig(fig, out_path)


def plot_threshold_sweep(
    sweep_df: pl.DataFrame,
    chamber: str,
    out_path: Path,
) -> None:
    """Plot network statistics vs Kappa threshold."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    thresholds = sweep_df["threshold"].to_list()
    components = sweep_df["n_components"].to_list()

    # Find the threshold where network splits into 2 components (party split)
    split_threshold = None
    for i in range(len(components) - 1):
        if components[i] == 1 and components[i + 1] >= 2:
            # Linear interpolation between the two thresholds
            split_threshold = (thresholds[i] + thresholds[i + 1]) / 2
            break
    # If already 2+ at first threshold, mark the first threshold
    if split_threshold is None and len(components) > 0 and components[0] >= 2:
        split_threshold = thresholds[0]

    metrics = [
        ("n_edges", "Number of Connections"),
        ("density", "Network Density"),
        ("n_components", "Number of Separate Groups"),
        ("modularity", "How Clustered (Modularity)"),
    ]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        vals = sweep_df[col].to_list()
        ax.plot(thresholds, vals, "o-", color="#333333", linewidth=2, markersize=6)
        ax.set_xlabel("Agreement Threshold (\u03ba)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add vertical line at party split threshold
        if split_threshold is not None:
            ax.axvline(
                split_threshold,
                color="#E81B23",
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
            )
            if col == "n_components":
                ax.annotate(
                    "Network splits\ninto two parties",
                    (split_threshold, max(vals) * 0.7),
                    fontsize=8,
                    fontweight="bold",
                    color="#E81B23",
                    ha="center",
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
                )

    fig.suptitle(
        f"{chamber} \u2014 How Does the Network Change as We Raise the Bar for Agreement?",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    save_fig(fig, out_path)


def plot_cross_chamber_network(
    G: nx.Graph,
    out_path: Path,
) -> None:
    """Plot combined cross-chamber network with equated ideal points."""
    if G.number_of_nodes() == 0:
        return

    pos = _compute_layout(G, xi_attr="xi_equated")

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    nodes = list(G.nodes())

    # Color by party, shape by chamber
    chambers = {n: G.nodes[n].get("chamber", "Unknown") for n in nodes}
    parties = {n: G.nodes[n].get("party", "Unknown") for n in nodes}

    for chamber_val, marker in [("House", "o"), ("Senate", "s")]:
        for party, color in PARTY_COLORS.items():
            subset = [n for n in nodes if chambers[n] == chamber_val and parties[n] == party]
            if not subset:
                continue
            x = [pos[n][0] for n in subset]
            y = [pos[n][1] for n in subset]
            ax.scatter(
                x,
                y,
                c=color,
                marker=marker,
                s=60,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                label=f"{chamber_val} {party}",
            )

    # Draw edges (light)
    for u, v, d in G.edges(data=True):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="#888888", alpha=0.05, linewidth=0.3)

    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Cross-Chamber Network (Equated Ideal Points)", fontsize=14, fontweight="bold")
    ax.axis("off")

    save_fig(fig, out_path)


def plot_edge_weight_distribution(
    G: nx.Graph,
    chamber: str,
    out_path: Path,
) -> None:
    """Histogram of edge weights by edge type (within-R, within-D, cross-party)."""
    within_r = []
    within_d = []
    cross = []

    for u, v, d in G.edges(data=True):
        pu = G.nodes[u].get("party", "")
        pv = G.nodes[v].get("party", "")
        w = d["weight"]
        if pu == "Republican" and pv == "Republican":
            within_r.append(w)
        elif pu == "Democrat" and pv == "Democrat":
            within_d.append(w)
        else:
            cross.append(w)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.linspace(0.3, 1.0, 30)
    if within_r:
        ax.hist(
            within_r,
            bins=bins,
            alpha=0.6,
            color=PARTY_COLORS["Republican"],
            label=f"R-R ({len(within_r)})",
            edgecolor="white",
        )
    if within_d:
        ax.hist(
            within_d,
            bins=bins,
            alpha=0.6,
            color=PARTY_COLORS["Democrat"],
            label=f"D-D ({len(within_d)})",
            edgecolor="white",
        )
    if cross:
        ax.hist(
            cross,
            bins=bins,
            alpha=0.6,
            color="#888888",
            label=f"Cross-party ({len(cross)})",
            edgecolor="white",
        )

    ax.set_xlabel("Kappa (Edge Weight)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{chamber} — Edge Weight Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    save_fig(fig, out_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from ks_vote_scraper.session import KSSession

    ks = KSSession.from_session_string(args.session)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("data") / ks.output_name

    results_root = Path("results") / ks.output_name

    if args.eda_dir:
        eda_dir = Path(args.eda_dir)
    else:
        eda_dir = results_root / "eda" / "latest"

    if args.irt_dir:
        irt_dir = Path(args.irt_dir)
    else:
        irt_dir = results_root / "irt" / "latest"

    if args.clustering_dir:
        clustering_dir = Path(args.clustering_dir)
    else:
        clustering_dir = results_root / "clustering" / "latest"

    with RunContext(
        session=args.session,
        analysis_name="network",
        params=vars(args),
        primer=NETWORK_PRIMER,
    ) as ctx:
        print(f"KS Legislature Network Analysis — Session {args.session}")
        print(f"Data:        {data_dir}")
        print(f"EDA:         {eda_dir}")
        print(f"IRT:         {irt_dir}")
        print(f"Clustering:  {clustering_dir}")
        print(f"Output:      {ctx.run_dir}")
        print(f"Threshold:   {args.kappa_threshold}")

        # ── Phase 1: Load data ──
        print_header("PHASE 1: LOADING DATA")
        kappa_house, kappa_senate = load_kappa_matrices(eda_dir)
        irt_house, irt_senate = load_ideal_points(irt_dir)
        vm_house, vm_senate = load_vote_matrices(eda_dir)
        bp_house, bp_senate = load_bill_params(irt_dir)
        clust_house, clust_senate = load_cluster_assignments(clustering_dir)
        loy_house, loy_senate = load_party_loyalty(clustering_dir)
        rollcalls, _legislators = load_metadata(data_dir)

        # Load equated ideal points (may not exist)
        equated_ip = None
        eq_path = irt_dir / "data" / "ideal_points_joint_equated.parquet"
        if eq_path.exists():
            equated_ip = load_equated_ideal_points(irt_dir)
            print(f"  Equated ideal points: {equated_ip.height} legislators")
        else:
            print("  Equated ideal points: not found (cross-chamber will be skipped)")
            args.skip_cross_chamber = True

        print(f"  Kappa House:  {kappa_house.height} x {len(kappa_house.columns) - 1}")
        print(f"  Kappa Senate: {kappa_senate.height} x {len(kappa_senate.columns) - 1}")
        print(f"  IRT House:    {irt_house.height} legislators")
        print(f"  IRT Senate:   {irt_senate.height} legislators")
        print(f"  Vote matrix House:  {vm_house.height} x {len(vm_house.columns) - 1}")
        print(f"  Vote matrix Senate: {vm_senate.height} x {len(vm_senate.columns) - 1}")
        print(f"  Bill params House:  {bp_house.height}")
        print(f"  Bill params Senate: {bp_senate.height}")
        print(f"  Rollcalls: {rollcalls.height}")
        print(f"  Legislators: {_legislators.height}")

        chamber_configs = [
            ("House", kappa_house, irt_house, vm_house, bp_house, clust_house, loy_house),
            ("Senate", kappa_senate, irt_senate, vm_senate, bp_senate, clust_senate, loy_senate),
        ]

        results: dict[str, dict] = {}

        for chamber, kappa, irt_ip, vm, bp, clust, loy in chamber_configs:
            if irt_ip.height < 5:
                print(f"\n  Skipping {chamber}: too few legislators ({irt_ip.height})")
                continue

            chamber_results: dict = {
                "ideal_points": irt_ip,
                "kappa_matrix": kappa,
                "vote_matrix": vm,
                "bill_params": bp,
                "cluster_assignments": clust,
                "loyalty": loy,
            }

            # ── Phase 2: Build Network ──
            print_header(f"PHASE 2: NETWORK CONSTRUCTION — {chamber}")
            G = build_kappa_network(kappa, irt_ip, clust, loy, threshold=args.kappa_threshold)
            summary = compute_network_summary(G)
            chamber_results["graph"] = G
            chamber_results["summary"] = summary

            print(f"  Nodes: {summary['n_nodes']}")
            print(f"  Edges: {summary['n_edges']}")
            print(f"  Density: {summary['density']}")
            print(f"  Components: {summary['n_components']}")
            print(f"  Avg clustering coeff: {summary['avg_clustering']}")
            print(f"  Transitivity: {summary['transitivity']}")
            print(f"  Party assortativity: {summary['assortativity_party']}")

            # ── Phase 3: Centrality ──
            print_header(f"PHASE 3: CENTRALITY MEASURES — {chamber}")
            centralities = compute_centralities(G)
            centralities.write_parquet(ctx.data_dir / f"centrality_{chamber.lower()}.parquet")
            print(f"  Saved: centrality_{chamber.lower()}.parquet")
            chamber_results["centralities"] = centralities

            bridges = identify_bridge_legislators(centralities, G)
            bridges.write_parquet(ctx.data_dir / f"bridge_legislators_{chamber.lower()}.parquet")
            print(f"  Saved: bridge_legislators_{chamber.lower()}.parquet")
            chamber_results["bridges"] = bridges

            party_summary = compute_party_centrality_summary(centralities)
            chamber_results["party_centrality"] = party_summary

            # Print top 5 by betweenness
            top5 = centralities.sort("betweenness", descending=True).head(5)
            print("  Top 5 by betweenness:")
            for row in top5.iter_rows(named=True):
                print(
                    f"    {row['full_name']}: betweenness={row['betweenness']:.4f}, "
                    f"party={row['party']}, xi={row['xi_mean']:+.3f}"
                )

            # ── Phase 4: Community Detection ──
            print_header(f"PHASE 4: COMMUNITY DETECTION — {chamber}")
            if G.number_of_edges() > 0:
                partitions, resolution_df = detect_communities_multi_resolution(G)
                resolution_df.write_parquet(
                    ctx.data_dir / f"community_resolution_{chamber.lower()}.parquet"
                )
                print(f"  Saved: community_resolution_{chamber.lower()}.parquet")

                # Pick best resolution by modularity
                best_idx = resolution_df["modularity"].arg_max()
                best_res = resolution_df["resolution"][best_idx]
                best_partition = partitions[best_res]
                chamber_results["best_partition"] = best_partition
                chamber_results["best_resolution"] = float(best_res)
                chamber_results["partitions"] = partitions
                chamber_results["resolution_df"] = resolution_df

                # Save community assignments at best resolution
                comm_df = pl.DataFrame(
                    [{"legislator_slug": n, "community": c} for n, c in best_partition.items()]
                )
                comm_df.write_parquet(ctx.data_dir / f"communities_{chamber.lower()}.parquet")
                print(f"  Saved: communities_{chamber.lower()}.parquet")

                # Composition
                composition = analyze_community_composition(best_partition, G, chamber)
                composition.write_parquet(
                    ctx.data_dir / f"community_composition_{chamber.lower()}.parquet"
                )
                print(f"  Saved: community_composition_{chamber.lower()}.parquet")
                chamber_results["community_composition"] = composition

                # Compare to party
                party_comparison = compare_communities_to_party(best_partition, G)
                chamber_results["community_vs_party"] = party_comparison
                print(
                    f"  Communities vs Party: NMI={party_comparison['nmi']:.4f}, "
                    f"ARI={party_comparison['ari']:.4f}"
                )
                if party_comparison["misclassified"]:
                    print(f"  Misclassified ({len(party_comparison['misclassified'])}):")
                    for m in party_comparison["misclassified"]:
                        print(f"    {m['full_name']} ({m['party']}) in community {m['community']}")

                # Compare to clustering
                clust_comparison = compare_communities_to_clusters(best_partition, clust, G)
                chamber_results["community_vs_clusters"] = clust_comparison
                print(
                    f"  Communities vs Clusters: NMI={clust_comparison.get('nmi')}, "
                    f"ARI={clust_comparison.get('ari')}"
                )
            else:
                print("  No edges — skipping community detection")
                chamber_results["best_partition"] = {}
                chamber_results["resolution_df"] = pl.DataFrame()

            # ── Phase 5: Within-Party Subnetworks ──
            print_header(f"PHASE 5: WITHIN-PARTY SUBNETWORKS — {chamber}")
            within_party_results: dict[str, dict] = {}

            for party in ["Republican", "Democrat"]:
                party_key = party.lower()
                G_party = build_within_party_network(
                    kappa, irt_ip, party, threshold=args.kappa_threshold
                )
                print(
                    f"  {party}: {G_party.number_of_nodes()} nodes, "
                    f"{G_party.number_of_edges()} edges"
                )

                if G_party.number_of_nodes() < 10:
                    within_party_results[party_key] = {
                        "skipped": True,
                        "reason": f"Too few {party} legislators ({G_party.number_of_nodes()})",
                    }
                    continue

                wp_result = analyze_within_party_communities(G_party, party=party, chamber=chamber)
                within_party_results[party_key] = wp_result

                # Save within-party community assignments
                if not wp_result.get("skipped", True) and wp_result.get("partition"):
                    wp_df = pl.DataFrame(
                        [
                            {"legislator_slug": n, "community": c}
                            for n, c in wp_result["partition"].items()
                        ]
                    )
                    wp_df.write_parquet(
                        ctx.data_dir / f"within_party_{party_key}_{chamber.lower()}.parquet"
                    )
                    print(f"  Saved: within_party_{party_key}_{chamber.lower()}.parquet")

            chamber_results["within_party"] = within_party_results

            # Tyson-Thompson edge weights (Senate only)
            tt_result = check_tyson_thompson_edge_weights(G, chamber)
            chamber_results["tyson_thompson"] = tt_result
            if tt_result:
                print(f"  R-R median edge weight: {tt_result['r_median_edge_weight']}")
                for leg in tt_result.get("legislators", []):
                    print(
                        f"    {leg['name']}: mean R weight={leg['mean_r_weight']:.4f} "
                        f"(vs R median: {leg['vs_r_median']:+.4f})"
                    )

            # ── Phase 6: Threshold Sensitivity ──
            print_header(f"PHASE 6: THRESHOLD SENSITIVITY — {chamber}")
            sweep = run_threshold_sweep(kappa, irt_ip)
            sweep.write_parquet(ctx.data_dir / f"threshold_sweep_{chamber.lower()}.parquet")
            print(f"  Saved: threshold_sweep_{chamber.lower()}.parquet")
            chamber_results["threshold_sweep"] = sweep

            # ── Phase 7: High-Discrimination Subnetwork ──
            if not args.skip_high_disc:
                print_header(f"PHASE 7: HIGH-DISC SUBNETWORK — {chamber}")
                G_hd = build_high_disc_network(vm, bp, irt_ip)
                if G_hd is not None:
                    hd_summary = compute_network_summary(G_hd)
                    chamber_results["high_disc_graph"] = G_hd
                    chamber_results["high_disc_summary"] = hd_summary
                    print(f"  Nodes: {hd_summary['n_nodes']}, Edges: {hd_summary['n_edges']}")
                else:
                    chamber_results["high_disc_graph"] = None
                    chamber_results["high_disc_summary"] = None
            else:
                print_header(f"PHASE 7: HIGH-DISC SUBNETWORK (SKIPPED) — {chamber}")
                chamber_results["high_disc_graph"] = None
                chamber_results["high_disc_summary"] = None

            # ── Phase 8: Veto Override Subnetwork ──
            print_header(f"PHASE 8: VETO OVERRIDE SUBNETWORK — {chamber}")
            G_veto = build_veto_override_network(
                vm, rollcalls, irt_ip, chamber, kappa_threshold=args.kappa_threshold
            )
            if G_veto is not None:
                veto_summary = compute_network_summary(G_veto)
                chamber_results["veto_graph"] = G_veto
                chamber_results["veto_summary"] = veto_summary
                print(f"  Nodes: {veto_summary['n_nodes']}, Edges: {veto_summary['n_edges']}")
            else:
                chamber_results["veto_graph"] = None
                chamber_results["veto_summary"] = None

            # ── Phase 10: Plots ──
            print_header(f"PHASE 10: PLOTS — {chamber}")

            # Network layout (party colors)
            pos = plot_network_layout(
                G,
                "party",
                chamber,
                "Co-Voting Network (Party Colors)",
                ctx.plots_dir / f"network_party_{chamber.lower()}.png",
            )
            # Network layout (IRT gradient)
            plot_network_layout(
                G,
                "xi_mean",
                chamber,
                "Co-Voting Network (IRT Gradient)",
                ctx.plots_dir / f"network_irt_{chamber.lower()}.png",
                pos=pos,
            )

            # Edge weight distribution
            plot_edge_weight_distribution(
                G, chamber, ctx.plots_dir / f"edge_weights_{chamber.lower()}.png"
            )

            # Centrality distributions
            plot_centrality_distributions(centralities, chamber, ctx.plots_dir)

            # Centrality scatter (betweenness vs eigenvector)
            plot_centrality_scatter(
                centralities,
                "betweenness",
                "eigenvector",
                chamber,
                ctx.plots_dir / f"centrality_scatter_{chamber.lower()}.png",
            )

            # Centrality vs IRT
            plot_centrality_vs_irt(
                centralities,
                chamber,
                ctx.plots_dir / f"centrality_vs_irt_{chamber.lower()}.png",
            )

            # Multi-resolution
            res_df = chamber_results.get("resolution_df")
            if res_df is not None and res_df.height > 0:
                plot_multi_resolution(
                    res_df,
                    chamber,
                    ctx.plots_dir / f"multi_resolution_{chamber.lower()}.png",
                )

            # Community network (party vs community side-by-side)
            if chamber_results.get("best_partition"):
                plot_community_network(
                    G,
                    chamber_results["best_partition"],
                    chamber,
                    chamber_results.get("best_resolution", 1.0),
                    ctx.plots_dir / f"community_network_{chamber.lower()}.png",
                    pos=pos,
                )

            # Threshold sweep
            plot_threshold_sweep(
                sweep,
                chamber,
                ctx.plots_dir / f"threshold_sweep_{chamber.lower()}.png",
            )

            # High-disc network
            if chamber_results.get("high_disc_graph") is not None:
                plot_network_layout(
                    chamber_results["high_disc_graph"],
                    "party",
                    chamber,
                    "High-Discrimination Subnetwork",
                    ctx.plots_dir / f"high_disc_network_{chamber.lower()}.png",
                )

            results[chamber] = chamber_results

        # ── Phase 9: Cross-Chamber Network ──
        if not args.skip_cross_chamber and equated_ip is not None:
            print_header("PHASE 9: CROSS-CHAMBER NETWORK")
            G_house = results.get("House", {}).get("graph")
            G_senate = results.get("Senate", {}).get("graph")
            if G_house is not None and G_senate is not None:
                G_cross = build_cross_chamber_network(equated_ip, G_house, G_senate)
                cross_summary = compute_network_summary(G_cross)
                results["cross_chamber"] = {
                    "graph": G_cross,
                    "summary": cross_summary,
                }
                print(f"  Nodes: {cross_summary['n_nodes']}, Edges: {cross_summary['n_edges']}")

                plot_cross_chamber_network(
                    G_cross,
                    ctx.plots_dir / "cross_chamber_network.png",
                )
            else:
                print("  Skipped: missing chamber graph(s)")

        # Save network summary
        summary_rows = []
        for chamber in ["House", "Senate"]:
            if chamber in results and "summary" in results[chamber]:
                row = {"chamber": chamber}
                row.update(results[chamber]["summary"])
                summary_rows.append(row)
        if summary_rows:
            summary_df = pl.DataFrame(summary_rows)
            summary_df.write_parquet(ctx.data_dir / "network_summary.parquet")
            print("\n  Saved: network_summary.parquet")

        # ── Filtering Manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "network",
            "constants": {
                "KAPPA_THRESHOLD_DEFAULT": KAPPA_THRESHOLD_DEFAULT,
                "KAPPA_THRESHOLD_SENSITIVITY": KAPPA_THRESHOLD_SENSITIVITY,
                "LOUVAIN_RESOLUTIONS": LOUVAIN_RESOLUTIONS,
                "HIGH_DISC_THRESHOLD": HIGH_DISC_THRESHOLD,
                "TOP_BRIDGE_N": TOP_BRIDGE_N,
                "RANDOM_SEED": RANDOM_SEED,
            },
            "kappa_threshold_used": args.kappa_threshold,
            "skip_cross_chamber": args.skip_cross_chamber,
            "skip_high_disc": args.skip_high_disc,
        }

        for chamber in ["House", "Senate"]:
            if chamber not in results:
                continue
            r = results[chamber]
            ch = chamber.lower()
            manifest[f"{ch}_n_nodes"] = r.get("summary", {}).get("n_nodes")
            manifest[f"{ch}_n_edges"] = r.get("summary", {}).get("n_edges")
            manifest[f"{ch}_density"] = r.get("summary", {}).get("density")
            manifest[f"{ch}_n_components"] = r.get("summary", {}).get("n_components")
            manifest[f"{ch}_assortativity_party"] = r.get("summary", {}).get("assortativity_party")
            if "community_vs_party" in r:
                manifest[f"{ch}_community_vs_party_nmi"] = r["community_vs_party"]["nmi"]
                manifest[f"{ch}_community_vs_party_ari"] = r["community_vs_party"]["ari"]
            if "community_vs_clusters" in r:
                manifest[f"{ch}_community_vs_clusters_nmi"] = r["community_vs_clusters"].get("nmi")
                manifest[f"{ch}_community_vs_clusters_ari"] = r["community_vs_clusters"].get("ari")
            if r.get("within_party"):
                wp_summary = {}
                for pk, pd in r["within_party"].items():
                    if isinstance(pd, dict):
                        wp_summary[pk] = {
                            k: v
                            for k, v in pd.items()
                            if k not in ("partition", "resolution_results")
                        }
                manifest[f"{ch}_within_party"] = wp_summary
            if r.get("tyson_thompson"):
                tt = r["tyson_thompson"]
                manifest[f"{ch}_tyson_thompson"] = {
                    k: v for k, v in tt.items() if k != "legislators"
                }
                if tt.get("legislators"):
                    manifest[f"{ch}_tyson_thompson"]["legislators"] = tt["legislators"]

        manifest_path = ctx.run_dir / "filtering_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print("  Saved: filtering_manifest.json")

        # ── HTML Report ──
        print_header("HTML REPORT")
        build_network_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
            kappa_threshold=args.kappa_threshold,
            skip_cross_chamber=args.skip_cross_chamber,
            skip_high_disc=args.skip_high_disc,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
