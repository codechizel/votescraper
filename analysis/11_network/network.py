"""
Kansas Legislature — Network Analysis (Phase 6)

Transforms Cohen's Kappa agreement matrices into weighted graphs, computes
centrality measures to identify structurally important legislators, and runs
Leiden community detection (modularity + CPM) at multiple resolutions to test
whether network-based grouping finds finer structure than the k=2 party split
from clustering.

Usage:
  uv run python analysis/network.py [--session 2025-26] [--kappa-threshold 0.40]
      [--skip-cross-chamber] [--skip-high-disc]

Outputs (in results/<session>/network/<date>/):
  - data/:   Parquet files (centrality, communities, threshold sweep)
  - plots/:  PNG visualizations (layouts, centrality, communities)
  - filtering_manifest.json, run_info.json, run_log.txt
  - network_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.network_report import build_network_report
except ModuleNotFoundError:
    from network_report import build_network_report  # type: ignore[no-redef]

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ImportError:
    from phase_utils import load_metadata, print_header, save_fig


# ── Constants ────────────────────────────────────────────────────────────────

KAPPA_THRESHOLD_DEFAULT = 0.40
KAPPA_THRESHOLD_SENSITIVITY = [0.30, 0.40, 0.50, 0.60]
CROSS_PARTY_BRIDGE_THRESHOLD = 0.30
LEIDEN_RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
CPM_GAMMAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
HIGH_DISC_THRESHOLD = 1.5
TOP_BRIDGE_N = 15
TOP_LABEL_N = 10
RANDOM_SEED = 42
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}

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
Two complementary algorithms via Leiden (Traag et al., 2019):
- **Modularity optimization** at 8 resolution parameters (0.5 to 3.0). Lower
  resolutions produce fewer, larger communities; higher resolutions split into
  potential subcaucuses.
- **CPM (Constant Potts Model)** at 8 gamma values (0.05 to 0.50). Resolution-
  limit-free — can detect subcaucuses of any size, including moderate Republican
  wings that fall below modularity's theoretical floor.
Communities compared to party labels and clustering assignments via NMI and ARI.

### Polarization and Backbone
- **Party modularity** (Waugh et al., 2009): modularity of the party-labeled
  partition — higher values indicate stronger partisan polarization.
- **Disparity filter** (Serrano et al., 2009): statistically significant edge
  extraction, keeping only edges whose weight is anomalously high for the node's
  degree. Reduces visual clutter while preserving structurally important connections.

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
| `community_resolution_{chamber}.parquet` | Leiden modularity at each resolution |
| `cpm_resolution_{chamber}.parquet` | CPM results at each gamma |
| `bridge_legislators_{chamber}.parquet` | Top betweenness bridge legislators |
| `threshold_sweep_{chamber}.parquet` | Network stats at each Kappa threshold |
| `within_party_{party}_{chamber}.parquet` | Within-party community results |
| `network_summary.parquet` | Summary statistics per chamber (incl. party modularity) |
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
- Leiden is stochastic; seed is fixed but results may vary across versions
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
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
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


def _build_ip_lookup(ideal_points: pl.DataFrame) -> dict[str, dict]:
    """Build a slug → row dict from an ideal points DataFrame."""
    return {row["legislator_slug"]: row for row in ideal_points.iter_rows(named=True)}


def _graph_from_kappa_matrix(
    kappa_mat: np.ndarray,
    slugs: list[str],
    ip_dict: dict[str, dict],
    threshold: float,
    extra_attrs: dict[str, dict[str, object]] | None = None,
) -> nx.Graph:
    """Build a weighted NetworkX graph from a kappa matrix and slug list.

    Shared by build_kappa_network() and _build_network_from_vote_subset().

    Args:
        kappa_mat: n×n symmetric matrix of pairwise Kappa values (NaN = missing).
        slugs: List of legislator slugs corresponding to matrix rows/columns.
        ip_dict: Mapping slug → {party, xi_mean, full_name, ...} from ideal points.
        threshold: Minimum Kappa for edge creation.
        extra_attrs: Optional per-slug additional node attributes.
    """
    n = len(slugs)
    G = nx.Graph()
    for slug in slugs:
        ip = ip_dict.get(slug, {})
        attrs = {
            "party": ip.get("party", "Unknown"),
            "xi_mean": ip.get("xi_mean", 0.0),
            "full_name": ip.get("full_name", slug),
        }
        if extra_attrs and slug in extra_attrs:
            attrs.update(extra_attrs[slug])
        G.add_node(slug, **attrs)

    for i in range(n):
        for j in range(i + 1, n):
            k = kappa_mat[i, j]
            if np.isnan(k) or k <= threshold:
                continue
            G.add_edge(slugs[i], slugs[j], weight=float(k), distance=1.0 / float(k))

    return G


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

    ip_dict = _build_ip_lookup(ideal_points)
    return _graph_from_kappa_matrix(kappa_mat, slugs, ip_dict, kappa_threshold)


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def _load_pair(
    base_dir: Path, pattern: str
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load house/senate parquet pair. Returns None per chamber if unavailable."""
    results: list[pl.DataFrame | None] = []
    for ch in ("house", "senate"):
        path = base_dir / "data" / pattern.format(ch=ch)
        results.append(pl.read_parquet(path) if path.exists() else None)
    return results[0], results[1]


def load_kappa_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load pairwise Kappa agreement matrices from EDA."""
    return _load_pair(eda_dir, "agreement_kappa_{ch}.parquet")


def load_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT ideal points for both chambers."""
    return _load_pair(irt_dir, "ideal_points_{ch}.parquet")


def load_equated_ideal_points(irt_dir: Path) -> pl.DataFrame:
    """Load cross-chamber equated ideal points."""
    return pl.read_parquet(irt_dir / "data" / "ideal_points_joint_equated.parquet")


def load_bill_params(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load bill discrimination parameters for high-disc subnetwork."""
    return _load_pair(irt_dir, "bill_params_{ch}.parquet")


def load_cluster_assignments(
    clustering_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load k-means cluster assignments from clustering phase."""
    return _load_pair(clustering_dir, "kmeans_assignments_{ch}.parquet")


def load_party_loyalty(
    clustering_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load party loyalty scores from clustering phase."""
    return _load_pair(clustering_dir, "party_loyalty_{ch}.parquet")


def load_vote_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load filtered binary vote matrices from EDA."""
    return _load_pair(eda_dir, "vote_matrix_{ch}_filtered.parquet")


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
    ip_dict = _build_ip_lookup(ideal_points)

    # Build extra attributes from cluster assignments and loyalty
    extra_attrs: dict[str, dict[str, object]] = {}

    cluster_dict: dict[str, int] = {}
    if cluster_assignments is not None:
        cluster_cols = [c for c in cluster_assignments.columns if c.startswith("cluster_k")]
        cluster_col = cluster_cols[0] if cluster_cols else None
        if cluster_col:
            for row in cluster_assignments.iter_rows(named=True):
                cluster_dict[row["legislator_slug"]] = row[cluster_col]

    loyalty_dict: dict[str, float] = {}
    if loyalty is not None:
        for row in loyalty.iter_rows(named=True):
            loyalty_dict[row["legislator_slug"]] = row["loyalty_rate"]

    for slug in slugs:
        ip = ip_dict.get(slug, {})
        extra_attrs[slug] = {
            "chamber": ip.get("chamber", "Unknown"),
            "xi_sd": ip.get("xi_sd", 1.0),
            "cluster": cluster_dict.get(slug, -1),
            "loyalty_rate": loyalty_dict.get(slug, None),
        }

    return _graph_from_kappa_matrix(mat, slugs, ip_dict, threshold, extra_attrs)


def generate_pyvis_network(G: nx.Graph, chamber: str, out_dir: Path) -> None:
    """Generate an interactive PyVis HTML network visualization.

    Nodes colored by party (Republican=#E81B23, Democrat=#0015BC, Independent=#999999),
    sized proportional to degree.
    """
    from pyvis.network import Network

    if G.number_of_nodes() == 0:
        return

    net = Network(height="600px", width="100%", notebook=False, cdn_resources="in_line")
    net.from_nx(G)

    # Compute degree for sizing
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    for node in net.nodes:
        node_id = node["id"]
        party = G.nodes[node_id].get("party", "Unknown")
        node["color"] = PARTY_COLORS.get(party, "#999999")
        deg = degrees.get(node_id, 1)
        node["size"] = 10 + 30 * (deg / max_deg)
        full_name = G.nodes[node_id].get("full_name", node_id)
        node["title"] = f"{full_name} ({party}), degree={deg}"

    out_path = out_dir / f"network_interactive_{chamber.lower()}.html"
    html = net.generate_html()
    out_path.write_text(html, encoding="utf-8")
    print(f"  Saved interactive network: {out_path.name}")


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


def compute_party_modularity(G: nx.Graph) -> float | None:
    """Compute modularity of the party-labeled partition (Waugh et al., 2009).

    Measures political polarization: high modularity means parties form distinct
    network clusters. Returns None if the graph has no edges or only one party.
    """
    if G.number_of_edges() == 0:
        return None

    party_communities: dict[str, set[str]] = {}
    for node in G.nodes():
        party = G.nodes[node].get("party", "Unknown")
        party_communities.setdefault(party, set()).add(node)

    if len(party_communities) < 2:
        return None

    return round(nx.community.modularity(G, party_communities.values(), weight="weight"), 4)


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

    # Harmonic centrality (handles disconnected graphs natively — finite for all nodes)
    harmonic = nx.harmonic_centrality(G, distance="distance")

    # Cross-party edge fraction: for each node, fraction of edges to other-party nodes
    cross_party_fraction: dict[str, float] = {}
    for n in nodes:
        n_party = G.nodes[n].get("party", "Unknown")
        total_edges = 0
        cross_edges = 0
        for _, neighbor in G.edges(n):
            total_edges += 1
            if G.nodes[neighbor].get("party", "Unknown") != n_party:
                cross_edges += 1
        cross_party_fraction[n] = cross_edges / total_edges if total_edges > 0 else 0.0

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
                "harmonic": harmonic.get(n, 0.0),
                "cross_party_fraction": cross_party_fraction.get(n, 0.0),
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


def _nx_to_igraph(G: nx.Graph) -> ig.Graph:
    """Convert a NetworkX Graph to igraph, preserving node names and edge weights."""
    G_ig = ig.Graph.from_networkx(G)
    return G_ig


def _leiden_partition_to_dict(
    G_ig: ig.Graph, partition: leidenalg.VertexPartition
) -> dict[str, int]:
    """Convert a leidenalg partition back to {node_name: community_id} dict."""
    return {G_ig.vs[i]["_nx_name"]: partition.membership[i] for i in range(G_ig.vcount())}


def _compute_modularity(G: nx.Graph, partition: dict[str, int]) -> float:
    """Compute modularity of a partition on a NetworkX graph."""
    communities: dict[int, set[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)
    return nx.community.modularity(G, communities.values(), weight="weight")


def detect_communities_multi_resolution(
    G: nx.Graph,
    resolutions: list[float] | None = None,
) -> tuple[dict[float, dict[str, int]], pl.DataFrame]:
    """Run Leiden community detection at multiple resolution parameters.

    Uses RBConfigurationVertexPartition (modularity optimization), the direct
    replacement for Louvain. Leiden guarantees well-connected communities
    (Traag et al., 2019).

    Returns:
        partitions_dict: {resolution: {node: community_id}}
        resolution_df: DataFrame with (resolution, n_communities, modularity)
    """
    if resolutions is None:
        resolutions = LEIDEN_RESOLUTIONS

    G_ig = _nx_to_igraph(G)
    partitions_dict: dict[float, dict[str, int]] = {}
    resolution_rows = []

    for res in resolutions:
        part = leidenalg.find_partition(
            G_ig,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=RANDOM_SEED,
        )
        partition = _leiden_partition_to_dict(G_ig, part)
        partitions_dict[res] = partition

        n_communities = len(set(partition.values()))
        modularity = _compute_modularity(G, partition)

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


def detect_communities_cpm(
    G: nx.Graph,
    gammas: list[float] | None = None,
) -> tuple[dict[float, dict[str, int]], pl.DataFrame]:
    """Run Leiden with CPM (Constant Potts Model) at multiple gamma values.

    CPM is resolution-limit-free: it can detect communities of any size,
    including subcaucuses smaller than sqrt(2m) — the theoretical floor
    for modularity-based methods (Fortunato & Barthelemy, 2007).

    Gamma sets the expected internal edge density: communities must have
    internal density exceeding gamma to be retained. Higher gamma → more,
    smaller communities.

    Returns:
        partitions_dict: {gamma: {node: community_id}}
        gamma_df: DataFrame with (gamma, n_communities, modularity)
    """
    if gammas is None:
        gammas = CPM_GAMMAS

    G_ig = _nx_to_igraph(G)
    partitions_dict: dict[float, dict[str, int]] = {}
    gamma_rows = []

    for gamma in gammas:
        part = leidenalg.find_partition(
            G_ig,
            leidenalg.CPMVertexPartition,
            weights="weight",
            resolution_parameter=gamma,
            seed=RANDOM_SEED,
        )
        partition = _leiden_partition_to_dict(G_ig, part)
        partitions_dict[gamma] = partition

        n_communities = len(set(partition.values()))
        modularity = _compute_modularity(G, partition)

        gamma_rows.append(
            {
                "gamma": gamma,
                "n_communities": n_communities,
                "modularity": round(modularity, 4),
            }
        )
        print(f"    CPM γ={gamma:.2f}: {n_communities} communities, modularity={modularity:.4f}")

    gamma_df = pl.DataFrame(gamma_rows)
    return partitions_dict, gamma_df


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

        n_other = n_total - parties.get("Republican", 0) - parties.get("Democrat", 0)
        row: dict = {
            "chamber": chamber,
            "community": comm_id,
            "n_legislators": n_total,
            "n_republican": parties.get("Republican", 0),
            "n_democrat": parties.get("Democrat", 0),
            "n_other": n_other,
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
    """Run Leiden community detection on a within-party subnetwork."""
    if resolutions is None:
        resolutions = LEIDEN_RESOLUTIONS

    if G_party.number_of_edges() == 0:
        return {"skipped": True, "reason": "No edges in party subnetwork"}

    G_ig = _nx_to_igraph(G_party)
    best_mod = -1.0
    best_res = resolutions[0]
    best_partition: dict[str, int] = {}
    all_results = []

    for res in resolutions:
        part = leidenalg.find_partition(
            G_ig,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=RANDOM_SEED,
        )
        partition = _leiden_partition_to_dict(G_ig, part)
        mod = _compute_modularity(G_party, partition)
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


def check_extreme_edge_weights(
    G: nx.Graph,
    chamber: str,
    top_n: int = 2,
) -> dict | None:
    """Find the most ideologically extreme majority-party legislators and compare
    their intra-party edge weights to the party median.

    Data-driven: selects the top_n majority-party legislators with highest |xi_mean|.
    """
    # Determine majority party
    party_counts: dict[str, int] = {}
    for n in G.nodes():
        p = G.nodes[n].get("party", "Unknown")
        party_counts[p] = party_counts.get(p, 0) + 1
    if not party_counts:
        return None
    majority_party = max(party_counts, key=party_counts.get)  # type: ignore[arg-type]

    # Get majority-party nodes
    maj_nodes = {n for n in G.nodes() if G.nodes[n].get("party") == majority_party}
    if len(maj_nodes) < 3:
        return None

    # All intra-majority edge weights
    maj_edges = []
    for u, v, d in G.edges(data=True):
        if u in maj_nodes and v in maj_nodes:
            maj_edges.append(d["weight"])
    if not maj_edges:
        return None

    maj_median = float(np.median(maj_edges))
    maj_mean = float(np.mean(maj_edges))

    # Find top_n most ideologically extreme majority-party legislators by |xi_mean|
    extreme = sorted(maj_nodes, key=lambda n: abs(G.nodes[n].get("xi_mean", 0.0)), reverse=True)[
        :top_n
    ]

    results: dict = {
        "majority_party": majority_party,
        "chamber": chamber,
        "r_median_edge_weight": round(maj_median, 4),
        "r_mean_edge_weight": round(maj_mean, 4),
        "r_n_edges": len(maj_edges),
        "legislators": [],
    }

    for slug in extreme:
        name = G.nodes[slug].get("full_name", slug)
        edges = [(v, d["weight"]) for _, v, d in G.edges(slug, data=True)]
        maj_only = [(v, w) for v, w in edges if G.nodes[v].get("party") == majority_party]
        if maj_only:
            weights = [w for _, w in maj_only]
            results["legislators"].append(
                {
                    "slug": slug,
                    "name": name,
                    "xi_mean": round(G.nodes[slug].get("xi_mean", 0.0), 3),
                    "n_r_edges": len(maj_only),
                    "mean_r_weight": round(float(np.mean(weights)), 4),
                    "median_r_weight": round(float(np.median(weights)), 4),
                    "min_r_weight": round(float(np.min(weights)), 4),
                    "vs_r_median": round(float(np.mean(weights)) - maj_median, 4),
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
            G_ig = _nx_to_igraph(G)
            part = leidenalg.find_partition(
                G_ig,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=1.0,
                seed=RANDOM_SEED,
            )
            partition = _leiden_partition_to_dict(G_ig, part)
            modularity = _compute_modularity(G, partition)

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


def disparity_filter(
    G: nx.Graph,
    alpha: float = 0.05,
) -> nx.Graph:
    """Extract the network backbone using the disparity filter (Serrano et al., 2009).

    For each node, tests whether each edge weight is statistically significant
    given the node's degree. An edge is retained if it is significant for
    at least one of its endpoints (p < alpha under the null hypothesis of
    uniform weight distribution).

    Returns a new graph containing only the statistically significant edges.
    """
    backbone = nx.Graph()
    backbone.add_nodes_from(G.nodes(data=True))

    for u in G.nodes():
        k = G.degree(u)
        if k < 2:
            # Single-edge nodes: always retain
            for v, data in G[u].items():
                if not backbone.has_edge(u, v):
                    backbone.add_edge(u, v, **data)
            continue

        # Total weight for this node
        strength = sum(d["weight"] for _, _, d in G.edges(u, data=True))
        if strength == 0:
            continue

        for v, data in G[u].items():
            w = data["weight"]
            p_ij = w / strength  # normalized weight
            # Disparity filter p-value: (1 - p_ij)^(k-1)
            p_value = (1.0 - p_ij) ** (k - 1)
            if p_value < alpha:
                if not backbone.has_edge(u, v):
                    backbone.add_edge(u, v, **data)

    return backbone


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
    subtitle: str | None = None,
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

    # Compute betweenness for bridge highlighting and labeling
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    top_bridge_slugs = sorted(betweenness, key=betweenness.get, reverse=True)[:3]  # type: ignore[arg-type]

    # Node sizes proportional to degree
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [100 + 300 * degrees[n] / max_deg for n in nodes]

    # Draw edges first (light gray)
    edge_weights = [d.get("weight", 0.5) for _, _, d in G.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.3 + 1.5 * w / max_w for w in edge_weights]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=edge_widths, edge_color="#888888")

    # Separate bridge vs non-bridge nodes
    non_bridge = [n for n in nodes if n not in top_bridge_slugs]
    bridge_nodes = [n for n in nodes if n in top_bridge_slugs]

    # Draw non-bridge nodes
    if non_bridge:
        nb_colors = (
            [node_colors[nodes.index(n)] for n in non_bridge]
            if isinstance(node_colors, list)
            else node_colors
        )
        nb_sizes = [node_sizes[nodes.index(n)] for n in non_bridge]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=non_bridge,
            ax=ax,
            node_color=nb_colors,
            node_size=nb_sizes,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Draw bridge nodes with red halo
    if bridge_nodes:
        br_colors = (
            [node_colors[nodes.index(n)] for n in bridge_nodes]
            if isinstance(node_colors, list)
            else node_colors
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bridge_nodes,
            ax=ax,
            node_color=br_colors,
            node_size=140,
            alpha=0.9,
            edgecolors="red",
            linewidths=2.5,
        )
        # Label bridge legislators
        for slug in bridge_nodes:
            name = G.nodes[slug].get("full_name", slug).split()[-1]
            ax.annotate(
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

    # Label top-N by betweenness (skip those already labeled as bridges)
    if label_top_n > 0:
        top_nodes = sorted(betweenness, key=betweenness.get, reverse=True)[:label_top_n]  # type: ignore[arg-type]
        labels = {
            n: G.nodes[n].get("full_name", n).split()[-1]
            for n in top_nodes
            if n not in top_bridge_slugs
        }
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    if legend_elements:
        legend_elements.append(
            Patch(facecolor="white", edgecolor="red", linewidth=2, label="Bridge legislator"),
        )
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    ax.set_title(f"{chamber} — {title}", fontsize=14, fontweight="bold")
    ax.axis("off")

    if subtitle:
        fig.text(
            0.5,
            0.96,
            subtitle,
            ha="center",
            fontsize=10,
            fontstyle="italic",
            color="#555555",
        )

    fig.text(
        0.5,
        0.01,
        "Red-ringed nodes connect otherwise-separate voting blocs (highest betweenness centrality)",
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95 if subtitle else 1))

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

    # Quadrant labels (italic gray, using axes-relative coordinates)
    quadrant_labels = {
        (0.95, 0.95): "Central players \u2014\nhighly connected\nand influential",
        (0.05, 0.95): "Unique connectors \u2014\nlink separate blocs",
        (0.95, 0.05): "Well-connected\nfollowers",
        (0.05, 0.05): "Rank and file",
    }
    for (qx, qy), qlabel in quadrant_labels.items():
        ax.text(
            qx,
            qy,
            qlabel,
            transform=ax.transAxes,
            fontsize=7,
            fontstyle="italic",
            color="#999999",
            ha="right" if qx > 0.5 else "left",
            va="top" if qy > 0.5 else "bottom",
            alpha=0.7,
        )

    # Betweenness inset diagram (when y_col is betweenness)
    if y_col == "betweenness":
        inset = ax.inset_axes([0.72, 0.72, 0.22, 0.22])
        inset.set_xlim(-0.5, 2.5)
        inset.set_ylim(-0.5, 1.0)
        inset.set_aspect("equal")
        inset.axis("off")
        # Draw 3 nodes: A(0,0), B(1,0), C(2,0)
        for nx_pos, label, color in [(0, "A", "#cccccc"), (1, "B", "#E81B23"), (2, "C", "#cccccc")]:
            circle = plt.Circle((nx_pos, 0.2), 0.15, color=color, ec="black", lw=1, zorder=3)
            inset.add_patch(circle)
            inset.text(
                nx_pos,
                0.2,
                label,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                zorder=4,
            )
        # Edges
        inset.plot([0.15, 0.85], [0.2, 0.2], color="#555555", lw=1.5, zorder=2)
        inset.plot([1.15, 1.85], [0.2, 0.2], color="#555555", lw=1.5, zorder=2)
        inset.text(1, -0.3, "B bridges A and C", ha="center", fontsize=6, fontstyle="italic")
        inset.patch.set_alpha(0.9)
        inset.patch.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_visible(False)

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


def plot_centrality_ranking(
    centralities: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Horizontal bar chart of ALL legislators, sorted by betweenness, party-colored."""
    if centralities.height == 0:
        return

    sorted_df = centralities.sort("betweenness")

    names = sorted_df["full_name"].to_list()
    scores = sorted_df["betweenness"].to_numpy()
    parties = sorted_df["party"].to_list()
    colors = [PARTY_COLORS.get(p, "#888888") for p in parties]

    fig_height = max(8, len(names) * 0.18)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(names))
    ax.barh(y, scores, color=colors, alpha=0.8, edgecolor="white", height=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel(
        "Network Influence (Betweenness \u2014 higher = connects more groups)", fontsize=10
    )
    ax.set_title(
        f"{chamber} \u2014 Who Holds the Most Influence in the Voting Network?",
        fontsize=14,
        fontweight="bold",
    )

    # Annotate top and bottom 5 with last names + scores
    for i in range(min(5, len(names))):
        last_name = names[i].split()[-1]
        ax.annotate(
            f"{last_name}: {scores[i]:.4f}",
            (scores[i] + 0.001, y[i]),
            fontsize=7,
            va="center",
            fontweight="bold",
        )
    for i in range(max(0, len(names) - 5), len(names)):
        last_name = names[i].split()[-1]
        ax.annotate(
            f"{last_name}: {scores[i]:.4f}",
            (scores[i] + 0.001, y[i]),
            fontsize=7,
            va="center",
            fontweight="bold",
        )

    # Legend
    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"centrality_ranking_{chamber.lower()}.png")


def _community_label(comm_id: int, composition: pl.DataFrame | None) -> str:
    """Return a plain-English label for a community based on its party composition."""
    if composition is None:
        return f"Community {comm_id}"
    row = composition.filter(pl.col("community") == comm_id)
    if row.height == 0:
        return f"Community {comm_id}"
    pct_r = row["pct_republican"][0]
    n = row["n_legislators"][0]
    if pct_r >= 80:
        return f"Mostly Republican ({pct_r:.0f}%, n={n})"
    if pct_r <= 20:
        pct_d = 100 - pct_r
        return f"Mostly Democrat ({pct_d:.0f}% D, n={n})"
    return f"Mixed ({pct_r:.0f}% R, n={n})"


def plot_community_network(
    G: nx.Graph,
    partition: dict[str, int],
    chamber: str,
    resolution: float,
    out_path: Path,
    pos: dict | None = None,
    community_composition: pl.DataFrame | None = None,
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
        Patch(
            facecolor=cmap(i / max(n_communities - 1, 1)),
            label=_community_label(i, community_composition),
        )
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

    fig.suptitle(f"{chamber} — Multi-Resolution Leiden", fontsize=14, fontweight="bold")
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

    # Detect stability zone: contiguous range where n_components stays constant
    stable_start = None
    stable_end = None
    if len(components) >= 2:
        # Find longest run of constant n_components
        best_run_start = 0
        best_run_len = 1
        cur_start = 0
        cur_len = 1
        for i in range(1, len(components)):
            if components[i] == components[cur_start]:
                cur_len += 1
            else:
                if cur_len > best_run_len:
                    best_run_len = cur_len
                    best_run_start = cur_start
                cur_start = i
                cur_len = 1
        if cur_len > best_run_len:
            best_run_len = cur_len
            best_run_start = cur_start
        if best_run_len >= 2:
            stable_start = thresholds[best_run_start]
            stable_end = thresholds[best_run_start + best_run_len - 1]

    # Detect further fragmentation: first threshold where n_components > 2
    frag_threshold = None
    for i, c in enumerate(components):
        if c > 2:
            frag_threshold = thresholds[i]
            break

    metrics = [
        ("n_edges", "How Many Connections Remain?"),
        ("density", "How Dense Is the Network?"),
        ("n_components", "When Does the Network Split?"),
        ("modularity", "How Clustered Are the Groups?"),
    ]

    for ax, (col, label) in zip(axes.flatten(), metrics):
        vals = sweep_df[col].to_list()
        ax.plot(thresholds, vals, "o-", color="#333333", linewidth=2, markersize=6)
        ax.set_xlabel("Agreement Threshold (\u03ba)", fontsize=10)
        ax.set_ylabel(label.split("?")[0].split()[-1] if "?" in label else label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Stability zone shading (all panels)
        if stable_start is not None and stable_end is not None:
            ax.axvspan(stable_start, stable_end, alpha=0.12, color="green", zorder=0)
            if col == "n_components":
                ax.text(
                    (stable_start + stable_end) / 2,
                    min(vals) + (max(vals) - min(vals)) * 0.15,
                    "Findings stable\nacross this range",
                    ha="center",
                    fontsize=7,
                    fontstyle="italic",
                    color="green",
                    alpha=0.8,
                )

        # Default threshold line (all panels)
        ax.axvline(
            KAPPA_THRESHOLD_DEFAULT,
            color="#0015BC",
            linestyle="--",
            alpha=0.5,
            linewidth=1.2,
        )
        ax.annotate(
            f"Default (κ={KAPPA_THRESHOLD_DEFAULT:.2f})",
            (KAPPA_THRESHOLD_DEFAULT, max(vals) * 0.92),
            fontsize=6,
            color="#0015BC",
            ha="center",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.8},
        )

        # Party split threshold (all panels)
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
            else:
                ax.annotate(
                    "Party split",
                    (split_threshold, max(vals) * 0.7),
                    fontsize=6,
                    color="#E81B23",
                    ha="center",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "alpha": 0.8},
                )

        # Further fragmentation marker (all panels)
        if frag_threshold is not None:
            ax.axvline(
                frag_threshold,
                color="#FF8C00",
                linestyle=":",
                alpha=0.6,
                linewidth=1.2,
            )
            if col == "n_components":
                ax.annotate(
                    "Further\nfragmentation",
                    (frag_threshold, max(vals) * 0.45),
                    fontsize=7,
                    color="#FF8C00",
                    ha="center",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8},
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
        w = d.get("weight", 0.0)
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

    # Default threshold line
    ax.axvline(
        KAPPA_THRESHOLD_DEFAULT,
        color="#0015BC",
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"Threshold (\u03ba={KAPPA_THRESHOLD_DEFAULT:.2f})",
    )

    ax.set_xlabel("Kappa (Edge Weight)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"{chamber} \u2014 How Strong Are Voting Connections?", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Cross-party gap annotation
    if cross:
        max_cross = max(cross)
        ax.annotate(
            f"Strongest cross-party: \u03ba={max_cross:.3f}",
            (max_cross, 0),
            xytext=(max_cross, ax.get_ylim()[1] * 0.5),
            fontsize=8,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2},
            bbox={"boxstyle": "round,pad=0.3", "fc": "lightyellow", "alpha": 0.85, "ec": "#cccccc"},
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No cross-party edges at this threshold",
            transform=ax.transAxes,
            fontsize=10,
            fontstyle="italic",
            color="#999999",
            ha="center",
            va="center",
        )

    # Interpretation text box
    ax.text(
        0.97,
        0.95,
        "Higher \u03ba = more similar voting patterns.\n"
        "Cross-party connections (gray) cluster at\n"
        "lower weights \u2014 the two parties agree less\n"
        "than members within each party.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.5", "fc": "lightyellow", "alpha": 0.85, "ec": "#cccccc"},
    )

    save_fig(fig, out_path)


def find_cross_party_bridge(
    kappa_df: pl.DataFrame,
    ideal_points: pl.DataFrame,
    threshold: float = CROSS_PARTY_BRIDGE_THRESHOLD,
) -> tuple[nx.Graph, str | None]:
    """Find the legislator with the most cross-party edges at a low threshold.

    Returns (graph, slug_of_bridge) or (graph, None) if no cross-party edges exist.
    Pure logic — no matplotlib dependency.
    """
    G = build_kappa_network(kappa_df, ideal_points, threshold=threshold)

    # Count cross-party edges per legislator
    cross_party_counts: dict[str, int] = {}
    for u, v in G.edges():
        pu = G.nodes[u].get("party", "Unknown")
        pv = G.nodes[v].get("party", "Unknown")
        if pu != pv:
            cross_party_counts[u] = cross_party_counts.get(u, 0) + 1
            cross_party_counts[v] = cross_party_counts.get(v, 0) + 1

    if not cross_party_counts:
        return G, None

    bridge_slug = max(cross_party_counts, key=cross_party_counts.get)  # type: ignore[arg-type]
    return G, bridge_slug


def plot_cross_party_bridge(
    kappa_df: pl.DataFrame,
    ideal_points: pl.DataFrame,
    chamber: str,
    out_path: Path,
    threshold: float = CROSS_PARTY_BRIDGE_THRESHOLD,
) -> dict | None:
    """Plot before/after network showing the effect of removing the top cross-party bridge.

    Left panel: full network at the low threshold with bridge's cross-party edges highlighted.
    Right panel: same network with bridge removed, showing disconnection.
    Returns metadata dict or None if no cross-party bridge exists.
    """
    G, bridge_slug = find_cross_party_bridge(kappa_df, ideal_points, threshold)
    if bridge_slug is None:
        print(f"  No cross-party bridge at κ={threshold} for {chamber} — skipping")
        return None

    bridge_name = G.nodes[bridge_slug].get("full_name", bridge_slug)
    bridge_party = G.nodes[bridge_slug].get("party", "Unknown")

    # Count cross-party edges for this bridge
    cross_edges = []
    for neighbor in G.neighbors(bridge_slug):
        if G.nodes[neighbor].get("party", "Unknown") != bridge_party:
            cross_edges.append((bridge_slug, neighbor))

    # Check if removal disconnects
    n_components_before = nx.number_connected_components(G)
    G_removed = G.copy()
    G_removed.remove_node(bridge_slug)
    n_components_after = nx.number_connected_components(G_removed)
    removal_disconnects = n_components_after > n_components_before

    # Compute layout on full graph
    pos = _compute_layout(G)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    nodes = list(G.nodes())

    # ── Left panel: full network with bridge highlighted ──
    node_colors_left = [PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in nodes]

    # Draw all edges light
    nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.1, edge_color="#888888")

    # Draw bridge's cross-party edges thick and gold
    if cross_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=cross_edges,
            ax=ax1,
            width=3.0,
            edge_color="#FFD700",
            alpha=0.9,
        )

    # Draw non-bridge nodes
    non_bridge = [n for n in nodes if n != bridge_slug]
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=non_bridge,
        ax=ax1,
        node_color=[node_colors_left[nodes.index(n)] for n in non_bridge],
        node_size=80,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )

    # Draw bridge node with gold halo
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[bridge_slug],
        ax=ax1,
        node_color=[PARTY_COLORS.get(bridge_party, "#999999")],
        node_size=200,
        alpha=0.95,
        edgecolors="#FFD700",
        linewidths=3.0,
    )
    # Callout for bridge
    last_name = bridge_name.split()[-1]
    ax1.annotate(
        f"{last_name}\n({len(cross_edges)} cross-party edges)",
        pos[bridge_slug],
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="bottom",
        xytext=(0, 15),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FFD700", "alpha": 0.85, "ec": "#B8860B"},
        arrowprops={"arrowstyle": "->", "color": "#B8860B", "lw": 1.5},
    )

    ax1.set_title(
        f"{chamber} — With {last_name} (κ={threshold})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.axis("off")

    legend_left = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
        Patch(facecolor="#FFD700", edgecolor="#B8860B", linewidth=2, label="Cross-party bridge"),
    ]
    ax1.legend(handles=legend_left, loc="upper left", fontsize=9)

    # ── Right panel: network with bridge removed ──
    nodes_right = [n for n in nodes if n != bridge_slug]
    node_colors_right = [
        PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in nodes_right
    ]

    # Use same positions (minus the removed node)
    pos_right = {n: pos[n] for n in nodes_right}

    nx.draw_networkx_edges(G_removed, pos_right, ax=ax2, alpha=0.1, edge_color="#888888")
    nx.draw_networkx_nodes(
        G_removed,
        pos_right,
        nodelist=nodes_right,
        ax=ax2,
        node_color=node_colors_right,
        node_size=80,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )

    # Show where bridge was with a faint X
    ax2.plot(
        pos[bridge_slug][0],
        pos[bridge_slug][1],
        "x",
        color="#999999",
        markersize=15,
        markeredgewidth=2,
        alpha=0.5,
    )

    if removal_disconnects:
        status = f"Network splits into {n_components_after} groups"
    else:
        status = "Network remains connected (alternative paths exist)"

    ax2.set_title(
        f"{chamber} — Without {last_name}\n({status})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.axis("off")

    fig.suptitle(
        "What Happens When You Remove the Cross-Party Bridge?",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        f"At κ={threshold}, {bridge_name} is the top cross-party connector. "
        + (
            "Remove them and the network splits."
            if removal_disconnects
            else "Alternative connections prevent a full split."
        ),
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    save_fig(fig, out_path)

    return {
        "bridge_slug": bridge_slug,
        "bridge_name": bridge_name,
        "n_cross_party_edges": len(cross_edges),
        "removal_disconnects": removal_disconnects,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = ks.data_dir

    results_root = ks.results_dir

    eda_dir = resolve_upstream_dir(
        "01_eda",
        results_root,
        args.run_id,
        Path(args.eda_dir) if args.eda_dir else None,
    )
    irt_dir = resolve_upstream_dir(
        "05_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )
    clustering_dir = resolve_upstream_dir(
        "09_clustering",
        results_root,
        args.run_id,
        Path(args.clustering_dir) if args.clustering_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="11_network",
        params=vars(args),
        primer=NETWORK_PRIMER,
        run_id=args.run_id,
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

        # Check if we have enough upstream data
        if kappa_house is None and kappa_senate is None:
            print("Phase 06 (Network): skipping — no EDA agreement matrices available")
            return
        if irt_house is None and irt_senate is None:
            print("Phase 06 (Network): skipping — no IRT ideal points available")
            return

        # Load equated ideal points (may not exist)
        equated_ip = None
        eq_path = irt_dir / "data" / "ideal_points_joint_equated.parquet"
        if eq_path.exists():
            equated_ip = load_equated_ideal_points(irt_dir)
            print(f"  Equated ideal points: {equated_ip.height} legislators")
        else:
            print("  Equated ideal points: not found (cross-chamber will be skipped)")
            args.skip_cross_chamber = True

        for label, df in [("Kappa House", kappa_house), ("Kappa Senate", kappa_senate)]:
            info = f"{df.height} x {len(df.columns) - 1}" if df is not None else "not available"
            print(f"  {label}:  {info}")
        for label, df in [("IRT House", irt_house), ("IRT Senate", irt_senate)]:
            info = f"{df.height} legislators" if df is not None else "not available"
            print(f"  {label}:    {info}")
        for label, df in [("Vote matrix House", vm_house), ("Vote matrix Senate", vm_senate)]:
            info = f"{df.height} x {len(df.columns) - 1}" if df is not None else "not available"
            print(f"  {label}:  {info}")
        for label, df in [("Bill params House", bp_house), ("Bill params Senate", bp_senate)]:
            info = f"{df.height}" if df is not None else "not available"
            print(f"  {label}:  {info}")
        print(f"  Rollcalls: {rollcalls.height}")
        print(f"  Legislators: {_legislators.height}")

        chamber_configs = [
            ("House", kappa_house, irt_house, vm_house, bp_house, clust_house, loy_house),
            ("Senate", kappa_senate, irt_senate, vm_senate, bp_senate, clust_senate, loy_senate),
        ]

        results: dict[str, dict] = {}

        for chamber, kappa, irt_ip, vm, bp, clust, loy in chamber_configs:
            if irt_ip is None or irt_ip.height < 5:
                n = irt_ip.height if irt_ip is not None else 0
                print(f"\n  Skipping {chamber}: too few legislators ({n})")
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

            # Party modularity (polarization metric, Waugh et al. 2009)
            party_mod = compute_party_modularity(G)
            chamber_results["party_modularity"] = party_mod
            print(f"  Party modularity (polarization): {party_mod}")

            # ── Phase 3: Centrality ──
            print_header(f"PHASE 3: CENTRALITY MEASURES — {chamber}")
            centralities = compute_centralities(G)
            centralities.write_parquet(ctx.data_dir / f"centrality_{chamber.lower()}.parquet")
            ctx.export_csv(
                centralities,
                f"centrality_{chamber.lower()}.csv",
                f"Network centrality measures for {chamber} legislators",
            )
            print(f"  Saved: centrality_{chamber.lower()}.parquet")
            chamber_results["centralities"] = centralities

            bridges = identify_bridge_legislators(centralities, G)
            bridges.write_parquet(ctx.data_dir / f"bridge_legislators_{chamber.lower()}.parquet")
            ctx.export_csv(
                bridges,
                f"bridge_legislators_{chamber.lower()}.csv",
                f"Cross-party bridge legislators in {chamber}",
            )
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
                ctx.export_csv(
                    comm_df,
                    f"communities_{chamber.lower()}.csv",
                    f"Network community assignments for {chamber} legislators",
                )
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

                # CPM sweep (resolution-limit-free)
                print("  CPM sweep (resolution-limit-free):")
                cpm_partitions, cpm_df = detect_communities_cpm(G)
                cpm_df.write_parquet(ctx.data_dir / f"cpm_resolution_{chamber.lower()}.parquet")
                print(f"  Saved: cpm_resolution_{chamber.lower()}.parquet")
                chamber_results["cpm_partitions"] = cpm_partitions
                chamber_results["cpm_df"] = cpm_df
            else:
                print("  No edges — skipping community detection")
                chamber_results["best_partition"] = {}
                chamber_results["resolution_df"] = pl.DataFrame()
                chamber_results["cpm_partitions"] = {}
                chamber_results["cpm_df"] = pl.DataFrame()

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

            # Extreme edge weight analysis (data-driven)
            extreme_result = check_extreme_edge_weights(G, chamber)
            chamber_results["extreme_edge_weights"] = extreme_result
            if extreme_result:
                maj = extreme_result["majority_party"]
                print(f"  {maj} median edge weight: {extreme_result['r_median_edge_weight']}")
                for leg in extreme_result.get("legislators", []):
                    print(
                        f"    {leg['name']} (xi={leg['xi_mean']:+.3f}): "
                        f"mean {maj[0]}-{maj[0]} weight={leg['mean_r_weight']:.4f} "
                        f"(vs median: {leg['vs_r_median']:+.4f})"
                    )

            # ── Phase 6: Threshold Sensitivity ──
            print_header(f"PHASE 6: THRESHOLD SENSITIVITY — {chamber}")
            sweep = run_threshold_sweep(kappa, irt_ip)
            sweep.write_parquet(ctx.data_dir / f"threshold_sweep_{chamber.lower()}.parquet")
            print(f"  Saved: threshold_sweep_{chamber.lower()}.parquet")
            chamber_results["threshold_sweep"] = sweep

            # Backbone extraction (disparity filter, Serrano et al. 2009)
            if G.number_of_edges() > 0:
                backbone = disparity_filter(G, alpha=0.05)
                backbone_summary = compute_network_summary(backbone)
                chamber_results["backbone"] = backbone
                chamber_results["backbone_summary"] = backbone_summary
                pct_retained = backbone.number_of_edges() / G.number_of_edges() * 100
                print(
                    f"  Backbone (α=0.05): {backbone.number_of_edges()} / "
                    f"{G.number_of_edges()} edges retained ({pct_retained:.1f}%)"
                )
            else:
                chamber_results["backbone"] = None
                chamber_results["backbone_summary"] = None

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
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            pos = plot_network_layout(
                G,
                "party",
                chamber,
                f"Who Votes With Whom? ({n_nodes} legislators, {n_edges} connections)",
                ctx.plots_dir / f"network_party_{chamber.lower()}.png",
            )

            # Interactive PyVis network
            generate_pyvis_network(G, chamber, ctx.plots_dir)

            # Network layout (IRT gradient)
            plot_network_layout(
                G,
                "xi_mean",
                chamber,
                "Ideology Gradient on the Same Network",
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

            # Centrality ranking (betweenness bar chart)
            plot_centrality_ranking(centralities, chamber, ctx.plots_dir)

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
                    community_composition=chamber_results.get("community_composition"),
                )

            # Threshold sweep
            plot_threshold_sweep(
                sweep,
                chamber,
                ctx.plots_dir / f"threshold_sweep_{chamber.lower()}.png",
            )

            # Cross-party bridge before/after
            bridge_result = plot_cross_party_bridge(
                kappa,
                irt_ip,
                chamber,
                ctx.plots_dir / f"cross_party_bridge_{chamber.lower()}.png",
            )
            if bridge_result:
                chamber_results["cross_party_bridge"] = bridge_result

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
                "LEIDEN_RESOLUTIONS": LEIDEN_RESOLUTIONS,
                "CPM_GAMMAS": CPM_GAMMAS,
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
            manifest[f"{ch}_party_modularity"] = r.get("party_modularity")
            if r.get("backbone_summary"):
                bs = r["backbone_summary"]
                manifest[f"{ch}_backbone_edges"] = bs["n_edges"]
                manifest[f"{ch}_backbone_density"] = bs["density"]
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
            if r.get("extreme_edge_weights"):
                ee = r["extreme_edge_weights"]
                manifest[f"{ch}_extreme_edge_weights"] = {
                    k: v for k, v in ee.items() if k != "legislators"
                }
                if ee.get("legislators"):
                    manifest[f"{ch}_extreme_edge_weights"]["legislators"] = ee["legislators"]
            if r.get("cross_party_bridge"):
                manifest[f"{ch}_cross_party_bridge"] = r["cross_party_bridge"]

        if not results:
            print("Phase 06 (Network): skipping — no chambers had sufficient data")
            return

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
