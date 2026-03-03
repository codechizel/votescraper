"""
Kansas Legislature — Bipartite Bill-Legislator Network Analysis (Phase 6b)

Preserves the two-mode (bipartite) structure of legislators × bills to enable
bill-centric analysis — bill polarization scores, bridge bills, bill clustering,
and statistically validated backbone extraction via the Bipartite Configuration
Model (BiCM). Companion to Phase 6 (Network), which builds a legislator-only
co-voting network weighted by Cohen's Kappa.

Key contributions:
  1. Bill polarization: |pct_R_yea − pct_D_yea| per bill
  2. Bridge bills: high bipartite betweenness + cross-party support
  3. Bill clustering: Leiden on Newman-weighted bill projection
  4. BiCM backbone: maximum-entropy null model with analytical p-values
  5. Phase 6 comparison: backbone Jaccard, community NMI/ARI

Library: BiCM (Python, MIT, PyPI, Saracco et al. 2015/2017).

Usage:
  uv run python analysis/06b_network_bipartite/bipartite.py [--session 2025-26]
      [--skip-phase6-comparison] [--run-id RUN_ID]

Outputs (in results/<session>/<run_id>/06b_network_bipartite/):
  - data/:   Parquet files (polarization, bridge bills, backbone, communities)
  - plots/:  PNG visualizations
  - filtering_manifest.json, run_info.json, run_log.txt
  - bipartite_report.html
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
from bicm import BipartiteGraph
from matplotlib.patches import Patch
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.bipartite_report import build_bipartite_report
except ModuleNotFoundError:
    from bipartite_report import build_bipartite_report  # type: ignore[no-redef]

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ImportError:
    from phase_utils import load_metadata, print_header, save_fig

# ── Constants ────────────────────────────────────────────────────────────────

BICM_SIGNIFICANCE = 0.01
"""P-value threshold for BiCM backbone extraction — House (conservative for dense matrix)."""

BICM_SIGNIFICANCE_SENATE = 0.05
"""P-value threshold for Senate backbone — relaxed due to ~10x fewer legislators
and correspondingly fewer multiple comparisons."""

BILL_POLARIZATION_MIN_VOTERS = 10
"""Minimum Yea+Nay votes for a bill to receive a polarization score."""

NEWMAN_PROJECTION = True
"""Use Newman 1/(N_p−1) weighting for bill-bill projection (default: True)."""

BILL_CLUSTER_RESOLUTIONS = [0.5, 1.0, 1.5, 2.0, 3.0]
"""Leiden resolution parameters for bill projection community detection."""

TOP_BRIDGE_BILLS = 20
"""Number of bridge bills to show in the report table."""

BACKBONE_COMPARISON_THRESHOLD = 0.40
"""Kappa threshold for Phase 6 comparison (matches Phase 6 default)."""

RANDOM_SEED = 42
"""Reproducibility seed for all stochastic operations."""

PARTY_COLORS = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}

COMMUNITY_CMAP = "Set2"
"""Colormap for bill community assignments."""

# ── Primer ───────────────────────────────────────────────────────────────────

BIPARTITE_PRIMER = """\
# Bipartite Bill-Legislator Network Analysis (Phase 6b)

## Purpose

Preserves the two-mode (bipartite) structure of legislators × bills to answer
**bill-centric** questions that Phase 6's legislator-only network cannot:
Which bills are most polarizing? Which create unexpected cross-party coalitions?
Do bills cluster into issue-area blocs by coalition support?

Also extracts a statistically validated legislator backbone using the Bipartite
Configuration Model (BiCM) — a maximum-entropy null model that analytically
computes p-values for co-voting strength. This provides a principled alternative
to Phase 6's Kappa thresholding + disparity filter.

## Method

### Bipartite Graph Construction
- **Two node types:** legislators (with party, ideal point) and bills (with
  vote_id, bill_number, discrimination)
- **Edges:** legislator voted Yea on bill → edge. Nay and absent → no edge.
- One graph per chamber (separate House/Senate analysis).

### Bill Polarization
For each bill with ≥10 Yea+Nay votes:
    polarization = |pct_R_yea − pct_D_yea|
Range [0, 1]: 0 = identical party support, 1 = perfect party-line vote.

### Bridge Bills
Bills with high bipartite betweenness centrality AND cross-party support.
Cross-referenced with IRT discrimination (beta) — bridge bills should have
low |beta| (easy votes attract bipartisan support).

### Bill Projection (Newman-Weighted)
Projects the bipartite graph onto the bill mode:
    w(b1, b2) = Σ_legislator [1/(k_legislator − 1)]
where k is the legislator's degree (number of Yea votes). Discounting by
1/(k−1) prevents high-activity legislators from dominating.

### Bill Communities (Leiden)
Leiden community detection on the bill projection at multiple resolutions.
Bills clustered by coalition support — not by topic, but by which legislators
vote Yea on them together.

### BiCM Backbone Extraction
The Bipartite Configuration Model (Saracco et al. 2015/2017) fits a maximum-
entropy null model preserving each node's degree. Projected co-voting weights
are tested against this null — only edges with p < 0.01 are retained. The
resulting backbone graph is the statistically validated legislator network.

### Phase 6 Comparison (Soft Dependency)
If Phase 6 results are available, compares:
- Edge Jaccard overlap between BiCM backbone and Phase 6 backbone
- Community agreement (NMI, ARI)
- "Hidden alliances": edges present in BiCM backbone but absent from Phase 6

## Inputs

Reads from `results/<session>/<run_id>/01_eda/data/`:
- `vote_matrix_house_filtered.parquet` — Filtered binary vote matrices
- `vote_matrix_senate_filtered.parquet`

Reads from `results/<session>/<run_id>/04_irt/data/`:
- `ideal_points_house.parquet` — IRT ideal points
- `ideal_points_senate.parquet`
- `bill_params_house.parquet` — Bill discrimination parameters
- `bill_params_senate.parquet`

Reads from `data/kansas/{legislature}_{start}-{end}/`:
- `{name}_rollcalls.csv` — Roll call metadata
- `{name}_legislators.csv` — Legislator metadata

## Outputs

All outputs land in `results/<session>/<run_id>/06b_network_bipartite/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `bill_polarization_{chamber}.parquet` | Per-bill polarization scores |
| `bridge_bills_{chamber}.parquet` | Top bridge bills by betweenness |
| `bill_communities_{chamber}.parquet` | Bill community assignments |
| `bill_community_sweep_{chamber}.parquet` | Resolution sweep results |
| `backbone_edges_{chamber}.parquet` | BiCM backbone edge list |
| `backbone_centrality_{chamber}.parquet` | Centrality on backbone graph |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `degree_dist_{chamber}.png` | Bipartite degree distributions |
| `polarization_hist_{chamber}.png` | Bill polarization distribution |
| `bridge_vs_beta_{chamber}.png` | Bridge betweenness vs IRT discrimination |
| `bill_cluster_heatmap_{chamber}.png` | Party support profile per bill cluster |
| `backbone_layout_{chamber}.png` | BiCM backbone network layout |
| `backbone_comparison_{chamber}.png` | BiCM vs Phase 6 backbone |
| `bipartite_layout_{chamber}.png` | Small bipartite graph (bridge bills) |

## Interpretation Guide

- **High polarization (>0.8):** Strict party-line vote — every R voted one way,
  every D the opposite. Common in Kansas (Republican supermajority).
- **Low polarization (<0.2):** Bipartisan or unanimous — both parties support/oppose.
- **Bridge bills:** Low polarization + high betweenness = connecting otherwise
  separate partisan blocs. Often procedural or bipartisan policy.
- **Bill clusters:** Group by coalition, not topic. Two clusters in the same
  policy area may reflect different partisan coalitions.
- **BiCM backbone:** Edges statistically significant beyond what degree
  distributions alone would predict. Denser than disparity filter but
  with statistical guarantees.

## Caveats

1. **Yea-only edges:** Only Yea votes create edges. Nay agreement (both voting
   against) is invisible in the bipartite structure. Phase 6's Kappa network
   captures Nay agreement; this phase does not.
2. **BiCM assumes independence:** The null model preserves marginal degrees but
   assumes edges are otherwise independent. Violated when party membership
   induces correlated voting — BiCM will flag all within-party edges as
   significant, which is expected but not informative.
3. **Newman projection discounts high-activity legislators:** A legislator who
   votes Yea on everything contributes little weight to bill-bill edges. This
   is usually desirable but may undercount genuine bridge behavior from
   bipartisan moderates who vote Yea frequently.
4. **Bill communities are NOT topic-based.** Leiden on the bill projection
   groups bills by who voted for them, not by subject matter.
"""

# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KS Legislature Bipartite Network Analysis (Phase 6b)"
    )
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument("--network-dir", default=None, help="Override Phase 6 network results")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--skip-phase6-comparison",
        action="store_true",
        help="Skip Phase 6 backbone comparison",
    )
    return parser.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


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


def load_vote_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load filtered binary vote matrices from EDA."""
    return _load_pair(eda_dir, "vote_matrix_{ch}_filtered.parquet")


def load_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT ideal points for both chambers."""
    return _load_pair(irt_dir, "ideal_points_{ch}.parquet")


def load_bill_params(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load bill discrimination parameters from IRT."""
    return _load_pair(irt_dir, "bill_params_{ch}.parquet")


# ── Phase 2: Bipartite Graph Construction ─────────────────────────────────


def build_bipartite_graph(
    vote_matrix: pl.DataFrame,
    ideal_points: pl.DataFrame,
    bill_params: pl.DataFrame | None = None,
    rollcalls: pl.DataFrame | None = None,
) -> nx.Graph:
    """Build a bipartite graph from the vote matrix.

    Legislator nodes have: party, xi_mean, full_name, bipartite=0
    Bill nodes have: vote_id, bipartite=1, beta_mean (if available)
    Edges: legislator voted Yea on bill.

    Args:
        vote_matrix: Polars DataFrame with legislator_slug as first column,
            remaining columns are vote_ids with values 1/0/null.
        ideal_points: IRT ideal points (legislator_slug, xi_mean, party, full_name).
        bill_params: Optional bill parameters (vote_id, beta_mean, bill_number).
        rollcalls: Optional rollcall metadata (vote_id, bill_number, short_title).

    Returns:
        NetworkX Graph with bipartite node attribute.
    """
    B = nx.Graph()
    slug_col = vote_matrix.columns[0]
    slugs = vote_matrix[slug_col].to_list()
    vote_ids = [c for c in vote_matrix.columns if c != slug_col]

    # Build lookup dicts
    ip_dict: dict[str, dict] = {}
    for row in ideal_points.iter_rows(named=True):
        ip_dict[row["legislator_slug"]] = {
            "xi_mean": row.get("xi_mean", 0.0),
            "party": row.get("party", ""),
            "full_name": row.get("full_name", row["legislator_slug"]),
        }

    bp_dict: dict[str, dict] = {}
    if bill_params is not None:
        beta_col = "beta_mean" if "beta_mean" in bill_params.columns else "discrimination_mean"
        for row in bill_params.iter_rows(named=True):
            bp_dict[row["vote_id"]] = {
                "beta_mean": row.get(beta_col, 0.0),
                "bill_number": row.get("bill_number", ""),
            }

    rc_dict: dict[str, dict] = {}
    if rollcalls is not None:
        for row in rollcalls.iter_rows(named=True):
            rc_dict[row["vote_id"]] = {
                "short_title": row.get("short_title", ""),
                "bill_number": row.get("bill_number", ""),
                "motion": row.get("motion", ""),
            }

    # Add legislator nodes (bipartite=0)
    for slug in slugs:
        attrs = ip_dict.get(slug, {"xi_mean": 0.0, "party": "", "full_name": slug})
        B.add_node(
            slug,
            bipartite=0,
            node_type="legislator",
            party=attrs.get("party", ""),
            xi_mean=attrs.get("xi_mean", 0.0),
            full_name=attrs.get("full_name", slug),
        )

    # Add bill nodes (bipartite=1)
    for vid in vote_ids:
        bill_attrs = bp_dict.get(vid, {})
        rc_attrs = rc_dict.get(vid, {})
        B.add_node(
            vid,
            bipartite=1,
            node_type="bill",
            beta_mean=bill_attrs.get("beta_mean", 0.0),
            bill_number=rc_attrs.get("bill_number", bill_attrs.get("bill_number", "")),
            short_title=rc_attrs.get("short_title", ""),
        )

    # Add edges for Yea votes
    vote_arr = vote_matrix.select(vote_ids).to_numpy().astype(float)
    for i, slug in enumerate(slugs):
        for j, vid in enumerate(vote_ids):
            if vote_arr[i, j] == 1.0:
                B.add_edge(slug, vid)

    return B


def compute_bipartite_summary(B: nx.Graph) -> dict:
    """Compute summary statistics for a bipartite graph."""
    legislators = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}
    bills = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}

    n_leg = len(legislators)
    n_bills = len(bills)
    n_edges = B.number_of_edges()
    max_edges = n_leg * n_bills
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Degree stats
    leg_degrees = [B.degree(n) for n in legislators]
    bill_degrees = [B.degree(n) for n in bills]

    return {
        "n_legislators": n_leg,
        "n_bills": n_bills,
        "n_edges": n_edges,
        "density": round(density, 4),
        "avg_legislator_degree": round(np.mean(leg_degrees), 1) if leg_degrees else 0.0,
        "avg_bill_degree": round(np.mean(bill_degrees), 1) if bill_degrees else 0.0,
        "max_legislator_degree": max(leg_degrees) if leg_degrees else 0,
        "max_bill_degree": max(bill_degrees) if bill_degrees else 0,
    }


# ── Phase 3: Bill Polarization ──────────────────────────────────────────────


def compute_bill_polarization(
    vote_matrix: pl.DataFrame,
    ideal_points: pl.DataFrame,
    rollcalls: pl.DataFrame | None = None,
    bill_params: pl.DataFrame | None = None,
    min_voters: int = BILL_POLARIZATION_MIN_VOTERS,
) -> pl.DataFrame:
    """Compute polarization score for each bill.

    polarization = |pct_R_yea − pct_D_yea|

    Args:
        vote_matrix: Binary vote matrix (legislator_slug as first column).
        ideal_points: IRT ideal points with party column.
        rollcalls: Optional rollcall metadata for bill_number/title.
        bill_params: Optional IRT bill params for beta_mean.
        min_voters: Minimum Yea+Nay votes for inclusion.

    Returns:
        Polars DataFrame with columns: vote_id, polarization, pct_r_yea,
        pct_d_yea, n_r, n_d, bill_number, short_title, beta_mean.
    """
    slug_col = vote_matrix.columns[0]
    vote_ids = [c for c in vote_matrix.columns if c != slug_col]
    slugs = vote_matrix[slug_col].to_list()

    # Party lookup
    party_map: dict[str, str] = {}
    for row in ideal_points.iter_rows(named=True):
        party_map[row["legislator_slug"]] = row.get("party", "")

    # Bill params lookup
    bp_dict: dict[str, dict] = {}
    if bill_params is not None:
        beta_col = "beta_mean" if "beta_mean" in bill_params.columns else "discrimination_mean"
        for row in bill_params.iter_rows(named=True):
            bp_dict[row["vote_id"]] = {
                "beta_mean": row.get(beta_col, 0.0),
                "bill_number": row.get("bill_number", ""),
            }

    # Rollcall lookup
    rc_dict: dict[str, dict] = {}
    if rollcalls is not None:
        for row in rollcalls.iter_rows(named=True):
            rc_dict[row["vote_id"]] = {
                "bill_number": row.get("bill_number", ""),
                "short_title": row.get("short_title", ""),
            }

    vote_arr = vote_matrix.select(vote_ids).to_numpy().astype(float)

    # Vectorized party-based tallies using integer masks
    parties = np.array([party_map.get(s, "") for s in slugs])
    is_r = (parties == "Republican").astype(np.int8)
    is_d = (parties == "Democrat").astype(np.int8)

    # voted = not NaN; yea = 1.0; cast to int for arithmetic dot product
    voted = (~np.isnan(vote_arr)).astype(np.int8)
    yea = (np.nan_to_num(vote_arr, nan=0.0) == 1.0).astype(np.int8)

    # Per-bill tallies via dot products (sum along legislator axis)
    r_total = is_r @ voted  # shape (n_bills,)
    r_yea = is_r @ (voted & yea)
    d_total = is_d @ voted
    d_yea = is_d @ (voted & yea)

    total_voters = r_total + d_total
    mask = total_voters >= min_voters

    # Safe division (filtered below, so zeros don't matter)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_r = np.where(r_total > 0, r_yea / r_total, 0.0)
        pct_d = np.where(d_total > 0, d_yea / d_total, 0.0)
    polarization = np.abs(pct_r - pct_d)

    rows: list[dict] = []
    for j in np.where(mask)[0]:
        vid = vote_ids[j]
        bp = bp_dict.get(vid, {})
        rc = rc_dict.get(vid, {})
        bill_number = rc.get("bill_number", bp.get("bill_number", ""))

        rows.append(
            {
                "vote_id": vid,
                "polarization": round(float(polarization[j]), 4),
                "pct_r_yea": round(float(pct_r[j]), 4),
                "pct_d_yea": round(float(pct_d[j]), 4),
                "n_r": int(r_total[j]),
                "n_d": int(d_total[j]),
                "bill_number": bill_number,
                "short_title": rc.get("short_title", ""),
                "beta_mean": bp.get("beta_mean", None),
            }
        )

    df = pl.DataFrame(rows)
    if df.height > 0:
        df = df.sort("polarization", descending=True)
    return df


# ── Phase 4: Bipartite Betweenness & Bridge Bills ──────────────────────────


def compute_bipartite_betweenness(B: nx.Graph) -> dict[str, float]:
    """Compute bipartite betweenness centrality for all nodes.

    Uses networkx.algorithms.bipartite.centrality.betweenness_centrality.

    Returns:
        Dict mapping node name to betweenness centrality value.
    """
    legislators = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}
    return nx.bipartite.betweenness_centrality(B, legislators)


def identify_bridge_bills(
    B: nx.Graph,
    betweenness: dict[str, float],
    polarization_df: pl.DataFrame,
    bill_params: pl.DataFrame | None = None,
    top_n: int = TOP_BRIDGE_BILLS,
) -> pl.DataFrame:
    """Identify bridge bills: high betweenness + low polarization.

    Bridge bills are bills that connect otherwise separate partisan blocs
    in the bipartite graph. They tend to have low polarization (bipartisan
    support) and high betweenness (connecting the two sides).

    Returns:
        Polars DataFrame sorted by betweenness (descending), top_n rows.
    """
    bills = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 1}
    bill_betweenness = {n: betweenness.get(n, 0.0) for n in bills}

    # Merge with polarization
    pol_dict: dict[str, dict] = {}
    if polarization_df.height > 0:
        for row in polarization_df.iter_rows(named=True):
            pol_dict[row["vote_id"]] = {
                "polarization": row["polarization"],
                "bill_number": row.get("bill_number", ""),
                "short_title": row.get("short_title", ""),
                "pct_r_yea": row.get("pct_r_yea", 0.0),
                "pct_d_yea": row.get("pct_d_yea", 0.0),
            }

    # Bill params lookup
    bp_dict: dict[str, float] = {}
    if bill_params is not None:
        beta_col = "beta_mean" if "beta_mean" in bill_params.columns else "discrimination_mean"
        for row in bill_params.iter_rows(named=True):
            bp_dict[row["vote_id"]] = row.get(beta_col, 0.0)

    rows: list[dict] = []
    for vid, btwn in bill_betweenness.items():
        pol = pol_dict.get(vid, {})
        rows.append(
            {
                "vote_id": vid,
                "betweenness": round(btwn, 6),
                "polarization": pol.get("polarization", None),
                "bill_number": pol.get("bill_number", B.nodes[vid].get("bill_number", "")),
                "short_title": pol.get("short_title", B.nodes[vid].get("short_title", "")),
                "pct_r_yea": pol.get("pct_r_yea", None),
                "pct_d_yea": pol.get("pct_d_yea", None),
                "beta_mean": bp_dict.get(vid, None),
                "degree": B.degree(vid),
            }
        )

    df = pl.DataFrame(rows)
    if df.height > 0:
        df = df.sort("betweenness", descending=True).head(top_n)
    return df


# ── Phase 5: Bill Projection ───────────────────────────────────────────────


def build_bill_projection(
    vote_matrix: pl.DataFrame,
    use_newman: bool = NEWMAN_PROJECTION,
) -> nx.Graph:
    """Project bipartite graph onto bill mode (bill-bill co-voting).

    Newman weighting: w(b1, b2) = Σ_legislator [1/(k_i − 1)]
    where k_i = number of Yea votes for legislator i.

    Args:
        vote_matrix: Binary vote matrix (legislator_slug first column).
        use_newman: Use Newman 1/(k-1) discount (default True).

    Returns:
        Weighted undirected graph on bill nodes.
    """
    slug_col = vote_matrix.columns[0]
    vote_ids = [c for c in vote_matrix.columns if c != slug_col]
    vote_arr = vote_matrix.select(vote_ids).to_numpy().astype(float)

    # Replace NaN with 0 for matrix multiplication
    binary = np.nan_to_num(vote_arr, nan=0.0)

    n_legs, n_bills = binary.shape

    if use_newman:
        # Newman weighting: each legislator contributes 1/(k-1)
        k = binary.sum(axis=1)  # degree per legislator
        weights = np.zeros(n_legs)
        for i in range(n_legs):
            if k[i] > 1:
                weights[i] = 1.0 / (k[i] - 1)
        # Weighted projection: B^T diag(weights) B
        proj = binary.T @ np.diag(weights) @ binary
    else:
        # Simple projection: B^T B
        proj = binary.T @ binary

    # Build graph
    G = nx.Graph()
    for j, vid in enumerate(vote_ids):
        G.add_node(vid)

    for j1 in range(n_bills):
        for j2 in range(j1 + 1, n_bills):
            w = proj[j1, j2]
            if w > 0:
                G.add_edge(vote_ids[j1], vote_ids[j2], weight=w)

    return G


# ── Phase 6: Bill Community Detection ──────────────────────────────────────


def detect_bill_communities(
    G_bills: nx.Graph,
    resolutions: list[float] = BILL_CLUSTER_RESOLUTIONS,
) -> tuple[dict[float, dict[str, int]], pl.DataFrame]:
    """Run Leiden modularity optimization on the bill projection at multiple resolutions.

    Returns:
        partitions: dict mapping resolution -> {bill_node: community_id}
        sweep_df: Polars DataFrame with resolution sweep results
    """
    if G_bills.number_of_edges() == 0:
        return {}, pl.DataFrame({"resolution": [], "n_communities": [], "modularity": []})

    # Convert to igraph
    ig_graph = ig.Graph.from_networkx(G_bills)

    partitions: dict[float, dict[str, int]] = {}
    sweep_rows: list[dict] = []

    for res in resolutions:
        part = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=res,
            weights="weight",
            seed=RANDOM_SEED,
        )

        # Map back to node names
        node_names = [ig_graph.vs[i]["_nx_name"] for i in range(ig_graph.vcount())]
        mapping = {node_names[i]: part.membership[i] for i in range(len(node_names))}
        partitions[res] = mapping

        sweep_rows.append(
            {
                "resolution": res,
                "n_communities": len(set(part.membership)),
                "modularity": round(part.modularity, 4),
            }
        )

    sweep_df = pl.DataFrame(sweep_rows)
    return partitions, sweep_df


def analyze_bill_community_profiles(
    partition: dict[str, int],
    vote_matrix: pl.DataFrame,
    ideal_points: pl.DataFrame,
) -> pl.DataFrame:
    """Compute party support profile for each bill community.

    For each community: mean pct_R_yea, mean pct_D_yea, community size.

    Returns:
        Polars DataFrame with columns: community, n_bills, mean_pct_r_yea,
        mean_pct_d_yea, mean_polarization.
    """
    slug_col = vote_matrix.columns[0]
    vote_ids = [c for c in vote_matrix.columns if c != slug_col]
    slugs = vote_matrix[slug_col].to_list()
    vote_arr = vote_matrix.select(vote_ids).to_numpy().astype(float)

    # Party lookup
    party_map: dict[str, str] = {}
    for row in ideal_points.iter_rows(named=True):
        party_map[row["legislator_slug"]] = row.get("party", "")

    r_indices = [i for i, s in enumerate(slugs) if party_map.get(s) == "Republican"]
    d_indices = [i for i, s in enumerate(slugs) if party_map.get(s) == "Democrat"]

    vid_to_col = {vid: j for j, vid in enumerate(vote_ids)}

    community_ids = sorted(set(partition.values()))
    rows: list[dict] = []

    for cid in community_ids:
        bills_in_comm = [b for b, c in partition.items() if c == cid and b in vid_to_col]
        if not bills_in_comm:
            continue

        pct_r_vals = []
        pct_d_vals = []
        for vid in bills_in_comm:
            j = vid_to_col[vid]
            r_votes = vote_arr[r_indices, j]
            d_votes = vote_arr[d_indices, j]
            r_valid = r_votes[~np.isnan(r_votes)]
            d_valid = d_votes[~np.isnan(d_votes)]
            pct_r = float(np.mean(r_valid)) if len(r_valid) > 0 else 0.0
            pct_d = float(np.mean(d_valid)) if len(d_valid) > 0 else 0.0
            pct_r_vals.append(pct_r)
            pct_d_vals.append(pct_d)

        mean_r = float(np.mean(pct_r_vals))
        mean_d = float(np.mean(pct_d_vals))
        mean_pol = float(np.mean([abs(r - d) for r, d in zip(pct_r_vals, pct_d_vals)]))

        rows.append(
            {
                "community": cid,
                "n_bills": len(bills_in_comm),
                "mean_pct_r_yea": round(mean_r, 4),
                "mean_pct_d_yea": round(mean_d, 4),
                "mean_polarization": round(mean_pol, 4),
            }
        )

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            {
                "community": [],
                "n_bills": [],
                "mean_pct_r_yea": [],
                "mean_pct_d_yea": [],
                "mean_polarization": [],
            }
        )
    )


# ── Phase 7: BiCM Backbone Extraction ──────────────────────────────────────


def extract_bicm_backbone(
    vote_matrix: pl.DataFrame,
    significance: float = BICM_SIGNIFICANCE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract statistically validated legislator projection via BiCM.

    Fits the Bipartite Configuration Model (maximum-entropy null preserving
    degree sequences), then computes p-values for each co-voting edge in the
    legislator projection. Edges with p < significance are retained.

    Args:
        vote_matrix: Binary vote matrix (legislator_slug first column).
        significance: P-value threshold for backbone retention.

    Returns:
        validated_matrix: (n_legs, n_legs) binary adjacency matrix
        pvalue_matrix: (n_legs, n_legs) p-value matrix
        slugs: list of legislator slugs
    """
    slug_col = vote_matrix.columns[0]
    slugs = vote_matrix[slug_col].to_list()
    vote_ids = [c for c in vote_matrix.columns if c != slug_col]
    vote_arr = vote_matrix.select(vote_ids).to_numpy().astype(float)

    # Replace NaN with 0 for BiCM (treats missing as no-vote)
    binary = np.nan_to_num(vote_arr, nan=0.0).astype(int)

    # Fit BiCM
    bg = BipartiteGraph()
    bg.set_biadjacency_matrix(binary)
    bg.solve_tool(light_mode=True)

    # Compute validated projection (rows = legislators)
    bg.compute_projection(rows=True, alpha=significance)

    # Extract p-value matrix and build validated adjacency from it.
    # BiCM's get_rows_projection() returns a dict (adjacency list) which may be
    # empty if no V-motifs pass at the given alpha. The p-value matrix is always
    # a dense numpy array, so we construct the validated matrix directly.
    pvalues = bg.get_projected_pvals_mat()
    if hasattr(pvalues, "toarray"):
        pvalues = pvalues.toarray()
    pvalues = np.asarray(pvalues, dtype=float)

    # Validated edges: p < significance AND p > 0 (p=0 means untested diagonal)
    validated = ((pvalues < significance) & (pvalues > 0)).astype(float)
    np.fill_diagonal(validated, 0.0)

    return validated, pvalues, slugs


def build_backbone_graph(
    validated_matrix: np.ndarray,
    slugs: list[str],
    ideal_points: pl.DataFrame,
) -> nx.Graph:
    """Build a NetworkX graph from the BiCM validated adjacency matrix.

    Args:
        validated_matrix: (n, n) binary matrix of validated edges.
        slugs: legislator slugs corresponding to matrix rows/columns.
        ideal_points: IRT ideal points for node attributes.

    Returns:
        Undirected graph with legislator nodes and validated edges.
    """
    G = nx.Graph()

    # Build IP lookup
    ip_dict: dict[str, dict] = {}
    for row in ideal_points.iter_rows(named=True):
        ip_dict[row["legislator_slug"]] = {
            "xi_mean": row.get("xi_mean", 0.0),
            "party": row.get("party", ""),
            "full_name": row.get("full_name", row["legislator_slug"]),
        }

    # Add nodes
    for slug in slugs:
        attrs = ip_dict.get(slug, {"xi_mean": 0.0, "party": "", "full_name": slug})
        G.add_node(
            slug,
            party=attrs["party"],
            xi_mean=attrs["xi_mean"],
            full_name=attrs["full_name"],
        )

    # Add validated edges
    n = len(slugs)
    for i in range(n):
        for j in range(i + 1, n):
            if validated_matrix[i, j] > 0:
                G.add_edge(slugs[i], slugs[j])

    return G


def compute_backbone_centrality(G: nx.Graph) -> pl.DataFrame:
    """Compute centrality measures on the BiCM backbone graph.

    Returns:
        Polars DataFrame with columns: legislator_slug, full_name, party,
        xi_mean, degree, betweenness, eigenvector, pagerank.
    """
    if G.number_of_nodes() == 0:
        return pl.DataFrame(
            {
                "legislator_slug": [],
                "full_name": [],
                "party": [],
                "xi_mean": [],
                "degree": [],
                "betweenness": [],
                "eigenvector": [],
                "pagerank": [],
            }
        )

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)

    # Eigenvector — may fail on disconnected graphs
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    rows = []
    for node in G.nodes():
        data = G.nodes[node]
        rows.append(
            {
                "legislator_slug": node,
                "full_name": data.get("full_name", node),
                "party": data.get("party", ""),
                "xi_mean": data.get("xi_mean", 0.0),
                "degree": degree[node],
                "betweenness": round(betweenness[node], 6),
                "eigenvector": round(eigenvector[node], 6),
                "pagerank": round(pagerank[node], 6),
            }
        )

    return pl.DataFrame(rows).sort("betweenness", descending=True)


def detect_backbone_communities(G: nx.Graph) -> tuple[dict[str, int], dict]:
    """Run Leiden modularity on the BiCM backbone graph.

    Returns:
        partition: dict mapping legislator slug to community ID
        comparison: dict with nmi, ari vs party labels
    """
    if G.number_of_edges() == 0:
        partition = {n: 0 for n in G.nodes()}
        return partition, {"nmi": 0.0, "ari": 0.0}

    ig_graph = ig.Graph.from_networkx(G)
    part = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        seed=RANDOM_SEED,
    )

    node_names = [ig_graph.vs[i]["_nx_name"] for i in range(ig_graph.vcount())]
    partition = {node_names[i]: part.membership[i] for i in range(len(node_names))}

    # Compare to party
    party_labels = []
    comm_labels = []
    for node in G.nodes():
        party = G.nodes[node].get("party", "")
        party_labels.append(party)
        comm_labels.append(partition[node])

    nmi = normalized_mutual_info_score(party_labels, comm_labels)
    ari = adjusted_rand_score(party_labels, comm_labels)

    return partition, {"nmi": round(nmi, 4), "ari": round(ari, 4)}


# ── Phase 8: Phase 6 Comparison ────────────────────────────────────────────


def build_kappa_network_for_comparison(
    vote_matrix: pl.DataFrame,
    ideal_points: pl.DataFrame,
    threshold: float = BACKBONE_COMPARISON_THRESHOLD,
) -> nx.Graph:
    """Build a Kappa network matching Phase 6 methodology.

    Recomputes pairwise Cohen's Kappa from the vote matrix and applies
    threshold filtering, then disparity filter for backbone.
    """
    slug_col = vote_matrix.columns[0]
    slugs = vote_matrix[slug_col].to_list()
    vote_ids = [c for c in vote_matrix.columns if c != slug_col]
    vote_arr = vote_matrix.select(vote_ids).to_numpy().astype(float)

    ip_dict: dict[str, dict] = {}
    for row in ideal_points.iter_rows(named=True):
        ip_dict[row["legislator_slug"]] = {
            "xi_mean": row.get("xi_mean", 0.0),
            "party": row.get("party", ""),
            "full_name": row.get("full_name", row["legislator_slug"]),
        }

    n = len(slugs)
    G = nx.Graph()
    for slug in slugs:
        attrs = ip_dict.get(slug, {"xi_mean": 0.0, "party": "", "full_name": slug})
        G.add_node(slug, **attrs)

    for i in range(n):
        for j in range(i + 1, n):
            vi = vote_arr[i]
            vj = vote_arr[j]
            mask = ~np.isnan(vi) & ~np.isnan(vj)
            if mask.sum() < 10:
                continue
            try:
                kappa = cohen_kappa_score(vi[mask].astype(int), vj[mask].astype(int))
            except Exception:
                continue
            if kappa > threshold:
                G.add_edge(slugs[i], slugs[j], weight=kappa)

    return G


def disparity_filter(G: nx.Graph, alpha: float = 0.05) -> nx.Graph:
    """Extract backbone using Serrano et al. (2009) disparity filter.

    Matches Phase 6 implementation for fair comparison.
    """
    backbone = nx.Graph()
    backbone.add_nodes_from(G.nodes(data=True))

    for u in G.nodes():
        k = G.degree(u)
        if k < 2:
            for v, data in G[u].items():
                if not backbone.has_edge(u, v):
                    backbone.add_edge(u, v, **data)
            continue

        strength = sum(d["weight"] for _, _, d in G.edges(u, data=True))
        if strength == 0:
            continue

        for v, data in G[u].items():
            w = data["weight"]
            p_ij = w / strength
            p_value = (1.0 - p_ij) ** (k - 1)
            if p_value < alpha:
                if not backbone.has_edge(u, v):
                    backbone.add_edge(u, v, **data)

    return backbone


def compare_backbones(
    bicm_backbone: nx.Graph,
    kappa_backbone: nx.Graph,
) -> dict:
    """Compare BiCM backbone with Phase 6 Kappa backbone.

    Returns:
        Dict with edge_jaccard, shared_edges, bicm_only, kappa_only,
        hidden_alliances (edges in BiCM but not Kappa).
    """
    bicm_edges = set(frozenset(e) for e in bicm_backbone.edges())
    kappa_edges = set(frozenset(e) for e in kappa_backbone.edges())

    intersection = bicm_edges & kappa_edges
    union = bicm_edges | kappa_edges
    jaccard = len(intersection) / len(union) if union else 1.0

    bicm_only = bicm_edges - kappa_edges
    kappa_only = kappa_edges - bicm_edges

    # Hidden alliances: cross-party edges in BiCM but not in Kappa
    hidden: list[dict] = []
    for edge in bicm_only:
        u, v = tuple(edge)
        if u in bicm_backbone.nodes and v in bicm_backbone.nodes:
            party_u = bicm_backbone.nodes[u].get("party", "")
            party_v = bicm_backbone.nodes[v].get("party", "")
            name_u = bicm_backbone.nodes[u].get("full_name", u)
            name_v = bicm_backbone.nodes[v].get("full_name", v)
            if party_u != party_v and party_u and party_v:
                hidden.append(
                    {
                        "legislator_1": name_u,
                        "party_1": party_u,
                        "legislator_2": name_v,
                        "party_2": party_v,
                    }
                )

    # Community comparison if both have edges
    comm_comparison: dict = {"nmi": None, "ari": None}
    shared_nodes = set(bicm_backbone.nodes()) & set(kappa_backbone.nodes())
    has_edges = bicm_backbone.number_of_edges() > 0 and kappa_backbone.number_of_edges() > 0
    if len(shared_nodes) > 2 and has_edges:
        # Leiden on both
        try:
            bicm_part, _ = detect_backbone_communities(bicm_backbone)
            kappa_part, _ = detect_backbone_communities(kappa_backbone)

            ordered_nodes = sorted(shared_nodes)
            bicm_labels = [bicm_part.get(n, 0) for n in ordered_nodes]
            kappa_labels = [kappa_part.get(n, 0) for n in ordered_nodes]

            comm_comparison["nmi"] = round(
                normalized_mutual_info_score(bicm_labels, kappa_labels), 4
            )
            comm_comparison["ari"] = round(adjusted_rand_score(bicm_labels, kappa_labels), 4)
        except Exception:
            pass

    return {
        "edge_jaccard": round(jaccard, 4),
        "shared_edges": len(intersection),
        "bicm_only": len(bicm_only),
        "kappa_only": len(kappa_only),
        "bicm_total": len(bicm_edges),
        "kappa_total": len(kappa_edges),
        "hidden_alliances": hidden,
        "community_comparison": comm_comparison,
    }


# ── Phase 9: Plots ─────────────────────────────────────────────────────────


def plot_degree_distributions(
    B: nx.Graph,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot legislator and bill degree distributions side by side."""
    legislators = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    bills = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    leg_degrees = [B.degree(n) for n in legislators]
    bill_degrees = [B.degree(n) for n in bills]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(leg_degrees, bins=20, color="#4a90d9", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Number of Yea Votes")
    axes[0].set_ylabel("Number of Legislators")
    axes[0].set_title(f"{chamber}: Legislator Degree Distribution")

    axes[1].hist(bill_degrees, bins=20, color="#d94a4a", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Number of Yea Votes Received")
    axes[1].set_ylabel("Number of Bills")
    axes[1].set_title(f"{chamber}: Bill Degree Distribution")

    fig.suptitle(f"{chamber} — Bipartite Degree Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, plots_dir / f"degree_dist_{chamber.lower()}.png")


def plot_polarization_histogram(
    polarization_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot distribution of bill polarization scores."""
    if polarization_df.height == 0:
        return

    vals = polarization_df["polarization"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=25, color="#6a4c93", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.set_xlabel("Polarization Score  |pct_R_yea − pct_D_yea|")
    ax.set_ylabel("Number of Bills")
    ax.set_title(f"{chamber}: Bill Polarization Distribution")
    ax.legend()

    # Annotate summary stats
    median = float(np.median(vals))
    ax.annotate(
        f"Median: {median:.2f}\nn={len(vals)} bills",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"polarization_hist_{chamber.lower()}.png")


def plot_bridge_vs_beta(
    bridge_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Scatter plot: bridge bill betweenness vs IRT discrimination (|beta|)."""
    if bridge_df.height == 0:
        return

    df = bridge_df.filter(pl.col("beta_mean").is_not_null())
    if df.height == 0:
        return

    btwn = df["betweenness"].to_numpy()
    beta = df["beta_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.abs(beta), btwn, alpha=0.7, s=40, c="#4a90d9", edgecolors="white", linewidth=0.5)
    ax.set_xlabel("IRT Discrimination |β|")
    ax.set_ylabel("Bipartite Betweenness")
    ax.set_title(f"{chamber}: Bridge Bills — Betweenness vs IRT Discrimination")

    # Annotate top 5 by betweenness
    for row in df.sort("betweenness", descending=True).head(5).iter_rows(named=True):
        label = row.get("bill_number", row["vote_id"])
        if label:
            ax.annotate(
                label,
                (abs(row["beta_mean"]), row["betweenness"]),
                fontsize=8,
                alpha=0.8,
            )

    fig.tight_layout()
    save_fig(fig, plots_dir / f"bridge_vs_beta_{chamber.lower()}.png")


def plot_bill_cluster_heatmap(
    profiles_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Heatmap of party support profiles per bill community."""
    if profiles_df.height == 0:
        return

    communities = profiles_df["community"].to_numpy()
    r_yea = profiles_df["mean_pct_r_yea"].to_numpy()
    d_yea = profiles_df["mean_pct_d_yea"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(3, profiles_df.height * 0.5 + 1)))

    data = np.column_stack([r_yea, d_yea])
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Republican %Yea", "Democrat %Yea"])
    ax.set_yticks(range(len(communities)))
    sizes = profiles_df["n_bills"].to_list()
    ax.set_yticklabels([f"Community {c} (n={s})" for c, s in zip(communities, sizes)])
    ax.set_title(f"{chamber}: Bill Community Party Support Profiles")

    # Add text annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] > 0.6 or data[i, j] < 0.4 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label="%Yea", shrink=0.8)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"bill_cluster_heatmap_{chamber.lower()}.png")


def plot_backbone_layout(
    G: nx.Graph,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Spring layout of the BiCM backbone graph colored by party."""
    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=1.5 / np.sqrt(G.number_of_nodes()))
    else:
        pos = nx.circular_layout(G)

    colors = [PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=40, alpha=0.8, ax=ax)

    # Label top 5 by degree
    degrees = dict(G.degree())
    top5 = sorted(degrees, key=degrees.get, reverse=True)[:5]
    labels = {n: G.nodes[n].get("full_name", n).split()[-1] for n in top5}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    legend_elements = [
        Patch(facecolor=c, label=p)
        for p, c in PARTY_COLORS.items()
        if any(G.nodes[n].get("party") == p for n in G.nodes())
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title(
        f"{chamber}: BiCM Backbone Network ({G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges)"
    )
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, plots_dir / f"backbone_layout_{chamber.lower()}.png")


def plot_backbone_comparison(
    bicm_backbone: nx.Graph,
    kappa_backbone: nx.Graph,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Side-by-side comparison of BiCM and Kappa backbones."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, G, title in [
        (axes[0], bicm_backbone, "BiCM Backbone"),
        (axes[1], kappa_backbone, "Kappa + Disparity Backbone"),
    ]:
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{chamber}: {title}")
            continue

        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=1.5 / max(1, np.sqrt(G.number_of_nodes())))
        colors = [PARTY_COLORS.get(G.nodes[n].get("party", ""), "#999999") for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=30, alpha=0.8, ax=ax)
        n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
        ax.set_title(f"{chamber}: {title}\n({n_nodes} nodes, {n_edges} edges)")
        ax.axis("off")

    legend_elements = [Patch(facecolor=c, label=p) for p, c in PARTY_COLORS.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3)
    fig.suptitle(f"{chamber}: Backbone Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, plots_dir / f"backbone_comparison_{chamber.lower()}.png")


def plot_bipartite_layout(
    B: nx.Graph,
    bridge_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Small bipartite layout showing bridge bills and their voting legislators."""
    if bridge_df.height == 0 or B.number_of_nodes() == 0:
        return

    # Take top 5 bridge bills and their neighbors
    top_bills = bridge_df.head(5)["vote_id"].to_list()
    subgraph_nodes = set(top_bills)
    for bill in top_bills:
        if bill in B:
            subgraph_nodes.update(B.neighbors(bill))

    sub = B.subgraph(subgraph_nodes).copy()
    if sub.number_of_nodes() == 0:
        return

    # Bipartite layout
    legislators = [n for n, d in sub.nodes(data=True) if d.get("bipartite") == 0]
    bills = [n for n, d in sub.nodes(data=True) if d.get("bipartite") == 1]

    pos = {}
    for i, n in enumerate(sorted(legislators)):
        pos[n] = (0, -i)
    for i, n in enumerate(sorted(bills)):
        pos[n] = (1, -i * len(legislators) / max(len(bills), 1))

    fig, ax = plt.subplots(figsize=(12, max(6, len(legislators) * 0.15 + 2)))

    # Draw edges
    nx.draw_networkx_edges(sub, pos, alpha=0.3, width=0.5, ax=ax)

    # Draw legislator nodes colored by party
    leg_colors = [PARTY_COLORS.get(sub.nodes[n].get("party", ""), "#999999") for n in legislators]
    nx.draw_networkx_nodes(
        sub, pos, nodelist=legislators, node_color=leg_colors, node_size=20, alpha=0.8, ax=ax
    )

    # Draw bill nodes
    nx.draw_networkx_nodes(
        sub,
        pos,
        nodelist=bills,
        node_color="#FFD700",
        node_size=60,
        node_shape="s",
        alpha=0.9,
        ax=ax,
    )

    # Label bills
    bill_labels = {}
    for n in bills:
        bn = sub.nodes[n].get("bill_number", "")
        bill_labels[n] = bn if bn else n[:15]
    nx.draw_networkx_labels(sub, pos, bill_labels, font_size=7, ax=ax)

    ax.set_title(f"{chamber}: Top 5 Bridge Bills — Bipartite Layout")
    legend_elements = [
        Patch(facecolor="#FFD700", label="Bill"),
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, plots_dir / f"bipartite_layout_{chamber.lower()}.png")


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
        "04_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )

    # Phase 6 is a soft dependency
    network_dir = None
    if not args.skip_phase6_comparison:
        try:
            network_dir = resolve_upstream_dir(
                "06_network",
                results_root,
                args.run_id,
                Path(args.network_dir) if args.network_dir else None,
            )
        except Exception:
            print("  Phase 6 network results not found — skipping comparison")
            network_dir = None

    with RunContext(
        session=args.session,
        analysis_name="06b_network_bipartite",
        params=vars(args),
        primer=BIPARTITE_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Bipartite Network Analysis — Session {args.session}")
        print(f"Data:    {data_dir}")
        print(f"EDA:     {eda_dir}")
        print(f"IRT:     {irt_dir}")
        print(f"Output:  {ctx.run_dir}")

        # ── Load data ──
        print_header("PHASE 1: LOADING DATA")
        vm_house, vm_senate = load_vote_matrices(eda_dir)
        irt_house, irt_senate = load_ideal_points(irt_dir)
        bp_house, bp_senate = load_bill_params(irt_dir)
        rollcalls, _legislators = load_metadata(data_dir)

        if vm_house is None and vm_senate is None:
            print("Phase 06b (Bipartite): skipping — no EDA vote matrices available")
            return
        if irt_house is None and irt_senate is None:
            print("Phase 06b (Bipartite): skipping — no IRT ideal points available")
            return

        for label, df in [("Vote matrix House", vm_house), ("Vote matrix Senate", vm_senate)]:
            info = f"{df.height} x {len(df.columns) - 1}" if df is not None else "not available"
            print(f"  {label}:  {info}")
        for label, df in [("IRT House", irt_house), ("IRT Senate", irt_senate)]:
            info = f"{df.height} legislators" if df is not None else "not available"
            print(f"  {label}:          {info}")
        for label, df in [("Bill params House", bp_house), ("Bill params Senate", bp_senate)]:
            info = f"{df.height}" if df is not None else "not available"
            print(f"  {label}:  {info}")
        print(f"  Rollcalls:          {rollcalls.height}")

        chamber_configs = [
            ("House", vm_house, irt_house, bp_house),
            ("Senate", vm_senate, irt_senate, bp_senate),
        ]

        results: dict[str, dict] = {}
        manifest: dict = {"chambers": {}}

        for chamber, vm, irt_ip, bp in chamber_configs:
            if vm is None or irt_ip is None or irt_ip.height < 5:
                n = irt_ip.height if irt_ip is not None else 0
                print(f"\n  Skipping {chamber}: too few legislators ({n})")
                continue

            ch_lower = chamber.lower()
            chamber_results: dict = {}

            # ── Build bipartite graph ──
            print_header(f"PHASE 2: BIPARTITE GRAPH — {chamber}")
            B = build_bipartite_graph(vm, irt_ip, bp, rollcalls)
            summary = compute_bipartite_summary(B)
            chamber_results["bipartite_graph"] = B
            chamber_results["summary"] = summary

            print(f"  Legislators: {summary['n_legislators']}")
            print(f"  Bills:       {summary['n_bills']}")
            print(f"  Edges:       {summary['n_edges']}")
            print(f"  Density:     {summary['density']}")
            print(f"  Avg legislator degree: {summary['avg_legislator_degree']}")
            print(f"  Avg bill degree:       {summary['avg_bill_degree']}")

            # ── Bill polarization ──
            print_header(f"PHASE 3: BILL POLARIZATION — {chamber}")
            polarization = compute_bill_polarization(vm, irt_ip, rollcalls, bp)
            polarization.write_parquet(ctx.data_dir / f"bill_polarization_{ch_lower}.parquet")
            chamber_results["polarization"] = polarization

            if polarization.height > 0:
                median_pol = float(polarization["polarization"].median())
                high_pol = polarization.filter(pl.col("polarization") > 0.8).height
                low_pol = polarization.filter(pl.col("polarization") < 0.2).height
                print(f"  Bills scored:   {polarization.height}")
                print(f"  Median polar.:  {median_pol:.3f}")
                print(f"  High (>0.8):    {high_pol}")
                print(f"  Low (<0.2):     {low_pol}")

            # ── Bipartite betweenness & bridge bills ──
            print_header(f"PHASE 4: BRIDGE BILLS — {chamber}")
            betweenness = compute_bipartite_betweenness(B)
            bridge_bills = identify_bridge_bills(B, betweenness, polarization, bp)
            bridge_bills.write_parquet(ctx.data_dir / f"bridge_bills_{ch_lower}.parquet")
            chamber_results["bridge_bills"] = bridge_bills
            chamber_results["betweenness"] = betweenness

            if bridge_bills.height > 0:
                print(f"  Top {min(5, bridge_bills.height)} bridge bills:")
                for row in bridge_bills.head(5).iter_rows(named=True):
                    bn = row.get("bill_number", "")
                    pol = row.get("polarization")
                    pol_str = f", pol={pol:.2f}" if pol is not None else ""
                    print(f"    {bn or row['vote_id']}: btwn={row['betweenness']:.4f}{pol_str}")

            # ── Bill projection & communities ──
            print_header(f"PHASE 5: BILL COMMUNITIES — {chamber}")
            G_bills = build_bill_projection(vm)
            partitions, sweep_df = detect_bill_communities(G_bills)
            sweep_df.write_parquet(ctx.data_dir / f"bill_community_sweep_{ch_lower}.parquet")
            chamber_results["bill_projection"] = G_bills
            chamber_results["bill_partitions"] = partitions
            chamber_results["bill_sweep"] = sweep_df

            if partitions:
                # Pick resolution with best modularity
                best_idx = sweep_df["modularity"].arg_max()
                best_res = float(sweep_df["resolution"][best_idx])
                best_partition = partitions[best_res]
                chamber_results["best_bill_partition"] = best_partition
                chamber_results["best_bill_resolution"] = best_res
                chamber_results["best_bill_modularity"] = float(sweep_df["modularity"][best_idx])

                # Save assignments
                comm_df = pl.DataFrame(
                    [{"vote_id": vid, "community": cid} for vid, cid in best_partition.items()]
                )
                comm_df.write_parquet(ctx.data_dir / f"bill_communities_{ch_lower}.parquet")

                # Community profiles
                profiles = analyze_bill_community_profiles(best_partition, vm, irt_ip)
                chamber_results["bill_community_profiles"] = profiles

                n_comms = len(set(best_partition.values()))
                print(f"  Best resolution: {best_res}")
                print(f"  Communities:     {n_comms}")
                if profiles.height > 0:
                    for row in profiles.iter_rows(named=True):
                        print(
                            f"    Community {row['community']}: {row['n_bills']} bills, "
                            f"R%={row['mean_pct_r_yea']:.2f}, D%={row['mean_pct_d_yea']:.2f}"
                        )

            # ── BiCM backbone ──
            print_header(f"PHASE 6: BiCM BACKBONE — {chamber}")
            try:
                bicm_alpha = BICM_SIGNIFICANCE_SENATE if chamber == "Senate" else BICM_SIGNIFICANCE
                validated, pvalues, backbone_slugs = extract_bicm_backbone(
                    vm, significance=bicm_alpha
                )
                backbone_G = build_backbone_graph(validated, backbone_slugs, irt_ip)
                chamber_results["backbone_graph"] = backbone_G
                chamber_results["backbone_validated"] = validated
                chamber_results["backbone_pvalues"] = pvalues

                print(f"  Backbone nodes:  {backbone_G.number_of_nodes()}")
                print(f"  Backbone edges:  {backbone_G.number_of_edges()}")

                # Centrality on backbone
                backbone_cent = compute_backbone_centrality(backbone_G)
                backbone_cent.write_parquet(
                    ctx.data_dir / f"backbone_centrality_{ch_lower}.parquet"
                )
                chamber_results["backbone_centrality"] = backbone_cent

                # Save backbone edges
                edge_rows = [{"legislator_1": u, "legislator_2": v} for u, v in backbone_G.edges()]
                if edge_rows:
                    pl.DataFrame(edge_rows).write_parquet(
                        ctx.data_dir / f"backbone_edges_{ch_lower}.parquet"
                    )

                # Community detection on backbone
                backbone_partition, backbone_vs_party = detect_backbone_communities(backbone_G)
                chamber_results["backbone_partition"] = backbone_partition
                chamber_results["backbone_vs_party"] = backbone_vs_party
                print(
                    f"  Backbone vs Party: NMI={backbone_vs_party['nmi']:.4f}, "
                    f"ARI={backbone_vs_party['ari']:.4f}"
                )

            except Exception as e:
                print(f"  BiCM backbone extraction failed: {e}")
                chamber_results["backbone_graph"] = nx.Graph()
                chamber_results["backbone_vs_party"] = {"nmi": None, "ari": None}

            # ── Phase 6 comparison ──
            if network_dir is not None and not args.skip_phase6_comparison:
                print_header(f"PHASE 7: PHASE 6 COMPARISON — {chamber}")
                try:
                    kappa_G = build_kappa_network_for_comparison(vm, irt_ip)
                    kappa_backbone = disparity_filter(kappa_G)
                    chamber_results["kappa_backbone"] = kappa_backbone

                    comparison = compare_backbones(
                        chamber_results.get("backbone_graph", nx.Graph()),
                        kappa_backbone,
                    )
                    chamber_results["backbone_comparison"] = comparison

                    print(f"  Edge Jaccard:    {comparison['edge_jaccard']:.4f}")
                    print(f"  Shared edges:    {comparison['shared_edges']}")
                    print(f"  BiCM-only:       {comparison['bicm_only']}")
                    print(f"  Kappa-only:      {comparison['kappa_only']}")
                    print(f"  Hidden alliances: {len(comparison['hidden_alliances'])}")
                    if comparison["community_comparison"]["nmi"] is not None:
                        print(f"  Community NMI:   {comparison['community_comparison']['nmi']}")
                        print(f"  Community ARI:   {comparison['community_comparison']['ari']}")
                except Exception as e:
                    print(f"  Phase 6 comparison failed: {e}")

            # ── Plots ──
            print_header(f"PHASE 8: PLOTS — {chamber}")
            plot_degree_distributions(B, chamber, ctx.plots_dir)
            plot_polarization_histogram(polarization, chamber, ctx.plots_dir)
            plot_bridge_vs_beta(bridge_bills, chamber, ctx.plots_dir)

            if "bill_community_profiles" in chamber_results:
                plot_bill_cluster_heatmap(
                    chamber_results["bill_community_profiles"], chamber, ctx.plots_dir
                )

            if "backbone_graph" in chamber_results:
                plot_backbone_layout(chamber_results["backbone_graph"], chamber, ctx.plots_dir)

            if "kappa_backbone" in chamber_results and "backbone_graph" in chamber_results:
                plot_backbone_comparison(
                    chamber_results["backbone_graph"],
                    chamber_results["kappa_backbone"],
                    chamber,
                    ctx.plots_dir,
                )

            plot_bipartite_layout(B, bridge_bills, chamber, ctx.plots_dir)

            # ── Manifest ──
            manifest["chambers"][chamber] = {
                "n_legislators": summary["n_legislators"],
                "n_bills": summary["n_bills"],
                "n_edges": summary["n_edges"],
                "n_polarized_bills": polarization.height,
                "n_bridge_bills": bridge_bills.height,
                "backbone_edges": chamber_results.get(
                    "backbone_graph", nx.Graph()
                ).number_of_edges(),
            }

            results[chamber] = chamber_results

        if not results:
            print("Phase 06b (Bipartite): skipping — no chambers had sufficient data")
            return

        # ── Save manifest and build report ──
        save_filtering_manifest(manifest, ctx.run_dir)

        print_header("BUILDING REPORT")
        build_bipartite_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
            skip_phase6=args.skip_phase6_comparison or network_dir is None,
        )

        print("\nBipartite network analysis complete.")
        print(f"Output: {ctx.run_dir}")


if __name__ == "__main__":
    main()
