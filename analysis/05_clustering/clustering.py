"""
Kansas Legislature — Clustering Analysis (Phase 5)

Identifies discrete voting blocs using multiple clustering methods on IRT ideal
points and agreement matrices. Supplements with a party loyalty metric to
distinguish ideology from caucus reliability (the Tyson paradox).

Usage:
  uv run python analysis/clustering.py [--session 2025-26] [--skip-sensitivity]
      [--skip-gmm] [--k INT]

Outputs (in results/<session>/clustering/<date>/):
  - data/:   Parquet files (cluster assignments, party loyalty, model selection)
  - plots/:  PNG visualizations (dendrograms, scatter plots, model selection)
  - filtering_manifest.json, run_info.json, run_log.txt
  - clustering_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import (
    cophenet,
    cut_tree,
    dendrogram,
    leaves_list,
    linkage,
)
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.clustering_report import build_clustering_report
except ModuleNotFoundError:
    from clustering_report import build_clustering_report  # type: ignore[no-redef]

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ImportError:
    from phase_utils import load_metadata, print_header, save_fig

# ── Primer ───────────────────────────────────────────────────────────────────

CLUSTERING_PRIMER = """\
# Clustering Analysis

## Purpose

Identifies discrete voting blocs (factions) among Kansas legislators using
multiple clustering methods for robustness. The expected structure is three
clusters: conservative Republicans, moderate Republicans, and Democrats.

Supplements IRT ideal points with a party loyalty metric to distinguish
"ideologically extreme" legislators from "unreliable caucus members" — the
Tyson paradox (see `analysis/design/tyson_paradox.md`).

## Method

### Three Complementary Approaches

1. **Hierarchical clustering** on agreement Kappa distance (average linkage)
   - Input: pairwise Kappa agreement matrices from EDA
   - Distance = 1 - Kappa; average linkage (valid for non-Euclidean metrics)
   - Produces dendrogram for visual inspection

2. **K-Means** on IRT ideal points
   - Input: 1D (xi_mean) and 2D (xi_mean, loyalty_rate)
   - Centroid-based; assumes spherical clusters
   - Elbow + silhouette for k selection

3. **Gaussian Mixture Model** on IRT ideal points
   - Input: xi_mean, weighted by 1/xi_sd (uncertain legislators down-weighted)
   - Probabilistic; provides soft cluster assignments (membership probabilities)
   - BIC for model selection

### Party Loyalty Metric

For each legislator: fraction of "contested" votes (>= 10% party dissent)
where the legislator agrees with the party median. Tyson scores low despite
extreme IRT position; core party members score high.

### Cross-Method Validation

Adjusted Rand Index (ARI) between each pair of methods. ARI > 0.7 indicates
strong agreement; methods recovering the same structure confirms robustness.

## Inputs

Reads from `results/<session>/irt/latest/data/`:
- `ideal_points_house.parquet` — House IRT ideal points + metadata
- `ideal_points_senate.parquet` — Senate ideal points

Reads from `results/<session>/eda/latest/data/`:
- `agreement_kappa_house.parquet` — House pairwise Kappa agreement
- `agreement_kappa_senate.parquet` — Senate Kappa agreement
- `vote_matrix_house_filtered.parquet` — Filtered binary vote matrices
- `vote_matrix_senate_filtered.parquet`

Reads from `results/<session>/pca/latest/data/`:
- `pc_scores_house.parquet` — PCA scores (for cross-validation)
- `pc_scores_senate.parquet`

Reads from `data/{legislature}_{start}-{end}/`:
- `{output_name}_rollcalls.csv` — Roll call metadata
- `{output_name}_legislators.csv` — Legislator metadata

## Outputs

All outputs land in `results/<session>/clustering/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `party_loyalty_house.parquet` | House party loyalty scores |
| `party_loyalty_senate.parquet` | Senate party loyalty scores |
| `hierarchical_assignments_house.parquet` | Hierarchical cluster assignments |
| `hierarchical_assignments_senate.parquet` | Senate hierarchical assignments |
| `kmeans_assignments_house.parquet` | K-Means cluster assignments |
| `kmeans_assignments_senate.parquet` | Senate K-Means assignments |
| `gmm_assignments_house.parquet` | GMM soft cluster assignments |
| `gmm_assignments_senate.parquet` | Senate GMM assignments |
| `model_selection_house.parquet` | Silhouette/BIC scores per k |
| `model_selection_senate.parquet` | Senate model selection |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `dendrogram_house.png` | Truncated dendrogram, party-colored |
| `dendrogram_senate.png` | Full dendrogram, party-colored |
| `model_selection_house.png` | Silhouette + elbow dual-axis plot |
| `model_selection_senate.png` | Senate model selection |
| `bic_aic_house.png` | GMM BIC/AIC per k |
| `bic_aic_senate.png` | Senate BIC/AIC |
| `irt_clusters_house.png` | IRT strip plot colored by cluster |
| `irt_clusters_senate.png` | Senate IRT clusters |
| `irt_loyalty_house.png` | 2D scatter: IRT x loyalty |
| `irt_loyalty_senate.png` | Senate IRT x loyalty |
| `cluster_composition_house.png` | Stacked bar: party composition |
| `cluster_composition_senate.png` | Senate cluster composition |
| `cluster_box_house.png` | Boxplot of xi_mean per cluster |
| `cluster_box_senate.png` | Senate cluster boxplot |

### Root files

| File | Description |
|------|-------------|
| `filtering_manifest.json` | All parameters, method results, ARI scores |
| `run_info.json` | Git commit, timestamp, Python version, parameters |
| `run_log.txt` | Full console output from the run |
| `clustering_report.html` | Self-contained HTML report with all tables and figures |

## Interpretation Guide

- **Dendrograms:** Height of merge = dissimilarity. Low merges = similar legislators.
  Color = party. Cutting at a given height produces k clusters.
- **Silhouette scores:** Range [-1, 1]. > 0.5 = good structure. > 0.7 = strong.
  < 0.25 = weak or no structure.
- **BIC (GMM):** Lower = better fit with fewer components. Minimum BIC = optimal k.
- **ARI:** Range [-0.5, 1]. > 0.7 = strong cross-method agreement. 1.0 = identical.
- **Party loyalty:** 0-1. Low loyalty + extreme ideology = maverick (Tyson).
  High loyalty + extreme ideology = reliable partisan.

## Caveats

- K-means assumes spherical clusters of equal size — violated if one faction is
  much larger than others (Republican supermajority).
- GMM weighting by 1/xi_sd is approximate (observation replication, not exact
  importance weighting).
- Hierarchical clustering on Kappa distance uses a different input space than
  k-means/GMM on IRT. Agreement is not always the same as whether the cluster
  structure exists, so cross-method comparison is essential.
- Veto override subgroup has only ~34 votes — cluster stability is limited.
"""

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
K_RANGE = range(2, 8)
COMPARISON_K = 3  # Forced k=3 cut for downstream comparison (k=2 confirmed optimal)
LINKAGE_METHOD = "average"  # Average linkage: valid for non-Euclidean distances (see ADR-0028)
COPHENETIC_THRESHOLD = 0.70
SILHOUETTE_GOOD = 0.50
GMM_COVARIANCE = "full"
GMM_N_INIT = 20
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 3
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
CLUSTER_CMAP = "Set2"
MINORITY_THRESHOLD = 0.025
SENSITIVITY_THRESHOLD = 0.10
MIN_VOTES = 20
CONTESTED_PARTY_THRESHOLD = 0.10
WITHIN_PARTY_MIN_SIZE = 15
# Plot sizing constants
DENDROGRAM_HEIGHT_PER_LEGISLATOR = 0.25
DENDROGRAM_TRUNCATED_HEIGHT = 10
VOTING_BLOCS_HEIGHT_PER_LEGISLATOR = 0.12
POLAR_SIZE_PER_LEGISLATOR = 0.18
NOTABLE_LOYALTY_THRESHOLD = 0.7
NOTABLE_XI_THRESHOLD = 3.5
EXTREME_PERCENTILE = 95
LABEL_STAGGER_RATIO = 0.8


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_display_labels(full_names: list[str]) -> dict[str, str]:
    """Build a mapping from full_name → short display label for plot annotations.

    Handles two problems:
      1. Leadership suffixes like " - Vice President of the Senate" are stripped
         before extracting the last name.
      2. Duplicate last names (e.g., two Claeys, two Smith) get disambiguated
         with a first-name initial (e.g., "J.R. Claeys" vs "Jo. Claeys").
    """
    from collections import Counter

    def _clean_name(name: str) -> str:
        """Strip leadership suffix: 'Tim Shallenburger - Vice President...' → 'Tim Shallenburger'"""
        return name.split(" - ")[0]

    def _last_name(clean: str) -> str:
        return clean.split()[-1] if clean else "?"

    # First pass: count last names to detect duplicates
    cleaned = {name: _clean_name(name) for name in full_names}
    last_names = {name: _last_name(c) for name, c in cleaned.items()}
    last_counts = Counter(last_names.values())

    # Second pass: build labels, disambiguating duplicates with first name
    labels: dict[str, str] = {}
    for name in full_names:
        clean = cleaned[name]
        last = last_names[name]
        if last_counts[last] > 1:
            # Include first name for disambiguation
            parts = clean.split()
            first = parts[0] if parts else ""
            # Abbreviate first name if long: "Joseph" → "Jo.", "J.R." stays as-is
            if len(first) > 3 and "." not in first:
                first = first[:2] + "."
            labels[name] = f"{first} {last}"
        else:
            labels[name] = last

    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Clustering Analysis")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis",
    )
    parser.add_argument(
        "--skip-gmm",
        action="store_true",
        help="Skip GMM (runs hierarchical + k-means only)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override optimal k (default: auto-select via silhouette)",
    )
    return parser.parse_args()


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_irt_ideal_points(irt_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load IRT ideal points for both chambers."""
    house = pl.read_parquet(irt_dir / "data" / "ideal_points_house.parquet")
    senate = pl.read_parquet(irt_dir / "data" / "ideal_points_senate.parquet")
    return house, senate


def load_agreement_matrices(eda_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load pairwise Kappa agreement matrices from EDA."""
    house = pl.read_parquet(eda_dir / "data" / "agreement_kappa_house.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "agreement_kappa_senate.parquet")
    return house, senate


def load_vote_matrices(eda_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load filtered binary vote matrices from EDA."""
    house = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
    return house, senate


def load_pca_scores(pca_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load PCA scores for cross-validation."""
    house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
    senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
    return house, senate


# ── Phase 2: Party Loyalty ───────────────────────────────────────────────────


def compute_party_loyalty(
    vote_matrix: pl.DataFrame,
    ideal_points: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Compute party loyalty rate for each legislator.

    For each contested vote (>= CONTESTED_PARTY_THRESHOLD of a party dissents),
    check whether the legislator agrees with the party median.

    Returns DataFrame with: legislator_slug, loyalty_rate, n_contested_votes,
    n_agree, party, full_name.
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in vote_matrix.columns if c != slug_col]

    # Merge party info
    meta = ideal_points.select("legislator_slug", "party", "full_name")
    leg_party = dict(
        zip(
            meta["legislator_slug"].to_list(),
            meta["party"].to_list(),
        )
    )

    # For each vote, compute the party median (majority position)
    # Build a long-format representation
    rows_data: list[dict] = []
    for row in vote_matrix.iter_rows(named=True):
        slug = row[slug_col]
        party = leg_party.get(slug)
        if not party:
            continue
        for vid in vote_cols:
            val = row[vid]
            if val is not None:
                rows_data.append(
                    {
                        "legislator_slug": slug,
                        "vote_id": vid,
                        "vote_val": int(val),
                        "party": party,
                    }
                )

    long = pl.DataFrame(rows_data)

    # For each vote x party, compute Yea rate and determine if contested
    party_vote_stats = long.group_by("vote_id", "party").agg(
        pl.col("vote_val").mean().alias("yea_rate"),
        pl.col("vote_val").len().alias("n_voters"),
    )

    # Determine party median position (Yea if yea_rate > 0.5, else Nay)
    # A vote is contested for a party if min(yea_rate, 1-yea_rate) >= threshold
    party_vote_stats = party_vote_stats.with_columns(
        pl.when(pl.col("yea_rate") > 0.5)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("party_median"),
        (
            pl.min_horizontal(pl.col("yea_rate"), 1 - pl.col("yea_rate"))
            >= CONTESTED_PARTY_THRESHOLD
        ).alias("is_contested"),
    )

    # Filter to contested votes only
    contested = party_vote_stats.filter(pl.col("is_contested"))

    # Join back to individual votes
    long_contested = long.join(
        contested.select("vote_id", "party", "party_median"),
        on=["vote_id", "party"],
        how="inner",
    )

    # For each legislator, compute agreement with party median on contested votes
    loyalty = (
        long_contested.with_columns(
            (pl.col("vote_val") == pl.col("party_median")).cast(pl.Int32).alias("agrees")
        )
        .group_by("legislator_slug", "party")
        .agg(
            pl.col("agrees").sum().alias("n_agree"),
            pl.col("agrees").len().alias("n_contested_votes"),
        )
        .with_columns((pl.col("n_agree") / pl.col("n_contested_votes")).alias("loyalty_rate"))
    )

    # Join full_name
    loyalty = loyalty.join(
        meta.select("legislator_slug", "full_name"),
        on="legislator_slug",
        how="left",
    )

    print(f"  {chamber}: {loyalty.height} legislators with party loyalty scores")
    print(f"  Contested votes (R): {contested.filter(pl.col('party') == 'Republican').height}")
    print(f"  Contested votes (D): {contested.filter(pl.col('party') == 'Democrat').height}")

    if loyalty.height > 0:
        sorted_loyalty = loyalty.sort("loyalty_rate")
        print(
            f"  Lowest loyalty: {sorted_loyalty['full_name'][0]} "
            f"({sorted_loyalty['loyalty_rate'][0]:.3f})"
        )
        print(
            f"  Highest loyalty: {sorted_loyalty['full_name'][-1]} "
            f"({sorted_loyalty['loyalty_rate'][-1]:.3f})"
        )

    return loyalty.sort("loyalty_rate", descending=True)


# ── Phase 3: Hierarchical Clustering ────────────────────────────────────────


def _kappa_to_distance(
    kappa_matrix: pl.DataFrame,
    chamber: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert Kappa agreement matrix to distance matrix.

    Returns (distance_array, slug_list).
    NaN entries (insufficient shared votes) filled with max finite distance.
    """
    slug_col = "legislator_slug"
    slugs = kappa_matrix[slug_col].to_list()
    data_cols = [c for c in kappa_matrix.columns if c != slug_col]

    kappa_arr = kappa_matrix.select(data_cols).to_numpy()
    distance_arr = 1.0 - kappa_arr

    # Ensure symmetry and zero diagonal
    distance_arr = (distance_arr + distance_arr.T) / 2
    np.fill_diagonal(distance_arr, 0.0)

    # Fill NaN distances (pairs with too few shared votes for Kappa)
    # with max finite distance — treats unknown pairs as maximally dissimilar
    nan_count = int(np.isnan(distance_arr).sum())
    if nan_count > 0:
        max_finite = float(np.nanmax(distance_arr))
        distance_arr = np.where(np.isnan(distance_arr), max_finite, distance_arr)
        if chamber:
            print(f"  {chamber}: Filled {nan_count} NaN distances with max={max_finite:.4f}")

    # Clip negative distances (kappa > 1 shouldn't happen, but be safe)
    distance_arr = np.clip(distance_arr, 0.0, None)

    return distance_arr, slugs


def _standardize_2d(xi: np.ndarray, loyalty: np.ndarray) -> np.ndarray:
    """Z-score standardize IRT ideal points and loyalty rates into a 2D array."""
    xi_std = (xi - xi.mean()) / (xi.std() + 1e-10)
    loy_std = (loyalty - loyalty.mean()) / (loyalty.std() + 1e-10)
    return np.column_stack([xi_std, loy_std])


def run_hierarchical(
    kappa_matrix: pl.DataFrame,
    chamber: str,
) -> tuple[np.ndarray, float, list[str], np.ndarray]:
    """Run hierarchical clustering on Kappa distance matrix.

    Returns (linkage_matrix, cophenetic_r, slug_list, distance_array).
    """
    distance_arr, slugs = _kappa_to_distance(kappa_matrix, chamber)

    # Convert to condensed form for scipy
    condensed = squareform(distance_arr, checks=False)

    # Linkage
    Z = linkage(condensed, method=LINKAGE_METHOD)

    # Cophenetic correlation
    coph_corr, _ = cophenet(Z, condensed)

    print(
        f"  {chamber}: Cophenetic correlation = {coph_corr:.4f} "
        f"({'OK' if coph_corr >= COPHENETIC_THRESHOLD else 'WARNING'})"
    )

    return Z, float(coph_corr), slugs, distance_arr


def find_optimal_k_hierarchical(
    Z: np.ndarray,
    distance_arr: np.ndarray,
    k_range: range,
    chamber: str,
) -> tuple[dict[int, float], int]:
    """Find optimal k by silhouette score on hierarchical cut.

    Returns (scores_dict, optimal_k).
    """
    scores: dict[int, float] = {}
    for k in k_range:
        labels = cut_tree(Z, n_clusters=k).flatten()
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(distance_arr, labels, metric="precomputed")
        scores[k] = float(sil)
        print(f"    k={k}: silhouette = {sil:.4f}")

    optimal_k = max(scores, key=scores.get) if scores else COMPARISON_K
    print(
        f"  {chamber} optimal k (hierarchical): {optimal_k} "
        f"(silhouette = {scores.get(optimal_k, 0):.4f})"
    )
    return scores, optimal_k


def plot_dendrogram(
    Z: np.ndarray,
    slugs: list[str],
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Plot dendrogram with party-colored leaf labels."""
    party_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["party"].to_list(),
        )
    )
    name_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["full_name"].to_list(),
        )
    )

    labels = [name_map.get(s, s) for s in slugs]
    truncate = chamber == "House"

    fig_height = (
        DENDROGRAM_TRUNCATED_HEIGHT
        if truncate
        else max(14, len(slugs) * DENDROGRAM_HEIGHT_PER_LEGISLATOR)
    )
    fig, ax = plt.subplots(figsize=(14, fig_height))

    if truncate:
        # Truncated dendrogram for House (130 members unreadable at full)
        dendrogram(
            Z,
            truncate_mode="lastp",
            p=12,
            ax=ax,
            orientation="top",
            leaf_font_size=9,
        )
        ax.set_title(f"{chamber} — Hierarchical Clustering Dendrogram (truncated, top 12)")
    else:
        # Full dendrogram for Senate (42 members readable)
        dendrogram(
            Z,
            labels=labels,
            ax=ax,
            orientation="left",
            leaf_font_size=7,
            no_plot=False,
        )

        # Color leaf labels by party
        ylbls = ax.get_yticklabels()
        for lbl in ylbls:
            name = lbl.get_text()
            # Find slug for this name
            slug = None
            for s, n in name_map.items():
                if n == name:
                    slug = s
                    break
            if slug:
                party = party_map.get(slug, "")
                color = PARTY_COLORS.get(party, "#888888")
                lbl.set_color(color)

        ax.set_title(f"{chamber} — Hierarchical Clustering Dendrogram")

    ax.set_xlabel("Distance (1 - Kappa)")
    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    fig.tight_layout()
    save_fig(fig, out_dir / f"dendrogram_{chamber.lower()}.png")


def plot_voting_blocs(
    Z: np.ndarray,
    slugs: list[str],
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Sorted dot plot: legislators ordered by dendrogram leaf order, colored by party.

    X-axis = IRT ideal point, y-axis = position in hierarchical leaf order.
    Reveals voting blocs more readably than a dendrogram, especially for the House.
    """
    leaf_order = leaves_list(Z)

    ip_slugs = ideal_points["legislator_slug"].to_list()
    party_map = dict(zip(ip_slugs, ideal_points["party"].to_list()))
    name_map = dict(zip(ip_slugs, ideal_points["full_name"].to_list()))
    ip_map = dict(zip(ip_slugs, ideal_points["xi_mean"].to_list()))

    # Build ordered lists from leaf order
    ordered_slugs = [slugs[i] for i in leaf_order]
    ordered_xi = [ip_map.get(s, 0.0) for s in ordered_slugs]
    ordered_parties = [party_map.get(s, "Unknown") for s in ordered_slugs]
    ordered_names = [name_map.get(s, s) for s in ordered_slugs]

    # Build display labels (handles leadership suffixes + duplicate last names)
    all_names = ideal_points["full_name"].to_list()
    display_labels = _build_display_labels(all_names)

    n = len(ordered_slugs)
    fig, ax = plt.subplots(figsize=(10, max(8, n * VOTING_BLOCS_HEIGHT_PER_LEGISLATOR)))

    for i in range(n):
        color = PARTY_COLORS.get(ordered_parties[i], "#888888")
        ax.scatter(ordered_xi[i], i, c=color, s=30, edgecolors="black", linewidth=0.3, zorder=2)

    # Vertical dashed line at ideological center
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.5)

    # Find the biggest gap in the leaf order by party — draw a horizontal separator
    party_seq = [1 if p == "Republican" else 0 for p in ordered_parties]
    max_gap_pos = 0
    max_gap_size = 0
    for i in range(1, n):
        if party_seq[i] != party_seq[i - 1]:
            # Count consecutive same-party on each side
            gap_size = 1
            max_gap_pos = i
            max_gap_size = max(max_gap_size, gap_size)
    if max_gap_pos > 0:
        ax.axhline(
            max_gap_pos - 0.5, color="#888888", linestyle="-", linewidth=1.0, alpha=0.4, zorder=1
        )

    # Label notable outliers — legislators at IRT extremes within each party
    xi_arr = np.array(ordered_xi)
    for i in range(n):
        is_extreme = abs(ordered_xi[i]) > np.percentile(np.abs(xi_arr), EXTREME_PERCENTILE)
        if is_extreme:
            label = display_labels.get(ordered_names[i], ordered_names[i].split()[-1])
            ax.annotate(
                label,
                (ordered_xi[i], i),
                fontsize=6,
                fontweight="bold",
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "wheat", "alpha": 0.7},
            )

    ax.set_xlabel("IRT Ideal Point (Liberal \u2190 \u2192 Conservative)")
    ax.set_ylabel("Position in Voting Similarity Order")
    ax.set_yticks([])
    ax.set_title(f"{chamber} \u2014 Voting Blocs (who clusters together)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"voting_blocs_{chamber.lower()}.png")


def plot_polar_dendrogram(
    Z: np.ndarray,
    slugs: list[str],
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Circular dendrogram: leaves around a circle, branches connect inward.

    More compact than a linear dendrogram and readable even for large chambers.
    """
    party_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["party"].to_list(),
        )
    )
    name_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["full_name"].to_list(),
        )
    )

    # Build display labels (handles leadership suffixes + duplicate last names)
    all_names = ideal_points["full_name"].to_list()
    display_labels = _build_display_labels(all_names)

    # Get dendrogram coordinates without plotting
    ddata = dendrogram(Z, no_plot=True)
    icoord = np.array(ddata["icoord"])
    dcoord = np.array(ddata["dcoord"])
    leaves = ddata["leaves"]

    n_leaves = len(leaves)
    max_dist = Z[:, 2].max()

    fig_size = max(14, n_leaves * POLAR_SIZE_PER_LEGISLATOR)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw={"projection": "polar"})

    # Map rectangular dendrogram x-coords to angle [0, 2*pi)
    # ddata x-coords are 5, 15, 25, ... (step 10) for leaves
    x_min = icoord.min()
    x_max = icoord.max()
    x_range = x_max - x_min if x_max > x_min else 1.0

    def x_to_angle(x: float) -> float:
        return 2 * np.pi * (x - x_min) / x_range

    # Map y-coords (merge distance) to radius: leaves at outer edge, root at center
    outer_r = 1.0
    inner_r = 0.2

    def y_to_radius(y: float) -> float:
        if max_dist == 0:
            return outer_r
        return outer_r - (y / max_dist) * (outer_r - inner_r)

    # Draw branches
    for i in range(len(icoord)):
        xs = icoord[i]
        ys = dcoord[i]
        # Each U-shaped branch has 4 points: (x0,y0)-(x0,y1)-(x3,y1)-(x3,y3)
        # In polar: draw two radial segments and one arc
        a0 = x_to_angle(xs[0])
        a1 = x_to_angle(xs[1])
        a2 = x_to_angle(xs[2])
        a3 = x_to_angle(xs[3])
        r0 = y_to_radius(ys[0])
        r1 = y_to_radius(ys[1])
        r3 = y_to_radius(ys[3])

        # Left vertical: (a0, r0) -> (a0, r1)
        ax.plot([a0, a0], [r0, r1], color="#555555", linewidth=0.5, alpha=0.6)
        # Horizontal arc: (a0, r1) -> (a3, r1)
        arc_angles = np.linspace(a1, a2, 30)
        arc_radii = np.full_like(arc_angles, r1)
        ax.plot(arc_angles, arc_radii, color="#555555", linewidth=0.5, alpha=0.6)
        # Right vertical: (a3, r1) -> (a3, r3)
        ax.plot([a3, a3], [r1, r3], color="#555555", linewidth=0.5, alpha=0.6)

    # Draw leaf dots and labels
    # The actual leaf x positions from ddata
    leaf_positions = sorted([(5 + 10 * i, leaves[i]) for i in range(n_leaves)], key=lambda t: t[0])

    # Compute leaf angles
    leaf_angles: list[float] = []
    for x_pos, _leaf_idx in leaf_positions:
        leaf_angles.append(x_to_angle(x_pos))

    # Detect crowded labels and stagger them radially.
    # When two adjacent labels are angularly close, the rotated text overlaps
    # even after nudging. Instead, place crowded labels at an outer radius so
    # they sit in a different "lane" from their neighbors.
    even_sep = 2 * np.pi / n_leaves
    label_radii: list[float] = []
    r_inner_label = outer_r + 0.05
    r_outer_label = outer_r + 0.18
    for i in range(len(leaf_angles)):
        if i == 0:
            label_radii.append(r_inner_label)
            continue
        gap = leaf_angles[i] - leaf_angles[i - 1]
        # If this label is closer than 80% of even spacing to the previous,
        # AND the previous was at the inner radius, push this one out
        if gap < even_sep * LABEL_STAGGER_RATIO and label_radii[i - 1] == r_inner_label:
            label_radii.append(r_outer_label)
        else:
            label_radii.append(r_inner_label)

    fontsize = max(5, min(7, 350 / n_leaves))

    for idx, (_x_pos, leaf_idx) in enumerate(leaf_positions):
        angle = leaf_angles[idx]
        slug = slugs[leaf_idx]
        party = party_map.get(slug, "Unknown")
        color = PARTY_COLORS.get(party, "#888888")
        name = name_map.get(slug, slug)
        label = display_labels.get(name, name.split()[-1])

        # Leaf dot
        ax.scatter(angle, outer_r, c=color, s=15, edgecolors="black", linewidth=0.3, zorder=3)

        # Label at staggered radius
        r_label = label_radii[idx]
        rotation = np.degrees(angle) - 90
        ha = "left"
        if 90 < np.degrees(angle) < 270:
            rotation += 180
            ha = "right"
        ax.text(
            angle,
            r_label,
            label,
            fontsize=fontsize,
            rotation=rotation,
            ha=ha,
            va="center",
            color=color,
        )

    ax.set_ylim(0, outer_r + 0.45)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"{chamber} \u2014 Voting Similarity Tree", pad=30, fontsize=14)
    ax.spines["polar"].set_visible(False)

    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(1.15, 0), fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"polar_dendrogram_{chamber.lower()}.png")


def plot_icicle(
    Z: np.ndarray,
    slugs: list[str],
    ideal_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Icicle (flame) chart: top-down rectangular hierarchy of voting blocs.

    Each merge is a horizontal span proportional to its child count.
    Y-axis = merge distance (top = most distant), bottom row = individual legislators.
    Colored by majority party in each subtree.
    """
    party_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["party"].to_list(),
        )
    )
    name_map = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            ideal_points["full_name"].to_list(),
        )
    )

    n = len(slugs)
    leaf_order = leaves_list(Z)

    # Build display labels (handles leadership suffixes + duplicate last names)
    all_names = ideal_points["full_name"].to_list()
    display_labels = _build_display_labels(all_names)

    # Build subtree info: for each node (leaf or internal), track its
    # position span in the leaf order and its majority party
    # Leaf nodes: 0..n-1, internal nodes: n..2n-2
    node_left: dict[int, float] = {}
    node_right: dict[int, float] = {}
    node_party_counts: dict[int, dict[str, int]] = {}
    node_height: dict[int, float] = {}

    # Position of each leaf in the leaf order
    leaf_pos = {leaf_id: pos for pos, leaf_id in enumerate(leaf_order)}

    # Initialize leaf nodes
    for i in range(n):
        pos = leaf_pos[i]
        node_left[i] = pos - 0.4
        node_right[i] = pos + 0.4
        node_height[i] = 0.0
        party = party_map.get(slugs[i], "Unknown")
        node_party_counts[i] = {party: 1}

    # Process internal nodes from the linkage matrix
    for idx in range(len(Z)):
        node_id = n + idx
        left_child = int(Z[idx, 0])
        right_child = int(Z[idx, 1])
        merge_dist = float(Z[idx, 2])

        node_left[node_id] = min(node_left[left_child], node_left[right_child])
        node_right[node_id] = max(node_right[left_child], node_right[right_child])
        node_height[node_id] = merge_dist

        # Merge party counts
        counts: dict[str, int] = {}
        for party, c in node_party_counts[left_child].items():
            counts[party] = counts.get(party, 0) + c
        for party, c in node_party_counts[right_child].items():
            counts[party] = counts.get(party, 0) + c
        node_party_counts[node_id] = counts

    max_dist = Z[:, 2].max() if len(Z) > 0 else 1.0
    fig_width = max(14, n * 0.15)
    fig_height = max(8, fig_width * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw rectangles for internal nodes
    for idx in range(len(Z)):
        node_id = n + idx
        left_child = int(Z[idx, 0])
        right_child = int(Z[idx, 1])

        x_left = node_left[node_id]
        x_right = node_right[node_id]
        y_bottom = node_height[node_id]

        # Height extends up to parent's merge distance (or max for root)
        # Find parent height
        parent_height = max_dist * 1.05
        for pidx in range(idx + 1, len(Z)):
            if int(Z[pidx, 0]) == node_id or int(Z[pidx, 1]) == node_id:
                parent_height = float(Z[pidx, 2])
                break

        # Determine color by majority party
        counts = node_party_counts[node_id]
        majority = max(counts, key=counts.get)
        color = PARTY_COLORS.get(majority, "#888888")

        rect = plt.Rectangle(
            (x_left, y_bottom),
            x_right - x_left,
            parent_height - y_bottom,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.5,
        )
        ax.add_patch(rect)

    # Draw bottom row: individual legislators
    fontsize = max(4, min(7, 500 / n))
    for i in range(n):
        leaf_id = leaf_order[i]
        slug = slugs[leaf_id]
        party = party_map.get(slug, "Unknown")
        color = PARTY_COLORS.get(party, "#888888")
        name = name_map.get(slug, slug)
        label = display_labels.get(name, name.split()[-1])

        ax.scatter(i, 0, c=color, s=20, edgecolors="black", linewidth=0.3, zorder=3)
        ax.text(
            i,
            -max_dist * 0.03,
            label,
            fontsize=fontsize,
            rotation=90,
            ha="center",
            va="top",
            color=color,
        )

    ax.set_xlim(-1, n)
    ax.set_ylim(-max_dist * 0.15, max_dist * 1.1)
    ax.set_ylabel("Merge Distance (Kappa)")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_title(f"{chamber} \u2014 Hierarchical Voting Blocs")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"icicle_{chamber.lower()}.png")


# ── Phase 4: K-Means on IRT ─────────────────────────────────────────────────


def run_kmeans_irt(
    ideal_points: pl.DataFrame,
    loyalty: pl.DataFrame | None,
    k_range: range,
    chamber: str,
) -> tuple[dict[int, dict], int]:
    """Run k-means on IRT ideal points (1D and 2D with loyalty).

    Returns (results_per_k, optimal_k).
    """
    xi = ideal_points["xi_mean"].to_numpy().reshape(-1, 1)

    # 1D k-means
    results: dict[int, dict] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(xi)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(xi, labels)) if len(set(labels)) > 1 else -1.0
        results[k] = {
            "labels_1d": labels,
            "inertia": inertia,
            "silhouette_1d": sil,
            "centroids_1d": km.cluster_centers_.flatten().tolist(),
        }
        print(f"    k={k}: inertia={inertia:.2f}, silhouette(1D)={sil:.4f}")

    # 2D k-means with loyalty
    if loyalty is not None:
        merged = ideal_points.select("legislator_slug", "xi_mean").join(
            loyalty.select("legislator_slug", "loyalty_rate"),
            on="legislator_slug",
            how="inner",
        )
        if merged.height >= max(k_range):
            X_2d = _standardize_2d(
                merged["xi_mean"].to_numpy(),
                merged["loyalty_rate"].to_numpy(),
            )

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                labels_2d = km.fit_predict(X_2d)
                sil_2d = (
                    float(silhouette_score(X_2d, labels_2d)) if len(set(labels_2d)) > 1 else -1.0
                )
                results[k]["labels_2d"] = labels_2d
                results[k]["silhouette_2d"] = sil_2d
                print(f"    k={k}: silhouette(2D)={sil_2d:.4f}")

    # Optimal k by 1D silhouette
    optimal_k = max(
        k_range,
        key=lambda k: results[k]["silhouette_1d"],
    )
    print(f"  {chamber} optimal k (k-means): {optimal_k}")
    return results, optimal_k


def plot_elbow_silhouette(
    results: dict[int, dict],
    chamber: str,
    out_dir: Path,
) -> None:
    """Dual y-axis plot: inertia (elbow) + silhouette."""
    ks = sorted(results.keys())
    inertias = [results[k]["inertia"] for k in ks]
    silhouettes = [results[k]["silhouette_1d"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "#4C72B0"
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (SSE)", color=color1)
    ax1.plot(ks, inertias, "o-", color=color1, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#E81B23"
    ax2.set_ylabel("Silhouette Score", color=color2)
    ax2.plot(ks, silhouettes, "s--", color=color2, label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Threshold line and optimal k annotation
    ax2.axhline(SILHOUETTE_GOOD, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax2.text(
        ks[-1] + 0.15,
        SILHOUETTE_GOOD,
        "Good threshold (0.50)",
        va="center",
        fontsize=7,
        color="#888888",
    )
    optimal_k = ks[int(np.argmax(silhouettes))]
    best_sil = max(silhouettes)
    ax2.plot(optimal_k, best_sil, marker="*", markersize=14, color=color2, zorder=5)
    ax2.annotate(
        f"k={optimal_k} optimal\n(silhouette={best_sil:.2f})",
        (optimal_k, best_sil),
        textcoords="offset points",
        xytext=(12, -8),
        fontsize=8,
        color=color2,
        fontweight="bold",
    )

    ax1.set_title(f"{chamber} \u2014 How Many Groups Exist in the Legislature?")

    # Add annotation at k=2 if it's optimal
    if optimal_k == 2:
        ax1.annotate(
            "The data strongly favors\nexactly 2 groups",
            xy=(2, inertias[0]),
            xytext=(4, inertias[0] * 0.8),
            fontsize=9,
            fontweight="bold",
            color="#333333",
            bbox={"boxstyle": "round,pad=0.3", "fc": "lightyellow", "alpha": 0.8},
            arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.2},
        )

    ax1.set_xticks(ks)
    fig.tight_layout()
    save_fig(fig, out_dir / f"model_selection_{chamber.lower()}.png")


def plot_irt_clusters(
    ideal_points: pl.DataFrame,
    labels: np.ndarray,
    k: int,
    chamber: str,
    out_dir: Path,
) -> None:
    """Strip plot of IRT ideal points colored by cluster, with party shapes."""
    cmap = plt.get_cmap(CLUSTER_CMAP)
    fig, ax = plt.subplots(figsize=(12, 5))

    parties = ideal_points["party"].to_list()
    xi_vals = ideal_points["xi_mean"].to_numpy()

    for cluster_id in range(k):
        mask = labels == cluster_id
        for party, marker in [("Republican", "o"), ("Democrat", "s")]:
            party_mask = np.array([p == party for p in parties])
            combined = mask & party_mask
            if combined.any():
                ax.scatter(
                    xi_vals[combined],
                    np.random.default_rng(RANDOM_SEED).uniform(-0.3, 0.3, combined.sum()),
                    c=[cmap(cluster_id / max(k - 1, 1))],
                    marker=marker,
                    s=50,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"Cluster {cluster_id} ({party[0]})" if party == "Republican" else None,
                )

    ax.axhline(0, color="gray", alpha=0.2)
    ax.set_xlabel("IRT Ideal Point (Liberal \u2190 \u2192 Conservative)")
    ax.set_ylabel("")
    ax.set_yticks([])

    # Use a descriptive title for k=2 (most common)
    if k == 2:
        ax.set_title(f"{chamber} \u2014 Two Groups \u2014 And They're Exactly the Two Parties")
        ax.annotate(
            "The data says there are exactly two groups\n"
            "\u2014 and they match party labels perfectly",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=9,
            fontstyle="italic",
            color="#555555",
            bbox={"boxstyle": "round,pad=0.4", "fc": "lightyellow", "alpha": 0.8},
        )
    else:
        ax.set_title(f"{chamber} \u2014 K-Means Clusters (k={k}) on IRT Ideal Points")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Custom legend
    handles = [Patch(facecolor=cmap(i / max(k - 1, 1)), label=f"Cluster {i}") for i in range(k)]
    handles.extend(
        [
            plt.Line2D([], [], marker="o", color="gray", linestyle="None", label="Republican"),
            plt.Line2D([], [], marker="s", color="gray", linestyle="None", label="Democrat"),
        ]
    )
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"irt_clusters_{chamber.lower()}.png")


def plot_irt_loyalty_clusters(
    ideal_points: pl.DataFrame,
    loyalty: pl.DataFrame,
    labels: np.ndarray,
    k: int,
    chamber: str,
    out_dir: Path,
) -> None:
    """2D scatter: IRT x-axis, loyalty y-axis, color=cluster, shape=party."""
    merged = ideal_points.select("legislator_slug", "xi_mean", "party", "full_name").join(
        loyalty.select("legislator_slug", "loyalty_rate"),
        on="legislator_slug",
        how="inner",
    )

    cmap = plt.get_cmap(CLUSTER_CMAP)
    fig, ax = plt.subplots(figsize=(12, 8))

    xi_vals = merged["xi_mean"].to_numpy()
    loy_vals = merged["loyalty_rate"].to_numpy()
    parties = merged["party"].to_list()
    names = merged["full_name"].to_list()

    # Map labels from ideal_points row order to merged row order via slug lookup
    slug_to_label = {s: labels[i] for i, s in enumerate(ideal_points["legislator_slug"].to_list())}
    plot_labels = np.array([slug_to_label[s] for s in merged["legislator_slug"].to_list()])

    for cluster_id in range(k):
        mask = plot_labels == cluster_id
        for party, marker in [("Republican", "o"), ("Democrat", "s")]:
            party_mask = np.array([p == party for p in parties])
            combined = mask & party_mask
            if combined.any():
                ax.scatter(
                    xi_vals[combined],
                    loy_vals[combined],
                    c=[cmap(cluster_id / max(k - 1, 1))],
                    marker=marker,
                    s=60,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

    # Annotate notable legislators (extreme IRT or low loyalty) with callout boxes
    for i in range(merged.height):
        if loy_vals[i] < NOTABLE_LOYALTY_THRESHOLD or abs(xi_vals[i]) > NOTABLE_XI_THRESHOLD:
            last_name = names[i].split()[-1] if names[i] else "?"
            ax.annotate(
                last_name,
                (xi_vals[i], loy_vals[i]),
                fontsize=7,
                fontweight="bold",
                xytext=(8, -8),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.2", "fc": "wheat", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.6},
            )

    # Add callout for low-loyalty extremists (Tyson/Thompson pattern)
    low_loyalty_extreme = [
        i for i in range(merged.height) if loy_vals[i] < 0.5 and abs(xi_vals[i]) > 3.0
    ]
    if low_loyalty_extreme:
        extremist_names = [names[i].split()[-1] for i in low_loyalty_extreme]
        callout = (
            f"{' and '.join(extremist_names)}: ideologically extreme\nbut unreliable caucus members"
        )
        # Place callout near the first extremist
        idx = low_loyalty_extreme[0]
        ax.annotate(
            callout,
            (xi_vals[idx], loy_vals[idx]),
            xytext=(xi_vals[idx] - 2, loy_vals[idx] + 0.15),
            fontsize=8,
            fontstyle="italic",
            color="#555555",
            bbox={"boxstyle": "round,pad=0.4", "fc": "lightyellow", "alpha": 0.8, "ec": "#cccccc"},
            arrowprops={"arrowstyle": "->", "color": "#E81B23", "lw": 1.2},
        )

    ax.set_xlabel("IRT Ideal Point (Liberal \u2190 \u2192 Conservative)")
    ax.set_ylabel("Party Loyalty (higher = more reliable caucus member)")
    ax.set_title(f"{chamber} \u2014 Who Marches to Their Own Beat?")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [Patch(facecolor=cmap(i / max(k - 1, 1)), label=f"Cluster {i}") for i in range(k)]
    handles.extend(
        [
            plt.Line2D([], [], marker="o", color="gray", linestyle="None", label="Republican"),
            plt.Line2D([], [], marker="s", color="gray", linestyle="None", label="Democrat"),
        ]
    )
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"irt_loyalty_{chamber.lower()}.png")


# ── Phase 5: Gaussian Mixture Model ─────────────────────────────────────────


def run_gmm_irt(
    ideal_points: pl.DataFrame,
    k_range: range,
    chamber: str,
) -> tuple[dict[int, dict], int]:
    """Run GMM on IRT ideal points, weighted by 1/xi_sd.

    Returns (results_per_k, optimal_k_by_bic).
    """
    xi = ideal_points["xi_mean"].to_numpy().reshape(-1, 1)
    xi_sd = ideal_points["xi_sd"].to_numpy()

    # Approximate importance weighting by repeating observations
    weights = 1.0 / (xi_sd + 1e-6)
    weights = weights / weights.sum()
    # Scale to effective sample count of ~len(xi) * 3 for stability
    repeat_counts = np.maximum(1, np.round(weights * len(xi) * 3).astype(int))
    xi_weighted = np.repeat(xi, repeat_counts, axis=0)

    results: dict[int, dict] = {}
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=GMM_COVARIANCE,
            n_init=GMM_N_INIT,
            random_state=RANDOM_SEED,
        )
        gmm.fit(xi_weighted)
        bic = float(gmm.bic(xi_weighted))
        aic = float(gmm.aic(xi_weighted))

        # Predict on original (unweighted) data
        probs = gmm.predict_proba(xi)
        labels = gmm.predict(xi)

        results[k] = {
            "bic": bic,
            "aic": aic,
            "labels": labels,
            "probs": probs,
            "max_prob": probs.max(axis=1),
        }
        print(f"    k={k}: BIC={bic:.1f}, AIC={aic:.1f}")

    optimal_k = min(k_range, key=lambda k: results[k]["bic"])
    print(f"  {chamber} optimal k (GMM/BIC): {optimal_k}")
    return results, optimal_k


def plot_bic_aic(
    results: dict[int, dict],
    chamber: str,
    out_dir: Path,
) -> None:
    """Plot BIC and AIC per k for GMM model selection."""
    ks = sorted(results.keys())
    bics = [results[k]["bic"] for k in ks]
    aics = [results[k]["aic"] for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, bics, "o-", color="#4C72B0", label="BIC")
    ax.plot(ks, aics, "s--", color="#E81B23", label="AIC")
    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("Information Criterion")
    ax.set_title(f"{chamber} — GMM Model Selection (BIC / AIC)")
    ax.set_xticks(ks)

    # Annotate BIC-optimal k
    bic_optimal_k = ks[int(np.argmin(bics))]
    bic_optimal_val = min(bics)
    ax.plot(bic_optimal_k, bic_optimal_val, marker="*", markersize=14, color="#4C72B0", zorder=5)
    ax.annotate(
        f"BIC min at k={bic_optimal_k}",
        (bic_optimal_k, bic_optimal_val),
        textcoords="offset points",
        xytext=(12, 8),
        fontsize=8,
        color="#4C72B0",
        fontweight="bold",
    )

    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"bic_aic_{chamber.lower()}.png")


def plot_gmm_probabilities(
    ideal_points: pl.DataFrame,
    probs: np.ndarray,
    k: int,
    chamber: str,
    out_dir: Path,
) -> None:
    """Strip plot with color intensity = max class probability."""
    xi_vals = ideal_points["xi_mean"].to_numpy()
    max_probs = probs.max(axis=1)
    labels = probs.argmax(axis=1)

    cmap = plt.get_cmap(CLUSTER_CMAP)
    fig, ax = plt.subplots(figsize=(12, 5))

    for cluster_id in range(k):
        mask = labels == cluster_id
        if mask.any():
            ax.scatter(
                xi_vals[mask],
                np.zeros(mask.sum()),
                c=[cmap(cluster_id / max(k - 1, 1))],
                s=80,
                alpha=max_probs[mask],
                edgecolors="black",
                linewidth=0.5,
            )

    ax.set_xlabel("IRT Ideal Point (Liberal ← → Conservative)")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(f"{chamber} — GMM Cluster Probabilities (k={k})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    handles = [Patch(facecolor=cmap(i / max(k - 1, 1)), label=f"Component {i}") for i in range(k)]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir / f"gmm_probs_{chamber.lower()}.png")


# ── Phase 5b: Spectral Clustering ─────────────────────────────────────────


def run_spectral_clustering(
    kappa_matrix: pl.DataFrame,
    k_range: range,
    chamber: str,
) -> tuple[dict[int, dict], int]:
    """Run spectral clustering on Kappa agreement matrix.

    Uses the agreement matrix directly as affinity (precomputed).
    Returns (results_per_k, optimal_k_by_silhouette).
    """
    distance_arr, slugs = _kappa_to_distance(kappa_matrix)

    # Convert distance to affinity: affinity = 1 - distance = kappa
    # Ensure non-negative (clip small negatives from floating point)
    affinity = np.clip(1.0 - distance_arr, 0.0, None)
    np.fill_diagonal(affinity, 1.0)

    results: dict[int, dict] = {}
    for k in k_range:
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=RANDOM_SEED,
        )
        labels = sc.fit_predict(affinity)
        sil = (
            float(silhouette_score(distance_arr, labels, metric="precomputed"))
            if len(set(labels)) > 1
            else -1.0
        )
        results[k] = {
            "labels": labels,
            "silhouette": sil,
        }
        print(f"    k={k}: silhouette = {sil:.4f}")

    optimal_k = max(k_range, key=lambda k: results[k]["silhouette"])
    print(f"  {chamber} optimal k (spectral): {optimal_k}")
    return results, optimal_k


# ── Phase 5c: HDBSCAN on PCA Embeddings ──────────────────────────────────


def run_hdbscan_pca(
    pca_scores: pl.DataFrame,
    ideal_points: pl.DataFrame,
    chamber: str,
) -> dict:
    """Run HDBSCAN on PCA embeddings to find density-based clusters.

    Unlike k-based methods, HDBSCAN does not require specifying k and
    designates noise points (label=-1) for outlier legislators.
    Returns dict with labels, n_clusters, n_noise, noise_slugs.
    """
    slug_col = "legislator_slug"

    # Align PCA scores to ideal_points slug order
    ip_slugs = ideal_points[slug_col].to_list()
    pca_slugs = pca_scores[slug_col].to_list()
    pc_cols = [c for c in pca_scores.columns if c.startswith("PC")]

    # Use available PC columns (typically PC1-PC5 or more)
    n_pcs = min(len(pc_cols), 10)
    use_cols = pc_cols[:n_pcs]

    # Filter to legislators present in both datasets
    common_slugs = [s for s in ip_slugs if s in pca_slugs]
    pca_filtered = pca_scores.filter(pl.col(slug_col).is_in(common_slugs))
    pca_filtered = pca_filtered.sort(slug_col)

    X = pca_filtered.select(use_cols).to_numpy().copy()

    # Standardize PCA scores
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hdb = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    labels = hdb.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    ordered_slugs = pca_filtered[slug_col].to_list()
    noise_slugs = [ordered_slugs[i] for i in range(len(labels)) if labels[i] == -1]

    # Get noise legislator names for reporting
    noise_names = []
    name_map = dict(zip(ip_slugs, ideal_points["full_name"].to_list()))
    for slug in noise_slugs:
        noise_names.append(name_map.get(slug, slug))

    print(f"  {chamber}: HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
    if noise_names:
        print(f"  Noise legislators: {', '.join(noise_names)}")

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_slugs": noise_slugs,
        "noise_names": noise_names,
        "slugs": ordered_slugs,
        "n_pcs_used": n_pcs,
    }


# ── Phase 6: Cross-Method Comparison ────────────────────────────────────────


def compare_methods(
    assignments: dict[str, np.ndarray],
    chamber: str,
    *,
    party_labels: np.ndarray | None = None,
) -> dict:
    """Compute ARI between each pair of clustering methods.

    assignments: {"hierarchical": labels, "kmeans": labels, "gmm": labels}
    party_labels: optional array of party labels (same length as assignment arrays)
        for ARI-against-party diagnostic.
    Returns dict with ARI pairs and consensus info.
    """
    methods = list(assignments.keys())
    ari_matrix: dict[str, float] = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1 :]:
            l1 = assignments[m1]
            l2 = assignments[m2]
            # Align lengths (some methods may have fewer observations)
            n = min(len(l1), len(l2))
            ari = float(adjusted_rand_score(l1[:n], l2[:n]))
            key = f"{m1}_vs_{m2}"
            ari_matrix[key] = ari
            print(f"    ARI({m1} vs {m2}): {ari:.4f}")

    mean_ari = np.mean(list(ari_matrix.values())) if ari_matrix else 0.0
    print(f"  {chamber}: mean ARI across all pairs = {mean_ari:.4f}")

    # ARI against party labels (diagnostic: how well does clustering recover party?)
    ari_vs_party: float | None = None
    if party_labels is not None:
        # Use k-means labels (first in dict) for the comparison
        primary_method = methods[0] if methods else None
        if primary_method is not None:
            primary_labels = assignments[primary_method]
            n = min(len(primary_labels), len(party_labels))
            ari_vs_party = float(adjusted_rand_score(primary_labels[:n], party_labels[:n]))
            print(f"  {chamber}: ARI(kmeans vs party) = {ari_vs_party:.4f}")

    k_values = {m: len(set(assignments[m])) for m in methods}
    n_common = min(len(v) for v in assignments.values())
    return {
        "ari_matrix": ari_matrix,
        "n_common": n_common,
        "mean_ari": float(mean_ari),
        "k_values": k_values,
        "ari_vs_party": ari_vs_party,
    }


# ── Phase 7: Cluster Characterization ───────────────────────────────────────


def characterize_clusters(
    ideal_points: pl.DataFrame,
    labels: np.ndarray,
    loyalty: pl.DataFrame | None,
    k: int,
    chamber: str,
) -> pl.DataFrame:
    """Summarize cluster characteristics: party composition, IRT stats, loyalty.

    Returns a summary DataFrame.
    """
    # Build a working frame
    ip = ideal_points.with_columns(pl.Series("cluster", labels.tolist()))

    if loyalty is not None:
        ip = ip.join(
            loyalty.select("legislator_slug", "loyalty_rate"),
            on="legislator_slug",
            how="left",
        )

    rows = []
    for cluster_id in range(k):
        subset = ip.filter(pl.col("cluster") == cluster_id)
        n = subset.height
        n_r = subset.filter(pl.col("party") == "Republican").height
        n_d = subset.filter(pl.col("party") == "Democrat").height
        n_other = n - n_r - n_d
        xi_mean = float(subset["xi_mean"].mean()) if n > 0 else 0.0
        xi_median = float(subset["xi_mean"].median()) if n > 0 else 0.0
        xi_sd = float(subset["xi_sd"].mean()) if n > 0 else 0.0

        loy_mean = None
        if loyalty is not None and "loyalty_rate" in subset.columns:
            non_null = subset.filter(pl.col("loyalty_rate").is_not_null())
            if non_null.height > 0:
                loy_mean = float(non_null["loyalty_rate"].mean())

        # Heuristic label
        dominant_party = "R" if n_r > n_d else "D"
        if dominant_party == "R" and xi_mean > 1.0:
            label = "Conservative R"
        elif dominant_party == "R":
            label = "Moderate R"
        elif dominant_party == "D":
            label = "Democrat"
        else:
            label = f"Mixed ({n_r}R/{n_d}D)"

        rows.append(
            {
                "cluster": cluster_id,
                "label": label,
                "n_legislators": n,
                "n_republican": n_r,
                "n_democrat": n_d,
                "n_other": n_other,
                "pct_republican": n_r / n * 100 if n > 0 else 0,
                "xi_mean": xi_mean,
                "xi_median": xi_median,
                "avg_xi_sd": xi_sd,
                "avg_loyalty": loy_mean,
            }
        )

    summary = pl.DataFrame(rows)
    print(f"\n  {chamber} Cluster Summary:")
    for row in summary.iter_rows(named=True):
        loy_str = f", loyalty={row['avg_loyalty']:.3f}" if row["avg_loyalty"] is not None else ""
        print(
            f"    Cluster {row['cluster']} ({row['label']}): "
            f"n={row['n_legislators']}, {row['n_republican']}R/{row['n_democrat']}D, "
            f"xi_mean={row['xi_mean']:.3f}{loy_str}"
        )

    return summary


def plot_cluster_composition(
    summary: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Stacked bar chart of party composition per cluster."""
    clusters = summary["cluster"].to_list()
    labels_list = summary["label"].to_list()
    n_r = summary["n_republican"].to_numpy()
    n_d = summary["n_democrat"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(clusters))
    width = 0.6

    ax.bar(x, n_r, width, color=PARTY_COLORS["Republican"], label="Republican")
    ax.bar(x, n_d, width, bottom=n_r, color=PARTY_COLORS["Democrat"], label="Democrat")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Cluster {c}\n{lbl}" for c, lbl in zip(clusters, labels_list)],
        fontsize=9,
    )
    ax.set_ylabel("Number of Legislators")
    ax.set_title(f"{chamber} — Cluster Party Composition")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"cluster_composition_{chamber.lower()}.png")


def plot_cluster_box(
    ideal_points: pl.DataFrame,
    labels: np.ndarray,
    k: int,
    chamber: str,
    out_dir: Path,
) -> None:
    """Boxplot of xi_mean per cluster."""
    xi_vals = ideal_points["xi_mean"].to_numpy()

    data_by_cluster = [xi_vals[labels == i] for i in range(k)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data_by_cluster,
        tick_labels=[f"Cluster {i}" for i in range(k)],
        patch_artist=True,
    )

    cmap = plt.get_cmap(CLUSTER_CMAP)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i / max(k - 1, 1)))
        patch.set_alpha(0.7)

    ax.set_ylabel("IRT Ideal Point")
    ax.set_title(f"{chamber} — Ideal Point Distribution by Cluster")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"cluster_box_{chamber.lower()}.png")


# ── Phase 7b: Within-Party Clustering ────────────────────────────────────────


def run_within_party_clustering(
    ideal_points: pl.DataFrame,
    loyalty: pl.DataFrame,
    k_range: range,
    chamber: str,
    out_dir: Path,
) -> dict:
    """Cluster each party caucus separately to find intra-party structure.

    Returns dict keyed by party name (lowercase) with per-party results:
    optimal_k, silhouette_scores, labels, slugs, structure_found.
    """
    results: dict = {}

    for party in ["Republican", "Democrat"]:
        party_ip = ideal_points.filter(pl.col("party") == party)
        if party_ip.height < WITHIN_PARTY_MIN_SIZE:
            print(
                f"  {chamber} {party}: {party_ip.height} legislators "
                f"< {WITHIN_PARTY_MIN_SIZE} minimum — skipping"
            )
            results[party.lower()] = {
                "skipped": True,
                "n_legislators": party_ip.height,
                "reason": f"caucus size {party_ip.height} < {WITHIN_PARTY_MIN_SIZE}",
            }
            continue

        party_slugs = party_ip["legislator_slug"].to_list()
        xi = party_ip["xi_mean"].to_numpy().reshape(-1, 1)

        # Merge loyalty
        merged = party_ip.select("legislator_slug", "xi_mean").join(
            loyalty.select("legislator_slug", "loyalty_rate"),
            on="legislator_slug",
            how="inner",
        )
        has_2d = merged.height >= max(k_range)

        # Standardize for 2D
        if has_2d:
            X_2d = _standardize_2d(
                merged["xi_mean"].to_numpy(),
                merged["loyalty_rate"].to_numpy(),
            )

        # Limit k_range to party size
        max_k = min(max(k_range), party_ip.height - 1)
        wp_k_range = range(2, max_k + 1)

        silhouette_1d: dict[int, float] = {}
        silhouette_2d: dict[int, float] = {}
        best_labels_1d: dict[int, np.ndarray] = {}
        best_labels_2d: dict[int, np.ndarray] = {}

        for k in wp_k_range:
            # 1D
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            labels_1d = km.fit_predict(xi)
            sil_1d = float(silhouette_score(xi, labels_1d)) if len(set(labels_1d)) > 1 else -1.0
            silhouette_1d[k] = sil_1d
            best_labels_1d[k] = labels_1d
            print(f"    {party} k={k}: silhouette(1D)={sil_1d:.4f}")

            # 2D
            if has_2d:
                km2 = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                labels_2d = km2.fit_predict(X_2d)
                sil_2d = (
                    float(silhouette_score(X_2d, labels_2d)) if len(set(labels_2d)) > 1 else -1.0
                )
                silhouette_2d[k] = sil_2d
                best_labels_2d[k] = labels_2d
                print(f"    {party} k={k}: silhouette(2D)={sil_2d:.4f}")

        # Best k by 1D silhouette
        optimal_k_1d = max(wp_k_range, key=lambda k: silhouette_1d[k]) if silhouette_1d else 2
        best_sil_1d = silhouette_1d.get(optimal_k_1d, -1.0)
        structure_found = best_sil_1d >= SILHOUETTE_GOOD

        # Best k by 2D silhouette
        optimal_k_2d = None
        best_sil_2d = None
        if silhouette_2d:
            optimal_k_2d = max(wp_k_range, key=lambda k: silhouette_2d.get(k, -1.0))
            best_sil_2d = silhouette_2d.get(optimal_k_2d, -1.0)

        status = (
            "discrete subclusters found"
            if structure_found
            else ("no discrete subclusters — continuous variation")
        )
        print(
            f"  {chamber} {party}: optimal k(1D)={optimal_k_1d} "
            f"(silhouette={best_sil_1d:.4f}) — {status}"
        )

        party_key = party.lower()
        results[party_key] = {
            "skipped": False,
            "n_legislators": party_ip.height,
            "optimal_k_1d": optimal_k_1d,
            "optimal_k_2d": optimal_k_2d,
            "best_silhouette_1d": best_sil_1d,
            "best_silhouette_2d": best_sil_2d,
            "silhouette_1d": silhouette_1d,
            "silhouette_2d": silhouette_2d,
            "structure_found": structure_found,
            "labels": best_labels_1d.get(optimal_k_1d, np.array([])),
            "slugs": party_slugs,
        }

        # Plot within-party model selection
        plot_within_party_model_selection(
            silhouette_1d,
            silhouette_2d,
            party,
            chamber,
            out_dir,
        )

        # Plot within-party clusters (2D scatter)
        if has_2d:
            use_labels = best_labels_2d.get(
                optimal_k_2d or optimal_k_1d,
                best_labels_1d.get(optimal_k_1d, np.array([])),
            )
            use_k = optimal_k_2d or optimal_k_1d
            use_sil = best_sil_2d if best_sil_2d is not None else best_sil_1d
            plot_within_party_clusters(
                merged,
                use_labels,
                use_k,
                use_sil,
                party,
                chamber,
                out_dir,
            )

    return results


def plot_within_party_model_selection(
    silhouette_1d: dict[int, float],
    silhouette_2d: dict[int, float],
    party: str,
    chamber: str,
    out_dir: Path,
) -> None:
    """Silhouette scores vs k for within-party k-means."""
    if not silhouette_1d:
        return

    ks = sorted(silhouette_1d.keys())
    sil_1d = [silhouette_1d[k] for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, sil_1d, "o-", color="#4C72B0", label="1D (IRT only)")

    if silhouette_2d:
        sil_2d = [silhouette_2d.get(k, float("nan")) for k in ks]
        ax.plot(ks, sil_2d, "s--", color="#E81B23", label="2D (IRT + loyalty)")

    # Threshold line
    ax.axhline(SILHOUETTE_GOOD, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(
        ks[-1] + 0.15,
        SILHOUETTE_GOOD,
        "Good threshold (0.50)",
        va="center",
        fontsize=7,
        color="#888888",
    )

    # Annotate optimal k (1D)
    optimal_k = ks[int(np.argmax(sil_1d))]
    best_sil = max(sil_1d)
    ax.plot(optimal_k, best_sil, marker="*", markersize=14, color="#4C72B0", zorder=5)
    ax.annotate(
        f"k={optimal_k} (sil={best_sil:.2f})",
        (optimal_k, best_sil),
        textcoords="offset points",
        xytext=(12, -8),
        fontsize=8,
        color="#4C72B0",
        fontweight="bold",
    )

    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"{chamber} — Within-{party} Model Selection")
    ax.set_xticks(ks)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    party_key = party.lower()
    save_fig(fig, out_dir / f"within_party_model_sel_{party_key}_{chamber.lower()}.png")


def plot_within_party_clusters(
    merged: pl.DataFrame,
    labels: np.ndarray,
    k: int,
    best_silhouette: float,
    party: str,
    chamber: str,
    out_dir: Path,
) -> None:
    """2D scatter: xi_mean (x) vs loyalty_rate (y), colored by within-party cluster."""
    xi_vals = merged["xi_mean"].to_numpy()
    loy_vals = merged["loyalty_rate"].to_numpy()

    cmap = plt.get_cmap(CLUSTER_CMAP)
    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in range(k):
        mask = labels == cluster_id
        if mask.any():
            ax.scatter(
                xi_vals[mask],
                loy_vals[mask],
                c=[cmap(cluster_id / max(k - 1, 1))],
                marker="o",
                s=60,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                label=f"Subcluster {cluster_id}",
            )

    # Annotate notable legislators (extreme IRT or low loyalty within party)
    if "legislator_slug" in merged.columns:
        slugs = merged["legislator_slug"].to_list()
        for i in range(merged.height):
            slug = slugs[i]
            is_notable = loy_vals[i] < 0.6 or (
                xi_vals[i] > np.percentile(xi_vals, 95) or xi_vals[i] < np.percentile(xi_vals, 5)
            )
            if is_notable:
                # Use slug as label (short)
                short = slug.split("_")[1] if "_" in slug else slug
                ax.annotate(
                    short.title(),
                    (xi_vals[i], loy_vals[i]),
                    fontsize=6,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    # If no structure, overlay annotation
    if best_silhouette < SILHOUETTE_GOOD:
        ax.text(
            0.5,
            0.97,
            f"No discrete subclusters (silhouette={best_silhouette:.2f} < {SILHOUETTE_GOOD})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#AA0000",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF3F3", "edgecolor": "#AA0000"},
        )

    ax.set_xlabel("IRT Ideal Point (Liberal ← → Conservative)")
    ax.set_ylabel("Party Loyalty Rate")
    ax.set_title(f"{chamber} — Within-{party} Clustering (k={k})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    party_key = party.lower()
    save_fig(fig, out_dir / f"within_party_clusters_{party_key}_{chamber.lower()}.png")


# ── Phase 8: Veto Override Subgroup ──────────────────────────────────────────


def analyze_veto_overrides(
    vote_matrix: pl.DataFrame,
    rollcalls: pl.DataFrame,
    ideal_points: pl.DataFrame,
    full_labels: np.ndarray,
    k: int,
    chamber: str,
) -> dict:
    """Analyze voting patterns on veto override votes.

    Compares cluster assignments on override votes vs full dataset.
    """
    slug_col = "legislator_slug"

    # Identify veto override vote IDs
    override_ids = set(
        rollcalls.filter(
            pl.col("motion").str.to_lowercase().str.contains("veto")
            & (pl.col("chamber") == chamber)
        )["vote_id"].to_list()
    )

    vote_cols = [c for c in vote_matrix.columns if c != slug_col]
    override_cols = [c for c in vote_cols if c in override_ids]

    if len(override_cols) < 2:
        print(f"  {chamber}: Only {len(override_cols)} veto override votes — skipping subgroup")
        return {"n_override_votes": len(override_cols), "skipped": True}

    print(f"  {chamber}: {len(override_cols)} veto override votes")

    # Build override vote sub-matrix
    override_matrix = vote_matrix.select([slug_col, *override_cols])

    # Compute per-legislator Yea rate on overrides
    slugs = override_matrix[slug_col].to_list()
    override_yea_rates = []
    for row in override_matrix.iter_rows(named=True):
        votes = [row[c] for c in override_cols if row[c] is not None]
        yea_rate = sum(votes) / len(votes) if votes else None
        override_yea_rates.append(yea_rate)

    # Create summary
    override_df = pl.DataFrame(
        {
            "legislator_slug": slugs,
            "override_yea_rate": override_yea_rates,
            "n_override_votes": [
                sum(1 for c in override_cols if row[c] is not None)
                for row in override_matrix.iter_rows(named=True)
            ],
        }
    ).drop_nulls(subset=["override_yea_rate"])

    # Join with cluster labels and party
    override_df = override_df.join(
        ideal_points.select("legislator_slug", "party", "full_name", "xi_mean"),
        on="legislator_slug",
        how="inner",
    )

    # Add full-dataset cluster
    slug_to_label = dict(
        zip(
            ideal_points["legislator_slug"].to_list(),
            full_labels.tolist(),
        )
    )
    override_df = override_df.with_columns(
        pl.col("legislator_slug").replace_strict(slug_to_label, default=-1).alias("full_cluster")
    )

    # Per-cluster override statistics
    cluster_stats = (
        override_df.group_by("full_cluster")
        .agg(
            pl.col("override_yea_rate").mean().alias("mean_override_yea_rate"),
            pl.col("override_yea_rate").std().alias("std_override_yea_rate"),
            pl.len().alias("n_legislators"),
        )
        .sort("full_cluster")
    )

    print("  Override Yea rates by cluster:")
    for row in cluster_stats.iter_rows(named=True):
        std_val = row["std_override_yea_rate"]
        std_str = f" +/- {std_val:.3f}" if std_val is not None else ""
        print(
            f"    Cluster {row['full_cluster']}: "
            f"mean={row['mean_override_yea_rate']:.3f}{std_str} "
            f"(n={row['n_legislators']})"
        )

    # Cross-party coalition: legislators who voted Yea on overrides
    # from both parties (unusual alignments)
    high_override_yea = override_df.filter(pl.col("override_yea_rate") > 0.7)
    n_r_yea = high_override_yea.filter(pl.col("party") == "Republican").height
    n_d_yea = high_override_yea.filter(pl.col("party") == "Democrat").height
    print(f"  High override support (>70% Yea): {n_r_yea}R, {n_d_yea}D")

    return {
        "n_override_votes": len(override_cols),
        "skipped": False,
        "cluster_stats": cluster_stats,
        "override_df": override_df,
        "n_high_yea_r": n_r_yea,
        "n_high_yea_d": n_d_yea,
    }


# ── Phase 9: Sensitivity Analysis ───────────────────────────────────────────


def run_sensitivity_clustering(
    vote_matrix: pl.DataFrame,
    ideal_points: pl.DataFrame,
    kappa_matrix: pl.DataFrame,
    default_labels: np.ndarray,
    default_k: int,
    rollcalls: pl.DataFrame,
    chamber: str,
) -> dict:
    """Re-run hierarchical + k-means at alternative filter threshold.

    Since IRT ideal points don't change (those are upstream), we re-run
    clustering at k=2, k=3, k=4 and compare via ARI.
    """
    findings: dict = {}

    # Compare k=2, k=3, k=4 via ARI against default
    xi = ideal_points["xi_mean"].to_numpy().reshape(-1, 1)
    for alt_k in [2, 3, 4]:
        km = KMeans(n_clusters=alt_k, random_state=RANDOM_SEED, n_init=10)
        alt_labels = km.fit_predict(xi)
        n = min(len(default_labels), len(alt_labels))
        ari = float(adjusted_rand_score(default_labels[:n], alt_labels[:n]))
        findings[f"k{default_k}_vs_k{alt_k}"] = {
            "ari": ari,
            "default_k": default_k,
            "alt_k": alt_k,
        }
        print(f"    ARI(k={default_k} vs k={alt_k}): {ari:.4f}")

    # Re-run hierarchical at alternative k values
    distance_arr, kappa_slugs = _kappa_to_distance(kappa_matrix)
    condensed = squareform(distance_arr, checks=False)
    Z = linkage(condensed, method=LINKAGE_METHOD)

    # Align hierarchical labels to IRT slug order (kappa matrix may have different ordering)
    irt_slugs = ideal_points["legislator_slug"].to_list()

    for alt_k in [2, 3, 4]:
        hier_labels_raw = cut_tree(Z, n_clusters=alt_k).flatten()
        slug_to_hier = dict(zip(kappa_slugs, hier_labels_raw.tolist()))
        hier_labels = np.array([slug_to_hier.get(s, -1) for s in irt_slugs])
        n = min(len(default_labels), len(hier_labels))
        ari = float(adjusted_rand_score(default_labels[:n], hier_labels[:n]))
        findings[f"kmeans_k{default_k}_vs_hier_k{alt_k}"] = {
            "ari": ari,
            "method_a": f"kmeans_k{default_k}",
            "method_b": f"hierarchical_k{alt_k}",
        }
        print(f"    ARI(kmeans_k={default_k} vs hier_k={alt_k}): {ari:.4f}")

    return findings


# ── Phase 10: Manifest + Report ──────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


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
    pca_dir = resolve_upstream_dir(
        "02_pca",
        results_root,
        args.run_id,
        Path(args.pca_dir) if args.pca_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="05_clustering",
        params=vars(args),
        primer=CLUSTERING_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature Clustering Analysis — Session {args.session}")
        print(f"Data:      {data_dir}")
        print(f"EDA:       {eda_dir}")
        print(f"IRT:       {irt_dir}")
        print(f"PCA:       {pca_dir}")
        print(f"Output:    {ctx.run_dir}")

        # ── Phase 1: Load data ──
        print_header("PHASE 1: LOADING DATA")
        irt_house, irt_senate = load_irt_ideal_points(irt_dir)
        kappa_house, kappa_senate = load_agreement_matrices(eda_dir)
        vm_house, vm_senate = load_vote_matrices(eda_dir)
        pca_house, pca_senate = load_pca_scores(pca_dir)
        rollcalls, legislators = load_metadata(data_dir)

        print(f"  IRT House:    {irt_house.height} legislators")
        print(f"  IRT Senate:   {irt_senate.height} legislators")
        print(f"  Kappa House:  {kappa_house.height} x {len(kappa_house.columns) - 1}")
        print(f"  Kappa Senate: {kappa_senate.height} x {len(kappa_senate.columns) - 1}")
        print(f"  Vote matrix House:  {vm_house.height} x {len(vm_house.columns) - 1}")
        print(f"  Vote matrix Senate: {vm_senate.height} x {len(vm_senate.columns) - 1}")
        print(f"  Rollcalls: {rollcalls.height}")
        print(f"  Legislators: {legislators.height}")

        chamber_configs = [
            ("House", irt_house, kappa_house, vm_house, pca_house),
            ("Senate", irt_senate, kappa_senate, vm_senate, pca_senate),
        ]

        results: dict[str, dict] = {}

        for chamber, irt_ip, kappa_mat, vm, pca_scores in chamber_configs:
            if irt_ip.height < 5:
                print(f"\n  Skipping {chamber}: too few legislators ({irt_ip.height})")
                continue

            chamber_results: dict = {
                "ideal_points": irt_ip,
                "kappa_matrix": kappa_mat,
                "vote_matrix": vm,
                "pca_scores": pca_scores,
            }

            # ── Phase 2: Party Loyalty ──
            print_header(f"PHASE 2: PARTY LOYALTY — {chamber}")
            loyalty = compute_party_loyalty(vm, irt_ip, chamber)
            loyalty.write_parquet(ctx.data_dir / f"party_loyalty_{chamber.lower()}.parquet")
            print(f"  Saved: party_loyalty_{chamber.lower()}.parquet")
            chamber_results["loyalty"] = loyalty

            # ── Phase 3: Hierarchical Clustering ──
            print_header(f"PHASE 3: HIERARCHICAL CLUSTERING — {chamber}")
            Z, coph_corr, slugs, distance_arr = run_hierarchical(kappa_mat, chamber)
            hier_scores, hier_optimal_k = find_optimal_k_hierarchical(
                Z, distance_arr, K_RANGE, chamber
            )

            # Use override k if provided
            use_k = args.k if args.k is not None else hier_optimal_k

            # Cut at optimal k and at COMPARISON_K for comparison
            hier_labels_optimal = cut_tree(Z, n_clusters=use_k).flatten()
            hier_labels_default = cut_tree(Z, n_clusters=COMPARISON_K).flatten()

            # Save hierarchical assignments
            hier_df = pl.DataFrame(
                {
                    "legislator_slug": slugs,
                    f"cluster_k{use_k}": hier_labels_optimal.tolist(),
                    f"cluster_k{COMPARISON_K}": hier_labels_default.tolist(),
                }
            )
            hier_df.write_parquet(
                ctx.data_dir / f"hierarchical_assignments_{chamber.lower()}.parquet"
            )
            print(f"  Saved: hierarchical_assignments_{chamber.lower()}.parquet")

            plot_dendrogram(Z, slugs, irt_ip, chamber, ctx.plots_dir)
            plot_voting_blocs(Z, slugs, irt_ip, chamber, ctx.plots_dir)
            plot_polar_dendrogram(Z, slugs, irt_ip, chamber, ctx.plots_dir)
            plot_icicle(Z, slugs, irt_ip, chamber, ctx.plots_dir)

            chamber_results["hierarchical"] = {
                "Z": Z,
                "cophenetic_r": coph_corr,
                "silhouette_scores": hier_scores,
                "optimal_k": use_k,
                "labels": hier_labels_optimal,
                "labels_default_k": hier_labels_default,
                "slugs": slugs,
            }

            # ── Phase 4: K-Means on IRT ──
            print_header(f"PHASE 4: K-MEANS ON IRT — {chamber}")
            km_results, km_optimal_k = run_kmeans_irt(irt_ip, loyalty, K_RANGE, chamber)

            km_k = args.k if args.k is not None else km_optimal_k
            km_labels = km_results[km_k]["labels_1d"]

            # Save k-means assignments
            km_df = pl.DataFrame(
                {
                    "legislator_slug": irt_ip["legislator_slug"].to_list(),
                    f"cluster_k{km_k}": km_labels.tolist(),
                    "distance_to_centroid": [
                        float(abs(xi - km_results[km_k]["centroids_1d"][lab]))
                        for xi, lab in zip(irt_ip["xi_mean"].to_list(), km_labels.tolist())
                    ],
                }
            )
            if "labels_2d" in km_results.get(km_k, {}):
                labels_2d = km_results[km_k]["labels_2d"].tolist()
                if len(labels_2d) == km_df.height:
                    km_df = km_df.with_columns(pl.Series(f"cluster_2d_k{km_k}", labels_2d))
                else:
                    print(
                        f"  Note: 2D labels length ({len(labels_2d)}) != legislator count "
                        f"({km_df.height}) — skipping 2D cluster column"
                    )
            km_df.write_parquet(ctx.data_dir / f"kmeans_assignments_{chamber.lower()}.parquet")
            ctx.export_csv(km_df, f"kmeans_assignments_{chamber.lower()}.csv", f"K-means cluster assignments for {chamber} legislators")
            print(f"  Saved: kmeans_assignments_{chamber.lower()}.parquet")

            plot_elbow_silhouette(km_results, chamber, ctx.plots_dir)
            plot_irt_clusters(irt_ip, km_labels, km_k, chamber, ctx.plots_dir)
            plot_irt_loyalty_clusters(irt_ip, loyalty, km_labels, km_k, chamber, ctx.plots_dir)

            chamber_results["kmeans"] = {
                "results": km_results,
                "optimal_k": km_k,
                "labels": km_labels,
            }

            # Save model selection data
            model_sel_rows = []
            for k_val in K_RANGE:
                row = {
                    "k": k_val,
                    "kmeans_inertia": km_results[k_val]["inertia"],
                    "kmeans_silhouette_1d": km_results[k_val]["silhouette_1d"],
                    "hier_silhouette": hier_scores.get(k_val),
                }
                if "silhouette_2d" in km_results.get(k_val, {}):
                    row["kmeans_silhouette_2d"] = km_results[k_val]["silhouette_2d"]
                model_sel_rows.append(row)

            # ── Phase 5: GMM ──
            gmm_optimal_k = None
            gmm_labels = None
            gmm_probs = None
            if not args.skip_gmm:
                print_header(f"PHASE 5: GMM ON IRT — {chamber}")
                gmm_results, gmm_optimal_k = run_gmm_irt(irt_ip, K_RANGE, chamber)

                gmm_k = args.k if args.k is not None else gmm_optimal_k
                gmm_labels = gmm_results[gmm_k]["labels"]
                gmm_probs = gmm_results[gmm_k]["probs"]

                # Save GMM assignments
                gmm_data = {
                    "legislator_slug": irt_ip["legislator_slug"].to_list(),
                    "cluster": gmm_labels.tolist(),
                    "max_prob": gmm_results[gmm_k]["max_prob"].tolist(),
                }
                for comp_i in range(gmm_k):
                    gmm_data[f"prob_{comp_i}"] = gmm_probs[:, comp_i].tolist()
                gmm_df = pl.DataFrame(gmm_data)
                gmm_df.write_parquet(ctx.data_dir / f"gmm_assignments_{chamber.lower()}.parquet")
                print(f"  Saved: gmm_assignments_{chamber.lower()}.parquet")

                plot_bic_aic(gmm_results, chamber, ctx.plots_dir)
                plot_gmm_probabilities(irt_ip, gmm_probs, gmm_k, chamber, ctx.plots_dir)

                # Add GMM BIC/AIC to model selection
                for i, k_val in enumerate(K_RANGE):
                    model_sel_rows[i]["gmm_bic"] = gmm_results[k_val]["bic"]
                    model_sel_rows[i]["gmm_aic"] = gmm_results[k_val]["aic"]

                chamber_results["gmm"] = {
                    "results": gmm_results,
                    "optimal_k": gmm_k,
                    "labels": gmm_labels,
                    "probs": gmm_probs,
                }
            else:
                print_header(f"PHASE 5: GMM (SKIPPED) — {chamber}")

            # ── Phase 5b: Spectral Clustering ──
            print_header(f"PHASE 5b: SPECTRAL CLUSTERING — {chamber}")
            spectral_results, spectral_optimal_k = run_spectral_clustering(
                kappa_mat, K_RANGE, chamber
            )
            spectral_k = args.k if args.k is not None else spectral_optimal_k
            spectral_labels = spectral_results[spectral_k]["labels"]

            # Save spectral assignments
            spectral_df = pl.DataFrame(
                {
                    "legislator_slug": _kappa_to_distance(kappa_mat)[1],
                    f"cluster_k{spectral_k}": spectral_labels.tolist(),
                }
            )
            spectral_df.write_parquet(
                ctx.data_dir / f"spectral_assignments_{chamber.lower()}.parquet"
            )
            print(f"  Saved: spectral_assignments_{chamber.lower()}.parquet")

            # Add spectral silhouette to model selection
            for i, k_val in enumerate(K_RANGE):
                model_sel_rows[i]["spectral_silhouette"] = spectral_results[k_val]["silhouette"]

            chamber_results["spectral"] = {
                "results": spectral_results,
                "optimal_k": spectral_k,
                "labels": spectral_labels,
            }

            # ── Phase 5c: HDBSCAN on PCA ──
            print_header(f"PHASE 5c: HDBSCAN ON PCA — {chamber}")
            hdbscan_result = run_hdbscan_pca(pca_scores, irt_ip, chamber)
            chamber_results["hdbscan"] = hdbscan_result

            if hdbscan_result["n_clusters"] > 0:
                hdb_df = pl.DataFrame(
                    {
                        "legislator_slug": hdbscan_result["slugs"],
                        "cluster": hdbscan_result["labels"].tolist(),
                    }
                )
                hdb_df.write_parquet(
                    ctx.data_dir / f"hdbscan_assignments_{chamber.lower()}.parquet"
                )
                print(f"  Saved: hdbscan_assignments_{chamber.lower()}.parquet")

            model_sel_df = pl.DataFrame(model_sel_rows)
            model_sel_df.write_parquet(ctx.data_dir / f"model_selection_{chamber.lower()}.parquet")

            # ── Phase 6: Cross-Method Comparison ──
            print_header(f"PHASE 6: CROSS-METHOD COMPARISON — {chamber}")
            # Align hierarchical labels to IRT slug order
            slug_to_hier = dict(zip(slugs, hier_labels_optimal.tolist()))
            hier_aligned = np.array(
                [slug_to_hier.get(s, -1) for s in irt_ip["legislator_slug"].to_list()]
            )

            # Align spectral labels to IRT slug order
            spectral_slugs = _kappa_to_distance(kappa_mat)[1]
            slug_to_spectral = dict(zip(spectral_slugs, spectral_labels.tolist()))
            spectral_aligned = np.array(
                [slug_to_spectral.get(s, -1) for s in irt_ip["legislator_slug"].to_list()]
            )

            method_assignments = {
                "hierarchical": hier_aligned,
                "kmeans": km_labels,
                "spectral": spectral_aligned,
            }
            if gmm_labels is not None:
                method_assignments["gmm"] = gmm_labels

            # Party labels aligned to IRT slug order for ARI-vs-party diagnostic
            party_arr = np.array(irt_ip["party"].to_list())
            comparison = compare_methods(
                method_assignments, chamber, party_labels=party_arr
            )
            chamber_results["comparison"] = comparison

            # ── Phase 7: Cluster Characterization ──
            print_header(f"PHASE 7: CLUSTER CHARACTERIZATION — {chamber}")
            # Use k-means labels as primary (centroid-based, on IRT)
            primary_k = km_k
            primary_labels = km_labels

            summary = characterize_clusters(irt_ip, primary_labels, loyalty, primary_k, chamber)
            chamber_results["cluster_summary"] = summary

            plot_cluster_composition(summary, chamber, ctx.plots_dir)
            plot_cluster_box(irt_ip, primary_labels, primary_k, chamber, ctx.plots_dir)

            # Flag notable legislators from data: lowest loyalty + highest uncertainty
            ip_with_loy = irt_ip.join(
                loyalty.select("legislator_slug", "loyalty_rate"),
                on="legislator_slug",
                how="left",
            )
            majority = ip_with_loy.group_by("party").len().sort("len", descending=True)["party"][0]
            low_loy = (
                ip_with_loy.filter(
                    pl.col("party") == majority, pl.col("loyalty_rate").is_not_null()
                )
                .sort("loyalty_rate")
                .head(2)["legislator_slug"]
                .to_list()
            )
            high_sd = irt_ip.sort("xi_sd", descending=True).head(2)["legislator_slug"].to_list()
            flagged_slugs = list(dict.fromkeys(low_loy + high_sd))
            ip_slugs = irt_ip["legislator_slug"].to_list()
            flagged = []
            for fs in flagged_slugs:
                if fs in ip_slugs:
                    idx = ip_slugs.index(fs)
                    loy_row = loyalty.filter(pl.col("legislator_slug") == fs)
                    loy_val = float(loy_row["loyalty_rate"][0]) if loy_row.height > 0 else None
                    flagged.append(
                        {
                            "legislator_slug": fs,
                            "full_name": irt_ip["full_name"][idx],
                            "party": irt_ip["party"][idx],
                            "xi_mean": float(irt_ip["xi_mean"][idx]),
                            "xi_sd": float(irt_ip["xi_sd"][idx]),
                            "cluster": int(primary_labels[idx]),
                            "loyalty_rate": loy_val,
                        }
                    )
                    loy_str = f", loyalty={loy_val:.3f}" if loy_val is not None else ""
                    print(
                        f"  {irt_ip['full_name'][idx]}: cluster={primary_labels[idx]}, "
                        f"xi={float(irt_ip['xi_mean'][idx]):+.3f}{loy_str}"
                    )
            chamber_results["flagged_legislators"] = flagged

            # ── Phase 7b: Within-Party Clustering ──
            print_header(f"PHASE 7b: WITHIN-PARTY CLUSTERING — {chamber}")
            within_party = run_within_party_clustering(
                irt_ip, loyalty, K_RANGE, chamber, ctx.plots_dir
            )
            chamber_results["within_party"] = within_party

            # Save within-party assignments
            for party_key, party_data in within_party.items():
                if isinstance(party_data, dict) and not party_data.get("skipped", True):
                    wp_labels = party_data.get("labels")
                    wp_slugs = party_data.get("slugs")
                    if wp_labels is not None and wp_slugs is not None and len(wp_labels) > 0:
                        wp_df = pl.DataFrame(
                            {
                                "legislator_slug": wp_slugs,
                                "within_cluster": wp_labels.tolist(),
                            }
                        )
                        wp_df.write_parquet(
                            ctx.data_dir / f"within_party_{party_key}_{chamber.lower()}.parquet"
                        )
                        print(f"  Saved: within_party_{party_key}_{chamber.lower()}.parquet")

            # ── Phase 8: Veto Override Subgroup ──
            print_header(f"PHASE 8: VETO OVERRIDE SUBGROUP — {chamber}")
            override_results = analyze_veto_overrides(
                vm, rollcalls, irt_ip, primary_labels, primary_k, chamber
            )
            chamber_results["veto_overrides"] = override_results

            # ── Phase 9: Sensitivity Analysis ──
            if not args.skip_sensitivity:
                print_header(f"PHASE 9: SENSITIVITY ANALYSIS — {chamber}")
                sensitivity = run_sensitivity_clustering(
                    vm, irt_ip, kappa_mat, primary_labels, primary_k, rollcalls, chamber
                )
                chamber_results["sensitivity"] = sensitivity
            else:
                print_header(f"PHASE 9: SENSITIVITY (SKIPPED) — {chamber}")
                chamber_results["sensitivity"] = {}

            results[chamber] = chamber_results

        # ── Phase 10: Manifest + Report ──
        print_header("PHASE 10: FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "clustering",
            "constants": {
                "RANDOM_SEED": RANDOM_SEED,
                "K_RANGE": list(K_RANGE),
                "COMPARISON_K": COMPARISON_K,
                "LINKAGE_METHOD": LINKAGE_METHOD,
                "COPHENETIC_THRESHOLD": COPHENETIC_THRESHOLD,
                "SILHOUETTE_GOOD": SILHOUETTE_GOOD,
                "GMM_COVARIANCE": GMM_COVARIANCE,
                "GMM_N_INIT": GMM_N_INIT,
                "MINORITY_THRESHOLD": MINORITY_THRESHOLD,
                "SENSITIVITY_THRESHOLD": SENSITIVITY_THRESHOLD,
                "MIN_VOTES": MIN_VOTES,
                "CONTESTED_PARTY_THRESHOLD": CONTESTED_PARTY_THRESHOLD,
            },
            "k_override": args.k,
            "skip_gmm": args.skip_gmm,
            "skip_sensitivity": args.skip_sensitivity,
        }

        for chamber, result in results.items():
            ch = chamber.lower()
            manifest[f"{ch}_n_legislators"] = result["ideal_points"].height
            manifest[f"{ch}_cophenetic_r"] = result["hierarchical"]["cophenetic_r"]
            manifest[f"{ch}_hier_optimal_k"] = result["hierarchical"]["optimal_k"]
            manifest[f"{ch}_kmeans_optimal_k"] = result["kmeans"]["optimal_k"]
            if "gmm" in result:
                manifest[f"{ch}_gmm_optimal_k"] = result["gmm"]["optimal_k"]
            if "spectral" in result:
                manifest[f"{ch}_spectral_optimal_k"] = result["spectral"]["optimal_k"]
            if "hdbscan" in result:
                manifest[f"{ch}_hdbscan_n_clusters"] = result["hdbscan"]["n_clusters"]
                manifest[f"{ch}_hdbscan_n_noise"] = result["hdbscan"]["n_noise"]
                manifest[f"{ch}_hdbscan_noise_legislators"] = result["hdbscan"]["noise_names"]
            if result.get("comparison"):
                manifest[f"{ch}_cross_method_ari"] = result["comparison"]["ari_matrix"]
                manifest[f"{ch}_mean_ari"] = result["comparison"]["mean_ari"]
            if result.get("sensitivity"):
                manifest[f"{ch}_sensitivity"] = result["sensitivity"]
            if result.get("within_party"):
                wp_summary = {}
                for pk, pd in result["within_party"].items():
                    if isinstance(pd, dict):
                        wp_summary[pk] = {
                            k: v for k, v in pd.items() if k not in ("labels", "slugs")
                        }
                manifest[f"{ch}_within_party"] = wp_summary

        save_filtering_manifest(manifest, ctx.run_dir)

        # ── HTML report ──
        print_header("HTML REPORT")
        build_clustering_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
            skip_gmm=args.skip_gmm,
            skip_sensitivity=args.skip_sensitivity,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
