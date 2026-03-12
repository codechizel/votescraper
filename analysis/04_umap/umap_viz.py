"""
Kansas Legislature — UMAP Ideological Landscape (Phase 2b)

Non-linear dimensionality reduction on the binary vote matrix using UMAP.
Produces an intuitive "ideological map" where nearby legislators vote alike.
More accessible to nontechnical audiences than PCA scatter plots.

Named umap_viz.py (not umap.py) to avoid shadowing the umap package.

Covers analytic method 11 from `Analytic_Methods/`.

Usage:
  uv run python analysis/umap_viz.py [--session 2025-26] [--eda-dir ...] \
      [--n-neighbors 15] [--min-dist 0.1] [--skip-sensitivity]

Outputs (in results/<session>/umap/<date>/):
  - data/:   Parquet files (UMAP embeddings, sensitivity sweep)
  - plots/:  PNG visualizations (landscape, PC1 gradient, IRT gradient, sensitivity)
  - filtering_manifest.json, run_info.json, run_log.txt
  - umap_report.html
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
from scipy.spatial import procrustes
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import load_legislators, print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import load_legislators, print_header, save_fig

try:
    from analysis.umap_report import build_umap_report
except ModuleNotFoundError:
    from umap_report import build_umap_report  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────
# Written to results/<session>/umap/README.md by RunContext on each run.

UMAP_PRIMER = """\
# UMAP Ideological Landscape

## Purpose

UMAP (Uniform Manifold Approximation and Projection) maps legislators into a 2D
space where **proximity equals voting similarity**. Unlike PCA (which finds the
best straight-line projection), UMAP preserves *nonlinear* relationships — it
can reveal factions, sub-clusters, and bridge legislators that blend together in
a linear PCA plot.

The result is an intuitive "ideological map" suitable for nontechnical audiences:
legislators who vote alike are nearby, legislators who vote differently are far
apart. Color by party to see the partisan split; color by PCA or IRT scores to
validate against upstream methods.

Covers analytic method 11 from `Analytic_Methods/`.

## Method

1. **Load filtered vote matrices** from the EDA phase (parquet files).
2. **Impute missing values** with row-mean (each legislator's Yea base rate).
3. **Fit UMAP** with cosine metric, n_neighbors=15, min_dist=0.1.
4. **Orient UMAP1** so Republicans are positive (same convention as PCA).
5. **Sensitivity sweep** — re-fit with n_neighbors in [5, 15, 30, 50], compare
   embeddings using Procrustes similarity (rotation-invariant).
6. **Validation** — Spearman correlation of UMAP1 vs PCA PC1 and IRT ideal points.

## Inputs

Reads from `results/<session>/eda/latest/data/`:
- `vote_matrix_house_filtered.parquet` — House binary vote matrix (EDA-filtered)
- `vote_matrix_senate_filtered.parquet` — Senate binary vote matrix (EDA-filtered)

Reads from `results/<session>/pca/latest/data/` (optional, for validation):
- `pc_scores_house.parquet`, `pc_scores_senate.parquet`

Reads from `results/<session>/irt/latest/data/` (optional, for validation):
- `ideal_points_house.parquet`, `ideal_points_senate.parquet`

Reads from `data/{legislature}_{start}-{end}/`:
- `{output_name}_legislators.csv` — Legislator metadata (names, parties, districts)

## Outputs

All outputs land in `results/<session>/umap/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `umap_embedding_house.parquet` | House legislator UMAP1-2 coordinates + metadata |
| `umap_embedding_senate.parquet` | Senate legislator UMAP1-2 coordinates + metadata |
| `sensitivity_sweep.parquet` | Procrustes similarities across n_neighbors values |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `umap_landscape_house.png` | UMAP1 vs UMAP2 scatter, party-colored, outlier labels |
| `umap_landscape_senate.png` | Landscape for Senate |
| `umap_pc1_gradient_house.png` | UMAP colored by PCA PC1 scores (RdBu_r gradient) |
| `umap_pc1_gradient_senate.png` | PC1 gradient for Senate |
| `umap_irt_gradient_house.png` | UMAP colored by IRT ideal points (RdBu_r gradient) |
| `umap_irt_gradient_senate.png` | IRT gradient for Senate |
| `umap_sensitivity_house.png` | 4-panel comparison across n_neighbors values |
| `umap_sensitivity_senate.png` | Sensitivity grid for Senate |

### Root files

| File | Description |
|------|-------------|
| `filtering_manifest.json` | EDA source, UMAP parameters, validation metrics |
| `run_info.json` | Git commit, timestamp, Python version, parameters |
| `run_log.txt` | Full console output from the run |
| `umap_report.html` | Self-contained HTML report with all tables and figures |

## Interpretation Guide

This is a **map of voting behavior**. Read it like a geographic map where
geography is replaced by voting patterns.

- **Proximity = similarity.** Two legislators near each other vote the same
  way most of the time. Two legislators far apart disagree often.
- **The two clusters are the two parties.** The gap between them is the
  partisan divide.
- **Position within a cluster** shows where a legislator falls within their
  party. Legislators at the edge nearest the other party are moderates.
  Legislators at the far edge are hardliners.
- **Isolated points are loners.** A legislator sitting away from their own
  party's cluster has an unusual voting pattern — low participation
  (imputation artifact), contrarianism, or genuine ideological independence.
- **Do not interpret the axes.** UMAP1 and UMAP2 are arbitrary coordinates
  — they do not measure ideology, partisanship, or any specific concept.
  UMAP1 is oriented so Republicans are positive only for visual consistency
  with PCA. The axes could be rotated and the map would be equally valid.
  Only the distances and relative positions carry meaning.
- **Do not compare axis values across chambers.** House UMAP1=+3 and Senate
  UMAP1=+3 do not mean the same thing. Each chamber's embedding is
  independent.

The PCA and IRT reports provide quantitative ideology scores. The UMAP
provides the picture — the visualization you show a journalist or
constituent to say "here is where your legislator sits relative to
everyone else."

## Caveats

- UMAP is stochastic — results vary slightly across runs (mitigated by
  fixed random_state=42).
- The axes (UMAP1, UMAP2) have no inherent meaning. Only distances and
  relative positions are interpretable.
- Small datasets (~40 senators) can produce unstable embeddings. Check
  the sensitivity sweep for robustness.
- Row-mean imputation treats absences as uninformative. Legislators with
  many absences may appear artificially moderate or may land in the wrong
  party cluster entirely (e.g., Silas Miller). Cross-party outliers are
  annotated on the plot when detected.
"""

# ── Constants ────────────────────────────────────────────────────────────────
# Explicit, named constants per the analytic-workflow rules.

DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_METRIC = "cosine"
RANDOM_STATE = 42
SENSITIVITY_N_NEIGHBORS = [5, 15, 30, 50]
STABILITY_SEEDS = [42, 123, 456, 789, 1337]
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
HIGH_IMPUTATION_PCT = 50.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature UMAP Visualization")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help="UMAP n_neighbors parameter",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=DEFAULT_MIN_DIST,
        help="UMAP min_dist parameter",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis (faster, for debugging)",
    )
    return parser.parse_args()


def find_irt_column(df: pl.DataFrame) -> str | None:
    """Find the IRT ideal point column name in a DataFrame.

    Tries xi_mean first (our IRT output), then common alternatives.
    """
    for candidate in ["xi_mean", "ideal_point", "theta", "xi"]:
        if candidate in df.columns:
            return candidate
    return None


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_eda_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """Load filtered vote matrices from the EDA phase output.

    Returns (house_filtered, senate_filtered), or None if unavailable.
    """
    house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
    senate_path = eda_dir / "data" / "vote_matrix_senate_filtered.parquet"
    if not house_path.exists() or not senate_path.exists():
        return None
    house = pl.read_parquet(house_path)
    senate = pl.read_parquet(senate_path)
    return house, senate


def load_pca_scores(pca_dir: Path) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load PCA scores for validation. Returns (house, senate) or None if unavailable."""
    house, senate = None, None
    try:
        house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
    except OSError:
        pass
    try:
        senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
    except OSError:
        pass
    return house, senate


def load_irt_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT ideal points for validation. Returns (house, senate) or None."""
    house, senate = None, None
    try:
        house = pl.read_parquet(irt_dir / "data" / "ideal_points_house.parquet")
    except OSError:
        pass
    try:
        senate = pl.read_parquet(irt_dir / "data" / "ideal_points_senate.parquet")
    except OSError:
        pass
    return house, senate


# ── Phase 2: UMAP per Chamber ───────────────────────────────────────────────


def impute_vote_matrix(matrix: pl.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert polars vote matrix to numpy, imputing nulls with row mean.

    Row-mean imputation: each legislator's missing votes are filled with their
    average Yea rate across non-missing votes. Duplicated from PCA for
    self-containment — changes to EDA filtering won't silently alter UMAP.

    Returns (X_imputed, slugs, vote_ids).
    """
    slug_col = "legislator_slug"
    slugs = matrix[slug_col].to_list()
    vote_ids = [c for c in matrix.columns if c != slug_col]
    X = matrix.select(vote_ids).to_numpy().astype(np.float64)

    # Impute row-by-row: fill NaN with that legislator's mean Yea rate
    for i in range(X.shape[0]):
        row = X[i]
        valid_mask = ~np.isnan(row)
        if valid_mask.any():
            row_mean = row[valid_mask].mean()
            X[i, ~valid_mask] = row_mean
        else:
            # Legislator has no votes at all — fill with 0.5 (uninformative)
            X[i] = 0.5

    return X, slugs, vote_ids


def compute_umap(
    X: np.ndarray,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    metric: str = DEFAULT_METRIC,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Fit UMAP and return 2D embedding array.

    Returns array of shape (n_samples, 2).
    """
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)
    return embedding


def orient_umap1(
    embedding: np.ndarray,
    slugs: list[str],
    legislators: pl.DataFrame,
) -> np.ndarray:
    """Center embedding at the origin and flip UMAP1 so Republicans are positive.

    UMAP axes have arbitrary translation and orientation. This function:
    1. Centers both axes at zero (subtracts column means)
    2. Flips UMAP1 sign if needed so Republicans have positive mean scores

    The centering is essential — without it, UMAP can place the entire embedding
    in one quadrant, making all scores negative (or all positive) even when the
    relative party ordering is correct.
    """
    # Center at origin — UMAP translation is arbitrary
    embedding = embedding - embedding.mean(axis=0)

    slug_to_party = dict(legislators.select("legislator_slug", "party").iter_rows())
    parties = [slug_to_party.get(s, "Unknown") for s in slugs]

    rep_scores = [embedding[i, 0] for i, p in enumerate(parties) if p == "Republican"]
    dem_scores = [embedding[i, 0] for i, p in enumerate(parties) if p == "Democrat"]

    rep_mean = np.mean(rep_scores) if rep_scores else 0.0
    dem_mean = np.mean(dem_scores) if dem_scores else 0.0

    if rep_mean < dem_mean:
        embedding[:, 0] *= -1
        print("  UMAP1 sign flipped (Republicans -> positive)")
    else:
        print("  UMAP1 orientation OK (Republicans already positive)")

    return embedding


def build_embedding_df(
    embedding: np.ndarray,
    slugs: list[str],
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Build a polars DataFrame of UMAP coordinates with legislator metadata."""
    df = pl.DataFrame(
        {
            "legislator_slug": slugs,
            "UMAP1": embedding[:, 0].tolist(),
            "UMAP2": embedding[:, 1].tolist(),
        }
    )

    # Join legislator metadata
    meta = legislators.select("legislator_slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, on="legislator_slug", how="left")
    return df


def compute_procrustes_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Procrustes similarity between two 2D embeddings.

    Returns 1 - disparity (higher = more similar, 1.0 = identical after
    rotation/reflection/scaling).

    Procrustes is the correct comparison for UMAP embeddings because UMAP axes
    are rotation-invariant — Pearson between UMAP1 across settings is meaningless.
    """
    # Center both embeddings
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)

    # scipy.spatial.procrustes returns (mtx1, mtx2, disparity)
    _, _, disparity = procrustes(a_centered, b_centered)
    return 1.0 - disparity


def compute_trustworthiness(
    X: np.ndarray,
    embedding: np.ndarray,
    n_neighbors: int,
) -> float:
    """Compute trustworthiness: fraction of embedding neighbors that were true neighbors.

    Uses sklearn.manifold.trustworthiness. Score > 0.80 is good; > 0.95 is excellent.
    n_neighbors is clamped to n_samples // 2 - 1 (sklearn requirement).
    """
    max_k = X.shape[0] // 2 - 1
    if max_k < 1:
        return float("nan")
    k = min(n_neighbors, max_k)
    return float(sklearn_trustworthiness(X, embedding, n_neighbors=k))


def compute_validation_correlations(
    embedding_df: pl.DataFrame,
    pca_scores: pl.DataFrame | None,
    irt_points: pl.DataFrame | None,
) -> dict:
    """Compute Spearman correlations of UMAP1 vs PCA PC1 and IRT ideal points.

    Spearman (not Pearson) because UMAP preserves rank ordering, not distances.
    """
    results: dict = {}

    if pca_scores is not None and "PC1" in pca_scores.columns:
        merged = embedding_df.join(
            pca_scores.select("legislator_slug", "PC1"),
            on="legislator_slug",
            how="inner",
        )
        if merged.height >= 5:
            rho, pval = spearmanr(
                merged["UMAP1"].to_numpy(),
                merged["PC1"].to_numpy(),
            )
            results["pca_pc1_spearman"] = float(rho)
            results["pca_pc1_pvalue"] = float(pval)
            results["pca_n_shared"] = merged.height
            print(f"    UMAP1 vs PCA PC1: rho={rho:.4f} (n={merged.height})")

    if irt_points is not None:
        ip_col = find_irt_column(irt_points)
        if ip_col is not None:
            merged = embedding_df.join(
                irt_points.select("legislator_slug", ip_col),
                on="legislator_slug",
                how="inner",
            )
            if merged.height >= 5:
                rho, pval = spearmanr(
                    merged["UMAP1"].to_numpy(),
                    merged[ip_col].to_numpy(),
                )
                results["irt_spearman"] = float(rho)
                results["irt_pvalue"] = float(pval)
                results["irt_n_shared"] = merged.height
                print(f"    UMAP1 vs IRT ideal point: rho={rho:.4f} (n={merged.height})")

    return results


def compute_imputation_pct(matrix: pl.DataFrame) -> pl.DataFrame:
    """Compute per-legislator imputation percentage from the raw vote matrix.

    Returns a DataFrame with columns: legislator_slug, imputation_pct.
    """
    vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
    n_votes = len(vote_cols)
    return matrix.select(
        "legislator_slug",
        (pl.sum_horizontal(pl.col(c).is_null() for c in vote_cols) / n_votes * 100)
        .round(1)
        .alias("imputation_pct"),
    )


def run_umap_for_chamber(
    matrix: pl.DataFrame,
    chamber: str,
    n_neighbors: int,
    min_dist: float,
    legislators: pl.DataFrame,
    pca_scores: pl.DataFrame | None,
    irt_points: pl.DataFrame | None,
) -> dict:
    """Run the full UMAP pipeline for one chamber.

    Returns dict with keys: embedding_df, embedding, slugs, vote_ids,
    validation.
    """
    print_header(f"UMAP -- {chamber}")
    n_members = matrix.height
    max_neighbors = max(2, n_members // 3)
    if n_neighbors > max_neighbors:
        print(
            f"  WARNING: n_neighbors={n_neighbors} exceeds {n_members}//3="
            f"{max_neighbors} — capping to {max_neighbors}"
        )
        n_neighbors = max_neighbors
    print(f"  Matrix: {n_members} legislators x {len(matrix.columns) - 1} votes")
    print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={DEFAULT_METRIC}")

    # Compute imputation rates before imputation
    imputation_df = compute_imputation_pct(matrix)

    X, slugs, vote_ids = impute_vote_matrix(matrix)
    embedding = compute_umap(X, n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = orient_umap1(embedding, slugs, legislators)
    embedding_df = build_embedding_df(embedding, slugs, legislators)

    # Add imputation percentage to embedding DataFrame
    embedding_df = embedding_df.join(imputation_df, on="legislator_slug", how="left")

    # Trustworthiness score
    trust = compute_trustworthiness(X, embedding, n_neighbors)
    print(f"\n  Trustworthiness (k={n_neighbors}): {trust:.4f}")

    # Print top/bottom UMAP1 legislators
    sorted_df = embedding_df.sort("UMAP1", descending=True)
    print("\n  Top 5 UMAP1 (most conservative):")
    for row in sorted_df.head(5).iter_rows(named=True):
        print(f"    {row['full_name']:30s}  {row['party']:12s}  UMAP1={row['UMAP1']:+.3f}")
    print("  Bottom 5 UMAP1 (most liberal):")
    for row in sorted_df.tail(5).iter_rows(named=True):
        print(f"    {row['full_name']:30s}  {row['party']:12s}  UMAP1={row['UMAP1']:+.3f}")

    # Validation correlations
    print("\n  Validation correlations:")
    validation = compute_validation_correlations(embedding_df, pca_scores, irt_points)
    validation["trustworthiness"] = trust

    return {
        "embedding_df": embedding_df,
        "embedding": embedding,
        "X_imputed": X,
        "slugs": slugs,
        "vote_ids": vote_ids,
        "validation": validation,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }


# ── Phase 3: Plots ──────────────────────────────────────────────────────────


def plot_umap_landscape(
    embedding_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """UMAP1 vs UMAP2 scatter plot, party-colored, with outlier labels."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for party, color in PARTY_COLORS.items():
        subset = embedding_df.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["UMAP1"].to_numpy(),
            subset["UMAP2"].to_numpy(),
            c=color,
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label=party,
        )

    # Label outliers: top 5 by |UMAP1| and top 5 by |UMAP2|
    labeled: set[str] = set()
    for col in ["UMAP1", "UMAP2"]:
        abs_vals = embedding_df[col].abs()
        top_idx = abs_vals.arg_sort(descending=True).head(5).to_list()
        for idx in top_idx:
            row = embedding_df.row(idx, named=True)
            slug = row["legislator_slug"]
            if slug in labeled:
                continue
            labeled.add(slug)
            name = row.get("full_name", slug)
            last_name = name.split()[-1] if name else slug
            ax.annotate(
                last_name,
                (row["UMAP1"], row["UMAP2"]),
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
                xytext=(6, 6),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.2", "fc": "wheat", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
            )

    # Label cross-party outliers: legislators in the opposite party's territory
    if "party" in embedding_df.columns:
        rep_umap1 = embedding_df.filter(pl.col("party") == "Republican")["UMAP1"]
        dem_umap1 = embedding_df.filter(pl.col("party") == "Democrat")["UMAP1"]
        if rep_umap1.len() > 0 and dem_umap1.len() > 0:
            rep_mean = float(rep_umap1.mean())
            dem_mean = float(dem_umap1.mean())
            for row in embedding_df.iter_rows(named=True):
                party = row.get("party", "")
                umap1 = row["UMAP1"]
                slug = row["legislator_slug"]
                # Democrat on the Republican side, or vice versa
                is_cross_party = (party == "Democrat" and umap1 > rep_mean * 0.5) or (
                    party == "Republican" and umap1 < dem_mean * 0.5
                )
                if is_cross_party and slug not in labeled:
                    labeled.add(slug)
                    name = row.get("full_name", slug)
                    imp_pct = row.get("imputation_pct", 0.0) or 0.0
                    if imp_pct >= HIGH_IMPUTATION_PCT:
                        note = f"(imputation artifact — {party},\n{imp_pct:.0f}% votes imputed)"
                    else:
                        note = f"(cross-party voter — {party})"
                    ax.annotate(
                        f"{name}\n{note}",
                        (umap1, row["UMAP2"]),
                        fontsize=8,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox={
                            "boxstyle": "round,pad=0.4",
                            "fc": "lightyellow",
                            "alpha": 0.9,
                            "ec": "#cc0000",
                        },
                        arrowprops={"arrowstyle": "->", "color": "#cc0000", "lw": 1.5},
                    )

    ax.set_xlabel("UMAP1 (oriented: positive = conservative)")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"{chamber} -- UMAP Ideological Landscape\n"
        "Nearby legislators vote alike; distance = voting dissimilarity"
    )
    # Build legend from parties present in the data
    parties_present = (
        embedding_df["party"].unique().sort().to_list() if "party" in embedding_df.columns else []
    )
    legend_handles = [
        Patch(facecolor=PARTY_COLORS.get(p, "#999999"), label=p)
        for p in parties_present
        if p in PARTY_COLORS
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"umap_landscape_{chamber.lower()}.png")


def plot_umap_gradient(
    embedding_df: pl.DataFrame,
    color_source: pl.DataFrame,
    color_col: str,
    chamber: str,
    out_dir: Path,
    *,
    colorbar_label: str,
    filename_suffix: str,
    title_method: str,
    title_detail: str,
) -> None:
    """UMAP scatter colored by a continuous score (RdBu_r gradient).

    Unified function for PCA PC1 and IRT ideal point gradient plots.
    """
    merged = embedding_df.join(
        color_source.select("legislator_slug", color_col),
        on="legislator_slug",
        how="inner",
    )
    if merged.height < 3:
        print(f"  Skipping {title_method} gradient for {chamber}: too few matched legislators")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        merged["UMAP1"].to_numpy(),
        merged["UMAP2"].to_numpy(),
        c=merged[color_col].to_numpy(),
        cmap="RdBu_r",
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label=colorbar_label)

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"{chamber} -- UMAP Colored by {title_method}\n{title_detail}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"umap_{filename_suffix}_{chamber.lower()}.png")


def plot_umap_colored_by_pc1(
    embedding_df: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """UMAP scatter colored by PCA PC1 scores (RdBu_r gradient)."""
    plot_umap_gradient(
        embedding_df,
        pca_scores,
        "PC1",
        chamber,
        out_dir,
        colorbar_label="PCA PC1 Score (red = conservative, blue = liberal)",
        filename_suffix="pc1_gradient",
        title_method="PCA PC1",
        title_detail="Smooth gradient validates that UMAP captures the same ideological dimension",
    )


def plot_umap_colored_by_irt(
    embedding_df: pl.DataFrame,
    irt_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """UMAP scatter colored by IRT ideal points (RdBu_r gradient)."""
    ip_col = find_irt_column(irt_points)
    if ip_col is None:
        print(f"  Skipping IRT gradient for {chamber}: no ideal_point column found")
        return
    plot_umap_gradient(
        embedding_df,
        irt_points,
        ip_col,
        chamber,
        out_dir,
        colorbar_label="IRT Ideal Point (red = conservative, blue = liberal)",
        filename_suffix="irt_gradient",
        title_method="IRT Ideal Point",
        title_detail="Smooth gradient validates that UMAP aligns with Bayesian ideal points",
    )


# ── Phase 4: Sensitivity Analysis ───────────────────────────────────────────


def run_stability_sweep(
    X: np.ndarray,
    slugs: list[str],
    legislators: pl.DataFrame,
    chamber: str,
    n_neighbors: int,
    min_dist: float,
) -> dict:
    """Run UMAP with multiple random seeds, compare via Procrustes.

    Tests whether embedding structure is robust to stochastic variation.
    Returns dict with per-pair Procrustes similarities.
    """
    print(f"\n  {chamber} multi-seed stability: seeds = {STABILITY_SEEDS}")

    embeddings: dict[int, np.ndarray] = {}
    for seed in STABILITY_SEEDS:
        emb = compute_umap(X, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        emb = orient_umap1(emb, slugs, legislators)
        embeddings[seed] = emb

    pairs: list[dict] = []
    seeds = STABILITY_SEEDS
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            sim = compute_procrustes_similarity(embeddings[seeds[i]], embeddings[seeds[j]])
            pairs.append({"seed_a": seeds[i], "seed_b": seeds[j], "procrustes_similarity": sim})

    mean_sim = np.mean([p["procrustes_similarity"] for p in pairs])
    min_sim = min(p["procrustes_similarity"] for p in pairs)
    print(f"    Mean Procrustes similarity: {mean_sim:.4f} (min: {min_sim:.4f})")

    return {"pairs": pairs, "mean_similarity": float(mean_sim), "min_similarity": float(min_sim)}


def run_sensitivity_sweep(
    X: np.ndarray,
    slugs: list[str],
    legislators: pl.DataFrame,
    chamber: str,
    min_dist: float,
    plots_dir: Path,
) -> dict:
    """Run UMAP with multiple n_neighbors values, compare via Procrustes.

    Returns dict with per-pair Procrustes similarities and per-setting embeddings.
    n_neighbors values exceeding n_samples are skipped.
    """
    n_samples = X.shape[0]
    nn_values = [nn for nn in SENSITIVITY_N_NEIGHBORS if nn < n_samples]
    skipped = [nn for nn in SENSITIVITY_N_NEIGHBORS if nn >= n_samples]
    if skipped:
        print(f"\n  {chamber} sensitivity sweep: n_neighbors = {nn_values}")
        print(f"    Skipped n_neighbors {skipped} (>= {n_samples} legislators)")
    else:
        print(f"\n  {chamber} sensitivity sweep: n_neighbors = {nn_values}")

    embeddings: dict[int, np.ndarray] = {}
    for nn in nn_values:
        emb = compute_umap(X, n_neighbors=nn, min_dist=min_dist)
        emb = orient_umap1(emb, slugs, legislators)
        embeddings[nn] = emb

    # Compute pairwise Procrustes similarities
    pairs: list[dict] = []
    nn_list = nn_values
    for i in range(len(nn_list)):
        for j in range(i + 1, len(nn_list)):
            sim = compute_procrustes_similarity(embeddings[nn_list[i]], embeddings[nn_list[j]])
            pairs.append(
                {
                    "nn_a": nn_list[i],
                    "nn_b": nn_list[j],
                    "procrustes_similarity": sim,
                }
            )
            print(
                f"    n_neighbors {nn_list[i]:2d} vs {nn_list[j]:2d}: "
                f"Procrustes similarity = {sim:.4f}"
            )

    # Plot sensitivity grid
    _plot_sensitivity_grid(embeddings, slugs, legislators, chamber, plots_dir)

    return {
        "embeddings": embeddings,
        "pairs": pairs,
    }


def _plot_sensitivity_grid(
    embeddings: dict[int, np.ndarray],
    slugs: list[str],
    legislators: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """4-panel comparison of UMAP embeddings across n_neighbors values."""
    nn_list = sorted(embeddings.keys())
    n_panels = len(nn_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    slug_to_party = dict(legislators.select("legislator_slug", "party").iter_rows())
    parties = [slug_to_party.get(s, "Unknown") for s in slugs]
    colors = [PARTY_COLORS.get(p, "#999999") for p in parties]

    for ax, nn in zip(axes, nn_list):
        emb = embeddings[nn]
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=colors,
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.3,
        )
        ax.set_title(f"n_neighbors = {nn}", fontsize=11)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{chamber} -- UMAP Sensitivity to n_neighbors\n"
        "Consistent structure across settings indicates robust results",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"umap_sensitivity_{chamber.lower()}.png")


# ── Phase 5: Filtering Manifest ─────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


# ── Phase 6: Main ───────────────────────────────────────────────────────────


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
    pca_dir = resolve_upstream_dir(
        "02_pca",
        results_root,
        args.run_id,
        Path(args.pca_dir) if args.pca_dir else None,
    )
    irt_dir = resolve_upstream_dir(
        "05_irt",
        results_root,
        args.run_id,
        Path(args.irt_dir) if args.irt_dir else None,
    )

    with RunContext(
        session=args.session,
        analysis_name="04_umap",
        params=vars(args),
        primer=UMAP_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature UMAP -- Session {args.session}")
        print(f"Data:     {data_dir}")
        print(f"EDA:      {eda_dir}")
        print(f"PCA:      {pca_dir}")
        print(f"IRT:      {irt_dir}")
        print(f"Output:   {ctx.run_dir}")
        print(f"Parameters: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")

        # ── Phase 1: Load data ──
        print_header("LOADING DATA")
        eda_result = load_eda_matrices(eda_dir)
        if eda_result is None:
            print("Phase 03 (UMAP): skipping — no EDA vote matrices available")
            return
        house_matrix, senate_matrix = eda_result
        if house_matrix.height == 0 and senate_matrix.height == 0:
            print("Phase 03 (UMAP): skipping — 0 legislators after EDA filtering")
            return
        legislators = load_legislators(data_dir)

        print(f"  House filtered: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate filtered: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")
        print(f"  Legislators: {legislators.height}")

        # Load upstream for validation (optional)
        pca_house, pca_senate = load_pca_scores(pca_dir)
        irt_house, irt_senate = load_irt_ideal_points(irt_dir)

        if pca_house is not None:
            sen_n = pca_senate.height if pca_senate is not None else 0
            print(f"  PCA scores loaded: House={pca_house.height}, Senate={sen_n}")
        else:
            print("  PCA scores: not available (validation will be skipped)")
        if irt_house is not None:
            sen_n = irt_senate.height if irt_senate is not None else 0
            print(f"  IRT ideal points loaded: House={irt_house.height}, Senate={sen_n}")
        else:
            print("  IRT ideal points: not available (validation will be skipped)")

        # ── Phase 2: UMAP per chamber ──
        results: dict[str, dict] = {}
        pca_map = {"House": pca_house, "Senate": pca_senate}
        irt_map = {"House": irt_house, "Senate": irt_senate}

        for label, matrix in [("House", house_matrix), ("Senate", senate_matrix)]:
            if matrix.height < 3:
                print(f"\n  Skipping {label}: too few legislators ({matrix.height})")
                continue

            result = run_umap_for_chamber(
                matrix,
                label,
                args.n_neighbors,
                args.min_dist,
                legislators,
                pca_map[label],
                irt_map[label],
            )
            results[label] = result

            # Save parquet
            result["embedding_df"].write_parquet(
                ctx.data_dir / f"umap_embedding_{label.lower()}.parquet"
            )
            print(f"  Saved: umap_embedding_{label.lower()}.parquet")

        # ── Phase 3: Plots ──
        print_header("GENERATING PLOTS")
        for label, result in results.items():
            plot_umap_landscape(result["embedding_df"], label, ctx.plots_dir)

            # PC1 gradient plot
            pca_scores = pca_map[label]
            if pca_scores is not None:
                plot_umap_colored_by_pc1(result["embedding_df"], pca_scores, label, ctx.plots_dir)

            # IRT gradient plot
            irt_points = irt_map[label]
            if irt_points is not None:
                plot_umap_colored_by_irt(result["embedding_df"], irt_points, label, ctx.plots_dir)

        # ── Phase 4: Sensitivity analysis ──
        sensitivity_findings: dict[str, dict] = {}
        stability_findings: dict[str, dict] = {}
        if not args.skip_sensitivity:
            print_header("SENSITIVITY ANALYSIS")
            for label, result in results.items():
                sweep = run_sensitivity_sweep(
                    result["X_imputed"],
                    result["slugs"],
                    legislators,
                    label,
                    args.min_dist,
                    ctx.plots_dir,
                )
                sensitivity_findings[label] = sweep

            # Save sensitivity sweep parquet
            all_pairs: list[dict] = []
            for label, sweep in sensitivity_findings.items():
                for pair in sweep["pairs"]:
                    all_pairs.append({"chamber": label, **pair})
            if all_pairs:
                sweep_df = pl.DataFrame(all_pairs)
                sweep_df.write_parquet(ctx.data_dir / "sensitivity_sweep.parquet")
                print("  Saved: sensitivity_sweep.parquet")

            # Multi-seed stability
            print_header("MULTI-SEED STABILITY")
            for label, result in results.items():
                stability = run_stability_sweep(
                    result["X_imputed"],
                    result["slugs"],
                    legislators,
                    label,
                    args.n_neighbors,
                    args.min_dist,
                )
                stability_findings[label] = stability

            # Save stability parquet
            stab_rows: list[dict] = []
            for label, stab in stability_findings.items():
                for pair in stab["pairs"]:
                    stab_rows.append({"chamber": label, **pair})
            if stab_rows:
                stab_df = pl.DataFrame(stab_rows)
                stab_df.write_parquet(ctx.data_dir / "stability_sweep.parquet")
                print("  Saved: stability_sweep.parquet")
        else:
            print_header("SENSITIVITY ANALYSIS (SKIPPED)")

        # ── Phase 5: Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "eda_source": str(eda_dir),
            "pca_source": str(pca_dir),
            "irt_source": str(irt_dir),
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": DEFAULT_METRIC,
            "random_state": RANDOM_STATE,
            "stability_seeds": STABILITY_SEEDS,
            "impute_method": "row_mean",
        }
        for label, result in results.items():
            manifest[f"{label.lower()}_n_legislators"] = result["embedding_df"].height
            manifest[f"{label.lower()}_n_votes"] = len(result["vote_ids"])
            manifest[f"{label.lower()}_validation"] = result["validation"]
        if sensitivity_findings:
            for label, sweep in sensitivity_findings.items():
                manifest[f"{label.lower()}_sensitivity_pairs"] = sweep["pairs"]
        if stability_findings:
            for label, stab in stability_findings.items():
                manifest[f"{label.lower()}_stability"] = {
                    "mean_similarity": stab["mean_similarity"],
                    "min_similarity": stab["min_similarity"],
                    "n_seeds": len(STABILITY_SEEDS),
                }
        save_filtering_manifest(manifest, ctx.run_dir)

        # ── Phase 6: HTML report ──
        print_header("HTML REPORT")
        build_umap_report(
            ctx.report,
            results=results,
            sensitivity_findings=sensitivity_findings,
            stability_findings=stability_findings,
            plots_dir=ctx.plots_dir,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
