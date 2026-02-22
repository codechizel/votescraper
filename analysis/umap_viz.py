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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from scipy.spatial import procrustes
from scipy.stats import spearmanr

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

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

- **Nearby legislators** vote alike. Distance = voting dissimilarity.
- **Party clusters**: Expect two main groups (Republican, Democrat). UMAP
  preserves global structure, so the gap between clusters reflects how
  different the parties' voting records are.
- **Sub-clusters within a party**: Possible factions (moderates vs. hardliners).
- **Bridge legislators** (between clusters): Moderates or cross-party voters.
- **Isolated points**: Unique voting patterns (extreme ideologues or chronic
  absentees with heavily imputed data).
- **UMAP1** is oriented so Republicans are positive (same as PCA PC1).
  But UMAP axes are arbitrary — only relative positions matter.

## Caveats

- UMAP is stochastic — results vary slightly across runs (mitigated by
  fixed random_state=42).
- The axes (UMAP1, UMAP2) have no inherent meaning. Only distances and
  relative positions are interpretable.
- Small datasets (~40 senators) can produce unstable embeddings. Check
  the sensitivity sweep for robustness.
- Row-mean imputation treats absences as uninformative. Legislators with
  many absences may appear more moderate than they are.
"""

# ── Constants ────────────────────────────────────────────────────────────────
# Explicit, named constants per the analytic-workflow rules.

DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_METRIC = "cosine"
RANDOM_STATE = 42
SENSITIVITY_N_NEIGHBORS = [5, 15, 30, 50]
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature UMAP Visualization")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
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


def print_header(title: str) -> None:
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_eda_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load filtered vote matrices from the EDA phase output.

    Returns (house_filtered, senate_filtered).
    """
    house = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
    return house, senate


def load_metadata(data_dir: Path) -> pl.DataFrame:
    """Load legislator CSV for metadata enrichment."""
    prefix = data_dir.name
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    return legislators


def load_pca_scores(pca_dir: Path) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load PCA scores for validation. Returns (house, senate) or None if unavailable."""
    house, senate = None, None
    try:
        house = pl.read_parquet(pca_dir / "data" / "pc_scores_house.parquet")
    except FileNotFoundError, OSError:
        pass
    try:
        senate = pl.read_parquet(pca_dir / "data" / "pc_scores_senate.parquet")
    except FileNotFoundError, OSError:
        pass
    return house, senate


def load_irt_ideal_points(
    irt_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """Load IRT ideal points for validation. Returns (house, senate) or None."""
    house, senate = None, None
    try:
        house = pl.read_parquet(irt_dir / "data" / "ideal_points_house.parquet")
    except FileNotFoundError, OSError:
        pass
    try:
        senate = pl.read_parquet(irt_dir / "data" / "ideal_points_senate.parquet")
    except FileNotFoundError, OSError:
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
    """Flip UMAP1 sign so Republicans have positive mean scores.

    UMAP axes have arbitrary orientation — this convention makes interpretation
    consistent: positive UMAP1 = conservative, negative UMAP1 = liberal.
    """
    slug_to_party = dict(legislators.select("slug", "party").iter_rows())
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
    meta = legislators.select("slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, left_on="legislator_slug", right_on="slug", how="left")
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
        # Match by legislator_slug
        slug_col = "legislator_slug"
        pca_slug_col = slug_col if slug_col in pca_scores.columns else "legislator_slug"
        merged = embedding_df.join(
            pca_scores.select(pca_slug_col, "PC1"),
            left_on="legislator_slug",
            right_on=pca_slug_col,
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
        # IRT ideal points column name varies
        ip_col = None
        for candidate in ["xi_mean", "ideal_point", "theta", "xi"]:
            if candidate in irt_points.columns:
                ip_col = candidate
                break

        if ip_col is not None:
            slug_col = "legislator_slug"
            irt_slug_col = slug_col if slug_col in irt_points.columns else "legislator_slug"
            merged = embedding_df.join(
                irt_points.select(irt_slug_col, ip_col),
                left_on="legislator_slug",
                right_on=irt_slug_col,
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
    print(f"  Matrix: {matrix.height} legislators x {len(matrix.columns) - 1} votes")
    print(f"  Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={DEFAULT_METRIC}")

    X, slugs, vote_ids = impute_vote_matrix(matrix)
    embedding = compute_umap(X, n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = orient_umap1(embedding, slugs, legislators)
    embedding_df = build_embedding_df(embedding, slugs, legislators)

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
    if not validation:
        print("    No upstream data available for validation")

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

    ax.set_xlabel("UMAP1 (oriented: positive = conservative)")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"{chamber} -- UMAP Ideological Landscape\n"
        "Nearby legislators vote alike; distance = voting dissimilarity"
    )
    ax.legend(
        handles=[
            Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
            Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
        ],
        loc="best",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"umap_landscape_{chamber.lower()}.png")


def plot_umap_colored_by_pc1(
    embedding_df: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """UMAP scatter colored by PCA PC1 scores (RdBu_r gradient)."""
    slug_col = "legislator_slug"
    pca_slug_col = slug_col if slug_col in pca_scores.columns else "legislator_slug"
    merged = embedding_df.join(
        pca_scores.select(pca_slug_col, "PC1"),
        left_on="legislator_slug",
        right_on=pca_slug_col,
        how="inner",
    )
    if merged.height < 3:
        print(f"  Skipping PC1 gradient for {chamber}: too few matched legislators")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        merged["UMAP1"].to_numpy(),
        merged["UMAP2"].to_numpy(),
        c=merged["PC1"].to_numpy(),
        cmap="RdBu_r",
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="PCA PC1 Score (red = conservative, blue = liberal)")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"{chamber} -- UMAP Colored by PCA PC1\n"
        "Smooth gradient validates that UMAP captures the same ideological dimension"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"umap_pc1_gradient_{chamber.lower()}.png")


def plot_umap_colored_by_irt(
    embedding_df: pl.DataFrame,
    irt_points: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """UMAP scatter colored by IRT ideal points (RdBu_r gradient)."""
    # Find the ideal point column
    ip_col = None
    for candidate in ["xi_mean", "ideal_point", "theta", "xi"]:
        if candidate in irt_points.columns:
            ip_col = candidate
            break
    if ip_col is None:
        print(f"  Skipping IRT gradient for {chamber}: no ideal_point column found")
        return

    slug_col = "legislator_slug"
    irt_slug_col = slug_col if slug_col in irt_points.columns else "legislator_slug"
    merged = embedding_df.join(
        irt_points.select(irt_slug_col, ip_col),
        left_on="legislator_slug",
        right_on=irt_slug_col,
        how="inner",
    )
    if merged.height < 3:
        print(f"  Skipping IRT gradient for {chamber}: too few matched legislators")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        merged["UMAP1"].to_numpy(),
        merged["UMAP2"].to_numpy(),
        c=merged[ip_col].to_numpy(),
        cmap="RdBu_r",
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="IRT Ideal Point (red = conservative, blue = liberal)")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(
        f"{chamber} -- UMAP Colored by IRT Ideal Point\n"
        "Smooth gradient validates that UMAP aligns with Bayesian ideal points"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"umap_irt_gradient_{chamber.lower()}.png")


# ── Phase 4: Sensitivity Analysis ───────────────────────────────────────────


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
    """
    print(f"\n  {chamber} sensitivity sweep: n_neighbors = {SENSITIVITY_N_NEIGHBORS}")

    embeddings: dict[int, np.ndarray] = {}
    for nn in SENSITIVITY_N_NEIGHBORS:
        emb = compute_umap(X, n_neighbors=nn, min_dist=min_dist)
        emb = orient_umap1(emb, slugs, legislators)
        embeddings[nn] = emb

    # Compute pairwise Procrustes similarities
    pairs: list[dict] = []
    nn_list = SENSITIVITY_N_NEIGHBORS
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

    slug_to_party = dict(legislators.select("slug", "party").iter_rows())
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

    pca_dir = Path(args.pca_dir) if args.pca_dir else results_root / "pca" / "latest"
    irt_dir = Path(args.irt_dir) if args.irt_dir else results_root / "irt" / "latest"

    with RunContext(
        session=args.session,
        analysis_name="umap",
        params=vars(args),
        primer=UMAP_PRIMER,
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
        house_matrix, senate_matrix = load_eda_matrices(eda_dir)
        legislators = load_metadata(data_dir)

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
            "impute_method": "row_mean",
        }
        for label, result in results.items():
            manifest[f"{label.lower()}_n_legislators"] = result["embedding_df"].height
            manifest[f"{label.lower()}_n_votes"] = len(result["vote_ids"])
            manifest[f"{label.lower()}_validation"] = result["validation"]
        if sensitivity_findings:
            for label, sweep in sensitivity_findings.items():
                manifest[f"{label.lower()}_sensitivity_pairs"] = sweep["pairs"]
        save_filtering_manifest(manifest, ctx.run_dir)

        # ── Phase 6: HTML report ──
        print_header("HTML REPORT")
        build_umap_report(
            ctx.report,
            results=results,
            sensitivity_findings=sensitivity_findings,
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
