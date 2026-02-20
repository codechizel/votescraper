"""
Kansas Legislature — Principal Component Analysis (Phase 2)

Covers analytic method 09: PCA on the binary vote matrix.
PCA is the cheapest dimensionality reduction — reveals the ideological landscape
and validates data before investing in Bayesian IRT (MCMC).

Usage:
  uv run python analysis/pca.py [--session 2025-26] [--eda-dir ...] \
      [--n-components 5] [--skip-sensitivity]

Outputs (in results/<session>/pca/<date>/):
  - data/:   Parquet files (PC scores, loadings, explained variance)
  - plots/:  PNG visualizations (scree, ideological map, PC1 distribution, sensitivity)
  - filtering_manifest.json, run_info.json, run_log.txt
  - pca_report.html
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.pca_report import build_pca_report
except ModuleNotFoundError:
    from pca_report import build_pca_report  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────
# Written to results/<session>/pca/README.md by RunContext on each run.

PCA_PRIMER = """\
# Principal Component Analysis (PCA)

## Purpose

PCA extracts the principal axes of variation in legislators' voting behavior.
The first principal component (PC1) almost always corresponds to the left-right
ideological spectrum. PC2 typically captures a second, cross-cutting dimension
(e.g., urban vs. rural, establishment vs. insurgent).

PCA is the mandatory step before Bayesian IRT (per the analytic workflow rules):
cheap, fast, and produces ideal-point estimates highly correlated (r > 0.95)
with NOMINATE scores.

Covers analytic method 09 from `Analytic_Methods/`.

## Method

1. **Load filtered vote matrices** from the EDA phase (parquet files).
2. **Impute missing values** with row-mean (each legislator's Yea base rate).
3. **Standardize** (center and scale) the matrix.
4. **Fit PCA** with 5 components per chamber.
5. **Orient PC1** so Republicans are positive (convention).
6. **Sensitivity analysis** — re-run at 10% minority threshold and compare.
7. **Holdout validation** — mask 20% of votes, reconstruct, measure accuracy.

## Inputs

Reads from `results/<session>/eda/latest/data/`:
- `vote_matrix_house_filtered.parquet` — House binary vote matrix (EDA-filtered)
- `vote_matrix_senate_filtered.parquet` — Senate binary vote matrix (EDA-filtered)
- `vote_matrix_full.parquet` — Full unfiltered vote matrix (for sensitivity)

Reads from `data/ks_{session}/`:
- `ks_{slug}_rollcalls.csv` — Roll call metadata (bill numbers, titles, motions)
- `ks_{slug}_legislators.csv` — Legislator metadata (names, parties, districts)

## Outputs

All outputs land in `results/<session>/pca/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `pc_scores_house.parquet` | House legislator PC1-5 scores + metadata |
| `pc_scores_senate.parquet` | Senate legislator PC1-5 scores + metadata |
| `pc_loadings_house.parquet` | House roll call PC1-5 loadings + bill metadata |
| `pc_loadings_senate.parquet` | Senate roll call PC1-5 loadings + bill metadata |
| `explained_variance.parquet` | Per-component explained variance (both chambers) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `scree_house.png` | Scree plot: individual + cumulative explained variance |
| `scree_senate.png` | Scree plot for Senate |
| `ideological_map_house.png` | PC1 vs PC2 scatter, party-colored, outlier labels |
| `ideological_map_senate.png` | Ideological map for Senate |
| `pc1_distribution_house.png` | Overlapping KDE of PC1 scores by party |
| `pc1_distribution_senate.png` | PC1 distribution for Senate |
| `sensitivity_pc1_house.png` | Default vs sensitivity PC1 scores scatter |
| `sensitivity_pc1_senate.png` | Sensitivity scatter for Senate |

### Root files

| File | Description |
|------|-------------|
| `filtering_manifest.json` | EDA source, n_components, impute method, metrics |
| `run_info.json` | Git commit, timestamp, Python version, parameters |
| `run_log.txt` | Full console output from the run |
| `pca_report.html` | Self-contained HTML report with all tables and figures |

## Interpretation Guide

- **PC1 explained variance of 30-50%** is typical for a partisan legislature.
  Higher values (>50%) indicate extreme partisanship.
- **PC1 scores**: Positive = Republican direction, negative = Democrat. Scores
  near zero are centrists / moderates.
- **PC2**: Examine top loadings to interpret. Common second dimensions: urban/rural,
  establishment/insurgent, fiscal/social splits.
- **Scree plot**: Sharp elbow after PC1 means legislature is one-dimensional.
  Gradual decline means multiple conflict dimensions coexist.
- **Sensitivity**: Pearson r > 0.95 between default (2.5%) and aggressive (10%)
  filtering means results are robust to the threshold choice.
- **Holdout validation**: Accuracy must exceed the ~82% Yea base rate to show
  that PCA captures real structure, not just the base rate.

## Caveats

- PCA is a linear model — assumes voting is a linear function of ideology.
  Bayesian IRT (Phase 4) uses a nonlinear link function that may fit better.
- Row-mean imputation treats each absence as "this legislator would have voted
  at their average rate." This is reasonable but not the only option.
- PCA gives point estimates, not uncertainty intervals. Use Bayesian IRT for
  posterior distributions on ideal points.
"""

# ── Constants ────────────────────────────────────────────────────────────────
# Explicit, named constants per the analytic-workflow rules.

DEFAULT_N_COMPONENTS = 5
MINORITY_THRESHOLD = 0.025  # Default: 2.5% (matches EDA)
SENSITIVITY_THRESHOLD = 0.10  # Sensitivity: 10% (per workflow rules)
MIN_VOTES = 20  # Minimum substantive votes per legislator
HOLDOUT_FRACTION = 0.20  # Random 20% of non-null cells
HOLDOUT_SEED = 42
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature PCA")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument(
        "--n-components", type=int, default=DEFAULT_N_COMPONENTS,
        help="Number of PCA components to extract",
    )
    parser.add_argument(
        "--skip-sensitivity", action="store_true",
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
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load filtered vote matrices from the EDA phase output.

    Returns (house_filtered, senate_filtered, full_matrix).
    """
    house = pl.read_parquet(eda_dir / "data" / "vote_matrix_house_filtered.parquet")
    senate = pl.read_parquet(eda_dir / "data" / "vote_matrix_senate_filtered.parquet")
    full = pl.read_parquet(eda_dir / "data" / "vote_matrix_full.parquet")
    return house, senate, full


def load_metadata(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load rollcall and legislator CSVs for metadata enrichment."""
    session_slug = data_dir.name.removeprefix("ks_")
    rollcalls = pl.read_csv(data_dir / f"ks_{session_slug}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"ks_{session_slug}_legislators.csv")
    return rollcalls, legislators


# ── Phase 2: PCA per Chamber ────────────────────────────────────────────────


def impute_vote_matrix(matrix: pl.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert polars vote matrix to numpy, imputing nulls with row mean.

    Row-mean imputation: each legislator's missing votes are filled with their
    average Yea rate across non-missing votes. This is the most principled
    default — it assumes absences are uninformative about ideology.

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


def fit_pca(
    X: np.ndarray, n_components: int,
) -> tuple[np.ndarray, np.ndarray, PCA, StandardScaler]:
    """Standardize and fit PCA. Returns (scores, loadings, pca, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_  # shape: (n_components, n_votes)
    return scores, loadings, pca, scaler


def orient_pc1(
    scores: np.ndarray,
    loadings: np.ndarray,
    slugs: list[str],
    legislators: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Flip PC1 sign so Republicans have positive mean scores.

    PCA components have arbitrary sign — this convention makes interpretation
    consistent: positive PC1 = conservative, negative PC1 = liberal.
    """
    slug_to_party = dict(legislators.select("slug", "party").iter_rows())
    parties = [slug_to_party.get(s, "Unknown") for s in slugs]

    rep_scores = [scores[i, 0] for i, p in enumerate(parties) if p == "Republican"]
    dem_scores = [scores[i, 0] for i, p in enumerate(parties) if p == "Democrat"]

    rep_mean = np.mean(rep_scores) if rep_scores else 0.0
    dem_mean = np.mean(dem_scores) if dem_scores else 0.0

    if rep_mean < dem_mean:
        scores[:, 0] *= -1
        loadings[0, :] *= -1
        print("  PC1 sign flipped (Republicans → positive)")
    else:
        print("  PC1 orientation OK (Republicans already positive)")

    return scores, loadings


def build_scores_df(
    scores: np.ndarray,
    slugs: list[str],
    n_components: int,
    legislators: pl.DataFrame,
) -> pl.DataFrame:
    """Build a polars DataFrame of PC scores with legislator metadata."""
    pc_cols = {f"PC{i+1}": scores[:, i].tolist() for i in range(n_components)}
    df = pl.DataFrame({"legislator_slug": slugs, **pc_cols})

    # Join legislator metadata
    meta = legislators.select("slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, left_on="legislator_slug", right_on="slug", how="left")
    return df


def build_loadings_df(
    loadings: np.ndarray,
    vote_ids: list[str],
    n_components: int,
    rollcalls: pl.DataFrame,
) -> pl.DataFrame:
    """Build a polars DataFrame of PC loadings with rollcall metadata."""
    pc_cols = {f"PC{i+1}": loadings[i, :].tolist() for i in range(n_components)}
    df = pl.DataFrame({"vote_id": vote_ids, **pc_cols})

    # Join rollcall metadata
    meta_cols = ["vote_id", "bill_number", "short_title", "motion", "vote_type", "chamber"]
    available = [c for c in meta_cols if c in rollcalls.columns]
    if available:
        meta = rollcalls.select(available)
        df = df.join(meta, on="vote_id", how="left")
    return df


def run_pca_for_chamber(
    matrix: pl.DataFrame,
    chamber: str,
    n_components: int,
    legislators: pl.DataFrame,
    rollcalls: pl.DataFrame,
) -> dict:
    """Run the full PCA pipeline for one chamber.

    Returns dict with keys: scores_df, loadings_df, pca, scaler, X_imputed,
    slugs, vote_ids, explained_variance.
    """
    print_header(f"PCA — {chamber}")
    print(f"  Matrix: {matrix.height} legislators x {len(matrix.columns) - 1} votes")

    X, slugs, vote_ids = impute_vote_matrix(matrix)
    n_comp = min(n_components, X.shape[0], X.shape[1])
    scores, loadings, pca, scaler = fit_pca(X, n_comp)
    scores, loadings = orient_pc1(scores, loadings, slugs, legislators)

    # Print explained variance
    ev = pca.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    print("\n  Explained variance:")
    for i in range(n_comp):
        print(f"    PC{i+1}: {ev[i]:.4f} ({100*ev[i]:.1f}%)  cumulative: {100*cumulative[i]:.1f}%")

    scores_df = build_scores_df(scores, slugs, n_comp, legislators)
    loadings_df = build_loadings_df(loadings, vote_ids, n_comp, rollcalls)

    # Print top/bottom PC1 legislators
    sorted_scores = scores_df.sort("PC1", descending=True)
    print("\n  Top 5 PC1 (most conservative):")
    for row in sorted_scores.head(5).iter_rows(named=True):
        print(f"    {row['full_name']:30s}  {row['party']:12s}  PC1={row['PC1']:+.3f}")
    print("  Bottom 5 PC1 (most liberal):")
    for row in sorted_scores.tail(5).iter_rows(named=True):
        print(f"    {row['full_name']:30s}  {row['party']:12s}  PC1={row['PC1']:+.3f}")

    return {
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "pca": pca,
        "scaler": scaler,
        "X_imputed": X,
        "slugs": slugs,
        "vote_ids": vote_ids,
        "explained_variance": ev.tolist(),
        "n_components": n_comp,
    }


# ── Phase 3: Plots ──────────────────────────────────────────────────────────


def plot_scree(pca_obj: PCA, chamber: str, out_dir: Path) -> None:
    """Scree plot with individual and cumulative explained variance (2 panels)."""
    ev = pca_obj.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    n = len(ev)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Individual explained variance
    axes[0].bar(
        range(1, n + 1), ev, color="#4C72B0", edgecolor="black", alpha=0.9,
    )
    for i, v in enumerate(ev):
        axes[0].text(i + 1, v + 0.005, f"{100*v:.1f}%", ha="center", fontsize=9)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title(f"{chamber} — Individual Explained Variance")
    axes[0].set_xticks(range(1, n + 1))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel 2: Cumulative explained variance
    axes[1].plot(range(1, n + 1), cumulative, "bo-", markersize=8)
    for i, v in enumerate(cumulative):
        axes[1].text(i + 1, v + 0.02, f"{100*v:.1f}%", ha="center", fontsize=9)
    axes[1].axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title(f"{chamber} — Cumulative Variance Explained")
    axes[1].set_xticks(range(1, n + 1))
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"scree_{chamber.lower()}.png")


def plot_ideological_map(
    scores_df: pl.DataFrame, chamber: str, out_dir: Path,
) -> None:
    """PC1 vs PC2 scatter plot, party-colored, with outlier labels."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for party, color in PARTY_COLORS.items():
        subset = scores_df.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["PC1"].to_numpy(),
            subset["PC2"].to_numpy(),
            c=color, s=60, alpha=0.7, edgecolors="black", linewidth=0.5,
            label=party,
        )

    # Label outliers: top 5 by |PC1| and top 5 by |PC2|
    labeled = set()
    for pc_col in ["PC1", "PC2"]:
        abs_vals = scores_df[pc_col].abs()
        top_idx = abs_vals.arg_sort(descending=True).head(5).to_list()
        for idx in top_idx:
            row = scores_df.row(idx, named=True)
            slug = row["legislator_slug"]
            if slug in labeled:
                continue
            labeled.add(slug)
            name = row.get("full_name", slug)
            # Use last name for concise labels
            last_name = name.split()[-1] if name else slug
            ax.annotate(
                last_name,
                (row["PC1"], row["PC2"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points",
            )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.2)
    ax.set_xlabel("PC1 (primary ideological dimension)")
    ax.set_ylabel("PC2 (secondary dimension)")
    ax.set_title(f"{chamber} — Ideological Map (PC1 vs PC2)")
    ax.legend(handles=[
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ], loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"ideological_map_{chamber.lower()}.png")


def plot_pc1_distribution(
    scores_df: pl.DataFrame, chamber: str, out_dir: Path,
) -> None:
    """Overlapping KDE of PC1 scores by party."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 5))

    for party, color in PARTY_COLORS.items():
        subset = scores_df.filter(pl.col("party") == party)
        if subset.height < 2:
            continue
        values = subset["PC1"].to_numpy()
        sns.kdeplot(values, ax=ax, color=color, fill=True, alpha=0.3, label=party)

    ax.set_xlabel("PC1 Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{chamber} — PC1 Distribution by Party")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"pc1_distribution_{chamber.lower()}.png")


# ── Phase 4: Sensitivity Analysis ───────────────────────────────────────────


def filter_vote_matrix_for_sensitivity(
    full_matrix: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
    minority_threshold: float = SENSITIVITY_THRESHOLD,
    min_votes: int = MIN_VOTES,
) -> pl.DataFrame:
    """Re-filter the full vote matrix at an alternative minority threshold.

    Duplicates the ~40-line filter logic from eda.py to avoid coupling.
    This is intentional — keeps the PCA phase self-contained.
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in full_matrix.columns if c != slug_col]

    # Restrict to chamber
    chamber_vote_ids = set(
        rollcalls.filter(pl.col("chamber") == chamber)["vote_id"].to_list()
    )
    prefix = "sen_" if chamber == "Senate" else "rep_"
    vote_cols = [c for c in vote_cols if c in chamber_vote_ids]
    matrix = full_matrix.filter(
        pl.col(slug_col).str.starts_with(prefix)
    ).select([slug_col, *vote_cols])

    # Filter 1: Drop near-unanimous votes
    contested_cols = []
    for col in vote_cols:
        series = matrix[col].drop_nulls()
        if series.len() == 0:
            continue
        yea_frac = series.mean()
        minority_frac = min(yea_frac, 1 - yea_frac)
        if minority_frac >= minority_threshold:
            contested_cols.append(col)

    if not contested_cols:
        return matrix.select([slug_col]).head(0)

    filtered = matrix.select([slug_col, *contested_cols])

    # Filter 2: Drop low-participation legislators
    non_null_counts = filtered.select(
        slug_col,
        pl.sum_horizontal(
            *[pl.col(c).is_not_null().cast(pl.Int32) for c in contested_cols]
        ).alias("n_votes"),
    )
    active_slugs = non_null_counts.filter(pl.col("n_votes") >= min_votes)[slug_col].to_list()
    filtered = filtered.filter(pl.col(slug_col).is_in(active_slugs))

    return filtered


def run_sensitivity(
    full_matrix: pl.DataFrame,
    default_results: dict[str, dict],
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    n_components: int,
    plots_dir: Path,
) -> dict:
    """Run PCA with 10% minority threshold and compare to default (2.5%).

    Returns sensitivity findings dict.
    """
    print_header("SENSITIVITY ANALYSIS (10% threshold)")
    findings: dict = {}

    for chamber, default in default_results.items():
        print(f"\n  {chamber}:")
        sens_matrix = filter_vote_matrix_for_sensitivity(
            full_matrix, rollcalls, chamber,
            minority_threshold=SENSITIVITY_THRESHOLD,
            min_votes=MIN_VOTES,
        )
        n_votes = len(sens_matrix.columns) - 1
        print(f"    Sensitivity matrix: {sens_matrix.height} legislators x {n_votes} votes")

        if sens_matrix.height < 3 or n_votes < 3:
            print("    Skipping: too few data points")
            findings[chamber] = {"skipped": True, "reason": "insufficient data"}
            continue

        X, slugs, vote_ids = impute_vote_matrix(sens_matrix)
        n_comp = min(n_components, X.shape[0], X.shape[1])
        scores, loadings, pca, scaler = fit_pca(X, n_comp)
        scores, loadings = orient_pc1(scores, loadings, slugs, legislators)

        # Match legislators between default and sensitivity by slug
        default_scores = default["scores_df"]
        default_slugs = set(default_scores["legislator_slug"].to_list())
        sens_score_map = dict(zip(slugs, scores[:, 0].tolist()))

        shared_slugs = sorted(default_slugs & set(slugs))
        if len(shared_slugs) < 5:
            print(f"    Skipping correlation: only {len(shared_slugs)} shared legislators")
            findings[chamber] = {"skipped": True, "reason": "too few shared legislators"}
            continue

        default_pc1 = []
        sens_pc1 = []
        for s in shared_slugs:
            row = default_scores.filter(pl.col("legislator_slug") == s)
            default_pc1.append(row["PC1"][0])
            sens_pc1.append(sens_score_map[s])

        default_arr = np.array(default_pc1)
        sens_arr = np.array(sens_pc1)
        correlation = float(np.corrcoef(default_arr, sens_arr)[0, 1])

        print(f"    Shared legislators: {len(shared_slugs)}")
        print(f"    Pearson r: {correlation:.4f}")
        ev_str = ", ".join(
            f"PC{i+1}={100*v:.1f}%"
            for i, v in enumerate(pca.explained_variance_ratio_)
        )
        print(f"    Sensitivity EV: {ev_str}")

        if correlation > 0.95:
            print("    Result: ROBUST (r > 0.95)")
        else:
            print("    Result: SENSITIVE (r <= 0.95) — investigate threshold dependence")

        findings[chamber] = {
            "default_threshold": MINORITY_THRESHOLD,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "default_n_legislators": default_scores.height,
            "sensitivity_n_legislators": sens_matrix.height,
            "default_n_votes": len(default["vote_ids"]),
            "sensitivity_n_votes": n_votes,
            "shared_legislators": len(shared_slugs),
            "pearson_r": correlation,
            "sensitivity_explained_variance": pca.explained_variance_ratio_.tolist(),
        }

        # Plot sensitivity comparison
        _plot_sensitivity_scatter(
            default_arr, sens_arr, correlation, chamber, plots_dir,
        )

    return findings


def _plot_sensitivity_scatter(
    default_pc1: np.ndarray,
    sens_pc1: np.ndarray,
    correlation: float,
    chamber: str,
    out_dir: Path,
) -> None:
    """Scatter plot comparing default and sensitivity PC1 scores."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(default_pc1, sens_pc1, c="#4C72B0", s=40, alpha=0.7, edgecolors="black",
               linewidth=0.5)

    # Identity line
    lims = [
        min(default_pc1.min(), sens_pc1.min()) - 0.5,
        max(default_pc1.max(), sens_pc1.max()) + 0.5,
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Identity line")

    ax.set_xlabel(f"PC1 (default: {MINORITY_THRESHOLD*100:.1f}% threshold)")
    ax.set_ylabel(f"PC1 (sensitivity: {SENSITIVITY_THRESHOLD*100:.0f}% threshold)")
    ax.set_title(f"{chamber} — PC1 Sensitivity (r = {correlation:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"sensitivity_pc1_{chamber.lower()}.png")


# ── Phase 5: Holdout Validation ─────────────────────────────────────────────


def run_holdout_validation(
    matrix: pl.DataFrame,
    chamber: str,
    n_components: int,
) -> dict:
    """Holdout validation: mask 20% of votes, fit PCA on rest, predict masked.

    Returns dict with accuracy, AUC-ROC, and reconstruction RMSE.
    """
    print(f"\n  {chamber} holdout validation:")

    slug_col = "legislator_slug"
    vote_ids = [c for c in matrix.columns if c != slug_col]
    X = matrix.select(vote_ids).to_numpy().astype(np.float64)

    # Identify non-null cells for masking
    non_null_mask = ~np.isnan(X)
    non_null_indices = list(zip(*np.where(non_null_mask)))

    # Randomly select 20% of non-null cells as holdout
    rng = np.random.default_rng(HOLDOUT_SEED)
    n_holdout = int(len(non_null_indices) * HOLDOUT_FRACTION)
    holdout_idx = rng.choice(len(non_null_indices), size=n_holdout, replace=False)
    holdout_cells = [non_null_indices[i] for i in holdout_idx]

    # Save true values and mask them
    true_values = np.array([X[r, c] for r, c in holdout_cells])
    X_train = X.copy()
    for r, c in holdout_cells:
        X_train[r, c] = np.nan

    # Impute training matrix with row mean
    for i in range(X_train.shape[0]):
        row = X_train[i]
        valid = ~np.isnan(row)
        if valid.any():
            X_train[i, ~valid] = row[valid].mean()
        else:
            X_train[i] = 0.5

    # Fit PCA on training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_scaled)

    # Reconstruct: X_hat = scores @ components, then un-standardize
    X_hat_scaled = scores @ pca.components_
    X_hat = scaler.inverse_transform(X_hat_scaled)

    # Extract predictions for holdout cells
    pred_continuous = np.array([X_hat[r, c] for r, c in holdout_cells])
    pred_binary = (pred_continuous >= 0.5).astype(float)

    # Metrics
    accuracy = float((pred_binary == true_values).mean())
    # Clip predictions for AUC-ROC (needs probabilities in [0,1])
    pred_clipped = np.clip(pred_continuous, 0, 1)
    try:
        auc = float(roc_auc_score(true_values, pred_clipped))
    except ValueError:
        auc = float("nan")
    rmse = float(np.sqrt(np.mean((pred_continuous - true_values) ** 2)))

    # Base rate for comparison
    base_rate = float(true_values.mean())
    base_accuracy = max(base_rate, 1 - base_rate)

    print(f"    Holdout cells: {n_holdout:,}")
    print(f"    Base rate (Yea): {base_rate:.3f}")
    print(f"    Base-rate accuracy: {base_accuracy:.3f}")
    print(f"    PCA accuracy: {accuracy:.3f}")
    print(f"    AUC-ROC: {auc:.3f}")
    print(f"    Reconstruction RMSE: {rmse:.3f}")

    if accuracy > base_accuracy:
        print(f"    Result: PASS (accuracy {accuracy:.3f} > base rate {base_accuracy:.3f})")
    else:
        print(f"    Result: FAIL (accuracy {accuracy:.3f} <= base rate {base_accuracy:.3f})")

    return {
        "chamber": chamber,
        "holdout_cells": n_holdout,
        "base_rate": base_rate,
        "base_accuracy": base_accuracy,
        "accuracy": accuracy,
        "auc_roc": auc,
        "rmse": rmse,
        "n_components": n_comp,
        "explained_variance_training": pca.explained_variance_ratio_.tolist(),
    }


# ── Phase 6: Filtering Manifest ─────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


# ── Phase 7: Main ───────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    session_slug = args.session.replace("-", "_")

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(f"data/ks_{session_slug}")

    if args.eda_dir:
        eda_dir = Path(args.eda_dir)
    else:
        session_full = args.session.replace("_", "-")
        # Expand abbreviated year: "2025-26" -> "2025-2026"
        parts = session_full.split("-")
        if len(parts) == 2 and len(parts[1]) == 2:
            century = parts[0][:2]
            session_full = f"{parts[0]}-{century}{parts[1]}"
        eda_dir = Path(f"results/{session_full}/eda/latest")

    with RunContext(
        session=args.session,
        analysis_name="pca",
        params=vars(args),
        primer=PCA_PRIMER,
    ) as ctx:
        print(f"KS Legislature PCA — Session {args.session}")
        print(f"Data:     {data_dir}")
        print(f"EDA:      {eda_dir}")
        print(f"Output:   {ctx.run_dir}")
        print(f"Components: {args.n_components}")

        # ── Phase 1: Load data ──
        print_header("LOADING DATA")
        house_matrix, senate_matrix, full_matrix = load_eda_matrices(eda_dir)
        rollcalls, legislators = load_metadata(data_dir)

        print(f"  House filtered: {house_matrix.height} x {len(house_matrix.columns) - 1}")
        print(f"  Senate filtered: {senate_matrix.height} x {len(senate_matrix.columns) - 1}")
        print(f"  Full matrix: {full_matrix.height} x {len(full_matrix.columns) - 1}")
        print(f"  Rollcalls: {rollcalls.height}")
        print(f"  Legislators: {legislators.height}")

        # ── Phase 2: PCA per chamber ──
        results: dict[str, dict] = {}
        ev_rows = []

        for label, matrix in [("House", house_matrix), ("Senate", senate_matrix)]:
            if matrix.height < 3:
                print(f"\n  Skipping {label}: too few legislators ({matrix.height})")
                continue

            result = run_pca_for_chamber(
                matrix, label, args.n_components, legislators, rollcalls,
            )
            results[label] = result

            # Save parquet files
            result["scores_df"].write_parquet(
                ctx.data_dir / f"pc_scores_{label.lower()}.parquet"
            )
            result["loadings_df"].write_parquet(
                ctx.data_dir / f"pc_loadings_{label.lower()}.parquet"
            )
            print(f"  Saved: pc_scores_{label.lower()}.parquet")
            print(f"  Saved: pc_loadings_{label.lower()}.parquet")

            # Collect explained variance for combined output
            for i, v in enumerate(result["explained_variance"]):
                ev_rows.append({
                    "chamber": label,
                    "component": f"PC{i+1}",
                    "explained_variance": v,
                    "cumulative": float(np.cumsum(result["explained_variance"])[i]),
                })

        # Save combined explained variance
        if ev_rows:
            ev_df = pl.DataFrame(ev_rows)
            ev_df.write_parquet(ctx.data_dir / "explained_variance.parquet")
            print("  Saved: explained_variance.parquet")

        # ── Phase 3: Plots ──
        print_header("GENERATING PLOTS")
        for label, result in results.items():
            plot_scree(result["pca"], label, ctx.plots_dir)
            plot_ideological_map(result["scores_df"], label, ctx.plots_dir)
            plot_pc1_distribution(result["scores_df"], label, ctx.plots_dir)

        # ── Phase 4: Sensitivity analysis ──
        sensitivity_findings: dict = {}
        if not args.skip_sensitivity:
            sensitivity_findings = run_sensitivity(
                full_matrix, results, rollcalls, legislators,
                args.n_components, ctx.plots_dir,
            )
        else:
            print_header("SENSITIVITY ANALYSIS (SKIPPED)")

        # ── Phase 5: Holdout validation ──
        print_header("HOLDOUT VALIDATION")
        validation_results: dict[str, dict] = {}
        for label, matrix in [("House", house_matrix), ("Senate", senate_matrix)]:
            if matrix.height < 3:
                continue
            validation_results[label] = run_holdout_validation(
                matrix, label, args.n_components,
            )

        # ── Phase 6: Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest = {
            "eda_source": str(eda_dir),
            "n_components": args.n_components,
            "impute_method": "row_mean",
            "minority_threshold_default": MINORITY_THRESHOLD,
            "minority_threshold_sensitivity": SENSITIVITY_THRESHOLD,
            "min_votes": MIN_VOTES,
            "holdout_fraction": HOLDOUT_FRACTION,
            "holdout_seed": HOLDOUT_SEED,
        }
        for label, result in results.items():
            manifest[f"{label.lower()}_explained_variance"] = result["explained_variance"]
            manifest[f"{label.lower()}_n_legislators"] = result["scores_df"].height
            manifest[f"{label.lower()}_n_votes"] = len(result["vote_ids"])
        if sensitivity_findings:
            manifest["sensitivity"] = sensitivity_findings
        if validation_results:
            manifest["validation"] = validation_results
        save_filtering_manifest(manifest, ctx.run_dir)

        # ── Phase 7: HTML report ──
        print_header("HTML REPORT")
        build_pca_report(
            ctx.report,
            results=results,
            sensitivity_findings=sensitivity_findings,
            validation_results=validation_results,
            plots_dir=ctx.plots_dir,
            n_components=args.n_components,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
