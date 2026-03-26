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
  - plots/:  PNG visualizations (scree, ideological map, PC1 distribution,
             scatter matrix, loading heatmap, sensitivity)
  - filtering_manifest.json, run_info.json, run_log.txt
  - pca_report.html
"""

import argparse
import json
import sys
from dataclasses import dataclass
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir

try:
    from analysis.phase_utils import load_metadata, print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import load_metadata, print_header, save_fig

try:
    from analysis.pca_report import build_pca_report
except ModuleNotFoundError:
    from pca_report import build_pca_report  # type: ignore[no-redef]

try:
    from analysis.tuning import CONTESTED_THRESHOLD, MIN_VOTES, PARTY_COLORS, SENSITIVITY_THRESHOLD
except ModuleNotFoundError:
    from tuning import CONTESTED_THRESHOLD, MIN_VOTES, PARTY_COLORS, SENSITIVITY_THRESHOLD

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

Reads from `data/{legislature}_{start}-{end}/`:
- `{output_name}_rollcalls.csv` — Roll call metadata (bill numbers, titles, motions)
- `{output_name}_legislators.csv` — Legislator metadata (names, parties, districts)

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
HOLDOUT_FRACTION = 0.20  # Random 20% of non-null cells
HOLDOUT_SEED = 42
PARALLEL_ANALYSIS_N_ITER = 100  # Horn's parallel analysis: random data iterations
RECONSTRUCTION_ERROR_THRESHOLD_SD = 2.0  # Flag legislators with RMSE > mean + 2σ


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature PCA")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help="Number of PCA components to extract",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis (faster, for debugging)",
    )
    return parser.parse_args()


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_eda_matrices(
    eda_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame] | None:
    """Load filtered vote matrices from the EDA phase output.

    Returns (house_filtered, senate_filtered, full_matrix), or None if unavailable.
    """
    house_path = eda_dir / "data" / "vote_matrix_house_filtered.parquet"
    senate_path = eda_dir / "data" / "vote_matrix_senate_filtered.parquet"
    full_path = eda_dir / "data" / "vote_matrix_full.parquet"
    if not house_path.exists() or not senate_path.exists():
        return None
    house = pl.read_parquet(house_path)
    senate = pl.read_parquet(senate_path)
    full = pl.read_parquet(full_path) if full_path.exists() else pl.DataFrame()
    return house, senate, full


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

    # Vectorized row-mean imputation: fill NaN with each legislator's mean Yea rate
    row_means = np.nanmean(X, axis=1, keepdims=True)
    # All-NaN rows (no votes at all) → fill with 0.5 (uninformative)
    row_means = np.where(np.isnan(row_means), 0.5, row_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.broadcast_to(row_means, X.shape)[nan_mask]

    return X, slugs, vote_ids


def parallel_analysis(
    n_obs: int,
    n_vars: int,
    n_components: int,
    n_iter: int = PARALLEL_ANALYSIS_N_ITER,
) -> np.ndarray:
    """95th-percentile eigenvalues from random data of same shape (Horn 1965).

    Generates n_iter random matrices, computes correlation-matrix eigenvalues,
    and returns the 95th percentile at each position. Components whose actual
    eigenvalues exceed these thresholds are statistically significant.

    Returns array of threshold eigenvalues for the first n_components components.
    """
    rng = np.random.default_rng(HOLDOUT_SEED)
    k = min(n_components, n_obs, n_vars)
    random_evals = np.zeros((n_iter, k))
    for i in range(n_iter):
        random_data = rng.standard_normal((n_obs, n_vars))
        corr = np.corrcoef(random_data, rowvar=False)
        evals = np.sort(np.linalg.eigvalsh(corr))[::-1][:k]
        random_evals[i] = evals
    return np.percentile(random_evals, 95, axis=0)


def compute_reconstruction_error(
    X_imputed: np.ndarray,
    scores: np.ndarray,
    pca_obj: PCA,
    scaler: StandardScaler,
    slugs: list[str],
) -> pl.DataFrame:
    """Per-legislator reconstruction RMSE from the PCA model.

    Legislators with high reconstruction error have voting patterns poorly
    explained by the dominant dimensions — candidates for IRT convergence issues
    or synthesis outliers.

    Flags legislators with RMSE > mean + RECONSTRUCTION_ERROR_THRESHOLD_SD × σ.
    """
    X_hat_scaled = scores @ pca_obj.components_
    X_hat = scaler.inverse_transform(X_hat_scaled)
    per_row_mse = np.mean((X_imputed - X_hat) ** 2, axis=1)
    per_row_rmse = np.sqrt(per_row_mse)

    mean_rmse = float(np.mean(per_row_rmse))
    std_rmse = float(np.std(per_row_rmse))
    threshold = mean_rmse + RECONSTRUCTION_ERROR_THRESHOLD_SD * std_rmse

    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "reconstruction_rmse": per_row_rmse.tolist(),
            "high_error": [bool(v > threshold) for v in per_row_rmse],
        }
    )


def fit_pca(
    X: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, PCA, StandardScaler]:
    """Standardize and fit PCA. Returns (scores, loadings, pca, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_  # shape: (n_components, n_votes)
    return scores, loadings, pca, scaler


def _build_slug_party_map(legislators: pl.DataFrame) -> dict[str, str]:
    """Build slug → party lookup from legislators DataFrame."""
    return dict(legislators.select("legislator_slug", "party").iter_rows())


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
    slug_to_party = dict(legislators.select("legislator_slug", "party").iter_rows())
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
    pc_cols = {f"PC{i + 1}": scores[:, i].tolist() for i in range(n_components)}
    df = pl.DataFrame({"legislator_slug": slugs, **pc_cols})

    # Join legislator metadata
    meta = legislators.select("legislator_slug", "full_name", "party", "district", "chamber")
    df = df.join(meta, on="legislator_slug", how="left")
    return df


def build_loadings_df(
    loadings: np.ndarray,
    vote_ids: list[str],
    n_components: int,
    rollcalls: pl.DataFrame,
) -> pl.DataFrame:
    """Build a polars DataFrame of PC loadings with rollcall metadata."""
    pc_cols = {f"PC{i + 1}": loadings[i, :].tolist() for i in range(n_components)}
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
    X_raw, slugs, vote_ids, explained_variance.
    """
    print_header(f"PCA — {chamber}")
    print(f"  Matrix: {matrix.height} legislators x {len(matrix.columns) - 1} votes")

    # Extract raw vote matrix (with NaN) before imputation — needed for absence diagnostics
    vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
    X_raw = matrix.select(vote_cols).to_numpy().astype(np.float64)

    X, slugs, vote_ids = impute_vote_matrix(matrix)
    n_comp = min(n_components, X.shape[0], X.shape[1])
    scores, loadings, pca, scaler = fit_pca(X, n_comp)
    scores, loadings = orient_pc1(scores, loadings, slugs, legislators)

    # Party separation on each PC (Cohen's d)
    # Detects axis instability: when PC2 separates parties better than PC1
    slug_party = _build_slug_party_map(legislators)
    pc_party_d: dict[str, float] = {}
    for pc_i in range(min(n_comp, 2)):
        pc_name = f"PC{pc_i + 1}"
        r_scores = scores[
            [i for i, s in enumerate(slugs) if slug_party.get(s) == "Republican"], pc_i
        ]
        d_scores = scores[[i for i, s in enumerate(slugs) if slug_party.get(s) == "Democrat"], pc_i]
        if len(r_scores) > 0 and len(d_scores) > 0:
            pooled_sd = np.sqrt((r_scores.std() ** 2 + d_scores.std() ** 2) / 2)
            d_val = abs(r_scores.mean() - d_scores.mean()) / pooled_sd if pooled_sd > 0 else 0.0
            pc_party_d[pc_name] = float(d_val)
        else:
            pc_party_d[pc_name] = 0.0

    if "PC1" in pc_party_d and "PC2" in pc_party_d:
        if pc_party_d["PC2"] > pc_party_d["PC1"] and pc_party_d["PC2"] > 2.0:
            print(
                f"\n  ⚠ AXIS SWAP: PC2 (d={pc_party_d['PC2']:.2f}) separates parties "
                f"more than PC1 (d={pc_party_d['PC1']:.2f}). "
                "PC1 captures intra-majority-party variation."
            )
        else:
            print(
                f"\n  Party separation: PC1 d={pc_party_d['PC1']:.2f}, "
                f"PC2 d={pc_party_d['PC2']:.2f}"
            )

    # Print explained variance
    ev = pca.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    print("\n  Explained variance:")
    for i in range(n_comp):
        print(
            f"    PC{i + 1}: {ev[i]:.4f} ({100 * ev[i]:.1f}%)"
            f"  cumulative: {100 * cumulative[i]:.1f}%"
        )

    # Eigenvalue ratio (lambda1/lambda2)
    eigenvalues = pca.explained_variance_
    has_two = n_comp >= 2 and eigenvalues[1] > 0
    eigenvalue_ratio = float(eigenvalues[0] / eigenvalues[1]) if has_two else float("inf")
    print(f"\n  Eigenvalue ratio (λ1/λ2): {eigenvalue_ratio:.2f}")
    if eigenvalue_ratio > 5:
        print("    → Strongly one-dimensional (ratio > 5)")
    elif eigenvalue_ratio > 3:
        print("    → Predominantly one-dimensional (ratio 3-5)")
    else:
        print("    → Meaningful second dimension (ratio < 3)")

    # Parallel analysis (Horn 1965)
    pa_thresholds = parallel_analysis(X.shape[0], X.shape[1], n_comp)
    n_significant = int(
        sum(actual > threshold for actual, threshold in zip(eigenvalues[:n_comp], pa_thresholds))
    )
    print("\n  Parallel analysis (95th pct. of random data):")
    for i in range(n_comp):
        sig = "✓" if eigenvalues[i] > pa_thresholds[i] else "✗"
        print(f"    PC{i + 1}: λ={eigenvalues[i]:.2f}  threshold={pa_thresholds[i]:.2f}  {sig}")
    print(f"    Significant dimensions: {n_significant}")

    # Per-legislator reconstruction error
    recon_df = compute_reconstruction_error(X, scores, pca, scaler, slugs)

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

    from analysis.tuning import EIGENVALUE_RATIO_AMBIGUOUS

    return {
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "pca": pca,
        "scaler": scaler,
        "X_imputed": X,
        "X_raw": X_raw,
        "slugs": slugs,
        "vote_ids": vote_ids,
        "explained_variance": ev.tolist(),
        "n_components": n_comp,
        "eigenvalue_ratio": eigenvalue_ratio,
        "axis_ambiguous": eigenvalue_ratio < EIGENVALUE_RATIO_AMBIGUOUS,
        "parallel_thresholds": pa_thresholds,
        "n_significant": n_significant,
        "reconstruction_error_df": recon_df,
        "pc_party_d": pc_party_d,
    }


# ── Phase 3: Plots ──────────────────────────────────────────────────────────


def plot_scree(
    pca_obj: PCA,
    chamber: str,
    out_dir: Path,
    parallel_thresholds: np.ndarray | None = None,
) -> None:
    """Scree plot with individual and cumulative explained variance (2 panels).

    If parallel_thresholds is provided, draws a reference line on panel 1 showing
    the 95th-percentile eigenvalues from random data (Horn's parallel analysis).
    Components above this line are statistically significant.
    """
    ev = pca_obj.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    n = len(ev)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Individual explained variance
    axes[0].bar(
        range(1, n + 1),
        ev,
        color="#4C72B0",
        edgecolor="black",
        alpha=0.9,
    )
    for i, v in enumerate(ev):
        axes[0].text(i + 1, v + 0.005, f"{100 * v:.1f}%", ha="center", fontsize=9)

    # Parallel analysis threshold line
    if parallel_thresholds is not None:
        n_features = pca_obj.n_features_in_
        threshold_ratios = parallel_thresholds[:n] / n_features
        axes[0].plot(
            range(1, n + 1),
            threshold_ratios,
            "r--o",
            markersize=5,
            alpha=0.7,
            label="95th pct. random data",
        )
        axes[0].legend(loc="upper right")

    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title(f"{chamber} \u2014 How Many Dimensions Does Kansas Politics Have?")
    axes[0].set_xticks(range(1, n + 1))
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Annotate the elbow (PC1 dominance)
    if n >= 2 and ev[0] > 2 * ev[1]:
        axes[0].annotate(
            "The sharp drop means Kansas is\nessentially a one-dimensional\n"
            "legislature \u2014 party affiliation\nexplains almost everything",
            xy=(1.5, (ev[0] + ev[1]) / 2),
            xytext=(3.0, ev[0] * 0.7),
            fontsize=8,
            fontstyle="italic",
            color="#555555",
            bbox={"boxstyle": "round,pad=0.4", "fc": "lightyellow", "alpha": 0.8, "ec": "#cccccc"},
            arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.2},
        )

    # Panel 2: Cumulative explained variance
    axes[1].plot(range(1, n + 1), cumulative, "bo-", markersize=8)
    for i, v in enumerate(cumulative):
        axes[1].text(i + 1, v + 0.02, f"{100 * v:.1f}%", ha="center", fontsize=9)
    axes[1].axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance")
    axes[1].set_title(f"{chamber} \u2014 Cumulative Variance Explained")
    axes[1].set_xticks(range(1, n + 1))
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"scree_{chamber.lower()}.png")


@dataclass(frozen=True)
class ExtremePC2Legislator:
    """A legislator with an extreme PC2 score (>3σ from median)."""

    slug: str
    full_name: str
    party: str
    pc1: float
    pc2: float
    pc2_std: float


def detect_extreme_pc2(scores_df: pl.DataFrame) -> ExtremePC2Legislator | None:
    """Detect the most extreme PC2 legislator if >3σ from the pack.

    Returns None if no legislator exceeds the 3σ threshold.
    """
    if "PC2" not in scores_df.columns or scores_df.height < 2:
        return None

    pc2_std = float(scores_df["PC2"].std())
    if pc2_std <= 0:
        return None

    pc2_min_idx = scores_df["PC2"].arg_min()
    row = scores_df.row(pc2_min_idx, named=True)
    pc2_min_val = row["PC2"]

    if abs(pc2_min_val) <= 3 * pc2_std:
        return None

    slug = row["legislator_slug"]
    raw_name = row.get("full_name") or slug
    full_name = raw_name.split(" - ")[0].strip()

    return ExtremePC2Legislator(
        slug=slug,
        full_name=full_name,
        party=row.get("party") or "Unknown",
        pc1=float(row["PC1"]),
        pc2=float(pc2_min_val),
        pc2_std=pc2_std,
    )


def plot_ideological_map(
    scores_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
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
            c=color,
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
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
            raw_name = row.get("full_name") or slug
            name = raw_name.split(" - ")[0].strip()
            last_name = name.split()[-1] if name else slug
            ax.annotate(
                last_name,
                (row["PC1"], row["PC2"]),
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
                xytext=(6, 6),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.2", "fc": "wheat", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
            )

    # Add callout box for extreme PC2 legislator (data-driven detection)
    extreme = detect_extreme_pc2(scores_df)
    if extreme is not None:
        last = extreme.full_name.split()[-1]
        ax.annotate(
            f"{last}: extreme contrarian\n"
            f"(PC2 = {extreme.pc2:.1f}, 3\u00d7 more extreme\nthan next legislator)",
            xy=(extreme.pc1, extreme.pc2),
            xytext=(extreme.pc1 + 2, extreme.pc2 + 3),
            fontsize=8,
            fontstyle="italic",
            color="#555555",
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": "lightyellow",
                "alpha": 0.8,
                "ec": "#cccccc",
            },
            arrowprops={"arrowstyle": "->", "color": "#E81B23", "lw": 1.5},
        )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.2)
    ax.set_xlabel("PC1 (primary ideological dimension)")
    ax.set_ylabel("PC2 (contrarianism \u2014 voting Nay on routine, near-unanimous bills)")
    ax.set_title(f"{chamber} \u2014 The Ideological Landscape of the Kansas Legislature")
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
    save_fig(fig, out_dir / f"ideological_map_{chamber.lower()}.png")


def plot_pc1_distribution(
    scores_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
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


def plot_score_scatter_matrix(
    scores_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
    n_significant: int,
) -> None:
    """Pairwise scatter matrix of significant PC scores, colored by party."""
    if n_significant < 2:
        return

    import seaborn as sns

    n = min(n_significant, 5)
    pc_cols = [f"PC{i}" for i in range(1, n + 1)]

    # Guard: skip if any PC column is missing
    for col in pc_cols:
        if col not in scores_df.columns:
            return

    # Build a pandas DataFrame for seaborn (pairplot requires pandas)
    pdf = scores_df.select([*pc_cols, "party"]).to_pandas()

    # Only include parties present in the data
    present_parties = [p for p in PARTY_COLORS if p in pdf["party"].values]
    palette = {p: PARTY_COLORS[p] for p in present_parties}

    g = sns.pairplot(
        pdf,
        hue="party",
        vars=pc_cols,
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "black", "linewidth": 0.3},
        diag_kws={"fill": True, "alpha": 0.3},
        height=2.5,
    )

    # Label top-3 outliers on off-diagonal panels
    for i, row_col in enumerate(pc_cols):
        for j, col_col in enumerate(pc_cols):
            if i == j:
                continue
            ax = g.axes[i, j]
            abs_dist = scores_df[row_col].abs() + scores_df[col_col].abs()
            top_idx = abs_dist.arg_sort(descending=True).head(3).to_list()
            for idx in top_idx:
                row = scores_df.row(idx, named=True)
                raw_name = row.get("full_name") or row["legislator_slug"]
                last_name = raw_name.split(" - ")[0].strip().split()[-1]
                ax.annotate(
                    last_name,
                    (row[col_col], row[row_col]),
                    fontsize=6,
                    ha="left",
                    va="bottom",
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    g.figure.suptitle(
        f"{chamber} — Score Scatter Matrix (PC1–PC{n})",
        y=1.02,
        fontsize=14,
    )
    g.figure.tight_layout()
    save_fig(g.figure, out_dir / f"scatter_matrix_{chamber.lower()}.png")
    plt.close(g.figure)


def plot_loading_heatmap(
    loadings_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
    n_significant: int,
) -> None:
    """Heatmap of top loadings across significant PCs."""
    if n_significant < 2:
        return

    import seaborn as sns

    n = min(n_significant, 5)
    pc_cols = [f"PC{i}" for i in range(1, n + 1)]

    # Guard: skip if any PC column is missing
    for col in pc_cols:
        if col not in loadings_df.columns:
            return

    # Collect union of top-5 absolute-loading bills per significant PC
    selected_indices: set[int] = set()
    for col in pc_cols:
        abs_vals = loadings_df[col].abs()
        top_idx = abs_vals.arg_sort(descending=True).head(5).to_list()
        selected_indices.update(top_idx)

    if not selected_indices:
        return

    subset = loadings_df[sorted(selected_indices)]

    # Build row labels: bill_number + short_title[:30]
    row_labels = []
    for row in subset.iter_rows(named=True):
        label = row.get("bill_number") or row["vote_id"]
        title = row.get("short_title")
        if title and str(title) != "None" and str(title).strip():
            label = f"{label} — {str(title)[:30]}"
        row_labels.append(label)

    # Extract loading values as numpy array
    data = subset.select(pc_cols).to_numpy()

    fig, ax = plt.subplots(figsize=(max(6, n * 1.5), max(8, len(row_labels) * 0.4)))
    sns.heatmap(
        data,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".3f",
        xticklabels=pc_cols,
        yticklabels=row_labels,
        linewidths=0.5,
        cbar_kws={"label": "Loading"},
    )
    ax.set_title(f"{chamber} — Top Loadings Across Significant Dimensions")
    ax.set_ylabel("Bill")
    ax.set_xlabel("Component")

    fig.tight_layout()
    save_fig(fig, out_dir / f"loading_heatmap_{chamber.lower()}.png")


def diagnose_pc2_horseshoe(scores_df: pl.DataFrame) -> dict:
    """Diagnose horseshoe artifact: fit PC2 ~ PC1 + PC1² and report R².

    Returns {"r_squared": float, "horseshoe_detected": bool}.
    """
    if "PC1" not in scores_df.columns or "PC2" not in scores_df.columns:
        return {"r_squared": 0.0, "horseshoe_detected": False}
    if scores_df.height < 5:
        return {"r_squared": 0.0, "horseshoe_detected": False}

    pc1 = scores_df["PC1"].to_numpy()
    pc2 = scores_df["PC2"].to_numpy()

    # Fit quadratic: PC2 = a*PC1² + b*PC1 + c
    coeffs = np.polyfit(pc1, pc2, 2)
    pc2_pred = np.polyval(coeffs, pc1)

    ss_res = np.sum((pc2 - pc2_pred) ** 2)
    ss_tot = np.sum((pc2 - np.mean(pc2)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"r_squared": float(r_squared), "horseshoe_detected": r_squared > 0.30}


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
    chamber_vote_ids = set(rollcalls.filter(pl.col("chamber") == chamber)["vote_id"].to_list())
    prefix = "sen_" if chamber == "Senate" else "rep_"
    vote_cols = [c for c in vote_cols if c in chamber_vote_ids]
    matrix = full_matrix.filter(pl.col(slug_col).str.starts_with(prefix)).select(
        [slug_col, *vote_cols]
    )

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
        pl.sum_horizontal(*[pl.col(c).is_not_null().cast(pl.Int32) for c in contested_cols]).alias(
            "n_votes"
        ),
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
            full_matrix,
            rollcalls,
            chamber,
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
            f"PC{i + 1}={100 * v:.1f}%" for i, v in enumerate(pca.explained_variance_ratio_)
        )
        print(f"    Sensitivity EV: {ev_str}")

        if correlation > 0.95:
            print("    Result: ROBUST (r > 0.95)")
        else:
            print("    Result: SENSITIVE (r <= 0.95) — investigate threshold dependence")

        findings[chamber] = {
            "default_threshold": CONTESTED_THRESHOLD,
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
            default_arr,
            sens_arr,
            correlation,
            chamber,
            plots_dir,
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
    ax.scatter(
        default_pc1, sens_pc1, c="#4C72B0", s=40, alpha=0.7, edgecolors="black", linewidth=0.5
    )

    # Identity line
    lims = [
        min(default_pc1.min(), sens_pc1.min()) - 0.5,
        max(default_pc1.max(), sens_pc1.max()) + 0.5,
    ]
    ax.plot(lims, lims, "r--", alpha=0.5, label="Identity line")

    ax.set_xlabel(f"PC1 (default: {CONTESTED_THRESHOLD * 100:.1f}% threshold)")
    ax.set_ylabel(f"PC1 (sensitivity: {SENSITIVITY_THRESHOLD * 100:.0f}% threshold)")
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

    with RunContext(
        session=args.session,
        analysis_name="02_pca",
        params=vars(args),
        primer=PCA_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"KS Legislature PCA — Session {args.session}")
        print(f"Data:     {data_dir}")
        print(f"EDA:      {eda_dir}")
        print(f"Output:   {ctx.run_dir}")
        print(f"Components: {args.n_components}")

        # ── Phase 1: Load data ──
        print_header("LOADING DATA")
        eda_result = load_eda_matrices(eda_dir)
        if eda_result is None:
            print("Phase 02 (PCA): skipping — no EDA vote matrices available")
            return
        house_matrix, senate_matrix, full_matrix = eda_result
        if house_matrix.height == 0 and senate_matrix.height == 0:
            print("Phase 02 (PCA): skipping — 0 legislators after EDA filtering")
            return
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
                matrix,
                label,
                args.n_components,
                legislators,
                rollcalls,
            )
            results[label] = result

            # Save parquet files
            result["scores_df"].write_parquet(ctx.data_dir / f"pc_scores_{label.lower()}.parquet")
            ctx.export_csv(
                result["scores_df"],
                f"pc_scores_{label.lower()}.csv",
                f"PCA scores for {label} legislators",
            )
            result["loadings_df"].write_parquet(
                ctx.data_dir / f"pc_loadings_{label.lower()}.parquet"
            )
            result["reconstruction_error_df"].write_parquet(
                ctx.data_dir / f"reconstruction_error_{label.lower()}.parquet"
            )
            print(f"  Saved: pc_scores_{label.lower()}.parquet")
            print(f"  Saved: pc_loadings_{label.lower()}.parquet")
            print(f"  Saved: reconstruction_error_{label.lower()}.parquet")

            # Collect explained variance for combined output
            for i, v in enumerate(result["explained_variance"]):
                ev_rows.append(
                    {
                        "chamber": label,
                        "component": f"PC{i + 1}",
                        "explained_variance": v,
                        "cumulative": float(np.cumsum(result["explained_variance"])[i]),
                    }
                )

        # Save combined explained variance
        if ev_rows:
            ev_df = pl.DataFrame(ev_rows)
            ev_df.write_parquet(ctx.data_dir / "explained_variance.parquet")
            print("  Saved: explained_variance.parquet")

        # ── Phase 3: Plots ──
        print_header("GENERATING PLOTS")
        for label, result in results.items():
            plot_scree(result["pca"], label, ctx.plots_dir, result["parallel_thresholds"])
            plot_ideological_map(result["scores_df"], label, ctx.plots_dir)
            plot_pc1_distribution(result["scores_df"], label, ctx.plots_dir)
            n_sig = result["n_significant"]
            if n_sig >= 2:
                plot_score_scatter_matrix(result["scores_df"], label, ctx.plots_dir, n_sig)
                plot_loading_heatmap(result["loadings_df"], label, ctx.plots_dir, n_sig)

        # ── Phase 4: Sensitivity analysis ──
        sensitivity_findings: dict = {}
        if not args.skip_sensitivity:
            sensitivity_findings = run_sensitivity(
                full_matrix,
                results,
                rollcalls,
                legislators,
                args.n_components,
                ctx.plots_dir,
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
                matrix,
                label,
                args.n_components,
            )

        # ── Phase 6: Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest = {
            "eda_source": str(eda_dir),
            "n_components": args.n_components,
            "impute_method": "row_mean",
            "minority_threshold_default": CONTESTED_THRESHOLD,
            "minority_threshold_sensitivity": SENSITIVITY_THRESHOLD,
            "min_votes": MIN_VOTES,
            "holdout_fraction": HOLDOUT_FRACTION,
            "holdout_seed": HOLDOUT_SEED,
        }
        for label, result in results.items():
            manifest[f"{label.lower()}_explained_variance"] = result["explained_variance"]
            manifest[f"{label.lower()}_n_legislators"] = result["scores_df"].height
            manifest[f"{label.lower()}_n_votes"] = len(result["vote_ids"])
            manifest[f"{label.lower()}_eigenvalue_ratio"] = result["eigenvalue_ratio"]
            manifest[f"{label.lower()}_n_significant_dimensions"] = result["n_significant"]
            manifest[f"{label.lower()}_parallel_thresholds"] = result[
                "parallel_thresholds"
            ].tolist()
            recon = result["reconstruction_error_df"]
            n_high = recon.filter(pl.col("high_error")).height
            manifest[f"{label.lower()}_reconstruction_error"] = {
                "mean_rmse": float(recon["reconstruction_rmse"].mean()),
                "max_rmse": float(recon["reconstruction_rmse"].max()),
                "n_high_error": n_high,
            }
        if sensitivity_findings:
            manifest["sensitivity"] = sensitivity_findings
        if validation_results:
            manifest["validation"] = validation_results
        save_filtering_manifest(manifest, ctx.run_dir)

        # ── Phase 6b: TEFI dimensionality comparison ──
        print_header("TEFI DIMENSIONALITY")
        try:
            from analysis.ega.tefi import compute_tefi

            for label, result in results.items():
                loadings = result["loadings_df"]
                n_sig = result.get("n_significant", 2)
                # Compute tetrachoric or use PCA loadings-based assignments
                # Assign each bill to its highest-loading PC
                pc_cols = [c for c in loadings.columns if c.startswith("PC")]
                if pc_cols:
                    loading_arr = loadings.select(pc_cols).to_numpy()
                    # Correlation matrix from bill loadings (p × p, bills as rows)
                    corr_approx = np.corrcoef(loading_arr)
                    p = loading_arr.shape[0]

                    tefi_scores: dict[int, float] = {}
                    for k in range(1, min(6, len(pc_cols) + 1)):
                        if k == 1:
                            assigns = np.zeros(p, dtype=np.int64)
                        else:
                            # Assign to highest-loading PC among first k
                            assigns = np.argmax(np.abs(loading_arr[:, :k]), axis=1).astype(np.int64)
                        tefi_scores[k] = compute_tefi(corr_approx, assigns)

                    best_k = min(tefi_scores, key=tefi_scores.get)
                    result["tefi_scores"] = tefi_scores
                    result["tefi_best_k"] = best_k
                    print(f"  {label}: TEFI best K={best_k}")

                    # Save TEFI JSON
                    import json as json_mod

                    tefi_path = ctx.data_dir / f"tefi_pca_{label.lower()}.json"
                    with open(tefi_path, "w") as f:
                        json_mod.dump(
                            {str(k): v for k, v in tefi_scores.items()},
                            f,
                            indent=2,
                        )

                    # Plot TEFI curve
                    fig_tefi, ax_tefi = plt.subplots(figsize=(8, 5))
                    ks = sorted(tefi_scores.keys())
                    vals = [tefi_scores[k] for k in ks]
                    ax_tefi.plot(ks, vals, "o-", color="#2C3E50", linewidth=2, markersize=8)
                    ax_tefi.axvline(
                        best_k,
                        color="#E74C3C",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Best K={best_k}",
                    )
                    ax_tefi.set_xlabel("Number of Dimensions (K)")
                    ax_tefi.set_ylabel("TEFI (lower = better)")
                    ax_tefi.set_title(f"TEFI — {label}")
                    ax_tefi.legend()
                    ax_tefi.set_xticks(ks)
                    save_fig(fig_tefi, ctx.plots_dir / f"tefi_{label.lower()}.png")
        except ImportError:
            print("  EGA library not available — skipping TEFI")

        # ── Phase 7: HTML report ──
        print_header("HTML REPORT")
        build_pca_report(
            ctx.report,
            results=results,
            sensitivity_findings=sensitivity_findings,
            validation_results=validation_results,
            plots_dir=ctx.plots_dir,
            n_components=args.n_components,
            session=args.session,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
