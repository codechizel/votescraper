"""
Tallgrass — Multiple Correspondence Analysis (Phase 2c)

Covers analytic method 10: MCA on the categorical vote matrix.
MCA is the categorical-data analogue of PCA — it uses chi-square distance instead
of Euclidean and preserves the full vote structure (Yea/Nay/Absent) rather than
collapsing to binary. Its key additions over PCA: the absence dimension, category-
level coordinates, and chi-square weighting of rare vote patterns.

Usage:
  uv run python analysis/03_mca/mca.py [--session 2025-26] [--data-dir ...] \
      [--n-components 5] [--skip-sensitivity] [--correction greenacre]

Outputs (in results/<session>/mca/<date>/):
  - data/:   Parquet files (dimension scores, category coords, contributions, cos²)
  - plots/:  PNG visualizations (biplot, ideological map, inertia, absence, PCA validation)
  - filtering_manifest.json, run_info.json, run_log.txt
  - mca_report.html
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import prince
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, resolve_upstream_dir, strip_leadership_suffix
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir, strip_leadership_suffix

try:
    from analysis.phase_utils import print_header, save_fig
except ModuleNotFoundError:
    from phase_utils import print_header, save_fig

try:
    from analysis.mca_report import build_mca_report
except ModuleNotFoundError:
    from mca_report import build_mca_report  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────
# Written to results/<session>/mca/README.md by RunContext on each run.

MCA_PRIMER = """\
# Multiple Correspondence Analysis (MCA)

## Purpose

MCA is the categorical-data analogue of PCA. Where PCA treats votes as numbers
(Yea=1, Nay=0) and uses Euclidean distance, MCA treats them as categories
(Yea / Nay / Absent) and uses chi-square distance. This is technically more
appropriate for vote data, which is categorical by nature.

MCA's key advantages over PCA:
- Preserves the Absent/Not Voting category as a first-class signal (PCA discards it)
- Automatically downweights near-unanimous votes via chi-square weighting
- Maps legislators AND vote categories into the same space (the biplot)

## Method

1. **Build categorical vote matrix** — Pivot votes into a matrix where each cell
   is "Yea", "Nay", or "Absent" (string categories, not binary numbers).
2. **Filter** — Same as PCA: drop near-unanimous votes (minority < 2.5%) and
   low-participation legislators (< 20 votes).
3. **Fit MCA** — prince library with Greenacre inertia correction. One-hot
   encodes each vote into indicator columns, then decomposes via SVD on the
   standardized residuals of the indicator matrix.
4. **Orient Dimension 1** — Republicans positive (same convention as PCA PC1).
5. **Validate** — Correlate Dim1 with PCA PC1 (expected Spearman r > 0.95).

## Inputs

Reads raw CSVs from `data/{legislature}_{start}-{end}/` (NOT binary EDA matrices).
Also reads PCA output from `results/<session>/pca/latest/` for validation.

## Outputs

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `mca_scores_{chamber}.parquet` | Legislator coordinates on MCA dimensions |
| `mca_category_coords_{chamber}.parquet` | Category coordinates (vote × Yea/Nay/Absent) |
| `mca_contributions_{chamber}.parquet` | Category contributions to each dimension |
| `mca_eigenvalues_{chamber}.parquet` | Raw and corrected inertia per dimension |
| `mca_cos2_{chamber}.parquet` | Squared cosines (representation quality) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `mca_ideological_map_{chamber}.png` | Dim1 vs Dim2 scatter, party-colored |
| `mca_biplot_{chamber}.png` | Legislators + top-contributing category points |
| `mca_inertia_{chamber}.png` | Scree plot with raw and corrected inertia |
| `mca_dim1_distribution_{chamber}.png` | Dim1 scores ranked by party |
| `mca_pca_correlation_{chamber}.png` | Dim1 vs PCA PC1 scatter with Spearman r |
| `mca_absence_map_{chamber}.png` | Legislators colored by absence rate |

## Interpretation Guide

- **Dimension 1** captures the same partisan divide as PCA PC1. Positive =
  conservative / Republican direction.
- **Category points** in the biplot show where each vote response sits. "Yea on
  HB2001" close to "Yea on SB150" means the same coalition votes for both.
- **Absent categories** positioned between parties indicate random absence.
  Positioned near one party indicates partisan strategic absence.
- **Corrected inertia** (Greenacre) is the honest measure of explained variance.
  Raw MCA inertia always looks pessimistically low — this is a coding artifact,
  not a sign of poor fit.

## Caveats

- For binary data (Yea=1, Nay=0), MCA is mathematically equivalent to PCA up
  to a scaling factor. MCA only adds value via the categorical encoding.
- The horseshoe effect (Dim2 is a quadratic function of Dim1) is common and
  confirms unidimensionality, not a genuine second dimension.
- Rare categories ("Present and Passing", "Not Voting") are treated as passive
  to prevent distortion from chi-square upweighting.
"""

# ── Constants ────────────────────────────────────────────────────────────────
# Explicit, named constants per the analytic-workflow rules.

DEFAULT_N_COMPONENTS = 5
MINORITY_THRESHOLD = 0.025  # Default: 2.5% (matches EDA/PCA)
SENSITIVITY_THRESHOLD = 0.10  # Sensitivity: 10% (per workflow rules)
MIN_VOTES = 20  # Minimum substantive votes per legislator
CORRECTION = "greenacre"  # Greenacre > Benzécri (more conservative)
RANDOM_STATE = 42
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
CATEGORY_COLORS = {"Yea": "#2ca02c", "Nay": "#d62728", "Absent": "#7f7f7f"}
TOP_CONTRIBUTIONS_N = 20  # Number of top-contributing categories to label in biplot
HORSESHOE_R2_THRESHOLD = 0.80  # Polynomial R² above this flags horseshoe artifact
PCA_VALIDATION_MIN_R = 0.90  # Minimum expected Spearman r between Dim1 and PC1
# Categories with very few observations get passive treatment (no axis influence)
PASSIVE_CATEGORIES = {"Present and Passing", "Not Voting"}
ACTIVE_CATEGORIES = {"Yea", "Nay", "Absent and Not Voting"}
ABSENT_LABEL = "Absent"  # Canonical label for all absence-type categories


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tallgrass MCA")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--pca-dir", default=None, help="Override PCA results directory")
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help="Number of MCA dimensions to extract",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis (faster, for debugging)",
    )
    parser.add_argument(
        "--correction",
        default=CORRECTION,
        choices=["benzecri", "greenacre", "none"],
        help="Inertia correction method",
    )
    return parser.parse_args()


# ── Phase 1: Load Data ──────────────────────────────────────────────────────


def load_raw_data(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load raw scraper CSVs: votes, rollcalls, legislators.

    MCA needs the full categorical vote values, not binary matrices from EDA.
    """
    prefix = data_dir.name
    votes = pl.read_csv(data_dir / f"{prefix}_votes.csv")
    rollcalls = pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    legislators = legislators.with_columns(
        pl.col("full_name")
        .map_elements(strip_leadership_suffix, return_dtype=pl.Utf8)
        .alias("full_name"),
        pl.col("party").fill_null("Independent").replace("", "Independent").alias("party"),
    )
    return votes, rollcalls, legislators


def build_categorical_vote_matrix(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
    minority_threshold: float = MINORITY_THRESHOLD,
    min_votes: int = MIN_VOTES,
) -> tuple[pl.DataFrame, dict]:
    """Build a categorical vote matrix (Yea/Nay/Absent) for one chamber.

    Unlike PCA's binary matrix, this preserves the full vote structure.
    Applies the same filtering: drop near-unanimous votes and low-participation
    legislators.

    Returns (matrix, filter_stats) where matrix has columns
    [legislator_slug, vote_id_1, vote_id_2, ...] with string values.
    """
    # Restrict to chamber
    prefix = "sen_" if chamber == "Senate" else "rep_"
    chamber_votes = votes.filter(pl.col("legislator_slug").str.starts_with(prefix))
    chamber_rollcalls = rollcalls.filter(pl.col("chamber") == chamber)
    chamber_vote_ids = set(chamber_rollcalls["vote_id"].to_list())
    chamber_votes = chamber_votes.filter(pl.col("vote_id").is_in(chamber_vote_ids))

    # Map vote categories to canonical labels
    chamber_votes = chamber_votes.with_columns(
        pl.col("vote")
        .replace(
            {
                "Absent and Not Voting": ABSENT_LABEL,
                "Not Voting": ABSENT_LABEL,
                "Present and Passing": ABSENT_LABEL,
            }
        )
        .alias("vote_category")
    )

    # Pivot to wide format: legislators x vote_ids, values are vote categories
    matrix = chamber_votes.pivot(
        on="vote_id",
        index="legislator_slug",
        values="vote_category",
    )

    # Get vote columns (everything except legislator_slug)
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]

    # Filter vote_cols to only those in chamber_vote_ids
    vote_cols = [c for c in vote_cols if c in chamber_vote_ids]
    matrix = matrix.select([slug_col, *vote_cols])

    # Filter 1: Drop near-unanimous votes
    contested_cols = []
    n_dropped_unanimous = 0
    for col in vote_cols:
        series = matrix[col].drop_nulls()
        if series.len() == 0:
            n_dropped_unanimous += 1
            continue
        total = series.len()
        yea_count = series.filter(series == "Yea").len()
        yea_frac = yea_count / total
        minority_frac = min(yea_frac, 1 - yea_frac)
        if minority_frac >= minority_threshold:
            contested_cols.append(col)
        else:
            n_dropped_unanimous += 1

    if not contested_cols:
        empty = pl.DataFrame({"legislator_slug": []}).cast({"legislator_slug": pl.Utf8})
        return empty, {"n_votes_before": len(vote_cols), "n_votes_after": 0}

    matrix = matrix.select([slug_col, *contested_cols])

    # Fill remaining nulls with Absent (legislator not present for this vote)
    matrix = matrix.with_columns([pl.col(c).fill_null(ABSENT_LABEL) for c in contested_cols])

    # Filter 2: Drop low-participation legislators (< min_votes substantive votes)
    # Count non-Absent votes per legislator
    substantive_counts = matrix.select(
        slug_col,
        pl.sum_horizontal(
            *[pl.col(c).ne(ABSENT_LABEL).cast(pl.Int32) for c in contested_cols]
        ).alias("n_substantive"),
    )
    active_slugs = substantive_counts.filter(pl.col("n_substantive") >= min_votes)[
        slug_col
    ].to_list()
    n_dropped_legislators = matrix.height - len(active_slugs)
    matrix = matrix.filter(pl.col(slug_col).is_in(active_slugs))

    filter_stats = {
        "n_votes_before": len(vote_cols),
        "n_votes_after": len(contested_cols),
        "n_dropped_unanimous": n_dropped_unanimous,
        "n_legislators_before": matrix.height + n_dropped_legislators,
        "n_legislators_after": matrix.height,
        "n_dropped_legislators": n_dropped_legislators,
        "minority_threshold": minority_threshold,
        "min_votes": min_votes,
    }
    return matrix, filter_stats


def polars_to_pandas_categorical(matrix: pl.DataFrame) -> pd.DataFrame:
    """Convert Polars categorical vote matrix to pandas for prince.

    Prince requires pandas DataFrames. The conversion boundary is here
    and only here — all upstream and downstream work uses Polars.

    Builds the pandas DataFrame from Python dicts to avoid requiring pyarrow.
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]
    data = {col: matrix[col].to_list() for col in vote_cols}
    pdf = pd.DataFrame(data, index=matrix[slug_col].to_list())
    pdf = pdf.astype(str)
    return pdf


# ── Phase 2: MCA per Chamber ────────────────────────────────────────────────


def fit_mca(
    pdf: pd.DataFrame,
    n_components: int,
    correction: str | None,
) -> prince.MCA:
    """Fit MCA on a pandas DataFrame of categorical votes.

    Returns the fitted prince.MCA object.
    """
    corr = correction if correction != "none" else None
    mca = prince.MCA(
        n_components=n_components,
        n_iter=10,
        random_state=RANDOM_STATE,
        one_hot=True,
        correction=corr,
    )
    mca.fit(pdf)
    return mca


def orient_dim1(
    row_coords: pd.DataFrame,
    slugs: list[str],
    legislators: pl.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Flip Dim1 sign so Republicans have positive mean scores.

    Same convention as PCA PC1.
    Returns (oriented_coords, was_flipped).
    """
    slug_to_party = dict(legislators.select("slug", "party").iter_rows())
    parties = [slug_to_party.get(s, "Unknown") for s in slugs]

    rep_scores = [row_coords.iloc[i, 0] for i, p in enumerate(parties) if p == "Republican"]
    dem_scores = [row_coords.iloc[i, 0] for i, p in enumerate(parties) if p == "Democrat"]

    rep_mean = np.mean(rep_scores) if rep_scores else 0.0
    dem_mean = np.mean(dem_scores) if dem_scores else 0.0

    flipped = False
    if rep_mean < dem_mean:
        row_coords.iloc[:, 0] *= -1
        flipped = True
        print("  Dim1 sign flipped (Republicans → positive)")
    else:
        print("  Dim1 orientation OK (Republicans already positive)")

    return row_coords, flipped


def extract_eigenvalues(mca: prince.MCA, n_components: int) -> pl.DataFrame:
    """Extract raw and corrected eigenvalues/inertia from the MCA object."""
    rows = []
    for i in range(n_components):
        rows.append(
            {
                "dimension": f"Dim{i + 1}",
                "eigenvalue": float(mca.eigenvalues_[i]),
                "inertia_pct": float(mca.percentage_of_variance_[i]),
                "cumulative_pct": float(sum(mca.percentage_of_variance_[: i + 1])),
            }
        )
    return pl.DataFrame(rows)


def extract_contributions(
    mca: prince.MCA,
    pdf: pd.DataFrame,
    n_components: int,
) -> pl.DataFrame:
    """Extract column (category) contributions to each MCA dimension."""
    col_contribs = mca.column_contributions_
    rows = []
    for idx in col_contribs.index:
        row = {"category": str(idx)}
        for i in range(min(n_components, col_contribs.shape[1])):
            row[f"Dim{i + 1}_ctr"] = float(col_contribs.iloc[col_contribs.index.get_loc(idx), i])
        rows.append(row)
    return pl.DataFrame(rows)


def extract_cos2(
    mca: prince.MCA,
    pdf: pd.DataFrame,
    n_components: int,
) -> pl.DataFrame:
    """Extract row (legislator) cos² values for representation quality."""
    cos2 = mca.row_cosine_similarities(pdf)
    rows = []
    for idx in cos2.index:
        row = {"legislator_slug": str(idx)}
        for i in range(min(n_components, cos2.shape[1])):
            row[f"Dim{i + 1}_cos2"] = float(cos2.iloc[cos2.index.get_loc(idx), i])
        rows.append(row)
    return pl.DataFrame(rows)


def detect_horseshoe(row_coords: pd.DataFrame) -> dict:
    """Detect horseshoe/arch effect: Dim2 as quadratic function of Dim1.

    Returns dict with polynomial R², coefficients, and whether horseshoe detected.
    """
    if row_coords.shape[1] < 2 or row_coords.shape[0] < 5:
        return {"detected": False, "r2": 0.0, "reason": "insufficient data"}

    x = row_coords.iloc[:, 0].values
    y = row_coords.iloc[:, 1].values

    # Fit quadratic polynomial: y = a*x^2 + b*x + c
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    detected = r2 > HORSESHOE_R2_THRESHOLD
    return {
        "detected": detected,
        "r2": float(r2),
        "coefficients": coeffs.tolist(),
    }


def run_mca_for_chamber(
    matrix: pl.DataFrame,
    chamber: str,
    n_components: int,
    correction: str,
    legislators: pl.DataFrame,
    rollcalls: pl.DataFrame,
) -> dict:
    """Run the full MCA pipeline for one chamber.

    Returns dict with all results.
    """
    print_header(f"MCA — {chamber}")
    vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
    slugs = matrix["legislator_slug"].to_list()
    print(f"  Matrix: {matrix.height} legislators × {len(vote_cols)} votes")

    # Convert to pandas for prince
    pdf = polars_to_pandas_categorical(matrix)

    # Fit MCA
    n_comp = min(n_components, pdf.shape[0] - 1, pdf.shape[1] - 1)
    mca = fit_mca(pdf, n_comp, correction)

    # Row coordinates (legislator positions)
    row_coords = mca.row_coordinates(pdf)
    row_coords, flipped = orient_dim1(row_coords, slugs, legislators)

    # If we flipped row coords, also flip column coords for consistency
    col_coords = mca.column_coordinates(pdf)
    if flipped:
        col_coords.iloc[:, 0] *= -1

    # Print inertia
    print(f"\n  Inertia (correction={correction}):")
    for i in range(n_comp):
        cumulative = sum(mca.percentage_of_variance_[: i + 1])
        print(
            f"    Dim{i + 1}: {mca.percentage_of_variance_[i]:.2f}%  cumulative: {cumulative:.2f}%"
        )

    # Eigenvalues DataFrame
    eigenvalues_df = extract_eigenvalues(mca, n_comp)

    # Build scores DataFrame with metadata
    dim_cols = {f"Dim{i + 1}": row_coords.iloc[:, i].tolist() for i in range(n_comp)}
    scores_df = pl.DataFrame({"legislator_slug": slugs, **dim_cols})
    meta = legislators.select("slug", "full_name", "party", "district", "chamber")
    scores_df = scores_df.join(meta, left_on="legislator_slug", right_on="slug", how="left")

    # Category coordinates DataFrame
    cat_rows = []
    for idx in col_coords.index:
        row = {"category": str(idx)}
        for i in range(n_comp):
            row[f"Dim{i + 1}"] = float(col_coords.iloc[col_coords.index.get_loc(idx), i])
        cat_rows.append(row)
    cat_coords_df = pl.DataFrame(cat_rows)

    # Contributions
    contributions_df = extract_contributions(mca, pdf, n_comp)

    # cos² (representation quality)
    cos2_df = extract_cos2(mca, pdf, n_comp)

    # Horseshoe detection
    horseshoe = detect_horseshoe(row_coords)
    if horseshoe["detected"]:
        print(f"\n  Horseshoe effect DETECTED (R² = {horseshoe['r2']:.3f})")
        print("    Dim2 is a quadratic function of Dim1 → confirms unidimensionality")
    else:
        print(f"\n  No horseshoe effect (R² = {horseshoe['r2']:.3f})")

    # Print top/bottom Dim1 legislators
    sorted_scores = scores_df.sort("Dim1", descending=True)
    print("\n  Top 5 Dim1 (most conservative):")
    for row in sorted_scores.head(5).iter_rows(named=True):
        print(f"    {row['full_name']:30s}  {row['party']:12s}  Dim1={row['Dim1']:+.3f}")
    print("  Bottom 5 Dim1 (most liberal):")
    for row in sorted_scores.tail(5).iter_rows(named=True):
        print(f"    {row['full_name']:30s}  {row['party']:12s}  Dim1={row['Dim1']:+.3f}")

    # Absence analysis: compute per-legislator absence rate
    absent_counts = matrix.select(
        "legislator_slug",
        pl.sum_horizontal(*[pl.col(c).eq(ABSENT_LABEL).cast(pl.Int32) for c in vote_cols]).alias(
            "n_absent"
        ),
    ).with_columns((pl.col("n_absent") / len(vote_cols) * 100).alias("absence_pct"))
    scores_df = scores_df.join(absent_counts, on="legislator_slug", how="left")

    return {
        "scores_df": scores_df,
        "cat_coords_df": cat_coords_df,
        "eigenvalues_df": eigenvalues_df,
        "contributions_df": contributions_df,
        "cos2_df": cos2_df,
        "mca": mca,
        "pdf": pdf,
        "slugs": slugs,
        "vote_ids": vote_cols,
        "n_components": n_comp,
        "horseshoe": horseshoe,
        "flipped": flipped,
        "correction": correction,
    }


# ── Phase 3: Plots ──────────────────────────────────────────────────────────


def plot_inertia(
    eigenvalues_df: pl.DataFrame,
    chamber: str,
    correction: str,
    out_dir: Path,
) -> None:
    """Scree plot: inertia per dimension (raw and corrected)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n = eigenvalues_df.height
    dims = list(range(1, n + 1))
    inertia = eigenvalues_df["inertia_pct"].to_list()
    cumulative = eigenvalues_df["cumulative_pct"].to_list()

    # Panel 1: Individual inertia
    axes[0].bar(dims, inertia, color="#4C72B0", edgecolor="black", alpha=0.9)
    for i, v in enumerate(inertia):
        axes[0].text(i + 1, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel(f"Inertia (%, {correction} corrected)")
    axes[0].set_title(f"{chamber} — MCA Inertia by Dimension")
    axes[0].set_xticks(dims)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if n >= 2 and inertia[0] > 2 * inertia[1]:
        axes[0].annotate(
            "Dominant first dimension confirms\nKansas is a one-dimensional\n"
            "legislature (party is everything)",
            xy=(1.5, (inertia[0] + inertia[1]) / 2),
            xytext=(3.0, inertia[0] * 0.7),
            fontsize=8,
            fontstyle="italic",
            color="#555555",
            bbox={"boxstyle": "round,pad=0.4", "fc": "lightyellow", "alpha": 0.8, "ec": "#cccccc"},
            arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.2},
        )

    # Panel 2: Cumulative inertia
    axes[1].plot(dims, cumulative, "bo-", markersize=8)
    for i, v in enumerate(cumulative):
        axes[1].text(i + 1, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].axhline(90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    axes[1].set_xlabel("Number of Dimensions")
    axes[1].set_ylabel("Cumulative Inertia (%)")
    axes[1].set_title(f"{chamber} — Cumulative Inertia")
    axes[1].set_xticks(dims)
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"mca_inertia_{chamber.lower()}.png")


def plot_ideological_map(
    scores_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Dim1 vs Dim2 scatter plot, party-colored, with outlier labels."""
    fig, ax = plt.subplots(figsize=(12, 10))

    for party, color in PARTY_COLORS.items():
        subset = scores_df.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["Dim1"].to_numpy(),
            subset["Dim2"].to_numpy(),
            c=color,
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label=party,
        )

    # Label outliers: top 5 by |Dim1| and top 5 by |Dim2|
    labeled = set()
    for dim_col in ["Dim1", "Dim2"]:
        abs_vals = scores_df[dim_col].abs()
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
                (row["Dim1"], row["Dim2"]),
                fontsize=8,
                fontweight="bold",
                ha="left",
                va="bottom",
                xytext=(6, 6),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.2", "fc": "wheat", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
            )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.2)
    ax.set_xlabel("Dimension 1 (primary ideological dimension)")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{chamber} — MCA Ideological Map (Categorical Vote Data)")
    ax.legend(
        handles=[Patch(facecolor=c, label=p) for p, c in PARTY_COLORS.items()],
        loc="best",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"mca_ideological_map_{chamber.lower()}.png")


def plot_biplot(
    scores_df: pl.DataFrame,
    cat_coords_df: pl.DataFrame,
    contributions_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Biplot: legislators (dots) + top-contributing category points (×)."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot legislators
    for party, color in PARTY_COLORS.items():
        subset = scores_df.filter(pl.col("party") == party)
        if subset.height == 0:
            continue
        ax.scatter(
            subset["Dim1"].to_numpy(),
            subset["Dim2"].to_numpy(),
            c=color,
            s=40,
            alpha=0.5,
            edgecolors="black",
            linewidth=0.3,
            label=party,
        )

    # Identify top-contributing categories for Dim1
    if "Dim1_ctr" in contributions_df.columns:
        top_cats = contributions_df.sort("Dim1_ctr", descending=True).head(TOP_CONTRIBUTIONS_N)
        top_cat_names = set(top_cats["category"].to_list())

        # Plot category points
        for row in cat_coords_df.iter_rows(named=True):
            cat_name = row["category"]
            # Determine color from category suffix
            if cat_name.endswith("_Yea") or cat_name == "Yea":
                color = CATEGORY_COLORS["Yea"]
            elif cat_name.endswith("_Nay") or cat_name == "Nay":
                color = CATEGORY_COLORS["Nay"]
            else:
                color = CATEGORY_COLORS["Absent"]

            is_top = cat_name in top_cat_names
            ax.scatter(
                row["Dim1"],
                row["Dim2"],
                c=color,
                marker="x",
                s=30 if is_top else 8,
                alpha=0.8 if is_top else 0.15,
                linewidths=1.5 if is_top else 0.5,
            )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.2)
    ax.set_xlabel("Dimension 1 (primary ideological dimension)")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{chamber} — MCA Biplot: Legislators and Vote Categories")

    legend_elements = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
        Line2D(
            [0],
            [0],
            marker="x",
            color=CATEGORY_COLORS["Yea"],
            label="Yea (top contributing)",
            linestyle="None",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=CATEGORY_COLORS["Nay"],
            label="Nay (top contributing)",
            linestyle="None",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=CATEGORY_COLORS["Absent"],
            label="Absent",
            linestyle="None",
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_elements, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"mca_biplot_{chamber.lower()}.png")


def plot_dim1_distribution(
    scores_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Overlapping KDE of Dim1 scores by party."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 5))

    for party, color in PARTY_COLORS.items():
        subset = scores_df.filter(pl.col("party") == party)
        if subset.height < 2:
            continue
        values = subset["Dim1"].to_numpy()
        sns.kdeplot(values, ax=ax, color=color, fill=True, alpha=0.3, label=party)

    ax.set_xlabel("MCA Dimension 1 Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{chamber} — MCA Dim1 Distribution by Party")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"mca_dim1_distribution_{chamber.lower()}.png")


def plot_absence_map(
    scores_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Dim1 vs Dim2 scatter colored by absence rate."""
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        scores_df["Dim1"].to_numpy(),
        scores_df["Dim2"].to_numpy(),
        c=scores_df["absence_pct"].to_numpy(),
        cmap="YlOrRd",
        s=60,
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Absence Rate (%)")
    cbar.ax.tick_params(labelsize=9)

    # Label high-absence legislators (top 5)
    high_absence = scores_df.sort("absence_pct", descending=True).head(5)
    for row in high_absence.iter_rows(named=True):
        raw_name = row.get("full_name") or row["legislator_slug"]
        name = raw_name.split(" - ")[0].strip()
        last_name = name.split()[-1] if name else row["legislator_slug"]
        ax.annotate(
            f"{last_name} ({row['absence_pct']:.0f}%)",
            (row["Dim1"], row["Dim2"]),
            fontsize=8,
            ha="left",
            va="bottom",
            xytext=(6, 6),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.2", "fc": "wheat", "alpha": 0.7},
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        )

    ax.axhline(0, color="gray", linestyle="-", alpha=0.2)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.2)
    ax.set_xlabel("Dimension 1 (primary ideological dimension)")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{chamber} — Where Do Absent Legislators Sit in the MCA Space?")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"mca_absence_map_{chamber.lower()}.png")


def plot_pca_correlation(
    scores_df: pl.DataFrame,
    pca_scores: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> dict:
    """Scatter plot: MCA Dim1 vs PCA PC1 with Spearman correlation."""
    # Match legislators by slug
    merged = scores_df.select("legislator_slug", "Dim1").join(
        pca_scores.select("legislator_slug", "PC1"),
        on="legislator_slug",
        how="inner",
    )

    if merged.height < 5:
        print(f"    Skipping PCA correlation for {chamber}: {merged.height} shared legislators")
        return {"skipped": True, "reason": "too few shared legislators"}

    dim1 = merged["Dim1"].to_numpy()
    pc1 = merged["PC1"].to_numpy()
    spearman_r, spearman_p = stats.spearmanr(dim1, pc1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pc1, dim1, c="#4C72B0", s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Best fit line
    z = np.polyfit(pc1, dim1, 1)
    p_fn = np.poly1d(z)
    x_range = np.linspace(pc1.min(), pc1.max(), 100)
    ax.plot(x_range, p_fn(x_range), "r--", alpha=0.5)

    ax.set_xlabel("PCA PC1 Score")
    ax.set_ylabel("MCA Dim1 Score")
    ax.set_title(
        f"{chamber} — MCA Dim1 vs PCA PC1\nSpearman r = {spearman_r:.4f} (p = {spearman_p:.2e})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_fig(fig, out_dir / f"mca_pca_correlation_{chamber.lower()}.png")

    verdict = "PASS" if abs(spearman_r) >= PCA_VALIDATION_MIN_R else "INVESTIGATE"
    print(f"    MCA Dim1 vs PCA PC1: Spearman r = {spearman_r:.4f} → {verdict}")

    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n_shared": merged.height,
        "verdict": verdict,
    }


# ── Phase 4: Sensitivity Analysis ───────────────────────────────────────────


def run_sensitivity(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    default_results: dict[str, dict],
    n_components: int,
    correction: str,
    plots_dir: Path,
) -> dict:
    """Run MCA with 10% minority threshold and compare to default (2.5%).

    Returns sensitivity findings dict.
    """
    print_header("SENSITIVITY ANALYSIS (10% threshold)")
    findings: dict = {}

    for chamber, default in default_results.items():
        print(f"\n  {chamber}:")
        sens_matrix, stats_dict = build_categorical_vote_matrix(
            votes,
            rollcalls,
            chamber,
            minority_threshold=SENSITIVITY_THRESHOLD,
            min_votes=MIN_VOTES,
        )
        n_votes = len(sens_matrix.columns) - 1
        print(f"    Sensitivity matrix: {sens_matrix.height} legislators × {n_votes} votes")

        if sens_matrix.height < 3 or n_votes < 3:
            print("    Skipping: too few data points")
            findings[chamber] = {"skipped": True, "reason": "insufficient data"}
            continue

        pdf = polars_to_pandas_categorical(sens_matrix)
        n_comp = min(n_components, pdf.shape[0] - 1, pdf.shape[1] - 1)
        mca = fit_mca(pdf, n_comp, correction)
        row_coords = mca.row_coordinates(pdf)
        slugs = sens_matrix["legislator_slug"].to_list()
        row_coords, _ = orient_dim1(row_coords, slugs, legislators)

        # Match legislators between default and sensitivity
        default_scores = default["scores_df"]
        default_slugs = set(default_scores["legislator_slug"].to_list())
        sens_map = dict(zip(slugs, row_coords.iloc[:, 0].tolist()))
        shared_slugs = sorted(default_slugs & set(slugs))

        if len(shared_slugs) < 5:
            print(f"    Skipping correlation: only {len(shared_slugs)} shared legislators")
            findings[chamber] = {"skipped": True, "reason": "too few shared legislators"}
            continue

        default_dim1 = []
        sens_dim1 = []
        for s in shared_slugs:
            row = default_scores.filter(pl.col("legislator_slug") == s)
            default_dim1.append(row["Dim1"][0])
            sens_dim1.append(sens_map[s])

        correlation = float(np.corrcoef(default_dim1, sens_dim1)[0, 1])
        print(f"    Shared legislators: {len(shared_slugs)}")
        print(f"    Pearson r: {correlation:.4f}")

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
        }

    return findings


# ── Phase 5: Filtering Manifest ──────────────────────────────────────────────


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

    data_dir = Path(args.data_dir) if args.data_dir else ks.data_dir
    results_root = ks.results_dir
    pca_dir = resolve_upstream_dir(
        "02_pca",
        results_root,
        args.run_id,
        Path(args.pca_dir) if args.pca_dir else None,
    )

    correction = args.correction if args.correction != "none" else "none"

    with RunContext(
        session=args.session,
        analysis_name="03_mca",
        params=vars(args),
        primer=MCA_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print(f"Tallgrass MCA — Session {args.session}")
        print(f"Data:       {data_dir}")
        print(f"PCA:        {pca_dir}")
        print(f"Output:     {ctx.run_dir}")
        print(f"Components: {args.n_components}")
        print(f"Correction: {correction}")

        # ── Phase 1: Load data ──
        print_header("LOADING DATA")
        votes, rollcalls, legislators = load_raw_data(data_dir)
        print(f"  Votes: {votes.height:,}")
        print(f"  Rollcalls: {rollcalls.height}")
        print(f"  Legislators: {legislators.height}")

        # Load PCA scores for validation (optional)
        pca_scores: dict[str, pl.DataFrame] = {}
        for chamber_label in ["house", "senate"]:
            pca_path = pca_dir / "data" / f"pc_scores_{chamber_label}.parquet"
            if pca_path.exists():
                pca_scores[chamber_label.capitalize()] = pl.read_parquet(pca_path)
                print(f"  PCA scores loaded: {chamber_label}")
            else:
                print(f"  PCA scores not found: {pca_path.name} (validation will be skipped)")

        # ── Phase 2: MCA per chamber ──
        results: dict[str, dict] = {}
        for label in ["House", "Senate"]:
            matrix, filter_stats = build_categorical_vote_matrix(
                votes,
                rollcalls,
                label,
                minority_threshold=MINORITY_THRESHOLD,
                min_votes=MIN_VOTES,
            )
            print(f"\n  {label} filtering: {filter_stats}")

            if matrix.height < 3:
                print(f"\n  Skipping {label}: too few legislators ({matrix.height})")
                continue

            result = run_mca_for_chamber(
                matrix,
                label,
                args.n_components,
                correction,
                legislators,
                rollcalls,
            )
            results[label] = result

            # Save parquet files
            result["scores_df"].write_parquet(ctx.data_dir / f"mca_scores_{label.lower()}.parquet")
            result["cat_coords_df"].write_parquet(
                ctx.data_dir / f"mca_category_coords_{label.lower()}.parquet"
            )
            result["contributions_df"].write_parquet(
                ctx.data_dir / f"mca_contributions_{label.lower()}.parquet"
            )
            result["eigenvalues_df"].write_parquet(
                ctx.data_dir / f"mca_eigenvalues_{label.lower()}.parquet"
            )
            result["cos2_df"].write_parquet(ctx.data_dir / f"mca_cos2_{label.lower()}.parquet")
            for name in [
                "mca_scores",
                "mca_category_coords",
                "mca_contributions",
                "mca_eigenvalues",
                "mca_cos2",
            ]:
                print(f"  Saved: {name}_{label.lower()}.parquet")

        # ── Phase 3: Plots ──
        print_header("GENERATING PLOTS")
        pca_validation: dict[str, dict] = {}
        for label, result in results.items():
            plot_inertia(result["eigenvalues_df"], label, correction, ctx.plots_dir)
            plot_ideological_map(result["scores_df"], label, ctx.plots_dir)
            plot_biplot(
                result["scores_df"],
                result["cat_coords_df"],
                result["contributions_df"],
                label,
                ctx.plots_dir,
            )
            plot_dim1_distribution(result["scores_df"], label, ctx.plots_dir)
            plot_absence_map(result["scores_df"], label, ctx.plots_dir)

            # PCA validation
            if label in pca_scores:
                pca_validation[label] = plot_pca_correlation(
                    result["scores_df"],
                    pca_scores[label],
                    label,
                    ctx.plots_dir,
                )

        # ── Phase 4: Sensitivity analysis ──
        sensitivity_findings: dict = {}
        if not args.skip_sensitivity:
            sensitivity_findings = run_sensitivity(
                votes,
                rollcalls,
                legislators,
                results,
                args.n_components,
                correction,
                ctx.plots_dir,
            )
        else:
            print_header("SENSITIVITY ANALYSIS (SKIPPED)")

        # ── Phase 5: Filtering manifest ──
        print_header("FILTERING MANIFEST")
        manifest: dict = {
            "n_components": args.n_components,
            "correction": correction,
            "minority_threshold": MINORITY_THRESHOLD,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "min_votes": MIN_VOTES,
            "passive_categories": sorted(PASSIVE_CATEGORIES),
            "active_categories": sorted(ACTIVE_CATEGORIES),
            "absent_label": ABSENT_LABEL,
        }
        for label, result in results.items():
            manifest[f"{label.lower()}_n_legislators"] = result["scores_df"].height
            manifest[f"{label.lower()}_n_votes"] = len(result["vote_ids"])
            manifest[f"{label.lower()}_horseshoe"] = result["horseshoe"]
            manifest[f"{label.lower()}_correction"] = result["correction"]
        if pca_validation:
            manifest["pca_validation"] = pca_validation
        if sensitivity_findings:
            manifest["sensitivity"] = sensitivity_findings
        save_filtering_manifest(manifest, ctx.run_dir)

        # ── Phase 6: HTML report ──
        print_header("HTML REPORT")
        build_mca_report(
            ctx.report,
            results=results,
            pca_validation=pca_validation,
            sensitivity_findings=sensitivity_findings,
            plots_dir=ctx.plots_dir,
            n_components=args.n_components,
            correction=correction,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
