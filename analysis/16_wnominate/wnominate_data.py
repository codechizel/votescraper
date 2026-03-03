"""Pure data logic for W-NOMINATE and Optimal Classification validation.

All functions are pure (no I/O, no subprocess calls). Vote matrix conversion,
result parsing, sign alignment, correlation computation, and comparison table
building live here so they can be tested with synthetic data.
"""

import numpy as np
import polars as pl
from scipy import stats as sp_stats

# ── Constants ────────────────────────────────────────────────────────────────

ROLLCALL_YEA = 1
ROLLCALL_NAY = 6
ROLLCALL_MISSING = 9

WNOMINATE_DIMS = 2
MIN_LEGISLATORS = 10
MIN_VOTES = 20
LOP_THRESHOLD = 0.025

STRONG_CORRELATION = 0.90
GOOD_CORRELATION = 0.85
CONCERN_CORRELATION = 0.70

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}


# ── Vote Matrix Conversion ──────────────────────────────────────────────────


def convert_vote_matrix_to_rollcall_csv(
    vote_matrix: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Convert EDA vote matrix (1/0/NaN) to pscl rollcall format (1/6/9).

    The EDA vote matrix has legislator_slug as the first column and vote_id
    columns with values 1.0 (Yea), 0.0 (Nay), or null (absent/missing).

    Returns (rollcall_df, legislator_slugs) where rollcall_df has legislators
    as rows and votes as columns, coded 1/6/9 for pscl::rollcall().
    """
    slug_col = vote_matrix.columns[0]
    vote_cols = vote_matrix.columns[1:]
    slugs = vote_matrix[slug_col].to_list()

    coded = vote_matrix.select(
        pl.col(slug_col),
        *[
            pl.when(pl.col(c) == 1.0)
            .then(pl.lit(ROLLCALL_YEA))
            .when(pl.col(c) == 0.0)
            .then(pl.lit(ROLLCALL_NAY))
            .otherwise(pl.lit(ROLLCALL_MISSING))
            .alias(c)
            for c in vote_cols
        ],
    )

    return coded, slugs


def select_polarity_legislator(
    pca_scores: pl.DataFrame,
    vote_matrix: pl.DataFrame,
    min_participation: float = 0.50,
) -> int:
    """Select the polarity legislator for W-NOMINATE identification.

    Picks the legislator with the highest PC1 score who has at least
    `min_participation` fraction of non-missing votes. Returns 1-based
    index into the vote matrix row order (R convention).

    Args:
        pca_scores: PCA output with legislator_slug and PC1 columns.
        vote_matrix: EDA vote matrix (legislator_slug as first col).
        min_participation: Minimum fraction of votes cast (default 0.50).

    Returns:
        1-based row index of the polarity legislator in vote_matrix.
    """
    slug_col = vote_matrix.columns[0]
    vote_cols = vote_matrix.columns[1:]
    n_votes = len(vote_cols)

    # Compute participation per legislator
    participation = vote_matrix.select(
        pl.col(slug_col),
        pl.concat_list([pl.col(c).is_not_null().cast(pl.Int64) for c in vote_cols])
        .list.sum()
        .truediv(n_votes)
        .alias("participation"),
    )

    # Join PCA scores
    pc1_col = "PC1" if "PC1" in pca_scores.columns else pca_scores.columns[1]
    merged = participation.join(
        pca_scores.select(pl.col(slug_col), pl.col(pc1_col).alias("_pc1")),
        on=slug_col,
        how="inner",
    )

    # Filter by participation
    eligible = merged.filter(pl.col("participation") >= min_participation)
    if eligible.height == 0:
        eligible = merged

    # Pick highest PC1
    best = eligible.sort("_pc1", descending=True).row(0, named=True)
    best_slug = best[slug_col]

    # Find 1-based index in vote matrix
    slugs = vote_matrix[slug_col].to_list()
    idx_0 = slugs.index(best_slug)
    return idx_0 + 1


# ── Result Parsing ───────────────────────────────────────────────────────────


def parse_wnominate_results(
    coords_df: pl.DataFrame,
    legislator_slugs: list[str],
) -> pl.DataFrame:
    """Parse W-NOMINATE coordinate CSV into a Polars DataFrame.

    The R script outputs a CSV with columns: coord1D, coord2D, se1, se2.
    Row order matches the input vote matrix (legislator_slugs).

    Returns DataFrame with: legislator_slug, wnom_dim1, wnom_dim2, wnom_se1, wnom_se2.
    """
    n_expected = len(legislator_slugs)
    n_actual = coords_df.height

    if n_actual != n_expected:
        # R may drop legislators with too few votes — map by index
        pass

    result = coords_df.with_columns(
        pl.Series("legislator_slug", legislator_slugs[:n_actual]),
    )

    # Rename to standard columns
    rename_map = {}
    for col in result.columns:
        if col.lower() in ("coord1d", "coord_1d"):
            rename_map[col] = "wnom_dim1"
        elif col.lower() in ("coord2d", "coord_2d"):
            rename_map[col] = "wnom_dim2"
        elif col.lower() == "se1":
            rename_map[col] = "wnom_se1"
        elif col.lower() == "se2":
            rename_map[col] = "wnom_se2"

    if rename_map:
        result = result.rename(rename_map)

    # Ensure expected columns exist
    for col in ["wnom_dim1", "wnom_dim2", "wnom_se1", "wnom_se2"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result.select("legislator_slug", "wnom_dim1", "wnom_dim2", "wnom_se1", "wnom_se2")


def parse_oc_results(
    coords_df: pl.DataFrame,
    legislator_slugs: list[str],
) -> pl.DataFrame:
    """Parse Optimal Classification coordinate CSV into a Polars DataFrame.

    The R script outputs: coord1D, coord2D, correctClassification.
    Row order matches the input vote matrix (legislator_slugs).

    Returns DataFrame with: legislator_slug, oc_dim1, oc_dim2, oc_correct_class.
    """
    n_actual = coords_df.height

    result = coords_df.with_columns(
        pl.Series("legislator_slug", legislator_slugs[:n_actual]),
    )

    rename_map = {}
    for col in result.columns:
        if col.lower() in ("coord1d", "coord_1d"):
            rename_map[col] = "oc_dim1"
        elif col.lower() in ("coord2d", "coord_2d"):
            rename_map[col] = "oc_dim2"
        elif col.lower() in ("correctclassification", "correct_classification"):
            rename_map[col] = "oc_correct_class"

    if rename_map:
        result = result.rename(rename_map)

    for col in ["oc_dim1", "oc_dim2", "oc_correct_class"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result.select("legislator_slug", "oc_dim1", "oc_dim2", "oc_correct_class")


def parse_fit_statistics(fit_json: dict) -> dict:
    """Parse fit statistics JSON from the R script.

    Expected keys per method: correctClassification, APRE, GMP.

    Returns dict with keys like wnominate_CC, wnominate_APRE, wnominate_GMP,
    oc_CC, oc_APRE, oc_GMP (if present).
    """
    result: dict = {}
    for method in ["wnominate", "oc"]:
        if method in fit_json:
            for stat in ["correctClassification", "APRE", "GMP"]:
                if stat in fit_json[method]:
                    result[f"{method}_{stat}"] = fit_json[method][stat]
    return result


def parse_eigenvalues(eigen_df: pl.DataFrame) -> pl.DataFrame:
    """Parse eigenvalue CSV from the R script.

    Returns DataFrame with columns: dimension, eigenvalue, pct_variance.
    """
    if "eigenvalue" not in eigen_df.columns and eigen_df.width >= 2:
        cols = eigen_df.columns
        eigen_df = eigen_df.rename({cols[0]: "dimension", cols[1]: "eigenvalue"})

    if "dimension" not in eigen_df.columns:
        eigen_df = eigen_df.with_row_index("dimension", offset=1)

    if "eigenvalue" in eigen_df.columns:
        total = eigen_df["eigenvalue"].sum()
        if total > 0:
            eigen_df = eigen_df.with_columns(
                (pl.col("eigenvalue") / total * 100).alias("pct_variance")
            )
        else:
            eigen_df = eigen_df.with_columns(pl.lit(0.0).alias("pct_variance"))

    return eigen_df


# ── Sign Alignment ──────────────────────────────────────────────────────────


def sign_align_scores(
    scores: pl.DataFrame,
    score_col: str,
    irt_df: pl.DataFrame,
    irt_col: str = "xi_mean",
    slug_col: str = "legislator_slug",
) -> pl.DataFrame:
    """Flip score_col sign if Pearson r with IRT is negative.

    Ensures that positive scores map to the same ideological direction as IRT.
    Returns the DataFrame with the column potentially sign-flipped.
    """
    merged = scores.join(
        irt_df.select(pl.col(slug_col), pl.col(irt_col)),
        on=slug_col,
        how="inner",
    )

    if merged.height < 3:
        return scores

    # Drop NaN pairs
    valid = merged.filter(pl.col(score_col).is_not_null() & pl.col(irt_col).is_not_null())
    if valid.height < 3:
        return scores

    x = valid[score_col].to_numpy().astype(float)
    y = valid[irt_col].to_numpy().astype(float)

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return scores

    r, _ = sp_stats.pearsonr(x, y)
    if r < 0:
        return scores.with_columns((-pl.col(score_col)).alias(score_col))
    return scores


# ── Correlations ─────────────────────────────────────────────────────────────


def compute_three_way_correlations(
    irt_df: pl.DataFrame,
    wnom_df: pl.DataFrame,
    oc_df: pl.DataFrame | None = None,
    irt_col: str = "xi_mean",
    wnom_col: str = "wnom_dim1",
    oc_col: str = "oc_dim1",
    slug_col: str = "legislator_slug",
) -> dict:
    """Compute 3x3 Pearson + Spearman correlation matrix (IRT/WNOM/OC).

    Returns dict with keys: irt_wnom, irt_oc, wnom_oc (each a sub-dict with
    pearson_r, pearson_p, spearman_rho, spearman_p, n). Also overall dict
    and within_party dict.
    """
    result: dict = {}

    # IRT vs WNOM
    merged_iw = irt_df.join(wnom_df.select(slug_col, wnom_col), on=slug_col, how="inner")
    result["irt_wnom"] = _pairwise_corr(merged_iw, irt_col, wnom_col)

    # IRT vs OC
    if oc_df is not None and oc_col in oc_df.columns:
        merged_io = irt_df.join(oc_df.select(slug_col, oc_col), on=slug_col, how="inner")
        result["irt_oc"] = _pairwise_corr(merged_io, irt_col, oc_col)
    else:
        result["irt_oc"] = _empty_corr()

    # WNOM vs OC
    if oc_df is not None and oc_col in oc_df.columns:
        merged_wo = wnom_df.join(oc_df.select(slug_col, oc_col), on=slug_col, how="inner")
        result["wnom_oc"] = _pairwise_corr(merged_wo, wnom_col, oc_col)
    else:
        result["wnom_oc"] = _empty_corr()

    return result


def compute_within_party_correlations(
    irt_df: pl.DataFrame,
    wnom_df: pl.DataFrame,
    oc_df: pl.DataFrame | None = None,
    irt_col: str = "xi_mean",
    wnom_col: str = "wnom_dim1",
    oc_col: str = "oc_dim1",
    slug_col: str = "legislator_slug",
) -> dict[str, dict]:
    """Compute within-party correlations (R and D separately).

    Returns dict: {"Republican": {...}, "Democrat": {...}}.
    """
    results: dict[str, dict] = {}

    # Merge all available scores
    merged = irt_df.join(wnom_df.select(slug_col, wnom_col), on=slug_col, how="inner")
    if oc_df is not None and oc_col in oc_df.columns:
        merged = merged.join(oc_df.select(slug_col, oc_col), on=slug_col, how="inner")

    for party in ["Republican", "Democrat"]:
        if "party" not in merged.columns:
            continue
        party_df = merged.filter(pl.col("party") == party)
        party_result: dict = {}

        party_result["irt_wnom"] = _pairwise_corr(party_df, irt_col, wnom_col)

        if oc_df is not None and oc_col in party_df.columns:
            party_result["irt_oc"] = _pairwise_corr(party_df, irt_col, oc_col)
            party_result["wnom_oc"] = _pairwise_corr(party_df, wnom_col, oc_col)
        else:
            party_result["irt_oc"] = _empty_corr()
            party_result["wnom_oc"] = _empty_corr()

        results[party] = party_result

    return results


def _pairwise_corr(df: pl.DataFrame, col_a: str, col_b: str) -> dict:
    """Compute Pearson + Spearman between two columns, handling NaN."""
    if col_a not in df.columns or col_b not in df.columns:
        return _empty_corr()

    valid = df.filter(pl.col(col_a).is_not_null() & pl.col(col_b).is_not_null())
    if valid.height < 3:
        return _empty_corr()

    a = valid[col_a].to_numpy().astype(float)
    b = valid[col_b].to_numpy().astype(float)

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if len(a) < 3:
        return _empty_corr()

    r, p_r = sp_stats.pearsonr(a, b)
    rho, p_rho = sp_stats.spearmanr(a, b)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "n": int(len(a)),
        "quality": _quality_label(abs(r)),
    }


def _empty_corr() -> dict:
    return {
        "pearson_r": float("nan"),
        "pearson_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "n": 0,
        "quality": "no_data",
    }


def _quality_label(abs_r: float) -> str:
    if abs_r >= STRONG_CORRELATION:
        return "strong"
    elif abs_r >= GOOD_CORRELATION:
        return "good"
    elif abs_r >= CONCERN_CORRELATION:
        return "moderate"
    else:
        return "concern"


# ── Comparison Table ─────────────────────────────────────────────────────────


def build_comparison_table(
    irt_df: pl.DataFrame,
    wnom_df: pl.DataFrame,
    oc_df: pl.DataFrame | None = None,
    irt_col: str = "xi_mean",
    wnom_col: str = "wnom_dim1",
    oc_col: str = "oc_dim1",
    slug_col: str = "legislator_slug",
) -> pl.DataFrame:
    """Build a joined comparison table with all three score sets + ranks.

    Columns: legislator_slug, full_name, party, irt_score, irt_rank,
    wnom_score, wnom_rank, oc_score, oc_rank, max_rank_diff.
    """
    # Start with IRT
    cols_to_keep = [slug_col, irt_col]
    if "full_name" in irt_df.columns:
        cols_to_keep.append("full_name")
    if "party" in irt_df.columns:
        cols_to_keep.append("party")

    table = irt_df.select([c for c in cols_to_keep if c in irt_df.columns]).rename(
        {irt_col: "irt_score"}
    )

    # Join WNOM
    table = table.join(
        wnom_df.select(slug_col, wnom_col).rename({wnom_col: "wnom_score"}),
        on=slug_col,
        how="inner",
    )

    # Join OC if available
    has_oc = oc_df is not None and oc_col in (oc_df.columns if oc_df is not None else [])
    if has_oc:
        table = table.join(
            oc_df.select(slug_col, oc_col).rename({oc_col: "oc_score"}),
            on=slug_col,
            how="left",
        )

    # Filter to valid rows (non-null IRT and WNOM)
    table = table.filter(pl.col("irt_score").is_not_null() & pl.col("wnom_score").is_not_null())

    # Add ranks (1 = most conservative / highest score)
    table = table.with_columns(
        pl.col("irt_score").rank(descending=True).cast(pl.Int64).alias("irt_rank"),
        pl.col("wnom_score").rank(descending=True).cast(pl.Int64).alias("wnom_rank"),
    )

    if has_oc and "oc_score" in table.columns:
        table = table.with_columns(
            pl.col("oc_score").rank(descending=True).cast(pl.Int64).alias("oc_rank"),
        )
        # Max rank difference
        table = table.with_columns(
            pl.max_horizontal(
                (pl.col("irt_rank") - pl.col("wnom_rank")).abs(),
                (pl.col("irt_rank") - pl.col("oc_rank")).abs(),
                (pl.col("wnom_rank") - pl.col("oc_rank")).abs(),
            ).alias("max_rank_diff")
        )
    else:
        table = table.with_columns(
            (pl.col("irt_rank") - pl.col("wnom_rank")).abs().alias("max_rank_diff")
        )

    return table.sort("irt_rank")
