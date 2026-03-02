"""
Kansas Legislature — Time Series Analysis (Phase 15)

Combines rolling-window PCA ideological drift detection with ruptures PELT
changepoint detection on Rice Index time series. Answers two questions:
  1. Did any legislator's voting behavior shift during the session?
  2. Were there structural breaks in party cohesion?

Usage:
  uv run python analysis/15_tsa/tsa.py [--session 2025-26] [--run-id ID]
      [--skip-drift] [--skip-changepoints] [--penalty 10.0] [--skip-r]

Outputs (in results/<session>/<run_id>/15_tsa/):
  - data/:   Parquet files (drift scores, Rice timeseries, changepoints)
  - plots/:  PNG visualizations (8 per chamber)
  - 15_tsa_report.html
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import ruptures as rpt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.run_context import RunContext, strip_leadership_suffix
except ModuleNotFoundError:
    from run_context import RunContext, strip_leadership_suffix

try:
    from analysis.tsa_report import build_tsa_report
except ModuleNotFoundError:
    from tsa_report import build_tsa_report  # type: ignore[no-redef]

try:
    from analysis.tsa_r_data import (
        find_crops_elbow,
        merge_bai_perron_with_pelt,
        parse_bai_perron_result,
        parse_crops_result,
        prepare_rice_signal_csv,
    )
except ModuleNotFoundError:
    from tsa_r_data import (  # type: ignore[no-redef]
        find_crops_elbow,
        merge_bai_perron_with_pelt,
        parse_bai_perron_result,
        parse_crops_result,
        prepare_rice_signal_csv,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

TSA_PRIMER = """\
# Time Series Analysis

## Purpose

Detects temporal patterns in legislative voting: ideological drift within a session
and structural breaks in party cohesion. Complements static analyses (PCA, IRT) with
a dynamic perspective.

## Method

Two independent analyses per chamber:

1. **Ideological Drift** — Rolling-window PCA on the vote matrix. Each window of 75
   consecutive roll calls produces a PC1 score per legislator. Tracking PC1 over time
   reveals who moved and when. Sign aligned so Republicans are positive.

2. **Changepoint Detection** — Per-vote Rice Index aggregated to weekly means, then
   PELT (Pruned Exact Linear Time) with RBF kernel detects structural breaks in party
   cohesion. Joint 2D detection finds breaks affecting both parties simultaneously.

## Inputs

- `{session}_votes.csv` — individual vote records
- `{session}_rollcalls.csv` — roll call metadata with timestamps
- `{session}_legislators.csv` — legislator metadata (party, district)

## Outputs

- Drift trajectories (party means, individual movers)
- Changepoint locations with dates and contextual annotations
- Penalty sensitivity analysis (robustness check)
- CROPS solution path (exact penalty thresholds, when R available)
- Bai-Perron 95% confidence intervals on break dates (when R available)

## Interpretation Guide

- **Party drift plot**: Diverging party means = increasing polarization.
  Converging = bipartisan periods.
- **Top movers**: Large |drift| = legislator changed position mid-session.
  Check bill context.
- **Changepoints**: Breaks in Rice = cohesion shift.
  Cross-reference with legislative calendar.
- **Penalty sensitivity**: Stable changepoints across penalties are robust;
  sensitive ones are artifacts.
- **CROPS solution path**: Step plot of changepoints vs penalty. The elbow marks
  the penalty where adding changepoints yields diminishing returns.
- **Bai-Perron CIs**: 95% confidence intervals on break dates. Narrow intervals
  = precisely dated breaks. Wide intervals = imprecise timing.

## Caveats

- Rolling PCA windows require sufficient data. Sessions with <150 roll calls per chamber
  will have very few windows, limiting drift detection power.
- PELT assumes i.i.d. within segments. Legislative voting has serial correlation, so
  changepoints should be interpreted as approximate boundaries, not exact transitions.
- Weekly Rice aggregation smooths daily variation. Individual high-drama days may be averaged away.
- CROPS and Bai-Perron require R with `changepoint`, `strucchange`, and `jsonlite` packages.
  When R is unavailable, the analysis runs Python-only PELT without these enrichments.
"""

# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_SIZE = 75
STEP_SIZE = 15
MIN_WINDOW_VOTES = 10
MIN_WINDOW_LEGISLATORS = 20
PELT_PENALTY_DEFAULT = 10.0
PELT_MIN_SIZE = 5
WEEKLY_AGG_DAYS = 7
SENSITIVITY_PENALTIES: list[float] = np.linspace(1, 50, 25).tolist()
TOP_MOVERS_N = 10
MIN_TOTAL_VOTES = 20

PARTY_COLORS = {
    "Republican": "#E81B23",
    "Democrat": "#0015BC",
    "Independent": "#999999",
}

# R enrichment constants
CROPS_PEN_MIN = 1.0
CROPS_PEN_MAX = 50.0
BAI_PERRON_MAX_BREAKS = 5
R_SUBPROCESS_TIMEOUT = 120

# ── Data Loading ─────────────────────────────────────────────────────────────


def load_data(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load votes, rollcalls, and legislators CSVs from the data directory."""
    csv_files = sorted(data_dir.glob("*_votes.csv"))
    if not csv_files:
        msg = f"No votes CSV found in {data_dir}"
        raise FileNotFoundError(msg)

    prefix = csv_files[0].stem.rsplit("_votes", 1)[0]
    votes = pl.read_csv(data_dir / f"{prefix}_votes.csv")
    rollcalls = pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    if "slug" in legislators.columns:
        legislators = legislators.rename({"slug": "legislator_slug"})

    # Fill Independent party
    if "party" in legislators.columns:
        legislators = legislators.with_columns(
            pl.col("party").fill_null("Independent").replace("", "Independent")
        )
    if "party" in votes.columns:
        votes = votes.with_columns(
            pl.col("party").fill_null("Independent").replace("", "Independent")
        )

    # Strip leadership suffixes
    if "full_name" in legislators.columns:
        legislators = legislators.with_columns(
            pl.col("full_name").map_elements(strip_leadership_suffix, return_dtype=pl.Utf8)
        )

    return votes, rollcalls, legislators


# ── Vote Matrix Construction ─────────────────────────────────────────────────


def build_vote_matrix(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Build a binary vote matrix (legislators x roll calls), chronologically ordered.

    Returns:
        matrix: (n_legislators, n_rollcalls) ndarray, 1=Yea, 0=Nay, NaN=absent
        slugs: legislator slugs (row labels)
        vote_ids: roll call vote IDs (column labels)
        datetimes: ISO datetime strings for each roll call
    """
    prefix = "sen_" if chamber == "Senate" else "rep_"
    chamber_votes = votes.filter(pl.col("legislator_slug").str.starts_with(prefix))

    # Filter to Yea/Nay only for binary encoding
    chamber_votes = chamber_votes.filter(pl.col("vote").is_in(["Yea", "Nay"]))
    chamber_votes = chamber_votes.with_columns(
        pl.when(pl.col("vote") == "Yea").then(1.0).otherwise(0.0).alias("vote_binary")
    )

    # Get chronological ordering from rollcalls
    rc_order = rollcalls.filter(
        pl.col("vote_id").is_in(chamber_votes["vote_id"].unique().to_list())
    )
    if "vote_datetime" in rc_order.columns:
        rc_order = rc_order.sort("vote_datetime")
    elif "vote_date" in rc_order.columns:
        rc_order = rc_order.sort("vote_date")

    ordered_vote_ids = rc_order["vote_id"].to_list()
    datetimes = []
    for col_name in ["vote_datetime", "vote_date"]:
        if col_name in rc_order.columns:
            datetimes = rc_order[col_name].cast(pl.Utf8).to_list()
            break
    if not datetimes:
        datetimes = [""] * len(ordered_vote_ids)

    # Filter legislators with minimum votes
    slug_counts = chamber_votes.group_by("legislator_slug").agg(pl.len().alias("n"))
    valid_slugs = slug_counts.filter(pl.col("n") >= MIN_TOTAL_VOTES)["legislator_slug"].to_list()
    valid_slugs.sort()

    # Pivot to matrix
    n_legs = len(valid_slugs)
    n_votes = len(ordered_vote_ids)
    matrix = np.full((n_legs, n_votes), np.nan)

    slug_idx = {s: i for i, s in enumerate(valid_slugs)}
    vid_idx = {v: i for i, v in enumerate(ordered_vote_ids)}

    for row in chamber_votes.filter(pl.col("legislator_slug").is_in(valid_slugs)).iter_rows(
        named=True
    ):
        si = slug_idx.get(row["legislator_slug"])
        vi = vid_idx.get(row["vote_id"])
        if si is not None and vi is not None:
            matrix[si, vi] = row["vote_binary"]

    return matrix, valid_slugs, ordered_vote_ids, datetimes


# ── Drift Analysis ───────────────────────────────────────────────────────────


def rolling_window_pca(
    matrix: np.ndarray,
    slugs: list[str],
    vote_ids: list[str],
    datetimes: list[str],
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    min_votes: int = MIN_WINDOW_VOTES,
    min_legislators: int = MIN_WINDOW_LEGISLATORS,
) -> pl.DataFrame:
    """Compute PC1 scores in rolling windows over the vote matrix.

    Returns a DataFrame with columns: slug, window_start, window_end,
    window_midpoint, window_idx, pc1_score.
    """
    n_legs, n_votes = matrix.shape

    if n_votes < window_size:
        warnings.warn(
            f"Too few roll calls ({n_votes}) for rolling PCA window size ({window_size}). "
            f"Drift detection requires at least {window_size} roll calls.",
            stacklevel=2,
        )

    rows: list[dict] = []
    window_idx = 0

    for start in range(0, n_votes - window_size + 1, step_size):
        end = start + window_size
        window = matrix[:, start:end]

        # Filter legislators with enough votes in this window
        valid_mask = np.sum(~np.isnan(window), axis=1) >= min_votes
        if np.sum(valid_mask) < min_legislators:
            continue

        sub_matrix = window[valid_mask]
        sub_slugs = [s for s, v in zip(slugs, valid_mask) if v]

        # Impute NaN with column mean for PCA
        col_means = np.nanmean(sub_matrix, axis=0)
        imputed = sub_matrix.copy()
        for j in range(imputed.shape[1]):
            nan_mask = np.isnan(imputed[:, j])
            imputed[nan_mask, j] = col_means[j]

        # Handle constant columns
        col_std = np.std(imputed, axis=0)
        vary_mask = col_std > 1e-10
        if np.sum(vary_mask) < 2:
            continue
        imputed = imputed[:, vary_mask]

        pca = PCA(n_components=1)
        scores = pca.fit_transform(imputed).ravel()

        mid = (start + end) // 2
        mid_date = datetimes[mid] if mid < len(datetimes) else ""
        start_date = datetimes[start] if start < len(datetimes) else ""
        end_date = datetimes[end - 1] if (end - 1) < len(datetimes) else ""

        for slug, score in zip(sub_slugs, scores):
            rows.append(
                {
                    "slug": slug,
                    "window_start": start_date,
                    "window_end": end_date,
                    "window_midpoint": mid_date,
                    "window_idx": window_idx,
                    "pc1_score": float(score),
                }
            )

        window_idx += 1

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "slug": pl.Utf8,
                "window_start": pl.Utf8,
                "window_end": pl.Utf8,
                "window_midpoint": pl.Utf8,
                "window_idx": pl.Int64,
                "pc1_score": pl.Float64,
            }
        )
    )


def align_pc_signs(
    rolling_df: pl.DataFrame,
    legislator_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Align PC1 signs so Republicans are positive (party convention).

    For each window, compute mean PC1 for Republicans. If negative, flip all scores.
    """
    if rolling_df.height == 0:
        return rolling_df

    # Build slug -> party mapping
    party_map = {}
    for row in legislator_meta.iter_rows(named=True):
        slug = row.get("legislator_slug", "")
        party = row.get("party", "")
        if slug and party:
            party_map[slug] = party

    # Process each window
    windows = rolling_df["window_idx"].unique().sort().to_list()
    result_rows: list[dict] = []

    for widx in windows:
        window_data = rolling_df.filter(pl.col("window_idx") == widx)
        scores = window_data["pc1_score"].to_numpy()
        slugs = window_data["slug"].to_list()

        # Compute Republican mean
        rep_scores = [s for slug, s in zip(slugs, scores) if party_map.get(slug) == "Republican"]
        flip = -1.0 if rep_scores and np.mean(rep_scores) < 0 else 1.0

        for row_dict in window_data.iter_rows(named=True):
            row_dict["pc1_score"] = row_dict["pc1_score"] * flip
            result_rows.append(row_dict)

    return pl.DataFrame(result_rows) if result_rows else rolling_df


def compute_party_trajectories(
    rolling_df: pl.DataFrame,
    legislator_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Compute party mean PC1 and polarization gap per window.

    Returns DataFrame with: window_idx, window_midpoint, party, mean_pc1,
    and a gap column (|Rep mean - Dem mean|).
    """
    if rolling_df.height == 0:
        return pl.DataFrame(
            schema={
                "window_idx": pl.Int64,
                "window_midpoint": pl.Utf8,
                "party": pl.Utf8,
                "mean_pc1": pl.Float64,
                "polarization_gap": pl.Float64,
            }
        )

    # Build slug -> party mapping
    party_map = {}
    for row in legislator_meta.iter_rows(named=True):
        slug = row.get("legislator_slug", "")
        party = row.get("party", "")
        if slug and party:
            party_map[slug] = party

    rolling_with_party = rolling_df.with_columns(
        pl.col("slug").replace_strict(party_map, default="Unknown").alias("party")
    )

    party_means = (
        rolling_with_party.filter(pl.col("party").is_in(["Republican", "Democrat"]))
        .group_by("window_idx", "window_midpoint", "party")
        .agg(pl.col("pc1_score").mean().alias("mean_pc1"))
        .sort("window_idx", "party")
    )

    # Compute polarization gap
    rep_means = party_means.filter(pl.col("party") == "Republican").select(
        "window_idx", pl.col("mean_pc1").alias("rep_mean")
    )
    dem_means = party_means.filter(pl.col("party") == "Democrat").select(
        "window_idx", pl.col("mean_pc1").alias("dem_mean")
    )

    gaps = rep_means.join(dem_means, on="window_idx", how="inner").with_columns(
        (pl.col("rep_mean") - pl.col("dem_mean")).abs().alias("polarization_gap")
    )

    party_means = party_means.join(
        gaps.select("window_idx", "polarization_gap"),
        on="window_idx",
        how="left",
    )

    return party_means


def compute_early_vs_late(
    matrix: np.ndarray,
    slugs: list[str],
    vote_ids: list[str],
    legislator_meta: pl.DataFrame,
) -> pl.DataFrame:
    """Compare PC1 scores from first half vs second half of session.

    Returns DataFrame with: slug, full_name, party, early_pc1, late_pc1, drift.
    """
    n_votes = matrix.shape[1]

    if n_votes < 2 * MIN_WINDOW_VOTES:
        warnings.warn(
            f"Too few roll calls ({n_votes}) for early/late comparison. "
            f"Need at least {2 * MIN_WINDOW_VOTES} for meaningful split.",
            stacklevel=2,
        )

    mid = n_votes // 2

    results = []
    for half_label, start, end in [("early", 0, mid), ("late", mid, n_votes)]:
        sub = matrix[:, start:end]
        valid_mask = np.sum(~np.isnan(sub), axis=1) >= MIN_WINDOW_VOTES
        if np.sum(valid_mask) < MIN_WINDOW_LEGISLATORS:
            continue

        sub_matrix = sub[valid_mask]
        sub_slugs = [s for s, v in zip(slugs, valid_mask) if v]

        # Impute
        col_means = np.nanmean(sub_matrix, axis=0)
        imputed = sub_matrix.copy()
        for j in range(imputed.shape[1]):
            nan_mask = np.isnan(imputed[:, j])
            imputed[nan_mask, j] = col_means[j]

        col_std = np.std(imputed, axis=0)
        vary_mask = col_std > 1e-10
        if np.sum(vary_mask) < 2:
            continue
        imputed = imputed[:, vary_mask]

        pca = PCA(n_components=1)
        scores = pca.fit_transform(imputed).ravel()

        # Align: Republicans positive
        party_map = {}
        for row in legislator_meta.iter_rows(named=True):
            party_map[row.get("legislator_slug", "")] = row.get("party", "")

        rep_scores = [
            s for slug, s in zip(sub_slugs, scores) if party_map.get(slug) == "Republican"
        ]
        if rep_scores and np.mean(rep_scores) < 0:
            scores = -scores

        results.append({half_label: dict(zip(sub_slugs, scores))})

    if len(results) != 2:
        return pl.DataFrame(
            schema={
                "slug": pl.Utf8,
                "full_name": pl.Utf8,
                "party": pl.Utf8,
                "early_pc1": pl.Float64,
                "late_pc1": pl.Float64,
                "drift": pl.Float64,
            }
        )

    early_scores = results[0]["early"]
    late_scores = results[1]["late"]
    common = sorted(set(early_scores) & set(late_scores))

    # Build name/party lookup
    meta_map: dict[str, dict[str, str]] = {}
    for row in legislator_meta.iter_rows(named=True):
        slug = row.get("legislator_slug", "")
        meta_map[slug] = {
            "full_name": row.get("full_name", slug),
            "party": row.get("party", ""),
        }

    rows = []
    for slug in common:
        e = early_scores[slug]
        la = late_scores[slug]
        info = meta_map.get(slug, {"full_name": slug, "party": ""})
        rows.append(
            {
                "slug": slug,
                "full_name": info["full_name"],
                "party": info["party"],
                "early_pc1": e,
                "late_pc1": la,
                "drift": la - e,
            }
        )

    return pl.DataFrame(rows)


def find_top_movers(drift_df: pl.DataFrame, n: int = TOP_MOVERS_N) -> pl.DataFrame:
    """Return the N legislators with largest |drift| between early and late session."""
    if drift_df.height == 0:
        return drift_df
    return (
        drift_df.with_columns(pl.col("drift").abs().alias("abs_drift"))
        .sort("abs_drift", descending=True)
        .head(n)
        .drop("abs_drift")
    )


def compute_imputation_sensitivity(
    matrix: np.ndarray,
    slugs: list[str],
    vote_ids: list[str],
    legislator_meta: pl.DataFrame,
) -> float | None:
    """Compare drift scores under column-mean vs listwise deletion imputation.

    Runs compute_early_vs_late() twice — once with the standard column-mean
    imputation (the default) and once with listwise deletion (rows with any NaN
    removed) — then correlates the resulting drift scores.

    Returns:
        Pearson correlation between drift scores under both methods, or None
        if insufficient data under either method.
    """
    # Column-mean is the default — just call as normal
    drift_mean = compute_early_vs_late(matrix, slugs, vote_ids, legislator_meta)
    if drift_mean.height < 3:
        return None

    # Listwise deletion: remove legislators with any NaN
    complete_mask = ~np.any(np.isnan(matrix), axis=1)
    if np.sum(complete_mask) < MIN_WINDOW_LEGISLATORS:
        return None

    listwise_matrix = matrix[complete_mask]
    listwise_slugs = [s for s, v in zip(slugs, complete_mask) if v]

    drift_listwise = compute_early_vs_late(
        listwise_matrix, listwise_slugs, vote_ids, legislator_meta
    )
    if drift_listwise.height < 3:
        return None

    # Join on slug and correlate
    merged = drift_mean.select(pl.col("slug"), pl.col("drift").alias("drift_mean")).join(
        drift_listwise.select(pl.col("slug"), pl.col("drift").alias("drift_listwise")),
        on="slug",
        how="inner",
    )

    if merged.height < 3:
        return None

    corr = np.corrcoef(
        merged["drift_mean"].to_numpy(),
        merged["drift_listwise"].to_numpy(),
    )[0, 1]

    return float(corr) if np.isfinite(corr) else None


# ── Changepoint Analysis ────────────────────────────────────────────────────


def desposato_corrected_rice(
    yea: int,
    nay: int,
    group_size: int,
    n_simulations: int = 10_000,
) -> float:
    """Compute Desposato-corrected Rice Index (Desposato 2005, BJPS).

    Subtracts the expected Rice under random voting for a group of this size,
    floored at 0.0. This removes the small-group inflation bias where smaller
    caucuses show higher Rice purely due to sampling variance.

    Args:
        yea: Number of Yea votes within the party group.
        nay: Number of Nay votes within the party group.
        group_size: Total party group size in the chamber (for simulation).
        n_simulations: Number of Monte Carlo draws (deterministic seed).

    Returns:
        Corrected Rice, floored at 0.0.
    """
    total = yea + nay
    if total == 0:
        return 0.0

    raw_rice = abs(yea - nay) / total

    # Simulate expected Rice under random (coin-flip) voting
    rng = np.random.default_rng(42)
    sim_yea = rng.binomial(group_size, 0.5, size=n_simulations)
    sim_nay = group_size - sim_yea
    sim_total = sim_yea + sim_nay  # always = group_size
    sim_rice = np.abs(sim_yea - sim_nay) / sim_total
    expected_rice = float(np.mean(sim_rice))

    return max(0.0, raw_rice - expected_rice)


def build_rice_timeseries(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
    correct_size_bias: bool = True,
) -> pl.DataFrame:
    """Compute per-vote Rice Index for each party.

    Args:
        correct_size_bias: If True, apply Desposato (2005) small-group correction
            to remove systematic Rice inflation for smaller caucuses.

    Returns DataFrame with: vote_id, vote_date, vote_datetime, party, rice,
    yea_count, nay_count, ordered chronologically.
    """
    prefix = "sen_" if chamber == "Senate" else "rep_"
    chamber_votes = votes.filter(
        pl.col("legislator_slug").str.starts_with(prefix) & pl.col("vote").is_in(["Yea", "Nay"])
    )

    # Attach party
    if "party" not in chamber_votes.columns:
        party_lookup = legislators.select("legislator_slug", "party")
        chamber_votes = chamber_votes.join(party_lookup, on="legislator_slug", how="left")

    # Compute party group sizes for Desposato correction
    party_group_sizes: dict[str, int] = {}
    if correct_size_bias:
        chamber_legs = legislators.filter(pl.col("legislator_slug").str.starts_with(prefix))
        for party in ["Republican", "Democrat"]:
            party_group_sizes[party] = chamber_legs.filter(pl.col("party") == party).height

    # Count Yea/Nay per vote per party
    party_counts = (
        chamber_votes.group_by("vote_id", "party", "vote")
        .agg(pl.len().alias("n"))
        .pivot(on="vote", index=["vote_id", "party"], values="n")
        .fill_null(0)
    )

    # Ensure columns exist
    if "Yea" not in party_counts.columns:
        party_counts = party_counts.with_columns(pl.lit(0).alias("Yea"))
    if "Nay" not in party_counts.columns:
        party_counts = party_counts.with_columns(pl.lit(0).alias("Nay"))

    party_counts = party_counts.rename({"Yea": "yea_count", "Nay": "nay_count"})

    # Compute Rice (raw formula first)
    party_counts = party_counts.with_columns(
        (
            (pl.col("yea_count") - pl.col("nay_count")).abs().cast(pl.Float64)
            / (pl.col("yea_count") + pl.col("nay_count")).cast(pl.Float64)
        ).alias("rice")
    )

    # Apply Desposato correction if requested
    if correct_size_bias and party_group_sizes:
        corrected_rows = []
        for row in party_counts.iter_rows(named=True):
            party = row["party"]
            group_size = party_group_sizes.get(party, 0)
            if group_size > 0:
                row["rice"] = desposato_corrected_rice(
                    row["yea_count"], row["nay_count"], group_size
                )
            corrected_rows.append(row)
        party_counts = pl.DataFrame(corrected_rows)

    # Join with rollcalls for dates
    date_cols = ["vote_id"]
    for col in ["vote_date", "vote_datetime"]:
        if col in rollcalls.columns:
            date_cols.append(col)

    rc_dates = rollcalls.select(date_cols).unique(subset=["vote_id"])
    rice_ts = party_counts.join(rc_dates, on="vote_id", how="left")

    # Sort chronologically
    sort_col = "vote_datetime" if "vote_datetime" in rice_ts.columns else "vote_date"
    if sort_col in rice_ts.columns:
        rice_ts = rice_ts.sort(sort_col, "party")

    # Filter to main parties
    rice_ts = rice_ts.filter(pl.col("party").is_in(["Republican", "Democrat"]))

    return rice_ts


def aggregate_weekly(rice_ts: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-vote Rice to weekly means.

    Returns DataFrame with: week_start, party, mean_rice, n_votes.
    """
    if rice_ts.height == 0:
        return pl.DataFrame(
            schema={
                "week_start": pl.Date,
                "party": pl.Utf8,
                "mean_rice": pl.Float64,
                "n_votes": pl.UInt32,
            }
        )

    # Parse date
    date_col = "vote_datetime" if "vote_datetime" in rice_ts.columns else "vote_date"
    if date_col not in rice_ts.columns:
        return pl.DataFrame(
            schema={
                "week_start": pl.Date,
                "party": pl.Utf8,
                "mean_rice": pl.Float64,
                "n_votes": pl.UInt32,
            }
        )

    ts = rice_ts.filter(pl.col(date_col).is_not_null()).with_columns(
        pl.col(date_col).cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d").alias("date")
    )

    weekly = (
        ts.group_by_dynamic("date", every=f"{WEEKLY_AGG_DAYS}d", group_by="party")
        .agg(
            pl.col("rice").mean().alias("mean_rice"),
            pl.len().alias("n_votes"),
        )
        .rename({"date": "week_start"})
        .sort("week_start", "party")
    )

    return weekly


def detect_changepoints_pelt(
    signal: np.ndarray,
    penalty: float = PELT_PENALTY_DEFAULT,
    min_size: int = PELT_MIN_SIZE,
) -> list[int]:
    """Detect changepoints in a 1D signal using PELT with RBF kernel.

    Returns list of changepoint indices (0-based, exclusive — matches ruptures convention).
    The last element is always len(signal).
    """
    if len(signal) < 2 * min_size:
        return [len(signal)]

    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
    result = algo.predict(pen=penalty)
    return result


def detect_changepoints_joint(
    rep_signal: np.ndarray,
    dem_signal: np.ndarray,
    penalty: float = PELT_PENALTY_DEFAULT,
    min_size: int = PELT_MIN_SIZE,
) -> list[int]:
    """Detect changepoints in a 2D signal (both parties jointly).

    Returns list of changepoint indices.
    """
    if len(rep_signal) != len(dem_signal):
        min_len = min(len(rep_signal), len(dem_signal))
        rep_signal = rep_signal[:min_len]
        dem_signal = dem_signal[:min_len]

    if len(rep_signal) < 2 * min_size:
        return [len(rep_signal)]

    signal_2d = np.column_stack([rep_signal, dem_signal])
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal_2d)
    result = algo.predict(pen=penalty)
    return result


def run_penalty_sensitivity(
    signal: np.ndarray,
    penalties: list[float] | None = None,
    min_size: int = PELT_MIN_SIZE,
) -> list[dict]:
    """Run PELT at multiple penalty values and return changepoint counts.

    Returns list of dicts: [{penalty, n_changepoints}, ...].
    """
    if penalties is None:
        penalties = SENSITIVITY_PENALTIES

    results = []
    for pen in penalties:
        cps = detect_changepoints_pelt(signal, penalty=pen, min_size=min_size)
        # Exclude the final point (always len(signal))
        n_cps = len(cps) - 1 if cps else 0
        results.append({"penalty": pen, "n_changepoints": max(0, n_cps)})

    return results


def cross_reference_veto_overrides(
    changepoint_dates: list[str],
    rollcalls: pl.DataFrame,
) -> pl.DataFrame:
    """Cross-reference changepoint dates with veto override votes.

    Returns a DataFrame with: changepoint_date, nearby_override_bill,
    nearby_override_date, days_apart.
    """
    overrides = rollcalls.filter(pl.col("motion").str.contains("(?i)override|veto"))

    if overrides.height == 0 or not changepoint_dates:
        return pl.DataFrame(
            schema={
                "changepoint_date": pl.Utf8,
                "nearby_override_bill": pl.Utf8,
                "nearby_override_date": pl.Utf8,
                "days_apart": pl.Int64,
            }
        )

    date_col = "vote_datetime" if "vote_datetime" in overrides.columns else "vote_date"
    if date_col not in overrides.columns:
        return pl.DataFrame(
            schema={
                "changepoint_date": pl.Utf8,
                "nearby_override_bill": pl.Utf8,
                "nearby_override_date": pl.Utf8,
                "days_apart": pl.Int64,
            }
        )

    rows = []
    for cp_date_str in changepoint_dates:
        try:
            from datetime import date

            cp_date = date.fromisoformat(cp_date_str[:10])
        except (ValueError, TypeError):  # fmt: skip
            continue

        for ov_row in overrides.iter_rows(named=True):
            ov_date_str = str(ov_row.get(date_col, ""))[:10]
            try:
                ov_date = date.fromisoformat(ov_date_str)
            except (ValueError, TypeError):  # fmt: skip
                continue

            days = abs((cp_date - ov_date).days)
            if days <= 14:
                rows.append(
                    {
                        "changepoint_date": cp_date_str[:10],
                        "nearby_override_bill": ov_row.get("bill_number", ""),
                        "nearby_override_date": ov_date_str,
                        "days_apart": days,
                    }
                )

    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "changepoint_date": pl.Utf8,
                "nearby_override_bill": pl.Utf8,
                "nearby_override_date": pl.Utf8,
                "days_apart": pl.Int64,
            }
        )
    )


# ── R Enrichment ────────────────────────────────────────────────────────────


def check_tsa_r_packages() -> bool:
    """Verify Rscript and required R packages (changepoint, strucchange, jsonlite)."""
    if shutil.which("Rscript") is None:
        return False

    pkgs = '"changepoint","strucchange","jsonlite"'
    check_script = f"cat(all(sapply(c({pkgs}), requireNamespace, quietly=TRUE)))"
    try:
        result = subprocess.run(
            ["Rscript", "-e", check_script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return "TRUE" in result.stdout
    except subprocess.TimeoutExpired, FileNotFoundError:
        return False


def run_r_tsa(
    weekly: pl.DataFrame,
    party: str,
    output_dir: Path,
    min_pen: float = CROPS_PEN_MIN,
    max_pen: float = CROPS_PEN_MAX,
    max_breaks: int = BAI_PERRON_MAX_BREAKS,
) -> bool:
    """Run CROPS + Bai-Perron via R subprocess for one party. Returns True on success."""
    r_script = Path(__file__).parent / "tsa_strucchange.R"
    if not r_script.exists():
        print(f"  ERROR: R script not found at {r_script}")
        return False

    signal_df = prepare_rice_signal_csv(weekly, party)
    if signal_df.height == 0:
        return False

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="tsa_signal_"
    ) as f:
        input_csv = Path(f.name)
        signal_df.write_csv(f.name)

    try:
        result = subprocess.run(
            [
                "Rscript",
                str(r_script),
                str(input_csv),
                str(output_dir),
                party,
                str(min_pen),
                str(max_pen),
                str(max_breaks),
            ],
            capture_output=True,
            text=True,
            timeout=R_SUBPROCESS_TIMEOUT,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  [R] {line}")

        if result.returncode != 0:
            print(f"  R enrichment failed for {party} (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:5]:
                    print(f"  [R stderr] {line}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"  ERROR: R subprocess timed out ({R_SUBPROCESS_TIMEOUT}s)")
        return False
    except FileNotFoundError:
        print("  ERROR: Rscript not found")
        return False
    finally:
        input_csv.unlink(missing_ok=True)


# ── Plotting ─────────────────────────────────────────────────────────────────


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def plot_party_drift(
    party_traj: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot party mean PC1 trajectories with shaded polarization gap."""
    if party_traj.height == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for party in ["Republican", "Democrat"]:
        pdata = party_traj.filter(pl.col("party") == party)
        if pdata.height == 0:
            continue
        ax.plot(
            pdata["window_idx"].to_numpy(),
            pdata["mean_pc1"].to_numpy(),
            color=PARTY_COLORS[party],
            linewidth=2,
            label=party,
        )

    # Shade polarization gap
    rep = party_traj.filter(pl.col("party") == "Republican").sort("window_idx")
    dem = party_traj.filter(pl.col("party") == "Democrat").sort("window_idx")
    if rep.height > 0 and dem.height > 0:
        common_idx = sorted(set(rep["window_idx"].to_list()) & set(dem["window_idx"].to_list()))
        if common_idx:
            rep_vals = (
                rep.filter(pl.col("window_idx").is_in(common_idx))
                .sort("window_idx")["mean_pc1"]
                .to_numpy()
            )
            dem_vals = (
                dem.filter(pl.col("window_idx").is_in(common_idx))
                .sort("window_idx")["mean_pc1"]
                .to_numpy()
            )
            ax.fill_between(common_idx, rep_vals, dem_vals, alpha=0.15, color="#888888")

    ax.set_xlabel("Window (chronological)")
    ax.set_ylabel("Mean PC1 Score")
    ax.set_title(f"{chamber} — Party Ideological Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"party_drift_{chamber.lower()}.png")


def plot_polarization_gap(
    party_traj: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot |Rep - Dem| polarization gap over time."""
    if party_traj.height == 0:
        return

    gaps = (
        party_traj.select("window_idx", "polarization_gap")
        .unique(subset=["window_idx"])
        .sort("window_idx")
        .drop_nulls()
    )

    if gaps.height == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        gaps["window_idx"].to_numpy(),
        gaps["polarization_gap"].to_numpy(),
        color="#333333",
        linewidth=2,
    )
    ax.fill_between(
        gaps["window_idx"].to_numpy(),
        gaps["polarization_gap"].to_numpy(),
        alpha=0.2,
        color="#333333",
    )
    ax.set_xlabel("Window (chronological)")
    ax.set_ylabel("|Republican Mean − Democrat Mean|")
    ax.set_title(f"{chamber} — Polarization Gap Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"polarization_gap_{chamber.lower()}.png")


def plot_top_movers(
    rolling_df: pl.DataFrame,
    top_movers_df: pl.DataFrame,
    legislator_meta: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot individual trajectories for the biggest drifters."""
    if top_movers_df.height == 0 or rolling_df.height == 0:
        return

    mover_slugs = top_movers_df["slug"].to_list()

    # Build name lookup
    name_map = {}
    party_map = {}
    for row in legislator_meta.iter_rows(named=True):
        slug = row.get("legislator_slug", "")
        name_map[slug] = row.get("full_name", slug)
        party_map[slug] = row.get("party", "")

    fig, ax = plt.subplots(figsize=(12, 6))

    for slug in mover_slugs:
        data = rolling_df.filter(pl.col("slug") == slug).sort("window_idx")
        if data.height == 0:
            continue
        name = name_map.get(slug, slug)
        party = party_map.get(slug, "")
        color = PARTY_COLORS.get(party, "#666666")
        ax.plot(
            data["window_idx"].to_numpy(),
            data["pc1_score"].to_numpy(),
            color=color,
            linewidth=1.5,
            alpha=0.8,
            label=name,
        )

    ax.set_xlabel("Window (chronological)")
    ax.set_ylabel("PC1 Score")
    ax.set_title(f"{chamber} — Top {len(mover_slugs)} Individual Drifters")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"top_movers_{chamber.lower()}.png")


def plot_early_vs_late(
    drift_df: pl.DataFrame,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Scatter plot: early PC1 vs late PC1 with identity line."""
    if drift_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    for party in ["Republican", "Democrat", "Independent"]:
        pdata = drift_df.filter(pl.col("party") == party)
        if pdata.height == 0:
            continue
        ax.scatter(
            pdata["early_pc1"].to_numpy(),
            pdata["late_pc1"].to_numpy(),
            color=PARTY_COLORS.get(party, "#666666"),
            alpha=0.6,
            s=40,
            label=party,
        )

    # Identity line
    all_vals = np.concatenate(
        [
            drift_df["early_pc1"].to_numpy(),
            drift_df["late_pc1"].to_numpy(),
        ]
    )
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.3, linewidth=1)

    # Label top movers
    top = find_top_movers(drift_df, 5)
    for row in top.iter_rows(named=True):
        ax.annotate(
            row["full_name"],
            (row["early_pc1"], row["late_pc1"]),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("Early Session PC1")
    ax.set_ylabel("Late Session PC1")
    ax.set_title(f"{chamber} — Early vs Late Ideological Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    save_fig(fig, plots_dir / f"early_vs_late_{chamber.lower()}.png")


def plot_changepoints(
    weekly: pl.DataFrame,
    changepoints: list[int],
    party: str,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot Rice timeseries with detected changepoints for one party."""
    pdata = weekly.filter(pl.col("party") == party).sort("week_start")
    if pdata.height == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    dates = pdata["week_start"].to_list()
    values = pdata["mean_rice"].to_numpy()

    ax.plot(dates, values, color=PARTY_COLORS.get(party, "#666666"), linewidth=1.5)
    ax.fill_between(dates, values, alpha=0.15, color=PARTY_COLORS.get(party, "#666666"))

    # Mark changepoints (exclude the last one which is len(signal))
    for cp in changepoints:
        if cp < len(dates):
            ax.axvline(dates[cp], color="red", linestyle="--", alpha=0.7, linewidth=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Mean Rice Index")
    ax.set_title(f"{chamber} {party} — Party Cohesion with Detected Breaks")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, plots_dir / f"changepoints_{party.lower()}_{chamber.lower()}.png")


def plot_changepoints_joint(
    weekly: pl.DataFrame,
    joint_cps: list[int],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot both parties' Rice with joint changepoints."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Use the common date axis from either party
    ref_party = weekly.filter(pl.col("party") == "Republican").sort("week_start")
    if ref_party.height == 0:
        ref_party = weekly.sort("week_start")
    dates = ref_party["week_start"].to_list()

    for party in ["Republican", "Democrat"]:
        pdata = weekly.filter(pl.col("party") == party).sort("week_start")
        if pdata.height == 0:
            continue
        ax.plot(
            pdata["week_start"].to_list(),
            pdata["mean_rice"].to_numpy(),
            color=PARTY_COLORS[party],
            linewidth=1.5,
            label=party,
        )

    for cp in joint_cps:
        if cp < len(dates):
            ax.axvline(dates[cp], color="red", linestyle="--", alpha=0.7, linewidth=1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Mean Rice Index")
    ax.set_title(f"{chamber} — Joint Changepoint Detection (Both Parties)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, plots_dir / f"changepoints_joint_{chamber.lower()}.png")


def plot_penalty_sensitivity(
    sensitivity: list[dict],
    chamber: str,
    plots_dir: Path,
) -> None:
    """Plot changepoint count vs penalty (elbow plot)."""
    if not sensitivity:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    penalties = [s["penalty"] for s in sensitivity]
    counts = [s["n_changepoints"] for s in sensitivity]

    ax.plot(penalties, counts, "o-", color="#333333", linewidth=2, markersize=8)
    ax.set_xlabel("PELT Penalty")
    ax.set_ylabel("Number of Changepoints")
    ax.set_title(f"{chamber} — Penalty Sensitivity (Elbow Plot)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"penalty_sensitivity_{chamber.lower()}.png")


def plot_crops_solution_path(
    crops_df: pl.DataFrame,
    elbow: float | None,
    party: str,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Step plot of CROPS solution path (changepoints vs penalty) with elbow marker."""
    if crops_df is None or crops_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    penalties = crops_df["penalty"].to_list()
    n_cps = crops_df["n_changepoints"].to_list()

    ax.step(penalties, n_cps, where="post", color=PARTY_COLORS.get(party, "#333333"), linewidth=2)
    ax.scatter(penalties, n_cps, color=PARTY_COLORS.get(party, "#333333"), s=30, zorder=5)

    if elbow is not None:
        # Find the n_changepoints at the elbow
        elbow_row = crops_df.filter(pl.col("penalty") == elbow)
        if elbow_row.height > 0:
            elbow_n = elbow_row["n_changepoints"][0]
            ax.axvline(elbow, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.scatter([elbow], [elbow_n], color="red", s=100, zorder=10, marker="D")
            ax.annotate(
                f"Elbow (pen={elbow:.1f})",
                xy=(elbow, elbow_n),
                xytext=(elbow + 2, elbow_n + 0.5),
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "red"},
                color="red",
            )

    ax.set_xlabel("Penalty")
    ax.set_ylabel("Number of Changepoints")
    ax.set_title(
        f"{chamber} {party} — CROPS Solution Path\n"
        "Each step = exact penalty threshold where optimal segmentation changes"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plots_dir / f"crops_{party.lower()}_{chamber.lower()}.png")


def plot_bai_perron_ci(
    bp_df: pl.DataFrame,
    weekly: pl.DataFrame,
    party: str,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Rice line + break lines + shaded CI bands from Bai-Perron."""
    if bp_df is None or bp_df.height == 0:
        return

    pdata = weekly.filter(pl.col("party") == party).sort("week_start")
    if pdata.height == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    dates = pdata["week_start"].to_list()
    values = pdata["mean_rice"].to_numpy()

    ax.plot(dates, values, color=PARTY_COLORS.get(party, "#666666"), linewidth=1.5)
    ax.fill_between(dates, values, alpha=0.1, color=PARTY_COLORS.get(party, "#666666"))

    for row in bp_df.iter_rows(named=True):
        bp_date = row["break_date"]
        ci_lo = row["ci_lower_date"]
        ci_hi = row["ci_upper_date"]

        # Break line
        ax.axvline(bp_date, color="red", linestyle="--", alpha=0.8, linewidth=1.5)

        # CI band
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="red")

    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Mean Rice Index")
    ax.set_title(
        f"{chamber} {party} — Bai-Perron Structural Breaks with 95% CIs\n"
        "Red dashed = break date, shaded = 95% confidence interval"
    )
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, plots_dir / f"bai_perron_{party.lower()}_{chamber.lower()}.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Time Series Analysis — ideological drift + changepoint detection"
    )
    parser.add_argument(
        "--session",
        default="2025-26",
        help="Session to analyze (e.g. 2025-26, 2023-24)",
    )
    parser.add_argument("--run-id", default=None, help="Run ID for pipeline grouping")
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="Skip ideological drift analysis",
    )
    parser.add_argument(
        "--skip-changepoints",
        action="store_true",
        help="Skip changepoint detection",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=PELT_PENALTY_DEFAULT,
        help=f"PELT penalty parameter (default: {PELT_PENALTY_DEFAULT})",
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Skip R enrichment (CROPS + Bai-Perron) even if R is available",
    )
    return parser.parse_args()


def main() -> None:
    """Run Time Series Analysis."""
    args = parse_args()

    # Resolve data directory
    from tallgrass.session import KSSession

    ks = KSSession.from_session_string(args.session)

    with RunContext(
        session=args.session,
        analysis_name="15_tsa",
        params=vars(args),
        primer=TSA_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        print_header("Time Series Analysis")
        print(f"Session: {ctx.session}")
        print(f"Run: {ctx.run_dir}")

        # Load data
        print("\n--- Loading data ---")
        votes, rollcalls, legislators = load_data(ks.data_dir)
        print(f"  Votes:      {votes.height:,} rows")
        print(f"  Roll calls: {rollcalls.height:,} rows")
        print(f"  Legislators: {legislators.height:,} rows")

        # R enrichment check
        r_available = not args.skip_r and check_tsa_r_packages()
        if args.skip_r:
            print("  R enrichment: skipped (--skip-r)")
        elif r_available:
            print("  R enrichment: available (CROPS + Bai-Perron)")
        else:
            print("  R enrichment: unavailable (install R + changepoint, strucchange, jsonlite)")

        results: dict[str, dict] = {}

        for chamber in ["House", "Senate"]:
            print_header(f"{chamber} Analysis")
            chamber_results: dict = {"chamber": chamber}

            # Build vote matrix
            matrix, slugs, vote_ids, datetimes = build_vote_matrix(votes, rollcalls, chamber)
            print(f"  Vote matrix: {matrix.shape[0]} legislators x {matrix.shape[1]} roll calls")
            chamber_results["n_legislators"] = matrix.shape[0]
            chamber_results["n_rollcalls"] = matrix.shape[1]

            # Get legislator metadata for this chamber
            prefix = "sen_" if chamber == "Senate" else "rep_"
            leg_meta = legislators.filter(pl.col("legislator_slug").str.starts_with(prefix))

            # --- Drift Analysis ---
            if not args.skip_drift:
                print("\n  --- Ideological Drift ---")

                rolling_df = rolling_window_pca(matrix, slugs, vote_ids, datetimes)
                n_windows = rolling_df["window_idx"].n_unique() if rolling_df.height > 0 else 0
                print(f"  Rolling PCA: {n_windows} windows")

                if rolling_df.height > 0:
                    rolling_df = align_pc_signs(rolling_df, leg_meta)
                    party_traj = compute_party_trajectories(rolling_df, leg_meta)
                    drift_df = compute_early_vs_late(matrix, slugs, vote_ids, leg_meta)
                    top_movers_df = find_top_movers(drift_df)

                    # Save data
                    ch_lower = chamber.lower()
                    rolling_df.write_parquet(ctx.data_dir / f"rolling_pca_{ch_lower}.parquet")
                    party_traj.write_parquet(
                        ctx.data_dir / f"party_trajectories_{ch_lower}.parquet"
                    )
                    drift_df.write_parquet(ctx.data_dir / f"drift_{ch_lower}.parquet")

                    # Print summary
                    if top_movers_df.height > 0:
                        print(f"\n  Top {top_movers_df.height} movers:")
                        for row in top_movers_df.iter_rows(named=True):
                            d = "→R" if row["drift"] > 0 else "→D"
                            name = row["full_name"]
                            drift = row["drift"]
                            print(f"    {name:30s} drift={drift:+.3f} {d}")

                    # Plots
                    plot_party_drift(party_traj, chamber, ctx.plots_dir)
                    plot_polarization_gap(party_traj, chamber, ctx.plots_dir)
                    plot_top_movers(rolling_df, top_movers_df, leg_meta, chamber, ctx.plots_dir)
                    plot_early_vs_late(drift_df, chamber, ctx.plots_dir)

                    # Imputation sensitivity
                    imp_corr = compute_imputation_sensitivity(matrix, slugs, vote_ids, leg_meta)
                    if imp_corr is not None:
                        print(f"  Imputation sensitivity: r={imp_corr:.3f}")
                    chamber_results["imputation_correlation"] = imp_corr

                    chamber_results["rolling_df"] = rolling_df
                    chamber_results["party_trajectories"] = party_traj
                    chamber_results["drift_df"] = drift_df
                    chamber_results["top_movers"] = top_movers_df
                    chamber_results["n_windows"] = n_windows
                else:
                    print("  Insufficient data for rolling PCA")

            # --- Changepoint Analysis ---
            if not args.skip_changepoints:
                print("\n  --- Changepoint Detection ---")

                rice_ts = build_rice_timeseries(votes, rollcalls, legislators, chamber)
                print(f"  Rice timeseries: {rice_ts.height} observations")

                weekly = aggregate_weekly(rice_ts)
                print(f"  Weekly aggregation: {weekly.height} weekly observations")

                if weekly.height > 0:
                    ch_l = chamber.lower()
                    rice_ts.write_parquet(ctx.data_dir / f"rice_ts_{ch_l}.parquet")
                    weekly.write_parquet(ctx.data_dir / f"rice_weekly_{ch_l}.parquet")
                    ctx.export_csv(
                        weekly,
                        f"rice_weekly_{ch_l}.csv",
                        f"Weekly Rice index time series for {ch_l.title()}",
                    )

                    # Per-party changepoints
                    cp_results = {}
                    for party in ["Republican", "Democrat"]:
                        pdata = weekly.filter(pl.col("party") == party).sort("week_start")
                        if pdata.height < 2 * PELT_MIN_SIZE:
                            warnings.warn(
                                f"{party}: too few weekly observations "
                                f"({pdata.height}) for changepoint detection "
                                f"(need {2 * PELT_MIN_SIZE})",
                                stacklevel=1,
                            )
                            continue

                        signal = pdata["mean_rice"].to_numpy()
                        cps = detect_changepoints_pelt(signal, penalty=args.penalty)
                        n_cps = len(cps) - 1  # exclude terminal
                        print(f"  {party}: {n_cps} changepoints at penalty={args.penalty}")

                        # Map changepoint indices to dates
                        dates = pdata["week_start"].cast(pl.Utf8).to_list()
                        cp_dates = [dates[i] for i in cps if i < len(dates)]

                        cp_results[party] = {
                            "changepoints": cps,
                            "n_changepoints": n_cps,
                            "cp_dates": cp_dates,
                        }

                        plot_changepoints(weekly, cps, party, chamber, ctx.plots_dir)

                    # Joint changepoints
                    rep_weekly = weekly.filter(pl.col("party") == "Republican").sort("week_start")
                    dem_weekly = weekly.filter(pl.col("party") == "Democrat").sort("week_start")

                    rep_ok = rep_weekly.height >= 2 * PELT_MIN_SIZE
                    dem_ok = dem_weekly.height >= 2 * PELT_MIN_SIZE
                    if rep_ok and dem_ok:
                        rep_signal = rep_weekly["mean_rice"].to_numpy()
                        dem_signal = dem_weekly["mean_rice"].to_numpy()
                        joint_cps = detect_changepoints_joint(
                            rep_signal, dem_signal, penalty=args.penalty
                        )
                        n_joint = len(joint_cps) - 1
                        print(f"  Joint: {n_joint} changepoints")

                        plot_changepoints_joint(weekly, joint_cps, chamber, ctx.plots_dir)
                        cp_results["joint"] = {
                            "changepoints": joint_cps,
                            "n_changepoints": n_joint,
                        }

                    # Penalty sensitivity (on Republican signal as reference)
                    if rep_weekly.height >= 2 * PELT_MIN_SIZE:
                        sensitivity = run_penalty_sensitivity(rep_weekly["mean_rice"].to_numpy())
                        plot_penalty_sensitivity(sensitivity, chamber, ctx.plots_dir)
                        cp_results["sensitivity"] = sensitivity

                    # R enrichment: CROPS + Bai-Perron (per-party)
                    if r_available:
                        print("\n  --- R Enrichment (CROPS + Bai-Perron) ---")
                        from datetime import date as _date

                        for party in ["Republican", "Democrat"]:
                            pdata = weekly.filter(pl.col("party") == party).sort("week_start")
                            if pdata.height < 2 * PELT_MIN_SIZE:
                                continue

                            success = run_r_tsa(weekly, party, ctx.data_dir)
                            if not success:
                                continue

                            # Parse CROPS result
                            crops_path = ctx.data_dir / f"crops_{party.lower()}.json"
                            if crops_path.exists():
                                with open(crops_path) as f:
                                    crops_json = json.load(f)
                                crops_df = parse_crops_result(crops_json)
                                if crops_df is not None:
                                    elbow = find_crops_elbow(crops_df)
                                    crops_df.write_parquet(
                                        ctx.data_dir
                                        / f"crops_{party.lower()}_{chamber.lower()}.parquet"
                                    )
                                    plot_crops_solution_path(
                                        crops_df, elbow, party, chamber, ctx.plots_dir
                                    )
                                    cp_results[f"{party}_crops"] = {
                                        "n_segmentations": crops_df.height,
                                        "elbow_penalty": elbow,
                                    }
                                    if elbow is not None:
                                        print(f"  {party} CROPS elbow: penalty={elbow:.1f}")

                            # Parse Bai-Perron result
                            bp_path = ctx.data_dir / f"bai_perron_{party.lower()}.json"
                            if bp_path.exists():
                                with open(bp_path) as f:
                                    bp_json = json.load(f)
                                weekly_dates = [
                                    _date.fromisoformat(str(d)[:10])
                                    for d in pdata["week_start"].to_list()
                                ]
                                bp_df = parse_bai_perron_result(bp_json, weekly_dates)
                                if bp_df is not None:
                                    bp_df.write_parquet(
                                        ctx.data_dir
                                        / f"bai_perron_{party.lower()}_{chamber.lower()}.parquet"
                                    )
                                    plot_bai_perron_ci(bp_df, weekly, party, chamber, ctx.plots_dir)
                                    n_bp = bp_df.height
                                    print(f"  {party} Bai-Perron: {n_bp} breaks with 95% CIs")
                                    cp_results[f"{party}_bai_perron"] = {
                                        "n_breaks": n_bp,
                                        "bp_df": bp_df,
                                    }

                                    # Merge with PELT
                                    pelt_dates = cp_results.get(party, {}).get("cp_dates", [])
                                    if pelt_dates:
                                        merged = merge_bai_perron_with_pelt(pelt_dates, bp_df)
                                        n_confirmed = merged.filter(pl.col("bp_confirmed")).height
                                        print(
                                            f"  {party} PELT/BP merge: "
                                            f"{n_confirmed}/{len(pelt_dates)} confirmed"
                                        )
                                        cp_results[f"{party}_merge"] = merged

                    # Veto override cross-reference
                    all_cp_dates = []
                    for party_key in ["Republican", "Democrat"]:
                        if party_key in cp_results:
                            all_cp_dates.extend(cp_results[party_key].get("cp_dates", []))

                    if all_cp_dates:
                        veto_xref = cross_reference_veto_overrides(all_cp_dates, rollcalls)
                        if veto_xref.height > 0:
                            print(f"  Veto override cross-references: {veto_xref.height}")
                            veto_xref.write_parquet(
                                ctx.data_dir / f"veto_crossref_{chamber.lower()}.parquet"
                            )
                            chamber_results["veto_crossref"] = veto_xref

                    chamber_results["changepoints"] = cp_results
                    chamber_results["weekly"] = weekly

            results[chamber] = chamber_results

        # Save filtering manifest
        manifest = {
            "constants": {
                "WINDOW_SIZE": WINDOW_SIZE,
                "STEP_SIZE": STEP_SIZE,
                "MIN_WINDOW_VOTES": MIN_WINDOW_VOTES,
                "MIN_WINDOW_LEGISLATORS": MIN_WINDOW_LEGISLATORS,
                "PELT_PENALTY": args.penalty,
                "PELT_MIN_SIZE": PELT_MIN_SIZE,
                "WEEKLY_AGG_DAYS": WEEKLY_AGG_DAYS,
                "MIN_TOTAL_VOTES": MIN_TOTAL_VOTES,
            },
            "chambers": {
                ch: {
                    "n_legislators": r.get("n_legislators", 0),
                    "n_rollcalls": r.get("n_rollcalls", 0),
                    "n_windows": r.get("n_windows", 0),
                    "imputation_correlation": r.get("imputation_correlation"),
                }
                for ch, r in results.items()
            },
        }
        if r_available:
            manifest["constants"]["CROPS_PEN_MIN"] = CROPS_PEN_MIN
            manifest["constants"]["CROPS_PEN_MAX"] = CROPS_PEN_MAX
            manifest["constants"]["BAI_PERRON_MAX_BREAKS"] = BAI_PERRON_MAX_BREAKS
        with open(ctx.run_dir / "filtering_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Build HTML report
        if ctx.report is not None:
            build_tsa_report(
                ctx.report,
                results=results,
                plots_dir=ctx.plots_dir,
                skip_drift=args.skip_drift,
                skip_changepoints=args.skip_changepoints,
                penalty=args.penalty,
                r_available=r_available,
            )

        print_header("Time Series Analysis Complete")


if __name__ == "__main__":
    main()
