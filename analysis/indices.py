"""
Kansas Legislature — Classical Indices Analysis (Phase 7)

Computes standard political science indices: Rice Index (party cohesion per vote),
Party Unity (CQ-standard per-legislator), Effective Number of Parties (Laakso-Taagepera),
and Maverick/Loyalty scores. Built for multi-session stacking from the start.

Usage:
  uv run python analysis/indices.py [--session 2025-26] [--skip-cross-ref]
      [--skip-sensitivity]

Outputs (in results/<session>/indices/<date>/):
  - data/:   Parquet files (party votes, Rice, unity, ENP, maverick, co-defection)
  - plots/:  PNG visualizations
  - filtering_manifest.json, run_info.json, run_log.txt
  - indices_report.html
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
from scipy import stats

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

try:
    from analysis.indices_report import build_indices_report
except ModuleNotFoundError:
    from indices_report import build_indices_report  # type: ignore[no-redef]

# ── Primer ───────────────────────────────────────────────────────────────────

INDICES_PRIMER = """\
# Classical Indices Analysis

## Purpose

Computes standard political science metrics that reviewers and journalists expect
when analyzing legislative voting data. These indices complement the model-based
approaches (IRT, clustering, network) with descriptive, assumption-light measures.

## Method

### Rice Index (per vote, per party)

Measures how unified a party is on a single roll call:

    Rice = |Yea - Nay| / (Yea + Nay)

Range: 0 (50-50 split) to 1 (unanimous). A Rice of 0.50 means a 75-25 split.
Computed on ALL roll calls (not just party votes).

### Party Unity (per legislator)

CQ-standard: fraction of "party votes" where a legislator votes with their
party's majority. A "party vote" is one where the majority of Republicans
oppose the majority of Democrats.

    Unity = (votes with party majority on party votes) / (party votes present)

### Effective Number of Parties (ENP)

Laakso-Taagepera index. Seat-based ENP is a static measure of party system
fragmentation. Vote-based ENP (per roll call) uses (party, direction) blocs
to capture when parties split internally.

    ENP = 1 / sum(p_i^2)

### Maverick Scores

Unweighted: 1 - unity (fraction of defections on party votes).
Weighted: defections weighted by chamber vote closeness (close votes matter more).

## Inputs

Reads from `data/{legislature}_{start}-{end}/`:
- `{output_name}_votes.csv` — Individual vote records (~68K rows)
- `{output_name}_rollcalls.csv` — Roll call metadata (~882 rows)
- `{output_name}_legislators.csv` — Legislator metadata (~172 rows)

Optionally reads from upstream results:
- `results/<session>/irt/latest/data/ideal_points_{chamber}.parquet`
- `results/<session>/network/latest/data/centrality_{chamber}.parquet`
- `results/<session>/clustering/latest/data/party_loyalty_{chamber}.parquet`

## Outputs

All outputs land in `results/<session>/indices/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `party_votes_{chamber}.parquet` | Per-vote: is_party_vote, majority positions, margins |
| `rice_index_{chamber}.parquet` | Per-vote per-party Rice index values |
| `fractured_votes_{chamber}.parquet` | Votes where majority party Rice < 0.50 |
| `party_unity_{chamber}.parquet` | Per-legislator CQ-standard unity scores |
| `enp_{chamber}.parquet` | Per-vote ENP from (party, direction) blocs |
| `maverick_scores_{chamber}.parquet` | Per-legislator maverick scores (weighted + unweighted) |
| `co_defection_{chamber}.parquet` | Pairwise co-defection counts among top defectors |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `rice_distribution_{chamber}.png` | "How Often Does Each Party Vote Together?" |
| `rice_over_time_{chamber}.png` | "Party Cohesion Through the Session" |
| `rice_by_vote_type_{chamber}.png` | Rice broken down by motion type |
| `party_unity_ranking_{chamber}.png` | "Who Are the Most Independent?" — all legislators |
| `enp_distribution_{chamber}.png` | "When Does Kansas Act Like More Than Two Parties?" |
| `enp_over_time_{chamber}.png` | ENP rolling average through the session |
| `maverick_landscape_{chamber}.png` | Unweighted vs weighted maverick scatter |
| `co_defection_heatmap_{chamber}.png` | "Who Breaks Ranks Together?" |
| `unity_vs_irt_{chamber}.png` | Party unity vs IRT ideal point scatter |

## Interpretation Guide

- **Rice = 1.0**: The party voted unanimously. Rice = 0.0: perfect 50-50 split.
- **Unity = 1.0**: Legislator always voted with their party on contested votes.
  Unity = 0.5: voted with party only half the time on contested votes.
- **ENP = 1.0**: One bloc controls the vote. ENP = 2.0: two equal-sized blocs.
  ENP > 2.5: three or more meaningful voting blocs.
- **Maverick (weighted) > maverick (unweighted)**: This legislator's defections
  tend to happen on close votes (more consequential).
- **Co-defection heatmap**: High counts indicate legislators who break ranks
  together — possible informal caucus or issue-specific alliance.

## Caveats

- Party unity is only defined on "party votes" (majority vs majority). A
  legislator who is always absent on party votes has an undefined unity score.
- Rice Index treats all votes equally. Some votes (final passage) are more
  substantively important than others (procedural motions).
- ENP blocs are (party, direction) pairs. With only 2 parties, the maximum
  ENP is 4 (both parties split 50-50), which is unrealistic in practice.
- Weighted maverick assigns more weight to close votes, which are often
  procedural. The weighting captures strategic importance, not necessarily
  policy importance.
"""

# ── Constants ────────────────────────────────────────────────────────────────

PARTY_VOTE_THRESHOLD = 0.50
RICE_FRACTURE_THRESHOLD = 0.50
MIN_PARTY_VOTERS = 2
MAVERICK_WEIGHT_FLOOR = 0.01
CO_DEFECTION_MIN = 3
ENP_MULTIPARTY_THRESHOLD = 2.5
ROLLING_WINDOW = 15
TOP_DEFECTORS_N = 20
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature Classical Indices Analysis")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    parser.add_argument("--irt-dir", default=None, help="Override IRT results directory")
    parser.add_argument("--network-dir", default=None, help="Override network results directory")
    parser.add_argument("--clustering-dir", default=None, help="Override clustering results dir")
    parser.add_argument("--eda-dir", default=None, help="Override EDA results directory")
    parser.add_argument(
        "--skip-cross-ref",
        action="store_true",
        help="Skip cross-referencing with IRT/network/clustering",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip sensitivity analysis on EDA-filtered data",
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


def _resolve_results_name(session: str) -> str:
    """Convert '2025-26' to biennium results directory name (e.g. '91st_2025-2026')."""
    from ks_vote_scraper.session import KSSession

    return KSSession.from_session_string(session).output_name


def _rice_from_counts(party_counts: pl.DataFrame) -> pl.DataFrame:
    """Add rice_index column to a party_counts DataFrame.

    Shared by the main Rice computation and veto override subgroup analysis.
    """
    return party_counts.with_columns(
        pl.when(pl.col("total_voters") >= MIN_PARTY_VOTERS)
        .then(
            (pl.col("yea_count").cast(pl.Int64) - pl.col("nay_count").cast(pl.Int64)).abs()
            / pl.col("total_voters").cast(pl.Float64)
        )
        .otherwise(pl.lit(None))
        .alias("rice_index"),
    )


# ── Phase 1: Load Data ───────────────────────────────────────────────────────


def load_raw_data(
    data_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load raw CSVs: votes, rollcalls, legislators."""
    prefix = data_dir.name
    votes = pl.read_csv(data_dir / f"{prefix}_votes.csv")
    rollcalls = pl.read_csv(data_dir / f"{prefix}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"{prefix}_legislators.csv")
    return votes, rollcalls, legislators


def load_upstream_optional(
    irt_dir: Path | None,
    network_dir: Path | None,
    clustering_dir: Path | None,
    chamber: str,
) -> dict[str, pl.DataFrame | None]:
    """Load upstream results for cross-referencing. Returns None for missing files."""
    result: dict[str, pl.DataFrame | None] = {
        "irt": None,
        "centrality": None,
        "clustering_loyalty": None,
    }
    ch = chamber.lower()
    if irt_dir:
        path = irt_dir / "data" / f"ideal_points_{ch}.parquet"
        if path.exists():
            result["irt"] = pl.read_parquet(path)
    if network_dir:
        path = network_dir / "data" / f"centrality_{ch}.parquet"
        if path.exists():
            result["centrality"] = pl.read_parquet(path)
    if clustering_dir:
        path = clustering_dir / "data" / f"party_loyalty_{ch}.parquet"
        if path.exists():
            result["clustering_loyalty"] = pl.read_parquet(path)
    return result


# ── Phase 2: Party Vote Identification ───────────────────────────────────────


def compute_party_majority_positions(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
) -> pl.DataFrame:
    """Compute per-vote per-party Yea/Nay counts and majority position.

    Returns DataFrame with columns: vote_id, party, yea_count, nay_count,
    total_voters, majority_position, yea_pct.
    """
    # Filter to chamber
    chamber_vote_ids = set(
        rollcalls.filter(pl.col("chamber") == chamber)["vote_id"].to_list()
    )

    # Join votes with legislator party
    vote_party = votes.join(
        legislators.select(pl.col("slug").alias("legislator_slug"), "party"),
        on="legislator_slug",
        how="left",
    ).filter(
        pl.col("vote_id").is_in(chamber_vote_ids)
        & pl.col("vote").is_in(["Yea", "Nay"])
    )

    # Group by vote_id and party
    party_counts = (
        vote_party.group_by("vote_id", "party")
        .agg(
            (pl.col("vote") == "Yea").sum().alias("yea_count"),
            (pl.col("vote") == "Nay").sum().alias("nay_count"),
        )
        .with_columns(
            (pl.col("yea_count") + pl.col("nay_count")).alias("total_voters"),
        )
        .with_columns(
            (pl.col("yea_count") / pl.col("total_voters")).alias("yea_pct"),
            pl.when(pl.col("yea_count") > pl.col("nay_count"))
            .then(pl.lit("Yea"))
            .otherwise(pl.lit("Nay"))
            .alias("majority_position"),
        )
    )

    return party_counts


def identify_party_votes(
    party_counts: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
    session: str,
) -> pl.DataFrame:
    """Identify party votes and compute margins from pre-computed party counts.

    A roll call is a "party vote" iff majority of Rs oppose majority of Ds,
    with each party having at least MIN_PARTY_VOTERS Yea+Nay voters.

    Returns DataFrame with one row per vote_id: is_party_vote, R/D majority
    positions, margins, closeness weight.
    """
    # Pivot to get R and D side-by-side per vote
    r_counts = (
        party_counts.filter(pl.col("party") == "Republican")
        .select(
            "vote_id",
            pl.col("yea_count").alias("r_yea"),
            pl.col("nay_count").alias("r_nay"),
            pl.col("total_voters").alias("r_total"),
            pl.col("majority_position").alias("r_majority"),
            pl.col("yea_pct").alias("r_yea_pct"),
        )
    )
    d_counts = (
        party_counts.filter(pl.col("party") == "Democrat")
        .select(
            "vote_id",
            pl.col("yea_count").alias("d_yea"),
            pl.col("nay_count").alias("d_nay"),
            pl.col("total_voters").alias("d_total"),
            pl.col("majority_position").alias("d_majority"),
            pl.col("yea_pct").alias("d_yea_pct"),
        )
    )

    combined = r_counts.join(d_counts, on="vote_id", how="full", coalesce=True)

    # A party vote requires: both parties have >= MIN_PARTY_VOTERS and majorities oppose
    combined = combined.with_columns(
        (
            (pl.col("r_total") >= MIN_PARTY_VOTERS)
            & (pl.col("d_total") >= MIN_PARTY_VOTERS)
            & (pl.col("r_majority") != pl.col("d_majority"))
        ).alias("is_party_vote"),
    )

    # Chamber-level margin and closeness weight
    combined = combined.with_columns(
        (pl.col("r_yea").fill_null(0) + pl.col("d_yea").fill_null(0)).alias("total_yea"),
        (pl.col("r_nay").fill_null(0) + pl.col("d_nay").fill_null(0)).alias("total_nay"),
    ).with_columns(
        (pl.col("total_yea") + pl.col("total_nay")).alias("total_voting"),
    ).with_columns(
        (
            (pl.col("total_yea").cast(pl.Int64) - pl.col("total_nay").cast(pl.Int64)).abs()
            / pl.col("total_voting").cast(pl.Float64)
        ).alias("chamber_margin"),
    ).with_columns(
        (
            1.0 / pl.max_horizontal(pl.col("chamber_margin"), pl.lit(MAVERICK_WEIGHT_FLOOR))
        ).alias("closeness_weight"),
        pl.lit(session).alias("session"),
    )

    # Join vote type and date from rollcalls
    chamber_rollcalls = rollcalls.filter(pl.col("chamber") == chamber).select(
        "vote_id", "vote_date", "motion", "bill_number"
    )
    combined = combined.join(chamber_rollcalls, on="vote_id", how="left")

    return combined


# ── Phase 3: Rice Index ──────────────────────────────────────────────────────


def compute_rice_index(
    party_counts: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str,
    session: str,
) -> pl.DataFrame:
    """Compute Rice index per vote per party from pre-computed party counts.

    Rice = |Yea - Nay| / (Yea + Nay). Null if fewer than MIN_PARTY_VOTERS.
    """
    rice = _rice_from_counts(party_counts).with_columns(
        pl.lit(session).alias("session"),
    )

    # Join vote metadata
    chamber_rollcalls = rollcalls.filter(pl.col("chamber") == chamber).select(
        "vote_id", "vote_date", "motion", "bill_number"
    )
    rice = rice.join(chamber_rollcalls, on="vote_id", how="left")

    return rice


def compute_rice_summary(rice_df: pl.DataFrame, chamber: str) -> dict[str, dict]:
    """Compute summary statistics per party."""
    summary: dict[str, dict] = {}
    for party in ["Republican", "Democrat"]:
        party_rice = rice_df.filter(
            (pl.col("party") == party) & pl.col("rice_index").is_not_null()
        )
        if party_rice.height == 0:
            continue
        vals = party_rice["rice_index"]
        summary[party] = {
            "n_votes": party_rice.height,
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "pct_perfect_unity": float(
                party_rice.filter(pl.col("rice_index") == 1.0).height / party_rice.height * 100
            ),
            "pct_fractured": float(
                party_rice.filter(pl.col("rice_index") < RICE_FRACTURE_THRESHOLD).height
                / party_rice.height
                * 100
            ),
        }
        print(
            f"  {chamber} {party}: mean Rice={summary[party]['mean']:.3f}, "
            f"median={summary[party]['median']:.3f}, "
            f"perfect unity={summary[party]['pct_perfect_unity']:.1f}%, "
            f"fractured={summary[party]['pct_fractured']:.1f}%"
        )
    return summary


def compute_rice_by_vote_type(rice_df: pl.DataFrame) -> pl.DataFrame:
    """Compute mean Rice by motion type and party."""
    return (
        rice_df.filter(pl.col("rice_index").is_not_null())
        .group_by("motion", "party")
        .agg(
            pl.col("rice_index").mean().alias("mean_rice"),
            pl.col("rice_index").median().alias("median_rice"),
            pl.col("rice_index").std().alias("std_rice"),
            pl.col("rice_index").len().alias("n_votes"),
        )
        .sort("party", "mean_rice")
    )


def find_fractured_votes(
    rice_df: pl.DataFrame,
    chamber: str,
    session: str,
) -> pl.DataFrame:
    """Find bills where the majority party has Rice < RICE_FRACTURE_THRESHOLD."""
    # Determine majority party by seat count
    party_sizes = (
        rice_df.filter(pl.col("rice_index").is_not_null())
        .select("vote_id", "party", "total_voters")
        .unique(subset=["vote_id", "party"])
        .group_by("party")
        .agg(pl.col("total_voters").mean().alias("avg_size"))
    )
    if party_sizes.height == 0:
        return pl.DataFrame()

    majority_party = party_sizes.sort("avg_size", descending=True)["party"][0]

    fractured = (
        rice_df.filter(
            (pl.col("party") == majority_party)
            & (pl.col("rice_index") < RICE_FRACTURE_THRESHOLD)
            & pl.col("rice_index").is_not_null()
        )
        .sort("rice_index")
        .with_columns(pl.lit(session).alias("session"))
    )

    print(
        f"  {chamber}: {fractured.height} fractured votes "
        f"({majority_party} Rice < {RICE_FRACTURE_THRESHOLD})"
    )
    return fractured


def plot_rice_distribution(
    rice_df: pl.DataFrame,
    rice_summary: dict[str, dict],
    chamber: str,
    out_dir: Path,
) -> None:
    """Histogram of Rice Index by party."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, party in zip(axes, ["Republican", "Democrat"]):
        vals = rice_df.filter(
            (pl.col("party") == party) & pl.col("rice_index").is_not_null()
        )["rice_index"].to_numpy()

        if len(vals) == 0:
            continue

        color = PARTY_COLORS[party]
        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor="white")

        mean_val = rice_summary.get(party, {}).get("mean", 0)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5)
        ax.text(
            mean_val + 0.02,
            ax.get_ylim()[1] * 0.9,
            f"Mean: {mean_val:.2f}",
            fontsize=9,
            fontweight="bold",
        )

        pct_unity = rice_summary.get(party, {}).get("pct_perfect_unity", 0)
        ax.set_title(f"{party}\n{pct_unity:.0f}% of votes are perfectly unified", fontsize=11)
        ax.set_xlabel("Rice Index (0 = split, 1 = unanimous)")
        ax.set_ylabel("Number of Roll Calls")

    fig.suptitle(
        f"{chamber} — How Often Does Each Party Vote Together?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"rice_distribution_{chamber.lower()}.png")


def plot_rice_over_time(
    rice_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Rolling average of Rice Index over time, per party."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for party in ["Republican", "Democrat"]:
        party_rice = (
            rice_df.filter(
                (pl.col("party") == party) & pl.col("rice_index").is_not_null()
            )
            .sort("vote_date", "vote_id")
        )
        if party_rice.height < ROLLING_WINDOW:
            continue

        vals = party_rice["rice_index"].to_numpy()
        # Rolling mean over ROLLING_WINDOW roll calls
        rolling = np.convolve(vals, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
        x = np.arange(len(rolling))

        color = PARTY_COLORS[party]
        ax.plot(x, rolling, color=color, linewidth=1.5, label=party)
        ax.fill_between(x, rolling, alpha=0.1, color=color)

    ax.set_xlabel(f"Roll Call Sequence (window = {ROLLING_WINDOW} votes)")
    ax.set_ylabel("Rice Index (rolling average)")
    ax.set_title(
        f"{chamber} — Party Cohesion Through the Session",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"rice_over_time_{chamber.lower()}.png")


def plot_rice_by_vote_type(
    rice_by_type: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Grouped bar chart of mean Rice by vote type and party."""
    if rice_by_type.height == 0:
        return

    # Get unique motions — limit to top 15 by total N
    top_motions = (
        rice_by_type.group_by("motion")
        .agg(pl.col("n_votes").sum().alias("total_n"))
        .sort("total_n", descending=True)
        .head(15)
    )
    motions = top_motions["motion"].to_list()
    parties = ["Republican", "Democrat"]

    if len(motions) == 0:
        return

    fig_width = max(10, len(motions) * 0.9)
    fig, ax = plt.subplots(figsize=(min(fig_width, 18), 6))
    x = np.arange(len(motions))
    width = 0.35

    for i, party in enumerate(parties):
        party_data = rice_by_type.filter(pl.col("party") == party)
        vals = []
        for motion in motions:
            row = party_data.filter(pl.col("motion") == motion)
            vals.append(float(row["mean_rice"][0]) if row.height > 0 else 0)

        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            color=PARTY_COLORS[party],
            alpha=0.8,
            label=party,
        )
        # Add count labels
        for j, motion in enumerate(motions):
            row = party_data.filter(pl.col("motion") == motion)
            if row.height > 0:
                n = int(row["n_votes"][0])
                ax.text(
                    x[j] + offset,
                    vals[j] + 0.01,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    # Truncate long motion names
    short_motions = [m[:30] + "..." if len(m) > 30 else m for m in motions]
    ax.set_xticks(x)
    ax.set_xticklabels(short_motions, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Rice Index")
    ax.set_title(
        f"{chamber} — Party Cohesion by Vote Type",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"rice_by_vote_type_{chamber.lower()}.png")


# ── Phase 4+6: Unity & Maverick (merged — single pass) ─────────────────────


def compute_unity_and_maverick(
    votes: pl.DataFrame,
    party_votes_df: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
    session: str,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, set[str]]]:
    """Compute CQ-standard party unity AND maverick scores in a single pass.

    Returns (unity_df, maverick_df, defection_sets).
    - unity_df has: legislator_slug, party, votes_with_party, party_votes_present,
      unity_score, maverick_rate, session, full_name, district
    - maverick_df adds: weighted_maverick, n_party_votes, n_defections, loyalty_zscore
    - defection_sets maps legislator_slug -> set of vote_ids where they defected
      (used downstream by co-defection matrix)
    """
    pv = party_votes_df.filter(pl.col("is_party_vote"))
    if pv.height == 0:
        print(f"  {chamber}: No party votes found!")
        return pl.DataFrame(), pl.DataFrame(), {}

    pv_ids = set(pv["vote_id"].to_list())
    r_majority = dict(zip(pv["vote_id"].to_list(), pv["r_majority"].to_list()))
    d_majority = dict(zip(pv["vote_id"].to_list(), pv["d_majority"].to_list()))
    closeness = dict(zip(pv["vote_id"].to_list(), pv["closeness_weight"].to_list()))

    leg_party = dict(
        zip(legislators["slug"].to_list(), legislators["party"].to_list())
    )

    indiv = votes.filter(
        pl.col("vote_id").is_in(pv_ids) & pl.col("vote").is_in(["Yea", "Nay"])
    )

    # Single pass: accumulate per-legislator stats and defection vote IDs
    leg_data: dict[str, dict] = {}
    defection_sets: dict[str, set[str]] = {}

    for row in indiv.iter_rows(named=True):
        slug = row["legislator_slug"]
        party = leg_party.get(slug)
        if not party:
            continue
        vid = row["vote_id"]
        vote_cat = row["vote"]
        party_maj = r_majority.get(vid) if party == "Republican" else d_majority.get(vid)
        if party_maj is None:
            continue

        if slug not in leg_data:
            leg_data[slug] = {
                "party": party,
                "n_party_votes": 0,
                "votes_with_party": 0,
                "n_defections": 0,
                "weighted_defections": 0.0,
                "total_weight": 0.0,
            }
            defection_sets[slug] = set()

        d = leg_data[slug]
        d["n_party_votes"] += 1
        w = closeness.get(vid, 1.0)
        d["total_weight"] += w

        if vote_cat == party_maj:
            d["votes_with_party"] += 1
        else:
            d["n_defections"] += 1
            d["weighted_defections"] += w
            defection_sets[slug].add(vid)

    if not leg_data:
        return pl.DataFrame(), pl.DataFrame(), {}

    # Build output rows
    rows = []
    for slug, d in leg_data.items():
        n = d["n_party_votes"]
        if n == 0:
            continue
        unity = d["votes_with_party"] / n
        mav_rate = d["n_defections"] / n
        w_mav = d["weighted_defections"] / d["total_weight"] if d["total_weight"] > 0 else 0.0
        rows.append(
            {
                "legislator_slug": slug,
                "party": d["party"],
                "votes_with_party": d["votes_with_party"],
                "party_votes_present": n,
                "unity_score": unity,
                "maverick_rate": mav_rate,
                "weighted_maverick": w_mav,
                "n_party_votes": n,
                "n_defections": d["n_defections"],
            }
        )

    if not rows:
        return pl.DataFrame(), pl.DataFrame(), {}

    combined_df = pl.DataFrame(rows)

    # Compute loyalty z-score within party
    party_dfs = []
    for party in ["Republican", "Democrat"]:
        party_sub = combined_df.filter(pl.col("party") == party)
        if party_sub.height < 3:
            party_dfs.append(
                party_sub.with_columns(pl.lit(None).cast(pl.Float64).alias("loyalty_zscore"))
            )
            continue
        mean_unity = float(party_sub["unity_score"].mean())
        std_unity = float(party_sub["unity_score"].std())
        if std_unity < 1e-10:
            party_dfs.append(party_sub.with_columns(pl.lit(0.0).alias("loyalty_zscore")))
        else:
            party_dfs.append(
                party_sub.with_columns(
                    ((pl.col("unity_score") - mean_unity) / std_unity).alias("loyalty_zscore")
                )
            )

    combined_df = pl.concat(party_dfs)

    # Join metadata
    combined_df = combined_df.join(
        legislators.select(
            pl.col("slug").alias("legislator_slug"), "full_name", "district"
        ),
        on="legislator_slug",
        how="left",
    ).with_columns(pl.lit(session).alias("session"))

    # Split into unity and maverick views
    unity_df = combined_df.select(
        "legislator_slug", "party", "votes_with_party", "party_votes_present",
        "unity_score", "maverick_rate", "session", "full_name", "district",
    ).sort("unity_score")

    maverick_df = combined_df.select(
        "legislator_slug", "party", "unity_score", "maverick_rate",
        "weighted_maverick", "n_party_votes", "n_defections", "loyalty_zscore",
        "full_name", "district", "session",
    )

    # Print summary
    print(
        f"  {chamber}: {unity_df.height} legislators with unity scores "
        f"(mean={float(unity_df['unity_score'].mean()):.3f})"
    )
    print(
        f"  {chamber}: mean maverick rate = {float(maverick_df['maverick_rate'].mean()):.3f}"
    )
    top_mav = maverick_df.sort("maverick_rate", descending=True).head(5)
    for row in top_mav.iter_rows(named=True):
        print(
            f"    {row['full_name']} ({row['party'][0]}): "
            f"maverick={row['maverick_rate']:.3f}, "
            f"weighted={row['weighted_maverick']:.3f}, "
            f"defections={row['n_defections']}/{row['n_party_votes']}"
        )

    return unity_df, maverick_df, defection_sets


def plot_party_unity_ranking(
    unity_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Horizontal bar chart of ALL legislators, sorted by unity, party-colored."""
    if unity_df.height == 0:
        return

    sorted_df = unity_df.sort("unity_score")

    names = sorted_df["full_name"].to_list()
    scores = sorted_df["unity_score"].to_numpy()
    parties = sorted_df["party"].to_list()
    colors = [PARTY_COLORS.get(p, "#888888") for p in parties]

    fig_height = max(8, len(names) * 0.18)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y = np.arange(len(names))
    ax.barh(y, scores, color=colors, alpha=0.8, edgecolor="white", height=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Party Unity Score (higher = more loyal to party)")
    ax.set_title(
        f"{chamber} — Who Are the Most Independent Legislators?",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlim(0, 1.05)

    # Annotate top and bottom 5
    for i in range(min(5, len(names))):
        ax.annotate(
            f"{scores[i]:.2f}",
            (scores[i] + 0.01, y[i]),
            fontsize=6,
            va="center",
            fontweight="bold",
        )
    for i in range(max(0, len(names) - 5), len(names)):
        ax.annotate(
            f"{scores[i]:.2f}",
            (scores[i] + 0.01, y[i]),
            fontsize=6,
            va="center",
            fontweight="bold",
        )

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"party_unity_ranking_{chamber.lower()}.png")


# ── Phase 5: Effective Number of Parties ─────────────────────────────────────


def compute_enp_seats(
    legislators: pl.DataFrame,
    chamber: str,
    session: str,
) -> pl.DataFrame:
    """Compute static seat-based ENP (Laakso-Taagepera)."""
    chamber_prefix = "sen_" if chamber == "Senate" else "rep_"
    chamber_legs = legislators.filter(pl.col("slug").str.starts_with(chamber_prefix))

    party_counts = chamber_legs.group_by("party").agg(pl.len().alias("seats"))
    total = float(party_counts["seats"].sum())

    if total == 0:
        return pl.DataFrame()

    party_counts = party_counts.with_columns(
        (pl.col("seats") / total).alias("seat_share"),
    )

    hhi = float((party_counts["seat_share"] ** 2).sum())
    enp = 1.0 / hhi if hhi > 0 else 0.0

    result = party_counts.with_columns(
        pl.lit(enp).alias("enp_seats"),
        pl.lit(chamber).alias("chamber"),
        pl.lit(session).alias("session"),
    )

    print(f"  {chamber}: ENP (seats) = {enp:.3f}")
    for row in result.iter_rows(named=True):
        print(f"    {row['party']}: {row['seats']} seats ({row['seat_share']:.3f})")

    return result


def compute_enp_per_vote(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
    session: str,
) -> pl.DataFrame:
    """Compute per-vote ENP from (party, direction) blocs.

    Vectorized with polars group_by — no Python-level vote iteration.
    """
    chamber_rollcalls = rollcalls.filter(pl.col("chamber") == chamber)
    chamber_vote_ids = set(chamber_rollcalls["vote_id"].to_list())

    vote_party = votes.join(
        legislators.select(pl.col("slug").alias("legislator_slug"), "party"),
        on="legislator_slug",
        how="left",
    ).filter(
        pl.col("vote_id").is_in(chamber_vote_ids)
        & pl.col("vote").is_in(["Yea", "Nay"])
    )

    # Count (party, vote_direction) blocs per vote
    blocs = vote_party.group_by("vote_id", "party", "vote").agg(
        pl.len().alias("bloc_size")
    )

    # Compute total voters per vote
    vote_totals = blocs.group_by("vote_id").agg(
        pl.col("bloc_size").sum().alias("total_voters"),
        pl.col("bloc_size").len().alias("n_blocs"),
    )

    # Join totals and compute share per bloc
    with_shares = blocs.join(vote_totals, on="vote_id").with_columns(
        (pl.col("bloc_size").cast(pl.Float64) / pl.col("total_voters")).alias("share")
    )

    # HHI = sum(share^2) per vote, then ENP = 1/HHI
    enp_df = (
        with_shares.group_by("vote_id")
        .agg(
            (pl.col("share") ** 2).sum().alias("hhi"),
            pl.col("n_blocs").first(),
            pl.col("total_voters").first(),
        )
        .with_columns(
            (1.0 / pl.col("hhi")).alias("enp"),
            pl.lit(session).alias("session"),
        )
        .drop("hhi")
    )

    if enp_df.height == 0:
        return pl.DataFrame()

    # Join vote metadata
    enp_df = enp_df.join(
        chamber_rollcalls.select("vote_id", "vote_date", "motion", "bill_number"),
        on="vote_id",
        how="left",
    )

    mean_enp = float(enp_df["enp"].mean())
    n_multiparty = enp_df.filter(pl.col("enp") > ENP_MULTIPARTY_THRESHOLD).height
    print(
        f"  {chamber}: mean ENP (votes) = {mean_enp:.3f}, "
        f"{n_multiparty} votes with ENP > {ENP_MULTIPARTY_THRESHOLD}"
    )

    return enp_df


def plot_enp_distribution(
    enp_df: pl.DataFrame,
    enp_seats: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Histogram of per-vote ENP."""
    if enp_df.height == 0:
        return

    vals = enp_df["enp"].to_numpy()
    seat_enp = float(enp_seats["enp_seats"][0]) if enp_seats.height > 0 else None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(vals, bins=30, color="#4C72B0", alpha=0.7, edgecolor="white")

    ax.axvline(2.0, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(2.02, ax.get_ylim()[1] * 0.95, "Two-party line", fontsize=8, color="#888888")

    ax.axvline(ENP_MULTIPARTY_THRESHOLD, color="#E81B23", linestyle="--", linewidth=1.5)
    ax.text(
        ENP_MULTIPARTY_THRESHOLD + 0.02,
        ax.get_ylim()[1] * 0.85,
        f"Multiparty threshold ({ENP_MULTIPARTY_THRESHOLD})",
        fontsize=8,
        color="#E81B23",
    )

    if seat_enp:
        ax.axvline(seat_enp, color="#0015BC", linestyle="--", linewidth=1.5)
        ax.text(
            seat_enp + 0.02,
            ax.get_ylim()[1] * 0.75,
            f"Seat-based ENP ({seat_enp:.2f})",
            fontsize=8,
            color="#0015BC",
        )

    n_multi = (vals > ENP_MULTIPARTY_THRESHOLD).sum()
    pct = n_multi / len(vals) * 100

    ax.set_xlabel("Effective Number of Parties (per vote)")
    ax.set_ylabel("Number of Roll Calls")
    ax.set_title(
        f"{chamber} — When Does Kansas Act Like More Than Two Parties?\n"
        f"{n_multi} of {len(vals)} votes ({pct:.0f}%) show multiparty-like behavior",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"enp_distribution_{chamber.lower()}.png")


def plot_enp_over_time(
    enp_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Rolling average of ENP over time."""
    if enp_df.height < ROLLING_WINDOW:
        return

    sorted_df = enp_df.sort("vote_date", "vote_id")
    vals = sorted_df["enp"].to_numpy()
    rolling = np.convolve(vals, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid")
    x = np.arange(len(rolling))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, rolling, color="#4C72B0", linewidth=1.5)
    ax.fill_between(x, rolling, alpha=0.1, color="#4C72B0")

    ax.axhline(2.0, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(ENP_MULTIPARTY_THRESHOLD, color="#E81B23", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel(f"Roll Call Sequence (window = {ROLLING_WINDOW} votes)")
    ax.set_ylabel("Effective Number of Parties (rolling average)")
    ax.set_title(
        f"{chamber} — Voting Fragmentation Through the Session",
        fontsize=14,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"enp_over_time_{chamber.lower()}.png")


# ── Co-Defection Matrix ────────────────────────────────────────────────────


def compute_co_defection_matrix(
    defection_sets: dict[str, set[str]],
    maverick_df: pl.DataFrame,
    chamber: str,
    session: str,
) -> pl.DataFrame | None:
    """Compute co-defection matrix for top defectors in the majority party.

    Uses pre-computed defection_sets from compute_unity_and_maverick() instead
    of re-iterating through votes.

    Returns long-format DataFrame: leg_a, leg_b, shared_defections, name_a, name_b.
    """
    # Find majority party
    party_counts = maverick_df.group_by("party").agg(pl.len().alias("n"))
    if party_counts.height == 0:
        return None
    majority_party = party_counts.sort("n", descending=True)["party"][0]

    # Top defectors in majority party
    party_mav = maverick_df.filter(pl.col("party") == majority_party).sort(
        "n_defections", descending=True
    )
    top_defectors = party_mav.head(TOP_DEFECTORS_N)

    if top_defectors.height < 2:
        print(f"  {chamber}: Fewer than 2 defectors — skipping co-defection matrix")
        return None

    top_slugs = set(top_defectors["legislator_slug"].to_list())
    slug_name = dict(
        zip(top_defectors["legislator_slug"].to_list(), top_defectors["full_name"].to_list())
    )

    # Build pairwise co-defection counts from pre-computed defection sets
    slugs_list = sorted(top_slugs)
    rows = []
    for i, sa in enumerate(slugs_list):
        sa_defections = defection_sets.get(sa, set())
        for sb in slugs_list[i + 1:]:
            shared = len(sa_defections & defection_sets.get(sb, set()))
            if shared >= CO_DEFECTION_MIN:
                rows.append(
                    {
                        "leg_a": sa,
                        "leg_b": sb,
                        "name_a": slug_name.get(sa, sa),
                        "name_b": slug_name.get(sb, sb),
                        "shared_defections": shared,
                        "session": session,
                    }
                )

    if not rows:
        print(f"  {chamber}: No pairs with >= {CO_DEFECTION_MIN} shared defections")
        return None

    co_df = pl.DataFrame(rows).sort("shared_defections", descending=True)
    print(
        f"  {chamber}: {co_df.height} co-defection pairs "
        f"(max = {int(co_df['shared_defections'].max())})"
    )

    return co_df


def plot_maverick_landscape(
    maverick_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Scatter plot: unweighted vs weighted maverick."""
    if maverick_df.height == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for party in ["Republican", "Democrat"]:
        p_df = maverick_df.filter(pl.col("party") == party)
        if p_df.height == 0:
            continue
        ax.scatter(
            p_df["maverick_rate"].to_numpy(),
            p_df["weighted_maverick"].to_numpy(),
            c=PARTY_COLORS[party],
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
            label=party,
        )

    # 1:1 line
    lim = max(
        float(maverick_df["maverick_rate"].max()),
        float(maverick_df["weighted_maverick"].max()),
    )
    ax.plot([0, lim * 1.1], [0, lim * 1.1], "k--", alpha=0.3, linewidth=1)

    # Quadrant labels
    ax.text(
        lim * 0.75,
        lim * 0.25,
        "Defects on\nblowout votes\n(performative)",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
        fontstyle="italic",
    )
    ax.text(
        lim * 0.25,
        lim * 0.75,
        "Defects on\nclose votes\n(strategic)",
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
        fontstyle="italic",
    )

    # Annotate top 5 mavericks
    top5 = maverick_df.sort("maverick_rate", descending=True).head(5)
    for row in top5.iter_rows(named=True):
        name = row["full_name"]
        # Use last name only
        short_name = name.split()[-1] if " " in name else name
        ax.annotate(
            short_name,
            (row["maverick_rate"], row["weighted_maverick"]),
            fontsize=7,
            fontweight="bold",
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Maverick Rate (unweighted)")
    ax.set_ylabel("Maverick Rate (weighted by vote closeness)")
    ax.set_title(
        f"{chamber} — Strategic vs Performative Independence",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"maverick_landscape_{chamber.lower()}.png")


def plot_co_defection_heatmap(
    co_defection_df: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Heatmap of co-defection counts."""
    if co_defection_df is None or co_defection_df.height == 0:
        return

    # Build matrix from long format
    all_names = sorted(
        set(co_defection_df["name_a"].to_list() + co_defection_df["name_b"].to_list())
    )
    # Use last names for display
    short_names = []
    for n in all_names:
        parts = n.split()
        short_names.append(parts[-1] if len(parts) > 1 else n)

    n = len(all_names)
    matrix = np.zeros((n, n))
    name_idx = {name: i for i, name in enumerate(all_names)}

    for row in co_defection_df.iter_rows(named=True):
        i = name_idx[row["name_a"]]
        j = name_idx[row["name_b"]]
        matrix[i, j] = row["shared_defections"]
        matrix[j, i] = row["shared_defections"]

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(7, n * 0.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if matrix[i, j] >= CO_DEFECTION_MIN:
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
                )

    fig.colorbar(im, ax=ax, label="Shared Defections")
    ax.set_title(
        f"{chamber} — Who Breaks Ranks Together?",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, out_dir / f"co_defection_heatmap_{chamber.lower()}.png")


# ── Phase 7: Veto Override Subgroup ──────────────────────────────────────────


def analyze_veto_override_indices(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    chamber: str,
) -> dict:
    """Compute Rice and unity specifically for veto override votes."""
    override_rollcalls = rollcalls.filter(
        pl.col("motion").str.to_lowercase().str.contains("veto")
        & (pl.col("chamber") == chamber)
    )
    n_overrides = override_rollcalls.height
    if n_overrides < 2:
        print(f"  {chamber}: Only {n_overrides} veto override votes — skipping")
        return {"n_overrides": n_overrides, "skipped": True}

    print(f"  {chamber}: {n_overrides} veto override votes")

    # Compute Rice on override votes only (uses its own party_counts — different data)
    override_ids = set(override_rollcalls["vote_id"].to_list())
    override_votes = votes.filter(pl.col("vote_id").is_in(override_ids))

    party_counts = compute_party_majority_positions(
        override_votes, override_rollcalls, legislators, chamber
    )
    rice_data = _rice_from_counts(party_counts)

    override_rice: dict[str, float] = {}
    for party in ["Republican", "Democrat"]:
        vals = rice_data.filter(
            (pl.col("party") == party) & pl.col("rice_index").is_not_null()
        )["rice_index"]
        if vals.len() > 0:
            override_rice[party] = float(vals.mean())
            print(f"    {party} override Rice: {override_rice[party]:.3f}")

    return {
        "n_overrides": n_overrides,
        "skipped": False,
        "rice_by_party": override_rice,
    }


# ── Phase 8: Cross-Referencing ───────────────────────────────────────────────


def cross_reference_upstream(
    unity_df: pl.DataFrame,
    maverick_df: pl.DataFrame,
    upstream: dict[str, pl.DataFrame | None],
    chamber: str,
    out_dir: Path,
) -> dict:
    """Cross-reference indices with IRT, network centrality, clustering loyalty."""
    result: dict = {}

    irt = upstream.get("irt")
    if irt is not None and unity_df.height > 0:
        # Unity vs IRT ideal points
        merged = unity_df.join(
            irt.select("legislator_slug", "xi_mean"),
            on="legislator_slug",
            how="inner",
        )
        if merged.height >= 10:
            rho, pval = stats.spearmanr(
                merged["unity_score"].to_numpy(),
                merged["xi_mean"].to_numpy(),
            )
            result["unity_vs_irt_rho"] = float(rho)
            result["unity_vs_irt_pval"] = float(pval)
            print(f"  Unity vs IRT (Spearman): rho={rho:.4f}, p={pval:.4g}")

            # Plot
            plot_unity_vs_irt(merged, chamber, rho, out_dir)

    centrality = upstream.get("centrality")
    if centrality is not None and maverick_df.height > 0:
        # Maverick vs betweenness centrality
        if "betweenness" in centrality.columns:
            merged = maverick_df.join(
                centrality.select("legislator_slug", "betweenness"),
                on="legislator_slug",
                how="inner",
            )
            if merged.height >= 10:
                rho, pval = stats.spearmanr(
                    merged["maverick_rate"].to_numpy(),
                    merged["betweenness"].to_numpy(),
                )
                result["maverick_vs_betweenness_rho"] = float(rho)
                result["maverick_vs_betweenness_pval"] = float(pval)
                print(f"  Maverick vs Betweenness (Spearman): rho={rho:.4f}, p={pval:.4g}")

    clustering_loyalty = upstream.get("clustering_loyalty")
    if clustering_loyalty is not None and unity_df.height > 0:
        # CQ unity vs clustering party loyalty
        merged = unity_df.join(
            clustering_loyalty.select("legislator_slug", "loyalty_rate"),
            on="legislator_slug",
            how="inner",
        )
        if merged.height >= 10:
            rho, pval = stats.spearmanr(
                merged["unity_score"].to_numpy(),
                merged["loyalty_rate"].to_numpy(),
            )
            result["unity_vs_clustering_loyalty_rho"] = float(rho)
            result["unity_vs_clustering_loyalty_pval"] = float(pval)
            print(f"  CQ Unity vs Clustering Loyalty (Spearman): rho={rho:.4f}, p={pval:.4g}")

    return result


def plot_unity_vs_irt(
    merged: pl.DataFrame,
    chamber: str,
    rho: float,
    out_dir: Path,
) -> None:
    """Scatter plot: party unity vs IRT ideal point."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for party in ["Republican", "Democrat"]:
        p_df = merged.filter(pl.col("party") == party)
        if p_df.height == 0:
            continue
        ax.scatter(
            p_df["xi_mean"].to_numpy(),
            p_df["unity_score"].to_numpy(),
            c=PARTY_COLORS[party],
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
            label=party,
        )

    # Annotate top/bottom unity
    sorted_m = merged.sort("unity_score")
    for i in range(min(3, sorted_m.height)):
        row = sorted_m.row(i, named=True)
        name = row["full_name"]
        short = name.split()[-1] if " " in name else name
        ax.annotate(
            short,
            (row["xi_mean"], row["unity_score"]),
            fontsize=7,
            fontweight="bold",
            xytext=(5, -10),
            textcoords="offset points",
        )

    ax.set_xlabel("IRT Ideal Point (Liberal <-- --> Conservative)")
    ax.set_ylabel("Party Unity Score (CQ standard)")
    ax.set_title(
        f"{chamber} — Does Ideology Predict Party Loyalty?\n"
        f"Spearman rho = {rho:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_dir / f"unity_vs_irt_{chamber.lower()}.png")


# ── Phase 9: Sensitivity ────────────────────────────────────────────────────


def run_sensitivity_analysis(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    eda_dir: Path | None,
    primary_unity: pl.DataFrame,
    chamber: str,
    session: str,
) -> dict:
    """Recompute unity on EDA-filtered votes and compare."""
    if eda_dir is None:
        print(f"  {chamber}: No EDA dir — skipping sensitivity")
        return {}

    # Load EDA filtered vote matrix to get the filtered vote IDs
    ch = chamber.lower()
    filtered_path = eda_dir / "data" / f"vote_matrix_{ch}_filtered.parquet"
    if not filtered_path.exists():
        print(f"  {chamber}: No filtered vote matrix at {filtered_path}")
        return {}

    filtered_vm = pl.read_parquet(filtered_path)
    filtered_vote_ids = set(c for c in filtered_vm.columns if c != "legislator_slug")

    # Filter rollcalls to only EDA-filtered votes
    filtered_rollcalls = rollcalls.filter(
        pl.col("vote_id").is_in(filtered_vote_ids)
    )

    print(
        f"  {chamber}: Recomputing on {filtered_rollcalls.height} EDA-filtered votes "
        f"(vs {rollcalls.filter(pl.col('chamber') == chamber).height} total)"
    )

    # Recompute party votes and unity on filtered data
    filtered_pc = compute_party_majority_positions(
        votes, filtered_rollcalls, legislators, chamber
    )
    filtered_pv = identify_party_votes(filtered_pc, filtered_rollcalls, chamber, session)
    n_party_votes_filtered = filtered_pv.filter(pl.col("is_party_vote")).height
    print(f"    Party votes (filtered): {n_party_votes_filtered}")

    filtered_unity, _, _ = compute_unity_and_maverick(
        votes, filtered_pv, legislators, chamber, session
    )
    if filtered_unity.height == 0 or primary_unity.height == 0:
        return {}

    # Compare: Spearman rho between primary and filtered unity scores
    merged = primary_unity.select("legislator_slug", "unity_score").join(
        filtered_unity.select(
            "legislator_slug",
            pl.col("unity_score").alias("unity_filtered"),
        ),
        on="legislator_slug",
        how="inner",
    )

    if merged.height < 10:
        return {}

    rho, pval = stats.spearmanr(
        merged["unity_score"].to_numpy(),
        merged["unity_filtered"].to_numpy(),
    )

    # Max rank change — sort first, THEN assign rank indices
    primary_ranks = merged.sort("unity_score").with_row_index("primary_rank")
    filtered_ranks = merged.sort("unity_filtered").with_row_index("filtered_rank")
    rank_change = (
        primary_ranks.select("legislator_slug", "primary_rank")
        .join(
            filtered_ranks.select("legislator_slug", "filtered_rank"),
            on="legislator_slug",
        )
        .with_columns(
            (pl.col("primary_rank").cast(pl.Int64) - pl.col("filtered_rank").cast(pl.Int64))
            .abs()
            .alias("rank_change")
        )
    )
    max_rank_change = int(rank_change["rank_change"].max())
    max_changer = rank_change.sort("rank_change", descending=True)["legislator_slug"][0]

    result = {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "max_rank_change": max_rank_change,
        "max_rank_changer": max_changer,
        "n_primary": primary_unity.height,
        "n_filtered": filtered_unity.height,
        "n_party_votes_filtered": n_party_votes_filtered,
    }

    print(
        f"  {chamber}: Spearman rho = {rho:.4f}, "
        f"max rank change = {max_rank_change} ({max_changer})"
    )

    return result


# ── Phase 10: Manifest & Report ──────────────────────────────────────────────


def save_filtering_manifest(manifest: dict, out_dir: Path) -> None:
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    from ks_vote_scraper.session import KSSession

    ks = KSSession.from_session_string(args.session)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path("data") / ks.output_name

    results_root = Path("results") / ks.output_name

    irt_dir = Path(args.irt_dir) if args.irt_dir else results_root / "irt" / "latest"
    network_dir = (
        Path(args.network_dir) if args.network_dir else results_root / "network" / "latest"
    )
    clustering_dir = (
        Path(args.clustering_dir)
        if args.clustering_dir
        else results_root / "clustering" / "latest"
    )
    eda_dir = Path(args.eda_dir) if args.eda_dir else results_root / "eda" / "latest"

    with RunContext(
        session=args.session,
        analysis_name="indices",
        params=vars(args),
        primer=INDICES_PRIMER,
    ) as ctx:
        print(f"KS Legislature Classical Indices Analysis — Session {args.session}")
        print(f"Data:       {data_dir}")
        print(f"IRT:        {irt_dir}")
        print(f"Network:    {network_dir}")
        print(f"Clustering: {clustering_dir}")
        print(f"EDA:        {eda_dir}")
        print(f"Output:     {ctx.run_dir}")

        # ── Phase 1: Load data ──
        print_header("PHASE 1: LOADING DATA")
        votes, rollcalls, legislators = load_raw_data(data_dir)
        print(f"  Votes:       {votes.height:,} rows")
        print(f"  Roll calls:  {rollcalls.height:,} rows")
        print(f"  Legislators: {legislators.height:,} rows")

        # Determine chambers
        chambers = sorted(rollcalls["chamber"].unique().to_list())
        print(f"  Chambers:    {chambers}")

        results: dict[str, dict] = {}

        for chamber in chambers:
            chamber_rollcalls = rollcalls.filter(pl.col("chamber") == chamber)
            print(f"\n  {chamber}: {chamber_rollcalls.height} roll calls")

            chamber_results: dict = {"n_rollcalls": chamber_rollcalls.height}

            # ── Phase 2: Party Vote Identification ──
            print_header(f"PHASE 2: PARTY VOTE IDENTIFICATION — {chamber}")
            party_counts = compute_party_majority_positions(
                votes, rollcalls, legislators, chamber
            )
            party_votes_df = identify_party_votes(
                party_counts, rollcalls, chamber, args.session
            )
            n_party_votes = party_votes_df.filter(pl.col("is_party_vote")).height
            n_total = party_votes_df.height
            print(
                f"  {chamber}: {n_party_votes} party votes of {n_total} total "
                f"({n_party_votes / n_total * 100:.1f}%)"
            )

            # Save
            party_votes_df.write_parquet(
                ctx.data_dir / f"party_votes_{chamber.lower()}.parquet"
            )
            chamber_results["party_votes"] = party_votes_df
            chamber_results["n_party_votes"] = n_party_votes
            chamber_results["n_total_votes"] = n_total

            # ── Phase 3: Rice Index (reuses party_counts) ──
            print_header(f"PHASE 3: RICE INDEX — {chamber}")
            rice_df = compute_rice_index(party_counts, rollcalls, chamber, args.session)
            rice_summary = compute_rice_summary(rice_df, chamber)
            rice_by_type = compute_rice_by_vote_type(rice_df)
            fractured = find_fractured_votes(rice_df, chamber, args.session)

            rice_df.write_parquet(ctx.data_dir / f"rice_index_{chamber.lower()}.parquet")
            if fractured.height > 0:
                fractured.write_parquet(
                    ctx.data_dir / f"fractured_votes_{chamber.lower()}.parquet"
                )

            chamber_results["rice_df"] = rice_df
            chamber_results["rice_summary"] = rice_summary
            chamber_results["rice_by_type"] = rice_by_type
            chamber_results["fractured_votes"] = fractured

            # Rice plots
            plot_rice_distribution(rice_df, rice_summary, chamber, ctx.plots_dir)
            plot_rice_over_time(rice_df, chamber, ctx.plots_dir)
            plot_rice_by_vote_type(rice_by_type, chamber, ctx.plots_dir)

            # ── Phase 4+6: Unity & Maverick (single pass) ──
            print_header(f"PHASE 4+6: UNITY & MAVERICK SCORES — {chamber}")
            unity_df, maverick_df, defection_sets = compute_unity_and_maverick(
                votes, party_votes_df, legislators, chamber, args.session
            )
            if unity_df.height > 0:
                unity_df.write_parquet(
                    ctx.data_dir / f"party_unity_{chamber.lower()}.parquet"
                )
                plot_party_unity_ranking(unity_df, chamber, ctx.plots_dir)

            co_defection = None
            if maverick_df.height > 0:
                maverick_df.write_parquet(
                    ctx.data_dir / f"maverick_scores_{chamber.lower()}.parquet"
                )
                plot_maverick_landscape(maverick_df, chamber, ctx.plots_dir)

                co_defection = compute_co_defection_matrix(
                    defection_sets, maverick_df, chamber, args.session
                )
                if co_defection is not None:
                    co_defection.write_parquet(
                        ctx.data_dir / f"co_defection_{chamber.lower()}.parquet"
                    )
                    plot_co_defection_heatmap(co_defection, chamber, ctx.plots_dir)

            chamber_results["unity"] = unity_df
            chamber_results["maverick"] = maverick_df
            chamber_results["co_defection"] = co_defection

            # ── Phase 5: ENP ──
            print_header(f"PHASE 5: EFFECTIVE NUMBER OF PARTIES — {chamber}")
            enp_seats = compute_enp_seats(legislators, chamber, args.session)
            enp_votes = compute_enp_per_vote(
                votes, rollcalls, legislators, chamber, args.session
            )

            if enp_votes.height > 0:
                enp_votes.write_parquet(ctx.data_dir / f"enp_{chamber.lower()}.parquet")
                plot_enp_distribution(enp_votes, enp_seats, chamber, ctx.plots_dir)
                plot_enp_over_time(enp_votes, chamber, ctx.plots_dir)

            chamber_results["enp_seats"] = enp_seats
            chamber_results["enp_votes"] = enp_votes

            # ── Phase 7: Veto Override Subgroup ──
            print_header(f"PHASE 7: VETO OVERRIDE INDICES — {chamber}")
            override_results = analyze_veto_override_indices(
                votes, rollcalls, legislators, chamber
            )
            chamber_results["veto_overrides"] = override_results

            # ── Phase 8: Cross-Referencing ──
            if not args.skip_cross_ref:
                print_header(f"PHASE 8: CROSS-REFERENCING — {chamber}")
                upstream = load_upstream_optional(
                    irt_dir, network_dir, clustering_dir, chamber
                )
                cross_ref = cross_reference_upstream(
                    unity_df, maverick_df, upstream, chamber, ctx.plots_dir
                )
                chamber_results["cross_ref"] = cross_ref
            else:
                print_header(f"PHASE 8: CROSS-REFERENCING (SKIPPED) — {chamber}")
                chamber_results["cross_ref"] = {}

            # ── Phase 9: Sensitivity ──
            if not args.skip_sensitivity:
                print_header(f"PHASE 9: SENSITIVITY ANALYSIS — {chamber}")
                sensitivity = run_sensitivity_analysis(
                    votes, rollcalls, legislators, eda_dir,
                    unity_df, chamber, args.session
                )
                chamber_results["sensitivity"] = sensitivity
            else:
                print_header(f"PHASE 9: SENSITIVITY (SKIPPED) — {chamber}")
                chamber_results["sensitivity"] = {}

            results[chamber] = chamber_results

        # ── Phase 10: Manifest + Report ──
        print_header("PHASE 10: FILTERING MANIFEST")
        manifest: dict = {
            "analysis": "indices",
            "session": args.session,
            "constants": {
                "PARTY_VOTE_THRESHOLD": PARTY_VOTE_THRESHOLD,
                "RICE_FRACTURE_THRESHOLD": RICE_FRACTURE_THRESHOLD,
                "MIN_PARTY_VOTERS": MIN_PARTY_VOTERS,
                "MAVERICK_WEIGHT_FLOOR": MAVERICK_WEIGHT_FLOOR,
                "CO_DEFECTION_MIN": CO_DEFECTION_MIN,
                "ENP_MULTIPARTY_THRESHOLD": ENP_MULTIPARTY_THRESHOLD,
                "ROLLING_WINDOW": ROLLING_WINDOW,
                "TOP_DEFECTORS_N": TOP_DEFECTORS_N,
            },
            "skip_cross_ref": args.skip_cross_ref,
            "skip_sensitivity": args.skip_sensitivity,
        }

        for chamber, result in results.items():
            ch = chamber.lower()
            manifest[f"{ch}_n_rollcalls"] = result["n_rollcalls"]
            manifest[f"{ch}_n_party_votes"] = result["n_party_votes"]
            manifest[f"{ch}_pct_party_votes"] = (
                result["n_party_votes"] / result["n_total_votes"] * 100
                if result["n_total_votes"] > 0
                else 0
            )
            if result.get("rice_summary"):
                manifest[f"{ch}_rice_summary"] = result["rice_summary"]
            if result.get("unity") is not None and result["unity"].height > 0:
                manifest[f"{ch}_mean_unity"] = float(result["unity"]["unity_score"].mean())
            if result.get("enp_seats") is not None and result["enp_seats"].height > 0:
                manifest[f"{ch}_enp_seats"] = float(result["enp_seats"]["enp_seats"][0])
            if result.get("enp_votes") is not None and result["enp_votes"].height > 0:
                manifest[f"{ch}_mean_enp_votes"] = float(result["enp_votes"]["enp"].mean())
            if result.get("veto_overrides"):
                manifest[f"{ch}_veto_overrides"] = result["veto_overrides"]
            if result.get("cross_ref"):
                manifest[f"{ch}_cross_ref"] = result["cross_ref"]
            if result.get("sensitivity"):
                manifest[f"{ch}_sensitivity"] = result["sensitivity"]

        save_filtering_manifest(manifest, ctx.run_dir)

        # ── HTML report ──
        print_header("HTML REPORT")
        build_indices_report(
            ctx.report,
            results=results,
            plots_dir=ctx.plots_dir,
            skip_cross_ref=args.skip_cross_ref,
            skip_sensitivity=args.skip_sensitivity,
        )

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
