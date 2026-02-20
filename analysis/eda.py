"""
Kansas Legislature — Exploratory Data Analysis (Phase 1)

Covers analytic methods 01-04:
  01: Vote matrix construction
  02: Descriptive statistics
  03: Participation analysis
  04: Agreement matrix & heatmap

Usage:
  uv run python analysis/eda.py [--session 2025-26] [--data-dir data/ks_2025_26]

Outputs (in results/<session>/eda/<date>/):
  - plots/:   8 PNG visualization plots
  - data/:    Parquet intermediates (vote matrices, agreement matrices)
  - filtering_manifest.json, run_info.json, run_log.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Use non-interactive backend so the script can run headless (no GUI window).
# Must be called before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

try:
    from analysis.run_context import RunContext
except ModuleNotFoundError:
    from run_context import RunContext

# ── Primer ───────────────────────────────────────────────────────────────────
# Written to results/<session>/eda/README.md by RunContext on each run.

EDA_PRIMER = """\
# Exploratory Data Analysis (EDA)

## Purpose

EDA is the mandatory first step before any modeling. It validates the scraped
data, quantifies its structure, and produces the filtered vote matrices that
all downstream analyses (PCA, IRT, clustering) consume.

Covers analytic methods 01-04 from `Analytic_Methods/`.

## Method

1. **Data loading & session summary** — Read the three scraper CSVs (votes,
   rollcalls, legislators) and print high-level counts.
2. **Data integrity checks** — Structural validation: seat counts, referential
   integrity, duplicate detection, tally consistency, chamber-slug matching,
   vote category validation, and near-duplicate rollcall detection.
3. **Statistical quality checks** — Rice Cohesion Index per party (flags bad
   party assignments) and perfect-partisan detection.
4. **Vote matrix construction** — Pivot individual votes into a binary matrix
   (legislators x rollcalls, Yea=1, Nay=0, absent=null).
5. **Filtering** — Drop near-unanimous votes (minority < 2.5%) and
   low-participation legislators (< 20 votes). Applied per chamber.
6. **Descriptive statistics** — Vote type distribution, passage rates, margin
   distribution, party-line classification.
7. **Participation analysis** — Per-legislator substantive voting rates.
8. **Agreement matrices** — Pairwise raw agreement and Cohen's Kappa, with
   hierarchically clustered heatmaps.

## Inputs

Reads from `data/ks_{session}/`:
- `ks_{slug}_votes.csv` — One row per legislator per roll call (~68K rows)
- `ks_{slug}_rollcalls.csv` — One row per roll call (~500-900 rows)
- `ks_{slug}_legislators.csv` — One row per legislator (~170 rows)

## Outputs

All outputs land in `results/<session>/eda/<date>/`:

### `data/` — Parquet intermediates

| File | Description |
|------|-------------|
| `vote_matrix_full.parquet` | Full binary vote matrix (all legislators x all rollcalls) |
| `vote_matrix_house_filtered.parquet` | House-only, contested votes, active legislators |
| `vote_matrix_senate_filtered.parquet` | Senate-only, contested votes, active legislators |
| `agreement_raw_house.parquet` | Pairwise raw agreement (House) |
| `agreement_raw_senate.parquet` | Pairwise raw agreement (Senate) |
| `agreement_kappa_house.parquet` | Pairwise Cohen's Kappa (House) |
| `agreement_kappa_senate.parquet` | Pairwise Cohen's Kappa (Senate) |

### `plots/` — PNG visualizations

| File | Description |
|------|-------------|
| `vote_type_distribution.png` | Bar chart of rollcall types (Final Action, Emergency, etc.) |
| `vote_margin_distribution.png` | Histogram of Yea% with 50% and 2/3 threshold lines |
| `temporal_activity.png` | Monthly rollcall counts by chamber |
| `party_vote_breakdown.png` | Vote category breakdown (Yea/Nay/Absent) by party |
| `participation_rates_house.png` | Per-legislator participation, colored by party |
| `participation_rates_senate.png` | Per-legislator participation, colored by party |
| `agreement_heatmap_house.png` | Clustered heatmap of pairwise agreement (House) |
| `agreement_heatmap_senate.png` | Clustered heatmap of pairwise agreement (Senate) |

### Root files

| File | Description |
|------|-------------|
| `filtering_manifest.json` | All filtering decisions, integrity findings, and statistical checks |
| `run_info.json` | Git commit, timestamp, Python version, parameters |
| `run_log.txt` | Full console output from the run |

## Interpretation Guide

- **Agreement heatmaps**: Look for block structure. In Kansas, expect a clear
  R/D split with visible intra-Republican factions (conservative vs moderate).
  Ward linkage dendrograms reveal coalition structure.
- **Participation rates**: Most legislators are >95%. Low outliers are often
  mid-session replacements (check the integrity findings).
- **Vote margin histogram**: Bimodal shape expected — large peak near 100%
  (near-unanimous) and a spread of contested votes. The gap near 50% reflects
  that close votes are rare in a supermajority legislature.
- **Rice Cohesion**: Both parties should be >0.7 mean. If either is <0.5,
  party assignments are likely wrong (see Bug 2 in lessons-learned).
- **Cohen's Kappa vs raw agreement**: Always prefer Kappa for similarity
  thresholds. Raw agreement is inflated by the ~82% Yea base rate.

## Caveats

- Filtering thresholds (2.5% minority, 20-vote minimum) are defaults. The
  analytic-workflow rules require sensitivity analyses with alternative settings.
- Mid-session replacements inflate legislator counts above seat counts. The
  integrity check flags these but does not remove them — downstream analyses
  should decide how to handle replacements.
- "Present and Passing" votes (~22 instances, mostly Senate) are excluded from
  the binary matrix. They represent deliberate abstentions, not missing data.
"""

# ── Constants ────────────────────────────────────────────────────────────────
# These are explicit, named constants per the analytic-workflow rules.
# Changing these values constitutes a sensitivity analysis — document why.

MINORITY_THRESHOLD = 0.025  # Drop votes where minority < 2.5% (VoteView standard)
MIN_VOTES = 20  # Drop legislators with fewer than 20 substantive votes
VOTE_CATEGORIES = [
    "Yea", "Nay", "Present and Passing", "Absent and Not Voting", "Not Voting"
]
PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC"}

# Kansas Legislature constitutional seat counts. These are fixed and serve as
# a data integrity guardrail: if we see more legislators than seats, we have
# mid-session replacements (normal) or a scraping bug (needs investigation).
HOUSE_SEATS = 125
SENATE_SEATS = 40

# Minimum number of shared votes between two legislators to compute a
# meaningful pairwise agreement score. Below this, agreement is too noisy.
MIN_SHARED_VOTES = 10

# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS Legislature EDA")
    parser.add_argument("--session", default="2025-26")
    parser.add_argument("--data-dir", default=None, help="Override data directory path")
    return parser.parse_args()


def print_header(title: str) -> None:
    """Print a visually distinct section header to stdout."""
    width = 80
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk and close it to free memory."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── 1. Data Loading ─────────────────────────────────────────────────────────


def load_data(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load the three CSVs produced by the scraper.

    File naming convention: ks_{session_slug}_{type}.csv
    The data_dir name encodes the session slug (e.g. 'ks_2025_26').
    """
    session_slug = data_dir.name.removeprefix("ks_")
    votes = pl.read_csv(data_dir / f"ks_{session_slug}_votes.csv")
    rollcalls = pl.read_csv(data_dir / f"ks_{session_slug}_rollcalls.csv")
    legislators = pl.read_csv(data_dir / f"ks_{session_slug}_legislators.csv")
    return votes, rollcalls, legislators


def print_session_summary(
    votes: pl.DataFrame, rollcalls: pl.DataFrame, legislators: pl.DataFrame
) -> None:
    """Print high-level counts: rollcalls, bills, legislators, date range, vote categories."""
    print_header("SESSION SUMMARY")

    dates = rollcalls["vote_datetime"].sort()
    print(f"  Roll calls:    {rollcalls.height}")
    print(f"  Unique bills:  {rollcalls['bill_number'].n_unique()}")
    print(f"  Legislators:   {legislators.height}")
    print(f"  Ind. votes:    {votes.height}")
    print(f"  Date range:    {dates.first()} → {dates.last()}")

    # Chamber/party breakdown — important to verify against known seat counts
    print("\n  By chamber:")
    for row in legislators.group_by("chamber").agg(pl.len()).sort("chamber").iter_rows(named=True):
        print(f"    {row['chamber']:8s}  {row['len']}")

    print("\n  By party:")
    for row in legislators.group_by("party").agg(pl.len()).sort("party").iter_rows(named=True):
        print(f"    {row['party']:12s}  {row['len']}")

    print("\n  By chamber × party:")
    cp = (
        legislators.group_by("chamber", "party")
        .agg(pl.len())
        .sort("chamber", "party")
    )
    for row in cp.iter_rows(named=True):
        print(f"    {row['chamber']:8s}  {row['party']:12s}  {row['len']}")

    # Vote category distribution — verify all 5 categories appear as expected
    print("\n  Vote categories:")
    vc = votes.group_by("vote").agg(pl.len()).sort("len", descending=True)
    total = votes.height
    for row in vc.iter_rows(named=True):
        pct = 100 * row["len"] / total
        print(f"    {row['vote']:28s}  {row['len']:>6,}  ({pct:5.1f}%)")


# ── 1b. Data Integrity Checks ────────────────────────────────────────────────
#
# These checks catch scraping bugs and validate structural assumptions before
# any analysis runs. Inspired by VoteView, OpenStates, and the IRT literature.
# See docs/lessons-learned.md for the mid-session replacement discovery.


def check_data_integrity(
    votes: pl.DataFrame, rollcalls: pl.DataFrame, legislators: pl.DataFrame
) -> dict:
    """Run all data integrity checks. Returns findings dict for the manifest.

    Checks performed:
      1. Seat count validation (House=125, Senate=40)
      2. Mid-session replacement detection via service windows
      3. Referential integrity (votes ↔ legislators)
      4. Duplicate vote detection
      5. Tally consistency (summary counts vs individual vote details)
      6. Chamber size bounds (no rollcall exceeds chamber capacity)
      7. Chamber-slug consistency (sen_* only in Senate, rep_* only in House)
      8. Vote category validation (only the 5 known categories)
      9. Near-duplicate rollcall detection (identical vote vectors)
    """
    print_header("DATA INTEGRITY CHECKS")
    findings: dict = {"warnings": [], "info": []}

    # ── 1. Seat count check ──
    # The Kansas House has exactly 125 seats and the Senate has exactly 40.
    # More legislators than seats means mid-session replacements (or a bug).
    expected = {"House": HOUSE_SEATS, "Senate": SENATE_SEATS}
    for chamber, expected_seats in expected.items():
        actual = legislators.filter(pl.col("chamber") == chamber).height
        diff = actual - expected_seats
        if diff == 0:
            msg = f"{chamber}: {actual} legislators = {expected_seats} seats (OK)"
            print(f"  [OK]   {msg}")
            findings["info"].append(msg)
        elif diff > 0:
            msg = (
                f"{chamber}: {actual} legislators > {expected_seats} seats "
                f"(+{diff} — likely mid-session replacements)"
            )
            print(f"  [WARN] {msg}")
            findings["warnings"].append(msg)
        else:
            msg = (
                f"{chamber}: {actual} legislators < {expected_seats} seats "
                f"({diff} — missing members)"
            )
            print(f"  [WARN] {msg}")
            findings["warnings"].append(msg)

    # ── 2. Identify likely replacements via service windows ──
    # Each legislator's first and last vote date defines their service window.
    # If two legislators share a district, their windows should NOT overlap.
    # Overlap = scraping bug. Non-overlap = legitimate replacement.
    service = (
        votes.group_by("legislator_slug")
        .agg(
            pl.col("vote_datetime").min().alias("first_vote"),
            pl.col("vote_datetime").max().alias("last_vote"),
            pl.len().alias("total_individual_votes"),
        )
        .join(
            legislators.select("slug", "full_name", "chamber", "party", "district"),
            left_on="legislator_slug",
            right_on="slug",
        )
        .sort("chamber", "district", "first_vote")
    )

    # Find districts with multiple legislators (replacement signal)
    district_counts = (
        service.group_by("chamber", "district")
        .agg(
            pl.len().alias("n_legislators"),
            pl.col("full_name").alias("names"),
            pl.col("first_vote").alias("first_votes"),
            pl.col("last_vote").alias("last_votes"),
            pl.col("total_individual_votes").alias("vote_counts"),
        )
        .filter(pl.col("n_legislators") > 1)
        .sort("chamber", "district")
    )

    if district_counts.height > 0:
        print(f"\n  Districts with multiple legislators ({district_counts.height} found):")
        for row in district_counts.iter_rows(named=True):
            names = row["names"]
            firsts = row["first_votes"]
            lasts = row["last_votes"]
            counts = row["vote_counts"]
            print(f"\n    {row['chamber']} District {row['district']}:")
            for name, first, last, count in zip(names, firsts, lasts, counts):
                print(f"      {name:30s}  {first[:10]} → {last[:10]}  ({count} votes)")

            # Check for overlapping service windows — overlap is a red flag
            periods = sorted(zip(firsts, lasts, names))
            for i in range(len(periods) - 1):
                if periods[i][1] >= periods[i + 1][0]:
                    msg = (
                        f"{row['chamber']} Dist {row['district']}: "
                        f"overlapping service — {periods[i][2]} "
                        f"(last: {periods[i][1][:10]}) vs "
                        f"{periods[i + 1][2]} "
                        f"(first: {periods[i + 1][0][:10]})"
                    )
                    print("      [WARN] Overlapping service windows!")
                    findings["warnings"].append(msg)
                else:
                    msg = (
                        f"{row['chamber']} Dist {row['district']}: "
                        f"clean handoff — {periods[i][2]} → {periods[i + 1][2]}"
                    )
                    findings["info"].append(msg)
    else:
        print("\n  No districts with multiple legislators found.")

    # ── 3. Referential integrity ──
    # Every slug in votes.csv should exist in legislators.csv and vice versa.
    # Mismatches indicate a name-matching failure in the scraper.
    all_slugs = set(legislators["slug"].to_list())
    voted_slugs = set(votes["legislator_slug"].unique().to_list())

    no_votes = all_slugs - voted_slugs
    if no_votes:
        msg = f"{len(no_votes)} legislators with zero votes: {sorted(no_votes)}"
        print(f"\n  [WARN] {msg}")
        findings["warnings"].append(msg)
    else:
        print("\n  [OK]   All legislators have at least one vote")

    unknown_slugs = voted_slugs - all_slugs
    if unknown_slugs:
        msg = (
            f"{len(unknown_slugs)} vote slugs not in legislators CSV: "
            f"{sorted(unknown_slugs)}"
        )
        print(f"  [WARN] {msg}")
        findings["warnings"].append(msg)
    else:
        print("  [OK]   All vote slugs found in legislators CSV")

    # ── 4. Duplicate vote check ──
    # Same legislator voting twice on the same rollcall is impossible.
    dupes = (
        votes.group_by("vote_id", "legislator_slug")
        .agg(pl.len())
        .filter(pl.col("len") > 1)
    )
    if dupes.height > 0:
        msg = f"{dupes.height} duplicate votes (same legislator + rollcall)"
        print(f"  [WARN] {msg}")
        findings["warnings"].append(msg)
    else:
        print("  [OK]   No duplicate votes found")

    # ── 5. Tally consistency ──
    # The rollcall summary counts (yea_count, nay_count, etc.) must match
    # the actual count of individual vote records for each rollcall.
    # A mismatch means the vote page was partially parsed or names were lost.
    detail_counts = (
        votes.group_by("vote_id", "vote")
        .agg(pl.len().alias("count"))
        .pivot(on="vote", index="vote_id", values="count")
        .fill_null(0)
    )
    # Rename columns to match rollcall field names
    col_map = {
        "Yea": "detail_yea",
        "Nay": "detail_nay",
        "Present and Passing": "detail_pp",
        "Absent and Not Voting": "detail_anv",
        "Not Voting": "detail_nv",
    }
    for old, new in col_map.items():
        if old in detail_counts.columns:
            detail_counts = detail_counts.rename({old: new})
        else:
            # Category might not appear at all (e.g. "Not Voting" = 0 everywhere)
            detail_counts = detail_counts.with_columns(pl.lit(0).alias(new))

    comparison = rollcalls.join(detail_counts, on="vote_id", how="left")

    # Check yea and nay counts (the most critical ones for analysis)
    tally_mismatches = comparison.filter(
        (pl.col("yea_count") != pl.col("detail_yea"))
        | (pl.col("nay_count") != pl.col("detail_nay"))
    )
    if tally_mismatches.height > 0:
        msg = (
            f"{tally_mismatches.height} rollcalls with tally mismatches "
            f"(summary ≠ individual vote details)"
        )
        print(f"  [WARN] {msg}")
        # Show first few mismatches for debugging
        for row in tally_mismatches.head(5).iter_rows(named=True):
            print(
                f"    {row['vote_id']}: summary yea={row['yea_count']} "
                f"nay={row['nay_count']}, "
                f"detail yea={row['detail_yea']} nay={row['detail_nay']}"
            )
        findings["warnings"].append(msg)
    else:
        print("  [OK]   All tally counts match individual vote details")

    # ── 6. Chamber size bounds ──
    # No rollcall can have more participants than seats in the chamber.
    # total_votes should be <= HOUSE_SEATS (125) or SENATE_SEATS (40).
    overcount = rollcalls.with_columns(
        pl.when(pl.col("chamber") == "Senate")
        .then(pl.lit(SENATE_SEATS))
        .otherwise(pl.lit(HOUSE_SEATS))
        .alias("chamber_size")
    ).filter(pl.col("total_votes") > pl.col("chamber_size"))

    if overcount.height > 0:
        msg = (
            f"{overcount.height} rollcalls exceed chamber seat count"
        )
        print(f"  [WARN] {msg}")
        for row in overcount.head(5).iter_rows(named=True):
            print(
                f"    {row['vote_id']}: {row['chamber']} has "
                f"{row['total_votes']} votes > {row['chamber_size']} seats"
            )
        findings["warnings"].append(msg)
    else:
        print("  [OK]   No rollcall exceeds chamber seat count")

    # ── 7. Chamber-slug consistency ──
    # sen_* slugs should only appear in Senate rollcalls, rep_* in House.
    # A mismatch means the scraper assigned a legislator to the wrong chamber.
    vote_with_chamber = votes.select("vote_id", "legislator_slug", "chamber")
    cross_chamber = vote_with_chamber.filter(
        (pl.col("legislator_slug").str.starts_with("sen_")
         & (pl.col("chamber") == "House"))
        | (pl.col("legislator_slug").str.starts_with("rep_")
           & (pl.col("chamber") == "Senate"))
    )
    if cross_chamber.height > 0:
        n_affected = cross_chamber["vote_id"].n_unique()
        msg = (
            f"{cross_chamber.height} votes with chamber-slug mismatch "
            f"across {n_affected} rollcalls"
        )
        print(f"  [WARN] {msg}")
        findings["warnings"].append(msg)
    else:
        print("  [OK]   All slugs match their chamber (sen_→Senate, rep_→House)")

    # ── 8. Vote category validation ──
    # Every vote value must be one of exactly 5 known categories.
    # Anything else means the scraper pulled unexpected text from a page.
    actual_categories = set(votes["vote"].unique().to_list())
    expected_categories = set(VOTE_CATEGORIES)
    unexpected = actual_categories - expected_categories
    if unexpected:
        msg = f"Unexpected vote categories found: {sorted(unexpected)}"
        print(f"  [WARN] {msg}")
        findings["warnings"].append(msg)
    else:
        print("  [OK]   All vote values are valid categories")

    # ── 9. Near-duplicate rollcall detection ──
    # Two rollcalls with nearly identical vote vectors (>99.9% cosine similarity)
    # may indicate the same vote scraped twice under different IDs, or a
    # procedural re-vote. The threshold is 0.999 (not 0.99) because the ~82%
    # Yea base rate makes many rollcalls naturally similar at 0.99.
    _check_near_duplicate_rollcalls(votes, findings)

    # ── Summary ──
    n_warn = len(findings["warnings"])
    if n_warn == 0:
        print("\n  All checks passed.")
    else:
        print(f"\n  {n_warn} warning(s) found — review above for details.")

    return findings


def _check_near_duplicate_rollcalls(
    votes: pl.DataFrame, findings: dict
) -> None:
    """Detect rollcalls with nearly identical vote vectors.

    Checks within each chamber separately, and ONLY among contested votes
    (minority >= 2.5%). This is critical: among near-unanimous votes, identical
    vectors are trivially expected (everyone voted Yea), not a data problem.
    Among contested votes, identical vectors are genuinely suspicious.

    Flags pairs where 0 or 1 legislator voted differently among shared voters.
    """
    all_near_dupes: list[tuple[str, str, str, int, int]] = []

    for chamber in ["House", "Senate"]:
        prefix = "sen_" if chamber == "Senate" else "rep_"
        chamber_votes = (
            votes.filter(
                pl.col("vote").is_in(["Yea", "Nay"])
                & pl.col("legislator_slug").str.starts_with(prefix)
            )
            .with_columns(
                pl.when(pl.col("vote") == "Yea")
                .then(1)
                .otherwise(0)
                .alias("vote_binary")
            )
        )

        binary = chamber_votes.pivot(
            on="vote_id", index="legislator_slug", values="vote_binary"
        )
        vote_ids = [c for c in binary.columns if c != "legislator_slug"]
        if len(vote_ids) < 2:
            continue

        # NaN = legislator didn't vote on that rollcall (absent, not in session)
        mat = binary.select(vote_ids).to_numpy().astype(np.float64)

        # Pre-filter to contested votes only (minority >= 2.5%)
        # Among unanimous votes, identical vectors are trivially expected.
        contested_mask = []
        for col_idx in range(mat.shape[1]):
            col = mat[:, col_idx]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                contested_mask.append(False)
                continue
            minority_frac = min(valid.mean(), 1 - valid.mean())
            contested_mask.append(minority_frac >= MINORITY_THRESHOLD)

        contested_indices = [i for i, keep in enumerate(contested_mask) if keep]
        contested_vote_ids = [vote_ids[i] for i in contested_indices]
        mat = mat[:, contested_indices]

        # For each pair of contested votes, count disagreements
        for i in range(len(contested_vote_ids)):
            for j in range(i + 1, len(contested_vote_ids)):
                mask = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
                shared = mask.sum()
                if shared < 10:
                    continue
                diffs = int((mat[mask, i] != mat[mask, j]).sum())
                if diffs <= 1:
                    all_near_dupes.append(
                        (contested_vote_ids[i], contested_vote_ids[j],
                         chamber, int(shared), diffs)
                    )

    if all_near_dupes:
        # Exact duplicates (0 diffs) = definite problem; 1-diff = informational
        exact = [d for d in all_near_dupes if d[4] == 0]
        one_diff = [d for d in all_near_dupes if d[4] == 1]

        if exact:
            # Identical vote patterns on contested votes are common in polarized
            # legislatures (party-line votes all look the same). This is INFO
            # not WARN — investigate only if the same bill appears twice.
            msg = (
                f"{len(exact)} contested-vote pairs with identical patterns "
                f"(common in party-line voting)"
            )
            print(f"  [INFO] {msg}")
            findings["info"].append(msg)

        if one_diff:
            msg = f"{len(one_diff)} near-duplicate pairs (1 legislator difference)"
            print(f"  [INFO] {msg}")
            findings["info"].append(msg)

        findings["near_duplicate_rollcalls"] = [
            {
                "vote_id_a": a, "vote_id_b": b, "chamber": ch,
                "shared_voters": s, "differences": d,
            }
            for a, b, ch, s, d in all_near_dupes
        ]
    else:
        print("  [OK]   No near-duplicate rollcalls found")


# ── 1c. Statistical Quality Checks ──────────────────────────────────────────
#
# These go beyond structural integrity to detect statistical anomalies that
# could contaminate downstream models (PCA, IRT, clustering).


def compute_rice_cohesion(
    votes: pl.DataFrame, legislators: pl.DataFrame
) -> pl.DataFrame:
    """Compute Rice Cohesion Index per party per rollcall.

    The Rice Index measures party unity:
      RICE = |%Yea - %Nay| among Yea+Nay voters only (excluding abstentions)

    Range: 0 (evenly split) to 1 (perfect unity).

    Returns DataFrame with columns: vote_id, party, rice_index, n_voting.
    """
    # Only count substantive Yea/Nay votes — abstentions excluded per Rice formula
    substantive = (
        votes.filter(pl.col("vote").is_in(["Yea", "Nay"]))
        .join(
            legislators.select("slug", "party"),
            left_on="legislator_slug",
            right_on="slug",
        )
    )

    rice = (
        substantive.group_by("vote_id", "party")
        .agg(
            (pl.col("vote") == "Yea").sum().alias("yea"),
            (pl.col("vote") == "Nay").sum().alias("nay"),
            pl.len().alias("n_voting"),
        )
        .with_columns(
            # Rice = |yea - nay| / total — equivalent to |%Yea - %Nay|
            # Cast to Int64 first: polars boolean .sum() returns UInt32,
            # and UInt32 subtraction overflows when nay > yea.
            (
                (pl.col("yea").cast(pl.Int64) - pl.col("nay").cast(pl.Int64)).abs()
                / pl.col("n_voting")
            ).alias("rice_index")
        )
    )
    return rice


def check_statistical_quality(
    votes: pl.DataFrame, rollcalls: pl.DataFrame, legislators: pl.DataFrame
) -> dict:
    """Run statistical sanity checks that gate downstream analysis.

    Checks:
      1. Rice cohesion index per party (mean < 0.5 = party data probably wrong)
      2. Perfect partisan detection (100% party-line on 50+ contested votes)

    Returns findings dict to merge into the manifest.
    """
    print_header("STATISTICAL QUALITY CHECKS")
    findings: dict = {"warnings": [], "info": []}

    # ── 1. Rice Cohesion Index ──
    # Expected for Kansas: R mean ~0.80-0.95, D mean ~0.80-0.95.
    # If mean < 0.5 for either party, the party column is probably wrong
    # (recall Bug 2 in lessons-learned.md where everyone was tagged Republican).
    rice = compute_rice_cohesion(votes, legislators)

    print("\n  Rice Cohesion Index by party:")
    print("  (0 = evenly split, 1 = perfect unity)")
    rice_summary = (
        rice.group_by("party")
        .agg(
            pl.col("rice_index").mean().alias("mean"),
            pl.col("rice_index").std().alias("std"),
            (pl.col("rice_index") == 1.0).mean().alias("pct_perfect"),
            pl.len().alias("n_party_votes"),
        )
        .sort("party")
    )
    for row in rice_summary.iter_rows(named=True):
        pct_perfect = 100 * (row["pct_perfect"] or 0)
        print(
            f"    {row['party']:12s}  mean={row['mean']:.3f}  "
            f"std={row['std']:.3f}  "
            f"perfect_unity={pct_perfect:.1f}%  "
            f"(n={row['n_party_votes']})"
        )
        # Flag if mean Rice is suspiciously low
        if row["mean"] < 0.5:
            msg = (
                f"{row['party']} mean Rice={row['mean']:.3f} < 0.5 — "
                f"party assignments may be incorrect"
            )
            findings["warnings"].append(msg)
            print(f"    [WARN] {msg}")

    findings["rice_summary"] = rice_summary.to_dicts()

    # ── 2. Perfect partisan detection ──
    # A legislator who votes 100% with their party on every contested vote
    # could be genuinely extreme, or could be a scraping artifact where their
    # name defaulted to a category. Flag for manual review.
    perfect, unity = _detect_perfect_partisans(votes, legislators)

    if perfect.height > 0:
        print("\n  Perfect partisans (100% party-line, >=50 contested votes):")
        for row in perfect.iter_rows(named=True):
            name_row = legislators.filter(pl.col("slug") == row["legislator_slug"])
            name = name_row["full_name"][0] if name_row.height > 0 else row["legislator_slug"]
            party = name_row["party"][0] if name_row.height > 0 else "?"
            print(
                f"    {name:30s}  {party:12s}  "
                f"{row['n_contested']:.0f} contested votes, "
                f"100% party-line"
            )
        msg = (
            f"{perfect.height} legislators with 100% party-line voting on "
            f"50+ contested votes — verify these are genuine"
        )
        findings["info"].append(msg)
    else:
        print("\n  [OK]   No perfect partisans found (100% on 50+ contested)")

    # Store the full party unity distribution for the manifest
    findings["party_unity_stats"] = {
        "mean": float(unity["party_unity_rate"].mean()),
        "median": float(unity["party_unity_rate"].median()),
        "min": float(unity["party_unity_rate"].min()),
        "max": float(unity["party_unity_rate"].max()),
    }

    return findings


def _detect_perfect_partisans(
    votes: pl.DataFrame, legislators: pl.DataFrame, min_contested: int = 50
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Find legislators who vote 100% with their party on contested votes.

    "Party direction" per rollcall = whichever way >50% of the party voted.
    A legislator is "perfect" if they never deviate from their party's
    majority direction on any contested vote where both parties participated.

    Returns (perfect_partisans_df, full_unity_df).
    """
    # Join votes with party info, keep only Yea/Nay
    v = (
        votes.filter(pl.col("vote").is_in(["Yea", "Nay"]))
        .join(
            legislators.select("slug", "party"),
            left_on="legislator_slug",
            right_on="slug",
        )
    )

    # Compute party majority direction per rollcall:
    # For each (vote_id, party), which way did the majority vote?
    party_majority = (
        v.group_by("vote_id", "party")
        .agg(
            (pl.col("vote") == "Yea").sum().alias("party_yea"),
            (pl.col("vote") == "Nay").sum().alias("party_nay"),
        )
        .with_columns(
            pl.when(pl.col("party_yea") >= pl.col("party_nay"))
            .then(pl.lit("Yea"))
            .otherwise(pl.lit("Nay"))
            .alias("party_direction")
        )
    )

    # Join back: did each legislator vote with their party?
    v = v.join(
        party_majority.select("vote_id", "party", "party_direction"),
        on=["vote_id", "party"],
    )
    v = v.with_columns(
        (pl.col("vote") == pl.col("party_direction")).alias("with_party")
    )

    # Aggregate per legislator
    unity = (
        v.group_by("legislator_slug")
        .agg(
            pl.col("with_party").mean().alias("party_unity_rate"),
            pl.col("with_party").len().alias("n_contested"),
        )
    )

    # Flag: 100% party line with sufficient contested votes
    perfect = unity.filter(
        (pl.col("party_unity_rate") == 1.0)
        & (pl.col("n_contested") >= min_contested)
    )

    return perfect, unity


# ── 2. Vote Matrix Construction ─────────────────────────────────────────────


def build_vote_matrix(votes: pl.DataFrame) -> pl.DataFrame:
    """Pivot votes into a binary matrix: rows=legislator_slug, cols=vote_id.

    Encoding: Yea=1, Nay=0, all others (absent, present, not voting)=null.
    Null represents "no substantive vote cast" — important for downstream
    analysis where we only want to model Yea/Nay choices.
    """
    binary = votes.with_columns(
        pl.when(pl.col("vote") == "Yea")
        .then(1)
        .when(pl.col("vote") == "Nay")
        .then(0)
        .otherwise(None)
        .alias("binary_vote")
    )
    matrix = binary.pivot(
        on="vote_id",
        index="legislator_slug",
        values="binary_vote",
    )
    return matrix


def filter_vote_matrix(
    matrix: pl.DataFrame,
    rollcalls: pl.DataFrame,
    chamber: str | None = None,
    minority_threshold: float = MINORITY_THRESHOLD,
    min_votes: int = MIN_VOTES,
) -> tuple[pl.DataFrame, dict]:
    """Filter a vote matrix for contested votes and active legislators.

    Two filters applied in sequence:
      1. Drop near-unanimous votes (minority side < threshold). These carry
         no ideological signal — everyone agrees.
      2. Drop legislators with too few substantive votes (< min_votes).
         Their ideal point estimates would be unreliable.

    If chamber is specified, restricts to that chamber's rollcalls and
    legislators only. Chambers must be analyzed separately because House
    and Senate vote on different bills (mixing creates block-diagonal NaN).

    Returns (filtered_matrix, manifest_dict) for reproducibility.
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]
    manifest: dict = {
        "chamber": chamber or "all",
        "minority_threshold": minority_threshold,
        "min_votes": min_votes,
    }

    # Restrict to chamber if specified — use slug prefix as chamber indicator
    # (sen_* = Senate, rep_* = House) and only include that chamber's rollcalls
    if chamber:
        chamber_vote_ids = set(
            rollcalls.filter(pl.col("chamber") == chamber)["vote_id"].to_list()
        )
        chamber_slugs_prefix = "sen_" if chamber == "Senate" else "rep_"
        vote_cols = [c for c in vote_cols if c in chamber_vote_ids]
        matrix = matrix.filter(
            pl.col(slug_col).str.starts_with(chamber_slugs_prefix)
        ).select([slug_col, *vote_cols])

    manifest["legislators_before"] = matrix.height
    manifest["votes_before"] = len(vote_cols)

    # Filter 1: Drop near-unanimous votes.
    # For each rollcall column, compute the minority fraction (smaller of
    # %Yea or %Nay among non-null values). If below threshold, drop it.
    contested_cols = []
    dropped_unanimous = []
    for col in vote_cols:
        series = matrix[col].drop_nulls()
        if series.len() == 0:
            dropped_unanimous.append(col)
            continue
        yea_frac = series.mean()
        minority_frac = min(yea_frac, 1 - yea_frac)
        if minority_frac < minority_threshold:
            dropped_unanimous.append(col)
        else:
            contested_cols.append(col)

    manifest["votes_dropped_unanimous"] = len(dropped_unanimous)
    manifest["votes_after_unanimous_filter"] = len(contested_cols)

    if not contested_cols:
        manifest["legislators_after"] = 0
        manifest["votes_after"] = 0
        return matrix.select([slug_col]).head(0), manifest

    filtered = matrix.select([slug_col, *contested_cols])

    # Filter 2: Drop low-participation legislators.
    # Count non-null (Yea or Nay) values per legislator across contested votes.
    non_null_counts = filtered.select(
        slug_col,
        pl.sum_horizontal(
            *[pl.col(c).is_not_null().cast(pl.Int32) for c in contested_cols]
        ).alias("n_votes"),
    )
    active_slugs = (
        non_null_counts.filter(pl.col("n_votes") >= min_votes)[slug_col].to_list()
    )
    dropped_legislators = manifest["legislators_before"] - len(active_slugs)
    filtered = filtered.filter(pl.col(slug_col).is_in(active_slugs))

    manifest["legislators_dropped_low_participation"] = dropped_legislators
    manifest["legislators_after"] = filtered.height
    manifest["votes_after"] = len(contested_cols)

    return filtered, manifest


def print_matrix_stats(
    full: pl.DataFrame, manifests: dict[str, dict]
) -> None:
    """Print vote matrix dimensions and filtering results."""
    print_header("VOTE MATRIX STATISTICS")
    slug_col = "legislator_slug"
    vote_cols = [c for c in full.columns if c != slug_col]

    # Sparsity: ~50% null is expected because House and Senate vote on
    # different bills, so every legislator has NaN for the other chamber's votes
    total_cells = full.height * len(vote_cols)
    null_cells = sum(full[c].null_count() for c in vote_cols)
    print(f"  Full matrix:  {full.height} legislators × {len(vote_cols)} votes")
    null_pct = 100 * null_cells / total_cells
    print(f"  Null cells:   {null_cells:,} / {total_cells:,} ({null_pct:.1f}%)")

    for label, m in manifests.items():
        print(f"\n  {label}:")
        print(f"    Before:      {m['legislators_before']} legislators × {m['votes_before']} votes")
        print(f"    Votes dropped (unanimous): {m['votes_dropped_unanimous']}")
        dropped = m["legislators_dropped_low_participation"]
        print(f"    Legislators dropped (< {m['min_votes']} votes): {dropped}")
        print(f"    After:       {m['legislators_after']} legislators × {m['votes_after']} votes")


# ── 3. Descriptive Statistics ────────────────────────────────────────────────


def print_descriptive_stats(rollcalls: pl.DataFrame) -> None:
    """Print vote type distribution, passage rates, margin stats, veto overrides."""
    print_header("DESCRIPTIVE STATISTICS")

    # Vote type distribution — shows the legislative process breakdown
    print("\n  Vote type distribution:")
    vt = rollcalls.group_by("vote_type").agg(pl.len()).sort("len", descending=True)
    for row in vt.iter_rows(named=True):
        pct = 100 * row["len"] / rollcalls.height
        print(f"    {row['vote_type']:30s}  {row['len']:>4}  ({pct:5.1f}%)")

    # Passage rate — expect high overall (most bills that reach a floor vote pass)
    has_result = rollcalls.filter(pl.col("passed").is_not_null())
    passed_count = has_result.filter(pl.col("passed") == True).height  # noqa: E712
    print(
        f"\n  Passage rate (overall): {passed_count}/{has_result.height}"
        f" ({100 * passed_count / has_result.height:.1f}%)"
    )

    print("\n  Passage rate by chamber:")
    for chamber in ["House", "Senate"]:
        ch = has_result.filter(pl.col("chamber") == chamber)
        ch_passed = ch.filter(pl.col("passed") == True).height  # noqa: E712
        if ch.height > 0:
            pct = 100 * ch_passed / ch.height
            print(f"    {chamber:8s}  {ch_passed}/{ch.height} ({pct:.1f}%)")

    print("\n  Passage rate by vote type:")
    for row in (
        has_result.group_by("vote_type")
        .agg(
            pl.len().alias("total"),
            (pl.col("passed") == True).sum().alias("passed_count"),  # noqa: E712
        )
        .sort("total", descending=True)
        .iter_rows(named=True)
    ):
        if row["total"] > 0:
            pct = 100 * row["passed_count"] / row["total"]
            print(
                f"    {row['vote_type']:30s}  "
                f"{row['passed_count']}/{row['total']} ({pct:5.1f}%)"
            )

    # Vote margin distribution — expect bimodal: peak near 1.0, spread for contested
    rc_with_margin = rollcalls.with_columns(
        (pl.col("yea_count") / pl.col("total_votes")).alias("yea_pct")
    )
    print("\n  Vote margin (Yea %):")
    print(f"    Mean:   {rc_with_margin['yea_pct'].mean():.3f}")
    print(f"    Median: {rc_with_margin['yea_pct'].median():.3f}")
    print(f"    Std:    {rc_with_margin['yea_pct'].std():.3f}")
    print(f"    Min:    {rc_with_margin['yea_pct'].min():.3f}")
    print(f"    Max:    {rc_with_margin['yea_pct'].max():.3f}")

    # Veto override votes — analytically rich (2/3 threshold, cross-party coalitions)
    veto = rollcalls.filter(pl.col("vote_type") == "Veto Override")
    print(f"\n  Veto override votes: {veto.height}")
    if veto.height > 0:
        veto_passed = veto.filter(pl.col("passed") == True).height  # noqa: E712
        print(f"    Override sustained: {veto_passed}/{veto.height}")


def classify_party_line(
    votes: pl.DataFrame, rollcalls: pl.DataFrame, legislators: pl.DataFrame
) -> pl.DataFrame:
    """Classify each rollcall as party-line, bipartisan, or mixed.

    Definition (using 90% threshold):
      - bipartisan: both parties >90% in the same direction (both Yea or both Nay)
      - party-line: parties >90% in opposite directions (R Yea + D Nay or vice versa)
      - mixed: everything else (significant cross-party voting)
    """
    vote_with_party = votes.join(
        legislators.select("slug", "party"),
        left_on="legislator_slug",
        right_on="slug",
    )

    # Only Yea/Nay — abstentions don't indicate party direction
    substantive = vote_with_party.filter(pl.col("vote").is_in(["Yea", "Nay"]))

    # Compute each party's Yea rate per rollcall
    party_agg = (
        substantive.group_by("vote_id", "party")
        .agg(
            (pl.col("vote") == "Yea").sum().alias("yea"),
            (pl.col("vote") == "Nay").sum().alias("nay"),
        )
        .with_columns(
            (pl.col("yea") / (pl.col("yea") + pl.col("nay"))).alias("yea_rate")
        )
    )

    # Pivot to get R and D yea_rates side by side
    pivoted = party_agg.pivot(on="party", index="vote_id", values="yea_rate")

    r_col = "Republican" if "Republican" in pivoted.columns else None
    d_col = "Democrat" if "Democrat" in pivoted.columns else None

    if r_col and d_col:
        classified = pivoted.with_columns(
            pl.when(
                # Both parties vote the same way (>90% agreement within each)
                ((pl.col(r_col) > 0.9) & (pl.col(d_col) > 0.9))
                | ((pl.col(r_col) < 0.1) & (pl.col(d_col) < 0.1))
            )
            .then(pl.lit("bipartisan"))
            .when(
                # Parties vote opposite ways
                ((pl.col(r_col) > 0.9) & (pl.col(d_col) < 0.1))
                | ((pl.col(r_col) < 0.1) & (pl.col(d_col) > 0.9))
            )
            .then(pl.lit("party-line"))
            .otherwise(pl.lit("mixed"))
            .alias("vote_alignment")
        )

        alignment_counts = (
            classified.group_by("vote_alignment")
            .agg(pl.len())
            .sort("len", descending=True)
        )
        print("\n  Vote alignment classification:")
        for row in alignment_counts.iter_rows(named=True):
            pct = 100 * row["len"] / classified.height
            print(f"    {row['vote_alignment']:14s}  {row['len']:>4}  ({pct:5.1f}%)")

        return classified.select("vote_id", "vote_alignment")

    return pl.DataFrame()


# ── 4. Participation Analysis ────────────────────────────────────────────────


def analyze_participation(
    votes: pl.DataFrame, rollcalls: pl.DataFrame, legislators: pl.DataFrame
) -> pl.DataFrame:
    """Compute per-legislator participation rates and print breakdown.

    Participation = (Yea + Nay votes) / (total rollcalls in their chamber).
    This measures substantive voting — abstentions and absences reduce the rate.
    Mid-session replacements will naturally have low rates (they weren't present
    for the full session), which is why the integrity check flags them separately.
    """
    print_header("PARTICIPATION ANALYSIS")

    # Count rollcalls per chamber — this is the denominator for participation
    chamber_rc_counts = dict(
        rollcalls.group_by("chamber")
        .agg(pl.len())
        .iter_rows()
    )

    # Count substantive (Yea or Nay) votes per legislator
    substantive = votes.filter(pl.col("vote").is_in(["Yea", "Nay"]))
    participation = (
        substantive.group_by("legislator_slug")
        .agg(pl.len().alias("substantive_votes"))
        .join(
            legislators.select("slug", "full_name", "chamber", "party"),
            left_on="legislator_slug",
            right_on="slug",
        )
    )

    # Compute participation rate = substantive votes / chamber rollcalls
    participation = participation.with_columns(
        pl.col("chamber")
        .replace_strict(chamber_rc_counts, default=0)
        .alias("chamber_rollcalls")
    ).with_columns(
        (pl.col("substantive_votes") / pl.col("chamber_rollcalls"))
        .alias("participation_rate")
    )

    participation = participation.sort("participation_rate", descending=True)

    # Summary stats
    rates = participation["participation_rate"]
    print(f"\n  Mean participation:   {rates.mean():.3f}")
    print(f"  Median participation: {rates.median():.3f}")
    print(f"  Min participation:    {rates.min():.3f}")
    print(f"  Max participation:    {rates.max():.3f}")

    # Top/bottom 10
    print("\n  Top 10 by participation:")
    for row in participation.head(10).iter_rows(named=True):
        print(
            f"    {row['full_name']:30s}  {row['party']:12s}  "
            f"{row['substantive_votes']}/{row['chamber_rollcalls']}  "
            f"({100 * row['participation_rate']:.1f}%)"
        )

    print("\n  Bottom 10 by participation:")
    for row in participation.tail(10).sort("participation_rate").iter_rows(named=True):
        print(
            f"    {row['full_name']:30s}  {row['party']:12s}  "
            f"{row['substantive_votes']}/{row['chamber_rollcalls']}  "
            f"({100 * row['participation_rate']:.1f}%)"
        )

    # Vote category breakdown by party — shows asymmetric absence patterns
    total_votes_per_party = (
        votes.join(
            legislators.select("slug", "party"),
            left_on="legislator_slug",
            right_on="slug",
        )
        .group_by("party", "vote")
        .agg(pl.len())
    )
    print("\n  Vote breakdown by party:")
    for party in ["Republican", "Democrat"]:
        party_data = total_votes_per_party.filter(pl.col("party") == party)
        total = party_data["len"].sum()
        print(f"\n    {party}:")
        for cat in VOTE_CATEGORIES:
            cat_row = party_data.filter(pl.col("vote") == cat)
            count = cat_row["len"].sum() if cat_row.height > 0 else 0
            pct = 100 * count / total if total > 0 else 0
            print(f"      {cat:28s}  {count:>6,}  ({pct:5.1f}%)")

    # "Present and Passing" is a deliberate abstention (distinct from absence).
    # In Kansas, it's rare (~22 instances) and almost always in the Senate.
    pnp = votes.filter(pl.col("vote") == "Present and Passing")
    print(f"\n  'Present and Passing' instances: {pnp.height}")
    if pnp.height > 0:
        pnp_detail = (
            pnp.join(
                legislators.select("slug", "full_name", "party"),
                left_on="legislator_slug",
                right_on="slug",
            )
            .select("full_name", "party", "bill_number", "vote_date", "chamber")
            .sort("vote_date", "full_name")
        )
        for row in pnp_detail.iter_rows(named=True):
            print(
                f"    {row['full_name']:30s}  {row['party']:12s}  "
                f"{row['bill_number']:10s}  {row['vote_date']}  {row['chamber']}"
            )

    return participation


# ── 5. Agreement Matrix & Heatmap ────────────────────────────────────────────


def compute_agreement_matrices(
    matrix: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise raw agreement and Cohen's Kappa from a binary vote matrix.

    For each pair of legislators (i, j):
      - Raw agreement = fraction of shared votes where both voted the same way
      - Cohen's Kappa = chance-corrected agreement: κ = (p_o - p_e) / (1 - p_e)
        where p_o = observed agreement, p_e = expected agreement by chance

    Kappa corrects for the ~82% Yea base rate in Kansas. Two legislators who
    both vote Yea 90% of the time would have ~82% raw agreement by chance alone.
    Kappa strips that out, revealing genuine ideological similarity.

    Pairs with fewer than MIN_SHARED_VOTES shared votes get NaN (unreliable).

    Returns (agreement_matrix, kappa_matrix), both shape (n, n).
    """
    slug_col = "legislator_slug"
    vote_cols = [c for c in matrix.columns if c != slug_col]

    # Convert to numpy for efficient vectorized pairwise computation
    arr = matrix.select(vote_cols).to_numpy().astype(np.float64)
    n = arr.shape[0]

    agreement = np.full((n, n), np.nan)
    kappa = np.full((n, n), np.nan)

    for i in range(n):
        agreement[i, i] = 1.0
        kappa[i, i] = 1.0
        for j in range(i + 1, n):
            # Only compare votes where BOTH legislators cast a Yea or Nay
            mask = ~np.isnan(arr[i]) & ~np.isnan(arr[j])
            shared = mask.sum()
            if shared < MIN_SHARED_VOTES:
                continue

            vi = arr[i, mask]
            vj = arr[j, mask]

            # Raw agreement: simple fraction of same-direction votes
            agree_frac = (vi == vj).sum() / shared
            agreement[i, j] = agree_frac
            agreement[j, i] = agree_frac

            # Cohen's Kappa: corrects for chance agreement.
            # p_e = P(both Yea by chance) + P(both Nay by chance)
            #      = p_i1 * p_j1 + (1-p_i1) * (1-p_j1)
            p_i1 = vi.mean()  # Legislator i's Yea rate
            p_j1 = vj.mean()  # Legislator j's Yea rate
            p_e = p_i1 * p_j1 + (1 - p_i1) * (1 - p_j1)
            if p_e < 1.0:
                k = (agree_frac - p_e) / (1 - p_e)
            else:
                k = 1.0  # Both vote identically — perfect agreement
            kappa[i, j] = k
            kappa[j, i] = k

    return agreement, kappa


def plot_agreement_heatmap(
    matrix: pl.DataFrame,
    agreement: np.ndarray,
    legislators: pl.DataFrame,
    chamber: str,
    out_dir: Path,
) -> None:
    """Generate a seaborn clustermap of pairwise agreement with party-colored annotations.

    Uses Ward linkage hierarchical clustering to order legislators by voting
    similarity. The dendrogram reveals coalition structure — expect a clear
    R/D split with visible intra-Republican factions in Kansas.
    """
    slug_col = "legislator_slug"
    slugs = matrix[slug_col].to_list()

    # Map each slug to its party color for the annotation sidebar
    slug_to_party = dict(
        legislators.select("slug", "party")
        .filter(pl.col("slug").is_in(slugs))
        .iter_rows()
    )
    parties = [slug_to_party.get(s, "Unknown") for s in slugs]
    row_colors = [PARTY_COLORS.get(p, "#999999") for p in parties]

    # Replace NaN with 0.5 (neutral) for clustering — NaN breaks Ward linkage
    agreement_clean = np.where(np.isnan(agreement), 0.5, agreement)

    # Seaborn clustermap requires pandas DataFrames
    import pandas as pd

    # Resolve slugs to human-readable names for axis labels
    labels = [
        legislators.filter(pl.col("slug") == s)["full_name"].to_list()[0]
        if legislators.filter(pl.col("slug") == s).height > 0
        else s
        for s in slugs
    ]
    df = pd.DataFrame(agreement_clean, index=labels, columns=labels)
    row_color_series = pd.Series(row_colors, index=labels, name="Party")

    # Scale figure size and font to the number of legislators
    n = len(slugs)
    size = max(8, n * 0.12)
    fontsize = max(3, min(7, 400 / n))

    g = sns.clustermap(
        df,
        method="ward",       # Ward linkage minimizes within-cluster variance
        metric="euclidean",
        row_colors=row_color_series,
        col_colors=row_color_series,
        cmap="RdYlGn",       # Red (disagree) → Yellow → Green (agree)
        vmin=0,
        vmax=1,
        figsize=(size, size),
        xticklabels=True,
        yticklabels=True,
        linewidths=0,
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), fontsize=fontsize, rotation=90
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=fontsize
    )
    g.fig.suptitle(
        f"{chamber} — Pairwise Agreement (Contested Votes)", y=1.01, fontsize=14
    )

    # Party color legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    g.ax_heatmap.legend(
        handles=legend_elements, loc="lower left", bbox_to_anchor=(0, -0.15),
        ncol=2, frameon=False, fontsize=9,
    )

    path = out_dir / f"agreement_heatmap_{chamber.lower()}.png"
    g.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  Saved: {path.name}")


# ── 6. Visualization Plots ──────────────────────────────────────────────────


def plot_vote_type_distribution(rollcalls: pl.DataFrame, out_dir: Path) -> None:
    """Horizontal bar chart of rollcall types (Final Action, Emergency, etc.)."""
    vt = rollcalls.group_by("vote_type").agg(pl.len()).sort("len", descending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    types = vt["vote_type"].to_list()
    counts = vt["len"].to_list()
    bars = ax.barh(types[::-1], counts[::-1], color="#4C72B0")
    ax.set_xlabel("Number of Roll Calls")
    ax.set_title("Roll Call Vote Types")
    for bar, count in zip(bars, counts[::-1]):
        ax.text(
            bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_dir / "vote_type_distribution.png")


def plot_vote_margin_distribution(rollcalls: pl.DataFrame, out_dir: Path) -> None:
    """Histogram of Yea% per rollcall with 50% and 2/3 threshold lines.

    Expected shape: bimodal — large peak near 1.0 (near-unanimous) and a
    spread of contested votes. The gap around 0.5 reflects that close votes
    are relatively rare in a supermajority legislature.
    """
    yea_pcts = (rollcalls["yea_count"] / rollcalls["total_votes"]).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(yea_pcts, bins=50, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.axvline(
        2 / 3, color="orange", linestyle="--", alpha=0.5,
        label="2/3 threshold (veto)",
    )
    ax.set_xlabel("Yea Percentage")
    ax.set_ylabel("Number of Roll Calls")
    ax.set_title("Vote Margin Distribution")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_dir / "vote_margin_distribution.png")


def plot_temporal_activity(rollcalls: pl.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart of monthly rollcall counts by chamber.

    Expect activity bursts near session deadlines (turnaround, sine die).
    """
    monthly = (
        rollcalls.with_columns(
            pl.col("vote_datetime").str.slice(0, 7).alias("month")
        )
        .group_by("month", "chamber")
        .agg(pl.len())
        .sort("month")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    months = sorted(monthly["month"].unique().to_list())
    house_counts = []
    senate_counts = []
    for m in months:
        h = monthly.filter(
            (pl.col("month") == m) & (pl.col("chamber") == "House")
        )
        s = monthly.filter(
            (pl.col("month") == m) & (pl.col("chamber") == "Senate")
        )
        house_counts.append(h["len"].sum() if h.height > 0 else 0)
        senate_counts.append(s["len"].sum() if s.height > 0 else 0)

    x = np.arange(len(months))
    width = 0.35
    ax.bar(x - width / 2, house_counts, width, label="House", color="#4C72B0")
    ax.bar(x + width / 2, senate_counts, width, label="Senate", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_ylabel("Number of Roll Calls")
    ax.set_title("Monthly Roll Call Activity by Chamber")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_dir / "temporal_activity.png")


def plot_party_vote_breakdown(
    votes: pl.DataFrame, legislators: pl.DataFrame, out_dir: Path
) -> None:
    """Side-by-side horizontal bar charts of vote categories by party.

    Shows how Rs and Ds differ in voting patterns — expect Rs to have higher
    Yea% (supermajority often passes their bills) and Ds to have higher Nay%.
    """
    vote_with_party = votes.join(
        legislators.select("slug", "party"),
        left_on="legislator_slug",
        right_on="slug",
    )
    breakdown = vote_with_party.group_by("party", "vote").agg(pl.len())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cat_colors = {
        "Yea": "#2ecc71",
        "Nay": "#e74c3c",
        "Present and Passing": "#f39c12",
        "Absent and Not Voting": "#95a5a6",
        "Not Voting": "#bdc3c7",
    }

    for ax, party in zip(axes, ["Republican", "Democrat"]):
        party_data = breakdown.filter(pl.col("party") == party)
        total = party_data["len"].sum()
        cats = []
        fracs = []
        colors = []
        for cat in VOTE_CATEGORIES:
            row = party_data.filter(pl.col("vote") == cat)
            count = row["len"].sum() if row.height > 0 else 0
            cats.append(cat)
            fracs.append(count / total if total > 0 else 0)
            colors.append(cat_colors[cat])

        bars = ax.barh(
            cats[::-1], [f * 100 for f in fracs[::-1]], color=colors[::-1]
        )
        ax.set_xlabel("Percentage of All Votes")
        ax.set_title(f"{party}")
        ax.set_xlim(0, 100)
        for bar, frac in zip(bars, fracs[::-1]):
            if frac > 0.01:
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{frac * 100:.1f}%", va="center", fontsize=9,
                )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Vote Category Breakdown by Party", fontsize=14)
    fig.tight_layout()
    save_fig(fig, out_dir / "party_vote_breakdown.png")


def plot_participation_rates(
    participation: pl.DataFrame, chamber: str, out_dir: Path
) -> None:
    """Horizontal bar chart of per-legislator participation rates, colored by party.

    Dashed lines at 95% (normal threshold) and 90% (notable threshold) help
    identify legislators with unusually low participation.
    """
    chamber_data = participation.filter(pl.col("chamber") == chamber).sort(
        "participation_rate", descending=False
    )
    names = chamber_data["full_name"].to_list()
    rates = (chamber_data["participation_rate"] * 100).to_list()
    parties = chamber_data["party"].to_list()
    colors = [PARTY_COLORS.get(p, "#999999") for p in parties]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.22)))
    ax.barh(names, rates, color=colors, height=0.8)
    ax.set_xlabel("Participation Rate (%)")
    ax.set_title(f"{chamber} — Legislator Participation Rates")
    ax.set_xlim(0, 105)

    # Reference thresholds from doc 03_EDA_vote_participation.md
    ax.axvline(95, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(90, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)

    ax.tick_params(axis="y", labelsize=max(4, min(7, 300 / len(names))))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PARTY_COLORS["Republican"], label="Republican"),
        Patch(facecolor=PARTY_COLORS["Democrat"], label="Democrat"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    save_fig(fig, out_dir / f"participation_rates_{chamber.lower()}.png")


# ── 7. Filtering Manifest ───────────────────────────────────────────────────


def save_filtering_manifest(manifests: dict, out_dir: Path) -> None:
    """Save all filtering decisions and integrity findings as JSON.

    This manifest is required by the analytic-workflow rules: every analysis
    must document which votes were dropped, which legislators excluded, and why.
    Downstream scripts can load this to reproduce exact filtering.
    """
    path = out_dir / "filtering_manifest.json"
    with open(path, "w") as f:
        json.dump(manifests, f, indent=2, default=str)
    print(f"  Saved: {path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    session_slug = args.session.replace("-", "_")

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(f"data/ks_{session_slug}")

    with RunContext(
        session=args.session,
        analysis_name="eda",
        params=vars(args),
        primer=EDA_PRIMER,
    ) as ctx:
        print(f"KS Legislature EDA — Session {args.session}")
        print(f"Data:   {data_dir}")
        print(f"Output: {ctx.run_dir}")

        # ── 1. Load data ──
        votes, rollcalls, legislators = load_data(data_dir)
        print_session_summary(votes, rollcalls, legislators)

        # ── 1b. Data integrity ──
        # Run structural checks before any analysis — fail-fast if data is bad.
        integrity_findings = check_data_integrity(votes, rollcalls, legislators)

        # ── 1c. Statistical quality ──
        # Rice cohesion and perfect partisan checks gate downstream modeling.
        stat_findings = check_statistical_quality(votes, rollcalls, legislators)

        # ── 2. Build vote matrix ──
        full_matrix = build_vote_matrix(votes)

        # Filter separately per chamber — the analytic-workflow rules require this
        # because House and Senate vote on different bills.
        manifests: dict[str, dict] = {}

        house_filtered, house_manifest = filter_vote_matrix(
            full_matrix, rollcalls, chamber="House"
        )
        manifests["House"] = house_manifest

        senate_filtered, senate_manifest = filter_vote_matrix(
            full_matrix, rollcalls, chamber="Senate"
        )
        manifests["Senate"] = senate_manifest

        # All-chambers pass for reference (not used in per-chamber analysis)
        _, all_manifest = filter_vote_matrix(full_matrix, rollcalls, chamber=None)
        manifests["All"] = all_manifest

        print_matrix_stats(full_matrix, manifests)

        # Save matrices as parquet — fast, typed, preserves nulls for downstream use
        full_matrix.write_parquet(ctx.data_dir / "vote_matrix_full.parquet")
        house_filtered.write_parquet(ctx.data_dir / "vote_matrix_house_filtered.parquet")
        senate_filtered.write_parquet(ctx.data_dir / "vote_matrix_senate_filtered.parquet")
        print("\n  Saved: vote_matrix_full.parquet")
        print("  Saved: vote_matrix_house_filtered.parquet")
        print("  Saved: vote_matrix_senate_filtered.parquet")

        # ── 3. Descriptive stats ──
        print_descriptive_stats(rollcalls)
        classify_party_line(votes, rollcalls, legislators)

        # ── 4. Participation analysis ──
        participation = analyze_participation(votes, rollcalls, legislators)

        # ── 5. Agreement matrices & heatmaps ──
        print_header("AGREEMENT MATRICES")

        for label, filtered in [("House", house_filtered), ("Senate", senate_filtered)]:
            if filtered.height < 2:
                print(f"  Skipping {label}: too few legislators after filtering")
                continue

            agreement, kappa = compute_agreement_matrices(filtered)

            # Save as parquet — convert numpy matrix to polars DataFrame
            slug_col = "legislator_slug"
            slugs = filtered[slug_col].to_list()

            def numpy_matrix_to_polars(
                mat: np.ndarray, slugs: list[str]
            ) -> pl.DataFrame:
                """Convert a square numpy matrix to a polars DataFrame with slug labels."""
                cols = {slug: mat[:, i].tolist() for i, slug in enumerate(slugs)}
                cols["legislator_slug"] = slugs
                return pl.DataFrame(cols).select(["legislator_slug", *slugs])

            agree_pl = numpy_matrix_to_polars(agreement, slugs)
            kappa_pl = numpy_matrix_to_polars(kappa, slugs)
            agree_pl.write_parquet(
                ctx.data_dir / f"agreement_raw_{label.lower()}.parquet"
            )
            kappa_pl.write_parquet(
                ctx.data_dir / f"agreement_kappa_{label.lower()}.parquet"
            )
            print(f"  Saved: agreement_raw_{label.lower()}.parquet")
            print(f"  Saved: agreement_kappa_{label.lower()}.parquet")

            # Plot heatmap
            plot_agreement_heatmap(
                filtered, agreement, legislators, label, ctx.plots_dir
            )

        # ── 6. Plots ──
        print_header("GENERATING PLOTS")
        plot_vote_type_distribution(rollcalls, ctx.plots_dir)
        plot_vote_margin_distribution(rollcalls, ctx.plots_dir)
        plot_temporal_activity(rollcalls, ctx.plots_dir)
        plot_party_vote_breakdown(votes, legislators, ctx.plots_dir)
        plot_participation_rates(participation, "House", ctx.plots_dir)
        plot_participation_rates(participation, "Senate", ctx.plots_dir)

        # ── 7. Filtering manifest ──
        # Bundle all findings into a single JSON for reproducibility
        print_header("FILTERING MANIFEST")
        manifests["data_integrity"] = integrity_findings
        manifests["statistical_quality"] = stat_findings
        save_filtering_manifest(manifests, ctx.run_dir)

        print_header("DONE")
        print(f"  All outputs in: {ctx.run_dir}")
        print(f"  Parquet files:  {len(list(ctx.data_dir.glob('*.parquet')))}")
        print(f"  PNG plots:      {len(list(ctx.plots_dir.glob('*.png')))}")
        print(f"  JSON manifests: {len(list(ctx.run_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
