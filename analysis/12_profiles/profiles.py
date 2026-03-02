"""Kansas Legislature — Legislator Profile Deep-Dives

Deep-dive profiles of individually notable legislators: scorecards, bill-type
breakdowns, defection analysis, voting neighbors, and surprising votes.

Reads upstream parquets (via synthesis.py infrastructure) and raw CSVs. Produces
a standalone HTML report with 5 plots per legislator.

Usage:
  uv run python analysis/profiles.py [--session 2025-26] [--slugs s1,s2] [--names "Name1"]

Outputs (in results/<session>/profiles/<date>/):
  - data/:   ProfileTarget list, per-legislator data as parquet
  - plots/:  5 PNGs per legislator
  - filtering_manifest.json, run_info.json, run_log.txt
  - profiles_report.html
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from analysis.profiles_data import (
        SCORECARD_METRICS,
        BillTypeBreakdown,
        ProfileTarget,
        build_full_voting_record,
        build_scorecard,
        compute_bill_type_breakdown,
        compute_sponsorship_stats,
        find_defection_bills,
        find_legislator_surprising_votes,
        find_voting_neighbors,
        gather_profile_targets,
        resolve_names,
    )
except ModuleNotFoundError:
    from profiles_data import (  # type: ignore[no-redef]
        SCORECARD_METRICS,
        BillTypeBreakdown,
        ProfileTarget,
        build_full_voting_record,
        build_scorecard,
        compute_bill_type_breakdown,
        compute_sponsorship_stats,
        find_defection_bills,
        find_legislator_surprising_votes,
        find_voting_neighbors,
        gather_profile_targets,
        resolve_names,
    )

try:
    from analysis.profiles_report import build_profiles_report
except ModuleNotFoundError:
    from profiles_report import build_profiles_report  # type: ignore[no-redef]

try:
    from analysis.run_context import RunContext, resolve_upstream_dir
except ModuleNotFoundError:
    from run_context import RunContext, resolve_upstream_dir  # type: ignore[no-redef]

try:
    from analysis.synthesis_data import (
        _read_parquet_safe,
        build_legislator_df,
        load_all_upstream,
    )
except ModuleNotFoundError:
    from synthesis_data import (  # type: ignore[no-redef]
        _read_parquet_safe,
        build_legislator_df,
        load_all_upstream,
    )

# ── Primer ───────────────────────────────────────────────────────────────────

PROFILES_PRIMER = """\
# Legislator Profiles

## Purpose

Deep-dive profiles of individually notable legislators, combining findings from
eight analysis phases into per-legislator scorecards, bill-type breakdowns,
defection analysis, voting neighbors, and surprising votes.

## Method

No new statistical computation. Joins upstream parquet outputs and raw vote CSVs
to build per-legislator views. Notable legislators are detected automatically
from data (mavericks, bridge-builders, metric paradoxes) via synthesis_detect.

## Inputs

- Upstream parquets from all 8 phases (via load_all_upstream)
- IRT bill_params (beta_mean for bill discrimination tiers)
- Raw votes CSV (for vote-level defection analysis)
- Raw rollcalls CSV (for bill metadata)

## Outputs

- `profiles_report.html` — Self-contained HTML with intro + per-legislator sections
- `plots/` — 5 PNGs per legislator: scorecard, bill type, position, defections, neighbors
- `data/` — Per-legislator data as parquet

## Interpretation Guide

Each legislator profile includes:
1. **Scorecard** — All metrics at a glance, compared to party average
2. **Bill Type Breakdown** — How they vote on partisan vs routine bills
3. **Position in Context** — Where they fall among same-party members
4. **Defection Votes** — Specific bills where they broke from party
5. **Surprising Votes** — Votes the prediction model got most wrong
6. **Voting Neighbors** — Who they vote most/least like

## Selecting Specific Legislators

Use `--names` to request profiles by legislator name (case-insensitive, supports
last name only or full name). Ambiguous matches (e.g., two "Carpenter"s) include
all matches. Use `--slugs` for exact legislator slug lookup. Both can be combined.

## Caveats

- All statistics inherit upstream filtering (near-unanimous votes removed, < 20 votes excluded)
- "Defection" means disagreeing with the party majority on a specific vote
- Legislators are detected dynamically; different sessions surface different individuals
"""

# ── Constants ────────────────────────────────────────────────────────────────

PARTY_COLORS = {"Republican": "#E81B23", "Democrat": "#0015BC", "Independent": "#999999"}
PARTY_COLORS_LIGHT = {"Republican": "#F5A0A5", "Democrat": "#8090E0", "Independent": "#CCCCCC"}


# ── Vote Data Prep ───────────────────────────────────────────────────────────


def prep_votes_long(votes_raw: pl.DataFrame) -> pl.DataFrame:
    """Convert raw votes CSV to long-form binary vote DataFrame.

    Returns DataFrame with columns: legislator_slug, vote_id, vote_binary, chamber.
    """
    return (
        votes_raw.filter(pl.col("vote").is_in(["Yea", "Nay"]))
        .with_columns(
            pl.when(pl.col("vote") == "Yea").then(1).otherwise(0).alias("vote_binary"),
            pl.col("chamber").str.to_lowercase().alias("chamber"),
        )
        .select("legislator_slug", "vote_id", "vote_binary", "chamber")
    )


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_enhanced_scorecard(
    scorecard: dict,
    target: ProfileTarget,
    plots_dir: Path,
) -> Path | None:
    """Horizontal bar chart with legislator values + party average markers."""
    # Collect metrics that are present
    metrics = []
    for col, label, fmt in SCORECARD_METRICS:
        if col in scorecard:
            metrics.append((col, label, fmt, scorecard[col], scorecard.get(f"{col}_party_avg")))

    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(metrics) * 0.7 + 1)))
    party_color = PARTY_COLORS.get(target.party, "#666666")
    light_color = PARTY_COLORS_LIGHT.get(target.party, "#cccccc")

    labels = [m[1] for m in metrics]
    values = [m[3] for m in metrics]
    party_avgs = [m[4] for m in metrics]

    y_pos = np.arange(len(metrics))

    # Background bars (full 0-1 range)
    ax.barh(y_pos, [1.0] * len(metrics), color=light_color, alpha=0.2, height=0.55, zorder=0)

    # Bars for legislator values
    ax.barh(y_pos, values, color=party_color, alpha=0.8, height=0.55, edgecolor="white")

    # Party average markers
    for i, avg in enumerate(party_avgs):
        if avg is not None:
            ax.plot(avg, i, "|", color="#555555", markersize=20, markeredgewidth=2)
            ax.plot(avg, i, "s", color="#555555", markersize=4, alpha=0.7)

    ax.set_xlim(0, 1.15)

    # Value labels
    for i, (col, label, fmt, val, _avg) in enumerate(metrics):
        formatted = f"{val:{fmt}}"
        if val > 0.85:
            ax.text(
                val - 0.02,
                i,
                formatted,
                va="center",
                ha="right",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                val + 0.02,
                i,
                formatted,
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Score (0 to 1)", fontsize=10)
    ax.invert_yaxis()

    ax.set_title(
        f"{target.title} — At a Glance",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    # Legend
    ax.text(
        0.98,
        0.02,
        "Gray marker = party average",
        transform=ax.transAxes,
        fontsize=9,
        fontstyle="italic",
        color="#555555",
        ha="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0f0f0", "edgecolor": "#cccccc"},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    out = plots_dir / f"scorecard_{slug_short}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out.name}")
    return out


def plot_bill_type_bars(
    breakdown: BillTypeBreakdown,
    target: ProfileTarget,
    plots_dir: Path,
) -> Path | None:
    """Grouped bar chart: legislator vs party on high/low discrimination bills."""
    fig, ax = plt.subplots(figsize=(10, 6))
    party_color = PARTY_COLORS.get(target.party, "#666666")

    categories = [
        f"Partisan Bills\n(n={breakdown.high_disc_n})",
        f"Routine Bills\n(n={breakdown.low_disc_n})",
    ]
    legislator_vals = [breakdown.high_disc_yea_rate, breakdown.low_disc_yea_rate]
    party_vals = [breakdown.party_high_disc_yea_rate, breakdown.party_low_disc_yea_rate]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, legislator_vals, width, label=target.full_name, color=party_color, alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2,
        party_vals,
        width,
        label=f"{target.party} Average",
        color="#999999",
        alpha=0.6,
    )

    # Value labels
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.0%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Yea Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"How {target.full_name} Votes by Bill Type",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)

    # Annotation if there's a notable gap
    gap = abs(breakdown.high_disc_yea_rate - breakdown.party_high_disc_yea_rate)
    if gap > 0.1:
        is_less = breakdown.high_disc_yea_rate < breakdown.party_high_disc_yea_rate
        direction = "less" if is_less else "more"
        ax.text(
            0.5,
            0.02,
            f"{target.full_name.split()[-1]} votes Yea {gap:.0%} {direction} often "
            "than their party on partisan bills",
            transform=ax.transAxes,
            fontsize=9,
            fontstyle="italic",
            color="#555555",
            ha="center",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f0f0f0", "edgecolor": "#cccccc"},
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    out = plots_dir / f"bill_type_{slug_short}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out.name}")
    return out


def plot_position_in_context(
    target: ProfileTarget,
    leg_df: pl.DataFrame,
    plots_dir: Path,
) -> Path | None:
    """Forest plot of same-party IRT ideal points with target highlighted."""
    if "xi_mean" not in leg_df.columns:
        return None

    party_df = leg_df.filter(pl.col("party") == target.party).sort("xi_mean")
    if party_df.height == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, max(5, party_df.height * 0.25 + 1)))
    party_color = PARTY_COLORS.get(target.party, "#666666")
    light_color = PARTY_COLORS_LIGHT.get(target.party, "#cccccc")

    names = party_df["full_name"].to_list()
    xi_means = party_df["xi_mean"].to_numpy()
    slugs = party_df["legislator_slug"].to_list()

    has_hdi = "xi_hdi_2.5" in party_df.columns and "xi_hdi_97.5" in party_df.columns

    y_pos = np.arange(len(names))

    for i, slug in enumerate(slugs):
        is_target = slug == target.slug
        color = "#FFD700" if is_target else light_color
        marker = "D" if is_target else "o"
        size = 80 if is_target else 30
        zorder = 10 if is_target else 3

        if is_target:
            ax.axhspan(i - 0.4, i + 0.4, color="#FFFDE7", alpha=0.6, zorder=1)

        # HDI error bars
        if has_hdi:
            row = party_df.row(i, named=True)
            lo = row.get("xi_hdi_2.5", xi_means[i])
            hi = row.get("xi_hdi_97.5", xi_means[i])
            ax.hlines(
                i,
                lo,
                hi,
                color=party_color if not is_target else "#B8860B",
                linewidth=1.5 if is_target else 0.8,
                zorder=zorder - 1,
            )

        ax.scatter(
            xi_means[i],
            i,
            c=color,
            s=size,
            marker=marker,
            edgecolors=party_color,
            linewidth=1.5,
            zorder=zorder,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)

    ax.set_xlabel("IRT Ideology (Liberal <-> Conservative)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Where {target.full_name} Stands Among {target.party}s ({target.chamber.title()})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    out = plots_dir / f"position_{slug_short}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out.name}")
    return out


def plot_defection_chart(
    defections: pl.DataFrame,
    target: ProfileTarget,
    plots_dir: Path,
) -> Path | None:
    """Horizontal bar chart of party Yea % on defection bills with legislator marker."""
    if defections.height == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, max(5, defections.height * 0.5 + 1)))

    bills = defections["bill_number"].to_list()
    party_pcts = defections["party_yea_pct"].to_numpy() / 100.0
    leg_votes = defections["legislator_vote"].to_list()

    y_pos = np.arange(len(bills))

    # Bars: party yea %
    ax.barh(
        y_pos,
        party_pcts,
        color="#cccccc",
        alpha=0.7,
        height=0.5,
        edgecolor="white",
        label="Party Yea %",
    )

    # Legislator vote markers
    party_color = PARTY_COLORS.get(target.party, "#666666")
    for i, vote in enumerate(leg_votes):
        x = 1.0 if vote == "Yea" else 0.0
        ax.scatter(
            x, i, c=party_color, s=100, marker="D", edgecolors="white", linewidth=1.5, zorder=5
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bills, fontsize=9)
    ax.set_xlim(-0.05, 1.15)
    ax.set_xlabel("Party Yea Rate", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    ax.set_title(
        f"Key Votes Where {target.full_name} Broke Ranks",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.text(
        0.98,
        0.98,
        f"Diamond = {target.full_name.split()[-1]}'s vote",
        transform=ax.transAxes,
        fontsize=8,
        fontstyle="italic",
        color="#555555",
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0f0f0", "edgecolor": "#cccccc"},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    out = plots_dir / f"defections_{slug_short}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out.name}")
    return out


def plot_neighbor_chart(
    neighbors: dict,
    target: ProfileTarget,
    plots_dir: Path,
) -> Path | None:
    """Two-panel horizontal bar chart: closest + most different."""
    closest = neighbors.get("closest", [])
    most_diff = neighbors.get("most_different", [])

    if not closest and not most_diff:
        return None

    fig, (ax_close, ax_diff) = plt.subplots(1, 2, figsize=(14, max(5, len(closest) * 0.6 + 1)))

    # Closest
    if closest:
        names = [f"{c['full_name']} ({c['party'][0]})" for c in closest]
        vals = [c["agreement"] for c in closest]
        colors = [PARTY_COLORS.get(c["party"], "#666666") for c in closest]
        y_pos = np.arange(len(names))
        ax_close.barh(y_pos, vals, color=colors, alpha=0.7, height=0.5)
        for i, v in enumerate(vals):
            ax_close.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, fontweight="bold")
        ax_close.set_yticks(y_pos)
        ax_close.set_yticklabels(names, fontsize=9)
        ax_close.set_xlim(0, 1.15)
        ax_close.set_title("Most Similar", fontsize=12, fontweight="bold")
        ax_close.invert_yaxis()
    ax_close.spines["top"].set_visible(False)
    ax_close.spines["right"].set_visible(False)
    ax_close.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax_close.set_axisbelow(True)

    # Most different
    if most_diff:
        names = [f"{c['full_name']} ({c['party'][0]})" for c in most_diff]
        vals = [c["agreement"] for c in most_diff]
        colors = [PARTY_COLORS.get(c["party"], "#666666") for c in most_diff]
        y_pos = np.arange(len(names))
        ax_diff.barh(y_pos, vals, color=colors, alpha=0.7, height=0.5)
        for i, v in enumerate(vals):
            ax_diff.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, fontweight="bold")
        ax_diff.set_yticks(y_pos)
        ax_diff.set_yticklabels(names, fontsize=9)
        ax_diff.set_xlim(0, 1.15)
        ax_diff.set_title("Most Different", fontsize=12, fontweight="bold")
        ax_diff.invert_yaxis()
    ax_diff.spines["top"].set_visible(False)
    ax_diff.spines["right"].set_visible(False)
    ax_diff.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax_diff.set_axisbelow(True)

    fig.suptitle(
        f"Who Does {target.full_name} Vote Like?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    out = plots_dir / f"neighbors_{slug_short}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved {out.name}")
    return out


# ── Argument Parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session",
        default="2025-26",
        help="Session identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--slugs",
        default=None,
        help="Comma-separated legislator slugs for extra profiles (e.g., rep_alcala_john_1)",
    )
    parser.add_argument(
        "--names",
        default=None,
        help=(
            "Comma-separated legislator names to profile "
            '(e.g., "Masterson,Blake Carpenter,Dietrich")'
        ),
    )
    parser.add_argument("--run-id", default=None, help="Run ID for grouped pipeline output")
    parser.add_argument(
        "--full-record",
        action="store_true",
        default=None,
        help="Include complete voting record per legislator (auto-enabled with --names)",
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def _resolve_name_args(
    names_arg: str | None,
    leg_dfs: dict[str, pl.DataFrame],
) -> list[str]:
    """Resolve --names argument to slugs, printing status for each query.

    Returns list of resolved slugs (may be empty).
    """
    if not names_arg:
        return []

    queries = [q.strip() for q in names_arg.split(",") if q.strip()]
    if not queries:
        return []

    matches = resolve_names(queries, leg_dfs)
    resolved_slugs: list[str] = []

    for match in matches:
        if match.status == "ok":
            m = match.matches[0]
            print(
                f"  Resolved '{match.query}' -> "
                f"{m['full_name']} ({m['party'][0]}-{m['district']}, {m['chamber']})"
            )
            resolved_slugs.append(m["slug"])
        elif match.status == "ambiguous":
            print(
                f"  NOTE: '{match.query}' matched {len(match.matches)} legislators (all included):"
            )
            for m in match.matches:
                print(f"    - {m['full_name']} ({m['party'][0]}-{m['district']}, {m['chamber']})")
                resolved_slugs.append(m["slug"])
        else:
            print(f"  WARNING: No legislator found matching '{match.query}'")

    return resolved_slugs


def main() -> None:
    args = parse_args()
    extra_slugs = args.slugs.split(",") if args.slugs else None

    # Default --full-record to on when --names is used
    include_full_record = args.full_record
    if include_full_record is None:
        include_full_record = args.names is not None

    with RunContext(
        session=args.session,
        analysis_name="12_profiles",
        params=vars(args),
        primer=PROFILES_PRIMER,
        run_id=args.run_id,
    ) as ctx:
        from tallgrass.session import STATE_DIR

        results_base = Path("results") / STATE_DIR / ctx.session
        print(f"Loading upstream data from {results_base}")

        # ── Load upstream ─────────────────────────────────────────────
        upstream = load_all_upstream(results_base, run_id=args.run_id)

        # ── Build legislator DataFrames ───────────────────────────────
        leg_dfs: dict[str, pl.DataFrame] = {}
        for chamber in ("house", "senate"):
            print(f"\nBuilding unified DataFrame: {chamber}")
            df = build_legislator_df(upstream, chamber)
            leg_dfs[chamber] = df
            print(f"  {chamber}: {df.height} legislators, {df.width} columns")

        # ── Resolve --names to slugs ──────────────────────────────────
        if args.names:
            print("\nResolving legislator names...")
            name_slugs = _resolve_name_args(args.names, leg_dfs)
            if extra_slugs:
                extra_slugs.extend(name_slugs)
            elif name_slugs:
                extra_slugs = name_slugs

        # ── Gather profile targets ────────────────────────────────────
        print("\nSelecting legislators for profiling...")
        targets = gather_profile_targets(leg_dfs, extra_slugs=extra_slugs)
        for t in targets:
            print(f"  {t.title} — {t.role}")

        if not targets:
            print("No legislators selected for profiling.")
            return

        # ── Load raw votes and rollcalls ──────────────────────────────
        data_dir = Path("data") / STATE_DIR / ctx.session
        votes_path = next(data_dir.glob("*_votes.csv"), None)
        rollcalls_path = next(data_dir.glob("*_rollcalls.csv"), None)

        if votes_path is None:
            print(f"ERROR: no votes CSV found in {data_dir}")
            return
        if rollcalls_path is None:
            print(f"ERROR: no rollcalls CSV found in {data_dir}")
            return

        print(f"\nLoading raw votes: {votes_path.name}")
        votes_raw = pl.read_csv(votes_path)
        votes_long = prep_votes_long(votes_raw)
        print(f"  {votes_long.height} Yea/Nay votes")

        print(f"Loading rollcalls: {rollcalls_path.name}")
        rollcalls = pl.read_csv(rollcalls_path)
        print(f"  {rollcalls.height} roll calls")

        # ── Load IRT bill_params ──────────────────────────────────────
        bill_params: dict[str, pl.DataFrame] = {}
        irt_phase_dir = resolve_upstream_dir("04_irt", results_base, args.run_id)
        for chamber in ("house", "senate"):
            bp = _read_parquet_safe(irt_phase_dir / "data" / f"bill_params_{chamber}.parquet")
            if bp is not None:
                bill_params[chamber] = bp

        # ── Profile each legislator ───────────────────────────────────
        all_data: dict[str, dict] = {}

        for target in targets:
            print(f"\n  Profiling: {target.full_name} ({target.role})")
            chamber = target.chamber
            chamber_df = leg_dfs[chamber]
            chamber_votes = votes_long.filter(pl.col("chamber") == chamber)

            # Party slugs for comparison
            party_slugs = chamber_df.filter(pl.col("party") == target.party)[
                "legislator_slug"
            ].to_list()

            # Scorecard
            scorecard = build_scorecard(chamber_df, target.slug)

            # Bill type breakdown
            bp = bill_params.get(chamber)
            breakdown = None
            if bp is not None:
                breakdown = compute_bill_type_breakdown(
                    target.slug, bp, chamber_votes, target.party, party_slugs
                )

            # Defection bills
            defections = find_defection_bills(
                target.slug, chamber_votes, rollcalls, target.party, party_slugs
            )

            # Sponsorship stats
            sponsorship = compute_sponsorship_stats(target.slug, rollcalls)
            if sponsorship is not None and sponsorship.height > 0:
                n_primary = sponsorship.filter(pl.col("is_primary")).height
                print(f"    Sponsored {sponsorship.height} bills ({n_primary} as primary)")

            # Voting neighbors
            neighbors = find_voting_neighbors(target.slug, chamber_votes, chamber_df)

            # Surprising votes
            sv = upstream.get(chamber, {}).get("surprising_votes")
            surprising = find_legislator_surprising_votes(target.slug, sv)

            # Full voting record (opt-in, auto-enabled with --names)
            full_record = None
            if include_full_record:
                full_record = build_full_voting_record(
                    target.slug,
                    chamber_votes,
                    rollcalls,
                    target.party,
                    party_slugs,
                )
                if full_record.height > 0:
                    print(f"    Full record: {full_record.height} votes")

            all_data[target.slug] = {
                "scorecard": scorecard,
                "breakdown": breakdown,
                "defections": defections,
                "sponsorship": sponsorship,
                "neighbors": neighbors,
                "surprising": surprising,
                "full_record": full_record,
            }

            # ── Plots ─────────────────────────────────────────────────
            if scorecard:
                plot_enhanced_scorecard(scorecard, target, ctx.plots_dir)

            if breakdown:
                plot_bill_type_bars(breakdown, target, ctx.plots_dir)

            plot_position_in_context(target, chamber_df, ctx.plots_dir)

            if defections.height > 0:
                plot_defection_chart(defections, target, ctx.plots_dir)

            if neighbors:
                plot_neighbor_chart(neighbors, target, ctx.plots_dir)

        # ── Build Report ──────────────────────────────────────────────
        print("\nBuilding profiles report...")
        ctx.report.title = f"Kansas Legislature {ctx.session} — Legislator Profiles"

        build_profiles_report(
            ctx.report,
            targets=targets,
            all_data=all_data,
            plots_dir=ctx.plots_dir,
            session=ctx.session,
        )

        # ── Manifest ──────────────────────────────────────────────────
        manifest = {
            "analysis": "profiles",
            "n_targets": len(targets),
            "targets": [
                {"slug": t.slug, "name": t.full_name, "role": t.role, "chamber": t.chamber}
                for t in targets
            ],
            "new_plots": sorted(str(p.name) for p in ctx.plots_dir.iterdir()),
            "report_sections": len(ctx.report._sections),
        }
        manifest_path = ctx.run_dir / "filtering_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"\nProfiles complete. Results: {ctx.run_dir}")


if __name__ == "__main__":
    main()
