"""Indices-specific HTML report builder.

Builds ~35 sections (tables, figures, and text) for the classical indices report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from indices.py):
    from analysis.indices_report import build_indices_report
    build_indices_report(ctx.report, results=results, ...)
"""

from pathlib import Path

import polars as pl

try:
    from analysis.report import (
        FigureSection,
        InteractiveSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )

# Import constants from indices.py (single source of truth).
# Circular-import safe: falls back to direct sibling import, then literals.
try:
    from analysis.indices import (
        CO_DEFECTION_MIN,
        ENP_MULTIPARTY_THRESHOLD,
        MAVERICK_WEIGHT_FLOOR,
        MIN_PARTY_VOTERS,
        PARTY_VOTE_THRESHOLD,
        RICE_FRACTURE_THRESHOLD,
        ROLLING_WINDOW,
        TOP_DEFECTORS_N,
    )
except ModuleNotFoundError, ImportError:
    try:
        from indices import (  # type: ignore[no-redef]
            CO_DEFECTION_MIN,
            ENP_MULTIPARTY_THRESHOLD,
            MAVERICK_WEIGHT_FLOOR,
            MIN_PARTY_VOTERS,
            PARTY_VOTE_THRESHOLD,
            RICE_FRACTURE_THRESHOLD,
            ROLLING_WINDOW,
            TOP_DEFECTORS_N,
        )
    except ModuleNotFoundError, ImportError:
        # Last resort: inline values (must match indices.py)
        PARTY_VOTE_THRESHOLD = 0.50  # type: ignore[assignment]
        RICE_FRACTURE_THRESHOLD = 0.50  # type: ignore[assignment]
        MIN_PARTY_VOTERS = 2  # type: ignore[assignment]
        MAVERICK_WEIGHT_FLOOR = 0.01  # type: ignore[assignment]
        CO_DEFECTION_MIN = 3  # type: ignore[assignment]
        ENP_MULTIPARTY_THRESHOLD = 2.5  # type: ignore[assignment]
        ROLLING_WINDOW = 15  # type: ignore[assignment]
        TOP_DEFECTORS_N = 20  # type: ignore[assignment]


def build_indices_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    skip_cross_ref: bool = False,
    skip_sensitivity: bool = False,
) -> None:
    """Build the full indices HTML report by adding sections to the ReportBuilder."""
    # Key findings
    findings = _generate_indices_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_data_summary(report, results)
    _add_how_to_read(report)
    _add_party_vote_summary(report, results)

    # Rice sections per chamber
    for chamber, result in results.items():
        _add_rice_summary_table(report, result, chamber)
        _add_rice_distribution_figure(report, plots_dir, chamber)
        _add_rice_over_time_figure(report, plots_dir, chamber)
        _add_rice_by_vote_type_figure(report, plots_dir, chamber)
        _add_rice_by_vote_type_table(report, result, chamber)
        _add_fractured_votes_table(report, result, chamber)

    _add_rice_interpretation(report)

    # Unity sections per chamber
    for chamber, result in results.items():
        _add_unity_summary_table(report, result, chamber)
        _add_unity_ranking_figure(report, plots_dir, chamber)
        _add_unity_full_table(report, result, chamber)

    _add_unity_interpretation(report)

    # ENP sections per chamber
    for chamber, result in results.items():
        _add_enp_seats_table(report, result, chamber)
        _add_enp_distribution_figure(report, plots_dir, chamber)
        _add_enp_over_time_figure(report, plots_dir, chamber)

    _add_enp_interpretation(report)

    # Maverick sections per chamber
    for chamber, result in results.items():
        _add_maverick_top_table(report, result, chamber)
        _add_maverick_landscape_figure(report, plots_dir, chamber)
        _add_maverick_full_table(report, result, chamber)
        _add_co_defection_figure(report, plots_dir, chamber)

    _add_maverick_interpretation(report)

    # Bipartisanship index
    for chamber, result in results.items():
        _add_bipartisanship_table(report, result, chamber)
        _add_bipartisanship_vs_maverick_figure(report, plots_dir, chamber)
    _add_bipartisanship_interpretation(report)

    # Plus-minus
    for chamber, result in results.items():
        _add_plus_minus_figure(report, plots_dir, chamber)
        _add_plus_minus_table(report, result, chamber)

    # Veto overrides
    _add_veto_override_table(report, results)

    # Cross-referencing
    if not skip_cross_ref:
        for chamber in results:
            _add_unity_vs_irt_figure(report, plots_dir, chamber)
            _add_unity_vs_irt_interactive(report, plots_dir, chamber)
        _add_cross_ref_table(report, results)
        _add_cross_ref_interpretation(report)

    # Sensitivity
    if not skip_sensitivity:
        _add_sensitivity_table(report, results)

    # Parameters
    _add_analysis_parameters(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_data_summary(report: ReportBuilder, results: dict[str, dict]) -> None:
    """Table: Data dimensions per chamber."""
    rows = []
    for chamber, result in results.items():
        rows.append(
            {
                "Chamber": chamber,
                "Roll Calls": result["n_rollcalls"],
                "Party Votes": result["n_party_votes"],
                "% Party Votes": (
                    result["n_party_votes"] / result["n_total_votes"] * 100
                    if result["n_total_votes"] > 0
                    else 0
                ),
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Classical Indices Data Summary",
        subtitle="Roll call counts and party vote identification per chamber",
        number_formats={"% Party Votes": ".1f"},
        source_note=(
            "Party vote: majority of Rs opposes majority of Ds, "
            f"each party has >= {MIN_PARTY_VOTERS} Yea+Nay voters."
        ),
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_how_to_read(report: ReportBuilder) -> None:
    """Text block: how to interpret this report."""
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html=(
                "<p>This report presents four classical legislative indices:</p>"
                "<ul>"
                "<li><strong>Rice Index</strong> — How often does each party vote together? "
                "Computed per roll call per party. A Rice of 1.0 means the party was "
                "completely unanimous; 0.0 means a perfect 50-50 split.</li>"
                "<li><strong>Party Unity</strong> — How loyal is each legislator to their "
                "party on contested votes? The Congressional Quarterly (CQ) standard metric. "
                "Unity = 1.0 means the legislator always votes with their party majority.</li>"
                "<li><strong>Effective Number of Parties (ENP)</strong> — Does Kansas behave "
                "like a two-party system, or does one party sometimes fracture into multiple "
                "blocs? ENP > 2.5 indicates multiparty-like behavior on that vote.</li>"
                "<li><strong>Maverick Scores</strong> — Who breaks ranks, and do their "
                "defections happen on close votes (strategic) or blowout votes "
                "(performative)?</li>"
                "</ul>"
                "<p>Unlike the model-based approaches (IRT, clustering), these indices are "
                "descriptive and assumption-light. They don't require priors or convergence — "
                "they count votes and compute ratios.</p>"
            ),
        )
    )


def _add_party_vote_summary(report: ReportBuilder, results: dict[str, dict]) -> None:
    """Table: Party vote breakdown by vote type."""
    rows = []
    for chamber, result in results.items():
        pv = result.get("party_votes")
        if pv is None or pv.height == 0:
            continue
        # Count party votes by motion type
        if "motion" in pv.columns:
            by_type = (
                pv.filter(pl.col("is_party_vote"))
                .group_by("motion")
                .agg(pl.len().alias("n_party_votes"))
                .sort("n_party_votes", descending=True)
            )
            for row in by_type.iter_rows(named=True):
                rows.append(
                    {
                        "Chamber": chamber,
                        "Motion Type": row["motion"],
                        "Party Votes": row["n_party_votes"],
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Party Votes by Motion Type",
        subtitle="Number of party votes broken down by the type of motion",
        source_note="A 'party vote' is one where majority-R opposes majority-D.",
    )
    report.add(
        TableSection(
            id="party-vote-summary",
            title="Party Votes by Motion Type",
            html=html,
        )
    )


def _add_rice_summary_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Rice Index summary statistics per party."""
    rice_summary = result.get("rice_summary", {})
    if not rice_summary:
        return

    rows = []
    for party, stats in rice_summary.items():
        rows.append(
            {
                "Party": party,
                "N Votes": stats["n_votes"],
                "Mean Rice": stats["mean"],
                "Median Rice": stats["median"],
                "Std Dev": stats["std"],
                "Min Rice": stats["min"],
                "% Perfect Unity": stats["pct_perfect_unity"],
                "% Fractured": stats["pct_fractured"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Rice Index Summary",
        subtitle="Per-party cohesion statistics across all roll calls",
        column_labels={
            "Party": "Party",
            "N Votes": "N Votes",
            "Mean Rice": "Mean",
            "Median Rice": "Median",
            "Std Dev": "Std Dev",
            "Min Rice": "Min",
            "% Perfect Unity": "% Unanimous",
            "% Fractured": "% Fractured",
        },
        number_formats={
            "Mean Rice": ".3f",
            "Median Rice": ".3f",
            "Std Dev": ".3f",
            "Min Rice": ".3f",
            "% Perfect Unity": ".1f",
            "% Fractured": ".1f",
        },
        source_note=f"Fractured = Rice < {RICE_FRACTURE_THRESHOLD}.",
    )
    report.add(
        TableSection(
            id=f"rice-summary-{chamber.lower()}",
            title=f"{chamber} Rice Index Summary",
            html=html,
        )
    )


def _add_rice_distribution_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"rice_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-rice-dist-{chamber.lower()}",
                f"{chamber} Rice Index Distribution",
                path,
                caption=(
                    "Distribution of Rice Index per party. Higher = more unified. "
                    "Dashed line = party mean."
                ),
                alt_text=(
                    f"Histogram of Rice Index values by party for {chamber}. "
                    f"Most votes cluster near 1.0, indicating high party cohesion."
                ),
            )
        )


def _add_rice_over_time_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"rice_over_time_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-rice-time-{chamber.lower()}",
                f"{chamber} Party Cohesion Over Time",
                path,
                caption=(
                    f"Rolling {ROLLING_WINDOW}-vote average of Rice Index through the session. "
                    "Drops indicate periods when a party was internally divided."
                ),
                alt_text=(
                    f"Line chart of rolling {ROLLING_WINDOW}-vote Rice Index over the "
                    f"{chamber} session. Dips reveal periods of intra-party division."
                ),
            )
        )


def _add_rice_by_vote_type_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"rice_by_vote_type_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-rice-type-{chamber.lower()}",
                f"{chamber} Rice by Vote Type",
                path,
                caption="Mean Rice Index by motion type and party.",
                alt_text=(
                    f"Grouped bar chart of mean Rice Index by motion type and party "
                    f"for {chamber}. Shows which types of votes produce the most party splits."
                ),
            )
        )


def _add_rice_by_vote_type_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Mean Rice by vote type."""
    rice_by_type = result.get("rice_by_type")
    if rice_by_type is None or rice_by_type.height == 0:
        return

    display = rice_by_type.select("motion", "party", "mean_rice", "median_rice", "n_votes")
    html = make_gt(
        display,
        title=f"{chamber} — Rice Index by Motion Type",
        column_labels={
            "motion": "Motion Type",
            "party": "Party",
            "mean_rice": "Mean Rice",
            "median_rice": "Median Rice",
            "n_votes": "N Votes",
        },
        number_formats={"mean_rice": ".3f", "median_rice": ".3f"},
    )
    report.add(
        TableSection(
            id=f"rice-by-type-{chamber.lower()}",
            title=f"{chamber} Rice by Motion Type",
            html=html,
        )
    )


def _add_fractured_votes_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Fractured votes where majority party Rice < threshold."""
    fractured = result.get("fractured_votes")
    if fractured is None or fractured.height == 0:
        return

    display_cols = ["vote_id", "bill_number", "motion", "vote_date", "rice_index"]
    available = [c for c in display_cols if c in fractured.columns]
    display = fractured.select(available).head(50)

    html = make_gt(
        display,
        title=f"{chamber} — Fractured Votes (Majority Party Rice < {RICE_FRACTURE_THRESHOLD})",
        subtitle=f"{fractured.height} votes where the majority party was more divided than united",
        column_labels={
            "vote_id": "Vote ID",
            "bill_number": "Bill",
            "motion": "Motion",
            "vote_date": "Date",
            "rice_index": "Rice Index",
        },
        number_formats={"rice_index": ".3f"},
        source_note=(
            f"Showing up to 50 of {fractured.height} fractured votes, "
            f"sorted by Rice (lowest first)."
            if fractured.height > 50
            else None
        ),
    )
    report.add(
        TableSection(
            id=f"fractured-{chamber.lower()}",
            title=f"{chamber} Fractured Votes",
            html=html,
        )
    )


def _add_rice_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting Rice Index."""
    report.add(
        TextSection(
            id="rice-interpretation",
            title="Interpreting the Rice Index",
            html=(
                "<p>The Rice Index measures <strong>within-party cohesion</strong> on a single "
                "roll call. It answers: 'How unified was this party on this vote?'</p>"
                "<p>Key benchmarks:</p>"
                "<ul>"
                "<li><strong>Rice = 1.0:</strong> Every party member who voted chose the same "
                "side. The party was completely unified.</li>"
                "<li><strong>Rice = 0.80:</strong> About a 90-10 split (e.g., 90% Yea, 10% Nay). "
                "Strong cohesion with a few dissenters.</li>"
                "<li><strong>Rice = 0.50:</strong> A 75-25 split. The party is more unified than "
                "divided, but a significant minority broke ranks.</li>"
                "<li><strong>Rice &lt; 0.50:</strong> The party is more divided than united. "
                "More than 25% of the party voted against the majority position.</li>"
                "</ul>"
                "<p>In a state like Kansas with a Republican supermajority, the Republican "
                "mean Rice is the key indicator. When it drops, it usually means moderate and "
                "conservative Republicans are splitting on a bill.</p>"
            ),
        )
    )


def _add_unity_summary_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Unity summary statistics per party."""
    unity = result.get("unity")
    if unity is None or unity.height == 0:
        return

    rows = []
    for party in ["Republican", "Democrat"]:
        party_unity = unity.filter(pl.col("party") == party)
        if party_unity.height == 0:
            continue
        vals = party_unity["unity_score"]
        rows.append(
            {
                "Party": party,
                "N Legislators": party_unity.height,
                "Mean Unity": float(vals.mean()),
                "Median Unity": float(vals.median()),
                "Std Dev": float(vals.std()),
                "Min Unity": float(vals.min()),
                "Max Unity": float(vals.max()),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Party Unity Summary",
        subtitle="CQ-standard: % of party votes where legislator sides with party majority",
        number_formats={
            "Mean Unity": ".3f",
            "Median Unity": ".3f",
            "Std Dev": ".3f",
            "Min Unity": ".3f",
            "Max Unity": ".3f",
        },
    )
    report.add(
        TableSection(
            id=f"unity-summary-{chamber.lower()}",
            title=f"{chamber} Unity Summary",
            html=html,
        )
    )


def _add_unity_ranking_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"party_unity_ranking_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-unity-rank-{chamber.lower()}",
                f"{chamber} Party Unity Ranking",
                path,
                caption=(
                    "All legislators ranked by party unity score. Red = Republican, "
                    "Blue = Democrat. Lower scores = more independent from party."
                ),
                alt_text=(
                    f"Horizontal bar chart ranking all {chamber} legislators by party unity "
                    f"score. Color-coded by party with low-unity legislators at the top."
                ),
            )
        )


def _add_unity_full_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: ALL legislators with unity scores — never truncated."""
    unity = result.get("unity")
    if unity is None or unity.height == 0:
        return

    display_cols = [
        "full_name",
        "party",
        "district",
        "unity_score",
        "maverick_rate",
        "votes_with_party",
        "party_votes_present",
    ]
    available = [c for c in display_cols if c in unity.columns]
    display = unity.sort("unity_score").select(available)

    html = make_interactive_table(
        display,
        title=f"{chamber} — Party Unity Scores ({display.height} legislators)",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "unity_score": "Unity Score",
            "maverick_rate": "Maverick Rate",
            "votes_with_party": "Votes With Party",
            "party_votes_present": "Party Votes Present",
        },
        number_formats={"unity_score": ".3f", "maverick_rate": ".3f"},
        caption=(
            "Unity = votes with party majority / party votes present. "
            "Maverick = 1 - unity. Higher unity = more loyal to party."
        ),
    )
    report.add(
        InteractiveTableSection(
            id=f"unity-full-{chamber.lower()}",
            title=f"{chamber} Full Unity Table",
            html=html,
        )
    )


def _add_unity_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting unity scores."""
    report.add(
        TextSection(
            id="unity-interpretation",
            title="Interpreting Party Unity",
            html=(
                "<p><strong>Party unity</strong> is the CQ (Congressional Quarterly) standard "
                "metric for measuring how loyal a legislator is to their party. It is only "
                "computed on 'party votes' — roll calls where the majority of Republicans "
                "opposes the majority of Democrats.</p>"
                "<p>This is intentionally different from the 'party loyalty' metric used in "
                "the clustering phase, which counts agreement with the party median on "
                "'contested' votes (where >= 10% of the party dissents). The CQ standard "
                "uses a simpler threshold: a party vote is any vote where the two parties' "
                "majorities are on opposite sides.</p>"
                "<p>A legislator with <strong>unity = 0.85</strong> voted with their party "
                "majority on 85% of party votes. The remaining 15% were defections — votes "
                "where they sided with the opposing party's majority.</p>"
                "<p>In Kansas, where Republicans hold a supermajority, the most interesting "
                "variation is among Republicans. Democrats tend to have high unity because "
                "they are a small, cohesive minority.</p>"
            ),
        )
    )


def _add_enp_seats_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Seat-based ENP."""
    enp_seats = result.get("enp_seats")
    if enp_seats is None or enp_seats.height == 0:
        return

    display_cols = ["party", "seats", "seat_share", "enp_seats"]
    available = [c for c in display_cols if c in enp_seats.columns]
    display = enp_seats.select(available)

    html = make_gt(
        display,
        title=f"{chamber} — Effective Number of Parties (Seats)",
        subtitle="Laakso-Taagepera index based on seat distribution",
        column_labels={
            "party": "Party",
            "seats": "Seats",
            "seat_share": "Seat Share",
            "enp_seats": "ENP (Seats)",
        },
        number_formats={"seat_share": ".3f", "enp_seats": ".3f"},
        source_note="ENP = 1/sum(p_i^2). ENP = 2.0 means two equal-sized parties.",
    )
    report.add(
        TableSection(
            id=f"enp-seats-{chamber.lower()}",
            title=f"{chamber} ENP (Seats)",
            html=html,
        )
    )


def _add_enp_distribution_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"enp_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-enp-dist-{chamber.lower()}",
                f"{chamber} ENP Distribution",
                path,
                caption=(
                    "Distribution of per-vote ENP. Votes to the right of the red line "
                    "show multiparty-like behavior (one party is splitting)."
                ),
                alt_text=(
                    f"Histogram of per-vote Effective Number of Parties for {chamber}. "
                    f"A red threshold line marks multiparty-like behavior "
                    f"above {ENP_MULTIPARTY_THRESHOLD}."
                ),
            )
        )


def _add_enp_over_time_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"enp_over_time_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-enp-time-{chamber.lower()}",
                f"{chamber} ENP Over Time",
                path,
                caption=(
                    f"Rolling {ROLLING_WINDOW}-vote average of ENP through the session. "
                    "Rises indicate periods when parties were splitting internally."
                ),
                alt_text=(
                    f"Line chart of rolling {ROLLING_WINDOW}-vote ENP over the {chamber} "
                    f"session. Spikes indicate votes where party blocs fragmented."
                ),
            )
        )


def _add_enp_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting ENP."""
    report.add(
        TextSection(
            id="enp-interpretation",
            title="Interpreting the Effective Number of Parties",
            html=(
                "<p>The <strong>Effective Number of Parties (ENP)</strong> measures how "
                "fragmented voting blocs are on a given roll call. It uses the Laakso-Taagepera "
                "index, which accounts for both the number and size of blocs.</p>"
                "<p>For per-vote ENP, blocs are defined as (party, vote direction) pairs. "
                "If all Republicans vote Yea and all Democrats vote Nay, there are 2 blocs "
                "and ENP = 2.0. If Republicans split 60-40 with all Democrats voting Nay, "
                "there are 3 blocs and ENP > 2.0.</p>"
                "<p>Key benchmarks:</p>"
                "<ul>"
                "<li><strong>ENP ~ 1.0:</strong> One bloc dominates — likely a near-unanimous "
                "vote.</li>"
                "<li><strong>ENP ~ 2.0:</strong> Two roughly equal blocs — a clean party-line "
                "vote.</li>"
                f"<li><strong>ENP > {ENP_MULTIPARTY_THRESHOLD}:</strong> Three or more "
                "meaningful blocs — one or both parties are splitting.</li>"
                "</ul>"
                "<p>The seat-based ENP is static (it just reflects the party balance). The "
                "per-vote ENP is dynamic and reveals which specific votes fracture the "
                "party system.</p>"
            ),
        )
    )


def _add_maverick_top_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Top 10 mavericks."""
    maverick = result.get("maverick")
    if maverick is None or maverick.height == 0:
        return

    top = maverick.sort("maverick_rate", descending=True).head(10)
    display_cols = [
        "full_name",
        "party",
        "maverick_rate",
        "weighted_maverick",
        "n_defections",
        "n_party_votes",
        "loyalty_zscore",
    ]
    available = [c for c in display_cols if c in top.columns]
    display = top.select(available)

    html = make_gt(
        display,
        title=f"{chamber} — Top 10 Mavericks",
        subtitle="Legislators who most often break ranks on party votes",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "maverick_rate": "Maverick Rate",
            "weighted_maverick": "Weighted Maverick",
            "n_defections": "Defections",
            "n_party_votes": "Party Votes",
            "loyalty_zscore": "Loyalty Z-Score",
        },
        number_formats={
            "maverick_rate": ".3f",
            "weighted_maverick": ".3f",
            "loyalty_zscore": ".2f",
        },
        source_note=(
            "Weighted maverick weights defections by chamber vote closeness. "
            "Loyalty Z-score is within-party (negative = less loyal than average)."
        ),
    )
    report.add(
        TableSection(
            id=f"maverick-top-{chamber.lower()}",
            title=f"{chamber} Top Mavericks",
            html=html,
        )
    )


def _add_maverick_landscape_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"maverick_landscape_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-maverick-land-{chamber.lower()}",
                f"{chamber} Maverick Landscape",
                path,
                caption=(
                    "Unweighted (x) vs weighted (y) maverick rate. Above the "
                    "diagonal = strategic; below = performative."
                ),
                alt_text=(
                    f"Scatter plot of unweighted vs weighted maverick rate for {chamber} "
                    f"legislators. Points above the diagonal are strategic defectors; "
                    f"below are performative."
                ),
            )
        )


def _add_maverick_full_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: ALL legislators with maverick scores — never truncated."""
    maverick = result.get("maverick")
    if maverick is None or maverick.height == 0:
        return

    display_cols = [
        "full_name",
        "party",
        "unity_score",
        "maverick_rate",
        "weighted_maverick",
        "n_defections",
        "n_party_votes",
    ]
    available = [c for c in display_cols if c in maverick.columns]
    display = maverick.sort("maverick_rate", descending=True).select(available)

    html = make_interactive_table(
        display,
        title=(
            f"{chamber} — Maverick Scores ({display.height} legislators, sorted by maverick rate)"
        ),
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "unity_score": "Unity",
            "maverick_rate": "Maverick",
            "weighted_maverick": "Weighted",
            "n_defections": "Defections",
            "n_party_votes": "Party Votes",
        },
        number_formats={
            "unity_score": ".3f",
            "maverick_rate": ".3f",
            "weighted_maverick": ".3f",
        },
    )
    report.add(
        InteractiveTableSection(
            id=f"maverick-full-{chamber.lower()}",
            title=f"{chamber} Full Maverick Table",
            html=html,
        )
    )


def _add_co_defection_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"co_defection_heatmap_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-codefect-{chamber.lower()}",
                f"{chamber} Co-Defection Heatmap",
                path,
                caption=(
                    f"Number of shared defections among top majority-party defectors. "
                    f"Minimum {CO_DEFECTION_MIN} shared defections to appear. "
                    "High counts suggest informal alliances."
                ),
                alt_text=(
                    f"Heatmap of shared defections among top {chamber} majority-party "
                    f"defectors. Darker cells indicate legislators who frequently break "
                    f"ranks together."
                ),
            )
        )


def _add_maverick_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting maverick scores."""
    report.add(
        TextSection(
            id="maverick-interpretation",
            title="Interpreting Maverick Scores",
            html=(
                "<p>The <strong>maverick rate</strong> is simply 1 - unity: the fraction of "
                "party votes where a legislator voted against their party's majority.</p>"
                "<p>The <strong>weighted maverick</strong> adjusts for the importance of each "
                "defection. Defecting on a vote that passed 90-10 is less consequential than "
                "defecting on a 51-49 vote. The weight is inversely proportional to the chamber "
                "vote margin: closer votes get higher weight.</p>"
                "<p>The <strong>maverick landscape</strong> plot compares these two scores:</p>"
                "<ul>"
                "<li><strong>Above the diagonal</strong> (weighted &gt; unweighted): "
                "This legislator's defections tend to happen on close votes. They are a "
                "strategic maverick — their votes could swing outcomes.</li>"
                "<li><strong>Below the diagonal</strong> (weighted &lt; unweighted): "
                "Defections happen "
                "on blowout votes. This is performative independence — they break ranks when it "
                "doesn't matter.</li>"
                "</ul>"
                "<p>The <strong>co-defection heatmap</strong> reveals which legislators break "
                "ranks together. If two legislators share many defections, they may have an "
                "informal alliance or be driven by the same policy concerns.</p>"
            ),
        )
    )


def _add_veto_override_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Veto override Rice by party."""
    rows = []
    for chamber, result in results.items():
        veto = result.get("veto_overrides", {})
        if veto.get("skipped"):
            rows.append(
                {
                    "Chamber": chamber,
                    "N Override Votes": veto.get("n_overrides", 0),
                    "Party": "N/A",
                    "Mean Override Rice": None,
                    "Note": "Insufficient override votes",
                }
            )
            continue

        rice_by_party = veto.get("rice_by_party", {})
        for party, rice_val in rice_by_party.items():
            rows.append(
                {
                    "Chamber": chamber,
                    "N Override Votes": veto["n_overrides"],
                    "Party": party,
                    "Mean Override Rice": rice_val,
                    "Note": "Near-perfect = strictly party-line" if rice_val > 0.95 else "",
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Veto Override Indices",
        subtitle="Rice Index for veto override votes by party and chamber",
        number_formats={"Mean Override Rice": ".3f"},
        source_note="Veto overrides require 2/3 supermajority.",
    )
    report.add(
        TableSection(
            id="veto-overrides",
            title="Veto Override Indices",
            html=html,
        )
    )


def _add_unity_vs_irt_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"unity_vs_irt_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-unity-irt-{chamber.lower()}",
                f"{chamber} Unity vs IRT",
                path,
                caption=(
                    "Party unity (CQ standard) vs IRT ideal point. Centrist legislators "
                    "tend to have lower unity because they are cross-pressured."
                ),
                alt_text=(
                    f"Scatter plot of party unity vs IRT ideal point for {chamber}. "
                    f"Centrist legislators near zero tend to have lower unity scores."
                ),
            )
        )


def _add_unity_vs_irt_interactive(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Embed Plotly interactive unity vs IRT scatter."""
    path = plots_dir / f"unity_vs_irt_interactive_{chamber.lower()}.html"
    if not path.exists():
        return
    html = path.read_text()
    report.add(
        InteractiveSection(
            id=f"interactive-unity-irt-{chamber.lower()}",
            title=f"{chamber} Unity vs IRT (Interactive)",
            html=html,
            caption="Hover over points to see legislator details.",
            aria_label=(
                f"Interactive scatter plot of party unity vs IRT ideal point for "
                f"{chamber} legislators. Hover over points to see individual details."
            ),
        )
    )


def _add_cross_ref_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Cross-reference correlations."""
    rows = []
    for chamber, result in results.items():
        cross_ref = result.get("cross_ref", {})
        if not cross_ref:
            continue
        for key, val in cross_ref.items():
            if key.endswith("_rho"):
                label = key.replace("_rho", "").replace("_", " ").title()
                pval = cross_ref.get(key.replace("_rho", "_pval"), None)
                rows.append(
                    {
                        "Chamber": chamber,
                        "Comparison": label,
                        "Spearman Rho": val,
                        "P-Value": pval,
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Cross-Reference Correlations",
        subtitle="Spearman rank correlation between indices and upstream results",
        number_formats={"Spearman Rho": ".4f", "P-Value": ".4g"},
        source_note="Rho > 0.7 = strong correlation. P < 0.05 = statistically significant.",
    )
    report.add(
        TableSection(
            id="cross-ref",
            title="Cross-Reference Correlations",
            html=html,
        )
    )


def _add_cross_ref_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting cross-references."""
    report.add(
        TextSection(
            id="cross-ref-interpretation",
            title="Interpreting Cross-References",
            html=(
                "<p>These correlations test whether the classical indices agree with the "
                "model-based measures from earlier phases:</p>"
                "<ul>"
                "<li><strong>Unity vs IRT ideal point:</strong> Do centrist legislators "
                "(near zero on the IRT scale) have lower party unity? If positive, extremists "
                "are more loyal — they have nowhere else to go.</li>"
                "<li><strong>Maverick vs betweenness centrality:</strong> Do mavericks serve "
                "as bridges in the voting network? A positive correlation would mean "
                "independent voters connect otherwise-separate voting blocs.</li>"
                "<li><strong>CQ unity vs clustering loyalty:</strong> These are conceptually "
                "similar but use different definitions. High correlation confirms they "
                "identify the same legislators as loyal/disloyal. Divergences point to "
                "cases where the definition matters.</li>"
                "</ul>"
            ),
        )
    )


def _add_sensitivity_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Sensitivity analysis — unity on all vs EDA-filtered votes."""
    rows = []
    for chamber, result in results.items():
        sensitivity = result.get("sensitivity", {})
        if not sensitivity:
            continue
        rows.append(
            {
                "Chamber": chamber,
                "Spearman Rho": sensitivity.get("spearman_rho"),
                "Max Rank Change": sensitivity.get("max_rank_change"),
                "Max Changer": sensitivity.get("max_rank_changer", ""),
                "N Primary": sensitivity.get("n_primary"),
                "N Filtered": sensitivity.get("n_filtered"),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Sensitivity Analysis — All Votes vs EDA-Filtered",
        subtitle="Comparing party unity on all votes vs near-unanimous votes removed",
        number_formats={"Spearman Rho": ".4f"},
        source_note=(
            "High rho (> 0.95) means filtering near-unanimous votes barely changes rankings. "
            "Max rank change shows the largest shift for any individual legislator."
        ),
    )
    report.add(
        TableSection(
            id="sensitivity",
            title="Sensitivity Analysis",
            html=html,
        )
    )


def _add_bipartisanship_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: ALL legislators with bipartisanship index — never truncated."""
    bpi = result.get("bipartisanship")
    if bpi is None or bpi.height == 0:
        return

    display_cols = [
        "full_name",
        "party",
        "district",
        "bpi_score",
        "votes_with_opposition",
        "party_votes_present",
    ]
    available = [c for c in display_cols if c in bpi.columns]
    display = bpi.sort("bpi_score", descending=True).select(available)

    html = make_interactive_table(
        display,
        title=f"{chamber} — Bipartisanship Index ({display.height} legislators)",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "bpi_score": "BPI",
            "votes_with_opposition": "Votes w/ Opposition",
            "party_votes_present": "Party Votes Present",
        },
        number_formats={"bpi_score": ".3f"},
        caption=(
            "BPI = votes with opposing party majority / party votes present. "
            "Higher BPI = more bipartisan. Distinct from maverick (which measures "
            "voting against own party)."
        ),
    )
    report.add(
        InteractiveTableSection(
            id=f"bipartisanship-{chamber.lower()}",
            title=f"{chamber} Bipartisanship Index",
            html=html,
        )
    )


def _add_bipartisanship_vs_maverick_figure(
    report: ReportBuilder, plots_dir: Path, chamber: str
) -> None:
    """Figure: BPI vs maverick scatter for a chamber."""
    fname = f"bpi_vs_maverick_{chamber.lower()}.png"
    path = plots_dir / fname
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            id=f"bpi-vs-maverick-{chamber.lower()}",
            title=f"{chamber} — BPI vs Maverick Rate",
            path=path,
            caption=(
                f"Bipartisanship Index vs maverick rate for {chamber} legislators. "
                "Points above the diagonal vote with the opposing party more than they "
                "vote against their own."
            ),
            alt_text=(
                f"Scatter plot comparing Bipartisanship Index to maverick rate for "
                f"{chamber}. Most points cluster along the diagonal, indicating the "
                f"two metrics are highly correlated."
            ),
        )
    )


def _add_bipartisanship_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting the bipartisanship index."""
    report.add(
        TextSection(
            id="bipartisanship-interpretation",
            title="Interpreting the Bipartisanship Index",
            html=(
                "<p>The <strong>Bipartisanship Index (BPI)</strong> measures how often a "
                "legislator votes with the <em>opposing</em> party's majority on party votes. "
                "It is inspired by the Lugar Center's Bipartisan Index for the U.S. Congress.</p>"
                "<p><strong>BPI vs Maverick:</strong> These are conceptually distinct:</p>"
                "<ul>"
                "<li><strong>Maverick rate</strong> = voting against your <em>own</em> party. "
                "A maverick might abstain or vote 'Present' rather than cross the aisle.</li>"
                "<li><strong>BPI</strong> = voting <em>with</em> the opposing party. This "
                "specifically measures cross-partisan cooperation.</li>"
                "</ul>"
                "<p>In practice, BPI and maverick scores are highly correlated because most "
                "party votes are binary (Yea/Nay). But they can diverge when legislators "
                "strategically abstain on party votes rather than crossing the aisle.</p>"
                "<p>High BPI legislators are the most likely bridge-builders — they actively "
                "vote with the other side, not just against their own.</p>"
            ),
        )
    )


def _add_plus_minus_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Figure: plus-minus dumbbell chart."""
    path = plots_dir / f"plus_minus_{chamber.lower()}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            id=f"plus-minus-{chamber.lower()}",
            title=f"{chamber} Plus-Minus",
            path=path,
            caption=(
                f"Dumbbell chart showing actual unity (colored dot) vs party mean "
                f"unity (gray mark) for {chamber} legislators. Distance right of "
                "the gray mark = more partisan than average; left = less partisan."
            ),
            alt_text=(
                f"Dumbbell chart of plus-minus scores for {chamber} legislators. "
                f"Each row shows actual unity vs party mean, revealing who is more "
                f"or less partisan than their party average."
            ),
        )
    )


def _add_plus_minus_table(report: ReportBuilder, result: dict, chamber: str) -> None:
    """Table: plus-minus scores for all legislators."""
    pm = result.get("plus_minus")
    if pm is None or pm.is_empty():
        return

    display_cols = [
        "full_name",
        "party",
        "unity_score",
        "party_mean_unity",
        "plus_minus",
        "party_votes_present",
    ]
    available = [c for c in display_cols if c in pm.columns]
    display = pm.sort("plus_minus").select(available)

    html = make_interactive_table(
        display,
        title=f"{chamber} — Plus-Minus ({display.height} legislators)",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "unity_score": "Actual Unity",
            "party_mean_unity": "Party Mean",
            "plus_minus": "Plus-Minus",
            "party_votes_present": "Party Votes",
        },
        number_formats={
            "unity_score": ".3f",
            "party_mean_unity": ".3f",
            "plus_minus": "+.3f",
        },
        caption=(
            "Plus-minus = actual unity − party mean unity. "
            "Positive = more partisan than average party member."
        ),
    )
    report.add(
        InteractiveTableSection(
            id=f"plus-minus-table-{chamber.lower()}",
            title=f"{chamber} Plus-Minus Scores",
            html=html,
        )
    )


def _generate_indices_key_findings(results: dict[str, dict]) -> list[str]:
    """Generate 3-5 key findings from indices results."""
    findings: list[str] = []

    for chamber, result in results.items():
        # Rice cohesion
        rice_summary = result.get("rice_summary", {})
        for party in ["Republican", "Democrat"]:
            stats = rice_summary.get(party)
            if stats:
                findings.append(
                    f"{chamber} {party}s maintained <strong>{stats['mean']:.0%}</strong> "
                    f"mean Rice cohesion ({stats['pct_perfect_unity']:.0f}% unanimous)."
                )
                break  # Just show majority party

        # Top maverick
        maverick = result.get("maverick")
        if maverick is not None and maverick.height > 0:
            top = maverick.sort("maverick_rate", descending=True).head(1)
            name = top["full_name"][0]
            rate = float(top["maverick_rate"][0])
            findings.append(
                f"Top {chamber} maverick: <strong>{name}</strong> ({rate:.0%} defection rate)."
            )

        break  # Only show first chamber to stay concise

    # Party vote fraction
    total_party = sum(r.get("n_party_votes", 0) for r in results.values())
    total_votes = sum(r.get("n_total_votes", 1) for r in results.values())
    if total_votes > 0:
        pct = 100 * total_party / total_votes
        findings.append(
            f"<strong>{pct:.0f}%</strong> of roll calls were party votes "
            f"(majority R vs majority D)."
        )

    return findings


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: All constants and settings used in this run."""
    df = pl.DataFrame(
        {
            "Parameter": [
                "Party Vote Threshold",
                "Rice Fracture Threshold",
                "Min Party Voters",
                "Maverick Weight Floor",
                "Co-Defection Minimum",
                "ENP Multiparty Threshold",
                "Rolling Window",
                "Top Defectors N",
            ],
            "Value": [
                f"{PARTY_VOTE_THRESHOLD:.2f} (> 50% of Yea+Nay = majority)",
                f"{RICE_FRACTURE_THRESHOLD:.2f}",
                str(MIN_PARTY_VOTERS),
                str(MAVERICK_WEIGHT_FLOOR),
                str(CO_DEFECTION_MIN),
                str(ENP_MULTIPARTY_THRESHOLD),
                f"{ROLLING_WINDOW} roll calls",
                str(TOP_DEFECTORS_N),
            ],
            "Description": [
                "CQ standard: >50% = party majority position",
                "Rice below this = party more divided than united",
                "Minimum Yea+Nay voters per party for Rice to be computed",
                "Floor on chamber margin weight to prevent division by near-zero",
                "Minimum shared defections to appear in co-defection heatmap",
                "Per-vote ENP above this indicates multiparty-like behavior",
                "Rolling average window for time series plots",
                "Number of top defectors shown in co-defection heatmap",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/indices.md for justification.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
