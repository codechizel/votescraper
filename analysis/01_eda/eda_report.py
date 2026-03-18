"""EDA-specific HTML report builder.

Builds ~19 sections (11 tables + 8 figures) for the Exploratory Data Analysis
report. Each section is a small function that slices/aggregates polars DataFrames
and calls make_gt() or FigureSection.from_file().

Usage (called from eda.py):
    from analysis.eda_report import build_eda_report
    build_eda_report(ctx.report, votes=votes, rollcalls=rollcalls, ...)
"""

from pathlib import Path

import polars as pl

try:
    from analysis.report import (
        FigureSection,
        InteractiveSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )
    from analysis.viz_helpers import PARTY_COLORS, make_hemicycle_chart
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )
    from viz_helpers import PARTY_COLORS, make_hemicycle_chart  # type: ignore[no-redef]


def build_eda_report(
    report: ReportBuilder,
    *,
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    manifests: dict,
    integrity_findings: dict,
    stat_findings: dict,
    participation: pl.DataFrame,
    party_unity: pl.DataFrame | None = None,
    eigenvalue_findings: dict | None = None,
    desposato_findings: dict | None = None,
    item_total_findings: dict | None = None,
    strategic_absence: pl.DataFrame | None = None,
    district_maps: dict[str, str] | None = None,
    bill_actions: pl.DataFrame | None = None,
    plots_dir: Path,
) -> None:
    """Build the full EDA HTML report by adding sections to the ReportBuilder."""
    # Key findings
    findings = _generate_key_findings(votes, rollcalls, legislators, stat_findings, participation)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_session_overview(report, votes, rollcalls, legislators)
    _add_chamber_party_composition(report, legislators)
    _add_vote_categories(report, votes)
    _add_vote_type_figure(report, plots_dir)
    _add_passage_rates(report, rollcalls)
    _add_margin_figure(report, plots_dir)
    _add_margin_stats(report, rollcalls)
    _add_vote_alignment(report, votes, rollcalls, legislators)
    _add_rice_cohesion(report, stat_findings)
    if desposato_findings:
        _add_desposato_rice(report, desposato_findings)
    _add_temporal_figure(report, plots_dir)
    _add_party_breakdown_figure(report, plots_dir)
    _add_participation_summary(report, participation)
    _add_participation_figure(report, plots_dir, "House")
    _add_participation_figure(report, plots_dir, "Senate")
    if strategic_absence is not None and not strategic_absence.is_empty():
        _add_absenteeism_analysis(report, strategic_absence)
    if party_unity is not None and not party_unity.is_empty():
        _add_party_unity(report, party_unity)
    if eigenvalue_findings:
        _add_eigenvalue_preview(report, eigenvalue_findings)
    if item_total_findings:
        _add_item_total_summary(report, item_total_findings)
    _add_filtering_decisions(report, manifests)
    _add_heatmap_figure(report, plots_dir, "House")
    _add_heatmap_figure(report, plots_dir, "Senate")
    if district_maps:
        for chamber_label, map_html in district_maps.items():
            report.add(
                InteractiveSection(
                    id=f"district-map-{chamber_label.lower()}",
                    title=f"Legislative District Map — {chamber_label}",
                    html=map_html,
                    caption=(
                        f"Interactive {chamber_label} district map. "
                        "Use layer control to toggle between party and ideology views. "
                        "Hover for district details."
                    ),
                    aria_label=(
                        f"Interactive choropleth map of {chamber_label} legislative districts "
                        f"with toggleable party affiliation and ideology layers."
                    ),
                )
            )
    if bill_actions is not None and not bill_actions.is_empty():
        _add_bill_lifecycle_sankey(report, bill_actions)
    _add_integrity_results(report, integrity_findings)
    _add_analysis_parameters(report)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ──────────────────────────────────────────────────


def _add_session_overview(
    report: ReportBuilder,
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
) -> None:
    """Table 1: High-level session counts and date range."""
    dates = rollcalls["vote_datetime"].sort()
    date_min = str(dates.first())[:10] if dates.len() > 0 else "N/A"
    date_max = str(dates.last())[:10] if dates.len() > 0 else "N/A"

    df = pl.DataFrame(
        {
            "Metric": [
                "Roll Calls",
                "Unique Bills",
                "Legislators",
                "Individual Votes",
                "First Vote",
                "Last Vote",
            ],
            "Value": [
                str(rollcalls.height),
                str(rollcalls["bill_number"].n_unique()),
                str(legislators.height),
                f"{votes.height:,}",
                date_min,
                date_max,
            ],
        }
    )
    html = make_gt(df, title="Session Overview")
    report.add(TableSection(id="session-overview", title="Session Overview", html=html))


def _add_chamber_party_composition(report: ReportBuilder, legislators: pl.DataFrame) -> None:
    """Table 2: Chamber x Party crosstab with totals."""
    cp = (
        legislators.group_by("chamber", "party")
        .agg(pl.len().alias("count"))
        .sort("chamber", "party")
    )
    # Pivot to wide format: rows = chamber, columns = parties
    wide = cp.pivot(on="party", index="chamber", values="count").fill_null(0)

    # Add total column
    party_cols = [c for c in wide.columns if c != "chamber"]
    wide = wide.with_columns(pl.sum_horizontal(*[pl.col(c) for c in party_cols]).alias("Total"))

    # Add total row
    totals = {"chamber": "Total"}
    for col in [*party_cols, "Total"]:
        totals[col] = wide[col].sum()
    wide = pl.concat([wide, pl.DataFrame([totals])], how="diagonal_relaxed")

    html = make_gt(
        wide,
        title="Chamber x Party Composition",
        column_labels={"chamber": "Chamber"},
        source_note="Counts exceed constitutional seat limits when mid-session replacements occur.",
    )
    report.add(TableSection(id="chamber-party", title="Chamber x Party Composition", html=html))

    # Hemicycle chart per chamber
    for chamber in ["House", "Senate"]:
        chamber_legs = legislators.filter(pl.col("chamber") == chamber)
        party_counts = chamber_legs.group_by("party").len().sort("party")
        if party_counts.is_empty():
            continue

        hemi_seats = [
            {
                "label": row["party"],
                "color": PARTY_COLORS.get(row["party"], "#808080"),
                "count": row["len"],
            }
            for row in party_counts.iter_rows(named=True)
        ]
        fig = make_hemicycle_chart(hemi_seats, f"{chamber} — Party Composition")
        hemi_html = fig.to_html(
            include_plotlyjs="cdn", full_html=False, div_id=f"hemicycle-{chamber.lower()}-plot"
        )
        report.add(
            InteractiveSection(
                id=f"hemicycle-{chamber.lower()}",
                title=f"{chamber} Party Composition (Hemicycle)",
                html=hemi_html,
                caption=f"Each dot represents one {chamber} seat, colored by party.",
            )
        )


def _add_vote_categories(report: ReportBuilder, votes: pl.DataFrame) -> None:
    """Table 3: Vote category distribution with counts and cumulative %."""
    vc = votes.group_by("vote").agg(pl.len().alias("count")).sort("count", descending=True)
    total = votes.height
    vc = vc.with_columns(
        (pl.col("count") / total * 100).alias("pct"),
    )
    # Compute cumulative percentage
    cum = []
    running = 0.0
    for row in vc.iter_rows(named=True):
        running += row["pct"]
        cum.append(running)
    vc = vc.with_columns(pl.Series("cum_pct", cum))

    html = make_gt(
        vc,
        title="Vote Category Distribution",
        subtitle=f"N = {total:,} individual votes",
        column_labels={
            "vote": "Category",
            "count": "Count",
            "pct": "%",
            "cum_pct": "Cumulative %",
        },
        number_formats={"count": ",.0f", "pct": ".1f", "cum_pct": ".1f"},
    )
    report.add(TableSection(id="vote-categories", title="Vote Category Distribution", html=html))


def _add_vote_type_figure(report: ReportBuilder, plots_dir: Path) -> None:
    """Figure 1: Vote type distribution bar chart."""
    path = plots_dir / "vote_type_distribution.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-vote-types",
                "Vote Type Distribution",
                path,
                caption="Horizontal bar chart of roll call vote types.",
                alt_text=(
                    "Horizontal bar chart showing the count of each roll call vote type "
                    "(e.g., Emergency Final Action, Final Action, veto overrides). "
                    "Bars are ordered by frequency."
                ),
            )
        )


def _add_passage_rates(report: ReportBuilder, rollcalls: pl.DataFrame) -> None:
    """Table 5: Passage rates by vote type."""
    has_result = rollcalls.filter(pl.col("passed").is_not_null())
    rates = (
        has_result.group_by("vote_type")
        .agg(
            pl.len().alias("total"),
            (pl.col("passed") == True).sum().alias("passed"),  # noqa: E712
        )
        .sort("total", descending=True)
        .with_columns(
            (pl.col("passed") / pl.col("total") * 100).alias("pass_rate"),
        )
    )
    html = make_gt(
        rates,
        title="Passage Rates by Vote Type",
        subtitle=f"N = {has_result.height} roll calls with a result",
        column_labels={
            "vote_type": "Vote Type",
            "total": "N",
            "passed": "Passed",
            "pass_rate": "Pass Rate (%)",
        },
        number_formats={"pass_rate": ".1f"},
    )
    report.add(TableSection(id="passage-rates", title="Passage Rates by Vote Type", html=html))


def _add_margin_figure(report: ReportBuilder, plots_dir: Path) -> None:
    """Figure 2: Vote margin distribution histogram."""
    path = plots_dir / "vote_margin_distribution.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-margins",
                "Vote Margin Distribution",
                path,
                caption=(
                    "Histogram of Yea % per roll call. Dashed lines at 50% "
                    "(simple majority) and 66.7% (veto override threshold)."
                ),
                alt_text=(
                    "Histogram of Yea percentage across all roll calls with dashed "
                    "reference lines at 50% simple majority and 66.7% veto override "
                    "threshold. Most votes cluster near 100%, reflecting a Republican "
                    "supermajority."
                ),
            )
        )


def _add_margin_stats(report: ReportBuilder, rollcalls: pl.DataFrame) -> None:
    """Table 7: Descriptive statistics for vote margins by chamber."""
    rc = rollcalls.with_columns(
        (pl.col("yea_count") / pl.col("total_votes") * 100).alias("yea_pct")
    )
    rows = []
    for chamber in ["House", "Senate", "Overall"]:
        subset = rc if chamber == "Overall" else rc.filter(pl.col("chamber") == chamber)
        if subset.height == 0:
            continue
        col = subset["yea_pct"]
        rows.append(
            {
                "chamber": chamber,
                "n": subset.height,
                "mean": col.mean(),
                "median": col.median(),
                "std": col.std(),
                "min": col.min(),
                "max": col.max(),
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Vote Margin Descriptive Statistics",
        subtitle="Yea % across roll calls",
        column_labels={
            "chamber": "Chamber",
            "n": "N",
            "mean": "Mean",
            "median": "Median",
            "std": "SD",
            "min": "Min",
            "max": "Max",
        },
        number_formats={
            "mean": ".1f",
            "median": ".1f",
            "std": ".1f",
            "min": ".1f",
            "max": ".1f",
        },
    )
    report.add(
        TableSection(id="margin-stats", title="Vote Margin Descriptive Statistics", html=html)
    )


def _add_vote_alignment(
    report: ReportBuilder,
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
) -> None:
    """Table 8: Vote alignment classification (bipartisan / party-line / mixed)."""
    # Replicate the classify_party_line logic to get alignment counts
    vote_with_party = votes.join(
        legislators.select("legislator_slug", "party"),
        on="legislator_slug",
    )
    substantive = vote_with_party.filter(pl.col("vote").is_in(["Yea", "Nay"]))
    party_agg = (
        substantive.group_by("vote_id", "party")
        .agg(
            (pl.col("vote") == "Yea").sum().alias("yea"),
            (pl.col("vote") == "Nay").sum().alias("nay"),
        )
        .with_columns((pl.col("yea") / (pl.col("yea") + pl.col("nay"))).alias("yea_rate"))
    )
    pivoted = party_agg.pivot(on="party", index="vote_id", values="yea_rate")

    r_col = "Republican" if "Republican" in pivoted.columns else None
    d_col = "Democrat" if "Democrat" in pivoted.columns else None

    if not (r_col and d_col):
        return

    classified = pivoted.with_columns(
        pl.when(
            ((pl.col(r_col) > 0.9) & (pl.col(d_col) > 0.9))
            | ((pl.col(r_col) < 0.1) & (pl.col(d_col) < 0.1))
        )
        .then(pl.lit("Bipartisan"))
        .when(
            ((pl.col(r_col) > 0.9) & (pl.col(d_col) < 0.1))
            | ((pl.col(r_col) < 0.1) & (pl.col(d_col) > 0.9))
        )
        .then(pl.lit("Party-Line"))
        .otherwise(pl.lit("Mixed"))
        .alias("alignment")
    )

    counts = (
        classified.group_by("alignment").agg(pl.len().alias("count")).sort("count", descending=True)
    )
    total = counts["count"].sum()
    counts = counts.with_columns((pl.col("count") / total * 100).alias("pct"))

    html = make_gt(
        counts,
        title="Vote Alignment Classification",
        subtitle=(
            "Bipartisan: both parties >90% same direction. Party-line: >90% opposite directions."
        ),
        column_labels={"alignment": "Alignment", "count": "N", "pct": "%"},
        number_formats={"pct": ".1f"},
    )
    report.add(TableSection(id="vote-alignment", title="Vote Alignment Classification", html=html))


def _add_rice_cohesion(report: ReportBuilder, stat_findings: dict) -> None:
    """Table 9: Rice Cohesion Index summary by party."""
    rice_data = stat_findings.get("rice_summary", [])
    if not rice_data:
        return

    rows = []
    for row in rice_data:
        rows.append(
            {
                "party": row["party"],
                "mean": row["mean"],
                "std": row["std"],
                "pct_perfect": (row["pct_perfect"] or 0) * 100,
                "n": row["n_party_votes"],
            }
        )
    df = pl.DataFrame(rows)

    html = make_gt(
        df,
        title="Rice Cohesion Index by Party",
        subtitle="0 = evenly split, 1 = perfect unity",
        column_labels={
            "party": "Party",
            "mean": "Mean",
            "std": "SD",
            "pct_perfect": "Perfect Unity (%)",
            "n": "N (party-votes)",
        },
        number_formats={"mean": ".3f", "std": ".3f", "pct_perfect": ".1f", "n": ",.0f"},
    )
    report.add(TableSection(id="rice-cohesion", title="Rice Cohesion Index by Party", html=html))


def _add_temporal_figure(report: ReportBuilder, plots_dir: Path) -> None:
    """Figure 3: Temporal activity plot."""
    path = plots_dir / "temporal_activity.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-temporal",
                "Temporal Activity",
                path,
                caption="Monthly roll call counts by chamber. "
                "Expect bursts near session deadlines.",
                alt_text=(
                    "Line chart showing monthly roll call counts by chamber across "
                    "the session. Activity typically spikes near session deadlines."
                ),
            )
        )


def _add_party_breakdown_figure(report: ReportBuilder, plots_dir: Path) -> None:
    """Figure 4: Party vote breakdown."""
    path = plots_dir / "party_vote_breakdown.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-party-breakdown",
                "Party Vote Breakdown",
                path,
                caption="Vote category breakdown (Yea/Nay/Absent) by party.",
                alt_text=(
                    "Stacked bar chart showing Yea, Nay, and Absent vote proportions "
                    "for each party. Compares voting patterns between Republicans and Democrats."
                ),
            )
        )


def _add_participation_summary(report: ReportBuilder, participation: pl.DataFrame) -> None:
    """Table 12: Participation rate summary statistics per chamber."""
    rows = []
    for chamber in ["House", "Senate", "Overall"]:
        subset = (
            participation
            if chamber == "Overall"
            else participation.filter(pl.col("chamber") == chamber)
        )
        if subset.height == 0:
            continue
        col = subset["participation_rate"] * 100
        rows.append(
            {
                "chamber": chamber,
                "n": subset.height,
                "mean": col.mean(),
                "median": col.median(),
                "std": col.std(),
                "min": col.min(),
                "max": col.max(),
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Participation Rate Summary",
        subtitle="Substantive votes (Yea + Nay) / chamber roll calls",
        column_labels={
            "chamber": "Chamber",
            "n": "N",
            "mean": "Mean (%)",
            "median": "Median (%)",
            "std": "SD",
            "min": "Min (%)",
            "max": "Max (%)",
        },
        number_formats={
            "mean": ".1f",
            "median": ".1f",
            "std": ".1f",
            "min": ".1f",
            "max": ".1f",
        },
    )
    report.add(
        TableSection(id="participation-summary", title="Participation Rate Summary", html=html)
    )


def _add_participation_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Figure 5/6: Per-legislator participation rates by chamber."""
    path = plots_dir / f"participation_rates_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-participation-{chamber.lower()}",
                f"{chamber} Participation Rates",
                path,
                caption=f"Per-legislator participation rates for the {chamber}, colored by party.",
                alt_text=(
                    f"Horizontal bar chart of per-legislator participation rates in the "
                    f"{chamber}, colored by party. Each bar shows the fraction of chamber "
                    f"roll calls where the legislator cast a substantive vote."
                ),
            )
        )


def _add_absenteeism_analysis(report: ReportBuilder, absence: pl.DataFrame) -> None:
    """Absenteeism analysis: ranked legislators by absence rate with strategic flags."""
    # Summary table: top absentees per chamber
    for chamber in ["House", "Senate"]:
        prefix = "rep_" if chamber == "House" else "sen_"
        chamber_df = absence.filter(pl.col("legislator_slug").str.starts_with(prefix))
        if chamber_df.is_empty():
            continue

        top = chamber_df.head(15).select(
            "full_name",
            "party",
            pl.col("overall_absence_rate").round(3),
            pl.col("pl_absence_rate").round(3),
            pl.col("absence_ratio").round(1),
            "absent_on_pl",
            "total_votes",
        )
        html = make_gt(
            top,
            title=f"{chamber} — Absenteeism Rankings",
            subtitle="Top 15 legislators by overall absence rate",
            column_labels={
                "full_name": "Legislator",
                "party": "Party",
                "overall_absence_rate": "Overall Absence %",
                "pl_absence_rate": "Party-Line Absence %",
                "absence_ratio": "Strategic Ratio",
                "absent_on_pl": "Absences on PL Votes",
                "total_votes": "Total Votes",
            },
            number_formats={
                "overall_absence_rate": "{:.2%}",
                "pl_absence_rate": "{:.2%}",
                "absence_ratio": "{:.1f}x",
            },
            source_note=(
                "Strategic ratio = party-line absence rate / overall absence rate. "
                "Ratio >= 2.0x with >= 3 party-line absences flags potential strategic absence."
            ),
        )
        report.add(
            TableSection(
                id=f"absenteeism-{chamber.lower()}", title=f"{chamber} Absenteeism", html=html
            )
        )

    # Flagged strategic absentees
    flagged = absence.filter((pl.col("absence_ratio") >= 2.0) & (pl.col("absent_on_pl") >= 3))
    if not flagged.is_empty():
        names = flagged["full_name"].to_list()
        n = flagged.height
        text = (
            f"<strong>{n} legislator(s) flagged for potential strategic absence:</strong> "
            + ", ".join(names)
            + ". These legislators miss party-line votes at more than twice their overall "
            "absence rate, suggesting absences may not be random."
        )
    else:
        text = (
            "No legislators were flagged for strategic absence. All absence patterns "
            "appear consistent with general absence rates."
        )
    report.add(
        TextSection(id="strategic-absence-flags", title="Strategic Absence Flags", html=text)
    )


def _add_filtering_decisions(report: ReportBuilder, manifests: dict) -> None:
    """Table 15: Filtering decisions summary (before/after/dropped per filter)."""
    rows = []
    for label in ["House", "Senate", "All"]:
        m = manifests.get(label, {})
        if not m:
            continue
        rows.append(
            {
                "scope": label,
                "votes_before": m.get("votes_before", 0),
                "votes_dropped": m.get("votes_dropped_unanimous", 0),
                "votes_after": m.get("votes_after", 0),
                "legislators_before": m.get("legislators_before", 0),
                "legislators_dropped": m.get("legislators_dropped_low_participation", 0),
                "legislators_after": m.get("legislators_after", 0),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Filtering Decisions",
        subtitle="Near-unanimous vote removal and low-participation legislator exclusion",
        column_labels={
            "scope": "Scope",
            "votes_before": "Votes (Before)",
            "votes_dropped": "Votes Dropped",
            "votes_after": "Votes (After)",
            "legislators_before": "Legislators (Before)",
            "legislators_dropped": "Legislators Dropped",
            "legislators_after": "Legislators (After)",
        },
        source_note=(
            "Votes dropped: minority < 2.5%. Legislators dropped: < 20 substantive votes."
        ),
    )
    report.add(TableSection(id="filtering-decisions", title="Filtering Decisions", html=html))


def _add_heatmap_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Figure 7/8: Agreement heatmap for a chamber."""
    path = plots_dir / f"agreement_heatmap_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-heatmap-{chamber.lower()}",
                f"{chamber} Agreement Heatmap",
                path,
                caption=(
                    f"Pairwise raw agreement among {chamber} legislators on contested votes. "
                    "Ward linkage clustering. Red sidebar = Republican, Blue = Democrat."
                ),
                alt_text=(
                    f"Clustered heatmap of pairwise voting agreement among {chamber} "
                    f"legislators on contested votes. Ward linkage clustering groups "
                    f"similar voters together. Party sidebar shows red for Republican "
                    f"and blue for Democrat."
                ),
            )
        )


def _add_integrity_results(report: ReportBuilder, integrity_findings: dict) -> None:
    """Table 18: Data integrity check results."""
    rows = []
    for item in integrity_findings.get("info", []):
        rows.append({"status": "OK", "detail": item})
    for item in integrity_findings.get("warnings", []):
        rows.append({"status": "WARN", "detail": item})

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Data Integrity Results",
        subtitle=f"{len(integrity_findings.get('warnings', []))} warnings, "
        f"{len(integrity_findings.get('info', []))} info",
        column_labels={"status": "Status", "detail": "Detail"},
    )
    report.add(TableSection(id="integrity-results", title="Data Integrity Results", html=html))


def _add_party_unity(report: ReportBuilder, party_unity: pl.DataFrame) -> None:
    """Table: Party Unity Scores — lowest-unity legislators."""
    # Show all legislators sorted by unity score
    display = (
        party_unity.select(
            "full_name", "chamber", "party", "party_unity_score", "n_party_line_votes"
        )
        .with_columns(
            (pl.col("party_unity_score") * 100).alias("party_unity_score"),
        )
        .sort("party_unity_score")
    )

    html = make_gt(
        display,
        title="Party Unity Scores",
        subtitle=(
            "Fraction of party-line votes where legislator voted with party majority. "
            "Lower = more independent."
        ),
        column_labels={
            "full_name": "Legislator",
            "chamber": "Chamber",
            "party": "Party",
            "party_unity_score": "Unity (%)",
            "n_party_line_votes": "Party-Line Votes",
        },
        number_formats={"party_unity_score": ".1f", "n_party_line_votes": ",.0f"},
    )
    report.add(TableSection(id="party-unity", title="Party Unity Scores", html=html))


def _add_eigenvalue_preview(report: ReportBuilder, eigenvalue_findings: dict) -> None:
    """Table: Eigenvalue preview — dimensionality diagnostic."""
    rows = []
    for chamber, ev_data in eigenvalue_findings.items():
        eigenvalues = ev_data.get("eigenvalues", [])
        ratio = ev_data.get("lambda_ratio")
        if eigenvalues:
            total = sum(eigenvalues)
            for i, ev in enumerate(eigenvalues):
                rows.append(
                    {
                        "chamber": chamber,
                        "component": f"λ{i + 1}",
                        "eigenvalue": ev,
                        "pct_variance": 100 * ev / total if total > 0 else 0,
                    }
                )
            if ratio is not None:
                rows.append(
                    {
                        "chamber": chamber,
                        "component": "λ1/λ2",
                        "eigenvalue": ratio,
                        "pct_variance": None,
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Eigenvalue Preview (Dimensionality Diagnostic)",
        subtitle="Top eigenvalues of the filtered vote matrix correlation matrix.",
        column_labels={
            "chamber": "Chamber",
            "component": "Component",
            "eigenvalue": "Value",
            "pct_variance": "% Variance",
        },
        number_formats={"eigenvalue": ".2f", "pct_variance": ".1f"},
        source_note="λ1/λ2 > 5 = strongly 1D. < 3 = second dimension may be meaningful.",
    )
    report.add(TableSection(id="eigenvalue-preview", title="Eigenvalue Preview", html=html))


def _add_desposato_rice(report: ReportBuilder, desposato_findings: dict) -> None:
    """Table: Desposato (2005) small-party correction for Rice index."""
    raw = desposato_findings.get("raw", {})
    corrected = desposato_findings.get("corrected", {})
    if not raw:
        return

    rows = []
    for party in sorted(raw.keys()):
        r = raw[party]
        c = corrected.get(party, r)
        rows.append(
            {
                "party": party,
                "raw_rice": r,
                "corrected_rice": c,
                "delta": c - r,
            }
        )

    df = pl.DataFrame(rows)
    min_party = desposato_findings.get("min_party", "?")
    min_size = desposato_findings.get("min_size", "?")
    html = make_gt(
        df,
        title="Desposato Rice Correction",
        subtitle=(
            f"Bootstrap correction for small-party inflation. "
            f"Larger parties resampled to n={min_size} (size of {min_party})."
        ),
        column_labels={
            "party": "Party",
            "raw_rice": "Raw Mean Rice",
            "corrected_rice": "Corrected Mean Rice",
            "delta": "Δ",
        },
        number_formats={"raw_rice": ".3f", "corrected_rice": ".3f", "delta": "+.3f"},
        source_note="Desposato (2005). Negative Δ = raw score was inflated by party size.",
    )
    report.add(TableSection(id="desposato-rice", title="Desposato Rice Correction", html=html))


def _add_item_total_summary(report: ReportBuilder, item_total_findings: dict) -> None:
    """Table: Item-total correlation summary per chamber."""
    rows = []
    for chamber, data in item_total_findings.items():
        n_analyzed = data.get("n_analyzed", 0)
        n_flagged = data.get("n_flagged", 0)
        pct_flagged = 100 * n_flagged / n_analyzed if n_analyzed > 0 else 0
        rows.append(
            {
                "chamber": chamber,
                "n_analyzed": n_analyzed,
                "n_flagged": n_flagged,
                "pct_flagged": pct_flagged,
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Item-Total Correlation Screening",
        subtitle="Roll calls with |r| < 0.1 are non-discriminating (procedural or cross-cutting).",
        column_labels={
            "chamber": "Chamber",
            "n_analyzed": "Contested Votes",
            "n_flagged": "Non-Discriminating",
            "pct_flagged": "% Flagged",
        },
        number_formats={"pct_flagged": ".1f"},
    )
    report.add(
        TableSection(id="item-total-corr", title="Item-Total Correlation Screening", html=html)
    )


def _add_bill_lifecycle_sankey(report: ReportBuilder, bill_actions: pl.DataFrame) -> None:
    """Interactive Sankey: bill flow through legislative stages."""
    try:
        from analysis.bill_lifecycle import plot_bill_lifecycle_sankey
    except ModuleNotFoundError:
        try:
            from bill_lifecycle import plot_bill_lifecycle_sankey  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return

    fig = plot_bill_lifecycle_sankey(bill_actions)
    if fig is None:
        return

    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    report.add(
        InteractiveSection(
            id="bill-lifecycle-sankey",
            title="Bill Lifecycle Flow",
            html=html,
            caption=(
                "Sankey diagram showing how bills flow through legislative stages. "
                "Width of each link is proportional to the number of bills making "
                "that transition. 'Died' is inferred for bills that stall in committee "
                "without reaching a floor vote."
            ),
        )
    )


def _generate_key_findings(
    votes: pl.DataFrame,
    rollcalls: pl.DataFrame,
    legislators: pl.DataFrame,
    stat_findings: dict,
    participation: pl.DataFrame,
) -> list[str]:
    """Generate 3-5 key findings from EDA results."""
    findings: list[str] = []

    # Roll call and legislator counts
    n_rollcalls = rollcalls.height
    n_legislators = legislators.height
    findings.append(
        f"Session covered <strong>{n_rollcalls:,}</strong> roll calls across "
        f"<strong>{n_legislators}</strong> legislators."
    )

    # Party-line vs bipartisan
    vote_with_party = votes.join(
        legislators.select("legislator_slug", "party"),
        on="legislator_slug",
    )
    substantive = vote_with_party.filter(pl.col("vote").is_in(["Yea", "Nay"]))
    party_agg = (
        substantive.group_by("vote_id", "party")
        .agg(
            (pl.col("vote") == "Yea").sum().alias("yea"),
            (pl.col("vote") == "Nay").sum().alias("nay"),
        )
        .with_columns((pl.col("yea") / (pl.col("yea") + pl.col("nay"))).alias("yea_rate"))
    )
    pivoted = party_agg.pivot(on="party", index="vote_id", values="yea_rate")
    if "Republican" in pivoted.columns and "Democrat" in pivoted.columns:
        party_line = pivoted.filter(
            ((pl.col("Republican") > 0.9) & (pl.col("Democrat") < 0.1))
            | ((pl.col("Republican") < 0.1) & (pl.col("Democrat") > 0.9))
        )
        pct_party_line = 100 * party_line.height / pivoted.height if pivoted.height > 0 else 0
        findings.append(
            f"<strong>{pct_party_line:.0f}%</strong> of votes were party-line "
            f"(each party >90% on opposite sides)."
        )

    # Lowest participation
    if participation.height > 0:
        worst = participation.sort("participation_rate").head(1)
        worst_name = worst["full_name"][0]
        worst_rate = float(worst["participation_rate"][0]) * 100
        findings.append(
            f"Lowest participation: <strong>{worst_name}</strong> ({worst_rate:.0f}% of "
            f"chamber roll calls)."
        )

    # Rice cohesion
    rice_data = stat_findings.get("rice_summary", [])
    for row in rice_data:
        if row["party"] == "Republican":
            findings.append(
                f"Republican mean Rice cohesion: <strong>{row['mean']:.2f}</strong> "
                f"({row['pct_perfect'] * 100:.0f}% of votes perfectly unified)."
            )
            break

    return findings


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: Analysis constants used in this run."""
    # Import the constants from eda.py
    try:
        from analysis.eda import (
            CONTESTED_THRESHOLD,
            HOUSE_SEATS,
            ITEM_TOTAL_CORRELATION_THRESHOLD,
            MIN_SHARED_VOTES,
            MIN_VOTES,
            RICE_BOOTSTRAP_ITERATIONS,
            SENATE_SEATS,
            STRATEGIC_ABSENCE_RATIO,
        )
    except ModuleNotFoundError:
        from eda import (  # type: ignore[no-redef]
            CONTESTED_THRESHOLD,
            HOUSE_SEATS,
            ITEM_TOTAL_CORRELATION_THRESHOLD,
            MIN_SHARED_VOTES,
            MIN_VOTES,
            RICE_BOOTSTRAP_ITERATIONS,
            SENATE_SEATS,
            STRATEGIC_ABSENCE_RATIO,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "Contested Threshold",
                "Min Substantive Votes",
                "Min Shared Votes (Agreement)",
                "House Seats",
                "Senate Seats",
                "Rice Bootstrap Iterations",
                "Strategic Absence Ratio",
                "Item-Total Correlation Threshold",
            ],
            "Value": [
                f"{CONTESTED_THRESHOLD:.3f} ({CONTESTED_THRESHOLD * 100:.1f}%)",
                str(MIN_VOTES),
                str(MIN_SHARED_VOTES),
                str(HOUSE_SEATS),
                str(SENATE_SEATS),
                str(RICE_BOOTSTRAP_ITERATIONS),
                f"{STRATEGIC_ABSENCE_RATIO:.1f}x",
                str(ITEM_TOTAL_CORRELATION_THRESHOLD),
            ],
            "Description": [
                "Drop votes where minority side < this fraction",
                "Drop legislators with fewer substantive votes",
                "Minimum shared votes for pairwise agreement",
                "Constitutional House seat count",
                "Constitutional Senate seat count",
                "Desposato bootstrap iterations for Rice correction",
                "Flag if party-line absence rate >= this × overall rate",
                "Flag roll calls with |r| below this as non-discriminating",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="Changing these values constitutes a sensitivity analysis.",
    )
    report.add(TableSection(id="analysis-params", title="Analysis Parameters", html=html))
