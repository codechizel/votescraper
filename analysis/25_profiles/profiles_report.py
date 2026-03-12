"""Legislator profiles report builder — per-legislator deep-dive sections.

Assembles scorecard, bill-type breakdown, defection analysis, voting neighbors,
and surprising votes into a self-contained HTML report. Called by profiles.py.
"""

from pathlib import Path

import polars as pl

try:
    from analysis.profiles_data import ProfileTarget
except ModuleNotFoundError:
    from profiles_data import ProfileTarget  # type: ignore[no-redef]

try:
    from analysis.report import (
        FigureSection,
        InteractiveTableSection,
        KeyFindingsSection,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveTableSection,
        KeyFindingsSection,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )


def build_profiles_report(
    report: object,
    *,
    targets: list[ProfileTarget],
    all_data: dict[str, dict],
    plots_dir: Path,
    session: str,
    horseshoe_status: dict[str, dict] | None = None,
) -> None:
    """Build the full profiles report with intro + per-legislator sections."""
    from analysis.phase_utils import horseshoe_warning_html

    findings = _generate_profiles_key_findings(targets, all_data)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    # Horseshoe warnings
    if horseshoe_status:
        for chamber, status in horseshoe_status.items():
            warning = horseshoe_warning_html(chamber, status)
            if warning:
                report.add(
                    TextSection(
                        id=f"horseshoe-warning-{chamber.lower()}",
                        title=f"{chamber} Horseshoe Warning",
                        html=warning,
                    )
                )

    _add_intro(report, targets, session)

    for target in targets:
        data = all_data.get(target.slug, {})
        slug_short = target.slug.replace("rep_", "").replace("sen_", "")

        _add_target_header(report, target)
        _add_scorecard_figure(report, target, slug_short, plots_dir)
        _add_bill_type_figure(report, target, slug_short, plots_dir)
        _add_position_figure(report, target, slug_short, plots_dir)
        _add_sponsorship_section(report, target, data.get("sponsorship"))
        _add_defections_table(report, target, data.get("defections"))
        _add_surprising_votes_table(report, target, data.get("surprising"))
        _add_neighbors_figure(report, target, slug_short, plots_dir)
        _add_full_voting_record(report, target, data.get("full_record"))

    print(f"  Report: {len(report._sections)} sections added")


# ── Intro ────────────────────────────────────────────────────────────────────


def _add_intro(report: object, targets: list[ProfileTarget], session: str) -> None:
    """Opening section explaining what this report is."""
    n = len(targets)
    role_list = []
    for t in targets:
        role_list.append(f"<strong>{t.full_name}</strong> ({t.role})")
    names_html = ", ".join(role_list)

    report.add(
        TextSection(
            id="profiles-intro",
            title="Legislator Profiles — Deep Dives",
            html=(
                f"<p>This report profiles <strong>{n} legislators</strong> who stand "
                "out statistically from the Kansas Legislature's "
                f"{session} session. Each profile combines findings from eight "
                "analysis phases into a single deep-dive.</p>"
                f"<p>Profiled legislators: {names_html}.</p>"
                "<p>Flagged legislators were detected automatically using data-driven "
                "thresholds — different sessions will surface different individuals. "
                "Each profile includes a scorecard, bill-type voting breakdown, "
                "key defection votes, prediction surprises, and voting neighbors.</p>"
            ),
        )
    )


# ── Per-Legislator Sections ─────────────────────────────────────────────────


def _add_target_header(report: object, target: ProfileTarget) -> None:
    """Section header for one legislator with role and narrative."""
    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    chamber = target.chamber.title()

    report.add(
        TextSection(
            id=f"header-{slug_short}",
            title=f"{target.title} — {target.role}",
            html=(
                f"<p><strong>{chamber}</strong> &middot; "
                f"{target.party} &middot; District {target.district}</p>"
                f"<p><em>{target.subtitle}</em></p>"
            ),
        )
    )


def _add_scorecard_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Enhanced scorecard bar chart."""
    path = plots_dir / f"scorecard_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"scorecard-{slug_short}",
            f"{target.full_name} — At a Glance",
            path,
            caption=(
                "Horizontal bars show this legislator's metric values (colored) "
                "alongside their party average (gray dashed line). Metrics include "
                "IRT ideology, party unity, loyalty, maverick rate, network centrality, "
                "and prediction accuracy."
            ),
            alt_text=(
                f"Horizontal bar chart comparing {target.full_name}'s key metrics "
                f"(ideology, party unity, loyalty, maverick rate, centrality, accuracy) "
                f"to their party average shown as a dashed reference line."
            ),
        )
    )


def _add_bill_type_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Bill type breakdown grouped bar chart."""
    path = plots_dir / f"bill_type_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"bill-type-{slug_short}",
            f"How {target.full_name} Votes by Bill Type",
            path,
            caption=(
                "Yea rates on high-discrimination bills (partisan, contested) vs "
                "low-discrimination bills (routine, bipartisan). The gap between a "
                "legislator and their party average on partisan bills reveals how "
                "much they break ranks on the votes that matter most."
            ),
            alt_text=(
                f"Grouped bar chart comparing {target.full_name}'s Yea rates on "
                f"partisan versus routine bills against their party average."
            ),
        )
    )


def _add_position_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Forest-style position plot among same-party members."""
    path = plots_dir / f"position_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"position-{slug_short}",
            f"Where {target.full_name} Stands Among {target.party}s",
            path,
            caption=(
                "IRT ideal point estimates for all same-party members in the same "
                "chamber. The profiled legislator is highlighted with a diamond "
                "marker and yellow background. Horizontal lines show 95% credible "
                "intervals (uncertainty)."
            ),
            alt_text=(
                f"Forest plot of IRT ideal points for all {target.party} members in "
                f"the {target.chamber}. {target.full_name} is highlighted with a "
                f"diamond marker. Horizontal lines show 95% credible intervals."
            ),
        )
    )


def _add_sponsorship_section(
    report: object, target: ProfileTarget, sponsored: "pl.DataFrame | None"
) -> None:
    """Sponsorship summary and table of sponsored bills."""
    if sponsored is None or sponsored.height == 0:
        return

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")
    n_total = sponsored.height
    n_primary = sponsored.filter(pl.col("is_primary")).height
    n_passed = sponsored.filter(pl.col("passed") == True).height  # noqa: E712
    n_with_outcome = sponsored.filter(pl.col("passed").is_not_null()).height
    passage_pct = f"{n_passed / n_with_outcome:.0%}" if n_with_outcome > 0 else "N/A"

    report.add(
        TextSection(
            id=f"sponsorship-summary-{slug_short}",
            title=f"Sponsorship — {target.full_name}",
            html=(
                f"<p>Sponsored <strong>{n_total}</strong> bill"
                f"{'s' if n_total != 1 else ''}"
                f" ({n_primary} as primary sponsor). "
                f"Passage rate: <strong>{passage_pct}</strong>.</p>"
            ),
        )
    )

    display = sponsored.select(
        pl.col("bill_number").alias("Bill"),
        pl.col("short_title").alias("Title"),
        pl.col("motion").alias("Motion"),
        pl.when(pl.col("passed").is_null())
        .then(pl.lit(""))
        .when(pl.col("passed"))
        .then(pl.lit("Passed"))
        .otherwise(pl.lit("Failed"))
        .alias("Outcome"),
        pl.when(pl.col("is_primary"))
        .then(pl.lit("Primary"))
        .otherwise(pl.lit("Co-sponsor"))
        .alias("Role"),
    )

    html = make_gt(
        display,
        title=f"Bills Sponsored by {target.full_name}",
        subtitle=f"{n_primary} as primary sponsor, {n_total - n_primary} as co-sponsor",
    )

    report.add(
        TableSection(
            id=f"sponsorship-{slug_short}",
            title=f"Sponsored Bills — {target.full_name}",
            html=html,
        )
    )


def _add_defections_table(
    report: object, target: ProfileTarget, defections: pl.DataFrame | None
) -> None:
    """Table of key votes where this legislator broke ranks."""
    from analysis.phase_utils import drop_empty_optional_columns

    if defections is None or defections.height == 0:
        return

    defections = drop_empty_optional_columns(defections, ["short_title"])
    slug_short = target.slug.replace("rep_", "").replace("sen_", "")

    display_cols = [
        pl.col("bill_number").alias("Bill"),
    ]
    if "short_title" in defections.columns:
        display_cols.append(pl.col("short_title").alias("Title"))
    display_cols += [
        pl.col("motion").alias("Motion"),
        pl.col("legislator_vote").alias("Their Vote"),
        pl.col("party_majority_vote").alias("Party Majority"),
        pl.col("party_yea_pct").alias("Party Yea %"),
    ]
    if "sponsor" in defections.columns:
        display_cols.append(pl.col("sponsor").alias("Sponsor"))

    display = defections.select(display_cols)

    html = make_gt(
        display,
        title=f"Key Votes Where {target.full_name} Broke Ranks",
        subtitle="Sorted by closeness of the party margin (tightest first)",
        number_formats={"Party Yea %": ".1f"},
        source_note=(
            "A 'defection' is any vote where this legislator disagreed with their "
            "party's majority. Party Yea % shows what fraction of the party voted Yea."
        ),
    )

    report.add(
        TableSection(
            id=f"defections-{slug_short}",
            title=f"Defection Votes — {target.full_name}",
            html=html,
        )
    )


def _add_surprising_votes_table(
    report: object, target: ProfileTarget, surprising: pl.DataFrame | None
) -> None:
    """Table of votes where the prediction model was most wrong about this legislator."""
    if surprising is None or surprising.height == 0:
        return

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")

    display = surprising.select(
        pl.col("bill_number").alias("Bill"),
        pl.col("motion").alias("Motion"),
        pl.when(pl.col("actual") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("Actual Vote"),
        pl.when(pl.col("predicted") == 1)
        .then(pl.lit("Yea"))
        .otherwise(pl.lit("Nay"))
        .alias("Predicted"),
        (pl.col("y_prob") * 100).round(1).alias("Model Confidence (%)"),
        (pl.col("confidence_error") * 100).round(1).alias("Surprise Score"),
    )

    html = make_gt(
        display,
        title=f"Most Surprising Votes — {target.full_name}",
        subtitle="Votes where the prediction model was most confident and most wrong",
        number_formats={"Model Confidence (%)": ".1f", "Surprise Score": ".1f"},
        source_note=(
            "The Surprise Score is how wrong the model was (|confidence - actual|). "
            "Higher = more surprising. These are the votes where this legislator's "
            "behavior defied the patterns learned from the full legislature."
        ),
    )

    report.add(
        TableSection(
            id=f"surprising-{slug_short}",
            title=f"Surprising Votes — {target.full_name}",
            html=html,
        )
    )


def _generate_profiles_key_findings(
    targets: list[ProfileTarget],
    all_data: dict[str, dict],
) -> list[str]:
    """Generate 2-4 key findings from profiles results."""
    findings: list[str] = []

    if not targets:
        return findings

    n_targets = len(targets)
    names = [t.full_name for t in targets[:3]]
    label = ", ".join(names)
    if n_targets > 3:
        label += f" (+{n_targets - 3} more)"
    plural = "s" if n_targets != 1 else ""
    findings.append(f"Profiling <strong>{n_targets}</strong> legislator{plural}: {label}.")

    # Key metrics from first target with data
    for target in targets:
        data = all_data.get(target.slug, {})
        defections = data.get("defections")
        if defections is not None and hasattr(defections, "height") and defections.height > 0:
            findings.append(
                f"<strong>{target.full_name}</strong>: {defections.height} defection"
                f"{'s' if defections.height != 1 else ''} from party majority."
            )
            break

    return findings


def _add_neighbors_figure(
    report: object, target: ProfileTarget, slug_short: str, plots_dir: Path
) -> None:
    """Voting neighbors bar chart."""
    path = plots_dir / f"neighbors_{slug_short}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            f"neighbors-{slug_short}",
            f"Who Does {target.full_name} Vote Like?",
            path,
            caption=(
                "Top 5 most similar legislators (highest agreement) and top 5 most "
                "different (lowest agreement) by simple vote-matching rate across "
                "all shared votes. Same-chamber only."
            ),
            alt_text=(
                f"Horizontal bar chart showing {target.full_name}'s top 5 voting "
                f"allies (highest agreement) and top 5 opponents (lowest agreement) "
                f"by vote-matching rate."
            ),
        )
    )


def _add_full_voting_record(
    report: object, target: ProfileTarget, full_record: "pl.DataFrame | None"
) -> None:
    """Searchable/sortable full voting record table via ITables."""
    if full_record is None or full_record.height == 0:
        return

    slug_short = target.slug.replace("rep_", "").replace("sen_", "")

    display = full_record.with_columns(
        pl.when(pl.col("with_party"))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("with_party_str"),
        pl.when(pl.col("passed").is_null())
        .then(pl.lit(""))
        .when(pl.col("passed"))
        .then(pl.lit("Passed"))
        .otherwise(pl.lit("Failed"))
        .alias("outcome"),
    ).select(
        "date",
        "bill_number",
        "short_title",
        "motion",
        "vote",
        "party_majority",
        "with_party_str",
        "outcome",
    )

    html = make_interactive_table(
        display,
        title=f"Complete Voting Record — {target.full_name} ({full_record.height} votes)",
        column_labels={
            "date": "Date",
            "bill_number": "Bill",
            "short_title": "Title",
            "motion": "Motion",
            "vote": "Vote",
            "party_majority": "Party Majority",
            "with_party_str": "With Party?",
            "outcome": "Outcome",
        },
        caption="All Yea/Nay votes cast, sorted by date (most recent first). "
        "Searchable and sortable.",
    )

    report.add(
        InteractiveTableSection(
            id=f"full-record-{slug_short}",
            title=f"Full Voting Record — {target.full_name}",
            html=html,
        )
    )
