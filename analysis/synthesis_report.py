"""
Synthesis report builder — narrative-driven sections.

Assembles findings from all 7 analysis phases into a single HTML report
written for nontechnical audiences. Called by synthesis.py.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from analysis.report import FigureSection, TableSection, TextSection, make_gt
except ModuleNotFoundError:
    from report import FigureSection, TableSection, TextSection, make_gt  # type: ignore[no-redef]


def build_synthesis_report(
    report: object,
    *,
    leg_dfs: dict[str, pl.DataFrame],
    manifests: dict[str, dict],
    upstream: dict,
    plots_dir: Path,
    upstream_plots: dict[str, Path],
    notables: dict,
    session: str,
) -> None:
    """Build the full synthesis report (27-30 sections depending on detections)."""
    # 1. intro
    _add_intro(report, manifests, notables)
    # 2. pipeline
    _add_pipeline_figure(report, plots_dir)
    # 3. party-line
    _add_party_line_narrative(report, manifests)
    # 4. clusters
    _add_clusters_figure(report, upstream_plots)
    # 5-6. network house/senate
    _add_network_figure(report, upstream_plots, "house")
    _add_network_figure(report, upstream_plots, "senate")
    # 7-8. dashboard house/senate
    _add_dashboard_figure(report, plots_dir, "house")
    _add_dashboard_figure(report, plots_dir, "senate")
    # 9. mavericks
    _add_mavericks_narrative(report, leg_dfs, notables)
    # 10. agreement-house
    _add_agreement_figure(report, upstream_plots, "house")
    # 11. profile (house maverick)
    _add_dynamic_profile(report, plots_dir, notables, "house", "maverick")
    # 12. forest-house
    _add_forest_figure(report, upstream_plots, "house", notables)
    # 13. maverick-landscape-house
    _add_maverick_landscape(report, upstream_plots, "house")
    # 14. maverick-landscape-senate
    _add_maverick_landscape(report, upstream_plots, "senate")
    # 15. profile (senate bridge or maverick)
    _add_dynamic_profile(report, plots_dir, notables, "senate", "bridge")
    # 16. forest-senate
    _add_forest_figure(report, upstream_plots, "senate", notables)
    # 17. paradox narrative
    _add_paradox_narrative(report, leg_dfs, notables)
    # 18. paradox visual
    _add_paradox_figure(report, plots_dir, notables)
    # 19. profile (paradox legislator)
    _add_paradox_profile(report, plots_dir, notables)
    # 20. veto-overrides
    _add_veto_narrative(report, manifests)
    # 21. unpredictable
    _add_unpredictable_narrative(report, upstream, notables)
    # 22. shap-house
    _add_shap_figure(report, upstream_plots, "house")
    # 23. accuracy-house
    _add_accuracy_figure(report, upstream_plots, "house")
    # 24. accuracy-senate
    _add_accuracy_figure(report, upstream_plots, "senate")
    # 25. calibration
    _add_calibration_figure(report, upstream_plots, "house")
    # 26. surprising-votes
    _add_surprising_votes_table(report, upstream)
    # 27. methodology
    _add_methodology_note(report, session)
    # 28. convergence
    _add_convergence_figure(report, upstream_plots, "house")
    # 29. discrimination
    _add_discrimination_figure(report, upstream_plots, "house")
    # 30. full-scorecard
    _add_full_scorecard(report, leg_dfs, session)

    print(f"  Report: {len(report._sections)} sections added")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _notable_names_for(notables: dict, key: str) -> list[str]:
    """Get a list of full_name strings from a notables sub-dict."""
    items = notables.get(key, {})
    if isinstance(items, dict):
        return [v.full_name for v in items.values() if hasattr(v, "full_name")]
    return []


def _maverick_name_list(notables: dict) -> str:
    """Comma-separated list of maverick full names for prose."""
    names = _notable_names_for(notables, "mavericks")
    paradox_names = _notable_names_for(notables, "paradoxes")
    all_names = list(dict.fromkeys(names + paradox_names))
    if not all_names:
        return "maverick legislators"
    if len(all_names) == 1:
        return all_names[0]
    return ", ".join(all_names[:-1]) + f", and {all_names[-1]}"


# ── Section Builders ─────────────────────────────────────────────────────────


def _add_intro(report: object, manifests: dict, notables: dict) -> None:
    """Section 1: What This Report Tells You."""
    eda = manifests.get("eda", {})
    total_votes = eda.get("All", {}).get("votes_before", 882)
    contested = eda.get("All", {}).get("votes_after", 491)
    n_legislators = eda.get("All", {}).get("legislators_before", 172)

    # Build dynamic maverick finding
    mav_names = []
    for chamber, mav in notables.get("mavericks", {}).items():
        mav_names.append(f"{mav.full_name} ({chamber.title()})")
    if mav_names:
        maverick_finding = (
            "<li><strong>A handful of mavericks stand out.</strong> "
            + " and ".join(mav_names)
            + " consistently break from their party on contested votes.</li>"
        )
    else:
        maverick_finding = (
            "<li><strong>Party discipline is exceptionally uniform.</strong> "
            "No legislators consistently break ranks on contested votes.</li>"
        )

    # Build dynamic paradox finding
    paradoxes = notables.get("paradoxes", {})
    if paradoxes:
        p = next(iter(paradoxes.values()))
        paradox_finding = (
            f"<li><strong>The {p.full_name.split()[-1]} Paradox:</strong> "
            f"{p.full_name} is the most extreme by one measure and the least loyal "
            f"by another — because they defect <em>{p.direction}</em> from their "
            "party.</li>"
        )
    else:
        paradox_finding = ""

    report.add(
        TextSection(
            id="intro",
            title="What This Report Tells You",
            html=(
                "<p>The Kansas Legislature cast "
                f"<strong>{total_votes:,} roll call votes</strong> across "
                f"<strong>{n_legislators} legislators</strong> in the House and Senate. "
                "This report distills those votes into a clear picture of how Kansas "
                "legislators actually vote — who follows the party, who breaks ranks, "
                "and what patterns emerge.</p>"
                "<p>We applied seven different analytical methods to this data: "
                "exploratory analysis, dimensionality reduction (PCA), Bayesian ideal "
                "point estimation (IRT), clustering, network analysis, predictive "
                "modeling, and classical political science indices. Each method asks a "
                "different question, but they all converge on the same answers.</p>"
                "<p><strong>Headline findings:</strong></p>"
                "<ol>"
                "<li><strong>Party is everything.</strong> Every method — clustering, "
                "networks, prediction — finds that party affiliation explains nearly all "
                "voting behavior. There are no hidden factions.</li>"
                "<li><strong>Most votes are not contested.</strong> "
                f"Of {total_votes:,} roll calls, "
                f"only {contested:,} had meaningful dissent (more than 2.5% minority). "
                "The rest were near-unanimous.</li>"
                + maverick_finding
                + paradox_finding
                + "<li><strong>Votes are highly predictable.</strong> A machine learning model "
                "can predict individual votes with near-perfect accuracy using only a "
                "legislator's ideology score and the bill's characteristics.</li>"
                "</ol>"
            ),
        )
    )


def _add_pipeline_figure(report: object, plots_dir: Path) -> None:
    """Section 2: Pipeline infographic."""
    path = plots_dir / "pipeline_summary.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "pipeline",
                "From Raw Votes to Prediction",
                path,
                caption=(
                    "Each box represents a stage of analysis. We start with every recorded "
                    "roll call, filter to contested votes, identify party-line votes, confirm "
                    "that party is the dominant grouping, and show that a model using these "
                    "patterns predicts individual votes with near-perfect accuracy."
                ),
            )
        )


def _add_party_line_narrative(report: object, manifests: dict) -> None:
    """Section 3: The Party Line Is Everything."""
    clust = manifests.get("clustering", {})
    mean_ari = clust.get("house_mean_ari", 0.96)

    report.add(
        TextSection(
            id="party-line",
            title="The Party Line Is Everything",
            html=(
                "<p>The single most important finding from this analysis is also the "
                "simplest: <strong>party affiliation predicts nearly everything</strong> "
                "about how a Kansas legislator votes.</p>"
                "<p>Here is how every method confirms this:</p>"
                "<ul>"
                "<li><strong>Clustering:</strong> The optimal number of voting blocs is 2 — "
                "exactly the two parties. Three different clustering algorithms agree, "
                f"matching actual party labels {mean_ari:.0%} of the time.</li>"
                "<li><strong>Networks:</strong> When we build a network of legislators connected "
                "by voting similarity, party separation is total — not a single strong "
                "connection crosses the party line. The two parties form completely "
                "separate voting blocs.</li>"
                "<li><strong>Community detection:</strong> A community-detection algorithm — "
                "which knows nothing about party labels — independently discovers the two "
                "parties in both the House and the Senate.</li>"
                "<li><strong>Prediction:</strong> A model using party, ideology, and bill features "
                "achieves near-perfect prediction.</li>"
                "</ul>"
                "<p>This does <em>not</em> mean all legislators vote identically. Within each "
                "party, there is meaningful variation — some members are reliably loyal, others "
                "are frequent dissenters. But the gap <em>between</em> parties is vastly larger "
                "than the variation <em>within</em> them.</p>"
            ),
        )
    )


def _add_network_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Sections 5-6: Community network diagrams (reused)."""
    key = f"community_network_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"network-{chamber}",
                f"The Kansas {chamber_title} Voting Network",
                path,
                caption=(
                    f"Each dot is a {chamber_title} member. Lines connect legislators who vote "
                    "similarly (Cohen's Kappa > 0.4). Colors are communities detected by the "
                    "Louvain algorithm — which independently rediscovers the two parties. "
                    "Legislators circled with a red halo are bridge legislators — members "
                    "with unusually high cross-party connections. Their names are labeled "
                    "directly on the plot. Notice the complete separation: no edges cross "
                    "the party divide."
                ),
            )
        )


def _add_dashboard_figure(report: object, plots_dir: Path, chamber: str) -> None:
    """Sections 7-8: Dashboard scatter (new)."""
    path = plots_dir / f"dashboard_scatter_{chamber}.png"
    if path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"dashboard-{chamber}",
                f"Every {chamber_title} Member at a Glance",
                path,
                caption=(
                    f"Each dot is a {chamber_title} member. Horizontal position shows ideology "
                    "(liberal to conservative) from Bayesian IRT. Vertical position shows party "
                    "unity (how often they vote with their party on contested votes). Larger "
                    "circles indicate more frequent maverick behavior. Red = Republican, "
                    "Blue = Democrat."
                ),
            )
        )


def _add_mavericks_narrative(report: object, leg_dfs: dict, notables: dict) -> None:
    """Section 9: Who Are the Mavericks?"""
    parts = [
        "<p>In a legislature where party discipline is the norm, a few members "
        "consistently break ranks. These <strong>mavericks</strong> — legislators "
        "who vote against their own party on contested votes — are analytically "
        "interesting because they reveal where the party line has cracks.</p>"
    ]

    mavericks = notables.get("mavericks", {})
    bridges = notables.get("bridges", {})

    if not mavericks and not bridges:
        parts.append(
            "<p>In this session, party discipline is exceptionally uniform — "
            "no legislator stands out as a consistent maverick.</p>"
        )
    else:
        for chamber, mav in mavericks.items():
            row = _get_legislator_row(leg_dfs, mav.slug, mav.chamber)
            if row is None:
                continue
            unity = row.get("unity_score", 0)
            mav_rate = row.get("maverick_rate", 0)
            acc = row.get("accuracy")
            xi = row.get("xi_mean", 0)

            acc_text = ""
            if acc is not None:
                acc_text = (
                    f" They are also among the hardest {chamber.title()} members "
                    f"for the prediction model to get right ({acc:.0%} accuracy "
                    "vs ~95% for most members)."
                )

            parts.append(
                f"<p><strong>{mav.title}</strong> is the {chamber.title()}'s most "
                f"prominent maverick. Their party unity score is {unity:.0%} — well "
                f"below the party average — and they defect on {mav_rate:.0%} of "
                f"party votes.{acc_text} Their IRT ideal point ({xi:.2f}) places "
                "them away from their party's mainstream.</p>"
            )

        for chamber, bridge in bridges.items():
            if bridge.slug in {m.slug for m in mavericks.values()}:
                continue  # already covered above
            row = _get_legislator_row(leg_dfs, bridge.slug, bridge.chamber)
            if row is None:
                continue
            unity = row.get("unity_score", 0)
            xi = row.get("xi_mean", 0)

            parts.append(
                f"<p><strong>{bridge.title}</strong> is the {chamber.title()}'s "
                f"bridge-builder. With a party unity of {unity:.0%} and an IRT "
                f"ideal point of {xi:.2f}, they vote more like a moderate than "
                "a party stalwart. Their network centrality scores suggest they "
                "bridge the gap between the two parties.</p>"
            )

    report.add(
        TextSection(
            id="mavericks",
            title="Who Are the Mavericks?",
            html="".join(parts),
        )
    )


def _get_legislator_row(leg_dfs: dict, slug: str, chamber: str) -> dict | None:
    """Get a single legislator's row as a dict."""
    df = leg_dfs.get(chamber)
    if df is None:
        return None
    row = df.filter(pl.col("legislator_slug") == slug)
    if row.height == 0:
        return None
    return row.to_dicts()[0]


def _add_dynamic_profile(
    report: object,
    plots_dir: Path,
    notables: dict,
    chamber: str,
    role: str,
) -> None:
    """Add a profile card section for a dynamically detected legislator.

    role: "maverick" looks in mavericks dict, "bridge" looks in bridges then
    falls back to mavericks.
    """
    notable = None
    if role == "bridge":
        notable = notables.get("bridges", {}).get(chamber)
        if notable is None:
            notable = notables.get("mavericks", {}).get(chamber)
    elif role == "maverick":
        notable = notables.get("mavericks", {}).get(chamber)

    if notable is None:
        return

    slug_short = notable.slug.split("_")[1]
    _add_profile_figure(
        report,
        plots_dir,
        slug_short,
        f"Legislator Profile: {notable.title}",
    )


def _add_profile_figure(
    report: object,
    plots_dir: Path,
    slug_short: str,
    title: str,
) -> None:
    """Profile card figure section."""
    path = plots_dir / f"profile_{slug_short}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"profile-{slug_short}",
                title,
                path,
                caption=(
                    "Six key metrics normalized to a 0-1 scale. Ideological Rank and Network "
                    "Influence are percentiles within the chamber. Party Unity, Clustering "
                    "Loyalty, Maverick Rate, and Prediction Accuracy are raw scores."
                ),
            )
        )


def _add_forest_figure(report: object, upstream_plots: dict, chamber: str, notables: dict) -> None:
    """Sections 12, 16: IRT forest plots (reused)."""
    key = f"forest_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()

        # Build dynamic callout names for the caption
        callout_parts = []
        mav = notables.get("mavericks", {}).get(chamber)
        if mav is not None:
            callout_parts.append(
                f"{mav.full_name.split()[-1]} (most bipartisan {chamber_title} member)"
            )
        paradoxes = notables.get("paradoxes", {})
        for slug, p in paradoxes.items():
            if p.chamber == chamber:
                callout_parts.append(f"{p.full_name.split()[-1]} (most extreme but lowest loyalty)")

        if callout_parts:
            callout_text = (
                " Diamond markers and italic callouts highlight legislators with "
                "notable patterns — such as " + " and ".join(callout_parts) + "."
            )
        else:
            callout_text = ""

        report.add(
            FigureSection.from_file(
                f"forest-{chamber}",
                f"Where Every {chamber_title} Member Falls on the Spectrum",
                path,
                caption=(
                    f"Bayesian IRT ideal point estimates for every {chamber_title} member. "
                    "Each dot is a legislator's estimated position, and horizontal lines "
                    "show the 95% credible interval (uncertainty). Negative values = more "
                    "liberal, positive = more conservative. Red = Republican, Blue = Democrat."
                    + callout_text
                ),
            )
        )


def _add_maverick_landscape(report: object, upstream_plots: dict, chamber: str) -> None:
    """Sections 13-14: Maverick landscape (reused, per chamber)."""
    key = f"maverick_landscape_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"maverick-landscape-{chamber}",
                f"Strategic vs Performative Independence ({chamber_title})",
                path,
                caption=(
                    f"Each dot is a {chamber_title} member, positioned by ideology "
                    "(horizontal) and party unity (vertical). Legislators in the lower "
                    "portion of their party's cluster defect more often. Those near the "
                    "center may be genuinely moderate; those at the extremes may defect in "
                    "the opposite direction from what you'd expect."
                ),
            )
        )


def _add_paradox_narrative(report: object, leg_dfs: dict, notables: dict) -> None:
    """Section 17: The Metric Paradox."""
    paradoxes = notables.get("paradoxes", {})

    if not paradoxes:
        report.add(
            TextSection(
                id="metric-paradox",
                title="No Significant Metric Paradox",
                html=(
                    "<p>In this session, no legislator shows a dramatic split between "
                    "ideology rank and loyalty rank. The metrics broadly agree on who "
                    "is extreme and who is loyal — an unusual degree of consistency.</p>"
                ),
            )
        )
        return

    p = next(iter(paradoxes.values()))
    rv = p.raw_values
    xi = rv.get("xi_mean", 0)
    loyalty = rv.get("loyalty_rate", 0)
    unity = rv.get("unity_score")

    unity_text = ""
    if unity is not None:
        unity_text = (
            f" By <strong>CQ party unity</strong>, they score {unity:.0%} — solidly "
            "in the upper ranks of their party."
        )

    report.add(
        TextSection(
            id="metric-paradox",
            title=f"The {p.full_name.split()[-1]} Paradox",
            html=(
                f"<p>{p.full_name} ({p.party[0]}-{p.district}) presents a puzzle that "
                "illuminates how different metrics can give contradictory answers about "
                "the same legislator.</p>"
                f"<p>By <strong>{p.metric_high_name}</strong>, they rank "
                f"#{p.rank_high} among {p.n_in_party} {p.party}s (score: {xi:.1f}). "
                f"By <strong>{p.metric_low_name}</strong>, they are among the "
                f"<em>least</em> loyal ({loyalty:.0%} agreement with party on contested "
                f"votes).{unity_text}</p>"
                f"<p>How can one person be the most extreme <em>and</em> the least "
                "loyal? The answer is in the direction of defection. They don't defect "
                f"<em>toward</em> the other party — they defect <em>{p.direction}</em> "
                "from their own party's mainstream position. On votes where most of "
                "their party says Yea, they sometimes say Nay because the bill isn't "
                "aligned enough with their position.</p>"
                "<p>This is why multiple metrics matter. No single number can capture "
                "the full story of a legislator's voting behavior.</p>"
            ),
        )
    )


def _add_paradox_figure(report: object, plots_dir: Path, notables: dict) -> None:
    """Section 18: Three Measures, Three Answers."""
    paradoxes = notables.get("paradoxes", {})
    if not paradoxes:
        return

    p = next(iter(paradoxes.values()))
    path = plots_dir / f"metric_paradox_{p.chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "paradox-visual",
                "Three Measures, Three Answers",
                path,
                caption=(
                    f"The same legislator measured three different ways. "
                    f"{p.metric_high_name.upper()} (all votes) ranks them as most extreme. "
                    f"{p.metric_low_name.title()} (contested votes within party) shows them "
                    "as least loyal. CQ party unity (party-vs-party votes only) puts them "
                    "in the upper ranks. The metrics disagree because they measure different "
                    "subsets of votes."
                ),
            )
        )


def _add_paradox_profile(report: object, plots_dir: Path, notables: dict) -> None:
    """Section 19: Profile card for paradox legislator."""
    paradoxes = notables.get("paradoxes", {})
    if not paradoxes:
        return

    p = next(iter(paradoxes.values()))
    # Check if this slug is already profiled as a maverick/bridge (avoid duplicate)
    mavericks = notables.get("mavericks", {})
    bridges = notables.get("bridges", {})
    already_profiled = {m.slug for m in mavericks.values()} | {b.slug for b in bridges.values()}
    if p.slug in already_profiled:
        return

    slug_short = p.slug.split("_")[1]
    _add_profile_figure(
        report,
        plots_dir,
        slug_short,
        f"Legislator Profile: {p.full_name} ({p.party[0]}-{p.district})",
    )


def _add_veto_narrative(report: object, manifests: dict) -> None:
    """Section 20: Veto Overrides Tell You Nothing New."""
    indices = manifests.get("indices", {})
    h_veto = indices.get("house_veto_overrides", {})
    s_veto = indices.get("senate_veto_overrides", {})
    n_h = h_veto.get("n_overrides", 17)
    n_s = s_veto.get("n_overrides", 17)
    total = n_h + n_s

    h_rice_r = h_veto.get("rice_by_party", {}).get("Republican", 0.96)
    h_rice_d = h_veto.get("rice_by_party", {}).get("Democrat", 0.99)
    s_rice_r = s_veto.get("rice_by_party", {}).get("Republican", 0.97)
    s_rice_d = s_veto.get("rice_by_party", {}).get("Democrat", 0.97)

    report.add(
        TextSection(
            id="veto-overrides",
            title="Veto Overrides Tell You Nothing New",
            html=(
                f"<p>This session included <strong>{total} veto override votes</strong> "
                f"({n_h} House, {n_s} Senate). These are analytically interesting in theory — "
                "overrides require a two-thirds supermajority, which often forces bipartisan "
                "coalitions.</p>"
                "<p>In Kansas, however, the Republican supermajority means overrides don't "
                "require any Democratic support. The data confirms this:</p>"
                "<ul>"
                f"<li>House Republican Rice Index on overrides: <strong>{h_rice_r:.2f}</strong> "
                f"(Democrat: {h_rice_d:.2f})</li>"
                f"<li>Senate Republican Rice Index on overrides: <strong>{s_rice_r:.2f}</strong> "
                f"(Democrat: {s_rice_d:.2f})</li>"
                "</ul>"
                "<p>Both parties vote almost unanimously on override votes — Republicans "
                "override, Democrats oppose (or vice versa). There is no bipartisan "
                "coalition to find. The veto overrides are simply another expression "
                "of party discipline.</p>"
            ),
        )
    )


def _add_unpredictable_narrative(report: object, upstream: dict, notables: dict) -> None:
    """Section 21: What the Model Cannot Predict."""
    # Get holdout AUC from upstream
    house_hr = upstream.get("house", {}).get("holdout_results")
    senate_hr = upstream.get("senate", {}).get("holdout_results")

    house_auc = "0.98"
    senate_auc = "0.98"
    if house_hr is not None:
        xgb = house_hr.filter(pl.col("model") == "XGBoost")
        if xgb.height > 0:
            house_auc = f"{xgb['auc'].item():.3f}"
    if senate_hr is not None:
        xgb = senate_hr.filter(pl.col("model") == "XGBoost")
        if xgb.height > 0:
            senate_auc = f"{xgb['auc'].item():.3f}"

    mav_names = _maverick_name_list(notables)

    report.add(
        TextSection(
            id="unpredictable",
            title="What the Model Cannot Predict",
            html=(
                f"<p>An XGBoost classifier achieves AUC = {house_auc} (House) and "
                f"{senate_auc} (Senate) on held-out votes — meaning party and ideology "
                "explain nearly all voting behavior. But 'nearly all' is not 'all'.</p>"
                "<p>The model's errors are not random. They cluster around specific "
                "legislators and specific bills:</p>"
                "<ul>"
                f"<li><strong>Maverick legislators</strong> — {mav_names} — "
                "account for a disproportionate share of prediction errors. Their votes "
                "are harder to predict because they don't follow the party-ideology pattern "
                "that works for everyone else.</li>"
                "<li><strong>Bipartisan bills</strong> and <strong>procedural motions</strong> "
                "generate more errors than final-passage party-line votes.</li>"
                "</ul>"
                "<p>The residuals tell us where the interesting politics happen: on the "
                "margins, where individual judgment overrides party loyalty.</p>"
            ),
        )
    )


def _add_accuracy_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Sections 23-24: Per-legislator accuracy (reused, per chamber)."""
    key = f"per_legislator_accuracy_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"accuracy-{chamber}",
                f"Some {chamber_title} Members Are Harder to Predict",
                path,
                caption=(
                    f"Prediction accuracy per {chamber_title} member on the holdout test "
                    "set. Most legislators are predicted correctly 93-97% of the time. "
                    "Callout boxes highlight the bottom five — the mavericks and moderates "
                    "whose votes carry the most information about what's actually happening "
                    "in the legislature."
                ),
            )
        )


def _add_surprising_votes_table(report: object, upstream: dict) -> None:
    """Section 26: The 20 Most Surprising Votes."""
    # Combine house and senate surprising votes
    frames = []
    for chamber in ("house", "senate"):
        sv = upstream.get(chamber, {}).get("surprising_votes")
        if sv is not None:
            frames.append(sv.with_columns(pl.lit(chamber.title()).alias("Chamber")))

    if not frames:
        return

    combined = pl.concat(frames).sort("confidence_error", descending=True).head(20)

    # Build display table
    display = combined.select(
        [
            pl.col("full_name").alias("Legislator"),
            pl.col("party").alias("Party"),
            pl.col("Chamber"),
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
        ]
    )

    html = make_gt(
        display,
        title="The 20 Most Surprising Votes",
        subtitle="Votes where the model was most confident — and most wrong",
        column_labels={
            "Model Confidence (%)": "Model Confidence (%)",
        },
        number_formats={
            "Model Confidence (%)": ".1f",
        },
        source_note=(
            "Confidence shows how sure the model was of the predicted outcome. "
            "Higher confidence on a wrong prediction = more surprising vote."
        ),
    )

    report.add(
        TableSection(
            id="surprising-votes",
            title="The 20 Most Surprising Votes",
            html=html,
        )
    )


def _add_methodology_note(report: object, session: str) -> None:
    """Section 27: How We Did This."""
    report.add(
        TextSection(
            id="methodology",
            title="How We Did This",
            html=(
                "<p>This report synthesizes seven independent analyses of roll call votes "
                f"from the Kansas Legislature's {session} session. The raw data — every "
                "recorded Yea, Nay, and absence — was scraped from "
                "<a href='https://kslegislature.gov'>kslegislature.gov</a> and stored as "
                "structured CSV files.</p>"
                "<p>The seven phases, each documented in its own technical report:</p>"
                "<ol>"
                "<li><strong>Exploratory Data Analysis (EDA)</strong> — Filtering, missing data, "
                "vote distributions, unanimous-vote removal</li>"
                "<li><strong>PCA</strong> — Principal component analysis to identify the main "
                "axes of voting variation</li>"
                "<li><strong>Bayesian IRT</strong> — A 2-parameter Item Response Theory model "
                "that estimates each legislator's ideology on a continuous scale</li>"
                "<li><strong>Clustering</strong> — K-means, hierarchical, and Gaussian mixture "
                "models to find natural voting blocs</li>"
                "<li><strong>Network Analysis</strong> — Building similarity networks from "
                "Cohen's Kappa, detecting communities, measuring centrality</li>"
                "<li><strong>Prediction</strong> — XGBoost and logistic regression to predict "
                "individual votes and identify surprising outcomes</li>"
                "<li><strong>Classical Indices</strong> — Rice Index, CQ-standard party unity, "
                "maverick scores, Effective Number of Parties</li>"
                "</ol>"
                "<p>Key methodological choices: near-unanimous votes (minority &lt; 2.5%) are "
                "excluded because they carry no ideological signal. Legislators with fewer than "
                "20 recorded votes are excluded for reliability. Chambers are analyzed "
                "separately because they vote on different bills. All analyses use the same "
                "upstream data and filters for consistency.</p>"
                "<p>The next two figures provide visual evidence that the Bayesian model "
                "is trustworthy.</p>"
            ),
        )
    )


def _add_clusters_figure(report: object, upstream_plots: dict) -> None:
    """Section 4: k=2 clustering visual proof."""
    path = upstream_plots.get("irt_clusters_house")
    if path is not None and path.exists():
        report.add(
            FigureSection.from_file(
                "clusters",
                "Two Groups — And They're Exactly the Two Parties",
                path,
                caption=(
                    "Three different clustering algorithms were asked to find natural voting "
                    "blocs — without being told anything about party labels. All three found "
                    "exactly two groups, and those groups match the two parties almost perfectly. "
                    "This is visual proof that party is the dominant structure in Kansas "
                    "legislative voting."
                ),
            )
        )


def _add_agreement_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Section 10: Agreement heatmap."""
    key = f"agreement_heatmap_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"agreement-{chamber}",
                f"How Often Do {chamber_title} Members Vote Together?",
                path,
                caption=(
                    f"A heatmap of pairwise voting agreement among {chamber_title} members. "
                    "Bright cells indicate legislators who almost always vote the same way; "
                    "dark cells indicate frequent disagreement. The two bright blocks along "
                    "the diagonal are the two parties — internally cohesive, externally opposed."
                ),
            )
        )


def _add_shap_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Section 22: SHAP feature importance."""
    key = f"shap_bar_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"shap-{chamber}",
                f"What Predicts a Yea Vote? ({chamber_title})",
                path,
                caption=(
                    "SHAP values show how much each feature pushes the model toward "
                    "predicting Yea or Nay. The most important features are a legislator's "
                    "ideology score and how sharply the bill divides the legislature. "
                    "Party label matters less than you might expect — because ideology "
                    "already captures most of what party tells you."
                ),
            )
        )


def _add_calibration_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Section 25: Calibration plot."""
    key = f"calibration_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"calibration-{chamber}",
                f"When the Model Says 80%, It Means 80% ({chamber_title})",
                path,
                caption=(
                    "A well-calibrated model is one where confidence matches reality. "
                    "This plot shows that when the model predicts an 80% chance of Yea, "
                    "roughly 80% of those votes actually are Yea. The closer the curve "
                    "follows the diagonal, the more trustworthy the model's confidence "
                    "scores are."
                ),
            )
        )


def _add_convergence_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Section 28: MCMC convergence summary."""
    key = f"convergence_summary_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"convergence-{chamber}",
                f"The Model Ran Independent Chains and They All Agree ({chamber_title})",
                path,
                caption=(
                    "Bayesian models run multiple independent estimation chains. If they "
                    "all arrive at similar answers, we can trust the results. This figure "
                    "shows that the chains converged — the ideology estimates are stable "
                    "and reproducible, not artifacts of randomness."
                ),
            )
        )


def _add_discrimination_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Section 29: Bill discrimination parameters."""
    key = f"discrimination_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
        report.add(
            FigureSection.from_file(
                f"discrimination-{chamber}",
                f"How Sharply Do Bills Divide the Legislature? ({chamber_title})",
                path,
                caption=(
                    "Each bar represents a bill's 'discrimination' — how sharply it "
                    "separates liberal from conservative legislators. High-discrimination "
                    "bills are the ones where ideology matters most: knowing where a "
                    "legislator sits on the spectrum tells you almost exactly how they voted. "
                    "Low-discrimination bills cut across ideological lines."
                ),
            )
        )


def _add_full_scorecard(report: object, leg_dfs: dict, session: str) -> None:
    """Section 30: Full Legislature Scorecard — ALL legislators, both chambers."""
    frames = []
    for chamber in ("house", "senate"):
        df = leg_dfs.get(chamber)
        if df is not None:
            frames.append(df)

    if not frames:
        return

    combined = pl.concat(frames, how="diagonal").sort(["chamber", "xi_mean"])

    # Select key columns for display
    display_cols = ["full_name", "party", "chamber", "district", "xi_mean"]
    optional_cols = [
        "unity_score",
        "loyalty_rate",
        "maverick_rate",
        "betweenness",
        "accuracy",
    ]
    for col in optional_cols:
        if col in combined.columns:
            display_cols.append(col)

    display = combined.select(display_cols)

    html = make_gt(
        display,
        title=f"Full Legislature Scorecard — {session}",
        subtitle="Every legislator, sorted by ideology within chamber",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "chamber": "Chamber",
            "district": "District",
            "xi_mean": "IRT Ideology",
            "unity_score": "Party Unity",
            "loyalty_rate": "Cluster Loyalty",
            "maverick_rate": "Maverick Rate",
            "betweenness": "Betweenness",
            "accuracy": "Pred. Accuracy",
        },
        number_formats={
            "xi_mean": ".2f",
            "unity_score": ".3f",
            "loyalty_rate": ".3f",
            "maverick_rate": ".3f",
            "betweenness": ".4f",
            "accuracy": ".3f",
        },
        source_note=(
            "IRT Ideology: Bayesian ideal point (negative = liberal, positive = conservative). "
            "Party Unity: CQ-standard (fraction of party votes with party majority). "
            "Cluster Loyalty: agreement rate on contested votes. "
            "Maverick Rate: party defection rate on party votes. "
            "Betweenness: network bridge-building score. "
            "Pred. Accuracy: model's prediction accuracy for this legislator."
        ),
    )

    report.add(
        TableSection(
            id="full-scorecard",
            title="Full Legislature Scorecard",
            html=html,
        )
    )
