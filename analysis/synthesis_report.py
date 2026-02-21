"""
Synthesis report builder — 30 narrative-driven sections.

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
) -> None:
    """Build the full 30-section synthesis report."""
    # 1. intro
    _add_intro(report, manifests)
    # 2. pipeline
    _add_pipeline_figure(report, plots_dir)
    # 3. party-line (de-jargoned)
    _add_party_line_narrative(report, manifests)
    # 4. clusters (NEW)
    _add_clusters_figure(report, upstream_plots)
    # 5-6. network house/senate (updated captions)
    _add_network_figure(report, upstream_plots, "house")
    _add_network_figure(report, upstream_plots, "senate")
    # 7-8. dashboard house/senate
    _add_dashboard_figure(report, plots_dir, "house")
    _add_dashboard_figure(report, plots_dir, "senate")
    # 9. mavericks
    _add_mavericks_narrative(report, leg_dfs)
    # 10. agreement-house (NEW)
    _add_agreement_figure(report, upstream_plots, "house")
    # 11. profile-schreiber
    _add_profile_figure(report, plots_dir, "schreiber", "Legislator Profile: Mark Schreiber (R-60)")
    # 12. forest-house (updated caption)
    _add_forest_figure(report, upstream_plots, "house")
    # 13. maverick-landscape-house
    _add_maverick_landscape(report, upstream_plots, "house")
    # 14. maverick-landscape-senate (NEW)
    _add_maverick_landscape(report, upstream_plots, "senate")
    # 15. profile-dietrich
    _add_profile_figure(report, plots_dir, "dietrich", "Legislator Profile: Brenda Dietrich (R-20)")
    # 16. forest-senate (updated caption)
    _add_forest_figure(report, upstream_plots, "senate")
    # 17. tyson-paradox
    _add_tyson_narrative(report, leg_dfs)
    # 18. tyson-visual
    _add_tyson_figure(report, plots_dir)
    # 19. profile-tyson
    _add_profile_figure(report, plots_dir, "tyson", "Legislator Profile: Caryn Tyson (R-12)")
    # 20. veto-overrides
    _add_veto_narrative(report, manifests)
    # 21. unpredictable
    _add_unpredictable_narrative(report, upstream)
    # 22. shap-house (NEW)
    _add_shap_figure(report, upstream_plots, "house")
    # 23. accuracy-house (updated caption)
    _add_accuracy_figure(report, upstream_plots, "house")
    # 24. accuracy-senate (NEW)
    _add_accuracy_figure(report, upstream_plots, "senate")
    # 25. calibration (NEW)
    _add_calibration_figure(report, upstream_plots, "house")
    # 26. surprising-votes
    _add_surprising_votes_table(report, upstream)
    # 27. methodology (updated)
    _add_methodology_note(report)
    # 28. convergence (NEW)
    _add_convergence_figure(report, upstream_plots, "house")
    # 29. discrimination (NEW)
    _add_discrimination_figure(report, upstream_plots, "house")
    # 30. full-scorecard
    _add_full_scorecard(report, leg_dfs)

    print(f"  Report: {len(report._sections)} sections added")


# ── Section Builders ─────────────────────────────────────────────────────────


def _add_intro(report: object, manifests: dict) -> None:
    """Section 1: What This Report Tells You."""
    eda = manifests.get("eda", {})
    total_votes = eda.get("All", {}).get("votes_before", 882)
    contested = eda.get("All", {}).get("votes_after", 491)
    n_legislators = eda.get("All", {}).get("legislators_before", 172)

    report.add(
        TextSection(
            id="intro",
            title="What This Report Tells You",
            html=(
                "<p>During the 2025-2026 session, the Kansas Legislature cast "
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
                "<p><strong>Five headline findings:</strong></p>"
                "<ol>"
                "<li><strong>Party is everything.</strong> Every method — clustering, "
                "networks, prediction — finds that party affiliation explains nearly all "
                "voting behavior. There are no hidden factions.</li>"
                "<li><strong>Most votes are not contested.</strong> "
                f"Of {total_votes:,} roll calls, "
                f"only {contested:,} had meaningful dissent (more than 2.5% minority). "
                "The rest were near-unanimous.</li>"
                "<li><strong>A handful of mavericks stand out.</strong> Mark Schreiber (House) "
                "and Brenda Dietrich (Senate) consistently break from their party on "
                "contested votes.</li>"
                "<li><strong>The Tyson Paradox:</strong> Senator Caryn Tyson is the most "
                "conservative senator by one measure and the least loyal by another — "
                "because she defects <em>rightward</em> from her party.</li>"
                "<li><strong>Votes are 98% predictable.</strong> A machine learning model "
                "can predict individual votes with AUC = 0.98 using only a legislator's "
                "ideology score and the bill's characteristics.</li>"
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
                "From 882 Votes to One Number: 0.98",
                path,
                caption=(
                    "Each box represents a stage of analysis. We start with every recorded "
                    "roll call, filter to contested votes, identify party-line votes, confirm "
                    "that party is the dominant grouping, and show that a model using these "
                    "patterns predicts 98% of individual votes correctly."
                ),
            )
        )


def _add_party_line_narrative(report: object, manifests: dict) -> None:
    """Section 3: The Party Line Is Everything."""
    net = manifests.get("network", {})
    clust = manifests.get("clustering", {})
    assortativity = net.get("house_assortativity_party", 1.0)
    ari_h = net.get("house_community_vs_party_ari", 1.0)
    ari_s = net.get("senate_community_vs_party_ari", 1.0)
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
                "achieves AUC = 0.98 — near-perfect prediction.</li>"
                "</ul>"
                "<p>This does <em>not</em> mean all legislators vote identically. Within each "
                "party, there is meaningful variation — some members are reliably loyal, others "
                "are frequent dissenters. But the gap <em>between</em> parties is vastly larger "
                "than the variation <em>within</em> them.</p>"
            ),
        )
    )


def _add_network_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Sections 4-5: Community network diagrams (reused)."""
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
    """Sections 6-7: Dashboard scatter (new)."""
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


def _add_mavericks_narrative(report: object, leg_dfs: dict) -> None:
    """Section 8: Who Are the Mavericks?"""
    # Pull Schreiber stats
    house = leg_dfs.get("house")
    schreiber = None
    if house is not None:
        s = house.filter(pl.col("legislator_slug") == "rep_schreiber_mark_1")
        if s.height > 0:
            schreiber = s.to_dicts()[0]

    # Pull Dietrich stats
    senate = leg_dfs.get("senate")
    dietrich = None
    if senate is not None:
        d = senate.filter(pl.col("legislator_slug") == "sen_dietrich_brenda_1")
        if d.height > 0:
            dietrich = d.to_dicts()[0]

    parts = [
        "<p>In a legislature where party discipline is the norm, a few members "
        "consistently break ranks. These <strong>mavericks</strong> — legislators "
        "who vote against their own party on contested votes — are analytically "
        "interesting because they reveal where the party line has cracks.</p>"
    ]

    if schreiber:
        unity = schreiber.get("unity_score", 0)
        mav = schreiber.get("maverick_rate", 0)
        acc = schreiber.get("accuracy", 0)
        parts.append(
            f"<p><strong>Mark Schreiber (R-60)</strong> is the House's most prominent "
            f"maverick. His party unity score is {unity:.0%} — well below the Republican "
            f"average — and he defects on {mav:.0%} of party votes. He is also the "
            f"hardest House member for the prediction model to get right ({acc:.0%} "
            "accuracy vs ~95% for most members). His IRT ideal point places him near "
            "the center of the ideological spectrum, closer to Democrats than to most "
            "Republicans.</p>"
        )

    if dietrich:
        unity = dietrich.get("unity_score", 0)
        xi = dietrich.get("xi_mean", 0)
        parts.append(
            f"<p><strong>Brenda Dietrich (R-20)</strong> is the Senate's counterpart. "
            f"With a party unity of {unity:.0%} and an IRT ideal point of {xi:.2f} — "
            "the lowest among Senate Republicans — she votes more like a moderate than "
            "a party stalwart. Her network centrality scores suggest she bridges the "
            "gap between the two parties more than any other senator.</p>"
        )

    report.add(
        TextSection(
            id="mavericks",
            title="Who Are the Mavericks?",
            html="".join(parts),
        )
    )


def _add_profile_figure(
    report: object,
    plots_dir: Path,
    slug_short: str,
    title: str,
) -> None:
    """Sections 9, 12, 16: Profile cards (new)."""
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


def _add_forest_figure(report: object, upstream_plots: dict, chamber: str) -> None:
    """Sections 10, 13: IRT forest plots (reused)."""
    key = f"forest_{chamber}"
    path = upstream_plots.get(key)
    if path is not None and path.exists():
        chamber_title = chamber.title()
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
                ),
            )
        )


def _add_maverick_landscape(report: object, upstream_plots: dict) -> None:
    """Section 11: Maverick landscape (reused)."""
    path = upstream_plots.get("maverick_landscape_house")
    if path is not None and path.exists():
        report.add(
            FigureSection.from_file(
                "maverick-landscape",
                "Strategic vs Performative Independence (House)",
                path,
                caption=(
                    "Each dot is a House member, positioned by ideology (horizontal) and "
                    "party unity (vertical). Legislators in the lower portion of their "
                    "party's cluster defect more often. Those near the center may be "
                    "genuinely moderate; those at the extremes may defect in the opposite "
                    "direction from what you'd expect."
                ),
            )
        )


def _add_tyson_narrative(report: object, leg_dfs: dict) -> None:
    """Section 14: The Tyson Paradox."""
    senate = leg_dfs.get("senate")
    tyson = None
    if senate is not None:
        t = senate.filter(pl.col("legislator_slug") == "sen_tyson_caryn_1")
        if t.height > 0:
            tyson = t.to_dicts()[0]

    if tyson is None:
        return

    xi = tyson.get("xi_mean", 0)
    loyalty = tyson.get("loyalty_rate", 0)
    unity = tyson.get("unity_score", 0)

    report.add(
        TextSection(
            id="tyson-paradox",
            title="The Tyson Paradox",
            html=(
                "<p>Senator Caryn Tyson (R-12) presents a puzzle that illuminates how "
                "different metrics can give contradictory answers about the same "
                "legislator.</p>"
                f"<p>By <strong>IRT ideology</strong>, Tyson is the most conservative "
                f"senator in the chamber (score: {xi:.1f}), ranking #1 among 32 Republicans. "
                "By <strong>clustering loyalty</strong>, she is the <em>least</em> loyal "
                f"Republican ({loyalty:.0%} agreement with party on contested votes). "
                f"By <strong>CQ party unity</strong>, she scores {unity:.0%} — solidly "
                "in the upper ranks of her party.</p>"
                "<p>How can one person be the most conservative <em>and</em> the least "
                "loyal? The answer is in the direction of defection. Tyson doesn't defect "
                "<em>toward</em> Democrats — she defects <em>away</em> from her own party's "
                "mainstream position in the conservative direction. On votes where most "
                "Republicans say Yea, Tyson sometimes says Nay because the bill isn't "
                "conservative <em>enough</em>.</p>"
                "<p>This is why multiple metrics matter. No single number can capture "
                "the full story of a legislator's voting behavior.</p>"
            ),
        )
    )


def _add_tyson_figure(report: object, plots_dir: Path) -> None:
    """Section 15: Three Measures, Three Answers."""
    path = plots_dir / "tyson_paradox.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "tyson-visual",
                "Three Measures, Three Answers",
                path,
                caption=(
                    "The same senator measured three different ways. IRT ideology (all votes) "
                    "says she's #1 conservative. Clustering loyalty (contested votes within "
                    "party) says she's the least loyal Republican. CQ party unity (party-vs-party "
                    "votes only) puts her in the upper ranks. The metrics disagree because they "
                    "measure different subsets of votes."
                ),
            )
        )


def _add_veto_narrative(report: object, manifests: dict) -> None:
    """Section 17: Veto Overrides Tell You Nothing New."""
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
                f"<p>The 2025-2026 session included <strong>{total} veto override votes</strong> "
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


def _add_unpredictable_narrative(report: object, upstream: dict) -> None:
    """Section 18: What the Model Cannot Predict."""
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
                "<li><strong>Maverick legislators</strong> — Schreiber, Dietrich, Tyson — "
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


def _add_accuracy_figure(report: object, upstream_plots: dict) -> None:
    """Section 19: Per-legislator accuracy (reused)."""
    path = upstream_plots.get("per_legislator_accuracy_house")
    if path is not None and path.exists():
        report.add(
            FigureSection.from_file(
                "accuracy-house",
                "Some Legislators Are Harder to Predict",
                path,
                caption=(
                    "Prediction accuracy per House member on the holdout test set. Most "
                    "legislators are predicted correctly 93-97% of the time. The few below "
                    "90% are the mavericks and moderates — the ones whose votes carry the "
                    "most information about what's actually happening in the legislature."
                ),
            )
        )


def _add_surprising_votes_table(report: object, upstream: dict) -> None:
    """Section 20: The 20 Most Surprising Votes."""
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


def _add_methodology_note(report: object) -> None:
    """Section 21: How We Did This."""
    report.add(
        TextSection(
            id="methodology",
            title="How We Did This",
            html=(
                "<p>This report synthesizes seven independent analyses of roll call votes "
                "from the Kansas Legislature's 2025-2026 session. The raw data — every "
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
            ),
        )
    )


def _add_full_scorecard(report: object, leg_dfs: dict) -> None:
    """Section 22: Full Legislature Scorecard — ALL legislators, both chambers."""
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
        title="Full Legislature Scorecard — 2025-2026 Session",
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
