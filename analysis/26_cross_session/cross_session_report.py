"""Cross-session validation HTML report builder.

Builds ~15 sections (tables, figures, and text) for the cross-session
comparison report. Each section is a small function that slices/aggregates
polars DataFrames and calls make_gt() or FigureSection.from_file().

Usage (called from cross_session.py):
    from analysis.cross_session_report import build_cross_session_report
    build_cross_session_report(ctx.report, results=..., plots_dir=...)
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

try:
    from analysis.cross_session_data import (
        ALIGNMENT_TRIM_PCT,
        CORRELATION_WARN,
        FEATURE_IMPORTANCE_TOP_K,
        SHIFT_THRESHOLD_SD,
    )
except ModuleNotFoundError:
    from cross_session_data import (  # type: ignore[no-redef]
        ALIGNMENT_TRIM_PCT,
        CORRELATION_WARN,
        FEATURE_IMPORTANCE_TOP_K,
        SHIFT_THRESHOLD_SD,
    )


def build_cross_session_report(
    report: ReportBuilder,
    *,
    results: dict,
    plots_dir: Path,
    session_a_label: str,
    session_b_label: str,
) -> None:
    """Build the full cross-session HTML report by adding sections."""
    findings = _generate_cross_session_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_overview(report, results, session_a_label, session_b_label)
    _add_matching_summary(report, results)

    for chamber in sorted(results["chambers"]):
        cr = results[chamber]
        r_val = cr.get("r_value")
        if r_val is not None and r_val < CORRELATION_WARN:
            report.add(
                TextSection(
                    id=f"alignment-warning-{chamber.lower()}",
                    title=f"{chamber} — Alignment Warning",
                    text=(
                        f"<div style='background:#fff3cd;border:1px solid #ffc107;"
                        f"padding:12px 16px;border-radius:6px;margin:8px 0'>"
                        f"<strong>Low ideology correlation (r = {r_val:.3f}).</strong> "
                        f"The cross-session IRT alignment for the {chamber} has "
                        f"r &lt; {CORRELATION_WARN}, which indicates that ideology "
                        f"scores may not be comparable across these two sessions. "
                        f"This can occur when major legislative turnover reshapes "
                        f"the ideological landscape (e.g., wave elections), or when "
                        f"the underlying IRT model captures different latent dimensions "
                        f"in each session. Downstream metrics (ideology shift, prediction "
                        f"transfer) should be interpreted with caution.</div>"
                    ),
                )
            )
        _add_ideology_scatter(report, plots_dir, chamber)
        _add_biggest_movers_figure(report, plots_dir, chamber)
        _add_biggest_movers_table(report, cr["shifted"], chamber)
        _add_shift_distribution(report, plots_dir, chamber)
        _add_turnover_figure(report, plots_dir, chamber)
        _add_metric_stability_table(report, cr["stability"], chamber)

    # Prediction transfer (if available)
    has_prediction = any(
        results.get(ch, {}).get("prediction") is not None for ch in results.get("chambers", [])
    )
    if has_prediction:
        for chamber in sorted(results["chambers"]):
            cr = results[chamber]
            if cr.get("prediction") is not None:
                _add_prediction_summary(
                    report,
                    cr["prediction"],
                    chamber,
                    session_a_label,
                    session_b_label,
                )
                _add_prediction_comparison_figure(report, plots_dir, chamber)
                _add_feature_importance_figure(report, plots_dir, chamber)

    # Freshmen cohort analysis (if available)
    for chamber in sorted(results["chambers"]):
        cr = results[chamber]
        freshmen = cr.get("freshmen")
        if freshmen is not None:
            _add_freshmen_cohort(report, freshmen, chamber, plots_dir)

    # Bloc stability (if available)
    for chamber in sorted(results["chambers"]):
        cr = results[chamber]
        bloc = cr.get("bloc_stability")
        if bloc is not None:
            _add_bloc_stability(report, bloc, chamber, plots_dir)

    _add_detection_validation(report, results)
    _add_methodology(report, results, session_a_label, session_b_label)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_overview(
    report: ReportBuilder,
    results: dict,
    session_a_label: str,
    session_b_label: str,
) -> None:
    matched = results["matched"]
    n_matched = matched.height
    n_chamber_switch = int(matched["is_chamber_switch"].sum())
    n_party_switch = int(matched["is_party_switch"].sum())
    total = n_matched + results["n_departing"] + results["n_new"]
    overlap_pct = n_matched / total * 100

    report.add(
        TextSection(
            id="overview",
            title="Overview",
            html=(
                f"<p>This report compares the <strong>{session_a_label}</strong> "
                f"and <strong>{session_b_label}</strong> Kansas Legislature sessions "
                "to answer four questions: <em>Who moved ideologically?</em> "
                "<em>Are our metrics stable?</em> "
                "<em>Are our predictive models honest?</em> "
                "<em>Do our detection methods generalize?</em></p>"
                f"<p><strong>{n_matched} legislators</strong> served in both "
                f"sessions ({overlap_pct:.0f}% "
                f"overlap). {results['n_departing']} departed after "
                f"{session_a_label}, "
                f"and {results['n_new']} are new in {session_b_label}."
                + (
                    f" {n_chamber_switch} legislator(s) switched chambers."
                    if n_chamber_switch
                    else ""
                )
                + (f" {n_party_switch} legislator(s) switched parties." if n_party_switch else "")
                + "</p>"
            ),
        )
    )


def _add_matching_summary(report: ReportBuilder, results: dict) -> None:
    matched = results["matched"]
    rows = []
    for chamber in ["House", "Senate"]:
        n = matched.filter(pl.col("chamber_b") == chamber).height
        if n > 0:
            rows.append({"Chamber": chamber, "Returning": n})

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title="Returning Legislators by Chamber",
            source_note="Matched by normalized full_name across sessions.",
        )
        report.add(TableSection(id="matching-summary", title="Matching Summary", html=html))


def _add_ideology_scatter(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"ideology_shift_scatter_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-shift-scatter-{chamber.lower()}",
                f"{chamber} — Who Moved?",
                path,
                caption=(
                    "Each dot is a returning legislator. The diagonal line marks 'no change' — "
                    "legislators above the line moved rightward (more conservative), below moved "
                    "leftward (more liberal). Labeled names are the biggest movers."
                ),
                alt_text=(
                    f"Scatter plot of previous vs current IRT ideal points for returning "
                    f"{chamber} legislators. Points off the diagonal indicate ideological "
                    f"shifts between sessions."
                ),
            )
        )


def _add_biggest_movers_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"biggest_movers_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-movers-{chamber.lower()}",
                f"{chamber} — Biggest Ideological Movers",
                path,
                caption=(
                    "Horizontal bars show the magnitude and direction of ideology shift. "
                    "Red bars = moved rightward (more conservative). Blue bars = moved leftward "
                    "(more liberal). Only the top movers are shown."
                ),
                alt_text=(
                    f"Horizontal bar chart of biggest ideological movers in the {chamber}. "
                    f"Red bars indicate rightward shifts, blue bars leftward shifts."
                ),
            )
        )


def _add_biggest_movers_table(report: ReportBuilder, shifted: pl.DataFrame, chamber: str) -> None:
    movers = shifted.filter(pl.col("is_significant_mover")).sort("abs_delta_xi", descending=True)
    if movers.height == 0:
        report.add(
            TextSection(
                id=f"movers-table-{chamber.lower()}",
                title=f"{chamber} — Significant Movers",
                html="<p>No legislators moved more than 1 standard deviation.</p>",
            )
        )
        return

    display = movers.select(
        "full_name",
        "party",
        "xi_a_aligned",
        "xi_b",
        "delta_xi",
        "shift_direction",
        "rank_shift",
    )

    html = make_gt(
        display,
        title=f"{chamber} — Significant Ideological Movers ({display.height} legislators)",
        subtitle="Legislators who shifted more than 1 SD between sessions",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "xi_a_aligned": "Previous Ideology",
            "xi_b": "Current Ideology",
            "delta_xi": "Shift",
            "shift_direction": "Direction",
            "rank_shift": "Rank Change",
        },
        number_formats={
            "xi_a_aligned": ".2f",
            "xi_b": ".2f",
            "delta_xi": "+.2f",
        },
        source_note=(
            "Previous Ideology is the aligned IRT ideal point from the earlier session. "
            "Positive shift = moved rightward (more conservative). "
            "Rank Change: positive = moved rightward in ranking."
        ),
    )
    report.add(
        TableSection(
            id=f"movers-table-{chamber.lower()}",
            title=f"{chamber} Significant Movers",
            html=html,
        )
    )


def _add_shift_distribution(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"shift_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-shift-dist-{chamber.lower()}",
                f"{chamber} — Distribution of Ideology Shifts",
                path,
                caption=(
                    "Histogram of ideology shifts for all returning legislators. The dashed lines "
                    "mark the 'significant mover' threshold (1 SD). Most legislators cluster near "
                    "zero (no change); outliers are the movers highlighted above."
                ),
                alt_text=(
                    f"Histogram of ideology shift magnitudes for returning {chamber} "
                    f"legislators. Most cluster near zero with dashed threshold lines "
                    f"marking significant movers."
                ),
            )
        )


def _add_turnover_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"turnover_impact_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-turnover-{chamber.lower()}",
                f"{chamber} — Turnover Impact on Ideology",
                path,
                caption=(
                    "Ideology distributions of departing, returning, and new legislators. "
                    "If new legislators are further right (or left) than departing ones, "
                    "the chamber's overall composition has shifted."
                ),
                alt_text=(
                    f"Density plot comparing ideology distributions of departing, returning, "
                    f"and new {chamber} legislators. Shows whether turnover shifted the "
                    f"chamber's ideological center."
                ),
            )
        )


def _add_metric_stability_table(
    report: ReportBuilder, stability: pl.DataFrame, chamber: str
) -> None:
    if stability.height == 0:
        return

    display_names = {
        "unity_score": "Party Unity (CQ)",
        "maverick_rate": "Maverick Rate",
        "weighted_maverick": "Weighted Maverick",
        "betweenness": "Network Influence",
        "eigenvector": "Eigenvector Centrality",
        "pagerank": "PageRank",
        "loyalty_rate": "Clustering Loyalty",
        "PC1": "PCA Dimension 1",
    }

    select_cols = ["Metric", "pearson_r", "spearman_rho", "n_legislators"]
    col_labels: dict[str, str] = {
        "Metric": "Metric",
        "pearson_r": "Pearson r",
        "spearman_rho": "Spearman rho",
        "n_legislators": "N",
    }
    num_formats: dict[str, str] = {"pearson_r": ".3f", "spearman_rho": ".3f"}

    # Include new columns if available
    if "icc" in stability.columns:
        select_cols.append("icc")
        col_labels["icc"] = "ICC"
        num_formats["icc"] = ".3f"
    if "stability_interpretation" in stability.columns:
        select_cols.append("stability_interpretation")
        col_labels["stability_interpretation"] = "Reliability"

    display = stability.with_columns(
        pl.col("metric")
        .map_elements(lambda m: display_names.get(m, m), return_dtype=pl.Utf8)
        .alias("Metric")
    ).select(select_cols)

    html = make_gt(
        display,
        title=f"{chamber} — Metric Stability Across Sessions",
        subtitle="How consistent are legislative metrics for returning legislators?",
        column_labels=col_labels,
        number_formats=num_formats,
        source_note=(
            f"Pearson r < {CORRELATION_WARN} would indicate weak stability. "
            "ICC = Intraclass Correlation (Koo & Li 2016): "
            "<0.50 poor, 0.50–0.75 moderate, 0.75–0.90 good, >0.90 excellent."
        ),
    )
    report.add(
        TableSection(
            id=f"stability-{chamber.lower()}",
            title=f"{chamber} Metric Stability",
            html=html,
        )
    )


def _add_prediction_summary(
    report: ReportBuilder,
    prediction: dict,
    chamber: str,
    session_a_label: str,
    session_b_label: str,
) -> None:
    rows = [
        {
            "Direction": f"Train {session_a_label} → Test {session_b_label}",
            "AUC-ROC": prediction["auc_ab"],
            "Accuracy": prediction["acc_ab"],
            "F1": prediction["f1_ab"],
        },
        {
            "Direction": f"Train {session_b_label} → Test {session_a_label}",
            "AUC-ROC": prediction["auc_ba"],
            "Accuracy": prediction["acc_ba"],
            "F1": prediction["f1_ba"],
        },
    ]

    within_rows = []
    if prediction.get("within_auc_a") is not None:
        within_rows.append(
            {
                "Direction": f"Within {session_a_label} (holdout)",
                "AUC-ROC": prediction["within_auc_a"],
                "Accuracy": None,
                "F1": None,
            }
        )
    if prediction.get("within_auc_b") is not None:
        within_rows.append(
            {
                "Direction": f"Within {session_b_label} (holdout)",
                "AUC-ROC": prediction["within_auc_b"],
                "Accuracy": None,
                "F1": None,
            }
        )

    df = pl.DataFrame(within_rows + rows)
    tau = prediction.get("kendall_tau", float("nan"))

    # Detect anti-prediction (AUC < 0.5 = worse than random)
    mean_auc = (prediction["auc_ab"] + prediction["auc_ba"]) / 2
    if mean_auc < 0.5:
        auc_warning = (
            f"⚠ Mean cross-session AUC = {mean_auc:.3f} (worse than random). "
            "This indicates a structural break between sessions — the model learned "
            "patterns that invert across the boundary. Common causes: major legislative "
            "turnover (e.g., Tea Party wave), IRT scale misalignment between sessions "
            "with different ideological structures, or genuine political realignment. "
            "Within-session AUCs remain strong, confirming the features are valid — "
            "the issue is domain transfer, not model quality."
        )
    elif mean_auc < 0.7:
        auc_warning = (
            f"⚠ Mean cross-session AUC = {mean_auc:.3f} (weak generalization). "
            "The model partially transfers but the two sessions have meaningfully "
            "different voting structures."
        )
    else:
        auc_warning = None

    source_note = (
        "Cross-session AUC below within-session AUC is expected — "
        "different bills and political contexts reduce performance. "
        f"Feature importance Kendall's tau = {tau:.3f} "
        f"(top {FEATURE_IMPORTANCE_TOP_K} features)."
    )
    if auc_warning:
        source_note = auc_warning + " " + source_note

    html = make_gt(
        df,
        title=f"{chamber} — Cross-Session Prediction Transfer",
        subtitle="Does a model trained on one session generalize to another?",
        number_formats={"AUC-ROC": ".3f", "Accuracy": ".3f", "F1": ".3f"},
        source_note=source_note,
    )
    report.add(
        TableSection(
            id=f"prediction-{chamber.lower()}",
            title=f"{chamber} Prediction Transfer",
            html=html,
        )
    )


def _add_prediction_comparison_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"prediction_comparison_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-pred-comparison-{chamber.lower()}",
                f"{chamber} — Prediction AUC Comparison",
                path,
                caption=(
                    "Green bars: within-session AUC (model trained and "
                    "tested on same session). Orange bars: cross-session "
                    "AUC (model trained on one session, tested on the "
                    "other). A small drop is expected; a large drop "
                    "indicates session-specific patterns."
                ),
                alt_text=(
                    f"Grouped bar chart comparing within-session and cross-session "
                    f"AUC-ROC for {chamber}. Cross-session bars are typically lower, "
                    f"showing the generalization gap."
                ),
            )
        )


def _add_feature_importance_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"feature_importance_comparison_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-feat-importance-{chamber.lower()}",
                f"{chamber} — Feature Importance Comparison",
                path,
                caption=(
                    "Mean |SHAP| values for the top features in each "
                    "session. Stable rankings indicate the model captures "
                    "generalizable patterns rather than session-specific "
                    "quirks."
                ),
                alt_text=(
                    f"Side-by-side bar chart of SHAP feature importance for {chamber} "
                    f"across two sessions. Consistent rankings suggest generalizable "
                    f"predictive patterns."
                ),
            )
        )


def _add_detection_validation(report: ReportBuilder, results: dict) -> None:
    rows = []
    for chamber in sorted(results["chambers"]):
        det = results[chamber].get("detection", {})
        for role in ["maverick", "bridge", "paradox"]:
            name_a = det.get(f"{role}_a")
            name_b = det.get(f"{role}_b")
            rows.append(
                {
                    "Chamber": chamber,
                    "Role": role.title(),
                    "Previous Session": name_a or "Not detected",
                    "Current Session": name_b or "Not detected",
                    "Same Person?": "Yes" if (name_a and name_b and name_a == name_b) else "No",
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Detection Threshold Validation",
        subtitle="Do the synthesis detection methods identify the same roles across sessions?",
        source_note=(
            "Mavericks, bridge-builders, and metric paradoxes are detected using the same "
            "thresholds on both sessions. Consistency suggests the thresholds generalize."
        ),
    )
    report.add(
        TableSection(
            id="detection-validation",
            title="Detection Validation",
            html=html,
        )
    )


def _generate_cross_session_key_findings(results: dict) -> list[str]:
    """Generate 2-4 key findings from cross-session results."""
    findings: list[str] = []

    matched = results.get("matched")
    if matched is not None and hasattr(matched, "height"):
        findings.append(f"<strong>{matched.height}</strong> legislators matched across sessions.")

    for chamber in sorted(results.get("chambers", [])):
        cr = results.get(chamber, {})

        # IRT correlation
        stability = cr.get("stability")
        if stability is not None and hasattr(stability, "columns"):
            if "pearson_r" in stability.columns and stability.height > 0:
                irt_r = float(stability["pearson_r"][0])
                findings.append(f"{chamber} IRT correlation: <strong>r = {irt_r:.3f}</strong>.")

        # Top shifter
        shifted = cr.get("shifted")
        if shifted is not None and hasattr(shifted, "height") and shifted.height > 0:
            sort_col = "abs_delta_xi" if "abs_delta_xi" in shifted.columns else "delta_xi"
            top = shifted.sort(sort_col, descending=True).head(1)
            name = top["full_name"][0]
            shift = float(top["delta_xi"][0])
            findings.append(
                f"{chamber} top shifter: <strong>{name}</strong> (shift = {shift:+.3f})."
            )

        break  # First chamber only

    return findings


def _add_freshmen_cohort(
    report: ReportBuilder,
    freshmen: object,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Add freshmen vs returning legislators comparison section."""
    chamber_cap = chamber.capitalize()

    lines = [
        f"<p><strong>{freshmen.n_new}</strong> new {chamber_cap} members entered this session, "
        f"joining <strong>{freshmen.n_returning}</strong> returning incumbents.</p>",
        "<table style='width:100%; border-collapse:collapse; font-size:14px; margin:12px 0;'>",
        "<thead><tr>"
        "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>Metric</th>"
        "<th style='text-align:center; border-bottom:2px solid #333; padding:6px;'>New</th>"
        "<th style='text-align:center; border-bottom:2px solid #333; padding:6px;'>Returning</th>"
        "<th style='text-align:center; border-bottom:2px solid #333; padding:6px;'>Test</th>"
        "</tr></thead><tbody>",
    ]

    # Ideology row
    if freshmen.ideology_new_mean is not None:
        sig = (
            f"<strong>p={freshmen.ideology_ks_p:.3f}</strong>"
            if freshmen.ideology_ks_p is not None and freshmen.ideology_ks_p < 0.05
            else f"p={freshmen.ideology_ks_p:.3f}"
            if freshmen.ideology_ks_p is not None
            else "—"
        )
        lines.append(
            "<tr>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee;'>Ideology (mean)</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"{freshmen.ideology_new_mean:+.3f}</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"{freshmen.ideology_returning_mean:+.3f}</td>"
            f"<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"KS {sig}</td></tr>"
        )

    # Unity row
    if freshmen.unity_new_mean is not None:
        sig = (
            f"<strong>p={freshmen.unity_t_p:.3f}</strong>"
            if freshmen.unity_t_p is not None and freshmen.unity_t_p < 0.05
            else f"p={freshmen.unity_t_p:.3f}"
            if freshmen.unity_t_p is not None
            else "—"
        )
        lines.append(
            "<tr>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee;'>Party Unity (mean)</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"{freshmen.unity_new_mean:.3f}</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"{freshmen.unity_returning_mean:.3f}</td>"
            f"<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"t-test {sig}</td></tr>"
        )

    # Maverick row
    if freshmen.maverick_new_mean is not None and freshmen.maverick_returning_mean is not None:
        lines.append(
            "<tr>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee;'>Maverick Rate (mean)</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"{freshmen.maverick_new_mean:.1%}</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            f"{freshmen.maverick_returning_mean:.1%}</td>"
            "<td style='padding:4px 6px; border-bottom:1px solid #eee; text-align:center;'>"
            "—</td></tr>"
        )

    lines.append("</tbody></table>")

    # Add freshmen density overlay plot if it exists
    fig_path = plots_dir / f"freshmen_ideology_{chamber}.png"
    if fig_path.exists():
        report.add(
            FigureSection.from_file(
                f"freshmen-density-{chamber}",
                f"Freshmen vs Returning Ideology — {chamber_cap}",
                fig_path,
                caption=(
                    "Ideology density overlay: new members vs returning incumbents. "
                    "KS test assesses whether the distributions differ significantly."
                ),
                alt_text=(
                    f"Overlapping density plot of ideology for freshmen vs returning "
                    f"{chamber_cap} legislators. Shows whether new members differ "
                    f"ideologically from incumbents."
                ),
            )
        )

    report.add(
        TextSection(
            id=f"freshmen-{chamber}",
            title=f"Freshmen Cohort Analysis — {chamber_cap}",
            html="\n".join(lines),
        )
    )


def _add_bloc_stability(
    report: ReportBuilder,
    bloc: dict,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Add voting bloc stability section with Sankey, transition table, and switchers."""
    chamber_cap = chamber.capitalize()
    ari = bloc["ari"]
    n_paired = bloc["n_paired"]

    # ARI interpretation
    if ari > 0.65:
        ari_label = "strong"
    elif ari > 0.25:
        ari_label = "moderate"
    else:
        ari_label = "weak"

    overview = (
        f"<p>Cluster assignments for <strong>{n_paired}</strong> returning "
        f"{chamber_cap} legislators were compared across sessions. "
        f"Adjusted Rand Index: <strong>{ari:.3f}</strong> ({ari_label} agreement).</p>"
    )

    report.add(
        TextSection(
            id=f"bloc-overview-{chamber}",
            title=f"Voting Bloc Stability — {chamber_cap}",
            html=overview,
        )
    )

    # Sankey diagram (if plotly available)
    sankey_path = plots_dir / f"bloc_sankey_{chamber}.html"
    if sankey_path.exists():
        report.add(
            InteractiveSection(
                id=f"bloc-sankey-{chamber}",
                title=f"Bloc Transition Flow — {chamber_cap}",
                html=sankey_path.read_text(encoding="utf-8"),
                caption="Sankey diagram showing how legislators moved between clusters.",
                aria_label=(
                    f"Interactive Sankey diagram showing {chamber_cap} legislator flow "
                    f"between voting bloc clusters across sessions."
                ),
            )
        )

    # Transition matrix table
    transition_df = bloc.get("transition_df")
    if transition_df is not None and transition_df.height > 0:
        html = make_gt(
            transition_df,
            title=f"Cluster Transition Matrix — {chamber_cap}",
            subtitle="Count of legislators moving from cluster A to cluster B",
            column_labels={
                "cluster_a": "Session A Cluster",
                "cluster_b": "Session B Cluster",
                "count": "Count",
            },
        )
        report.add(
            TableSection(
                id=f"bloc-transition-{chamber}",
                title=f"Transition Matrix — {chamber_cap}",
                html=html,
            )
        )

    # Switchers table
    switchers = bloc.get("switchers")
    if switchers is not None and switchers.height > 0:
        display_cols = ["slug_b", "cluster_a", "cluster_b"]
        labels = {
            "slug_b": "Legislator",
            "cluster_a": "Old Cluster",
            "cluster_b": "New Cluster",
        }
        if "full_name" in switchers.columns:
            display_cols = ["full_name", "party", "cluster_a", "cluster_b"]
            labels = {
                "full_name": "Legislator",
                "party": "Party",
                "cluster_a": "Old Cluster",
                "cluster_b": "New Cluster",
            }

        avail_cols = [c for c in display_cols if c in switchers.columns]
        avail_labels = {k: v for k, v in labels.items() if k in avail_cols}

        html = make_interactive_table(
            switchers.select(avail_cols),
            title=f"Cluster Switchers — {chamber_cap} ({switchers.height})",
            column_labels=avail_labels,
            caption="Legislators who changed cluster assignment between sessions.",
        )
        report.add(
            InteractiveTableSection(
                id=f"bloc-switchers-{chamber}",
                title=f"Cluster Switchers — {chamber_cap}",
                html=html,
            )
        )


def _add_methodology(
    report: ReportBuilder,
    results: dict,
    session_a_label: str,
    session_b_label: str,
) -> None:
    a_coeff = results.get("alignment_coefficients", {})
    alignment_para = ""
    if a_coeff:
        coeff_str = ", ".join(
            f"{ch}: A={coefs['A']:.3f}, B={coefs['B']:.3f}" for ch, coefs in sorted(a_coeff.items())
        )
        alignment_para = f"<p><strong>Alignment coefficients:</strong> {coeff_str}.</p>"

    report.add(
        TextSection(
            id="methodology",
            title="Methodology Notes",
            html=(
                "<p><strong>IRT Scale Alignment:</strong> IRT ideal points from each session "
                "are fitted independently, producing scores on different scales. To compare them, "
                "we use a robust affine transformation (xi_aligned = A &times; xi + B) fitted on "
                f"the {results.get('n_matched', '?')} returning legislators. The top/bottom "
                f"{ALIGNMENT_TRIM_PCT}% of residuals are trimmed before the final fit to prevent "
                "genuine movers from distorting the alignment.</p>"
                + alignment_para
                + "<p><strong>Significant mover threshold:</strong> A legislator is flagged as a "
                f"significant mover if |shift| > {SHIFT_THRESHOLD_SD} &times; SD(all shifts). "
                "This adapts to the overall session-to-session variability.</p>"
                "<p><strong>Legislator matching:</strong> Legislators are matched by normalized "
                "full name (lowercased, leadership suffixes removed). Fuzzy matching is not used "
                "— only exact name matches are included.</p>"
                f"<p><strong>Reference session:</strong> {session_b_label} is the reference scale. "
                f"{session_a_label} ideal points are transformed onto that scale.</p>"
                "<p>See <code>analysis/design/cross_session.md</code> and "
                "<code>docs/adr/0019-cross-session-validation.md</code> for full details.</p>"
            ),
        )
    )
