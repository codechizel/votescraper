"""External-validation-specific HTML report builder.

Builds sections for the Shor-McCarty external validation report:
matching summary, correlation tables, scatter plots, outlier analysis,
intra-party correlations, pooled analysis, and interpretation guide.

Usage (called from external_validation.py):
    from analysis.external_validation_report import build_external_validation_report
    build_external_validation_report(ctx.report, all_results=..., ...)
"""

from pathlib import Path

import polars as pl

try:
    from analysis.report import (
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )

try:
    from analysis.external_validation_data import (
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
    )
except ModuleNotFoundError:
    from external_validation_data import (  # type: ignore[no-redef]
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
    )


def build_external_validation_report(
    report: ReportBuilder,
    *,
    all_results: dict[str, dict],
    pooled_results: dict[str, dict],
    sm_total: int,
    sessions: list[str],
    models: list[str],
    plots_dir: Path,
) -> None:
    """Build the full external validation HTML report."""
    findings = _generate_external_key_findings(all_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_how_to_read(report)
    _add_sm_summary(report, sm_total, sessions)
    _add_matching_summary(report, all_results)
    _add_correlation_summary(report, all_results, pooled_results)

    # Scatter plots per session/chamber/model
    for key, data in sorted(all_results.items()):
        model = data["model"]
        chamber = data["chamber"]
        session = data["session"]
        _add_scatter_figure(report, plots_dir, model, chamber, session)

    _add_intra_party_correlations(report, all_results)
    _add_outlier_analysis(report, all_results)

    if pooled_results:
        _add_pooled_analysis(report, pooled_results, plots_dir)

    _add_interpretation_guide(report)
    _add_limitations(report)
    _add_analysis_parameters(report)
    _add_references(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_how_to_read(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html=(
                "<p>This report compares our IRT ideal points to the Shor-McCarty dataset — "
                "the most widely used measure of state legislator ideology in political science. "
                "High correlation means our scores agree with the field standard.</p>"
                "<p>Key things to look for:</p>"
                "<ul>"
                "<li><strong>Correlation Summary table</strong> — "
                "The single most important result. Pearson r > 0.90 is strong "
                "validation; 0.85-0.90 is good; below 0.85, investigate.</li>"
                "<li><strong>Scatter plots</strong> — Visual check. "
                "Points should cluster tightly along a line, "
                "with Republicans and Democrats clearly separated.</li>"
                "<li><strong>Outlier table</strong> — Legislators whose "
                "scores disagree most. These may indicate name matching "
                "errors, convergence issues, or genuine ideological shifts.</li>"
                "<li><strong>Intra-party correlations</strong> — "
                "Within-party agreement. Lower than overall because "
                "within-party variation is smaller, but still positive.</li>"
                "</ul>"
            ),
        )
    )


def _add_sm_summary(report: ReportBuilder, sm_total: int, sessions: list[str]) -> None:
    report.add(
        TextSection(
            id="sm-summary",
            title="Shor-McCarty Dataset Summary",
            html=(
                "<p><strong>Source:</strong> Shor, B. and N. McCarty. 2011. "
                '"The Ideological Mapping of American Legislatures." '
                "<em>American Political Science Review</em> 105(3): 530-551.</p>"
                "<p><strong>Data:</strong> Individual State Legislator Ideology Data (April 2023), "
                '<a href="https://dataverse.harvard.edu/dataset.xhtml?'
                'persistentId=doi:10.7910/DVN/NWSYOS">Harvard Dataverse</a>. CC0 license.</p>'
                "<p><strong>Methodology:</strong> Maps state legislators onto a common ideological "
                "scale using a bridge-legislator approach: members who serve in both state "
                "legislature and U.S. Congress anchor the scale. Then extends to all legislators "
                "via co-voting patterns within each state.</p>"
                f"<p><strong>Kansas legislators:</strong> {sm_total} with ideology scores "
                f"(np_score), covering 1996-2020.</p>"
                f"<p><strong>Sessions analyzed:</strong> {', '.join(sessions)}</p>"
                "<p><strong>Key limitation:</strong> SM scores are career-fixed (one score per "
                "legislator across their entire career). Our IRT scores vary by biennium. "
                "The comparison tests rank-order agreement, not scale equivalence.</p>"
            ),
        )
    )


def _add_matching_summary(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    if not all_results:
        return

    rows = []
    for key, data in sorted(all_results.items()):
        matched_n = data["matched"].height
        rows.append(
            {
                "Session": data["session"],
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "Our N": data["n_ours"],
                "SM N": data["n_sm"],
                "Matched": matched_n,
                "Match Rate": f"{data['match_rate_ours']:.1%}",
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Legislator Matching Summary",
        subtitle="How many legislators were matched between our data and Shor-McCarty?",
        source_note=(
            "Match Rate = Matched / Our N. Phase 1: exact normalized name. "
            "Phase 2: last-name + district tiebreaker."
        ),
    )
    report.add(TableSection(id="matching-summary", title="Matching Summary", html=html))


def _add_correlation_summary(
    report: ReportBuilder,
    all_results: dict[str, dict],
    pooled_results: dict[str, dict],
) -> None:
    rows = []

    for key, data in sorted(all_results.items()):
        corr = data["correlations"]
        rows.append(
            {
                "Session": data["session"],
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "n": corr["n"],
                "Pearson r": corr["pearson_r"],
                "Spearman ρ": corr["spearman_rho"],
                "CI Lower": corr["ci_lower"],
                "CI Upper": corr["ci_upper"],
                "Quality": corr["quality"],
            }
        )

    for key, data in sorted(pooled_results.items()):
        corr = data["correlations"]
        rows.append(
            {
                "Session": "Pooled",
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "n": corr["n"],
                "Pearson r": corr["pearson_r"],
                "Spearman ρ": corr["spearman_rho"],
                "CI Lower": corr["ci_lower"],
                "CI Upper": corr["ci_upper"],
                "Quality": corr["quality"],
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Correlation Summary — Our IRT vs Shor-McCarty",
        subtitle="Pearson r and Spearman ρ per session/chamber/model",
        number_formats={
            "Pearson r": ".3f",
            "Spearman ρ": ".3f",
            "CI Lower": ".3f",
            "CI Upper": ".3f",
        },
        source_note=(
            f"Quality: strong (r ≥ {STRONG_CORRELATION}), "
            f"good ({GOOD_CORRELATION} ≤ r < {STRONG_CORRELATION}), "
            f"moderate ({CONCERN_CORRELATION} ≤ r < {GOOD_CORRELATION}), "
            f"concern (r < {CONCERN_CORRELATION}). "
            f"CI = 95% Fisher z confidence interval for Pearson r."
        ),
    )
    report.add(TableSection(id="correlation-summary", title="Correlation Summary", html=html))


def _add_scatter_figure(
    report: ReportBuilder,
    plots_dir: Path,
    model: str,
    chamber: str,
    session: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"scatter_{model}_{ch}.png"
    if not path.exists():
        # Try session-specific path
        path = plots_dir / f"scatter_{model}_{ch}_{session}.png"
    if path.exists():
        model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
        report.add(
            FigureSection.from_file(
                f"fig-scatter-{model}-{ch}-{session}",
                f"{chamber} {model_label} vs Shor-McCarty ({session})",
                path,
                caption=(
                    "Each dot is a legislator. X-axis: our IRT ideal point. "
                    "Y-axis: Shor-McCarty score. Tight clustering along the "
                    "dashed line indicates strong agreement."
                ),
                alt_text=(
                    f"Scatter plot of {model_label} ideal points vs "
                    f"Shor-McCarty scores for {chamber} ({session}). "
                    "Points cluster along the diagonal, "
                    "showing agreement."
                ),
            )
        )


def _add_intra_party_correlations(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for key, data in sorted(all_results.items()):
        intra = data.get("intra_party", {})
        for party in ["Republican", "Democrat"]:
            if party not in intra:
                continue
            pc = intra[party]
            if pc.get("quality") == "insufficient_data":
                continue
            rows.append(
                {
                    "Session": data["session"],
                    "Model": data["model"].capitalize(),
                    "Chamber": data["chamber"],
                    "Party": party,
                    "n": pc["n"],
                    "Pearson r": pc["pearson_r"],
                    "Spearman ρ": pc["spearman_rho"],
                    "Quality": pc["quality"],
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Intra-Party Correlations",
        subtitle="Within-party agreement (lower than overall due to smaller variance)",
        number_formats={"Pearson r": ".3f", "Spearman ρ": ".3f"},
        source_note=(
            "Intra-party correlations are expected to be lower than overall "
            "because within-party variation is smaller. Positive values indicate "
            "our scores capture within-party ordering."
        ),
    )
    report.add(TableSection(id="intra-party", title="Intra-Party Correlations", html=html))


def _add_outlier_analysis(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for key, data in sorted(all_results.items()):
        outliers = data.get("outliers")
        if outliers is None or outliers.height == 0:
            continue
        for row in outliers.iter_rows(named=True):
            rows.append(
                {
                    "Session": data["session"],
                    "Model": data["model"].capitalize(),
                    "Chamber": data["chamber"],
                    "Name": row.get("full_name", row.get("sm_name", "")),
                    "Party": row.get("party", ""),
                    "Our xi": row.get("xi_mean", float("nan")),
                    "SM np": row.get("np_score", float("nan")),
                    "Discrepancy (z)": row.get("discrepancy_z", float("nan")),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"Top {OUTLIER_TOP_N} Outliers by Score Discrepancy",
        subtitle="Legislators whose IRT and SM scores disagree most (z-standardized)",
        number_formats={"Our xi": ".3f", "SM np": ".3f", "Discrepancy (z)": ".2f"},
        source_note=(
            "Both scores are z-standardized before computing discrepancy. "
            "High discrepancy may indicate name matching errors, convergence "
            "issues, or genuine ideological shifts between our session-specific "
            "scores and SM's career-level scores."
        ),
    )
    report.add(TableSection(id="outliers", title="Outlier Analysis", html=html))


def _add_pooled_analysis(
    report: ReportBuilder,
    pooled_results: dict[str, dict],
    plots_dir: Path,
) -> None:
    rows = []
    for key, data in sorted(pooled_results.items()):
        corr = data["correlations"]
        rows.append(
            {
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "n": corr["n"],
                "Pearson r": corr["pearson_r"],
                "Spearman ρ": corr["spearman_rho"],
                "Quality": corr["quality"],
            }
        )

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title="Pooled Correlations Across All Bienniums",
            subtitle="Unique legislators (most recent session scores used for duplicates)",
            number_formats={"Pearson r": ".3f", "Spearman ρ": ".3f"},
        )
        report.add(TableSection(id="pooled-table", title="Pooled Correlation Summary", html=html))

    # Pooled scatter plots
    for key, data in sorted(pooled_results.items()):
        model = data["model"]
        chamber = data["chamber"]
        ch = chamber.lower()
        path = plots_dir / f"scatter_{model}_{ch}.png"
        if path.exists():
            model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
            report.add(
                FigureSection.from_file(
                    f"fig-pooled-{model}-{ch}",
                    f"Pooled {chamber} {model_label} vs Shor-McCarty",
                    path,
                    caption="Pooled across all overlapping bienniums (unique legislators only).",
                    alt_text=(
                        f"Scatter plot of pooled {model_label} ideal points vs Shor-McCarty scores "
                        f"for {chamber}. Unique legislators across all overlapping bienniums."
                    ),
                )
            )


def _add_interpretation_guide(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="interpretation",
            title="Interpretation Guide",
            html=(
                "<p>This table summarizes how to interpret the correlation results:</p>"
                "<table style='width:100%; border-collapse:collapse; font-size:14px; "
                "margin-top:12px;'>"
                "<thead><tr>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Pearson r</th>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Interpretation</th>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Action</th>"
                "</tr></thead><tbody>"
                f"<tr><td style='padding:4px 6px;'>≥ {STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Strong external validation</td>"
                "<td style='padding:4px 6px;'>Our scores are essentially interchangeable with "
                "the field standard</td></tr>"
                f"<tr><td style='padding:4px 6px;'>{GOOD_CORRELATION}–{STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Good agreement</td>"
                "<td style='padding:4px 6px;'>Differences likely reflect session-specific "
                "dynamics vs career averages</td></tr>"
                f"<tr><td style='padding:4px 6px;'>{CONCERN_CORRELATION}–{GOOD_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Moderate agreement</td>"
                "<td style='padding:4px 6px;'>Investigate specific sessions/chambers; check "
                "convergence diagnostics</td></tr>"
                f"<tr><td style='padding:4px 6px;'>< {CONCERN_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Concern</td>"
                "<td style='padding:4px 6px;'>Check for data quality issues, convergence "
                "failures, or systematic methodology differences</td></tr>"
                "</tbody></table>"
            ),
        )
    )


def _add_limitations(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="limitations",
            title="Limitations",
            html=(
                "<ol>"
                "<li><strong>Career-fixed vs session-specific.</strong> SM scores are computed "
                "across a legislator's entire career; our scores are session-specific. A "
                "legislator who genuinely shifted ideology between sessions will show a "
                "discrepancy that is not a measurement error.</li>"
                "<li><strong>Name matching is imperfect.</strong> Middle names, nicknames, and "
                "name changes (e.g., marriage) may cause mismatches. The unmatched report "
                "identifies these for manual review.</li>"
                "<li><strong>SM methodology.</strong> The bridge-legislator approach anchors "
                "state scores to Congress members. Kansas has few bridge legislators, which "
                "may reduce SM precision for some members.</li>"
                "<li><strong>Coverage gap.</strong> SM data ends at 2020 (88th biennium). "
                "The 89th-91st bienniums (2021-2026) cannot be externally validated by this "
                "method.</li>"
                "<li><strong>84th biennium quality.</strong> ODT-based vote data for 2011-2012 "
                "has ~30% missing (committee-of-the-whole tallies). IRT convergence may be "
                "weaker, reducing correlation.</li>"
                "</ol>"
            ),
        )
    )


def _add_analysis_parameters(report: ReportBuilder) -> None:
    df = pl.DataFrame(
        {
            "Parameter": [
                "Minimum Matched Legislators",
                "Strong Correlation Threshold",
                "Good Correlation Threshold",
                "Concern Correlation Threshold",
                "Outlier Top-N",
                "Name Matching",
                "Correlation Methods",
            ],
            "Value": [
                str(MIN_MATCHED),
                str(STRONG_CORRELATION),
                str(GOOD_CORRELATION),
                str(CONCERN_CORRELATION),
                str(OUTLIER_TOP_N),
                "Two-phase: exact normalized name, then last-name + district",
                "Pearson r, Spearman ρ, Fisher z 95% CI",
            ],
            "Description": [
                "Minimum legislators per chamber to compute correlations",
                "Pearson |r| at or above this is 'strong'",
                "Pearson |r| at or above this is 'good'",
                "Pearson |r| below this is 'concern'",
                "Number of outliers to report per session/chamber/model",
                "Phase 1 handles ~90% of matches; Phase 2 catches middle-name mismatches",
                "Pearson for linear agreement; Spearman for rank-order; Fisher z for CI",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/external_validation.md for justification.",
    )
    report.add(TableSection(id="analysis-params", title="Analysis Parameters", html=html))


def _generate_external_key_findings(all_results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from external validation results."""
    findings: list[str] = []

    if not all_results:
        return findings

    # Count correlations and find strongest
    n_correlations = len(all_results)
    best_r = 0.0
    best_session = ""
    total_r = 0.0
    n_valid = 0
    for key, data in all_results.items():
        r = data.get("pearson_r")
        if r is not None:
            abs_r = abs(r)
            total_r += abs_r
            n_valid += 1
            if abs_r > best_r:
                best_r = abs_r
                best_session = f"{data.get('session', '?')} {data.get('chamber', '?')}"

    findings.append(f"<strong>{n_correlations}</strong> IRT-vs-Shor-McCarty correlations computed.")

    if n_valid > 0:
        mean_r = total_r / n_valid
        findings.append(
            f"Mean |r| = <strong>{mean_r:.3f}</strong> across all session-chamber pairs."
        )

    if best_session:
        findings.append(
            f"Strongest correlation: <strong>{best_session}</strong> (|r| = {best_r:.3f})."
        )

    return findings


def _add_references(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="references",
            title="References",
            html=(
                "<ul style='font-size:14px;'>"
                "<li>Shor, B. and N. McCarty. 2011. "
                '"The Ideological Mapping of American Legislatures." '
                "<em>American Political Science Review</em> 105(3): 530-551. "
                '<a href="https://doi.org/10.1017/S0003055411000153">'
                "doi:10.1017/S0003055411000153</a></li>"
                "<li>Shor, B. and N. McCarty. 2023. "
                '"Individual State Legislator Ideology Data." '
                "Harvard Dataverse. "
                '<a href="https://doi.org/10.7910/DVN/NWSYOS">'
                "doi:10.7910/DVN/NWSYOS</a> (CC0 license)</li>"
                "</ul>"
            ),
        )
    )
