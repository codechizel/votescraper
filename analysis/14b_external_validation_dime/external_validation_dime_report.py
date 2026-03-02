"""DIME/CFscore external-validation-specific HTML report builder.

Builds sections for the DIME external validation report:
matching summary, correlation tables, scatter plots, static vs dynamic
comparison, intra-party correlations, outlier analysis, pooled analysis,
DIME vs Shor-McCarty side-by-side, interpretation guide, and references.

Usage (called from external_validation_dime.py):
    from analysis.external_validation_dime_report import build_dime_report
    build_dime_report(ctx.report, all_results=..., ...)
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
    from analysis.external_validation_dime_data import (
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_GIVERS,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
    )
except ModuleNotFoundError:
    from external_validation_dime_data import (  # type: ignore[no-redef]
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        MIN_GIVERS,
        MIN_MATCHED,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
    )


def build_dime_report(
    report: ReportBuilder,
    *,
    all_results: dict[str, dict],
    pooled_results: dict[str, dict],
    sm_comparison: dict[str, dict] | None,
    dime_total: int,
    sessions: list[str],
    models: list[str],
    plots_dir: Path,
) -> None:
    """Build the full DIME external validation HTML report."""
    findings = _generate_dime_key_findings(all_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_how_to_read(report)
    _add_dime_summary(report, dime_total, sessions)
    _add_matching_summary(report, all_results)
    _add_correlation_summary(report, all_results, pooled_results)

    # Scatter plots per session/chamber/model
    for key, data in sorted(all_results.items()):
        model = data["model"]
        chamber = data["chamber"]
        session = data["session"]
        _add_scatter_figure(report, plots_dir, model, chamber, session)

    _add_static_vs_dynamic(report, all_results)
    _add_intra_party_correlations(report, all_results)
    _add_outlier_analysis(report, all_results)

    if pooled_results:
        _add_pooled_analysis(report, pooled_results, plots_dir)

    if sm_comparison:
        _add_sm_comparison(report, sm_comparison)

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
                "<p>This report compares our IRT ideal points to DIME/CFscores — "
                "campaign-finance-based ideology scores from Stanford's DIME project. "
                "CFscores measure who <em>funds</em> a legislator, while our IRT scores "
                "measure how they <em>vote</em>. High correlation means the money agrees "
                "with the votes.</p>"
                "<p>Key things to look for:</p>"
                "<ul>"
                "<li><strong>Correlation Summary table</strong> — "
                "The single most important result. Overall Pearson r of 0.75-0.90 is expected "
                "(literature range for state legislatures). Below 0.70, investigate.</li>"
                "<li><strong>Scatter plots</strong> — Visual check. "
                "Points should cluster along a line, with clear party separation.</li>"
                "<li><strong>Static vs Dynamic comparison</strong> — "
                "Static CFscores (career) should correlate similarly to Shor-McCarty. "
                "Dynamic CFscores (per-cycle) may be noisier but more temporally aligned.</li>"
                "<li><strong>Intra-party correlations</strong> — "
                "Expected to be lower (r ≈ 0.50-0.70) because CFscores discriminate "
                "poorly within parties due to access-motivated donations.</li>"
                "<li><strong>DIME vs SM side-by-side</strong> — "
                "For bienniums with both sources, triangulation strengthens validation.</li>"
                "</ul>"
            ),
        )
    )


def _add_dime_summary(report: ReportBuilder, dime_total: int, sessions: list[str]) -> None:
    report.add(
        TextSection(
            id="dime-summary",
            title="DIME/CFscores Methodology",
            html=(
                "<p><strong>Source:</strong> Bonica, Adam. 2024. "
                '"Database on Ideology, Money in Politics, and Elections (DIME)." '
                "Stanford University Libraries. "
                '<a href="https://data.stanford.edu/dime">data.stanford.edu/dime</a>.</p>'
                "<p><strong>License:</strong> ODC-BY (Open Data Commons Attribution).</p>"
                "<p><strong>Methodology:</strong> CFscores estimate candidate ideology from "
                "campaign finance records using correspondence analysis (SVD) on the "
                "donor-recipient matrix. Donors who give to multiple candidates create "
                "ideological links between those candidates (the 'donor bridge' mechanism).</p>"
                "<p><strong>Two score types:</strong></p>"
                "<ul>"
                "<li><strong>Static CFscore</strong> (<code>recipient.cfscore</code>): "
                "career-level score across all cycles. Primary comparison target.</li>"
                "<li><strong>Dynamic CFscore</strong> (<code>recipient.cfscore.dyn</code>): "
                "per-cycle score. Noisier but temporally aligned with our "
                "session-specific IRT.</li>"
                "</ul>"
                f"<p><strong>Kansas state legislators in DIME:</strong> {dime_total} records.</p>"
                f"<p><strong>Sessions analyzed:</strong> {', '.join(sessions)}</p>"
                "<p><strong>Key difference from Shor-McCarty:</strong> CFscores measure "
                "<em>donor ideology</em> (who funds you), not <em>voting ideology</em> "
                "(how you vote). These are related but distinct constructs. Overall "
                "correlation between CFscores and roll-call ideology is typically "
                "r ≈ 0.75-0.90 at the state level.</p>"
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
                "DIME N": data["n_dime"],
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
        subtitle="How many legislators were matched between our data and DIME?",
        source_note=(
            "Match Rate = Matched / Our N. Phase 1: exact normalized name. "
            "Phase 2: last-name fallback. "
            f"DIME filtered to incumbents with ≥ {MIN_GIVERS} donors."
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
        title="Correlation Summary — Our IRT vs DIME CFscores",
        subtitle="Pearson r and Spearman ρ per session/chamber/model (static CFscore)",
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
            "CI = 95% Fisher z confidence interval. "
            "Literature expectation for state legislatures: r ≈ 0.75-0.90 overall."
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
    path = plots_dir / f"scatter_dime_{model}_{ch}_{session}.png"
    if path.exists():
        model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
        report.add(
            FigureSection.from_file(
                f"fig-scatter-dime-{model}-{ch}-{session}",
                f"{chamber} {model_label} vs DIME CFscore ({session})",
                path,
                caption=(
                    "Each dot is a legislator. X-axis: our IRT ideal point. "
                    "Y-axis: DIME static CFscore. Tight clustering along the "
                    "dashed line indicates strong agreement."
                ),
                alt_text=(
                    f"Scatter plot of {model_label} ideal points vs "
                    f"DIME CFscores for {chamber} ({session}). "
                    "Points cluster along the diagonal, "
                    "showing agreement."
                ),
            )
        )


def _add_static_vs_dynamic(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for key, data in sorted(all_results.items()):
        static_corr = data.get("correlations", {})
        dyn_corr = data.get("dynamic_correlations")
        if not dyn_corr or dyn_corr.get("quality") == "insufficient_data":
            continue
        rows.append(
            {
                "Session": data["session"],
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "Static r": static_corr.get("pearson_r", float("nan")),
                "Dynamic r": dyn_corr.get("pearson_r", float("nan")),
                "Static ρ": static_corr.get("spearman_rho", float("nan")),
                "Dynamic ρ": dyn_corr.get("spearman_rho", float("nan")),
                "n": static_corr.get("n", 0),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Static vs Dynamic CFscore Comparison",
        subtitle="Career-level (static) vs per-cycle (dynamic) CFscore correlations with our IRT",
        number_formats={
            "Static r": ".3f",
            "Dynamic r": ".3f",
            "Static ρ": ".3f",
            "Dynamic ρ": ".3f",
        },
        source_note=(
            "Static = recipient.cfscore (career average). "
            "Dynamic = recipient.cfscore.dyn (per-cycle). "
            "Dynamic may be noisier for candidates with few per-cycle donors."
        ),
    )
    report.add(
        TableSection(
            id="static-vs-dynamic",
            title="Static vs Dynamic CFscore Comparison",
            html=html,
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
        subtitle=(
            "Within-party agreement (expected to be lower for CFscores "
            "due to access-motivated giving)"
        ),
        number_formats={"Pearson r": ".3f", "Spearman ρ": ".3f"},
        source_note=(
            "CFscore intra-party correlations are typically r ≈ 0.50-0.70 in the literature. "
            "Access-motivated donations compress within-party CFscore spread, "
            "especially for Kansas Republicans in safe districts."
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
                    "Name": row.get("full_name", row.get("dime_name", "")),
                    "Party": row.get("party", ""),
                    "Our xi": row.get("xi_mean", float("nan")),
                    "CFscore": row.get("recipient_cfscore", float("nan")),
                    "Discrepancy (z)": row.get("discrepancy_z", float("nan")),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"Top {OUTLIER_TOP_N} Outliers by Score Discrepancy",
        subtitle="Legislators whose IRT and CFscore disagree most (z-standardized)",
        number_formats={"Our xi": ".3f", "CFscore": ".3f", "Discrepancy (z)": ".2f"},
        source_note=(
            "Both scores are z-standardized before computing discrepancy. "
            "High discrepancy may indicate access-motivated funding (moderate voting, "
            "centrist donors), ideological shifts, or name matching errors."
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
        path = plots_dir / f"scatter_dime_{model}_{ch}_pooled.png"
        if path.exists():
            model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
            report.add(
                FigureSection.from_file(
                    f"fig-pooled-dime-{model}-{ch}",
                    f"Pooled {chamber} {model_label} vs DIME CFscore",
                    path,
                    caption="Pooled across all overlapping bienniums (unique legislators only).",
                    alt_text=(
                        f"Scatter plot of pooled {model_label} ideal points vs DIME CFscores "
                        f"for {chamber}. Unique legislators across all overlapping bienniums."
                    ),
                )
            )


def _add_sm_comparison(report: ReportBuilder, sm_comparison: dict[str, dict]) -> None:
    rows = []
    for key, data in sorted(sm_comparison.items()):
        rows.append(
            {
                "Session": data["session"],
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "SM r": data.get("sm_pearson_r", float("nan")),
                "DIME r": data.get("dime_pearson_r", float("nan")),
                "SM ρ": data.get("sm_spearman_rho", float("nan")),
                "DIME ρ": data.get("dime_spearman_rho", float("nan")),
                "SM n": data.get("sm_n", 0),
                "DIME n": data.get("dime_n", 0),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="DIME vs Shor-McCarty Side-by-Side (84th-88th Overlap)",
        subtitle="Two independent validation sources compared",
        number_formats={
            "SM r": ".3f",
            "DIME r": ".3f",
            "SM ρ": ".3f",
            "DIME ρ": ".3f",
        },
        source_note=(
            "SM = Shor-McCarty (roll-call ideology). "
            "DIME = CFscore (campaign-finance ideology). "
            "Triangulation: both should positively correlate with our IRT scores, "
            "but SM should correlate more strongly (same construct — voting)."
        ),
    )
    report.add(
        TableSection(
            id="sm-comparison",
            title="DIME vs Shor-McCarty Side-by-Side",
            html=html,
        )
    )


def _add_interpretation_guide(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="interpretation",
            title="Interpretation Guide",
            html=(
                "<p>This table summarizes how to interpret the correlation results. "
                "Note that expected correlations with CFscores are <strong>lower</strong> than "
                "with Shor-McCarty because CFscores measure a different construct "
                "(donor ideology vs voting ideology).</p>"
                "<table style='width:100%; border-collapse:collapse; font-size:14px; "
                "margin-top:12px;'>"
                "<thead><tr>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Pearson r</th>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Interpretation</th>"
                "<th style='text-align:left; border-bottom:2px solid #333; padding:6px;'>"
                "Notes</th>"
                "</tr></thead><tbody>"
                f"<tr><td style='padding:4px 6px;'>≥ {STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Strong agreement</td>"
                "<td style='padding:4px 6px;'>Exceptional — donor ideology closely tracks "
                "voting behavior</td></tr>"
                f"<tr><td style='padding:4px 6px;'>{GOOD_CORRELATION}–{STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Good agreement</td>"
                "<td style='padding:4px 6px;'>Expected for overall (cross-party) correlations "
                "at the state level</td></tr>"
                f"<tr><td style='padding:4px 6px;'>{CONCERN_CORRELATION}–{GOOD_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Moderate agreement</td>"
                "<td style='padding:4px 6px;'>May be normal for intra-party comparisons; "
                "investigate if overall</td></tr>"
                f"<tr><td style='padding:4px 6px;'>< {CONCERN_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Weak agreement</td>"
                "<td style='padding:4px 6px;'>Expected for intra-party; concerning "
                "if overall correlation</td></tr>"
                "</tbody></table>"
            ),
        )
    )


def _add_limitations(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="limitations",
            title="Limitations & Caveats",
            html=(
                "<ol>"
                "<li><strong>Different constructs.</strong> CFscores measure donor ideology; "
                "our IRT scores measure voting ideology. Disagreement does not necessarily "
                "indicate a measurement error — a legislator may vote differently from what "
                "their donors expect.</li>"
                "<li><strong>Within-party discrimination is limited.</strong> CFscores "
                "discriminate poorly within parties (r ≈ 0.60-0.70 in literature). "
                "Access-motivated donations compress within-party spread. Lower "
                "intra-party correlations are expected, not a failure.</li>"
                "<li><strong>Democratic divergence.</strong> Since ~2010, Democratic candidates' "
                "donor networks have shifted left faster than their voting records (Bonica & "
                "Tausanovitch 2022). Kansas Democrats may show CFscores more liberal than "
                "their IRT ideal points.</li>"
                f"<li><strong>Donor count threshold.</strong> Legislators with fewer than "
                f"{MIN_GIVERS} unique donors are excluded. Their CFscores may be unreliable "
                "noise.</li>"
                "<li><strong>Incumbent-only matching.</strong> We match against incumbent "
                "records only. Challengers have CFscores but no roll-call record.</li>"
                "<li><strong>Name matching is imperfect.</strong> Middle names, nicknames, "
                "and name changes may cause mismatches. The unmatched report identifies "
                "these for manual review.</li>"
                "<li><strong>Coverage ends at 2022.</strong> The DIME dataset has no Kansas "
                "state legislature data for the 2024 cycle. The 90th and 91st bienniums "
                "cannot be meaningfully validated (2022 CFscores are too stale).</li>"
                "</ol>"
            ),
        )
    )


def _add_analysis_parameters(report: ReportBuilder) -> None:
    df = pl.DataFrame(
        {
            "Parameter": [
                "Minimum Matched Legislators",
                "Minimum Unique Donors",
                "Strong Correlation Threshold",
                "Good Correlation Threshold",
                "Concern Correlation Threshold",
                "Outlier Top-N",
                "Name Matching",
                "Correlation Methods",
                "Incumbent Filter",
            ],
            "Value": [
                str(MIN_MATCHED),
                str(MIN_GIVERS),
                str(STRONG_CORRELATION),
                str(GOOD_CORRELATION),
                str(CONCERN_CORRELATION),
                str(OUTLIER_TOP_N),
                "Two-phase: exact normalized name, then last-name fallback",
                "Pearson r, Spearman ρ, Fisher z 95% CI",
                "ico.status == 'I' (incumbents only)",
            ],
            "Description": [
                "Minimum legislators per chamber to compute correlations",
                "Minimum unique donors for a reliable CFscore",
                "Pearson |r| at or above this is 'strong'",
                "Pearson |r| at or above this is 'good'",
                "Pearson |r| below this is 'concern'",
                "Number of outliers to report per session/chamber/model",
                "Phase 1 handles ~90% of matches; Phase 2 catches name divergences",
                "Pearson for linear; Spearman for rank-order; Fisher z for CI",
                "Only match legislators who actually served (not challengers)",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/external_validation_dime.md for justification.",
    )
    report.add(TableSection(id="analysis-params", title="Analysis Parameters", html=html))


def _generate_dime_key_findings(all_results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from DIME validation results."""
    findings: list[str] = []

    if not all_results:
        return findings

    # Match rate and correlation
    n_results = len(all_results)
    total_matched = 0
    total_r = 0.0
    n_valid = 0
    for data in all_results.values():
        matched = data.get("n_matched", 0)
        total_matched += matched
        r = data.get("pearson_r")
        if r is not None:
            total_r += abs(r)
            n_valid += 1

    if n_valid > 0:
        mean_r = total_r / n_valid
        findings.append(
            f"<strong>{n_results}</strong> IRT-vs-CFscore correlations, "
            f"mean |r| = <strong>{mean_r:.3f}</strong>."
        )

    if total_matched > 0:
        findings.append(
            f"<strong>{total_matched}</strong> total legislator-CFscore matches "
            f"across all sessions."
        )

    # SM comparison note (if sm_comparison is in any result)
    for data in all_results.values():
        sm_r = data.get("sm_pearson_r")
        dime_r = data.get("pearson_r")
        if sm_r is not None and dime_r is not None:
            findings.append(
                f"SM |r| = {abs(sm_r):.3f} vs DIME |r| = {abs(dime_r):.3f} "
                f"for {data.get('session', '?')} {data.get('chamber', '?')}."
            )
            break

    return findings


def _add_references(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="references",
            title="References",
            html=(
                "<ul style='font-size:14px;'>"
                "<li>Bonica, A. 2014. "
                '"Mapping the Ideological Marketplace." '
                "<em>American Journal of Political Science</em> 58(2): 367-386. "
                '<a href="https://doi.org/10.1111/ajps.12062">'
                "doi:10.1111/ajps.12062</a></li>"
                "<li>Bonica, A. 2018. "
                '"Inferring Roll-Call Scores from Campaign Contributions Using '
                'Supervised Machine Learning." '
                "<em>American Journal of Political Science</em> 62(4): 830-848. "
                '<a href="https://doi.org/10.1111/ajps.12376">'
                "doi:10.1111/ajps.12376</a></li>"
                "<li>Bonica, A. 2024. "
                '"Database on Ideology, Money in Politics, and Elections (DIME)." '
                "Stanford University Libraries. "
                '<a href="https://data.stanford.edu/dime">'
                "data.stanford.edu/dime</a> (ODC-BY license)</li>"
                "<li>Bonica, A. and C. Tausanovitch. 2022. "
                '"The Ideological Nationalization of Congressional Campaigns." '
                "<em>Legislative Studies Quarterly</em> 47(4): 921-953.</li>"
                "<li>Hill, S.J. and G.A. Huber. 2019. "
                '"On the Meaning of Campaign Contributions." '
                "<em>Journal of Politics</em> 81(2): 755-762.</li>"
                "</ul>"
            ),
        )
    )
