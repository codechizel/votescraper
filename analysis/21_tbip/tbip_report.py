"""Text-Based Ideal Points (Phase 18b) HTML report builder.

Builds sections for the TBIP report: key findings, methodology,
Kansas context, dataset summary, correlation table, scatter plots,
party distributions, intra-party correlations, outliers, PCA scree,
interpretation guide, and analysis parameters.

Usage (called from tbip.py):
    from analysis.tbip_report import build_tbip_report
    build_tbip_report(ctx.report, all_results=..., ...)
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
    from analysis.tbip_data import (
        GOOD_CORRELATION,
        MIN_BILLS,
        MIN_MATCHED,
        MODERATE_CORRELATION,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
    )
except ModuleNotFoundError:
    from tbip_data import (  # type: ignore[no-redef]
        GOOD_CORRELATION,
        MIN_BILLS,
        MIN_MATCHED,
        MODERATE_CORRELATION,
        OUTLIER_TOP_N,
        STRONG_CORRELATION,
    )


def build_tbip_report(
    report: ReportBuilder,
    *,
    all_results: dict[str, dict],
    plots_dir: Path,
    session: str,
    models: list[str],
    embedding_model: str,
    min_votes: int,
) -> None:
    """Build the full TBIP HTML report."""
    findings = _generate_key_findings(all_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_methodology(report)
    _add_kansas_context(report)
    _add_dataset_summary(report, all_results)
    _add_correlation_summary(report, all_results)

    # Scatter plots per chamber/model
    for key, data in sorted(all_results.items()):
        model = data["model"]
        chamber = data["chamber"]
        _add_scatter_figure(report, plots_dir, model, chamber)

    # Party distribution plots
    for key, data in sorted(all_results.items()):
        model = data["model"]
        chamber = data["chamber"]
        _add_party_dist_figure(report, plots_dir, model, chamber)

    _add_intra_party_correlations(report, all_results)
    _add_outlier_analysis(report, all_results)

    # PCA scree plots (one per chamber)
    seen_chambers = set()
    for key, data in sorted(all_results.items()):
        chamber = data["chamber"]
        if chamber not in seen_chambers:
            _add_pca_scree_figure(report, plots_dir, chamber)
            seen_chambers.add(chamber)

    _add_interpretation_guide(report)
    _add_analysis_parameters(report, session, models, embedding_model, min_votes)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _generate_key_findings(all_results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from TBIP results."""
    findings: list[str] = []

    if not all_results:
        return findings

    # Average correlation
    valid_rs = []
    for data in all_results.values():
        corr = data.get("correlations", {})
        r = corr.get("pearson_r")
        if r is not None and corr.get("quality") != "insufficient_data":
            valid_rs.append(abs(r))

    if valid_rs:
        mean_r = sum(valid_rs) / len(valid_rs)
        findings.append(
            f"<strong>{len(valid_rs)}</strong> text-vs-IRT correlations, "
            f"mean |r| = <strong>{mean_r:.3f}</strong>."
        )

    # Bills matched
    for data in all_results.values():
        n_bills = data.get("n_bills_matched", 0)
        n_leg = data.get("n_legislators", 0)
        if n_bills > 0:
            findings.append(
                f"<strong>{n_bills}</strong> bills with both text and roll calls, "
                f"<strong>{n_leg}</strong> legislators profiled."
            )
            break

    # PC1 variance
    for data in all_results.values():
        var_ratio = data.get("pc1_variance_ratio")
        if var_ratio is not None:
            findings.append(
                f"PC1 explains <strong>{var_ratio:.1%}</strong> of variance "
                f"in legislator text profiles ({data['chamber']})."
            )
            break

    return findings


def _add_methodology(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="methodology",
            title="Methodology",
            html=(
                "<p>This phase derives text-based ideal points using an "
                "<strong>embedding-vote approach</strong>:</p>"
                "<ol>"
                "<li><strong>Embed bill texts</strong> using FastEmbed (bge-small-en-v1.5, "
                "384 dimensions) — reuses Phase 18 cached embeddings.</li>"
                "<li><strong>Build vote matrix</strong>: each legislator's votes on bills "
                "with embeddings are encoded as +1 (Yea), -1 (Nay), 0 (absent).</li>"
                "<li><strong>Compute text profiles</strong>: for each legislator, "
                "multiply their vote vector by the bill embedding matrix and normalize "
                "by the number of non-absent votes.</li>"
                "<li><strong>Extract PC1</strong>: PCA on the legislator×384-dim "
                "profile matrix. The first principal component is the text-derived "
                "ideal point.</li>"
                "<li><strong>Align signs</strong>: flip text scores if negatively "
                "correlated with IRT ideal points (Republicans positive convention).</li>"
                "<li><strong>Validate</strong>: Pearson r, Spearman ρ, and Fisher z CI "
                "against IRT ideal points from Phase 04 (flat) and Phase 10 (hierarchical).</li>"
                "</ol>"
                "<p>This is <em>not</em> the Text-Based Ideal Points (TBIP) model of "
                "Vafa et al. (2020), which requires individual authorship. Kansas bills "
                "are ~92% committee-sponsored, making classic TBIP inapplicable. "
                "Instead, we use voting behavior to weight bill embeddings — a simpler "
                "but interpretable alternative.</p>"
            ),
        )
    )


def _add_kansas_context(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="kansas-context",
            title="Kansas Context",
            html=(
                "<p>Kansas legislative sponsorship patterns make this approach "
                "particularly appropriate:</p>"
                "<ul>"
                "<li><strong>~92% committee sponsorship</strong> — individual bill authorship "
                "is rare, ruling out authorship-based text models (TBIP, Wordshoal).</li>"
                "<li><strong>Only ~27 individual sponsors</strong> across ~38 bills in the "
                "91st Legislature — insufficient for stable estimates.</li>"
                "<li><strong>Vote-weighted embeddings</strong> use the signal we <em>do</em> "
                "have: how legislators vote on bills whose content we can embed.</li>"
                "</ul>"
                "<p>The key assumption: legislators who vote similarly on bills with "
                "similar text content have similar text-informed ideological profiles.</p>"
            ),
        )
    )


def _add_dataset_summary(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for key, data in sorted(all_results.items()):
        rows.append(
            {
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "Bills Matched": data.get("n_bills_matched", 0),
                "Legislators": data.get("n_legislators", 0),
                "PC1 Var%": f"{data.get('pc1_variance_ratio', 0):.1%}",
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Dataset Summary",
        subtitle="Bills with both text embeddings and roll calls, by chamber",
        source_note=(
            "Bills Matched = intersection of Phase 18 bill texts and rollcall data. "
            "PC1 Var% = fraction of variance in text profiles explained by the first "
            "principal component."
        ),
    )
    report.add(TableSection(id="data-summary", title="Dataset Summary", html=html))


def _add_correlation_summary(
    report: ReportBuilder,
    all_results: dict[str, dict],
) -> None:
    rows = []
    for key, data in sorted(all_results.items()):
        corr = data.get("correlations", {})
        if corr.get("quality") == "insufficient_data":
            continue
        rows.append(
            {
                "Model": data["model"].capitalize(),
                "Chamber": data["chamber"],
                "n": corr.get("n", 0),
                "Pearson r": corr.get("pearson_r", float("nan")),
                "Spearman ρ": corr.get("spearman_rho", float("nan")),
                "CI Lower": corr.get("ci_lower", float("nan")),
                "CI Upper": corr.get("ci_upper", float("nan")),
                "Quality": corr.get("quality", ""),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Correlation Summary — Text Ideal Points vs IRT",
        subtitle="Pearson r and Spearman ρ per chamber/model",
        number_formats={
            "Pearson r": ".3f",
            "Spearman ρ": ".3f",
            "CI Lower": ".3f",
            "CI Upper": ".3f",
        },
        source_note=(
            f"Quality: strong (r ≥ {STRONG_CORRELATION}), "
            f"good ({GOOD_CORRELATION} ≤ r < {STRONG_CORRELATION}), "
            f"moderate ({MODERATE_CORRELATION} ≤ r < {GOOD_CORRELATION}), "
            f"weak (r < {MODERATE_CORRELATION}). "
            "CI = 95% Fisher z confidence interval. "
            "Lower thresholds than Phase 14 — text is further removed from ideology."
        ),
    )
    report.add(TableSection(id="correlation-summary", title="Correlation Summary", html=html))


def _add_scatter_figure(
    report: ReportBuilder,
    plots_dir: Path,
    model: str,
    chamber: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"scatter_{model}_{ch}.png"
    if path.exists():
        model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
        report.add(
            FigureSection.from_file(
                f"fig-scatter-{model}-{ch}",
                f"{chamber} Text Ideal Points vs {model_label}",
                path,
                caption=(
                    "Each dot is a legislator. X-axis: IRT ideal point. "
                    "Y-axis: text-derived ideal point (PC1 of vote-weighted embeddings). "
                    "Tight clustering along the dashed line indicates strong agreement."
                ),
                alt_text=(
                    f"Scatter plot of {model_label} ideal points vs text-derived ideal "
                    f"points for {chamber}. Points cluster along the diagonal, "
                    "showing agreement between text and voting ideology."
                ),
            )
        )


def _add_party_dist_figure(
    report: ReportBuilder,
    plots_dir: Path,
    model: str,
    chamber: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"party_dist_{model}_{ch}.png"
    if path.exists():
        model_label = "Flat IRT" if model == "flat" else "Hierarchical IRT"
        report.add(
            FigureSection.from_file(
                f"fig-party-dist-{model}-{ch}",
                f"{chamber} Party Separation in Text Ideal Points ({model_label})",
                path,
                caption=(
                    "Strip plot showing text-derived ideal point distribution by party. "
                    "Clear separation indicates that text profiles capture partisan "
                    "differences in bill content preferences."
                ),
                alt_text=(
                    f"Strip plot of text-derived ideal points by party for {chamber}. "
                    "Republicans cluster on the right, Democrats on the left."
                ),
            )
        )


def _add_intra_party_correlations(
    report: ReportBuilder,
    all_results: dict[str, dict],
) -> None:
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
        subtitle="Within-party agreement between text and IRT ideal points",
        number_formats={"Pearson r": ".3f", "Spearman ρ": ".3f"},
        source_note=(
            "Intra-party correlations are expected to be lower than overall correlations. "
            "Text profiles capture broad partisan direction well but have less discriminating "
            "power within parties."
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
                    "Model": data["model"].capitalize(),
                    "Chamber": data["chamber"],
                    "Name": row.get("full_name", row.get("legislator_slug", "")),
                    "Party": row.get("party", ""),
                    "IRT xi": row.get("xi_mean", float("nan")),
                    "Text Score": row.get("text_score", float("nan")),
                    "Discrepancy (z)": row.get("discrepancy_z", float("nan")),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"Top {OUTLIER_TOP_N} Outliers by Score Discrepancy",
        subtitle="Legislators whose text and IRT ideal points disagree most",
        number_formats={
            "IRT xi": ".3f",
            "Text Score": ".3f",
            "Discrepancy (z)": ".2f",
        },
        source_note=(
            "Both scores are z-standardized before computing discrepancy. "
            "Outliers may indicate legislators who vote on bills outside their "
            "typical policy area, or whose text profile is shaped by a few "
            "high-impact votes on unusual legislation."
        ),
    )
    report.add(TableSection(id="outliers", title="Outlier Analysis", html=html))


def _add_pca_scree_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    ch = chamber.lower()
    path = plots_dir / f"pca_scree_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-pca-scree-{ch}",
                f"{chamber} PCA Scree Plot (Text Profiles)",
                path,
                caption=(
                    "Explained variance ratio by principal component. A dominant PC1 "
                    "suggests the primary dimension of variation in text profiles "
                    "is ideological (like standard PCA on the vote matrix)."
                ),
                alt_text=(
                    f"Scree plot showing explained variance by principal component "
                    f"for {chamber} text profiles. PC1 dominates."
                ),
            )
        )


def _add_interpretation_guide(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="interpretation",
            title="Interpretation Guide",
            html=(
                "<p>How to interpret the text-vs-IRT correlations:</p>"
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
                "<td style='padding:4px 6px;'>Text profiles capture ideology well</td></tr>"
                f"<tr><td style='padding:4px 6px;'>"
                f"{GOOD_CORRELATION}–{STRONG_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Good agreement</td>"
                "<td style='padding:4px 6px;'>Expected range — text adds signal beyond "
                "vote direction alone</td></tr>"
                f"<tr><td style='padding:4px 6px;'>"
                f"{MODERATE_CORRELATION}–{GOOD_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Moderate agreement</td>"
                "<td style='padding:4px 6px;'>Text captures partisan direction but less "
                "within-party variation</td></tr>"
                f"<tr><td style='padding:4px 6px;'>< {MODERATE_CORRELATION}</td>"
                "<td style='padding:4px 6px;'>Weak agreement</td>"
                "<td style='padding:4px 6px;'>Text and voting may capture different "
                "dimensions — investigate</td></tr>"
                "</tbody></table>"
                "<p style='margin-top:12px;'><strong>Why lower thresholds than "
                "Phase 14?</strong> Text-derived scores are twice removed from ideology: "
                "bill text → embedding → vote weighting → PCA. Each step adds noise. "
                "Phase 14 compares IRT to other direct ideology measures (SM, DIME).</p>"
            ),
        )
    )


def _add_analysis_parameters(
    report: ReportBuilder,
    session: str,
    models: list[str],
    embedding_model: str,
    min_votes: int,
) -> None:
    df = pl.DataFrame(
        {
            "Parameter": [
                "Session",
                "IRT Models",
                "Embedding Model",
                "Embedding Dimensions",
                "Minimum Non-Absent Votes",
                "Minimum Matched Legislators",
                "Minimum Bills",
                "Strong Correlation",
                "Good Correlation",
                "Moderate Correlation",
                "Outlier Top-N",
                "Sign Convention",
                "PCA Components Computed",
            ],
            "Value": [
                session,
                ", ".join(m.capitalize() for m in models),
                embedding_model,
                "384",
                str(min_votes),
                str(MIN_MATCHED),
                str(MIN_BILLS),
                str(STRONG_CORRELATION),
                str(GOOD_CORRELATION),
                str(MODERATE_CORRELATION),
                str(OUTLIER_TOP_N),
                "Republicans positive (aligned with IRT)",
                "min(n_legislators, 384, 10)",
            ],
            "Description": [
                "Legislative session analyzed",
                "IRT model variants validated against",
                "FastEmbed sentence transformer model",
                "bge-small-en-v1.5 output dimensionality",
                "Legislators with fewer non-absent votes excluded",
                "Minimum legislators per chamber to compute correlations",
                "Minimum bills with both embeddings and roll calls",
                f"Pearson |r| ≥ {STRONG_CORRELATION} for 'strong'",
                f"Pearson |r| ≥ {GOOD_CORRELATION} for 'good'",
                f"Pearson |r| ≥ {MODERATE_CORRELATION} for 'moderate'",
                "Number of outliers to report per chamber/model",
                "Flipped if negative Pearson r with IRT xi_mean",
                "Up to 10 components for scree plot",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/tbip.md for methodology justification.",
    )
    report.add(TableSection(id="analysis-params", title="Analysis Parameters", html=html))
