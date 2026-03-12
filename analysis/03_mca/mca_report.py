"""MCA-specific HTML report builder.

Builds ~12 sections (tables + figures) for the Multiple Correspondence Analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from mca.py):
    from analysis.mca_report import build_mca_report
    build_mca_report(ctx.report, results=results, ...)
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


TOP_CONTRIBUTIONS = 20  # Number of top contributions to show per dimension


def build_mca_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    pca_validation: dict[str, dict],
    sensitivity_findings: dict,
    plots_dir: Path,
    n_components: int,
    correction: str,
) -> None:
    """Build the full MCA HTML report by adding sections to the ReportBuilder."""
    findings = _generate_mca_key_findings(results, pca_validation)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    for chamber, result in results.items():
        _add_inertia_summary(report, result, chamber, correction)
        _add_inertia_figure(report, plots_dir, chamber)
        _add_ideological_map_figure(report, plots_dir, chamber)
        _add_biplot_figure(report, plots_dir, chamber)
        _add_dim1_distribution_figure(report, plots_dir, chamber)
        _add_legislator_scores(report, result, chamber)
        _add_top_contributions(report, result, chamber)
        _add_absence_map_figure(report, plots_dir, chamber)
        _add_horseshoe(report, result, chamber)

    if pca_validation:
        _add_pca_validation(report, pca_validation)
        for chamber in results:
            _add_pca_correlation_figure(report, plots_dir, chamber)

    if sensitivity_findings:
        _add_sensitivity_table(report, sensitivity_findings)

    _add_analysis_parameters(report, n_components, correction)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_inertia_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
    correction: str,
) -> None:
    """Table: MCA inertia summary (dimension, eigenvalue, inertia %, cumulative %)."""
    ev_df = result["eigenvalues_df"]

    html = make_gt(
        ev_df,
        title=f"{chamber} — MCA Inertia Summary",
        subtitle=(
            f"{result['scores_df'].height} legislators × {len(result['vote_ids'])} votes "
            f"(correction: {correction})"
        ),
        column_labels={
            "dimension": "Dimension",
            "eigenvalue": "Eigenvalue",
            "inertia_pct": "Inertia (%)",
            "cumulative_pct": "Cumulative (%)",
        },
        number_formats={
            "eigenvalue": ".4f",
            "inertia_pct": ".2f",
            "cumulative_pct": ".2f",
        },
        source_note=(
            f"Inertia corrected using {correction} method. "
            "Raw MCA percentages are much lower than PCA due to indicator matrix inflation — "
            "the correction compensates for this. "
            "Corrected cumulative inertia may exceed 100% — this is a known artifact of "
            "the Greenacre correction and does not indicate an error."
        ),
    )
    report.add(
        TableSection(
            id=f"inertia-summary-{chamber.lower()}",
            title=f"{chamber} MCA Inertia",
            html=html,
        )
    )


def _add_inertia_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"mca_inertia_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-inertia-{chamber.lower()}",
                f"{chamber} MCA Inertia Plot",
                path,
                caption=(
                    f"Individual and cumulative inertia for {chamber}. "
                    "Sharp drop after Dim1 indicates a one-dimensional legislature."
                ),
                alt_text=(
                    f"Bar-and-line chart showing MCA inertia by dimension for {chamber}. "
                    "First dimension captures most inertia with a sharp drop-off."
                ),
            )
        )


def _add_ideological_map_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"mca_ideological_map_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-ideo-{chamber.lower()}",
                f"{chamber} MCA Ideological Map",
                path,
                caption=(
                    f"Legislators in MCA Dim1-Dim2 space ({chamber}). Red = Republican, "
                    "Blue = Democrat. Dim1 positive = conservative direction. "
                    "Unlike PCA, this map uses chi-square distance on categorical votes."
                ),
                alt_text=(
                    f"Scatter plot of legislators in MCA Dim1-Dim2 space for {chamber}. "
                    "Red Republican and blue Democrat clusters separate along Dim1."
                ),
            )
        )


def _add_biplot_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"mca_biplot_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-biplot-{chamber.lower()}",
                f"{chamber} MCA Biplot",
                path,
                caption=(
                    f"Legislators (dots) and vote categories (× markers) in the same space "
                    f"({chamber}). Green × = Yea, Red × = Nay, Gray × = Absent. "
                    "Top-contributing categories are larger. Categories near a legislator "
                    "indicate that legislator frequently chose that response."
                ),
                alt_text=(
                    f"Biplot showing legislators and vote categories in MCA space for {chamber}. "
                    "Yea, Nay, and Absent categories positioned relative to legislator clusters."
                ),
            )
        )


def _add_dim1_distribution_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"mca_dim1_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-dim1dist-{chamber.lower()}",
                f"{chamber} MCA Dim1 Distribution",
                path,
                caption=(
                    f"Kernel density estimate of MCA Dim1 scores by party ({chamber}). "
                    "Comparable to PCA PC1 distribution."
                ),
                alt_text=(
                    f"Density plot of MCA Dim1 scores by party for {chamber}. "
                    "Party distributions show clear separation with some overlap in the center."
                ),
            )
        )


def _add_legislator_scores(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators ranked by Dim1 score."""
    scores = result["scores_df"].sort("Dim1", descending=True)

    display_cols = ["full_name", "party", "district", "Dim1", "Dim2", "absence_pct"]
    available = [c for c in display_cols if c in scores.columns]
    df = scores.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Legislator MCA Scores (ranked by Dim1)",
        subtitle=f"{df.height} legislators, positive Dim1 = conservative",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "Dim1": "Dim1",
            "Dim2": "Dim2",
            "absence_pct": "Absence %",
        },
        number_formats={"Dim1": ".3f", "Dim2": ".3f", "absence_pct": ".1f"},
    )
    report.add(
        TableSection(
            id=f"scores-{chamber.lower()}",
            title=f"{chamber} Legislator Scores",
            html=html,
        )
    )


def _add_top_contributions(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Top contributing categories for Dim1 and Dim2."""
    contribs = result["contributions_df"]

    for dim in [1, 2]:
        ctr_col = f"Dim{dim}_ctr"
        if ctr_col not in contribs.columns:
            continue

        top = contribs.sort(ctr_col, descending=True).head(TOP_CONTRIBUTIONS)
        display = top.select("category", ctr_col)

        html = make_gt(
            display,
            title=f"{chamber} — Top Dim{dim} Contributions",
            subtitle=f"Top {TOP_CONTRIBUTIONS} categories defining Dim{dim}",
            column_labels={
                "category": "Category",
                ctr_col: "Contribution",
            },
            number_formats={ctr_col: ".4f"},
            source_note=(
                "Contributions measure how much each category shapes a dimension. "
                "High-contribution categories are the votes that most separate legislators."
            ),
        )
        report.add(
            TableSection(
                id=f"contributions-dim{dim}-{chamber.lower()}",
                title=f"{chamber} Dim{dim} Contributions",
                html=html,
            )
        )

        # Warn if dimension is absence-dominated
        top5_cats = top.head(5)["category"].to_list()
        n_absent = sum(1 for c in top5_cats if "__Absent" in str(c))
        if n_absent >= 4:
            report.add(
                TextSection(
                    id=f"absence-warning-dim{dim}-{chamber.lower()}",
                    title=f"{chamber} Dim{dim} Absence Warning",
                    html=(
                        f'<div style="background:#fff3cd; border:1px solid #ffc107; '
                        f'border-radius:6px; padding:12px 16px; margin:8px 0;">'
                        f"<strong>Absence-Dominated Dimension:</strong> "
                        f"{n_absent} of the top 5 Dim{dim} contributions are "
                        f"absence categories. This dimension primarily captures "
                        f"attendance patterns rather than ideological variation. "
                        f"Interpret Dim{dim} scores as an attendance axis, not a "
                        f"policy axis.</div>"
                    ),
                )
            )


def _add_absence_map_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"mca_absence_map_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-absence-{chamber.lower()}",
                f"{chamber} Absence Map",
                path,
                caption=(
                    f"Legislators colored by absence rate ({chamber}). "
                    "MCA's advantage over PCA: absences are positioned in the ideological "
                    "space rather than imputed away. If high-absence legislators cluster "
                    "near one party, absence patterns are partisan."
                ),
                alt_text=(
                    f"Scatter plot of legislators colored by absence rate for {chamber}. "
                    "Color gradient shows whether high-absence legislators cluster by party."
                ),
            )
        )


def _add_horseshoe(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Text section: horseshoe effect analysis."""
    hs = result["horseshoe"]
    if hs["detected"]:
        text = (
            f"<p><strong>Horseshoe effect detected</strong> (quadratic R² = {hs['r2']:.3f}).</p>"
            "<p>Dim2 is a quadratic function of Dim1 — this is a mathematical artifact, "
            "not a genuine second dimension. It confirms that the Kansas Legislature is "
            "fundamentally one-dimensional (party explains everything). The arch pattern "
            "is well-documented in correspondence analysis of gradient-structured data "
            "(Greenacre 2017, Le Roux & Rouanet 2004).</p>"
        )
    else:
        text = (
            f"<p>No horseshoe effect detected (quadratic R² = {hs['r2']:.3f}, "
            f"threshold = {0.80}).</p>"
            "<p>Dim2 appears to capture genuine variation beyond the partisan divide.</p>"
        )

    report.add(
        TextSection(
            id=f"horseshoe-{chamber.lower()}",
            title=f"{chamber} Horseshoe Effect",
            html=text,
        )
    )


def _add_pca_validation(report: ReportBuilder, validation: dict[str, dict]) -> None:
    """Table: MCA Dim1 vs PCA PC1 correlation."""
    rows = []
    for chamber, data in validation.items():
        if isinstance(data, dict) and data.get("skipped"):
            continue
        rows.append(
            {
                "chamber": chamber,
                "n_shared": data["n_shared"],
                "spearman_r": data["spearman_r"],
                "spearman_p": data["spearman_p"],
                "verdict": data["verdict"],
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="MCA Dim1 vs PCA PC1 Validation",
        subtitle="Spearman rank correlation between MCA and PCA ideological orderings",
        column_labels={
            "chamber": "Chamber",
            "n_shared": "Shared Legislators",
            "spearman_r": "Spearman r",
            "spearman_p": "p-value",
            "verdict": "Verdict",
        },
        number_formats={
            "spearman_r": ".4f",
            "spearman_p": ".2e",
        },
        source_note=(
            "Spearman r > 0.90 indicates MCA and PCA produce essentially the same "
            "ideological ordering, validating that PCA's linear assumptions don't "
            "distort results for this dataset."
        ),
    )
    report.add(
        TableSection(
            id="pca-validation",
            title="PCA Validation",
            html=html,
        )
    )


def _add_pca_correlation_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"mca_pca_correlation_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-pca-corr-{chamber.lower()}",
                f"{chamber} MCA vs PCA",
                path,
                caption=(
                    f"MCA Dim1 vs PCA PC1 for {chamber}. Points near the trend line "
                    "indicate MCA and PCA agree on the ideological ordering."
                ),
                alt_text=(
                    f"Scatter plot comparing MCA Dim1 and PCA PC1 scores for {chamber}. "
                    "Tight clustering along the trend line shows strong agreement."
                ),
            )
        )


def _add_sensitivity_table(report: ReportBuilder, findings: dict) -> None:
    """Table: Sensitivity comparison across chambers."""
    rows = []
    for chamber, data in findings.items():
        if isinstance(data, dict) and data.get("skipped"):
            continue
        if not isinstance(data, dict):
            continue
        rows.append(
            {
                "chamber": chamber,
                "default_threshold": f"{data['default_threshold'] * 100:.1f}%",
                "sensitivity_threshold": f"{data['sensitivity_threshold'] * 100:.0f}%",
                "default_n_legislators": data["default_n_legislators"],
                "sensitivity_n_legislators": data["sensitivity_n_legislators"],
                "shared_legislators": data["shared_legislators"],
                "pearson_r": data["pearson_r"],
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Sensitivity Analysis — Dim1 Correlation",
        subtitle="Comparing default (2.5%) vs. aggressive (10%) minority thresholds",
        column_labels={
            "chamber": "Chamber",
            "default_threshold": "Default",
            "sensitivity_threshold": "Sensitivity",
            "default_n_legislators": "N Leg. (Default)",
            "sensitivity_n_legislators": "N Leg. (Sens.)",
            "shared_legislators": "Shared Leg.",
            "pearson_r": "Pearson r",
        },
        number_formats={"pearson_r": ".4f"},
        source_note="r > 0.95 indicates robust results.",
    )
    report.add(
        TableSection(
            id="sensitivity",
            title="Sensitivity Analysis",
            html=html,
        )
    )


def _generate_mca_key_findings(
    results: dict[str, dict],
    pca_validation: dict[str, dict],
) -> list[str]:
    """Generate 2-4 key findings from MCA results."""
    findings: list[str] = []

    for chamber, result in results.items():
        ev_df = result.get("eigenvalues_df")
        if ev_df is not None and ev_df.height > 0:
            dim1_inertia = float(ev_df["inertia_pct"][0])
            findings.append(
                f"{chamber} Dim1 captures <strong>{dim1_inertia:.1f}%</strong> of "
                f"corrected inertia."
            )

        hs = result.get("horseshoe", {})
        if hs.get("detected"):
            findings.append(
                f"{chamber} <strong>horseshoe effect detected</strong> "
                f"(R² = {hs['r2']:.3f}) — Dim2 is an artifact, not a genuine dimension."
            )
        elif hs.get("r2") is not None:
            findings.append(
                f"{chamber} no horseshoe effect (R² = {hs['r2']:.3f}) — "
                f"Dim2 captures genuine variation."
            )
        break  # First chamber only

    # PCA agreement
    for chamber, data in pca_validation.items():
        if isinstance(data, dict) and not data.get("skipped"):
            r = data.get("spearman_r")
            if r is not None:
                findings.append(
                    f"MCA Dim1 vs PCA PC1: Spearman <strong>r = {r:.3f}</strong> "
                    f"({data.get('verdict', 'N/A')})."
                )
            break

    return findings


def _add_analysis_parameters(
    report: ReportBuilder,
    n_components: int,
    correction: str,
) -> None:
    """Table: Analysis parameters used in this run."""
    try:
        from analysis.mca import (
            ABSENT_LABEL,
            HORSESHOE_R2_THRESHOLD,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            PCA_VALIDATION_MIN_R,
            SENSITIVITY_THRESHOLD,
            TOP_CONTRIBUTIONS_N,
        )
    except ModuleNotFoundError:
        from mca import (  # type: ignore[no-redef]
            ABSENT_LABEL,
            HORSESHOE_R2_THRESHOLD,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            PCA_VALIDATION_MIN_R,
            SENSITIVITY_THRESHOLD,
            TOP_CONTRIBUTIONS_N,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "N Components",
                "Inertia Correction",
                "Minority Threshold (Default)",
                "Minority Threshold (Sensitivity)",
                "Min Substantive Votes",
                "Absent Label",
                "Horseshoe R² Threshold",
                "PCA Validation Min r",
                "Top Contributions Shown",
            ],
            "Value": [
                str(n_components),
                correction,
                f"{MINORITY_THRESHOLD:.3f} ({MINORITY_THRESHOLD * 100:.1f}%)",
                f"{SENSITIVITY_THRESHOLD:.2f} ({SENSITIVITY_THRESHOLD * 100:.0f}%)",
                str(MIN_VOTES),
                ABSENT_LABEL,
                str(HORSESHOE_R2_THRESHOLD),
                str(PCA_VALIDATION_MIN_R),
                str(TOP_CONTRIBUTIONS_N),
            ],
            "Description": [
                "MCA dimensions extracted per chamber",
                "Benzécri or Greenacre adjustment to raw inertia",
                "Drop votes where minority side < this fraction",
                "Alternative threshold for sensitivity analysis",
                "Drop legislators with fewer substantive (non-Absent) votes",
                "Canonical label for all absence-type categories",
                "Quadratic R² above this flags horseshoe artifact",
                "Minimum Spearman r between MCA Dim1 and PCA PC1",
                "Number of top-contributing categories shown in report",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="Changing these values constitutes a sensitivity analysis.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
