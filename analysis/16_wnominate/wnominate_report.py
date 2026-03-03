"""W-NOMINATE + OC validation HTML report builder.

Builds sections for the Phase 17 report: correlation matrices, scatter plots,
2D W-NOMINATE space, scree plots, fit statistics, OC classification, and the
full legislator comparison table.

Usage (called from wnominate.py):
    from analysis.wnominate_report import build_wnominate_report
    build_wnominate_report(ctx.report, all_results=..., ...)
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
    from analysis.wnominate_data import (
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        STRONG_CORRELATION,
    )
except ModuleNotFoundError:
    from wnominate_data import (  # type: ignore[no-redef]
        CONCERN_CORRELATION,
        GOOD_CORRELATION,
        STRONG_CORRELATION,
    )


def build_wnominate_report(
    report: ReportBuilder,
    *,
    all_results: dict[str, dict],
    session: str,
    dims: int,
    plots_dir: Path,
) -> None:
    """Build the full W-NOMINATE + OC validation HTML report."""
    findings = _generate_wnominate_key_findings(all_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_how_to_read(report)
    _add_executive_summary(report, all_results, session)
    _add_correlation_matrix(report, all_results)
    _add_within_party_correlations(report, all_results)

    for ch, data in sorted(all_results.items()):
        chamber = data["chamber"]
        _add_irt_vs_wnom_scatter(report, plots_dir, ch, chamber, session)
        _add_irt_vs_oc_scatter(report, plots_dir, ch, chamber, session)
        _add_wnom_2d(report, plots_dir, ch, chamber, session)
        _add_scree(report, plots_dir, ch, chamber, session)

    _add_fit_statistics(report, all_results)
    _add_oc_classification(report, all_results)

    for ch, data in sorted(all_results.items()):
        _add_comparison_table(report, data)

    _add_methodology(report, dims)
    _add_references(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_how_to_read(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html=(
                "<p>This report compares our Bayesian IRT ideal points against two "
                "field-standard legislative scaling methods: <strong>W-NOMINATE</strong> "
                "(Poole & Rosenthal) and <strong>Optimal Classification</strong> "
                "(Poole 2000). High correlation validates that our scores measure the "
                "same construct as the methods used in virtually every published paper "
                "on legislative voting.</p>"
                "<p>Key things to look for:</p>"
                "<ul>"
                "<li><strong>3x3 Correlation Matrix</strong> — The headline result. "
                "IRT vs W-NOMINATE r > 0.95 is the standard finding in the literature.</li>"
                "<li><strong>Scatter plots</strong> — Visual check. Points should cluster "
                "tightly along the diagonal, with parties clearly separated.</li>"
                "<li><strong>W-NOMINATE 2D plot</strong> — Shows the 2D ideological space. "
                "Dim 1 = left-right; dim 2 captures residual structure.</li>"
                "<li><strong>Scree plot</strong> — Eigenvalue drop-off. Sharp drop after "
                "dimension 1 confirms unidimensionality.</li>"
                "<li><strong>Fit statistics</strong> — Correct Classification, APRE, GMP "
                "summarize model fit.</li>"
                "<li><strong>Comparison table</strong> — Every legislator's scores and "
                "ranks across all three methods (never truncated).</li>"
                "</ul>"
            ),
        )
    )


def _add_executive_summary(
    report: ReportBuilder, all_results: dict[str, dict], session: str
) -> None:
    lines = [f"<p><strong>Session:</strong> {session}</p>"]
    for ch, data in sorted(all_results.items()):
        chamber = data["chamber"]
        iw = data["correlations"]["irt_wnom"]
        io = data["correlations"]["irt_oc"]

        lines.append(f"<p><strong>{chamber}:</strong></p><ul>")
        lines.append(
            f"<li>IRT vs W-NOMINATE: r = {iw['pearson_r']:.3f}, "
            f"rho = {iw['spearman_rho']:.3f} ({iw['quality']})</li>"
        )
        if io["n"] > 0:
            lines.append(
                f"<li>IRT vs OC: r = {io['pearson_r']:.3f}, "
                f"rho = {io['spearman_rho']:.3f} ({io['quality']})</li>"
            )
        else:
            lines.append("<li>IRT vs OC: not available</li>")

        wo = data["correlations"]["wnom_oc"]
        if wo["n"] > 0:
            lines.append(
                f"<li>W-NOMINATE vs OC: r = {wo['pearson_r']:.3f}, "
                f"rho = {wo['spearman_rho']:.3f} ({wo['quality']})</li>"
            )
        lines.append(f"<li>Legislators: {data['n_legislators']}</li>")
        lines.append("</ul>")

    report.add(
        TextSection(
            id="executive-summary",
            title="Executive Summary",
            html="".join(lines),
        )
    )


def _add_correlation_matrix(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for ch, data in sorted(all_results.items()):
        chamber = data["chamber"]
        corr = data["correlations"]

        for pair_key, pair_label in [
            ("irt_wnom", "IRT vs W-NOMINATE"),
            ("irt_oc", "IRT vs OC"),
            ("wnom_oc", "W-NOMINATE vs OC"),
        ]:
            pc = corr[pair_key]
            if pc["n"] == 0:
                continue
            rows.append(
                {
                    "Chamber": chamber,
                    "Comparison": pair_label,
                    "Pearson r": pc["pearson_r"],
                    "Spearman rho": pc["spearman_rho"],
                    "n": pc["n"],
                    "Quality": pc["quality"],
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="3x3 Correlation Matrix — IRT / W-NOMINATE / OC",
        subtitle="Pearson r and Spearman rho per chamber and method pair",
        number_formats={"Pearson r": ".3f", "Spearman rho": ".3f"},
        source_note=(
            f"Quality: strong (r >= {STRONG_CORRELATION}), "
            f"good ({GOOD_CORRELATION} <= r < {STRONG_CORRELATION}), "
            f"moderate ({CONCERN_CORRELATION} <= r < {GOOD_CORRELATION}), "
            f"concern (r < {CONCERN_CORRELATION})."
        ),
    )
    report.add(TableSection(id="correlation-matrix", title="3x3 Correlation Matrix", html=html))


def _add_within_party_correlations(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for ch, data in sorted(all_results.items()):
        chamber = data["chamber"]
        wp = data.get("within_party", {})
        for party in ["Republican", "Democrat"]:
            if party not in wp:
                continue
            for pair_key, pair_label in [
                ("irt_wnom", "IRT vs W-NOMINATE"),
                ("irt_oc", "IRT vs OC"),
                ("wnom_oc", "W-NOMINATE vs OC"),
            ]:
                pc = wp[party].get(pair_key, {})
                if not pc or pc.get("n", 0) == 0:
                    continue
                rows.append(
                    {
                        "Chamber": chamber,
                        "Party": party,
                        "Comparison": pair_label,
                        "Pearson r": pc["pearson_r"],
                        "Spearman rho": pc["spearman_rho"],
                        "n": pc["n"],
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Within-Party Correlations",
        subtitle="Method agreement within each party (lower than overall due to smaller variance)",
        number_formats={"Pearson r": ".3f", "Spearman rho": ".3f"},
        source_note=(
            "Intra-party correlations are expected to be lower because within-party "
            "variation is smaller. Positive values indicate our scores preserve "
            "within-party ordering."
        ),
    )
    report.add(TableSection(id="within-party", title="Within-Party Correlations", html=html))


def _add_irt_vs_wnom_scatter(
    report: ReportBuilder, plots_dir: Path, ch: str, chamber: str, session: str
) -> None:
    path = plots_dir / f"scatter_irt_vs_wnom_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-wnom-{ch}",
                f"{chamber} — IRT vs W-NOMINATE ({session})",
                path,
                caption=(
                    "Each dot is a legislator. X-axis: our IRT ideal point. "
                    "Y-axis: W-NOMINATE dimension 1. Tight clustering along the "
                    "dashed line indicates strong agreement between methods."
                ),
                alt_text=(
                    f"Scatter plot of IRT ideal points vs W-NOMINATE dimension 1 for {chamber}. "
                    "Tight clustering along the diagonal confirms strong agreement."
                ),
            )
        )


def _add_irt_vs_oc_scatter(
    report: ReportBuilder, plots_dir: Path, ch: str, chamber: str, session: str
) -> None:
    path = plots_dir / f"scatter_irt_vs_oc_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-oc-{ch}",
                f"{chamber} — IRT vs Optimal Classification ({session})",
                path,
                caption=(
                    "Each dot is a legislator. X-axis: our IRT ideal point. "
                    "Y-axis: OC dimension 1. OC is nonparametric, so slightly "
                    "lower correlation than W-NOMINATE is normal."
                ),
                alt_text=(
                    "Scatter plot of IRT ideal points vs Optimal "
                    f"Classification dimension 1 for {chamber}. Points "
                    "cluster along the diagonal with slightly more "
                    "spread than W-NOMINATE."
                ),
            )
        )


def _add_wnom_2d(
    report: ReportBuilder, plots_dir: Path, ch: str, chamber: str, session: str
) -> None:
    path = plots_dir / f"wnom_2d_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-wnom-2d-{ch}",
                f"{chamber} — W-NOMINATE 2D Space ({session})",
                path,
                caption=(
                    "W-NOMINATE places legislators inside a unit circle. "
                    "Dimension 1 (x-axis) = left-right ideology. "
                    "Dimension 2 (y-axis) captures residual structure beyond "
                    "the main ideological divide."
                ),
                alt_text=(
                    f"Scatter plot of legislators in W-NOMINATE 2D space for {chamber}. "
                    "Points within a unit circle; dimension 1 separates parties left to right."
                ),
            )
        )


def _add_scree(report: ReportBuilder, plots_dir: Path, ch: str, chamber: str, session: str) -> None:
    path = plots_dir / f"scree_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-scree-{ch}",
                f"{chamber} — W-NOMINATE Eigenvalues ({session})",
                path,
                caption=(
                    "Eigenvalue scree plot. A sharp drop after dimension 1 confirms "
                    "that Kansas voting is primarily unidimensional (left-right), "
                    "supporting our 1D IRT model as the right default."
                ),
                alt_text=(
                    f"Scree plot of W-NOMINATE eigenvalues for {chamber}. "
                    "Sharp drop after dimension 1 confirms unidimensional voting structure."
                ),
            )
        )


def _add_fit_statistics(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for ch, data in sorted(all_results.items()):
        chamber = data["chamber"]
        fs = data.get("fit_stats", {})
        if not fs:
            continue

        for method in ["wnominate", "oc"]:
            cc = fs.get(f"{method}_correctClassification")
            apre = fs.get(f"{method}_APRE")
            gmp = fs.get(f"{method}_GMP")
            if cc is not None:

                def _to_float(v: object) -> float:
                    if v is None or v == "NA":
                        return float("nan")
                    return float(v)

                rows.append(
                    {
                        "Chamber": chamber,
                        "Method": "W-NOMINATE" if method == "wnominate" else "OC",
                        "Correct Classification": _to_float(cc),
                        "APRE": _to_float(apre),
                        "GMP": _to_float(gmp),
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Fit Statistics",
        subtitle="Correct Classification, APRE, and GMP by method and chamber",
        number_formats={
            "Correct Classification": ".3f",
            "APRE": ".3f",
            "GMP": ".3f",
        },
        source_note=(
            "CC = proportion of votes correctly classified. "
            "APRE = aggregate proportional reduction in error. "
            "GMP = geometric mean probability."
        ),
    )
    report.add(TableSection(id="fit-statistics", title="Fit Statistics", html=html))


def _add_oc_classification(report: ReportBuilder, all_results: dict[str, dict]) -> None:
    rows = []
    for ch, data in sorted(all_results.items()):
        chamber = data["chamber"]
        oc_df = data.get("oc_df")
        if oc_df is None or "oc_correct_class" not in oc_df.columns:
            continue

        valid = oc_df.filter(pl.col("oc_correct_class").is_not_null())
        if valid.height == 0:
            continue

        cc_vals = valid["oc_correct_class"].cast(pl.Float64).to_numpy()
        rows.append(
            {
                "Chamber": chamber,
                "Legislators": valid.height,
                "Mean CC": float(cc_vals.mean()),
                "Median CC": float(sorted(cc_vals)[len(cc_vals) // 2]),
                "Min CC": float(cc_vals.min()),
                "Max CC": float(cc_vals.max()),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="OC Per-Legislator Correct Classification",
        subtitle="Summary of individual classification accuracy",
        number_formats={"Mean CC": ".3f", "Median CC": ".3f", "Min CC": ".3f", "Max CC": ".3f"},
        source_note="CC = proportion of votes correctly classified for each legislator.",
    )
    report.add(TableSection(id="oc-classification", title="OC Classification Summary", html=html))


def _add_comparison_table(report: ReportBuilder, data: dict) -> None:
    """Full legislator comparison table — never truncated."""
    chamber = data["chamber"]
    comparison = data["comparison"]

    if comparison.height == 0:
        return

    # Select display columns
    display_cols = ["legislator_slug"]
    if "full_name" in comparison.columns:
        display_cols.append("full_name")
    if "party" in comparison.columns:
        display_cols.append("party")
    display_cols.extend(["irt_score", "irt_rank", "wnom_score", "wnom_rank"])

    if "oc_score" in comparison.columns:
        display_cols.extend(["oc_score", "oc_rank"])

    display_cols.append("max_rank_diff")

    display = comparison.select([c for c in display_cols if c in comparison.columns])

    number_fmts = {"irt_score": ".3f", "wnom_score": ".3f"}
    if "oc_score" in display.columns:
        number_fmts["oc_score"] = ".3f"

    html = make_gt(
        display,
        title=f"{chamber} — Full Legislator Comparison",
        subtitle="All three methods side-by-side with ranks (sorted by IRT rank)",
        number_formats=number_fmts,
        source_note=(
            "Rank 1 = most conservative. max_rank_diff = largest rank disagreement "
            "across methods. Large differences may warrant investigation."
        ),
    )
    report.add(
        TableSection(
            id=f"comparison-{chamber.lower()}",
            title=f"{chamber} Legislator Comparison",
            html=html,
        )
    )


def _add_methodology(report: ReportBuilder, dims: int) -> None:
    report.add(
        TextSection(
            id="methodology",
            title="Methodology Notes",
            html=(
                "<h4>W-NOMINATE</h4>"
                "<p>W-NOMINATE (Weighted NOMINAl Three-step Estimation) is the field-standard "
                "method for scaling roll-call votes. It models each legislator as a point in "
                f"a {dims}-dimensional space and each vote as a pair of outcome points. "
                "Legislators vote for the closer outcome. Parameters are estimated via "
                "conditional maximum likelihood (alternating between legislator and vote "
                "parameters). W-NOMINATE constrains legislators to the unit circle and "
                "includes a signal-to-noise (weight) parameter per dimension.</p>"
                "<h4>Optimal Classification</h4>"
                "<p>OC is a nonparametric alternative. It finds the cutting plane that "
                "maximizes correct classification of votes — no distributional assumptions "
                "about error terms. This makes it robust but potentially noisier for "
                "individual-level comparisons.</p>"
                "<h4>Polarity Identification</h4>"
                "<p>Both methods require a polarity legislator to orient the ideological "
                "direction. We select the legislator with the highest PCA PC1 score and at "
                "least 50% vote participation. This is equivalent to the convention used in "
                "pscl::ideal (PCA-informed identification). After estimation, we verify "
                "sign alignment against our IRT ideal points.</p>"
                "<h4>Why R Subprocess (Not rpy2)?</h4>"
                "<p>The wnominate and oc packages are R-only with complex Fortran backends. "
                "A subprocess call with CSV I/O is simpler, more portable, and matches the "
                "pattern used for emIRT in Phase 16. No compiled R-Python bridge needed.</p>"
            ),
        )
    )


def _generate_wnominate_key_findings(all_results: dict[str, dict]) -> list[str]:
    """Generate key findings for the W-NOMINATE + OC validation report."""
    findings: list[str] = []

    # Collect IRT vs W-NOMINATE correlations across chambers
    irt_wnom_rs: list[float] = []
    for data in all_results.values():
        corr = data.get("correlations", {})
        iw = corr.get("irt_wnom", {})
        r = iw.get("pearson_r")
        if r is not None:
            irt_wnom_rs.append(r)

    if irt_wnom_rs:
        mean_r = sum(irt_wnom_rs) / len(irt_wnom_rs)
        findings.append(
            f"IRT vs W-NOMINATE: mean Pearson r = <strong>{mean_r:.3f}</strong> "
            f"across <strong>{len(irt_wnom_rs)}</strong> chamber(s) "
            f"— {'exceeds' if mean_r >= 0.95 else 'below'} the 0.95 literature benchmark."
        )

    # Collect fit statistics — correct classification rates
    cc_values: list[tuple[str, str, float]] = []
    for data in all_results.values():
        chamber = data.get("chamber", "?")
        fs = data.get("fit_stats", {})
        for method, label in [("wnominate", "W-NOMINATE"), ("oc", "OC")]:
            cc = fs.get(f"{method}_correctClassification")
            if cc is not None:
                cc_values.append((chamber, label, cc))

    if cc_values:
        best = max(cc_values, key=lambda x: x[2])
        findings.append(
            f"Best classification: <strong>{best[1]}</strong> in <strong>{best[0]}</strong> "
            f"correctly classifies <strong>{best[2]:.1%}</strong> of votes."
        )

    # IRT vs OC correlation
    irt_oc_rs: list[float] = []
    for data in all_results.values():
        corr = data.get("correlations", {})
        io = corr.get("irt_oc", {})
        r = io.get("pearson_r")
        n = io.get("n", 0)
        if r is not None and n > 0:
            irt_oc_rs.append(r)

    if irt_oc_rs:
        mean_oc_r = sum(irt_oc_rs) / len(irt_oc_rs)
        findings.append(
            f"IRT vs Optimal Classification: mean r = <strong>{mean_oc_r:.3f}</strong> "
            f"— nonparametric OC confirms the scaling."
        )

    return findings


def _add_references(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="references",
            title="References",
            html=(
                "<ul style='font-size:14px;'>"
                "<li>Poole, K.T. and H. Rosenthal. 1997. <em>Congress: A "
                "Political-Economic History of Roll Call Voting</em>. Oxford University Press.</li>"
                "<li>Poole, K.T. and H. Rosenthal. 2007. <em>Ideology & Congress</em>. "
                "Transaction Publishers. (2nd ed.)</li>"
                "<li>Poole, K.T. 2000. "
                '"Non-Parametric Unfolding of Binary Choice Data." '
                "<em>Political Analysis</em> 8(3): 211-237. "
                "(Optimal Classification method)</li>"
                "<li>Carroll, R., J.B. Lewis, J. Lo, K.T. Poole, and H. Rosenthal. 2013. "
                '"The Structure of Utility in Spatial Models of Voting." '
                "<em>American Journal of Political Science</em> 57(4): 1008-1028. "
                "(wnominate R package)</li>"
                "<li>Clinton, J., S. Jackman, and D. Rivers. 2004. "
                '"The Statistical Analysis of Roll Call Data." '
                "<em>American Political Science Review</em> 98(2): 355-370. "
                "(Bayesian IRT comparison benchmark)</li>"
                "</ul>"
            ),
        )
    )
