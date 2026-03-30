"""Bifactor IRT HTML report builder (EXPERIMENTAL).

Builds sections for the bifactor IRT report: experimental banner,
model specification, bill classification, ECV/omega_h diagnostics,
convergence, ideal points, factor loadings, and comparison with upstream.

Usage (called from bifactor.py):
    from analysis.bifactor_report import build_bifactor_report
    build_bifactor_report(ctx.report, results=results, ...)
"""

from pathlib import Path

import polars as pl

try:
    from analysis.report import (
        FigureSection,
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
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
        make_interactive_table,
    )


def build_bifactor_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> None:
    """Build the full bifactor IRT HTML report."""
    findings = _generate_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_experimental_banner(report)
    _add_model_specification(report)

    for chamber, result in results.items():
        _add_bill_classification_table(report, result, chamber)
        _add_ecv_section(report, result, chamber)
        _add_convergence_table(report, result, chamber)
        _add_ideal_point_table(report, result, chamber)
        _add_general_forest_figure(report, plots_dir, chamber)
        _add_scatter_figure(report, plots_dir, chamber)
        _add_ecv_bar_figure(report, plots_dir, chamber)
        _add_loadings_figure(report, plots_dir, chamber)
        _add_general_vs_1d_figure(report, plots_dir, chamber)
        _add_correlation_table(report, result, chamber)

    _add_interpretation_guide(report)
    _add_analysis_parameters(report, n_samples, n_tune, n_chains)
    print(f"  Report: {len(report._sections)} sections added")


# ── Key findings ─────────────────────────────────────────────────────────────


def _generate_key_findings(results: dict[str, dict]) -> list[str]:
    findings: list[str] = []
    for chamber, result in results.items():
        bf = result["bifactor_diagnostics"]
        ecv = bf["ecv"]
        omega_h = bf["omega_h"]
        findings.append(
            f"<strong>{chamber}</strong>: ECV = {ecv:.3f}, omega_h = {omega_h:.3f} "
            f"({bf['interpretation']})"
        )
        corrs = result["correlations"]
        if "general_vs_1d_pearson" in corrs:
            r = corrs["general_vs_1d_pearson"]
            findings.append(f"{chamber} theta_G vs 1D IRT: r = {r:.4f}")
    return findings


# ── Private section builders ─────────────────────────────────────────────────


def _add_experimental_banner(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="experimental-status",
            title="Experimental Status",
            html=(
                '<div style="border: 3px solid #E81B23; border-radius: 8px; '
                'padding: 16px; margin: 16px 0; background: #FFF5F5;">'
                '<h3 style="color: #E81B23; margin-top: 0;">EXPERIMENTAL: '
                "Bifactor IRT Model</h3>"
                "<p>This phase fits a <strong>bifactor IRT model</strong> with one general "
                "factor (ideology, loading on all bills) and two specific factors "
                "(partisan bills, bipartisan bills). Bills are classified using Phase 05 "
                "IRT discrimination parameters.</p>"
                "<p>Uses <strong>relaxed convergence thresholds</strong> "
                "(R-hat &lt; 1.05, ESS &gt; 200, divergences &lt; 50). "
                "Specific factor credible intervals may be unreliable for most legislators.</p>"
                "</div>"
            ),
        )
    )


def _add_model_specification(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="model-specification",
            title="Model Specification",
            html=(
                "<p><strong>Bifactor 2-Parameter Logistic IRT</strong></p>"
                "<pre>"
                "P(Yea) = logit^-1(\n"
                "    a_G[j] * theta_G[i]\n"
                "    + a_S1[j] * theta_S1[i] * mask_high[j]\n"
                "    + a_S2[j] * theta_S2[i] * mask_low[j]\n"
                "    - d[j]\n"
                ")\n\n"
                "theta_G, theta_S1, theta_S2  ~ Normal(0, 1)  per legislator\n"
                "a_G                          ~ Normal(0, 1)  all bills\n"
                "a_S1_raw, a_S2_raw           ~ Normal(0, 1)  (masked by bill group)\n"
                "d                            ~ Normal(0, 5)  all bills\n"
                "</pre>"
                "<p><strong>Identification:</strong> Orthogonal by construction "
                "(independent priors). Post-hoc sign flip on theta_G "
                "(Republican mean &gt; 0). No PLT constraints needed.</p>"
                "<p><strong>Bill groups:</strong> High-disc (|beta| &gt; 1.5) "
                "→ specific factor 1; Low-disc (|beta| &lt; 0.5) → specific "
                "factor 2; Medium → general only.</p>"
            ),
        )
    )


def _add_bill_classification_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    classification = result["classification"]
    method = result["classification_method"]

    counts = classification.group_by("bill_group").agg(pl.len().alias("count")).sort("bill_group")

    rows = [
        {
            "Bill Group": row["bill_group"],
            "Count": row["count"],
            "Loads On": (
                "General + Specific 1"
                if row["bill_group"] == "high_disc"
                else "General + Specific 2"
                if row["bill_group"] == "low_disc"
                else "General only"
            ),
        }
        for row in counts.iter_rows(named=True)
    ]

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Bill Classification ({method})",
        source_note=f"Classification method: {method}",
    )
    report.add(
        TableSection(
            id=f"bill-classification-{chamber.lower()}",
            title=f"{chamber} Bill Classification",
            html=html,
        )
    )


def _add_ecv_section(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    bf = result["bifactor_diagnostics"]
    ecv = bf["ecv"]
    omega_h = bf["omega_h"]
    interpretation = bf["interpretation"]

    color = "#28a745" if ecv > 0.70 else "#ffc107" if ecv > 0.60 else "#dc3545"

    report.add(
        TextSection(
            id=f"ecv-bifactor-{chamber.lower()}",
            title=f"{chamber} ECV and Omega Hierarchical",
            html=(
                f'<div style="border-left: 4px solid {color}; padding: 12px 16px; '
                f'margin: 12px 0; background: #f8f9fa;">'
                f"<p><strong>ECV = {ecv:.3f}</strong> &mdash; {interpretation}</p>"
                f"<p><strong>omega_h = {omega_h:.3f}</strong> &mdash; proportion of "
                "total score variance due to the general factor</p>"
                "<p>ECV measures the share of discriminating variance from the general "
                "factor. omega_h is the bifactor reliability of the general factor. "
                "Both should be high (&gt; 0.70) for the general factor to be a "
                "trustworthy ideology score.</p>"
                f"<p>Variance breakdown: General = {bf['sum_aG_sq']:.1f}, "
                f"S1 (partisan) = {bf['sum_aS1_sq']:.1f}, "
                f"S2 (bipartisan) = {bf['sum_aS2_sq']:.1f}</p>"
                "</div>"
            ),
        )
    )


def _add_convergence_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    diag = result["diagnostics"]
    rows = []

    for name in ["theta_G", "theta_S1", "theta_S2", "a_G", "a_S1_raw", "a_S2_raw", "d"]:
        key = f"{name}_rhat_max"
        if key in diag:
            val = diag[key]
            rows.append(
                {
                    "Metric": f"R-hat ({name}) max",
                    "Value": f"{val:.4f}",
                    "Threshold": "< 1.05",
                    "Status": "OK" if val < 1.05 else "WARNING",
                }
            )

    for name in ["theta_G", "theta_S1", "theta_S2", "a_G"]:
        key = f"{name}_ess_min"
        if key in diag:
            val = diag[key]
            rows.append(
                {
                    "Metric": f"ESS ({name}) min",
                    "Value": f"{val:.0f}",
                    "Threshold": "> 200",
                    "Status": "OK" if val > 200 else "WARNING",
                }
            )

    rows.append(
        {
            "Metric": "Divergences",
            "Value": str(diag.get("divergences", "?")),
            "Threshold": "< 50",
            "Status": "OK" if diag.get("divergences", 999) < 50 else "WARNING",
        }
    )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Bifactor Convergence (Relaxed Thresholds)",
        source_note="Relaxed thresholds for experimental bifactor model.",
    )
    report.add(
        TableSection(
            id=f"convergence-bifactor-{chamber.lower()}",
            title=f"{chamber} Convergence (Bifactor)",
            html=html,
        )
    )


def _add_ideal_point_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    ip = result["ideal_points"].sort("theta_G_mean", descending=True)

    display = ip.select(
        "legislator_slug",
        "full_name",
        "party",
        pl.col("theta_G_mean").round(3).alias("theta_G"),
        pl.col("theta_G_hdi_3%").round(3).alias("G_lo"),
        pl.col("theta_G_hdi_97%").round(3).alias("G_hi"),
        pl.col("theta_S1_mean").round(3).alias("theta_S1"),
        pl.col("theta_S2_mean").round(3).alias("theta_S2"),
    )

    html = make_interactive_table(
        display,
        caption=f"{chamber} — Bifactor Ideal Points (sorted by general factor)",
    )
    report.add(
        InteractiveTableSection(
            id=f"ideal-points-bifactor-{chamber.lower()}",
            title=f"{chamber} Bifactor Ideal Points",
            html=html,
        )
    )


def _add_general_forest_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"general_forest_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"general-forest-{chamber.lower()}",
                title=f"{chamber} General Factor Forest Plot",
                path=path,
                caption="General factor (theta_G) with 95% HDIs, ranked.",
                alt_text=f"Forest plot of bifactor general factor for {chamber}",
            )
        )


def _add_scatter_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"bifactor_scatter_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"bifactor-scatter-{chamber.lower()}",
                title=f"{chamber} Bifactor Scatter (theta_G vs theta_S1)",
                path=path,
                caption="General factor vs specific factor 1 (partisan bills), party-colored.",
                alt_text=f"Bifactor scatter plot for {chamber}",
            )
        )


def _add_ecv_bar_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"ecv_bar_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"ecv-bar-{chamber.lower()}",
                title=f"{chamber} Variance Decomposition",
                path=path,
                caption="Proportion of discriminating variance: general vs specific factors.",
                alt_text=f"ECV bar chart for {chamber}",
            )
        )


def _add_loadings_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"factor_loadings_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"factor-loadings-{chamber.lower()}",
                title=f"{chamber} Factor Loadings Heatmap",
                path=path,
                caption="Discrimination parameters (a_G, a_S1, a_S2) for top 50 bills by |a_G|.",
                alt_text=f"Factor loadings heatmap for {chamber}",
            )
        )


def _add_general_vs_1d_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"general_vs_1d_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"general-vs-1d-{chamber.lower()}",
                title=f"{chamber} General Factor vs 1D IRT",
                path=path,
                caption="Correlation between bifactor general factor and 1D IRT ideal points.",
                alt_text=f"Bifactor theta_G vs 1D IRT scatter for {chamber}",
            )
        )


def _add_correlation_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    corrs = result["correlations"]
    if not corrs:
        return

    rows = []
    if "general_vs_1d_pearson" in corrs:
        rows.append(
            {
                "Comparison": "theta_G vs 1D IRT",
                "Pearson r": f"{corrs['general_vs_1d_pearson']:.4f}",
                "Spearman rho": f"{corrs['general_vs_1d_spearman']:.4f}",
            }
        )
    if "general_vs_2d_dim1_pearson" in corrs:
        rows.append(
            {
                "Comparison": "theta_G vs 2D Dim 1",
                "Pearson r": f"{corrs['general_vs_2d_dim1_pearson']:.4f}",
                "Spearman rho": f"{corrs['general_vs_2d_dim1_spearman']:.4f}",
            }
        )

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(df, title=f"{chamber} — Correlations with Upstream Models")
        report.add(
            TableSection(
                id=f"correlations-bifactor-{chamber.lower()}",
                title=f"{chamber} Upstream Correlations",
                html=html,
            )
        )


def _add_interpretation_guide(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="interpretation-guide",
            title="Interpretation Guide",
            html=(
                "<h4>General Factor (theta_G)</h4>"
                "<p>The general factor loads on <strong>all bills</strong>. It captures "
                "the broadest ideological dimension — what is common to all voting behavior. "
                "Positive = conservative, negative = liberal. This should correlate "
                "strongly (r &gt; 0.90) with 1D IRT ideal points.</p>"
                "<h4>Specific Factor 1 (theta_S1, partisan bills)</h4>"
                "<p>Loads only on high-discrimination bills (|beta| &gt; 1.5 from Phase 05). "
                "Captures partisan-specific voting patterns <em>after</em> removing general "
                "ideology. Extreme values indicate legislators whose partisan behavior "
                "diverges from what their general ideology would predict.</p>"
                "<h4>Specific Factor 2 (theta_S2, bipartisan bills)</h4>"
                "<p>Loads only on low-discrimination bills (|beta| &lt; 0.5). Captures the "
                "contrarian/establishment axis — how legislators vote on routine bipartisan "
                "legislation. This is the dimension that the Tyson paradox lives on.</p>"
                "<h4>ECV (Explained Common Variance)</h4>"
                "<p>Proportion of total discriminating variance from the general factor. "
                "ECV &gt; 0.70: unidimensional model adequate. "
                "ECV 0.60-0.70: moderate multidimensionality. "
                "ECV &lt; 0.60: strong multidimensional structure.</p>"
                "<h4>omega_h (Omega Hierarchical)</h4>"
                "<p>Proportion of total score variance attributable to the general factor, "
                "accounting for unique variance. The reliability of using theta_G as an "
                "ideology measure.</p>"
            ),
        )
    )


def _add_analysis_parameters(
    report: ReportBuilder,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> None:
    rows = [
        {"Parameter": "Posterior draws", "Value": str(n_samples)},
        {"Parameter": "Tuning steps", "Value": str(n_tune)},
        {"Parameter": "Chains", "Value": str(n_chains)},
        {"Parameter": "Sampler", "Value": "nutpie (Rust NUTS)"},
        {"Parameter": "R-hat threshold", "Value": "< 1.05 (relaxed)"},
        {"Parameter": "ESS threshold", "Value": "> 200 (relaxed)"},
        {"Parameter": "Max divergences", "Value": "< 50"},
        {"Parameter": "High-disc threshold", "Value": "|beta| > 1.5"},
        {"Parameter": "Low-disc threshold", "Value": "|beta| < 0.5"},
    ]
    df = pl.DataFrame(rows)
    html = make_gt(df, title="Analysis Parameters")
    report.add(
        TableSection(
            id="analysis-parameters",
            title="Analysis Parameters",
            html=html,
        )
    )
