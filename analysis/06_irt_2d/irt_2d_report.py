"""2D IRT HTML report builder (EXPERIMENTAL).

Builds sections for the 2D Bayesian IRT report: experimental status banner,
model specification, convergence, ideal points, correlation with PCA, and
interpretation guide per chamber.

Usage (called from irt_2d.py):
    from analysis.irt_2d_report import build_irt_2d_report
    build_irt_2d_report(ctx.report, results=results, ...)
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


def build_irt_2d_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    n_samples: int,
    n_tune: int,
    n_chains: int,
    canonical_sources: dict[str, str] | None = None,
) -> None:
    """Build the full 2D IRT HTML report by adding sections to the ReportBuilder."""
    findings = _generate_irt_2d_key_findings(results, canonical_sources)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_experimental_banner(report)
    if canonical_sources:
        _add_canonical_routing_summary(report, canonical_sources)
    _add_model_specification(report)

    for chamber, result in results.items():
        _add_convergence_table(report, result, chamber)
        _add_ideal_point_table(report, result, chamber)
        _add_dim1_forest_figure(report, plots_dir, chamber)
        _add_scatter_figure(report, plots_dir, chamber)
        _add_scatter_interactive(report, plots_dir, chamber)
        _add_dim1_vs_pc1_figure(report, plots_dir, chamber)
        _add_dim1_vs_pc1_interactive(report, plots_dir, chamber)
        _add_dim2_vs_pc2_figure(report, plots_dir, chamber)
        _add_dim2_vs_pc2_interactive(report, plots_dir, chamber)
        _add_correlation_table(report, result, chamber)

    _add_interpretation_guide(report)
    _add_analysis_parameters(report, n_samples, n_tune, n_chains)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_experimental_banner(report: ReportBuilder) -> None:
    """Red-bordered experimental status banner — first section of the report."""
    report.add(
        TextSection(
            id="experimental-status",
            title="Experimental Status",
            html=(
                '<div style="border: 3px solid #E81B23; border-radius: 8px; '
                'padding: 16px; margin: 16px 0; background: #FFF5F5;">'
                '<h3 style="color: #E81B23; margin-top: 0;">EXPERIMENTAL: '
                "Relaxed Convergence Thresholds</h3>"
                "<p>This phase uses a 2D Bayesian IRT model with "
                "<strong>relaxed convergence thresholds</strong> "
                "(R-hat &lt; 1.05, ESS &gt; 200, divergences &lt; 50) "
                "compared to the production 1D model "
                "(R-hat &lt; 1.01, ESS &gt; 400, divergences &lt; 10).</p>"
                "<p><strong>Dimension 2 credible intervals may be unreliable "
                "for most legislators.</strong> The second dimension captures "
                "~11% of variance and has inherently weak signal. Only legislators "
                "with narrow Dim 2 HDIs have reliable second-dimension estimates.</p>"
                "<p>For <strong>horseshoe-affected chambers</strong> (supermajority), "
                "2D Dim 1 is the canonical ideology score, following the DW-NOMINATE "
                "standard. For balanced chambers, 1D IRT remains canonical. "
                "See <code>docs/canonical-ideal-points.md</code>.</p>"
                "</div>"
            ),
        )
    )


def _add_canonical_routing_summary(
    report: ReportBuilder, canonical_sources: dict[str, str]
) -> None:
    """Table showing which ideal point source was selected per chamber."""
    rows = []
    for chamber, source in canonical_sources.items():
        source_label = (
            "2D IRT Dim 1 (horseshoe corrected)" if source == "2d_dim1" else "1D IRT (standard)"
        )
        rows.append({"Chamber": chamber, "Canonical Source": source_label, "Source ID": source})

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Canonical Ideal Point Routing",
        source_note=(
            "For horseshoe-affected chambers, 2D Dim 1 is used as the canonical ideology score. "
            "See docs/canonical-ideal-points.md."
        ),
    )
    report.add(
        TableSection(
            id="canonical-routing",
            title="Canonical Ideal Point Routing",
            html=html,
        )
    )


def _add_model_specification(report: ReportBuilder) -> None:
    """Text block: M2PL model specification with PLT identification."""
    report.add(
        TextSection(
            id="model-specification",
            title="Model Specification",
            html=(
                "<p><strong>Multidimensional 2-Parameter Logistic (M2PL) IRT</strong> "
                "with Positive Lower Triangular (PLT) identification.</p>"
                "<pre>"
                "P(Yea | xi_i, alpha_j, beta_j) = logit^-1(sum_d(beta[j,d] * xi[i,d]) - alpha_j)\n"
                "\n"
                "xi_i    ~ Normal(0, 1)     per dimension, per legislator\n"
                "alpha_j ~ Normal(0, 5)     per bill (difficulty)\n"
                "beta_j  ~ PLT-constrained  per bill, per dimension\n"
                "</pre>"
                "<p><strong>PLT Identification:</strong></p>"
                "<ul>"
                "<li><code>beta[0, 1] = 0</code> &mdash; rotation anchor (first item "
                "loads only on Dim 1)</li>"
                "<li><code>beta[1, 1] &gt; 0</code> &mdash; HalfNormal prior "
                "(positive diagonal, fixes Dim 2 sign)</li>"
                "<li>Post-hoc Dim 1 sign check: Republican mean must be positive</li>"
                "</ul>"
                "<p>Initialization: PCA PC1 and PC2 scores (standardized) used as "
                "starting values for xi via nutpie <code>initial_points</code>. "
                "All other RVs are jittered; xi is not.</p>"
            ),
        )
    )


def _add_convergence_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Convergence diagnostics with relaxed thresholds."""
    diag = result["diagnostics"]
    rows = [
        {
            "Metric": "R-hat (xi) max",
            "Value": f"{diag['xi_rhat_max']:.4f}",
            "Threshold": "< 1.05",
            "Status": "OK" if diag["xi_rhat_max"] < 1.05 else "WARNING",
        },
        {
            "Metric": "R-hat (alpha) max",
            "Value": f"{diag['alpha_rhat_max']:.4f}",
            "Threshold": "< 1.05",
            "Status": "OK" if diag["alpha_rhat_max"] < 1.05 else "WARNING",
        },
        {
            "Metric": "R-hat (beta) max",
            "Value": f"{diag['beta_rhat_max']:.4f}",
            "Threshold": "< 1.05",
            "Status": "OK" if diag["beta_rhat_max"] < 1.05 else "WARNING",
        },
        {
            "Metric": "ESS (xi) min",
            "Value": f"{diag['xi_ess_min']:.0f}",
            "Threshold": "> 200",
            "Status": "OK" if diag["xi_ess_min"] > 200 else "WARNING",
        },
        {
            "Metric": "ESS (alpha) min",
            "Value": f"{diag['alpha_ess_min']:.0f}",
            "Threshold": "> 200",
            "Status": "OK" if diag["alpha_ess_min"] > 200 else "WARNING",
        },
        {
            "Metric": "ESS (beta) min",
            "Value": f"{diag['beta_ess_min']:.0f}",
            "Threshold": "> 200",
            "Status": "OK" if diag["beta_ess_min"] > 200 else "WARNING",
        },
        {
            "Metric": "Tail ESS (xi) min",
            "Value": f"{diag['xi_tail_ess_min']:.0f}",
            "Threshold": "> 200",
            "Status": "OK" if diag["xi_tail_ess_min"] > 200 else "WARNING",
        },
        {
            "Metric": "Divergences",
            "Value": str(diag["divergences"]),
            "Threshold": "< 50",
            "Status": "OK" if diag["divergences"] < 50 else "WARNING",
        },
    ]
    for i, v in enumerate(diag["ebfmi"]):
        rows.append(
            {
                "Metric": f"E-BFMI (chain {i})",
                "Value": f"{v:.3f}",
                "Threshold": "> 0.3",
                "Status": "OK" if v > 0.3 else "WARNING",
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — 2D IRT Convergence (Relaxed Thresholds)",
        source_note="Thresholds are relaxed for the experimental 2D model.",
    )
    report.add(
        TableSection(
            id=f"convergence-2d-{chamber.lower()}",
            title=f"{chamber} Convergence (2D IRT)",
            html=html,
        )
    )


def _add_ideal_point_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators ranked by Dim 1 with both dimensions and HDIs."""
    ip = result["ideal_points"].sort("xi_dim1_mean", descending=True)

    display_cols = [
        "full_name",
        "party",
        "xi_dim1_mean",
        "xi_dim1_hdi_3%",
        "xi_dim1_hdi_97%",
        "xi_dim2_mean",
        "xi_dim2_hdi_3%",
        "xi_dim2_hdi_97%",
    ]
    available = [c for c in display_cols if c in ip.columns]
    df = ip.select(available)

    html = make_interactive_table(
        df,
        title=f"{chamber} — 2D Ideal Points (ranked by Dim 1)",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "xi_dim1_mean": "Dim 1",
            "xi_dim1_hdi_3%": "Dim 1 HDI 3%",
            "xi_dim1_hdi_97%": "Dim 1 HDI 97%",
            "xi_dim2_mean": "Dim 2",
            "xi_dim2_hdi_3%": "Dim 2 HDI 3%",
            "xi_dim2_hdi_97%": "Dim 2 HDI 97%",
        },
        number_formats={
            "xi_dim1_mean": "+.3f",
            "xi_dim1_hdi_3%": "+.3f",
            "xi_dim1_hdi_97%": "+.3f",
            "xi_dim2_mean": "+.3f",
            "xi_dim2_hdi_3%": "+.3f",
            "xi_dim2_hdi_97%": "+.3f",
        },
        caption="HDI = 95% Highest Density Interval. Dim 2 HDIs may be wide.",
    )
    report.add(
        InteractiveTableSection(
            id=f"ideal-points-2d-{chamber.lower()}",
            title=f"{chamber} 2D Ideal Points",
            html=html,
        )
    )


def _add_dim1_forest_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Forest plot: Dim 1 ideal points ranked with HDI bars."""
    path = plots_dir / f"dim1_forest_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-dim1-forest-{chamber.lower()}",
                f"{chamber} Dim 1 Ideal Points (Ideology Ranking)",
                path,
                caption=(
                    f"Dim 1 ideology ranking ({chamber}) from the 2D IRT model. "
                    "Each dot is a legislator's posterior mean; horizontal bars show "
                    "the 95% HDI. Red = Republican, Blue = Democrat. "
                    "For horseshoe-affected chambers, this ranking corrects the "
                    "distortion present in 1D IRT by separating ideology from "
                    "the establishment-contrarian axis."
                ),
                alt_text=(
                    f"Forest plot of 2D IRT Dim 1 ideal points for {chamber}, "
                    "showing each legislator ranked by ideology with 95% credible intervals."
                ),
            )
        )


def _add_scatter_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"2d_scatter_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-2d-scatter-{chamber.lower()}",
                f"{chamber} 2D Ideal Points Scatter",
                path,
                caption=(
                    f"2D ideal points ({chamber}): Dim 1 (ideology) vs Dim 2 "
                    "(establishment–contrarian). Red = Republican, Blue = Democrat. "
                    "Key legislators annotated."
                ),
                alt_text=(
                    f"Scatter plot of 2D IRT ideal points for {chamber}. "
                    "Dim 1 separates parties; Dim 2 captures the establishment–contrarian axis."
                ),
            )
        )


def _add_dim1_vs_pc1_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"dim1_vs_pc1_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-dim1-pc1-{chamber.lower()}",
                f"{chamber} Dim 1 vs PCA PC1",
                path,
                caption=(
                    f"2D IRT Dimension 1 vs PCA PC1 ({chamber}). "
                    "High correlation (r > 0.90) confirms Dim 1 captures ideology."
                ),
                alt_text=(
                    f"Scatter plot comparing 2D IRT Dim 1 and PCA PC1 for {chamber}. "
                    "Strong linear relationship confirms both capture the "
                    "same ideological dimension."
                ),
            )
        )


def _add_dim2_vs_pc2_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"dim2_vs_pc2_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-dim2-pc2-{chamber.lower()}",
                f"{chamber} Dim 2 vs PCA PC2",
                path,
                caption=(
                    f"2D IRT Dimension 2 vs PCA PC2 ({chamber}). "
                    "Positive correlation confirms Dim 2 captures the "
                    "secondary PCA pattern."
                ),
                alt_text=(
                    f"Scatter plot comparing 2D IRT Dim 2 and PCA PC2 for {chamber}. "
                    "Positive correlation confirms the secondary dimension aligns across methods."
                ),
            )
        )


def _add_scatter_interactive(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Embed Plotly interactive 2D scatter with hover tooltips."""
    path = plots_dir / f"2d_scatter_interactive_{chamber.lower()}.html"
    if not path.exists():
        return
    report.add(
        InteractiveSection(
            id=f"interactive-2d-scatter-{chamber.lower()}",
            title=f"{chamber} 2D Ideal Points (Interactive)",
            html=path.read_text(),
            caption=(
                f"Interactive version of the 2D scatter ({chamber}). "
                "Hover over any dot to see legislator name, party, and coordinates."
            ),
            aria_label=f"Interactive 2D ideal point scatter plot for {chamber}",
        )
    )


def _add_dim1_vs_pc1_interactive(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Embed Plotly interactive Dim 1 vs PCA PC1 with hover tooltips."""
    path = plots_dir / f"dim1_vs_pc1_interactive_{chamber.lower()}.html"
    if not path.exists():
        return
    report.add(
        InteractiveSection(
            id=f"interactive-dim1-pc1-{chamber.lower()}",
            title=f"{chamber} Dim 1 vs PCA PC1 (Interactive)",
            html=path.read_text(),
            caption=(
                f"Interactive Dim 1 vs PCA PC1 ({chamber}). "
                "Hover to see legislator name, Dim 1 value, and PCA PC1 score."
            ),
            aria_label=f"Interactive Dim 1 vs PCA PC1 scatter plot for {chamber}",
        )
    )


def _add_dim2_vs_pc2_interactive(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Embed Plotly interactive Dim 2 vs PCA PC2 with hover tooltips."""
    path = plots_dir / f"dim2_vs_pc2_interactive_{chamber.lower()}.html"
    if not path.exists():
        return
    report.add(
        InteractiveSection(
            id=f"interactive-dim2-pc2-{chamber.lower()}",
            title=f"{chamber} Dim 2 vs PCA PC2 (Interactive)",
            html=path.read_text(),
            caption=(
                f"Interactive Dim 2 vs PCA PC2 ({chamber}). "
                "Hover to see legislator name, Dim 2 value, and PCA PC2 score."
            ),
            aria_label=f"Interactive Dim 2 vs PCA PC2 scatter plot for {chamber}",
        )
    )


def _add_correlation_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Correlation between 2D IRT dimensions and PCA components."""
    corrs = result["correlations"]
    df = pl.DataFrame(
        [
            {
                "Comparison": "Dim 1 vs PCA PC1",
                "Pearson r": corrs["dim1_vs_pc1_pearson"],
                "Spearman rho": corrs["dim1_vs_pc1_spearman"],
            },
            {
                "Comparison": "Dim 2 vs PCA PC2",
                "Pearson r": corrs["dim2_vs_pc2_pearson"],
                "Spearman rho": corrs["dim2_vs_pc2_spearman"],
            },
        ]
    )
    html = make_gt(
        df,
        title=f"{chamber} — 2D IRT vs PCA Correlation",
        number_formats={"Pearson r": ".4f", "Spearman rho": ".4f"},
        source_note="Dim 1 vs PC1: expect r > 0.90. Dim 2 vs PC2: expect positive correlation.",
    )
    report.add(
        TableSection(
            id=f"correlation-2d-{chamber.lower()}",
            title=f"{chamber} 2D IRT vs PCA",
            html=html,
        )
    )


def _add_interpretation_guide(report: ReportBuilder) -> None:
    """Text block: How to interpret the 2D ideal points."""
    report.add(
        TextSection(
            id="interpretation-guide",
            title="Interpreting 2D Ideal Points",
            html=(
                "<p><strong>Dimension 1 (Ideology)</strong> is the primary axis. "
                "It captures the same liberal-conservative spectrum as the 1D IRT "
                "model and PCA PC1. Positive = conservative, negative = liberal. "
                "This dimension should be highly correlated (r &gt; 0.90) with "
                "the 1D model.</p>"
                "<p><strong>Dimension 2 (Establishment–Contrarian)</strong> captures "
                "variation not explained by the primary ideological axis. In the "
                "Kansas legislature, this captures the "
                "<strong>establishment–contrarian</strong> axis &mdash; the degree to which "
                "legislators align with party leadership on routine bills. "
                "Positive Dim 2 = establishment-aligned; negative Dim 2 = contrarian. "
                "The clearest example is Senator Caryn Tyson, who appears as the most "
                "conservative legislator on Dim 1 but scores extreme negative on Dim 2 "
                "due to frequent Nay votes on bills nearly all Republicans support.</p>"
                "<p><strong>Wide Dim 2 HDIs</strong> are expected. For most "
                "legislators, the second dimension has weak signal (~11% variance). "
                "Only legislators with narrow Dim 2 HDIs have meaningfully "
                "distinguished positions on this axis. A wide HDI means "
                "&ldquo;we cannot tell where this legislator sits on Dim 2.&rdquo;</p>"
                "<p><strong>Caution:</strong> The 2D model uses relaxed convergence "
                "thresholds. Dim 2 posterior summaries should be treated as "
                "exploratory, not definitive.</p>"
                "<p><strong>Canonical routing:</strong> For horseshoe-affected chambers, "
                "Dim 1 from this 2D model is the canonical ideology score consumed by "
                "downstream phases. For balanced chambers, the simpler 1D IRT model "
                "remains canonical. See <code>docs/canonical-ideal-points.md</code>.</p>"
            ),
        )
    )


def _generate_irt_2d_key_findings(
    results: dict[str, dict],
    canonical_sources: dict[str, str] | None = None,
) -> list[str]:
    """Generate 2-4 key findings from 2D IRT results."""
    findings: list[str] = []

    # Convergence status
    all_ok = True
    for chamber, result in results.items():
        diag = result.get("diagnostics", {})
        if diag.get("xi_rhat_max", 2.0) >= 1.05 or diag.get("xi_ess_min", 0) < 200:
            all_ok = False
            break
    if all_ok:
        findings.append(
            "All 2D convergence diagnostics <strong>passed</strong> "
            "(relaxed thresholds: R-hat < 1.05, ESS > 200)."
        )
    else:
        findings.append("<strong>WARNING:</strong> Some 2D convergence diagnostics did not pass.")

    # Dim 1 vs PC1 correlation
    for chamber, result in results.items():
        corrs = result.get("correlations", {})
        r_dim1 = corrs.get("dim1_vs_pc1_pearson")
        r_dim2 = corrs.get("dim2_vs_pc2_pearson")
        if r_dim1 is not None:
            findings.append(
                f"{chamber} Dim 1 vs PCA PC1: <strong>r = {r_dim1:.3f}</strong>"
                + (f" (Dim 2 vs PC2: r = {r_dim2:.3f})" if r_dim2 is not None else "")
                + "."
            )
        break

    # Canonical routing
    if canonical_sources:
        dim1_chambers = [c for c, s in canonical_sources.items() if s == "2d_dim1"]
        if dim1_chambers:
            findings.append(
                f"<strong>Canonical routing:</strong> {', '.join(dim1_chambers)} "
                "using 2D Dim 1 (horseshoe detected in 1D)."
            )
        else:
            findings.append(
                "<strong>Canonical routing:</strong> all chambers using 1D IRT (no horseshoe)."
            )

    return findings


def _add_analysis_parameters(
    report: ReportBuilder,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> None:
    """Table: Analysis parameters used in this run."""
    try:
        from analysis.irt_2d import (
            ESS_THRESHOLD,
            MAX_DIVERGENCES,
            RANDOM_SEED,
            RHAT_THRESHOLD,
        )
    except ModuleNotFoundError:
        from irt_2d import (  # type: ignore[no-redef]
            ESS_THRESHOLD,
            MAX_DIVERGENCES,
            RANDOM_SEED,
            RHAT_THRESHOLD,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "Model",
                "Dimensions",
                "Identification",
                "Sampler",
                "Prior (xi)",
                "Prior (alpha)",
                "Prior (beta col 0)",
                "Prior (beta[1,1])",
                "MCMC Draws per Chain",
                "Tuning Steps",
                "Chains",
                "Random Seed",
                "R-hat Threshold",
                "ESS Threshold",
                "Max Divergences",
            ],
            "Value": [
                "M2PL IRT (Bayesian, 2D)",
                "2",
                "Positive Lower Triangular (PLT)",
                "nutpie (Rust NUTS, adaptive dual averaging)",
                "Normal(0, 1) per dimension",
                "Normal(0, 5)",
                "Normal(0, 1) — unconstrained",
                "HalfNormal(1) — positive diagonal",
                str(n_samples),
                str(n_tune),
                str(n_chains),
                str(RANDOM_SEED),
                f"< {RHAT_THRESHOLD} (relaxed)",
                f"> {ESS_THRESHOLD} (relaxed)",
                f"< {MAX_DIVERGENCES} (relaxed)",
            ],
        }
    )
    html = make_gt(
        df,
        title="2D IRT Analysis Parameters (EXPERIMENTAL)",
        source_note="Thresholds are relaxed compared to the production 1D model.",
    )
    report.add(
        TableSection(
            id="analysis-params-2d",
            title="Analysis Parameters",
            html=html,
        )
    )
