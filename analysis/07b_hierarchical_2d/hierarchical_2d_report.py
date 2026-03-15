"""Hierarchical 2D IRT report builder.

Builds sections for the Phase 07b HTML report: 2D scatter, party posteriors,
shrinkage comparison, group parameters, and convergence diagnostics.

Usage (called from hierarchical_2d.py):
    from analysis.hierarchical_2d_report import build_hierarchical_2d_report
    build_hierarchical_2d_report(ctx.report, chamber_results=..., ...)
"""

from pathlib import Path

try:
    from analysis.report import (
        FigureSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TextSection,
        make_interactive_table,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveTableSection,
        KeyFindingsSection,
        ReportBuilder,
        TextSection,
        make_interactive_table,
    )


def build_hierarchical_2d_report(
    report: ReportBuilder,
    *,
    chamber_results: dict[str, dict],
    plots_dir: Path,
    session: str,
) -> None:
    """Build the Hierarchical 2D IRT HTML report."""

    # ── Key Findings ──
    findings = []
    for chamber, results in chamber_results.items():
        convergence = results["convergence"]
        ideal_points = results["ideal_points"]
        n_leg = ideal_points.height
        status = "converged" if convergence.get("all_ok", False) else "convergence issues"
        findings.append(f"{chamber}: {n_leg} legislators, {status}")

    report.add_section(
        KeyFindingsSection(
            title="Key Findings",
            findings=findings,
        )
    )

    # ── Per-chamber sections ──
    for chamber, results in chamber_results.items():
        ch = chamber.lower()
        ideal_points = results["ideal_points"]
        group_params = results["group_params"]
        convergence = results["convergence"]
        sampling_time = results["sampling_time"]

        # 2D Scatter
        scatter_path = plots_dir / f"2d_scatter_{ch}.png"
        if scatter_path.exists():
            report.add_section(
                FigureSection(
                    title=f"{chamber} — 2D Ideal Points",
                    image_path=scatter_path,
                    caption=(
                        f"Hierarchical 2D IRT ideal points for the Kansas {chamber}. "
                        "Dim 1 (x-axis) = ideology, Dim 2 (y-axis) = secondary axis. "
                        "Party pooling regularizes sparse legislators."
                    ),
                    alt_text=(
                        f"Scatter plot of 2D ideal points for {chamber} "
                        "legislators colored by party"
                    ),
                )
            )

        # Party Posteriors
        posteriors_path = plots_dir / f"party_posteriors_{ch}.png"
        if posteriors_path.exists():
            report.add_section(
                FigureSection(
                    title=f"{chamber} — Party Mean Posteriors",
                    image_path=posteriors_path,
                    caption=(
                        "Posterior distributions of party mean ideal points for each dimension. "
                        "Separation indicates polarization on that axis."
                    ),
                    alt_text=(
                        f"Histograms of party mean posteriors for "
                        f"{chamber} on both dimensions"
                    ),
                )
            )

        # Shrinkage vs Flat 2D
        shrinkage_path = plots_dir / f"shrinkage_vs_flat2d_{ch}.png"
        if shrinkage_path.exists():
            report.add_section(
                FigureSection(
                    title=f"{chamber} — Shrinkage vs Flat 2D",
                    image_path=shrinkage_path,
                    caption=(
                        "Comparison of hierarchical 2D vs flat 2D ideal points. Points off "
                        "the diagonal indicate shrinkage toward party means."
                    ),
                    alt_text=(
                        f"Scatter plot comparing hierarchical vs flat 2D ideal points for {chamber}"
                    ),
                )
            )

        # Group Parameters Table
        report.add_section(
            InteractiveTableSection(
                title=f"{chamber} — Group Parameters",
                table_html=make_interactive_table(
                    group_params,
                    table_id=f"group_params_{ch}",
                ),
                description=(
                    "Per-party per-dimension group parameters: mu (party mean) and "
                    "sigma_within (within-party spread)."
                ),
            )
        )

        # Ideal Points Table
        display_cols = [
            "legislator_slug",
            "full_name",
            "party",
            "xi_dim1_mean",
            "xi_dim1_hdi_3%",
            "xi_dim1_hdi_97%",
            "xi_dim2_mean",
            "xi_dim2_hdi_3%",
            "xi_dim2_hdi_97%",
        ]
        available = [c for c in display_cols if c in ideal_points.columns]
        report.add_section(
            InteractiveTableSection(
                title=f"{chamber} — Ideal Points",
                table_html=make_interactive_table(
                    ideal_points.select(available).sort("xi_dim1_mean"),
                    table_id=f"ideal_points_h2d_{ch}",
                ),
                description="All legislators ranked by Dim 1 (ideology).",
            )
        )

        # Convergence Summary
        conv_text = f"**Sampling time:** {sampling_time:.1f}s\n\n"
        conv_text += f"**Overall:** {'PASSED' if convergence.get('all_ok') else 'ISSUES'}\n\n"
        if "xi_rhat_max" in convergence:
            conv_text += f"- R-hat (xi): {convergence['xi_rhat_max']:.4f}\n"
        if "xi_ess_min" in convergence:
            conv_text += f"- ESS (xi): {convergence['xi_ess_min']:.0f}\n"
        if "divergences" in convergence:
            conv_text += f"- Divergences: {convergence['divergences']}\n"

        report.add_section(
            TextSection(
                title=f"{chamber} — Convergence Diagnostics",
                text=conv_text,
            )
        )
