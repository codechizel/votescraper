"""Hierarchical IRT-specific HTML report builder.

Builds ~18-22 sections (tables, figures, and text) for the hierarchical Bayesian
IRT report. Covers per-chamber group posteriors, variance decomposition, shrinkage
comparison, forest plots, dispersion, and optionally the joint cross-chamber model.

Usage (called from hierarchical.py):
    from analysis.hierarchical_report import build_hierarchical_report
    build_hierarchical_report(ctx.report, chamber_results=..., ...)
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


def build_hierarchical_report(
    report: ReportBuilder,
    *,
    chamber_results: dict[str, dict],
    joint_results: dict | None,
    linking_results: dict | None = None,
    plots_dir: Path,
) -> None:
    """Build the full hierarchical IRT HTML report by adding sections."""
    findings = _generate_hierarchical_key_findings(chamber_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_intro(report)
    _add_how_to_read(report)

    for chamber in sorted(chamber_results.keys()):
        res = chamber_results[chamber]

        _add_group_params_table(report, res["group_params"], chamber)
        _add_party_posteriors_figure(report, plots_dir, chamber)
        _add_icc_figure(report, plots_dir, chamber)
        _add_variance_decomposition_text(report, res["icc_df"], chamber)
        _add_dispersion_figure(report, plots_dir, chamber)
        _add_shrinkage_scatter_figure(report, plots_dir, chamber)
        _add_shrinkage_table(report, res["ideal_points"], chamber)
        _add_forest_figure(report, plots_dir, chamber)
        _add_convergence_text(report, res["convergence"], res["sampling_time"], chamber)

    # Cross-chamber comparison
    if len(chamber_results) > 1:
        _add_cross_chamber_comparison(report, chamber_results)

    # Joint model
    if joint_results is not None:
        _add_joint_model_section(report, joint_results, plots_dir)

    # Stocking-Lord linking (cross-chamber alignment without joint model)
    if linking_results is not None:
        _add_linking_section(report, linking_results)

    # Flat vs hierarchical comparison
    _add_flat_vs_hier_comparison(report, chamber_results)

    _add_analysis_parameters(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_intro(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="intro",
            title="What Is a Hierarchical Model?",
            html=(
                "<p>The standard IRT model treats every legislator as independent — it doesn't "
                "know that legislators belong to parties. The hierarchical model fixes this by "
                "adding a <strong>party layer</strong>: each party has its own average ideology, "
                "and individual legislators are drawn from their party's distribution.</p>"
                "<p>This matters because:</p>"
                "<ul>"
                "<li><strong>Partial pooling:</strong> Legislators with few votes get pulled "
                "toward their party's average, producing more reliable estimates.</li>"
                "<li><strong>Variance decomposition:</strong> We can now say exactly how much "
                "of the ideological variation is explained by party vs. individual "
                "differences.</li>"
                "<li><strong>Shrinkage comparison:</strong> By comparing with the flat model, "
                "we can see which legislators' estimates changed the most — and whether they "
                "moved toward their party.</li>"
                "</ul>"
            ),
        )
    )


def _add_how_to_read(report: ReportBuilder) -> None:
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html=(
                "<p>This report has five key visualizations per chamber:</p>"
                "<ol>"
                "<li><strong>Party Posteriors</strong> — Where does each party stand on the "
                "ideological spectrum? Separate curves = polarized.</li>"
                "<li><strong>ICC (Variance Decomposition)</strong> — What percentage of "
                "ideological variation is explained by party? Higher = more party-driven.</li>"
                "<li><strong>Dispersion</strong> — Which party has more internal disagreement? "
                "The party with a wider, more rightward curve is less cohesive.</li>"
                "<li><strong>Shrinkage Scatter</strong> — How did accounting for party change "
                "each legislator's estimate? Points off the diagonal moved.</li>"
                "<li><strong>Forest Plot</strong> — Every legislator's ideal point with "
                "uncertainty bars, using the hierarchical estimates.</li>"
                "</ol>"
            ),
        )
    )


def _add_group_params_table(
    report: ReportBuilder,
    group_params: pl.DataFrame,
    chamber: str,
) -> None:
    if group_params.height == 0:
        return

    display = group_params.select(
        "party",
        "n_legislators",
        "mu_mean",
        "mu_sd",
        "mu_hdi_2.5",
        "mu_hdi_97.5",
        "sigma_within_mean",
        "sigma_within_sd",
    )

    html = make_gt(
        display,
        title=f"{chamber} — Party-Level Parameters",
        subtitle="Posterior summaries of party mean ideal points and within-party dispersion",
        column_labels={
            "party": "Party",
            "n_legislators": "N",
            "mu_mean": "Party Mean",
            "mu_sd": "Mean SD",
            "mu_hdi_2.5": "Mean HDI 2.5%",
            "mu_hdi_97.5": "Mean HDI 97.5%",
            "sigma_within_mean": "Within-Party SD",
            "sigma_within_sd": "SD of SD",
        },
        number_formats={
            "mu_mean": "+.3f",
            "mu_sd": ".3f",
            "mu_hdi_2.5": "+.3f",
            "mu_hdi_97.5": "+.3f",
            "sigma_within_mean": ".3f",
            "sigma_within_sd": ".3f",
        },
        source_note=(
            "Party Mean = posterior mean of the party-level ideal point (positive = conservative). "
            "Within-Party SD = how spread out the party's members are (higher = more internal "
            "disagreement)."
        ),
    )
    report.add(
        TableSection(
            id=f"group-params-{chamber.lower()}",
            title=f"{chamber} Party-Level Parameters",
            html=html,
        )
    )


def _add_party_posteriors_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"party_posteriors_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-party-post-{chamber.lower()}",
                f"{chamber} — Where Do the Parties Stand?",
                path,
                caption=(
                    "Posterior distributions of party mean ideal points. Separation between "
                    "the curves indicates polarization. The dashed lines mark the posterior means."
                ),
                alt_text=(
                    f"KDE plot of {chamber} posterior distributions for Republican and Democrat "
                    "party mean ideal points. Separation between curves indicates the degree of "
                    "partisan polarization."
                ),
            )
        )


def _add_icc_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"icc_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-icc-{chamber.lower()}",
                f"{chamber} — How Much Does Party Explain?",
                path,
                caption=(
                    "The Intraclass Correlation Coefficient (ICC) shows what share of "
                    "ideological variance is explained by party membership vs. individual "
                    "differences. The error bar shows the 95% credible interval."
                ),
                alt_text=(
                    f"Bar chart of the {chamber} Intraclass Correlation Coefficient (ICC) "
                    "showing the share of ideological variance explained by party membership, "
                    "with a 95% credible interval error bar."
                ),
            )
        )


def _add_variance_decomposition_text(
    report: ReportBuilder,
    icc_df: pl.DataFrame,
    chamber: str,
) -> None:
    if icc_df.height == 0:
        return

    icc_mean = float(icc_df["icc_mean"][0])
    icc_lo = float(icc_df["icc_ci_2.5"][0])
    icc_hi = float(icc_df["icc_ci_97.5"][0])

    if icc_mean > 0.7:
        interpretation = (
            "Party is the dominant factor. Most of the variation in how legislators "
            "vote is explained by which party they belong to."
        )
    elif icc_mean > 0.4:
        interpretation = (
            "Party matters a lot, but individual differences also play a meaningful role. "
            "Some legislators deviate substantially from their party line."
        )
    else:
        interpretation = (
            "Individual differences outweigh party. Legislators vote based on personal "
            "ideology as much as — or more than — party affiliation."
        )

    report.add(
        TextSection(
            id=f"icc-text-{chamber.lower()}",
            title=f"{chamber} — Variance Decomposition",
            html=(
                f"<p>In the {chamber}, <strong>{icc_mean:.0%}</strong> of the variation in "
                f"legislators' ideological positions is explained by party membership "
                f"(95% CI: [{icc_lo:.0%}, {icc_hi:.0%}]).</p>"
                f"<p>{interpretation}</p>"
            ),
        )
    )


def _add_dispersion_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"dispersion_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-dispersion-{chamber.lower()}",
                f"{chamber} — Which Party Has More Internal Disagreement?",
                path,
                caption=(
                    "Posterior distributions of within-party standard deviation. A wider curve "
                    "shifted to the right means that party's members have more diverse voting "
                    "patterns."
                ),
                alt_text=(
                    f"KDE plot of {chamber} posterior distributions for within-party standard "
                    "deviation by party. A rightward-shifted curve indicates greater internal "
                    "ideological disagreement within that party."
                ),
            )
        )


def _add_shrinkage_scatter_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"shrinkage_scatter_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-shrinkage-{chamber.lower()}",
                f"{chamber} — How Does Accounting for Party Change Estimates?",
                path,
                caption=(
                    "Each dot is a legislator. The diagonal line marks 'no change.' "
                    "Points pulled off the diagonal moved when the model accounted for party "
                    "structure. The 5 biggest movers are labeled."
                ),
                alt_text=(
                    f"Scatter plot comparing {chamber} flat IRT ideal points (x-axis) to "
                    "hierarchical ideal points (y-axis). Points off the diagonal indicate "
                    "shrinkage toward the party mean; the 5 biggest movers are labeled."
                ),
            )
        )


def _add_shrinkage_table(
    report: ReportBuilder,
    ideal_points: pl.DataFrame,
    chamber: str,
) -> None:
    """Full table of hierarchical ideal points — never truncated."""
    if ideal_points.height == 0:
        return

    cols = [
        "full_name",
        "party",
        "district",
        "xi_mean",
        "xi_sd",
        "xi_hdi_2.5",
        "xi_hdi_97.5",
        "party_mean",
    ]
    if "delta_from_flat" in ideal_points.columns:
        cols.extend(["delta_from_flat", "shrinkage_pct", "toward_party_mean"])

    display = ideal_points.sort("xi_mean", descending=True).select(
        [c for c in cols if c in ideal_points.columns]
    )

    labels = {
        "full_name": "Legislator",
        "party": "Party",
        "district": "District",
        "xi_mean": "Ideal Point",
        "xi_sd": "Uncertainty",
        "xi_hdi_2.5": "HDI Low",
        "xi_hdi_97.5": "HDI High",
        "party_mean": "Party Mean",
        "delta_from_flat": "Change from Flat",
        "shrinkage_pct": "Shrinkage %",
        "toward_party_mean": "Toward Party?",
    }

    formats = {
        "xi_mean": "+.3f",
        "xi_sd": ".3f",
        "xi_hdi_2.5": "+.3f",
        "xi_hdi_97.5": "+.3f",
        "party_mean": "+.3f",
        "delta_from_flat": "+.3f",
        "shrinkage_pct": ".1f",
    }

    html = make_interactive_table(
        display,
        title=f"{chamber} — Hierarchical Ideal Points ({display.height} legislators)",
        column_labels={k: v for k, v in labels.items() if k in display.columns},
        number_formats={k: v for k, v in formats.items() if k in display.columns},
        caption=(
            "Ideal Point: positive = conservative, negative = liberal. "
            "Party Mean: the posterior mean for this legislator's party. "
            "Change from Flat: how much the estimate moved compared to the standard IRT model. "
            "Toward Party? = True if the estimate moved closer to the party mean."
        ),
    )
    report.add(
        InteractiveTableSection(
            id=f"ideal-points-{chamber.lower()}",
            title=f"{chamber} Hierarchical Ideal Points",
            html=html,
        )
    )


def _add_forest_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"forest_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-forest-{chamber.lower()}",
                f"{chamber} — Hierarchical Ideal Point Estimates",
                path,
                caption=(
                    "Every legislator's ideal point estimate with 95% HDI. "
                    "Points are colored by party. The hierarchical model pulls "
                    "uncertain legislators toward their party mean."
                ),
                alt_text=(
                    f"Forest plot of all {chamber} legislators' hierarchical ideal point "
                    "estimates with 95% HDI error bars, colored by party. Legislators are "
                    "sorted by ideal point from liberal to conservative."
                ),
            )
        )


def _add_convergence_text(
    report: ReportBuilder,
    convergence: dict,
    sampling_time: float,
    chamber: str,
) -> None:
    all_ok = convergence.get("all_ok", False)
    divergences = convergence.get("divergences", -1)
    status = "All convergence checks passed." if all_ok else "Some convergence checks failed."

    parts = [
        f"<p><strong>{status}</strong></p>",
        f"<p>Sampling time: {sampling_time:.0f} seconds. Divergences: {divergences}.</p>",
    ]

    # Key diagnostics
    for var in ["xi", "mu_party", "sigma_within"]:
        rhat_key = f"{var}_rhat_max"
        ess_key = f"{var}_ess_min"
        if rhat_key in convergence:
            parts.append(
                f"<p>{var}: R-hat max = {convergence[rhat_key]:.4f}, "
                f"ESS min = {convergence.get(ess_key, 0):.0f}</p>"
            )

    report.add(
        TextSection(
            id=f"convergence-{chamber.lower()}",
            title=f"{chamber} — Convergence Diagnostics",
            html="".join(parts),
        )
    )


def _add_cross_chamber_comparison(
    report: ReportBuilder,
    chamber_results: dict[str, dict],
) -> None:
    """Compare ICC and party parameters across chambers."""
    rows = []
    for chamber in sorted(chamber_results.keys()):
        res = chamber_results[chamber]
        icc_mean = float(res["icc_df"]["icc_mean"][0])
        for row in res["group_params"].iter_rows(named=True):
            rows.append(
                {
                    "Chamber": chamber,
                    "Party": row["party"],
                    "N Legislators": row["n_legislators"],
                    "Party Mean": row["mu_mean"],
                    "Within-Party SD": row["sigma_within_mean"],
                    "ICC": icc_mean,
                    "Flat Correlation": res["flat_corr"],
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Cross-Chamber Comparison",
        subtitle="How do the House and Senate hierarchical structures compare?",
        number_formats={
            "Party Mean": "+.3f",
            "Within-Party SD": ".3f",
            "ICC": ".3f",
            "Flat Correlation": ".4f",
        },
        source_note=(
            "ICC = fraction of ideological variance explained by party. "
            "Flat Correlation = Pearson r between hierarchical and flat IRT ideal points."
        ),
    )
    report.add(
        TableSection(
            id="cross-chamber",
            title="Cross-Chamber Comparison",
            html=html,
        )
    )


def _add_joint_model_section(
    report: ReportBuilder,
    joint_results: dict,
    plots_dir: Path,
) -> None:
    """Sections for the joint cross-chamber model."""
    convergence = joint_results["convergence"]
    all_ok = convergence.get("all_ok", False)

    report.add(
        TextSection(
            id="joint-intro",
            title="Joint Cross-Chamber Model",
            html=(
                "<p>The joint model places all legislators — House and Senate — on a common "
                "ideological scale. It adds a <strong>chamber layer</strong> above the party "
                "layer, allowing the model to learn whether the House or Senate is more "
                "polarized.</p>"
                f"<p>Convergence: {'All checks passed.' if all_ok else 'Some checks failed.'} "
                f"Sampling time: {joint_results['sampling_time']:.0f}s. "
                f"Divergences: {convergence.get('divergences', -1)}.</p>"
            ),
        )
    )

    # Joint party spread plot
    path = plots_dir / "joint_party_spread.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-joint-spread",
                "Are Parties More Polarized in the House or Senate?",
                path,
                caption=(
                    "Group mean posteriors for all four party-chamber combinations. "
                    "Solid lines = House, dashed = Senate. Greater separation between "
                    "party curves within a chamber indicates more polarization."
                ),
                alt_text=(
                    "KDE plot of group mean posterior distributions for all four party-chamber "
                    "combinations. Solid lines represent House, dashed lines represent Senate. "
                    "Greater separation between party curves indicates more polarization."
                ),
            )
        )

    # Joint forest plot
    path = plots_dir / "forest_joint.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-forest-joint",
                "Joint Model — All Legislators on a Common Scale",
                path,
                caption=(
                    "All House and Senate legislators placed on the same ideological scale "
                    "by the joint hierarchical model."
                ),
                alt_text=(
                    "Forest plot of all House and Senate legislators on a common ideological "
                    "scale from the joint hierarchical model, with 95% HDI error bars colored "
                    "by party."
                ),
            )
        )


def _add_linking_section(report: ReportBuilder, linking_results: dict) -> None:
    """Section for Stocking-Lord IRT scale linking results."""
    all_methods = linking_results.get("all_methods", {})
    linked_df = linking_results.get("linked_df")
    matched_bills = linking_results.get("matched_bills", [])

    # Linking coefficients table (all 4 methods)
    if all_methods:
        rows = []
        for method_name, coeffs in sorted(all_methods.items()):
            rows.append(
                {
                    "method": method_name,
                    "A (slope)": round(float(coeffs.get("A", 0)), 4),
                    "B (intercept)": round(float(coeffs.get("B", 0)), 4),
                }
            )
        if rows:
            coeff_df = pl.DataFrame(rows)
            html = make_gt(
                coeff_df,
                title="IRT Scale Linking Coefficients",
                subtitle=(
                    f"Cross-chamber alignment using {len(matched_bills)} shared anchor bills. "
                    "Senate scale is transformed to House scale: θ* = A·θ + B."
                ),
                column_labels={
                    "method": "Method",
                    "A (slope)": "A (Slope)",
                    "B (intercept)": "B (Intercept)",
                },
            )
            report.add(
                TableSection(
                    id="linking-coefficients",
                    title="Cross-Chamber Scale Linking (Stocking-Lord)",
                    html=html,
                )
            )

    # Linked ideal points interactive table
    if linked_df is not None and linked_df.height > 0:
        display = linked_df.select("legislator_slug", "chamber", "xi_linked", "xi_sd")
        html = make_interactive_table(
            display,
            title="Linked Ideal Points (Common Scale)",
            column_labels={
                "legislator_slug": "Legislator",
                "chamber": "Chamber",
                "xi_linked": "Ideal Point (Linked)",
                "xi_sd": "Posterior SD",
            },
            number_formats={
                "xi_linked": "+.3f",
                "xi_sd": ".3f",
            },
        )
        report.add(
            InteractiveTableSection(
                id="linking-ideal-points",
                title="All Legislators on a Common Scale (Stocking-Lord Linking)",
                html=html,
            )
        )


def _add_flat_vs_hier_comparison(
    report: ReportBuilder,
    chamber_results: dict[str, dict],
) -> None:
    """Text summary comparing flat vs hierarchical IRT."""
    parts = ["<p>Key differences between the flat and hierarchical models:</p><ul>"]

    for chamber in sorted(chamber_results.keys()):
        res = chamber_results[chamber]
        flat_corr = res["flat_corr"]
        icc_mean = float(res["icc_df"]["icc_mean"][0])

        ip = res["ideal_points"]
        if "toward_party_mean" in ip.columns and "delta_from_flat" in ip.columns:
            non_null = ip.drop_nulls(subset=["toward_party_mean"])
            toward_pct = (
                non_null.filter(pl.col("toward_party_mean")).height / non_null.height
                if non_null.height > 0
                else 0.0
            )
            avg_delta = (
                float(ip.select(pl.col("delta_from_flat").abs().mean())[0, 0])
                if ip.drop_nulls(subset=["delta_from_flat"]).height > 0
                else 0.0
            )

            parts.append(
                f"<li><strong>{chamber}:</strong> Correlation with flat IRT = {flat_corr:.4f}. "
                f"ICC = {icc_mean:.0%}. "
                f"{toward_pct:.0%} of legislators moved toward their party mean. "
                f"Average absolute change = {avg_delta:.3f}.</li>"
            )
        else:
            parts.append(
                f"<li><strong>{chamber}:</strong> Correlation with flat IRT = {flat_corr:.4f}. "
                f"ICC = {icc_mean:.0%}.</li>"
            )

    parts.append("</ul>")

    report.add(
        TextSection(
            id="flat-vs-hier",
            title="Flat vs. Hierarchical IRT Comparison",
            html="".join(parts),
        )
    )


def _generate_hierarchical_key_findings(
    chamber_results: dict[str, dict],
) -> list[str]:
    """Generate 2-4 key findings from hierarchical IRT results."""
    findings: list[str] = []

    # Convergence
    all_converged = True
    for chamber, res in chamber_results.items():
        conv = res.get("convergence", {})
        if conv.get("max_rhat", 2.0) >= 1.01 or conv.get("min_ess", 0) < 400:
            all_converged = False
            break
    if all_converged:
        findings.append(
            "All hierarchical models <strong>converged</strong> (R-hat < 1.01, ESS > 400)."
        )
    else:
        findings.append(
            "<strong>WARNING:</strong> Some hierarchical convergence diagnostics did not pass."
        )

    # Party posteriors
    for chamber, res in chamber_results.items():
        gp = res.get("group_params")
        if gp is not None and gp.height >= 2:
            r_row = gp.filter(pl.col("party") == "Republican")
            d_row = gp.filter(pl.col("party") == "Democrat")
            if r_row.height > 0 and d_row.height > 0:
                r_mean = float(r_row["mu_mean"][0])
                d_mean = float(d_row["mu_mean"][0])
                sep = abs(r_mean - d_mean)
                findings.append(
                    f"{chamber} party means: R = <strong>{r_mean:+.2f}</strong>, "
                    f"D = <strong>{d_mean:+.2f}</strong> (separation = {sep:.2f})."
                )
        break  # First chamber only

    # Shrinkage summary
    for chamber, res in chamber_results.items():
        ip = res.get("ideal_points")
        if ip is not None and "shrinkage" in ip.columns:
            mean_shrink = float(ip["shrinkage"].mean())
            findings.append(
                f"{chamber} mean shrinkage: <strong>{mean_shrink:.1%}</strong> toward party mean."
            )
        break

    # Small-group warning (e.g. Senate Democrats N=7-11)
    small_group_threshold = 20
    for chamber, res in chamber_results.items():
        gp = res.get("group_params")
        if gp is None:
            continue
        for row in gp.iter_rows(named=True):
            n = row.get("n_legislators", 0)
            party = row.get("party", "Unknown")
            if 0 < n < small_group_threshold:
                findings.append(
                    f"<strong>Note:</strong> {chamber} {party}s ({n} legislators) fall below "
                    f"the recommended group size for reliable hierarchical shrinkage "
                    f"(Gelman &amp; Hill 2007). Flat IRT ideal points may be more "
                    f"trustworthy for individual {chamber} {party} positions."
                )

    return findings


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: All constants and settings used in this run."""
    df = pl.DataFrame(
        {
            "Parameter": [
                "MCMC Samples",
                "Tuning Steps",
                "Chains",
                "Target Acceptance",
                "Identification",
                "Parameterization",
                "Party Prior Mean",
                "Party Prior SD",
                "Within-Party SD Prior",
            ],
            "Value": [
                "2000 per chain",
                "1500 (discarded)",
                "2",
                "0.95",
                "Ordering constraint (sort)",
                "Non-centered",
                "Normal(0, 2)",
                "—",
                "HalfNormal(1)",
            ],
            "Description": [
                "Posterior draws used for inference",
                "Higher than flat IRT (1000) to handle hierarchical geometry",
                "Independent MCMC chains for convergence checks",
                "Higher than flat IRT (0.9) to reduce divergences",
                "mu_party[D] < mu_party[R] via pt.sort — no anchored legislators",
                "xi_offset ~ Normal(0,1), xi = mu + sigma * offset",
                "Weakly informative prior on party-level ideal points",
                "—",
                "Regularizes toward 0; prevents unrealistically wide party distributions",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/hierarchical.md for justification.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
