"""Dynamic IRT HTML report builder.

Builds ~14 sections (tables, figures, and text) for the dynamic ideal point
estimation report.  Each section is a small function that slices/aggregates
polars DataFrames and calls make_gt() or FigureSection.from_file().

Usage (called from dynamic_irt.py):
    from analysis.dynamic_irt_report import build_dynamic_irt_report
    build_dynamic_irt_report(ctx.report, results=..., plots_dir=...)
"""

from pathlib import Path

try:
    from analysis.report import (
        FigureSection,
        InteractiveSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        InteractiveSection,
        KeyFindingsSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )

try:
    from analysis.dynamic_irt_data import MIN_BRIDGE_OVERLAP
except ModuleNotFoundError:
    from dynamic_irt_data import MIN_BRIDGE_OVERLAP  # type: ignore[no-redef]


def build_dynamic_irt_report(
    report: ReportBuilder,
    *,
    results: dict,
    plots_dir: Path,
    biennium_labels: list[str],
) -> None:
    """Build the full dynamic IRT HTML report by adding sections."""
    chambers = results.get("chambers", [])

    findings = _generate_dynamic_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_overview(report, results, biennium_labels)

    for chamber in sorted(chambers):
        if chamber not in results:
            continue
        cr = results[chamber]

        _add_bridge_coverage(report, cr, plots_dir, chamber)
        _add_sign_corrections(report, cr, chamber)
        _add_global_roster(report, cr, chamber)
        _add_polarization_trend(report, plots_dir, chamber)
        _add_ridgeline(report, plots_dir, chamber)
        _add_animated_scatter(report, plots_dir, chamber)
        _add_trajectories(report, plots_dir, chamber)
        _add_tau_posterior(report, cr, plots_dir, chamber)
        _add_top_movers_table(report, cr, chamber)
        _add_top_movers_bar(report, plots_dir, chamber)
        _add_conversion_replacement_table(report, cr, chamber)
        _add_conversion_replacement_plot(report, plots_dir, chamber)
        _add_static_correlation(report, cr, plots_dir, chamber)

        if cr.get("emirt_results") is not None:
            _add_emirt_comparison(report, cr, chamber)

        _add_convergence(report, cr, chamber)

    _add_model_priors(report, results)
    _add_methodology(report)


# ── Individual Sections ──────────────────────────────────────────────────────


def _add_overview(
    report: ReportBuilder,
    results: dict,
    biennium_labels: list[str],
) -> None:
    """Overview section with key statistics."""
    chambers = results.get("chambers", [])
    lines = [
        "<h3>Dynamic Ideal Point Estimation</h3>",
        f"<p><strong>Bienniums:</strong> {', '.join(biennium_labels)} "
        f"({len(biennium_labels)} sessions)</p>",
        "<table style='border-collapse: collapse; margin: 1em 0;'>",
        "<tr><th style='padding: 4px 12px; text-align: left;'>Chamber</th>"
        "<th style='padding: 4px 12px;'>Legislators</th>"
        "<th style='padding: 4px 12px;'>Bills</th>"
        "<th style='padding: 4px 12px;'>Observations</th>"
        "<th style='padding: 4px 12px;'>Sampling Time</th></tr>",
    ]

    for chamber in sorted(chambers):
        if chamber not in results:
            continue
        cr = results[chamber]
        s = cr["stacked"]
        t = cr["sampling_time"]
        lines.append(
            f"<tr><td style='padding: 4px 12px;'>{chamber.capitalize()}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{s['n_legislators']}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{s['n_bills']:,}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{s['n_obs']:,}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{t:.0f}s</td></tr>"
        )

    lines.append("</table>")

    # Tau estimates
    for chamber in sorted(chambers):
        if chamber not in results:
            continue
        cr = results[chamber]
        tau = cr.get("tau")
        if tau is not None and tau.height > 0:
            cap = chamber.capitalize()
            lines.append(f"<p><strong>{cap} Evolution Variance (tau):</strong></p>")
            lines.append("<ul>")
            for row in tau.iter_rows(named=True):
                lines.append(
                    f"<li>{row['party']}: tau = {row['tau_mean']:.4f} "
                    f"[{row['tau_hdi_2.5']:.4f}, {row['tau_hdi_97.5']:.4f}]</li>"
                )
            lines.append("</ul>")

    report.add(TextSection(id="overview", title="Overview", html="\n".join(lines)))


def _add_bridge_coverage(
    report: ReportBuilder,
    cr: dict,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Bridge coverage heatmap and adjacent bridge table."""
    chamber_cap = chamber.capitalize()

    # Heatmap
    heatmap_path = plots_dir / f"bridge_coverage_{chamber}.png"
    if heatmap_path.exists():
        report.add(
            FigureSection.from_file(
                f"bridge-heatmap-{chamber}",
                f"{chamber_cap} — Bridge Coverage Heatmap",
                heatmap_path,
                caption=(
                    "Number of legislators serving in both bienniums. The Markov chain "
                    "of bridges requires sufficient overlap between consecutive periods."
                ),
                alt_text=(
                    f"Heatmap of {chamber_cap} bridge legislator counts between all pairs of "
                    "bienniums. Brighter cells indicate more shared legislators linking periods."
                ),
            )
        )

    # Adjacent bridges table
    bridge_adj = cr.get("bridge_adj")
    if bridge_adj is not None and bridge_adj.height > 0:
        display = bridge_adj.select(
            "pair", "shared_count", "total_a", "total_b", "overlap_pct", "sufficient"
        )
        report.add(
            TableSection(
                id=f"bridge-adjacent-{chamber}",
                title=f"{chamber_cap} — Adjacent Bridge Coverage",
                html=make_gt(
                    display,
                    title=f"{chamber_cap} Bridge Coverage",
                    subtitle=f"Minimum shared legislators for valid bridge: {MIN_BRIDGE_OVERLAP}",
                    column_labels={
                        "pair": "Transition",
                        "shared_count": "Shared",
                        "total_a": "Period A",
                        "total_b": "Period B",
                        "overlap_pct": "Overlap %",
                        "sufficient": "Sufficient",
                    },
                    number_formats={"overlap_pct": ".1f"},
                ),
            )
        )


def _add_sign_corrections(
    report: ReportBuilder,
    cr: dict,
    chamber: str,
) -> None:
    """Sign correction transparency section."""
    chamber_cap = chamber.capitalize()
    corrections = cr.get("sign_corrections", [])

    if not corrections:
        report.add(
            TextSection(
                id=f"sign-corrections-{chamber}",
                title=f"{chamber_cap} — Sign Corrections",
                html=(
                    "<p>No sign corrections were required. All periods had positive "
                    "Pearson r with static IRT (Phase 04).</p>"
                ),
            )
        )
        return

    lines = [
        "<p>The following periods had their ideal point signs corrected based on "
        "correlation with static IRT (Phase 04). A negative Pearson r indicates "
        "the dynamic model found a sign-flipped mode — positive beta alone is "
        "insufficient for sign identification when the random walk chain is broken "
        "(e.g., missing biennium data creating 0 bridge legislators).</p>",
        "<table style='border-collapse: collapse; margin: 1em 0;'>",
        "<tr>"
        "<th style='padding: 4px 12px; text-align: left;'>Biennium</th>"
        "<th style='padding: 4px 12px;'>r (before)</th>"
        "<th style='padding: 4px 12px;'>r (after)</th>"
        "<th style='padding: 4px 12px;'>N Matched</th>"
        "</tr>",
    ]

    for c in corrections:
        lines.append(
            f"<tr>"
            f"<td style='padding: 4px 12px;'>{c['label']}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{c['r_before']:.3f}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{c['r_after']:.3f}</td>"
            f"<td style='padding: 4px 12px; text-align: right;'>{c['n_matched']}</td>"
            f"</tr>"
        )
    lines.append("</table>")

    # Reference legislators for each corrected period
    for c in corrections:
        lines.append(f"<p><strong>{c['label']} reference legislators:</strong></p>")
        lines.append("<ul>")
        for name, dyn_xi, static_xi in c["reference_legs"]:
            lines.append(f"<li>{name}: dynamic = {dyn_xi:+.2f}, static = {static_xi:+.2f}</li>")
        lines.append("</ul>")

    report.add(
        TextSection(
            id=f"sign-corrections-{chamber}",
            title=f"{chamber_cap} — Sign Corrections",
            html="\n".join(lines),
        )
    )


def _add_global_roster(
    report: ReportBuilder,
    cr: dict,
    chamber: str,
) -> None:
    """Global roster table sorted by periods served."""
    chamber_cap = chamber.capitalize()
    roster = cr.get("roster")
    if roster is None or roster.height == 0:
        return

    display = roster.select(
        "full_name", "parties", "first_period", "last_period", "n_periods"
    ).sort("n_periods", descending=True)

    report.add(
        TableSection(
            id=f"roster-{chamber}",
            title=f"{chamber_cap} — Global Legislator Roster",
            html=make_gt(
                display,
                title=f"{chamber_cap} Legislators Across Bienniums",
                subtitle=f"{roster.height} unique legislators",
                column_labels={
                    "full_name": "Name",
                    "parties": "Party",
                    "first_period": "First",
                    "last_period": "Last",
                    "n_periods": "Bienniums",
                },
            ),
            caption="Sorted by number of bienniums served.",
        )
    )


def _add_polarization_trend(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Polarization trend figure."""
    chamber_cap = chamber.capitalize()
    path = plots_dir / f"polarization_trend_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"polarization-{chamber}",
                f"{chamber_cap} — Polarization Trend",
                path,
                caption=(
                    "Party mean ideal points across bienniums with 95% bands. "
                    "Increasing distance between party means indicates growing polarization."
                ),
                alt_text=(
                    f"Line chart of {chamber_cap} party mean ideal points across bienniums with "
                    "95% credible interval bands. Growing separation between party lines indicates "
                    "increasing polarization."
                ),
            )
        )


def _add_ridgeline(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Add ridgeline ideology distribution plot."""
    chamber_cap = chamber.capitalize()
    path = plots_dir / f"ridgeline_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"ridgeline-{chamber}",
                f"{chamber_cap} — Ideological Ridgeline",
                path,
                caption=(
                    "Kernel density estimates of ideal point distributions for each biennium. "
                    "Republicans (red) and Democrats (blue) shown separately. "
                    "Wider peaks indicate more within-party ideological variation. "
                    "Increasing separation between party peaks indicates growing polarization."
                ),
                alt_text=(
                    f"Ridgeline plot of {chamber_cap} ideological distributions stacked "
                    "vertically by biennium. Republican (red) and Democrat (blue) kernel "
                    "density curves show party ideal point distributions over time."
                ),
            )
        )


def _add_animated_scatter(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Add animated ideal point scatter (Gapminder-style)."""
    path = plots_dir / f"animated_scatter_{chamber}.html"
    if not path.exists():
        return
    report.add(
        InteractiveSection(
            id=f"animated-scatter-{chamber.lower()}",
            title=f"{chamber.capitalize()} — Ideal Point Animation",
            html=path.read_text(),
            caption=(
                "Press play to animate legislator positions across bienniums. "
                "X-axis: ideal point (left=liberal, right=conservative). "
                "Y-axis: estimation uncertainty (higher = less certain). "
                "Hover over any dot for legislator details and 95% credible interval."
            ),
            aria_label=(
                f"Animated scatter plot of {chamber.capitalize()} legislator ideal points "
                "across bienniums. Each frame shows one biennium with legislators positioned "
                "by ideal point (x) and uncertainty (y), colored by party."
            ),
        )
    )


def _add_trajectories(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Individual trajectories spaghetti plot."""
    chamber_cap = chamber.capitalize()
    path = plots_dir / f"trajectories_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"trajectories-{chamber}",
                f"{chamber_cap} — Individual Ideal Point Trajectories",
                path,
                caption=(
                    "Spaghetti plot of individual trajectories. Top movers by total "
                    "ideological shift are highlighted with labels."
                ),
                alt_text=(
                    f"Spaghetti plot of {chamber_cap} individual ideal point "
                    "trajectories across bienniums. Top movers with the largest "
                    "total ideological shift are highlighted and labeled."
                ),
            )
        )


def _add_tau_posterior(
    report: ReportBuilder,
    cr: dict,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Evolution variance tau posterior."""
    chamber_cap = chamber.capitalize()

    # Table
    tau = cr.get("tau")
    if tau is not None and tau.height > 0:
        report.add(
            TableSection(
                id=f"tau-table-{chamber}",
                title=f"{chamber_cap} — Evolution Variance (tau) Posterior",
                html=make_gt(
                    tau,
                    title=f"{chamber_cap} tau Posterior",
                    column_labels={
                        "party": "Party",
                        "tau_mean": "Mean",
                        "tau_sd": "SD",
                        "tau_hdi_2.5": "HDI 2.5%",
                        "tau_hdi_97.5": "HDI 97.5%",
                    },
                    number_formats={
                        "tau_mean": ".4f",
                        "tau_sd": ".4f",
                        "tau_hdi_2.5": ".4f",
                        "tau_hdi_97.5": ".4f",
                    },
                ),
            )
        )

    # Figure
    path = plots_dir / f"tau_posterior_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"tau-kde-{chamber}",
                f"{chamber_cap} — tau Posterior Distribution",
                path,
                caption="Larger tau indicates more ideological movement between bienniums.",
                alt_text=(
                    f"KDE plot of the {chamber_cap} evolution variance (tau) "
                    "posterior distribution by party. Larger tau values indicate "
                    "more ideological movement between bienniums."
                ),
            )
        )


def _add_top_movers_table(
    report: ReportBuilder,
    cr: dict,
    chamber: str,
) -> None:
    """Top movers table."""
    chamber_cap = chamber.capitalize()
    top_movers = cr.get("top_movers")
    if top_movers is None or top_movers.height == 0:
        return

    display = top_movers.select(
        "full_name",
        "party",
        "n_periods",
        "first_period",
        "last_period",
        "total_movement",
        "net_movement",
        "direction",
    )

    report.add(
        TableSection(
            id=f"movers-table-{chamber}",
            title=f"{chamber_cap} — Top {display.height} Movers",
            html=make_gt(
                display,
                title=f"{chamber_cap} Top Movers by Total Ideological Shift",
                column_labels={
                    "full_name": "Name",
                    "party": "Party",
                    "n_periods": "Bienniums",
                    "first_period": "First",
                    "last_period": "Last",
                    "total_movement": "Total",
                    "net_movement": "Net",
                    "direction": "Direction",
                },
                number_formats={
                    "total_movement": ".3f",
                    "net_movement": ".3f",
                },
            ),
        )
    )


def _add_top_movers_bar(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Top movers bar chart."""
    chamber_cap = chamber.capitalize()
    path = plots_dir / f"top_movers_bar_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"movers-bar-{chamber}",
                f"{chamber_cap} — Top Movers Net Shift",
                path,
                caption="Red = rightward shift, blue = leftward shift.",
                alt_text=(
                    f"Horizontal bar chart of {chamber_cap} top movers ranked by "
                    "net ideological shift. Red bars indicate rightward movement; "
                    "blue bars indicate leftward movement."
                ),
            )
        )


def _add_conversion_replacement_table(
    report: ReportBuilder,
    cr: dict,
    chamber: str,
) -> None:
    """Conversion vs replacement decomposition table."""
    chamber_cap = chamber.capitalize()
    decomposition = cr.get("decomposition")
    if decomposition is None or decomposition.height == 0:
        return

    report.add(
        TableSection(
            id=f"decomp-table-{chamber}",
            title=f"{chamber_cap} — Conversion vs. Replacement",
            html=make_gt(
                decomposition,
                title=f"{chamber_cap} Polarization Decomposition",
                subtitle="Total shift = conversion (returning members' movement) + replacement "
                "(new vs departing)",
                column_labels={
                    "pair": "Transition",
                    "party": "Party",
                    "total_shift": "Total",
                    "conversion": "Conversion",
                    "replacement": "Replacement",
                    "n_returning": "Returning",
                    "n_departing": "Departing",
                    "n_new": "New",
                },
                number_formats={
                    "total_shift": ".4f",
                    "conversion": ".4f",
                    "replacement": ".4f",
                },
            ),
        )
    )


def _add_conversion_replacement_plot(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Conversion vs replacement stacked bar chart."""
    chamber_cap = chamber.capitalize()
    path = plots_dir / f"conversion_replacement_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"decomp-plot-{chamber}",
                f"{chamber_cap} — Conversion vs. Replacement",
                path,
                caption=(
                    "Stacked bars: conversion = shift in returning members' positions, "
                    "replacement = effect of turnover (new members replacing departing)."
                ),
                alt_text=(
                    f"Stacked bar chart of {chamber_cap} polarization decomposition "
                    "per transition. Each bar splits total shift into conversion "
                    "(returning members' movement) and replacement "
                    "(effect of member turnover)."
                ),
            )
        )


def _add_static_correlation(
    report: ReportBuilder,
    cr: dict,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Static IRT correlation table and figure."""
    chamber_cap = chamber.capitalize()
    correlation = cr.get("correlation")
    if correlation is None or correlation.height == 0:
        return

    # Build column labels dynamically based on available columns
    col_labels: dict[str, str] = {
        "biennium": "Biennium",
        "n_matched": "N Matched",
        "pearson_r": "Pearson r",
        "spearman_rho": "Spearman ρ",
    }
    if "sign_corrected" in correlation.columns:
        col_labels["sign_corrected"] = "Sign Corrected"

    report.add(
        TableSection(
            id=f"static-corr-{chamber}",
            title=f"{chamber_cap} — Dynamic vs. Static IRT Correlation",
            html=make_gt(
                correlation,
                title=f"{chamber_cap} Correlation with Per-Biennium Static IRT",
                column_labels=col_labels,
                number_formats={
                    "pearson_r": ".3f",
                    "spearman_rho": ".3f",
                },
            ),
        )
    )

    path = plots_dir / f"static_correlation_{chamber}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"static-corr-fig-{chamber}",
                f"{chamber_cap} — Static IRT Correlation",
                path,
                caption="Per-biennium correlation between dynamic and static IRT ideal points.",
                alt_text=(
                    f"Bar chart or scatter of {chamber_cap} per-biennium Pearson "
                    "correlation between dynamic and static IRT ideal points, "
                    "validating cross-method consistency."
                ),
            )
        )


def _add_emirt_comparison(
    report: ReportBuilder,
    cr: dict,
    chamber: str,
) -> None:
    """emIRT vs PyMC comparison (conditional section)."""
    chamber_cap = chamber.capitalize()
    emirt = cr.get("emirt_results")
    if emirt is None or emirt.height == 0:
        return

    report.add(
        TextSection(
            id=f"emirt-{chamber}",
            title=f"{chamber_cap} — emIRT Comparison",
            html=(
                f"<p>emIRT::dynIRT results available for {chamber_cap}. "
                "See scatter plot in output directory for visual comparison "
                "with PyMC estimates.</p>"
            ),
        )
    )


def _add_convergence(
    report: ReportBuilder,
    cr: dict,
    chamber: str,
) -> None:
    """Convergence diagnostics summary."""
    chamber_cap = chamber.capitalize()
    conv = cr.get("convergence")
    if conv is None:
        return

    passed_str = "PASSED" if conv["passed"] else "FAILED"
    color = "green" if conv["passed"] else "red"

    lines = [
        f"<h3 style='color: {color};'>Convergence: {passed_str}</h3>",
        "<table style='border-collapse: collapse; margin: 1em 0;'>",
        f"<tr><td style='padding: 4px 12px;'>R-hat max</td>"
        f"<td style='padding: 4px 12px; text-align: right;'>{conv['rhat_max']:.4f}</td>"
        f"<td style='padding: 4px 12px;'>(threshold: {conv['rhat_threshold']})</td></tr>",
        f"<tr><td style='padding: 4px 12px;'>ESS bulk min</td>"
        f"<td style='padding: 4px 12px; text-align: right;'>{conv['ess_bulk_min']:.0f}</td>"
        f"<td style='padding: 4px 12px;'>(threshold: {conv['ess_threshold']})</td></tr>",
        f"<tr><td style='padding: 4px 12px;'>ESS tail min</td>"
        f"<td style='padding: 4px 12px; text-align: right;'>{conv['ess_tail_min']:.0f}</td>"
        f"<td style='padding: 4px 12px;'></td></tr>",
        f"<tr><td style='padding: 4px 12px;'>Divergences</td>"
        f"<td style='padding: 4px 12px; text-align: right;'>{conv['n_divergences']}</td>"
        f"<td style='padding: 4px 12px;'>(max: {MAX_DIVERGENCES})</td></tr>",
        f"<tr><td style='padding: 4px 12px;'>Total params</td>"
        f"<td style='padding: 4px 12px; text-align: right;'>{conv['n_params']:,}</td>"
        f"<td style='padding: 4px 12px;'></td></tr>",
        "</table>",
    ]

    report.add(
        TextSection(
            id=f"convergence-{chamber}",
            title=f"{chamber_cap} — Convergence Summary",
            html="\n".join(lines),
        )
    )


def _add_model_priors(report: ReportBuilder, results: dict) -> None:
    """Model priors and tuning parameters transparency section."""
    lines = [
        "<h3>Prior Distributions</h3>",
        "<table style='border-collapse: collapse; margin: 1em 0;'>",
        "<tr>"
        "<th style='padding: 4px 12px; text-align: left;'>Parameter</th>"
        "<th style='padding: 4px 12px; text-align: left;'>Prior</th>"
        "<th style='padding: 4px 12px; text-align: left;'>Purpose</th>"
        "</tr>",
        "<tr><td style='padding: 4px 12px;'>tau</td>"
        "<td style='padding: 4px 12px;'>HalfNormal(0.5), per party</td>"
        "<td style='padding: 4px 12px;'>Evolution SD (ideal point drift)</td></tr>",
        "<tr><td style='padding: 4px 12px;'>xi_init</td>"
        "<td style='padding: 4px 12px;'>Normal(0, 1)</td>"
        "<td style='padding: 4px 12px;'>Initial ideal points (PCA-initialized)</td></tr>",
        "<tr><td style='padding: 4px 12px;'>xi_innovations</td>"
        "<td style='padding: 4px 12px;'>Normal(0, 1)</td>"
        "<td style='padding: 4px 12px;'>Non-centered random walk steps</td></tr>",
        "<tr><td style='padding: 4px 12px;'>alpha</td>"
        "<td style='padding: 4px 12px;'>Normal(0, 5), per bill</td>"
        "<td style='padding: 4px 12px;'>Bill difficulty</td></tr>",
        "<tr><td style='padding: 4px 12px;'>beta</td>"
        "<td style='padding: 4px 12px;'>HalfNormal(2.5), per bill</td>"
        "<td style='padding: 4px 12px;'>Bill discrimination (positive for sign ID)</td></tr>",
        "</table>",
    ]

    # MCMC settings from first available chamber
    chambers = results.get("chambers", [])
    for chamber in sorted(chambers):
        if chamber in results and "mcmc_params" in results[chamber]:
            params = results[chamber]["mcmc_params"]
            pca_range = results[chamber].get("pca_init_range")
            lines.append("<h3>MCMC Settings</h3>")
            lines.append("<table style='border-collapse: collapse; margin: 1em 0;'>")
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>Samples</td>"
                f"<td style='padding: 4px 12px;'>{params['n_samples']}</td></tr>"
            )
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>Tuning</td>"
                f"<td style='padding: 4px 12px;'>{params['n_tune']}</td></tr>"
            )
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>Chains</td>"
                f"<td style='padding: 4px 12px;'>{params['n_chains']}</td></tr>"
            )
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>Seed</td>"
                f"<td style='padding: 4px 12px;'>{params['seed']}</td></tr>"
            )
            lines.append(
                "<tr><td style='padding: 4px 12px;'>Sampler</td>"
                "<td style='padding: 4px 12px;'>nutpie (Rust NUTS)</td></tr>"
            )
            if pca_range:
                lines.append(
                    f"<tr><td style='padding: 4px 12px;'>PCA init range</td>"
                    f"<td style='padding: 4px 12px;'>"
                    f"[{pca_range[0]:.2f}, {pca_range[1]:.2f}]</td></tr>"
                )
            lines.append("</table>")

            lines.append("<h3>Convergence Thresholds</h3>")
            lines.append("<table style='border-collapse: collapse; margin: 1em 0;'>")
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>R-hat</td>"
                f"<td style='padding: 4px 12px;'>{RHAT_THRESHOLD}</td></tr>"
            )
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>ESS min</td>"
                f"<td style='padding: 4px 12px;'>{ESS_THRESHOLD}</td></tr>"
            )
            lines.append(
                f"<tr><td style='padding: 4px 12px;'>Max divergences</td>"
                f"<td style='padding: 4px 12px;'>{MAX_DIVERGENCES}</td></tr>"
            )
            lines.append("</table>")

            lines.append(
                "<p><strong>Sign correction:</strong> If a period's dynamic xi correlates "
                "negatively with static IRT (Phase 04), xi is negated for that period. "
                "Corrected periods are documented in the Sign Corrections section above.</p>"
            )
            break  # only need settings once

    report.add(
        TextSection(
            id="model-priors",
            title="Model Priors & Tuning Parameters",
            html="\n".join(lines),
        )
    )


def _generate_dynamic_key_findings(results: dict) -> list[str]:
    """Generate 2-4 key findings from dynamic IRT results."""
    findings: list[str] = []

    chambers = results.get("chambers", [])
    n_periods = results.get("n_periods") or results.get("n_bienniums")
    if n_periods is not None:
        findings.append(f"Dynamic IRT estimated across <strong>{n_periods}</strong> time periods.")

    for chamber in sorted(chambers):
        cr = results.get(chamber, {})

        # Polarization trend
        polarization = cr.get("polarization_trend", {})
        slope = polarization.get("slope")
        if slope is not None:
            direction = "increasing" if slope > 0 else "decreasing"
            findings.append(
                f"{chamber} polarization: <strong>{direction}</strong> trend (slope = {slope:.4f})."
            )

        # Top movers
        movers = cr.get("top_movers")
        if movers is not None and hasattr(movers, "height") and movers.height > 0:
            top = movers.head(1)
            name_col = "full_name" if "full_name" in top.columns else "legislator_slug"
            name = top[name_col][0]
            shift_col = "total_shift" if "total_shift" in top.columns else "total_drift"
            if shift_col in top.columns:
                shift = float(top[shift_col][0])
                findings.append(
                    f"{chamber} largest mover: <strong>{name}</strong> (total shift = {shift:.3f})."
                )
            else:
                findings.append(f"{chamber} largest mover: <strong>{name}</strong>.")

        break  # First chamber only

    return findings


def _add_methodology(report: ReportBuilder) -> None:
    """Methodology section describing the model."""
    html = """
    <h3>Model Specification</h3>
    <p>Martin-Quinn style state-space 2PL IRT with non-centered random walk:</p>
    <pre>
tau ~ HalfNormal(0.5, dims="party")          # per-party evolution SD
xi_init ~ Normal(0, 1, dims="legislator")    # initial ideal points
xi_innovations ~ Normal(0, 1)                # non-centered innovations
xi[0] = xi_init
xi[t] = xi[t-1] + tau[party] * innovation    # random walk
alpha ~ Normal(0, 5, dims="bill")            # bill difficulty
beta ~ HalfNormal(2.5, dims="bill")          # bill discrimination (positive)
obs ~ Bernoulli(logit(beta * xi - alpha))
    </pre>

    <h3>Identification</h3>
    <p>Positive beta (HalfNormal prior) provides sign identification without anchor
    constraints. Cross-period scale linking is achieved through bridge legislators
    (those serving multiple bienniums), which anchor the random walk across time.</p>

    <h3>Identification Caveat</h3>
    <p>Positive beta alone is insufficient when the random walk chain is broken —
    for example, when a biennium's data is missing, creating 0 bridge legislators
    on both sides of the gap. In such cases, a post-hoc sign correction step
    (ADR-0068) validates each period against static IRT and negates xi if the
    correlation is negative. The Senate is less susceptible due to higher
    legislator continuity from 4-year staggered terms.</p>

    <h3>Decomposition</h3>
    <p>For each adjacent biennium pair, per party:</p>
    <ul>
        <li><strong>Total shift</strong> = change in party mean ideal point</li>
        <li><strong>Conversion</strong> = movement of returning members</li>
        <li><strong>Replacement</strong> = effect of new members replacing departing ones</li>
        <li>Total ≈ conversion + replacement</li>
    </ul>

    <h3>Sampler</h3>
    <p>nutpie (Rust NUTS sampler) with PCA-informed initialization for xi_init.
    Non-centered parameterization prevents funnel geometry in the random walk.</p>
    """
    report.add(TextSection(id="methodology", title="Methodology", html=html))


# Import constants from the main module for report sections
try:
    from analysis.dynamic_irt import ESS_THRESHOLD, MAX_DIVERGENCES, RHAT_THRESHOLD
except ModuleNotFoundError, ImportError:
    MAX_DIVERGENCES = 50  # type: ignore[assignment]
    RHAT_THRESHOLD = 1.05  # type: ignore[assignment]
    ESS_THRESHOLD = 400  # type: ignore[assignment]
