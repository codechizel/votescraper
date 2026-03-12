"""TSA-specific HTML report builder.

Builds sections (tables, figures, and text) for the time series analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from tsa.py):
    from analysis.tsa_report import build_tsa_report
    build_tsa_report(ctx.report, results=results, ...)
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

# Constants duplicated from tsa.py to avoid circular import
WINDOW_SIZE = 75
STEP_SIZE = 15
MIN_WINDOW_VOTES = 10
MIN_WINDOW_LEGISLATORS = 20
PELT_MIN_SIZE = 5
WEEKLY_AGG_DAYS = 7
TOP_MOVERS_N = 10
MIN_TOTAL_VOTES = 20


def build_tsa_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    skip_drift: bool = False,
    skip_changepoints: bool = False,
    penalty: float = 10.0,
    r_available: bool = False,
    horseshoe_status: dict[str, dict] | None = None,
) -> None:
    """Build the full TSA HTML report by adding sections to the ReportBuilder."""
    from analysis.phase_utils import horseshoe_warning_html

    findings = _generate_tsa_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    # Horseshoe warnings
    if horseshoe_status:
        for chamber, status in horseshoe_status.items():
            warning = horseshoe_warning_html(chamber, status)
            if warning:
                report.add(
                    TextSection(
                        id=f"horseshoe-warning-{chamber.lower()}",
                        title=f"{chamber} Horseshoe Warning",
                        html=warning,
                    )
                )

    _add_data_summary(report, results)
    _add_how_to_read(report)

    if not skip_drift:
        for chamber, result in results.items():
            _add_drift_figures(report, plots_dir, chamber)
            _add_top_movers_table(report, result, chamber)
            _add_early_late_table(report, result, chamber)
        _add_drift_interpretation(report)

    if not skip_changepoints:
        for chamber, result in results.items():
            _add_changepoint_figures(report, plots_dir, chamber)
            _add_changepoint_summary_table(report, result, chamber)
        _add_changepoint_interpretation(report)

        # R enrichment sections (CROPS + Bai-Perron)
        if r_available:
            for chamber, result in results.items():
                _add_crops_figures(report, plots_dir, chamber)
            for chamber, result in results.items():
                _add_bai_perron_table(report, result, chamber)
                _add_bai_perron_figures(report, plots_dir, chamber)
            for chamber, result in results.items():
                _add_pelt_bp_crossref(report, result, chamber)
            _add_r_enrichment_interpretation(report)

        _add_veto_crossref_table(report, results)

    _add_analysis_parameters(report, penalty, results=results, r_available=r_available)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_data_summary(report: ReportBuilder, results: dict[str, dict]) -> None:
    """Table: Data dimensions per chamber."""
    rows = []
    for chamber, result in results.items():
        rows.append(
            {
                "Chamber": chamber,
                "Legislators": result.get("n_legislators", 0),
                "Roll Calls": result.get("n_rollcalls", 0),
                "PCA Windows": result.get("n_windows", 0),
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Time Series Analysis Data Summary",
        subtitle="Vote matrix dimensions and rolling PCA window count per chamber",
        source_note=(
            f"Window size: {WINDOW_SIZE} roll calls, step: {STEP_SIZE}. "
            f"Legislators with < {MIN_TOTAL_VOTES} total votes excluded."
        ),
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_how_to_read(report: ReportBuilder) -> None:
    """Text block: how to interpret this report."""
    report.add(
        TextSection(
            id="how-to-read",
            title="How to Read This Report",
            html=(
                "<p>This report presents two types of temporal analysis:</p>"
                "<ul>"
                "<li><strong>Ideological Drift</strong> — Did anyone change their voting "
                "position during the session? Rolling-window PCA tracks each legislator's "
                "ideological score over time. Party means show whether polarization increased "
                "or decreased. Individual trajectories reveal who moved the most.</li>"
                "<li><strong>Changepoint Detection</strong> — Were there structural breaks in "
                "party cohesion? The Rice Index (party unity per vote) is aggregated weekly and "
                "analyzed for abrupt shifts. These might correspond to legislative events like "
                "veto overrides, leadership changes, or end-of-session deal-making.</li>"
                "</ul>"
                "<p><strong>Party drift plot:</strong> Lines moving apart = increasing "
                "polarization. Lines converging = bipartisan periods.</p>"
                "<p><strong>Early vs late scatter:</strong> Points on the diagonal = no change. "
                "Points above = moved toward Republican positions. Below = moved toward Democrat "
                "positions.</p>"
                "<p><strong>Changepoint lines:</strong> Red dashed lines mark detected structural "
                "breaks. Stable across different penalty values = robust finding.</p>"
            ),
        )
    )


def _add_drift_figures(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Add the four drift figures for a chamber."""
    ch = chamber.lower()

    path = plots_dir / f"party_drift_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-party-drift-{ch}",
                f"{chamber} Party Ideological Trajectories",
                path,
                caption=(
                    "Party mean PC1 scores across rolling PCA windows. "
                    "Shaded area = polarization gap between parties."
                ),
                alt_text=(
                    f"Line chart of {chamber} party mean PC1 scores across rolling "
                    "PCA windows. Shaded region between Republican and Democrat lines "
                    "shows the polarization gap over time."
                ),
            )
        )

    path = plots_dir / f"polarization_gap_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-pol-gap-{ch}",
                f"{chamber} Polarization Gap",
                path,
                caption="Absolute difference between Republican and Democrat mean PC1 over time.",
                alt_text=(
                    f"Line chart of the {chamber} polarization gap over time, "
                    "showing the absolute difference between Republican and Democrat "
                    "mean PC1 scores across rolling windows."
                ),
            )
        )

    path = plots_dir / f"top_movers_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-top-movers-{ch}",
                f"{chamber} Top Individual Drifters",
                path,
                caption=(
                    f"PC1 trajectories for the {TOP_MOVERS_N} legislators with the largest "
                    "position change between early and late session."
                ),
                alt_text=(
                    f"Spaghetti plot of PC1 trajectories for the {TOP_MOVERS_N} {chamber} "
                    "legislators with the largest ideological drift between early and late session."
                ),
            )
        )

    path = plots_dir / f"early_vs_late_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-early-late-{ch}",
                f"{chamber} Early vs Late Session Position",
                path,
                caption=(
                    "Each point is one legislator. Diagonal = no change. "
                    "Distance from diagonal = magnitude of drift."
                ),
                alt_text=(
                    f"Scatter plot comparing each {chamber} legislator's early-session "
                    "PC1 score to their late-session score. Points off the diagonal "
                    "indicate ideological drift."
                ),
            )
        )


def _add_top_movers_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Top movers with drift scores."""
    top_movers = result.get("top_movers")
    if top_movers is None or top_movers.height == 0:
        return

    display_cols = ["full_name", "party", "early_pc1", "late_pc1", "drift"]
    available = [c for c in display_cols if c in top_movers.columns]
    display = top_movers.select(available)

    html = make_gt(
        display,
        title=f"{chamber} — Top Movers (Largest Ideological Drift)",
        subtitle="Legislators whose PC1 position changed most between early and late session",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "early_pc1": "Early PC1",
            "late_pc1": "Late PC1",
            "drift": "Drift",
        },
        number_formats={
            "early_pc1": ".3f",
            "late_pc1": ".3f",
            "drift": ".3f",
        },
        source_note=(
            "Positive drift = moved toward Republican positions. "
            "Negative = moved toward Democrat positions."
        ),
    )
    report.add(
        TableSection(
            id=f"top-movers-{chamber.lower()}",
            title=f"{chamber} Top Movers",
            html=html,
        )
    )


def _add_early_late_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Early vs late PC1 for all legislators."""
    drift_df = result.get("drift_df")
    if drift_df is None or drift_df.height == 0:
        return

    display = drift_df.sort("drift").select("full_name", "party", "early_pc1", "late_pc1", "drift")

    html = make_gt(
        display,
        title=f"{chamber} — All Legislators Early vs Late ({display.height} legislators)",
        subtitle="PC1 scores from first half and second half of session (never truncated)",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "early_pc1": "Early PC1",
            "late_pc1": "Late PC1",
            "drift": "Drift",
        },
        number_formats={
            "early_pc1": ".3f",
            "late_pc1": ".3f",
            "drift": ".3f",
        },
    )
    report.add(
        TableSection(
            id=f"early-late-all-{chamber.lower()}",
            title=f"{chamber} Early vs Late (All)",
            html=html,
        )
    )


def _add_drift_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting drift results."""
    report.add(
        TextSection(
            id="drift-interpretation",
            title="Interpreting Ideological Drift",
            html=(
                "<p><strong>Rolling PCA</strong> recomputes the first principal component of the "
                f"vote matrix in overlapping windows of {WINDOW_SIZE} consecutive roll calls. "
                "PC1 captures the dominant dimension of ideological variation — in Kansas, this "
                "is the party divide.</p>"
                "<p>By tracking each legislator's PC1 score across windows, we can identify:</p>"
                "<ul>"
                "<li><strong>Party-level trends:</strong> Are the parties moving apart (increasing "
                "polarization) or converging (bipartisan periods)?</li>"
                "<li><strong>Individual movers:</strong> Which legislators changed position the "
                "most? This could indicate genuine ideological shifts, changed issue priorities, "
                "or responsiveness to constituent pressure.</li>"
                "</ul>"
                "<p>The <strong>early vs late</strong> comparison divides the session at the "
                "midpoint. Points far from the diagonal represent legislators whose voting "
                "patterns changed substantially between the first and second halves.</p>"
                "<p><strong>Caveat:</strong> PCA captures the dominant dimension, which may shift "
                "meaning across windows if the issue space changes. The sign convention "
                "(Republicans = positive) is enforced per window to maintain consistency.</p>"
            ),
        )
    )


def _add_changepoint_figures(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Add changepoint figures for a chamber."""
    ch = chamber.lower()

    for party in ["republican", "democrat"]:
        path = plots_dir / f"changepoints_{party}_{ch}.png"
        if path.exists():
            party_label = party.title()
            report.add(
                FigureSection.from_file(
                    f"fig-cp-{party}-{ch}",
                    f"{chamber} {party_label} Changepoints",
                    path,
                    caption=(
                        f"Weekly mean Rice Index for {party_label}s with PELT-detected "
                        "structural breaks (red dashed lines)."
                    ),
                    alt_text=(
                        f"Time series of weekly Rice Index for {chamber} {party_label}s "
                        "with red dashed vertical lines marking PELT-detected changepoints."
                    ),
                )
            )

    path = plots_dir / f"changepoints_joint_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-cp-joint-{ch}",
                f"{chamber} Joint Changepoints",
                path,
                caption=(
                    "Both parties' Rice with jointly detected breaks. "
                    "These are shifts affecting both parties simultaneously."
                ),
                alt_text=(
                    f"Time series of weekly Rice Index for both {chamber} parties "
                    "with jointly detected structural breaks marking shifts "
                    "affecting both parties simultaneously."
                ),
            )
        )

    path = plots_dir / f"penalty_sensitivity_{ch}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-sensitivity-{ch}",
                f"{chamber} Penalty Sensitivity",
                path,
                caption=(
                    "Number of detected changepoints vs PELT penalty. "
                    "Flat regions = robust changepoints. Steep drops = sensitive to tuning."
                ),
                alt_text=(
                    f"Step chart of {chamber} changepoint count versus PELT penalty "
                    "value. Flat regions indicate robust changepoints; steep drops "
                    "indicate sensitivity to tuning."
                ),
            )
        )


def _add_changepoint_summary_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Changepoint summary per party."""
    cp_results = result.get("changepoints", {})
    if not cp_results:
        return

    rows = []
    for key in ["Republican", "Democrat", "joint"]:
        if key not in cp_results:
            continue
        cp_data = cp_results[key]
        rows.append(
            {
                "Analysis": key.title() if key != "joint" else "Joint (Both Parties)",
                "N Changepoints": cp_data.get("n_changepoints", 0),
                "Dates": ", ".join(cp_data.get("cp_dates", [])[:5]),
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Changepoint Summary",
        subtitle="Number and timing of detected structural breaks in party cohesion",
        source_note="Dates shown for up to 5 changepoints. Joint = 2D multivariate detection.",
    )
    report.add(
        TableSection(
            id=f"cp-summary-{chamber.lower()}",
            title=f"{chamber} Changepoint Summary",
            html=html,
        )
    )


def _add_changepoint_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting changepoints."""
    report.add(
        TextSection(
            id="changepoint-interpretation",
            title="Interpreting Changepoints",
            html=(
                "<p><strong>PELT</strong> (Pruned Exact Linear Time) detects abrupt changes in "
                "the statistical properties of a time series. It looks for moments where the "
                "distribution of weekly Rice Index values shifts — either the mean level changes, "
                "the variance changes, or both.</p>"
                "<p>The <strong>RBF kernel</strong> makes PELT sensitive to changes in both the "
                "mean and spread of cohesion. This is important because a party might maintain the "
                "same average cohesion while becoming more erratic (higher variance).</p>"
                "<p>The <strong>penalty parameter</strong> controls how many changepoints are "
                "detected. Higher penalties require stronger evidence before declaring a break, "
                "producing fewer changepoints. The sensitivity plot shows how changepoint count "
                "varies with penalty — flat regions indicate robust changepoints.</p>"
                "<p><strong>Joint detection</strong> finds breaks affecting both parties "
                "simultaneously. These often correspond to session-wide events (leadership "
                "changes, major legislation, end-of-session dynamics) rather than "
                "party-specific shifts.</p>"
                "<p>Changepoints near <strong>veto override votes</strong> suggest that override "
                "coalitions disrupted normal party cohesion patterns.</p>"
            ),
        )
    )


def _add_crops_figures(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Add CROPS solution path figures for a chamber."""
    ch = chamber.lower()

    for party in ["republican", "democrat"]:
        path = plots_dir / f"crops_{party}_{ch}.png"
        if path.exists():
            party_label = party.title()
            report.add(
                FigureSection.from_file(
                    f"fig-crops-{party}-{ch}",
                    f"{chamber} {party_label} CROPS Solution Path",
                    path,
                    caption=(
                        f"Exact penalty thresholds where the optimal segmentation changes "
                        f"for {party_label}s. Each step = one fewer changepoint. Diamond = "
                        f"elbow (penalty of diminishing returns)."
                    ),
                    alt_text=(
                        f"Step function of {chamber} {party_label} CROPS solution path showing "
                        "changepoint count versus exact penalty thresholds. Diamond marks the "
                        "elbow where diminishing returns begin."
                    ),
                )
            )


def _add_bai_perron_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Bai-Perron break dates with 95% confidence intervals."""
    cp_results = result.get("changepoints", {})
    if not cp_results:
        return

    rows = []
    for party in ["Republican", "Democrat"]:
        bp_key = f"{party}_bai_perron"
        if bp_key not in cp_results:
            continue
        bp_data = cp_results[bp_key]
        bp_df = bp_data.get("bp_df")
        if bp_df is None or bp_df.height == 0:
            continue

        for bp_row in bp_df.iter_rows(named=True):
            ci_lo = str(bp_row["ci_lower_date"])[:10]
            ci_hi = str(bp_row["ci_upper_date"])[:10]
            bp_date = str(bp_row["break_date"])[:10]
            ci_lo_d = __import__("datetime").date.fromisoformat(ci_lo)
            ci_hi_d = __import__("datetime").date.fromisoformat(ci_hi)
            window = (ci_hi_d - ci_lo_d).days

            rows.append(
                {
                    "Party": party,
                    "Break Date": bp_date,
                    "95% CI Lower": ci_lo,
                    "95% CI Upper": ci_hi,
                    "Window (days)": window,
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Bai-Perron Structural Breaks with Confidence Intervals",
        subtitle="Formal 95% CIs on break date locations (Bai & Perron 2003)",
        source_note=(
            "Narrow windows = precisely dated breaks. Wide windows = uncertainty in timing. "
            "Bai-Perron uses regression-based inference (F-tests), complementing PELT's "
            "penalty-based approach."
        ),
    )
    report.add(
        TableSection(
            id=f"bp-ci-{chamber.lower()}",
            title=f"{chamber} Bai-Perron Confidence Intervals",
            html=html,
        )
    )


def _add_bai_perron_figures(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Add Bai-Perron CI figures for a chamber."""
    ch = chamber.lower()

    for party in ["republican", "democrat"]:
        path = plots_dir / f"bai_perron_{party}_{ch}.png"
        if path.exists():
            party_label = party.title()
            report.add(
                FigureSection.from_file(
                    f"fig-bp-{party}-{ch}",
                    f"{chamber} {party_label} Bai-Perron Breaks with 95% CIs",
                    path,
                    caption=(
                        f"Weekly Rice Index for {party_label}s with Bai-Perron structural "
                        f"breaks (red dashed lines) and 95% confidence intervals (shaded bands)."
                    ),
                    alt_text=(
                        f"Time series of weekly Rice Index for {chamber} "
                        f"{party_label}s with Bai-Perron structural break lines "
                        "and shaded 95% confidence interval bands."
                    ),
                )
            )


def _add_pelt_bp_crossref(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Compare PELT break dates with Bai-Perron CIs — confirmed vs unconfirmed."""
    import datetime

    cp_results = result.get("changepoints", {})
    if not cp_results:
        return

    rows = []
    for party in ["Republican", "Democrat"]:
        # PELT break dates
        pelt_key = party
        pelt_data = cp_results.get(pelt_key, {})
        pelt_dates = pelt_data.get("dates", [])

        # Bai-Perron breaks
        bp_key = f"{party}_bai_perron"
        bp_data = cp_results.get(bp_key, {})
        bp_df = bp_data.get("bp_df")
        if not pelt_dates or bp_df is None or bp_df.height == 0:
            continue

        for pelt_date in pelt_dates:
            pelt_d = (
                datetime.date.fromisoformat(str(pelt_date)[:10])
                if not isinstance(pelt_date, datetime.date)
                else pelt_date
            )
            confirmed = False
            bp_match = "—"
            for bp_row in bp_df.iter_rows(named=True):
                ci_lo = datetime.date.fromisoformat(str(bp_row["ci_lower_date"])[:10])
                ci_hi = datetime.date.fromisoformat(str(bp_row["ci_upper_date"])[:10])
                if ci_lo <= pelt_d <= ci_hi:
                    confirmed = True
                    bp_match = str(bp_row["break_date"])[:10]
                    break
            rows.append(
                {
                    "Party": party,
                    "PELT Date": str(pelt_d),
                    "Confirmed by BP": "Yes" if confirmed else "No",
                    "BP Match": bp_match,
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — PELT vs Bai-Perron Cross-Reference",
        subtitle="PELT breaks confirmed when they fall within a Bai-Perron 95% CI",
        source_note=(
            "Confirmed breaks are supported by two independent methods "
            "(PELT penalized likelihood + Bai-Perron F-tests)."
        ),
    )
    report.add(
        TableSection(
            id=f"pelt-bp-crossref-{chamber.lower()}",
            title=f"{chamber} Break Confirmation",
            html=html,
        )
    )


def _add_r_enrichment_interpretation(report: ReportBuilder) -> None:
    """Text block: interpreting CROPS and Bai-Perron results."""
    report.add(
        TextSection(
            id="r-enrichment-interpretation",
            title="Interpreting CROPS and Bai-Perron Results",
            html=(
                "<p><strong>CROPS</strong> (Changepoints for a Range of Penalties, Haynes et al. "
                "2017) finds the <em>exact</em> penalty thresholds where the optimal segmentation "
                "changes. Unlike a manual penalty sweep, CROPS guarantees that no intermediate "
                "penalties produce different results. The solution path shows a step function — "
                "each step is one fewer changepoint. The <strong>elbow</strong> is the penalty "
                "where adding more changepoints yields the steepest diminishing returns.</p>"
                "<p><strong>Bai-Perron</strong> (1998, 2003) complements PELT by providing formal "
                "95% confidence intervals on break date locations. PELT gives point estimates; "
                "Bai-Perron gives intervals. A narrow confidence interval means the break is "
                "precisely dated — the data strongly pinpoints when cohesion changed. A wide "
                "interval means the transition was gradual or the evidence is weaker.</p>"
                "<p><strong>PELT/BP cross-reference:</strong> When a PELT break falls within a "
                "Bai-Perron confidence interval, the break is <em>confirmed</em> by two "
                "independent methods using different statistical frameworks (penalized likelihood "
                "vs regression F-tests). Confirmed breaks are the most robust findings.</p>"
            ),
        )
    )


def _add_veto_crossref_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Veto override cross-references across all chambers."""
    all_rows = []
    for chamber, result in results.items():
        veto_xref = result.get("veto_crossref")
        if veto_xref is not None and veto_xref.height > 0:
            for row in veto_xref.iter_rows(named=True):
                all_rows.append(
                    {
                        "Chamber": chamber,
                        "Changepoint Date": row.get("changepoint_date", ""),
                        "Override Bill": row.get("nearby_override_bill", ""),
                        "Override Date": row.get("nearby_override_date", ""),
                        "Days Apart": row.get("days_apart", 0),
                    }
                )

    if not all_rows:
        return

    df = pl.DataFrame(all_rows)
    html = make_gt(
        df,
        title="Changepoint–Veto Override Cross-Reference",
        subtitle="Changepoints within 14 days of a veto override vote",
        source_note=(
            "Proximity suggests the override may have disrupted normal cohesion patterns. "
            "Correlation is not causation — other events may coincide."
        ),
    )
    report.add(
        TableSection(
            id="veto-crossref",
            title="Veto Override Cross-Reference",
            html=html,
        )
    )


def _generate_tsa_key_findings(results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from TSA results."""
    findings: list[str] = []

    for chamber, result in results.items():
        # Changepoints
        cp_summary = result.get("changepoint_summary")
        if cp_summary is not None and hasattr(cp_summary, "height"):
            n_cp = cp_summary.height
            findings.append(
                f"{chamber}: <strong>{n_cp}</strong> changepoint"
                f"{'s' if n_cp != 1 else ''} detected in party cohesion."
            )

        # Drift trend
        drift_summary = result.get("drift_summary", {})
        overall_trend = drift_summary.get("overall_trend")
        if overall_trend is not None:
            direction = "increasing" if overall_trend > 0 else "decreasing"
            findings.append(
                f"{chamber} ideological drift: <strong>{direction}</strong> "
                f"trend (slope = {overall_trend:.4f})."
            )

        # Top mover
        top_movers = result.get("top_movers")
        if top_movers is not None and hasattr(top_movers, "height") and top_movers.height > 0:
            top = top_movers.head(1)
            name_col = "full_name" if "full_name" in top.columns else "legislator_slug"
            name = top[name_col][0]
            drift_col = "total_drift" if "total_drift" in top.columns else "drift"
            drift = float(top[drift_col][0])
            findings.append(
                f"{chamber} most volatile: <strong>{name}</strong> (drift = {drift:.3f})."
            )

        break  # First chamber only

    return findings


def _add_analysis_parameters(
    report: ReportBuilder,
    penalty: float,
    results: dict[str, dict] | None = None,
    r_available: bool = False,
) -> None:
    """Table: All constants and settings used in this run."""
    params = [
        "Window Size",
        "Step Size",
        "Min Window Votes",
        "Min Window Legislators",
        "PELT Penalty",
        "PELT Min Size",
        "Weekly Aggregation",
        "Top Movers N",
        "Min Total Votes",
        "Desposato Correction",
    ]
    values = [
        f"{WINDOW_SIZE} roll calls",
        f"{STEP_SIZE} roll calls (75% overlap)",
        str(MIN_WINDOW_VOTES),
        str(MIN_WINDOW_LEGISLATORS),
        str(penalty),
        str(PELT_MIN_SIZE),
        f"{WEEKLY_AGG_DAYS} days",
        str(TOP_MOVERS_N),
        str(MIN_TOTAL_VOTES),
        "Enabled (Desposato 2005)",
    ]
    descs = [
        "Number of consecutive roll calls per PCA window",
        "Offset between consecutive windows",
        "Minimum votes per legislator to include in a window",
        "Minimum cross-section size for a valid window",
        "PELT penalty: higher = fewer changepoints",
        "Minimum segment size between changepoints",
        "Rice Index aggregation window",
        "Number of biggest drifters highlighted",
        "Session-wide minimum votes for inclusion",
        "Small-group Rice bias correction via Monte Carlo simulation",
    ]

    # Add imputation sensitivity if available
    if results:
        imp_corrs = []
        for result in results.values():
            ic = result.get("imputation_correlation")
            if ic is not None:
                imp_corrs.append(ic)
        if imp_corrs:
            mean_corr = sum(imp_corrs) / len(imp_corrs)
            params.append("Imputation Sensitivity")
            values.append(f"r = {mean_corr:.3f}")
            descs.append("Correlation between column-mean and listwise deletion drift scores")

    # Add R enrichment parameters
    if r_available:
        params.extend(["CROPS Penalty Range", "Bai-Perron Max Breaks", "R Enrichment"])
        values.extend(["[1.0, 50.0]", "5", "Enabled (changepoint + strucchange)"])
        descs.extend(
            [
                "CROPS penalty search range for exact solution path",
                "Maximum number of Bai-Perron structural breaks",
                "R subprocess for CROPS penalty selection and Bai-Perron CIs",
            ]
        )

    df = pl.DataFrame({"Parameter": params, "Value": values, "Description": descs})
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/tsa.md for justification.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
