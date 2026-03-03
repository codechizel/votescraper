"""IRT-specific HTML report builder.

Builds ~16 sections (tables + figures) for the Bayesian IRT report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from irt.py):
    from analysis.irt_report import build_irt_report
    build_irt_report(ctx.report, results=results, ...)
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

TOP_DISCRIMINATING = 15  # Number of top/bottom discriminating votes to show


def build_irt_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    pca_comparisons: dict[str, dict],
    ppc_results: dict[str, dict],
    validation_results: dict[str, dict],
    sensitivity_findings: dict,
    plots_dir: Path,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> None:
    """Build the full IRT HTML report by adding sections to the ReportBuilder."""
    # Key findings
    findings = _generate_irt_key_findings(results, pca_comparisons)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    # Model config + convergence
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        _add_model_summary(report, result, chamber, n_samples, n_tune, n_chains)
        _add_convergence_table(report, result, chamber)
    _add_convergence_interpretation(report)

    # Ideal points
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        _add_forest_figure(report, plots_dir, chamber)
        _add_party_density_figure(report, plots_dir, chamber)
        _add_paradox_spotlight_figure(report, plots_dir, chamber, result)
        _add_ideal_point_table(report, result, chamber)
        _add_ideal_points_interactive(report, plots_dir, chamber)
    _add_ideal_point_interpretation(report)

    # Discrimination
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        _add_discrimination_figure(report, plots_dir, chamber)
        _add_icc_curves_figure(report, plots_dir, chamber)
        _add_top_discriminating_votes(report, result, chamber)
    _add_discrimination_interpretation(report)

    # Cutting lines + swing votes
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        _add_cutting_lines_figure(report, plots_dir, chamber)
        _add_swing_vote_table(report, result, chamber)

    # Traces
    for chamber in results:
        if chamber == "Joint":
            continue
        _add_trace_figure(report, plots_dir, chamber)
    _add_trace_interpretation(report)

    # PPC
    for chamber in results:
        if chamber == "Joint":
            continue
        _add_ppc_figure(report, plots_dir, chamber)
    _add_ppc_interpretation(report)

    # Cross-chamber sections
    if pca_comparisons:
        for chamber in results:
            if chamber == "Joint":
                continue
            _add_pca_comparison_figure(report, plots_dir, chamber)
            _add_irt_vs_pca_interactive(report, plots_dir, chamber)
        _add_pca_comparison_table(report, pca_comparisons)

    if validation_results:
        _add_validation_table(report, validation_results)

    if ppc_results:
        _add_ppc_summary_table(report, ppc_results)

    if sensitivity_findings:
        _add_sensitivity_table(report, sensitivity_findings)
        for chamber in results:
            if chamber == "Joint":
                continue
            _add_sensitivity_figure(report, plots_dir, chamber)
        _add_sensitivity_interpretation(report, sensitivity_findings)

    # Joint model sections (test equating, not MCMC)
    if "Joint" in results:
        _add_joint_model_interpretation(report)
        _add_equating_summary(report, results["Joint"])
        _add_forest_figure(report, plots_dir, "Joint")
        _add_ideal_point_table(report, results["Joint"], "Joint")
        _add_joint_comparison_figure(report, plots_dir)
        _add_joint_comparison_table(report, results)

    _add_analysis_parameters(report, n_samples, n_tune, n_chains)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_model_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> None:
    """Table: Model summary (dimensions, priors, anchors, sampling time)."""
    data = result["data"]

    # Anchor rows differ: per-chamber has 2 (cons + lib), joint has 4
    if "anchor_slugs" in result:
        anchor_labels = [f"Anchor {i + 1}" for i in range(len(result["anchor_slugs"]))]
        anchor_values = result["anchor_slugs"]
    else:
        anchor_labels = ["Conservative Anchor", "Liberal Anchor"]
        anchor_values = [result["cons_slug"], result["lib_slug"]]

    df = pl.DataFrame(
        {
            "Property": [
                "Model",
                "Legislators",
                "Roll Calls",
                "Observed Cells",
                "Observation Rate",
                "Prior (xi)",
                "Prior (alpha)",
                "Prior (beta)",
                *anchor_labels,
                "MCMC Draws",
                "Tuning Steps",
                "Chains",
                "Sampling Time",
            ],
            "Value": [
                "2PL IRT (Bayesian)",
                str(data["n_legislators"]),
                str(data["n_votes"]),
                f"{data['n_obs']:,}",
                f"{100 * data['n_obs'] / (data['n_legislators'] * data['n_votes']):.1f}%",
                "Normal(0, 1) + anchors",
                "Normal(0, 5)",
                "Normal(0, 1)",
                *anchor_values,
                str(n_samples),
                str(n_tune),
                str(n_chains),
                f"{result['sampling_time']:.1f}s",
            ],
        }
    )

    html = make_gt(
        df,
        title=f"{chamber} — IRT Model Summary",
    )
    report.add(
        TableSection(
            id=f"model-summary-{chamber.lower()}",
            title=f"{chamber} Model Summary",
            html=html,
        )
    )


def _add_convergence_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Convergence diagnostics (R-hat, ESS, divergences, E-BFMI)."""
    diag = result["diagnostics"]
    rows = [
        {
            "Metric": "R-hat (xi) max",
            "Value": f"{diag['xi_rhat_max']:.4f}",
            "Threshold": "< 1.01",
            "Status": "OK" if diag["xi_rhat_max"] < 1.01 else "WARNING",
        },
        {
            "Metric": "R-hat (alpha) max",
            "Value": f"{diag['alpha_rhat_max']:.4f}",
            "Threshold": "< 1.01",
            "Status": "OK" if diag["alpha_rhat_max"] < 1.01 else "WARNING",
        },
        {
            "Metric": "R-hat (beta) max",
            "Value": f"{diag['beta_rhat_max']:.4f}",
            "Threshold": "< 1.01",
            "Status": "OK" if diag["beta_rhat_max"] < 1.01 else "WARNING",
        },
        {
            "Metric": "ESS (xi) min",
            "Value": f"{diag['xi_ess_min']:.0f}",
            "Threshold": "> 400",
            "Status": "OK" if diag["xi_ess_min"] > 400 else "WARNING",
        },
        {
            "Metric": "ESS (alpha) min",
            "Value": f"{diag['alpha_ess_min']:.0f}",
            "Threshold": "> 400",
            "Status": "OK" if diag["alpha_ess_min"] > 400 else "WARNING",
        },
        {
            "Metric": "ESS (beta) min",
            "Value": f"{diag['beta_ess_min']:.0f}",
            "Threshold": "> 400",
            "Status": "OK" if diag["beta_ess_min"] > 400 else "WARNING",
        },
        {
            "Metric": "Divergences",
            "Value": str(diag["divergences"]),
            "Threshold": "< 10",
            "Status": "OK" if diag["divergences"] < 10 else "WARNING",
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
        title=f"{chamber} — Convergence Diagnostics",
        source_note="All metrics must pass for reliable inference.",
    )
    report.add(
        TableSection(
            id=f"convergence-{chamber.lower()}",
            title=f"{chamber} Convergence Diagnostics",
            html=html,
        )
    )


def _add_forest_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"forest_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-forest-{chamber.lower()}",
                f"{chamber} Ideal Points (Forest Plot)",
                path,
                caption=(
                    f"Legislator ideal points with 95% HDI ({chamber}). "
                    "Red = Republican, Blue = Democrat. Positive = conservative. "
                    "Flagged legislators detected automatically based on "
                    "statistical outlier criteria."
                ),
                alt_text=(
                    f"Forest plot of Bayesian IRT ideal points with 95% credible intervals "
                    f"for {chamber} legislators. Republicans cluster on the positive "
                    f"(conservative) side, Democrats on the negative (liberal) side."
                ),
            )
        )


def _add_party_density_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Figure: Party ideal point density overlay."""
    path = plots_dir / f"party_density_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-party-density-{chamber.lower()}",
                f"{chamber} Party Ideal Point Distributions",
                path,
                caption=(
                    f"Overlapping KDE density curves of ideal points by party ({chamber}). "
                    "Dashed lines = party means. Overlap region indicates cross-pressured "
                    "legislators whose positions cannot be distinguished by party alone."
                ),
                alt_text=(
                    f"Density plot of IRT ideal points by party for the {chamber}. "
                    f"Republican and Democrat distributions are shown as overlapping curves "
                    f"with dashed lines marking party means."
                ),
            )
        )


def _add_icc_curves_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Figure: Item Characteristic Curves for top discriminating bills."""
    path = plots_dir / f"icc_curves_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-icc-{chamber.lower()}",
                f"{chamber} Item Characteristic Curves",
                path,
                caption=(
                    f"P(Yea | theta) for the most discriminating bills ({chamber}). "
                    "Red = conservative-Yea (beta > 0), Blue = liberal-Yea (beta < 0). "
                    "Steeper curves = stronger party separation. The crossing point "
                    "(P = 0.5) is the bill's cutting point on the ideology scale."
                ),
                alt_text=(
                    f"Item characteristic curves showing probability of Yea vote as a "
                    f"function of ideal point for the most discriminating {chamber} bills. "
                    f"Steeper sigmoid curves indicate stronger ideological separation."
                ),
            )
        )


def _add_paradox_spotlight_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
    result: dict,
) -> None:
    """Figure: Paradox spotlight (side-by-side voting pattern + forest position)."""
    paradox = result.get("paradox")
    if paradox is None:
        return

    path = plots_dir / f"paradox_spotlight_{chamber.lower()}.png"
    if not path.exists():
        return

    name = paradox["full_name"]
    direction = "conservative" if paradox["xi_mean"] > 0 else "liberal"
    low_pct = f"{paradox['low_disc_yea_rate']:.0%}"
    party_low_pct = f"{paradox['party_low_disc_yea_rate']:.0%}"

    report.add(
        FigureSection.from_file(
            f"fig-paradox-{chamber.lower()}",
            f"{chamber} Paradox Spotlight: {name}",
            path,
            caption=(
                f"{name} is the most {direction} legislator by IRT in the {chamber}, "
                f"yet votes Yea on only {low_pct} of routine bills "
                f"(vs. the party average of {party_low_pct}). "
                f"({paradox['n_high_disc']} partisan bills, "
                f"{paradox['n_low_disc']} routine bills.)"
            ),
            alt_text=(
                f"Scatter plot spotlighting {name} in the {chamber}, showing "
                f"the paradox between extreme IRT ideal point and low Yea rate "
                f"on routine bills compared to party average."
            ),
        )
    )


def _add_ideal_point_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators ranked by ideal point with HDI."""
    ip = result["ideal_points"].sort("xi_mean", descending=True)

    display_cols = [
        "full_name",
        "party",
        "district",
        "xi_mean",
        "xi_sd",
        "xi_hdi_2.5",
        "xi_hdi_97.5",
    ]
    available = [c for c in display_cols if c in ip.columns]
    df = ip.select(available)

    html = make_interactive_table(
        df,
        title=(
            f"{chamber} — Legislator Ideal Points "
            f"({df.height} legislators, positive = conservative)"
        ),
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "xi_mean": "Ideal Point",
            "xi_sd": "Std Dev",
            "xi_hdi_2.5": "HDI 2.5%",
            "xi_hdi_97.5": "HDI 97.5%",
        },
        number_formats={
            "xi_mean": ".3f",
            "xi_sd": ".3f",
            "xi_hdi_2.5": ".3f",
            "xi_hdi_97.5": ".3f",
        },
        caption="HDI = 95% Highest Density Interval.",
    )
    report.add(
        InteractiveTableSection(
            id=f"ideal-points-{chamber.lower()}",
            title=f"{chamber} Ideal Points",
            html=html,
        )
    )


def _add_discrimination_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"discrimination_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-discrim-{chamber.lower()}",
                f"{chamber} Discrimination Distribution",
                path,
                caption=(
                    f"Distribution of roll call discrimination parameters ({chamber}). "
                    "Positive beta = conservatives favor Yea. "
                    "Negative beta = liberals favor Yea. "
                    "Higher |beta| = more ideologically discriminating."
                ),
                alt_text=(
                    f"Histogram of bill discrimination parameters (beta) for the {chamber}. "
                    f"Distribution centered near zero with tails indicating highly "
                    f"partisan bills in both conservative and liberal directions."
                ),
            )
        )


def _add_top_discriminating_votes(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Top 15 most discriminating (by |beta|) in each direction."""
    bp = result["bill_params"].with_columns(pl.col("beta_mean").abs().alias("abs_beta"))

    # Top conservative-Yea bills (highest positive beta)
    top_pos = (
        bp.filter(pl.col("beta_mean") > 0)
        .sort("abs_beta", descending=True)
        .head(TOP_DISCRIMINATING)
    )
    # Top liberal-Yea bills (most negative beta, i.e. highest |beta| where beta < 0)
    top_neg = (
        bp.filter(pl.col("beta_mean") < 0)
        .sort("abs_beta", descending=True)
        .head(TOP_DISCRIMINATING)
    )
    combined = pl.concat([top_pos, top_neg]).drop("abs_beta")

    display_cols = ["beta_mean", "beta_sd", "alpha_mean", "vote_id", "bill_number"]
    for opt_col in ["short_title", "motion", "is_veto_override"]:
        if opt_col in combined.columns:
            display_cols.append(opt_col)
    combined = combined.select([c for c in display_cols if c in combined.columns])

    labels: dict[str, str] = {
        "beta_mean": "Discrimination",
        "beta_sd": "Beta SD",
        "alpha_mean": "Difficulty",
        "vote_id": "Vote ID",
        "bill_number": "Bill",
    }
    if "short_title" in combined.columns:
        labels["short_title"] = "Title"
    if "motion" in combined.columns:
        labels["motion"] = "Motion"
    if "is_veto_override" in combined.columns:
        labels["is_veto_override"] = "Veto Override"

    html = make_interactive_table(
        combined,
        title=(
            f"{chamber} — Most Discriminating Roll Calls: "
            f"top {TOP_DISCRIMINATING} conservative-Yea (β > 0) and "
            f"{TOP_DISCRIMINATING} liberal-Yea (β < 0)"
        ),
        column_labels=labels,
        number_formats={"beta_mean": ".3f", "beta_sd": ".3f", "alpha_mean": ".3f"},
    )
    report.add(
        InteractiveTableSection(
            id=f"discrim-votes-{chamber.lower()}",
            title=f"{chamber} Top Discriminating Votes",
            html=html,
        )
    )


def _add_trace_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"trace_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-trace-{chamber.lower()}",
                f"{chamber} Trace Plots",
                path,
                caption=(
                    f"Trace plots for selected ideal points ({chamber}). "
                    "Good mixing = fuzzy caterpillars with no trends."
                ),
                alt_text=(
                    f"MCMC trace plots for selected {chamber} ideal point parameters. "
                    f"Multiple chains shown as overlapping time series to assess convergence."
                ),
            )
        )


def _add_ppc_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"ppc_yea_rate_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-ppc-{chamber.lower()}",
                f"{chamber} PPC: Yea Rate",
                path,
                caption=(
                    f"Posterior predictive check ({chamber}): replicated vs observed Yea rate. "
                    "Red line = observed. Bayesian p-value in [0.1, 0.9] = well-calibrated."
                ),
                alt_text=(
                    f"Histogram of replicated Yea rates from posterior predictive simulation "
                    f"for the {chamber}. Vertical red line marks the observed Yea rate "
                    f"for model calibration assessment."
                ),
            )
        )


def _add_pca_comparison_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"irt_vs_pca_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-pca-{chamber.lower()}",
                f"{chamber} IRT vs PCA Comparison",
                path,
                caption=(
                    f"IRT ideal points vs PCA PC1 scores ({chamber}). "
                    "High correlation (r > 0.95) confirms consistent ideological estimates."
                ),
                alt_text=(
                    f"Scatter plot comparing IRT ideal points to PCA first principal component "
                    f"scores for {chamber} legislators. Points fall along a strong linear "
                    f"trend, confirming consistency between the two methods."
                ),
            )
        )


def _add_ideal_points_interactive(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Embed Plotly interactive ideal point scatter."""
    path = plots_dir / f"ideal_points_interactive_{chamber.lower()}.html"
    if not path.exists():
        return
    html = path.read_text()
    report.add(
        InteractiveSection(
            id=f"interactive-ideal-points-{chamber.lower()}",
            title=f"{chamber} Ideal Points (Interactive)",
            html=html,
            caption="Hover over points to see legislator details, ideal point, and HDI.",
            aria_label=(
                f"Interactive scatter plot of {chamber} legislator ideal points. "
                f"Hover over points for legislator name, party, ideal point, and credible interval."
            ),
        )
    )


def _add_irt_vs_pca_interactive(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Embed Plotly interactive IRT vs PCA scatter."""
    path = plots_dir / f"irt_vs_pca_interactive_{chamber.lower()}.html"
    if not path.exists():
        return
    html = path.read_text()
    report.add(
        InteractiveSection(
            id=f"interactive-irt-pca-{chamber.lower()}",
            title=f"{chamber} IRT vs PCA (Interactive)",
            html=html,
            caption="Hover over points to see both IRT and PCA scores.",
            aria_label=(
                f"Interactive scatter plot comparing IRT ideal points to PCA scores "
                f"for {chamber} legislators. Hover for both scores per legislator."
            ),
        )
    )


def _add_pca_comparison_table(
    report: ReportBuilder,
    comparisons: dict[str, dict],
) -> None:
    """Table: PCA comparison metrics (Pearson r, Spearman rho)."""
    rows = []
    for chamber, data in comparisons.items():
        rows.append(
            {
                "chamber": chamber,
                "n_shared": data["n_shared"],
                "pearson_r": data["pearson_r"],
                "spearman_rho": data["spearman_rho"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="IRT vs PCA Comparison",
        subtitle="Correlation between IRT ideal points and PCA PC1 scores",
        column_labels={
            "chamber": "Chamber",
            "n_shared": "Shared Legislators",
            "pearson_r": "Pearson r",
            "spearman_rho": "Spearman rho",
        },
        number_formats={"pearson_r": ".4f", "spearman_rho": ".4f"},
        source_note="r > 0.95 expected for a well-behaved 1D model.",
    )
    report.add(
        TableSection(
            id="pca-comparison",
            title="IRT vs PCA Comparison",
            html=html,
        )
    )


def _add_validation_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Holdout validation metrics (in-sample prediction)."""
    rows = []
    for chamber, data in results.items():
        rows.append(
            {
                "chamber": chamber,
                "holdout_cells": data["holdout_cells"],
                "base_rate": data["base_rate"],
                "base_accuracy": data["base_accuracy"],
                "accuracy": data["accuracy"],
                "auc_roc": data["auc_roc"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Holdout Validation Metrics",
        subtitle="In-sample prediction on random 20% of observed cells (seed=42)",
        column_labels={
            "chamber": "Chamber",
            "holdout_cells": "Holdout Cells",
            "base_rate": "Base Rate (Yea)",
            "base_accuracy": "Base Accuracy",
            "accuracy": "IRT Accuracy",
            "auc_roc": "AUC-ROC",
        },
        number_formats={
            "holdout_cells": ",.0f",
            "base_rate": ".3f",
            "base_accuracy": ".3f",
            "accuracy": ".3f",
            "auc_roc": ".3f",
        },
        source_note=(
            "In-sample prediction (model saw all data). "
            "PPC provides the proper Bayesian validation."
        ),
    )
    report.add(
        TableSection(
            id="validation",
            title="Holdout Validation",
            html=html,
        )
    )


def _add_ppc_summary_table(
    report: ReportBuilder,
    ppc_results: dict[str, dict],
) -> None:
    """Table: PPC summary (test statistics, Bayesian p-values)."""
    rows = []
    for chamber, data in ppc_results.items():
        rows.append(
            {
                "chamber": chamber,
                "observed_yea_rate": data["observed_yea_rate"],
                "replicated_mean": data["replicated_yea_rate_mean"],
                "replicated_sd": data["replicated_yea_rate_sd"],
                "bayesian_p": data["bayesian_p_yea_rate"],
                "mean_rep_accuracy": data["mean_replicated_accuracy"],
                "n_replications": data["n_replications"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Posterior Predictive Check Summary",
        subtitle="Bayesian p-values should be in [0.1, 0.9] for well-calibrated model",
        column_labels={
            "chamber": "Chamber",
            "observed_yea_rate": "Obs. Yea Rate",
            "replicated_mean": "Rep. Yea Rate (mean)",
            "replicated_sd": "Rep. Yea Rate (SD)",
            "bayesian_p": "Bayesian p-value",
            "mean_rep_accuracy": "Mean Rep. Accuracy",
            "n_replications": "N Replications",
        },
        number_formats={
            "observed_yea_rate": ".3f",
            "replicated_mean": ".3f",
            "replicated_sd": ".4f",
            "bayesian_p": ".3f",
            "mean_rep_accuracy": ".3f",
        },
        source_note="p-values outside [0.1, 0.9] indicate potential model misfit.",
    )
    report.add(
        TableSection(
            id="ppc-summary",
            title="Posterior Predictive Checks",
            html=html,
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
        r = data["pearson_r"]
        status = "ROBUST" if abs(r) > 0.95 else "SENSITIVE"
        rows.append(
            {
                "chamber": chamber,
                "status": status,
                "default_threshold": f"{data['default_threshold'] * 100:.1f}%",
                "sensitivity_threshold": f"{data['sensitivity_threshold'] * 100:.0f}%",
                "default_n_legislators": data["default_n_legislators"],
                "sensitivity_n_legislators": data["sensitivity_n_legislators"],
                "default_n_votes": data["default_n_votes"],
                "sensitivity_n_votes": data["sensitivity_n_votes"],
                "shared_legislators": data["shared_legislators"],
                "pearson_r": r,
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="IRT Sensitivity Analysis — Ideal Point Correlation",
        subtitle="Comparing default (2.5%) vs. aggressive (10%) minority thresholds",
        column_labels={
            "chamber": "Chamber",
            "status": "Status",
            "default_threshold": "Default",
            "sensitivity_threshold": "Sensitivity",
            "default_n_legislators": "N Leg. (Default)",
            "sensitivity_n_legislators": "N Leg. (Sens.)",
            "default_n_votes": "N Votes (Default)",
            "sensitivity_n_votes": "N Votes (Sens.)",
            "shared_legislators": "Shared Leg.",
            "pearson_r": "Pearson r",
        },
        number_formats={"pearson_r": ".4f"},
        source_note="ROBUST: |r| > 0.95 — stable across threshold choices. SENSITIVE: |r| ≤ 0.95.",
    )
    report.add(
        TableSection(
            id="sensitivity",
            title="Sensitivity Analysis",
            html=html,
        )
    )


def _add_sensitivity_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"sensitivity_xi_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-sensitivity-{chamber.lower()}",
                f"{chamber} Sensitivity Scatter",
                path,
                caption=(
                    f"Default vs. sensitivity ideal points ({chamber}). "
                    "Points near the identity line indicate stable results."
                ),
                alt_text=(
                    f"Scatter plot of default vs sensitivity-analysis ideal points "
                    f"for the {chamber}. Points near the diagonal identity line "
                    f"indicate robust estimates under alternative filter settings."
                ),
            )
        )


def _add_sensitivity_interpretation(report: ReportBuilder, findings: dict) -> None:
    """Text block: Interpret sensitivity results with ROBUST/SENSITIVE classification."""
    parts = []
    for chamber, data in findings.items():
        if not isinstance(data, dict) or data.get("skipped"):
            continue
        r = data.get("pearson_r")
        if r is None:
            continue
        raw_r = data.get("raw_pearson_r", r)
        sign_flipped = raw_r < 0 if raw_r is not None else False
        status = "ROBUST" if abs(r) > 0.95 else "SENSITIVE"
        style = (
            "color: green; font-weight: bold"
            if status == "ROBUST"
            else "color: orange; font-weight: bold"
        )

        part = (
            f'<li><strong>{chamber}:</strong> <span style="{style}">{status}</span> '
            f"(|r| = {abs(r):.4f})"
        )
        if sign_flipped:
            part += (
                " — <em>sign flip detected</em> (raw r = "
                f"{raw_r:.4f}). The aggressive threshold reversed the ideological "
                "scale, which is expected when near-unanimous votes carry intra-party signal."
            )
        part += "</li>"
        parts.append(part)

    if not parts:
        return

    report.add(
        TextSection(
            id="sensitivity-interpretation",
            title="Interpreting Sensitivity Results",
            html=(
                "<ul>" + "".join(parts) + "</ul>"
                "<p><strong>What this means:</strong> Sensitivity analysis compares ideal "
                "points estimated with two different vote-filtering thresholds — the default "
                "(2.5% minority) and an aggressive cutoff (10%). ROBUST (r &gt; 0.95) means "
                "ideal point estimates are stable regardless of which near-unanimous votes "
                "are included.</p>"
                "<p><strong>Why chambers differ:</strong> The Kansas House has a Republican "
                "supermajority (~72%), so many votes are near-unanimous. Including or "
                "excluding these votes changes the information available to the model, "
                "potentially making House estimates more sensitive than Senate estimates.</p>"
                "<p><strong>SENSITIVE does not mean incorrect.</strong> It means the "
                "ideological scale contains meaningful intra-party signal from near-unanimous "
                "votes. This is a well-known property of IRT in supermajority settings "
                "(Clinton, Jackman &amp; Rivers 2004). Both W-NOMINATE and DW-NOMINATE "
                "handle this via vote pre-selection thresholds.</p>"
            ),
        )
    )


def _add_convergence_interpretation(report: ReportBuilder) -> None:
    """Text block: How to interpret convergence diagnostics."""
    report.add(
        TextSection(
            id="convergence-interpretation",
            title="Interpreting Convergence Diagnostics",
            html=(
                "<p><strong>R-hat</strong> (potential scale reduction factor) compares "
                "between-chain and within-chain variance. Values &lt; 1.01 indicate that "
                "chains have converged to the same distribution. R-hat &gt; 1.01 means "
                "chains disagree — increase tuning steps or check for multimodality.</p>"
                "<p><strong>ESS</strong> (effective sample size) measures the number of "
                "independent samples after accounting for autocorrelation. ESS &gt; 400 "
                "is needed for reliable posterior summaries and HDIs. Low ESS means the "
                "sampler is moving slowly through parameter space — increase draws or "
                "reparameterize the model.</p>"
                "<p><strong>Divergences</strong> are NUTS transitions where the numerical "
                "integrator diverged from the true trajectory. A few (&lt; 10) are "
                "tolerable; many indicate the posterior has sharp features the sampler "
                "cannot navigate. Increase target_accept toward 0.99, or simplify "
                "the model.</p>"
                "<p><strong>E-BFMI</strong> (energy Bayesian fraction of missing "
                "information) measures how well the sampler explores the posterior's "
                "energy distribution. Values &gt; 0.3 are acceptable. Low E-BFMI "
                "suggests the posterior has a funnel-shaped geometry — consider "
                "reparameterization (e.g., non-centered parameterization).</p>"
            ),
        )
    )


def _add_ideal_point_interpretation(report: ReportBuilder) -> None:
    """Text block: How to interpret ideal points."""
    report.add(
        TextSection(
            id="ideal-point-interpretation",
            title="Interpreting Ideal Points",
            html=(
                "<p>Each legislator's <strong>ideal point (xi)</strong> represents their "
                "estimated position on a latent ideological spectrum. The scale is "
                "anchored by two legislators fixed at +1 (conservative) and -1 (liberal), "
                "so positive values indicate conservative positions and negative values "
                "indicate liberal positions.</p>"
                "<p>The <strong>95% HDI</strong> (highest density interval) is the "
                "narrowest interval containing 95% of the posterior probability. "
                "If two legislators' HDIs overlap, their positions are statistically "
                "indistinguishable — you cannot reliably rank them.</p>"
                "<p><strong>Wide intervals</strong> indicate uncertainty, typically "
                "from few votes or inconsistent voting patterns. Legislators with "
                "wide HDIs have low-confidence ideal point estimates.</p>"
                "<p><strong>Anchored legislators</strong> have their ideal points "
                "fixed (not estimated). They appear as point estimates with no "
                "uncertainty. The model uses them to identify the scale's location, "
                "spread, and direction.</p>"
            ),
        )
    )


def _add_discrimination_interpretation(report: ReportBuilder) -> None:
    """Text block: How to interpret discrimination parameters."""
    report.add(
        TextSection(
            id="discrimination-interpretation",
            title="Interpreting Discrimination Parameters",
            html=(
                "<p>Each roll call has two IRT parameters:</p>"
                "<ul>"
                "<li><strong>Discrimination (beta)</strong>: How sharply the vote "
                "separates legislators along the ideological spectrum. Higher |beta| = "
                "more partisan/ideological. |beta| &gt; 1.5 is highly discriminating; "
                "|beta| &lt; 0.5 is weakly discriminating.</li>"
                "<li><strong>Difficulty (alpha)</strong>: Where on the spectrum the "
                "vote 'flips' from likely Nay to likely Yea. High alpha = harder to "
                "pass (requires more extreme position).</li>"
                "</ul>"
                "<p><strong>Sign of beta</strong>: Positive beta means conservatives "
                "(high xi) are more likely to vote Yea. Negative beta means liberals "
                "(low xi) are more likely to vote Yea. This is determined by the "
                "anchor legislators — it is not arbitrary.</p>"
                "<p>Bills with beta near zero are <strong>non-informative</strong> — "
                "ideology does not predict the vote. These are typically procedural "
                "or unanimous bills that survived the EDA filtering.</p>"
            ),
        )
    )


def _add_cutting_lines_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Figure: VoteView-style cutting line visualization."""
    path = plots_dir / f"cutting_lines_{chamber.lower()}.png"
    if not path.exists():
        return
    report.add(
        FigureSection.from_file(
            id=f"cutting-lines-{chamber.lower()}",
            title=f"{chamber} Cutting Lines",
            path=path,
            caption=(
                f"Cutting lines for the most discriminating bills ({chamber}). "
                "Each panel shows legislator ideal points along a horizontal axis. "
                "Green triangles = Yea, red triangles = Nay. The dashed vertical "
                "line is the cutting point where P(Yea) = 0.5."
            ),
            alt_text=(
                f"Multi-panel cutting line plot for the most discriminating {chamber} "
                f"bills. Each panel shows legislator positions on the ideology axis with "
                f"Yea and Nay votes marked, and a dashed line at the 50% probability point."
            ),
        )
    )


def _add_swing_vote_table(report: ReportBuilder, result: dict, chamber: str) -> None:
    """Table: swing legislators near cutting points on close votes."""
    swing = result.get("swing_votes")
    if swing is None or swing.is_empty():
        return

    display = swing.head(20).select("full_name", "party", "swing_count", "ideal_point")
    html = make_gt(
        display,
        title=f"{chamber} — Swing Legislators on Close Votes",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "swing_count": "Close Votes Near Cutting Point",
            "ideal_point": "Ideal Point",
        },
        number_formats={"ideal_point": ".3f"},
        source_note=(
            "Swing legislators have ideal points within 0.5 IRT units of the "
            "cutting point on close votes (margin <= 5). Higher count = more "
            "often in a pivotal position."
        ),
    )
    report.add(
        TableSection(
            id=f"swing-votes-{chamber.lower()}",
            title=f"{chamber} Swing Legislators",
            html=html,
        )
    )


def _add_trace_interpretation(report: ReportBuilder) -> None:
    """Text block: How to interpret trace plots."""
    report.add(
        TextSection(
            id="trace-interpretation",
            title="Interpreting Trace Plots",
            html=(
                "<p>Trace plots show the sampled values of selected ideal points "
                "across MCMC iterations. Each color represents a different chain.</p>"
                "<p><strong>Good traces</strong> look like 'fuzzy caterpillars' — "
                "stationary, well-mixed, with chains overlapping. The posterior "
                "density (left panel) should be smooth and unimodal.</p>"
                "<p><strong>Bad signs</strong> to watch for:</p>"
                "<ul>"
                "<li><strong>Trending</strong>: Chains drifting up or down, indicating "
                "the sampler has not converged.</li>"
                "<li><strong>Stuck</strong>: Chains spending long periods at the same "
                "value, indicating poor mixing.</li>"
                "<li><strong>Multimodal</strong>: Chains visiting different regions, "
                "suggesting the posterior has multiple modes.</li>"
                "<li><strong>Chain disagreement</strong>: One chain exploring a different "
                "region than the other — will show up as R-hat &gt; 1.01.</li>"
                "</ul>"
                "<p>If traces look problematic, increase tuning steps, increase "
                "target_accept, or add more chains to diagnose the issue.</p>"
            ),
        )
    )


def _add_ppc_interpretation(report: ReportBuilder) -> None:
    """Text block: How to interpret posterior predictive checks."""
    report.add(
        TextSection(
            id="ppc-interpretation",
            title="Interpreting Posterior Predictive Checks",
            html=(
                "<p>Posterior predictive checks (PPCs) assess whether the fitted model "
                "can generate data that resembles the observed data. The model draws "
                "parameter values from the posterior and simulates new datasets.</p>"
                "<p>The <strong>Bayesian p-value</strong> is the fraction of replicated "
                "datasets where the test statistic (e.g., overall Yea rate) equals or "
                "exceeds the observed value. A well-calibrated model produces p-values "
                "in [0.1, 0.9] — the observed data looks typical of what the model "
                "generates.</p>"
                "<ul>"
                "<li>p near 0.5: Model fits perfectly for this statistic.</li>"
                "<li>p &lt; 0.1 or p &gt; 0.9: The model consistently over- or "
                "under-predicts this statistic — potential misfit.</li>"
                "<li>p = 0.0 or p = 1.0: The model never generates data like the "
                "observed — serious misfit for this statistic.</li>"
                "</ul>"
                "<p>PPCs are more informative than holdout accuracy because they test "
                "the full posterior distribution, not just the posterior mean. The "
                "holdout validation uses posterior means only and is documented as "
                "in-sample prediction (the model saw all data during fitting).</p>"
            ),
        )
    )


def _add_joint_model_interpretation(report: ReportBuilder) -> None:
    """Text block: How to interpret the cross-chamber equating."""
    report.add(
        TextSection(
            id="joint-model-interpretation",
            title="Interpreting Cross-Chamber Equating",
            html=(
                "<p>Per-chamber IRT models estimate ideal points on separate, "
                "incomparable scales &mdash; a House score of +2.0 and a Senate "
                "score of +2.0 do not mean the same thing. <strong>Test equating"
                "</strong> places all legislators on a single common scale using "
                "the House scale as the reference.</p>"
                "<p><strong>How it works (mean/sigma method):</strong> Bills that "
                "passed through both chambers receive IRT discrimination (beta) "
                "and difficulty (alpha) estimates from each per-chamber model "
                "independently. The ratio of discrimination standard deviations "
                "across chambers gives the scale factor <em>A</em>, and the mean "
                "difficulty difference gives the location shift <em>B</em>. Senate "
                "ideal points are then transformed: xi_equated = A &times; "
                "xi_senate + B. House ideal points remain unchanged.</p>"
                "<p><strong>Why not a joint MCMC model?</strong> A full joint "
                "IRT model was attempted but does not converge for this data. "
                "With ~70 shared bills and ~170 legislators (0.42 bills per "
                "legislator in the joint matrix), the posterior is severely "
                "under-identified, producing R-hat &gt; 1.7 despite 4 anchors "
                "and 4 chains. Test equating sidesteps this by using the "
                "already-converged per-chamber estimates.</p>"
                "<p><strong>Limitations:</strong></p>"
                "<ul>"
                "<li>Assumes shared bills have the same ideological content in "
                "both chambers. If a bill is substantively amended between "
                "chambers, the equating is weakened for that bill.</li>"
                "<li>Senate uncertainty (HDI width) is scaled by |A| but the "
                "correlation structure between legislators is not preserved "
                "&mdash; these are transformed marginals, not a joint "
                "posterior.</li>"
                "<li>Per-chamber models remain primary for within-chamber "
                "analyses. Equated scores are for cross-chamber comparison "
                "only.</li>"
                "</ul>"
                "<p><strong>Validation:</strong> Equated ideal points for House "
                "legislators should correlate perfectly (r = 1.0) with their "
                "per-chamber scores (they are unchanged). Senate legislators "
                "should show r &gt; 0.99 (only a linear transformation).</p>"
            ),
        )
    )


def _add_equating_summary(
    report: ReportBuilder,
    result: dict,
) -> None:
    """Table: Test equating transformation summary."""
    eq = result.get("equating", {})
    xform = eq.get("transformation", {})

    df = pl.DataFrame(
        {
            "Property": [
                "Method",
                "Reference Scale",
                "Shared Bills (concordant / total)",
                "Scale Factor (A)",
                "Location Shift (B)",
                "Transformation",
                "Total Legislators",
            ],
            "Value": [
                "Mean/Sigma Test Equating",
                "House",
                f"{xform.get('n_usable_bills', '?')} / {xform.get('n_total_shared', '?')}",
                f"{xform.get('A', 0):.4f}",
                f"{xform.get('B', 0):.4f}",
                f"xi_equated = {xform.get('A', 0):.4f} * xi_senate + {xform.get('B', 0):.4f}",
                str(result["ideal_points"].height),
            ],
        }
    )

    html = make_gt(
        df,
        title="Cross-Chamber Equating Summary",
        subtitle="Senate ideal points transformed to House scale",
    )
    report.add(
        TableSection(
            id="equating-summary",
            title="Equating Summary",
            html=html,
        )
    )


def _add_joint_comparison_figure(report: ReportBuilder, plots_dir: Path) -> None:
    """Figure: Equated vs per-chamber ideal points scatter."""
    path = plots_dir / "joint_vs_chamber.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-joint-vs-chamber",
                "Equated vs Per-Chamber Ideal Points",
                path,
                caption=(
                    "Equated ideal points (x) vs per-chamber ideal points (y). "
                    "House legislators are unchanged (identity line). Senate "
                    "legislators show the linear transformation. "
                    "Bridging legislators are highlighted with diamond markers."
                ),
                alt_text=(
                    "Scatter plot comparing equated cross-chamber ideal points to "
                    "per-chamber ideal points. House legislators follow the identity line; "
                    "Senate legislators show the Stocking-Lord linear transformation."
                ),
            )
        )


def _add_joint_comparison_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Correlation between equated and per-chamber ideal points."""
    rows = []
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        corr = result.get("joint_correlation")
        if corr is not None:
            rows.append(
                {
                    "Chamber": chamber,
                    "Shared Legislators": corr.get("n_shared", 0),
                    "Pearson r": corr.get("pearson_r"),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Equated vs Per-Chamber Correlation",
        subtitle="Pearson r between equated and per-chamber ideal points",
        number_formats={"Pearson r": ".4f"},
        source_note=(
            "House r = 1.0 expected (unchanged). Senate r > 0.99 expected "
            "(linear transformation preserves rank order)."
        ),
    )
    report.add(
        TableSection(
            id="joint-comparison",
            title="Equated vs Per-Chamber Comparison",
            html=html,
        )
    )


def _generate_irt_key_findings(
    results: dict[str, dict],
    pca_comparisons: dict[str, dict],
) -> list[str]:
    """Generate 3-5 key findings from IRT results."""
    findings: list[str] = []

    # Convergence status
    all_converged = True
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        diag = result.get("diagnostics", {})
        if diag.get("xi_rhat_max", 2.0) >= 1.01 or diag.get("xi_ess_min", 0) < 400:
            all_converged = False
            break
    if all_converged:
        findings.append(
            "All convergence diagnostics <strong>passed</strong> (R-hat < 1.01, ESS > 400)."
        )
    else:
        findings.append("<strong>WARNING:</strong> Some convergence diagnostics did not pass.")

    # Party separation per chamber
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        ip = result.get("ideal_points")
        if ip is None or ip.height == 0:
            continue
        r_mean = ip.filter(pl.col("party") == "Republican")["xi_mean"].mean()
        d_mean = ip.filter(pl.col("party") == "Democrat")["xi_mean"].mean()
        if r_mean is not None and d_mean is not None:
            separation = abs(float(r_mean) - float(d_mean))
            findings.append(
                f"{chamber} party separation: <strong>{separation:.2f}</strong> IRT units "
                f"(R mean={float(r_mean):.2f}, D mean={float(d_mean):.2f})."
            )

    # Most extreme legislator
    for chamber, result in results.items():
        if chamber == "Joint":
            continue
        ip = result.get("ideal_points")
        if ip is None or ip.height == 0:
            continue
        most_extreme = ip.sort("xi_mean", descending=True).head(1)
        name = most_extreme["full_name"][0]
        xi = float(most_extreme["xi_mean"][0])
        direction = "conservative" if xi > 0 else "liberal"
        findings.append(
            f"Most extreme {chamber} legislator: <strong>{name}</strong> "
            f"(xi={xi:.2f}, {direction})."
        )
        break  # Only show for first chamber to keep it concise

    # PCA correlation
    for chamber, data in pca_comparisons.items():
        r = data.get("pearson_r", 0)
        findings.append(f"IRT-PCA correlation ({chamber}): <strong>r={r:.3f}</strong>.")
        break  # One is enough

    return findings


def _add_analysis_parameters(
    report: ReportBuilder,
    n_samples: int,
    n_tune: int,
    n_chains: int,
) -> None:
    """Table: Analysis parameters used in this run."""
    try:
        from analysis.irt import (
            HOLDOUT_FRACTION,
            HOLDOUT_SEED,
            MIN_PARTICIPATION_FOR_ANCHOR,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            RANDOM_SEED,
            SENSITIVITY_THRESHOLD,
            TARGET_ACCEPT,
        )
    except ModuleNotFoundError:
        from irt import (  # type: ignore[no-redef]
            HOLDOUT_FRACTION,
            HOLDOUT_SEED,
            MIN_PARTICIPATION_FOR_ANCHOR,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            RANDOM_SEED,
            SENSITIVITY_THRESHOLD,
            TARGET_ACCEPT,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "Model",
                "Prior (xi)",
                "Prior (alpha)",
                "Prior (beta)",
                "MCMC Draws per Chain",
                "Tuning Steps",
                "Chains",
                "Target Accept Rate",
                "Random Seed",
                "Minority Threshold (Default)",
                "Minority Threshold (Sensitivity)",
                "Min Substantive Votes",
                "Min Participation for Anchor",
                "Holdout Fraction",
                "Holdout Random Seed",
            ],
            "Value": [
                "2PL Bayesian IRT",
                "Normal(0, 1) + two anchors fixed at +1/-1",
                "Normal(0, 5)",
                "Normal(0, 1)",
                str(n_samples),
                str(n_tune),
                str(n_chains),
                str(TARGET_ACCEPT),
                str(RANDOM_SEED),
                f"{MINORITY_THRESHOLD:.3f} ({MINORITY_THRESHOLD * 100:.1f}%)",
                f"{SENSITIVITY_THRESHOLD:.2f} ({SENSITIVITY_THRESHOLD * 100:.0f}%)",
                str(MIN_VOTES),
                f"{MIN_PARTICIPATION_FOR_ANCHOR:.0%}",
                f"{HOLDOUT_FRACTION:.2f} ({HOLDOUT_FRACTION * 100:.0f}%)",
                str(HOLDOUT_SEED),
            ],
            "Description": [
                "Two-parameter logistic IRT with Bayesian estimation via MCMC",
                "Standard normal prior; two anchors fixed for identification",
                "Diffuse prior on bill difficulty",
                "Unconstrained; sign determined by anchors (+ = conservative Yea)",
                "Posterior samples per chain (after tuning)",
                "Adaptation samples (discarded)",
                "Independent Markov chains for convergence assessment",
                "NUTS target acceptance probability",
                "For MCMC reproducibility",
                "Drop votes where minority side < this fraction",
                "Alternative threshold for sensitivity analysis",
                "Drop legislators with fewer substantive votes",
                "Anchors must have participated in at least this fraction of votes",
                "Fraction of observed cells used for holdout validation",
                "NumPy random seed for reproducible holdout selection",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="Changing MCMC parameters or priors constitutes a sensitivity analysis.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
