"""IRT-specific HTML report builder.

Builds ~16 sections (tables + figures) for the Bayesian IRT report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from irt.py):
    from analysis.irt_report import build_irt_report
    build_irt_report(ctx.report, results=results, ...)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, make_gt
except ModuleNotFoundError:
    from report import FigureSection, ReportBuilder, TableSection, make_gt

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
    """Build the full IRT HTML report by adding ~16 sections to the ReportBuilder."""
    for chamber, result in results.items():
        _add_model_summary(report, result, chamber, n_samples, n_tune, n_chains)
        _add_convergence_table(report, result, chamber)
        _add_forest_figure(report, plots_dir, chamber)
        _add_ideal_point_table(report, result, chamber)
        _add_discrimination_figure(report, plots_dir, chamber)
        _add_top_discriminating_votes(report, result, chamber)
        _add_trace_figure(report, plots_dir, chamber)
        _add_ppc_figure(report, plots_dir, chamber)

    # Cross-chamber sections
    if pca_comparisons:
        for chamber in results:
            _add_pca_comparison_figure(report, plots_dir, chamber)
        _add_pca_comparison_table(report, pca_comparisons)

    if validation_results:
        _add_validation_table(report, validation_results)

    if ppc_results:
        _add_ppc_summary_table(report, ppc_results)

    if sensitivity_findings:
        _add_sensitivity_table(report, sensitivity_findings)
        for chamber in results:
            _add_sensitivity_figure(report, plots_dir, chamber)

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
                "Conservative Anchor",
                "Liberal Anchor",
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
                result["cons_slug"],
                result["lib_slug"],
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
                    "Red = Republican, Blue = Democrat. Positive = conservative."
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

    html = make_gt(
        df,
        title=f"{chamber} — Legislator Ideal Points (ranked by xi_mean)",
        subtitle=f"{df.height} legislators, positive = conservative",
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
        source_note="HDI = 95% Highest Density Interval.",
    )
    report.add(
        TableSection(
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
    top_pos = bp.filter(pl.col("beta_mean") > 0).sort("abs_beta", descending=True).head(
        TOP_DISCRIMINATING
    )
    # Top liberal-Yea bills (most negative beta, i.e. highest |beta| where beta < 0)
    top_neg = bp.filter(pl.col("beta_mean") < 0).sort("abs_beta", descending=True).head(
        TOP_DISCRIMINATING
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

    html = make_gt(
        combined,
        title=f"{chamber} — Most Discriminating Roll Calls",
        subtitle=(
            f"Top {TOP_DISCRIMINATING} conservative-Yea (β > 0) and "
            f"{TOP_DISCRIMINATING} liberal-Yea (β < 0)"
        ),
        column_labels=labels,
        number_formats={"beta_mean": ".3f", "beta_sd": ".3f", "alpha_mean": ".3f"},
    )
    report.add(
        TableSection(
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
        rows.append(
            {
                "chamber": chamber,
                "default_threshold": f"{data['default_threshold'] * 100:.1f}%",
                "sensitivity_threshold": f"{data['sensitivity_threshold'] * 100:.0f}%",
                "default_n_legislators": data["default_n_legislators"],
                "sensitivity_n_legislators": data["sensitivity_n_legislators"],
                "default_n_votes": data["default_n_votes"],
                "sensitivity_n_votes": data["sensitivity_n_votes"],
                "shared_legislators": data["shared_legislators"],
                "pearson_r": data["pearson_r"],
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
        source_note="r > 0.95 indicates robust results across threshold choices.",
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
            )
        )


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
