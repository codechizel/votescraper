"""PCA-specific HTML report builder.

Builds ~14 sections (tables + figures) for the Principal Component Analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from pca.py):
    from analysis.pca_report import build_pca_report
    build_pca_report(ctx.report, results=results, ...)
"""

from pathlib import Path

import numpy as np
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

TOP_LOADINGS = 15  # Number of top positive/negative loadings to show


def build_pca_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    sensitivity_findings: dict,
    validation_results: dict[str, dict],
    plots_dir: Path,
    n_components: int,
) -> None:
    """Build the full PCA HTML report by adding ~14 sections to the ReportBuilder."""
    findings = _generate_pca_key_findings(results, validation_results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    for chamber, result in results.items():
        _add_pca_summary(report, result, chamber)
        _add_dimensionality_diagnostics(report, result, chamber)
        _add_scree_figure(report, plots_dir, chamber, result)
        _add_ideological_map_figure(report, plots_dir, chamber)
        _add_pc1_distribution_figure(report, plots_dir, chamber)
        _add_top_loadings(report, result, chamber, pc=1)
        _add_top_loadings(report, result, chamber, pc=2)
        _add_legislator_scores(report, result, chamber)
        _add_reconstruction_error(report, result, chamber)

    if sensitivity_findings:
        _add_sensitivity_table(report, sensitivity_findings)
        for chamber in results:
            _add_sensitivity_figure(report, plots_dir, chamber)
        # Warn when sensitivity r is low
        for chamber, data in sensitivity_findings.items():
            r = data.get("pearson_r", 1.0)
            if r < 0.95:
                report.add(
                    TextSection(
                        id=f"sensitivity-warning-{chamber.lower()}",
                        title=f"{chamber} Sensitivity Warning",
                        html=(
                            '<div style="background:#fff3cd; border:1px solid #ffc107; '
                            'border-radius:6px; padding:12px 16px; margin:8px 0;">'
                            f"<strong>Low Sensitivity Correlation ({chamber}):</strong> "
                            f"r = {r:.3f} between default and sensitivity filter thresholds. "
                            f"PC1 scores are sensitive to the vote-filtering threshold, "
                            f"meaning results may change materially with different inclusion "
                            f"criteria. Interpret with caution.</div>"
                        ),
                    )
                )

    if validation_results:
        _add_validation_table(report, validation_results)

    _add_analysis_parameters(report, n_components)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_pca_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: PCA summary (component, eigenvalue, explained var, cumulative)."""
    pca = result["pca"]
    ev = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(ev)

    rows = []
    for i in range(len(ev)):
        rows.append(
            {
                "component": f"PC{i + 1}",
                "eigenvalue": float(eigenvalues[i]),
                "explained_var_pct": float(ev[i]) * 100,
                "cumulative_pct": float(cumulative[i]) * 100,
            }
        )
    df = pl.DataFrame(rows)

    html = make_gt(
        df,
        title=f"{chamber} — PCA Summary",
        subtitle=f"{result['scores_df'].height} legislators x {len(result['vote_ids'])} votes",
        column_labels={
            "component": "Component",
            "eigenvalue": "Eigenvalue",
            "explained_var_pct": "Explained Var. (%)",
            "cumulative_pct": "Cumulative (%)",
        },
        number_formats={
            "eigenvalue": ".2f",
            "explained_var_pct": ".1f",
            "cumulative_pct": ".1f",
        },
    )
    report.add(
        TableSection(
            id=f"pca-summary-{chamber.lower()}",
            title=f"{chamber} PCA Summary",
            html=html,
        )
    )


def _add_dimensionality_diagnostics(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: eigenvalue ratio and parallel analysis results."""
    pca = result["pca"]
    eigenvalues = pca.explained_variance_
    pa_thresholds = result["parallel_thresholds"]
    n_comp = result["n_components"]

    rows = []
    for i in range(n_comp):
        rows.append(
            {
                "component": f"PC{i + 1}",
                "eigenvalue": float(eigenvalues[i]),
                "random_threshold": float(pa_thresholds[i]),
                "significant": "Yes" if eigenvalues[i] > pa_thresholds[i] else "No",
            }
        )

    df = pl.DataFrame(rows)

    eigenvalue_ratio = result["eigenvalue_ratio"]
    n_sig = result["n_significant"]
    if eigenvalue_ratio > 5:
        interpretation = "strongly one-dimensional"
    elif eigenvalue_ratio > 3:
        interpretation = "predominantly one-dimensional"
    else:
        interpretation = "meaningful second dimension present"

    html = make_gt(
        df,
        title=f"{chamber} — Dimensionality Diagnostics",
        subtitle=(
            f"λ1/λ2 = {eigenvalue_ratio:.2f} ({interpretation}). "
            f"Parallel analysis: {n_sig} significant dimension(s)."
        ),
        column_labels={
            "component": "Component",
            "eigenvalue": "Eigenvalue",
            "random_threshold": "95th Pct. Random",
            "significant": "Significant?",
        },
        number_formats={
            "eigenvalue": ".2f",
            "random_threshold": ".2f",
        },
        source_note="Horn's parallel analysis (1965): retain components with eigenvalues above "
        "the 95th percentile of random data of the same shape.",
    )
    report.add(
        TableSection(
            id=f"dimensionality-{chamber.lower()}",
            title=f"{chamber} Dimensionality Diagnostics",
            html=html,
        )
    )


def _add_reconstruction_error(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: legislators with high reconstruction error (> mean + 2σ)."""
    recon_df = result["reconstruction_error_df"]
    high_error = recon_df.filter(pl.col("high_error"))

    if high_error.height == 0:
        return

    # Join with scores_df for metadata
    scores = result["scores_df"]
    display = high_error.join(
        scores.select("legislator_slug", "full_name", "party", "PC1"),
        on="legislator_slug",
        how="left",
    ).sort("reconstruction_rmse", descending=True)

    display = display.select(
        "full_name",
        "party",
        "PC1",
        "reconstruction_rmse",
    )

    mean_rmse = float(recon_df["reconstruction_rmse"].mean())
    std_rmse = float(recon_df["reconstruction_rmse"].std())

    html = make_gt(
        display,
        title=f"{chamber} — High Reconstruction Error Legislators",
        subtitle=(
            f"Legislators with RMSE > {mean_rmse:.4f} + 2 × {std_rmse:.4f} = "
            f"{mean_rmse + 2 * std_rmse:.4f}. "
            "These voting patterns are poorly explained by the PCA model."
        ),
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "PC1": "PC1",
            "reconstruction_rmse": "RMSE",
        },
        number_formats={"PC1": ".3f", "reconstruction_rmse": ".4f"},
    )
    report.add(
        TableSection(
            id=f"recon-error-{chamber.lower()}",
            title=f"{chamber} Reconstruction Error",
            html=html,
        )
    )


def _add_scree_figure(report: ReportBuilder, plots_dir: Path, chamber: str, result: dict) -> None:
    path = plots_dir / f"scree_{chamber.lower()}.png"
    if path.exists():
        eigenvalue_ratio = result.get("eigenvalue_ratio", 0.0)
        if eigenvalue_ratio > 5:
            scree_caption = (
                "Sharp elbow after PC1 indicates a strongly one-dimensional legislature."
            )
        elif eigenvalue_ratio > 3:
            scree_caption = "Elbow after PC1 indicates a predominantly one-dimensional legislature."
        else:
            scree_caption = (
                f"Eigenvalue ratio ({eigenvalue_ratio:.1f}) suggests a meaningful "
                "second dimension is present."
            )
        report.add(
            FigureSection.from_file(
                f"fig-scree-{chamber.lower()}",
                f"{chamber} Scree Plot",
                path,
                caption=(
                    f"Individual and cumulative explained variance for {chamber}. {scree_caption}"
                ),
                alt_text=(
                    f"Bar-and-line chart showing PCA eigenvalues for {chamber}. "
                    "First component explains most variance with a sharp elbow."
                ),
            )
        )


def _add_ideological_map_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"ideological_map_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-ideo-{chamber.lower()}",
                f"{chamber} Ideological Map",
                path,
                caption=(
                    f"Legislators in PC1-PC2 space ({chamber}). Red = Republican, "
                    "Blue = Democrat. PC1 positive = conservative direction."
                ),
                alt_text=(
                    f"Scatter plot of legislators in PC1-PC2 space for {chamber}. "
                    "Red Republican and blue Democrat clusters separate along PC1."
                ),
            )
        )


def _add_pc1_distribution_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"pc1_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-pc1dist-{chamber.lower()}",
                f"{chamber} PC1 Distribution",
                path,
                caption=(
                    f"Kernel density estimate of PC1 scores by party ({chamber}). "
                    "Overlap region shows moderate/swing legislators."
                ),
                alt_text=(
                    f"Density plot of PC1 scores by party for {chamber}. "
                    "Republican and Democrat distributions overlap in the moderate center."
                ),
            )
        )


def _add_top_loadings(
    report: ReportBuilder,
    result: dict,
    chamber: str,
    pc: int,
) -> None:
    """Table: Top positive and negative loadings for a PC."""
    loadings_df = result["loadings_df"]
    pc_col = f"PC{pc}"

    if pc_col not in loadings_df.columns:
        return

    # Top positive loadings
    top_pos = loadings_df.sort(pc_col, descending=True).head(TOP_LOADINGS)
    # Top negative loadings
    top_neg = loadings_df.sort(pc_col, descending=False).head(TOP_LOADINGS)

    from analysis.phase_utils import drop_empty_optional_columns

    combined = pl.concat([top_pos, top_neg])
    combined = drop_empty_optional_columns(combined, ["short_title"])

    # Select columns to display
    display_cols = [pc_col, "vote_id", "bill_number"]
    # Add optional metadata columns if they exist
    for opt_col in ["short_title", "motion", "vote_type"]:
        if opt_col in combined.columns:
            display_cols.append(opt_col)
    combined = combined.select([c for c in display_cols if c in combined.columns])

    labels: dict[str, str] = {
        pc_col: f"{pc_col} Loading",
        "vote_id": "Vote ID",
        "bill_number": "Bill",
    }
    if "short_title" in combined.columns:
        labels["short_title"] = "Title"
    if "motion" in combined.columns:
        labels["motion"] = "Motion"
    if "vote_type" in combined.columns:
        labels["vote_type"] = "Type"

    html = make_gt(
        combined,
        title=f"{chamber} — Top {pc_col} Loadings",
        subtitle=f"Top {TOP_LOADINGS} positive and {TOP_LOADINGS} negative loadings",
        column_labels=labels,
        number_formats={pc_col: ".4f"},
    )
    report.add(
        TableSection(
            id=f"loadings-pc{pc}-{chamber.lower()}",
            title=f"{chamber} Top PC{pc} Loadings",
            html=html,
        )
    )


def _add_legislator_scores(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators ranked by PC1 score."""
    scores = result["scores_df"].sort("PC1", descending=True)

    # Select display columns
    display_cols = ["full_name", "party", "district", "PC1", "PC2"]
    available = [c for c in display_cols if c in scores.columns]
    df = scores.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Legislator Scores (ranked by PC1)",
        subtitle=f"{df.height} legislators, positive PC1 = conservative",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "PC1": "PC1",
            "PC2": "PC2",
        },
        number_formats={"PC1": ".3f", "PC2": ".3f"},
    )
    report.add(
        TableSection(
            id=f"scores-{chamber.lower()}",
            title=f"{chamber} Legislator Scores",
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
        title="Sensitivity Analysis — PC1 Correlation",
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
        source_note="r > 0.95 indicates robust results.",
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
    path = plots_dir / f"sensitivity_pc1_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-sensitivity-{chamber.lower()}",
                f"{chamber} Sensitivity Scatter",
                path,
                caption=(
                    f"Default vs. sensitivity PC1 scores ({chamber}). "
                    "Points near the identity line indicate stable results."
                ),
                alt_text=(
                    f"Scatter plot comparing default and sensitivity PC1 scores for {chamber}. "
                    "Points cluster tightly along the identity line, indicating stable results."
                ),
            )
        )


def _add_validation_table(report: ReportBuilder, results: dict[str, dict]) -> None:
    """Table: Holdout validation metrics."""
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
                "rmse": data["rmse"],
                "n_components": data["n_components"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Holdout Validation Metrics",
        subtitle="20% random holdout, seed=42. Accuracy must beat base rate.",
        column_labels={
            "chamber": "Chamber",
            "holdout_cells": "Holdout Cells",
            "base_rate": "Base Rate (Yea)",
            "base_accuracy": "Base Accuracy",
            "accuracy": "PCA Accuracy",
            "auc_roc": "AUC-ROC",
            "rmse": "RMSE",
            "n_components": "Components",
        },
        number_formats={
            "holdout_cells": ",.0f",
            "base_rate": ".3f",
            "base_accuracy": ".3f",
            "accuracy": ".3f",
            "auc_roc": ".3f",
            "rmse": ".3f",
        },
        source_note="PCA accuracy must exceed base-rate accuracy to demonstrate predictive power.",
    )
    report.add(
        TableSection(
            id="validation",
            title="Holdout Validation",
            html=html,
        )
    )


def _generate_pca_key_findings(
    results: dict[str, dict],
    validation_results: dict[str, dict],
) -> list[str]:
    """Generate 2-4 key findings from PCA results."""
    findings: list[str] = []

    for chamber, result in results.items():
        pca = result.get("pca")
        if pca is None:
            continue
        ev = pca.explained_variance_ratio_
        if len(ev) > 0:
            pc1_pct = float(ev[0]) * 100
            findings.append(
                f"{chamber} PC1 explains <strong>{pc1_pct:.1f}%</strong> of vote variance "
                f"(eigenvalue ratio = {result.get('eigenvalue_ratio', 0):.1f})."
            )
        break  # Only first chamber for conciseness

    # Validation accuracy
    for chamber, data in validation_results.items():
        acc = data.get("accuracy")
        auc = data.get("auc_roc")
        base = data.get("base_accuracy")
        if acc is not None and auc is not None:
            findings.append(
                f"{chamber} holdout validation: <strong>{acc:.1%}</strong> accuracy "
                f"(AUC-ROC = {auc:.3f}, base rate = {base:.1%})."
            )
            break

    # Parallel analysis
    for chamber, result in results.items():
        n_sig = result.get("n_significant")
        if n_sig is not None:
            findings.append(
                f"Parallel analysis retains <strong>{n_sig}</strong> significant "
                f"dimension{'s' if n_sig != 1 else ''} in {chamber}."
            )
            break

    return findings


def _add_analysis_parameters(report: ReportBuilder, n_components: int) -> None:
    """Table: Analysis parameters used in this run."""
    try:
        from analysis.pca import (
            HOLDOUT_FRACTION,
            HOLDOUT_SEED,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            PARALLEL_ANALYSIS_N_ITER,
            RECONSTRUCTION_ERROR_THRESHOLD_SD,
            SENSITIVITY_THRESHOLD,
        )
    except ModuleNotFoundError:
        from pca import (  # type: ignore[no-redef]
            HOLDOUT_FRACTION,
            HOLDOUT_SEED,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            PARALLEL_ANALYSIS_N_ITER,
            RECONSTRUCTION_ERROR_THRESHOLD_SD,
            SENSITIVITY_THRESHOLD,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "N Components",
                "Imputation Method",
                "Minority Threshold (Default)",
                "Minority Threshold (Sensitivity)",
                "Min Substantive Votes",
                "Holdout Fraction",
                "Holdout Random Seed",
                "Parallel Analysis Iterations",
                "Reconstruction Error Threshold",
            ],
            "Value": [
                str(n_components),
                "Row mean (legislator Yea rate)",
                f"{MINORITY_THRESHOLD:.3f} ({MINORITY_THRESHOLD * 100:.1f}%)",
                f"{SENSITIVITY_THRESHOLD:.2f} ({SENSITIVITY_THRESHOLD * 100:.0f}%)",
                str(MIN_VOTES),
                f"{HOLDOUT_FRACTION:.2f} ({HOLDOUT_FRACTION * 100:.0f}%)",
                str(HOLDOUT_SEED),
                str(PARALLEL_ANALYSIS_N_ITER),
                f"mean + {RECONSTRUCTION_ERROR_THRESHOLD_SD:.0f}σ",
            ],
            "Description": [
                "Principal components extracted per chamber",
                "Missing votes filled with each legislator's average Yea rate",
                "Drop votes where minority side < this fraction",
                "Alternative threshold for sensitivity analysis",
                "Drop legislators with fewer substantive votes",
                "Fraction of non-null cells randomly masked for validation",
                "NumPy random seed for reproducible holdout and parallel analysis",
                "Random matrices generated for Horn's parallel analysis",
                "Flag legislators with reconstruction RMSE above this threshold",
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
