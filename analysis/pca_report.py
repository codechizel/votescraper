"""PCA-specific HTML report builder.

Builds ~14 sections (tables + figures) for the Principal Component Analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from pca.py):
    from analysis.pca_report import build_pca_report
    build_pca_report(ctx.report, results=results, ...)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, make_gt
except ModuleNotFoundError:
    from report import FigureSection, ReportBuilder, TableSection, make_gt

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
    for chamber, result in results.items():
        _add_pca_summary(report, result, chamber)
        _add_scree_figure(report, plots_dir, chamber)
        _add_ideological_map_figure(report, plots_dir, chamber)
        _add_pc1_distribution_figure(report, plots_dir, chamber)
        _add_top_loadings(report, result, chamber, pc=1)
        _add_top_loadings(report, result, chamber, pc=2)
        _add_legislator_scores(report, result, chamber)

    if sensitivity_findings:
        _add_sensitivity_table(report, sensitivity_findings)
        for chamber in results:
            _add_sensitivity_figure(report, plots_dir, chamber)

    if validation_results:
        _add_validation_table(report, validation_results)

    _add_analysis_parameters(report, n_components)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_pca_summary(
    report: ReportBuilder, result: dict, chamber: str,
) -> None:
    """Table: PCA summary (component, eigenvalue, explained var, cumulative)."""
    pca = result["pca"]
    ev = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_
    cumulative = np.cumsum(ev)

    rows = []
    for i in range(len(ev)):
        rows.append({
            "component": f"PC{i+1}",
            "eigenvalue": float(eigenvalues[i]),
            "explained_var_pct": float(ev[i]) * 100,
            "cumulative_pct": float(cumulative[i]) * 100,
        })
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
    report.add(TableSection(
        id=f"pca-summary-{chamber.lower()}", title=f"{chamber} PCA Summary", html=html,
    ))


def _add_scree_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"scree_{chamber.lower()}.png"
    if path.exists():
        report.add(FigureSection.from_file(
            f"fig-scree-{chamber.lower()}",
            f"{chamber} Scree Plot",
            path,
            caption=(
                f"Individual and cumulative explained variance for {chamber}. "
                "Sharp elbow after PC1 indicates a one-dimensional legislature."
            ),
        ))


def _add_ideological_map_figure(
    report: ReportBuilder, plots_dir: Path, chamber: str,
) -> None:
    path = plots_dir / f"ideological_map_{chamber.lower()}.png"
    if path.exists():
        report.add(FigureSection.from_file(
            f"fig-ideo-{chamber.lower()}",
            f"{chamber} Ideological Map",
            path,
            caption=(
                f"Legislators in PC1-PC2 space ({chamber}). Red = Republican, "
                "Blue = Democrat. PC1 positive = conservative direction."
            ),
        ))


def _add_pc1_distribution_figure(
    report: ReportBuilder, plots_dir: Path, chamber: str,
) -> None:
    path = plots_dir / f"pc1_distribution_{chamber.lower()}.png"
    if path.exists():
        report.add(FigureSection.from_file(
            f"fig-pc1dist-{chamber.lower()}",
            f"{chamber} PC1 Distribution",
            path,
            caption=(
                f"Kernel density estimate of PC1 scores by party ({chamber}). "
                "Overlap region shows moderate/swing legislators."
            ),
        ))


def _add_top_loadings(
    report: ReportBuilder, result: dict, chamber: str, pc: int,
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

    combined = pl.concat([top_pos, top_neg])

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
    report.add(TableSection(
        id=f"loadings-pc{pc}-{chamber.lower()}",
        title=f"{chamber} Top PC{pc} Loadings",
        html=html,
    ))


def _add_legislator_scores(
    report: ReportBuilder, result: dict, chamber: str,
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
    report.add(TableSection(
        id=f"scores-{chamber.lower()}",
        title=f"{chamber} Legislator Scores",
        html=html,
    ))


def _add_sensitivity_table(report: ReportBuilder, findings: dict) -> None:
    """Table: Sensitivity comparison across chambers."""
    rows = []
    for chamber, data in findings.items():
        if isinstance(data, dict) and data.get("skipped"):
            continue
        if not isinstance(data, dict):
            continue
        rows.append({
            "chamber": chamber,
            "default_threshold": f"{data['default_threshold']*100:.1f}%",
            "sensitivity_threshold": f"{data['sensitivity_threshold']*100:.0f}%",
            "default_n_legislators": data["default_n_legislators"],
            "sensitivity_n_legislators": data["sensitivity_n_legislators"],
            "default_n_votes": data["default_n_votes"],
            "sensitivity_n_votes": data["sensitivity_n_votes"],
            "shared_legislators": data["shared_legislators"],
            "pearson_r": data["pearson_r"],
        })

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
    report.add(TableSection(
        id="sensitivity", title="Sensitivity Analysis", html=html,
    ))


def _add_sensitivity_figure(
    report: ReportBuilder, plots_dir: Path, chamber: str,
) -> None:
    path = plots_dir / f"sensitivity_pc1_{chamber.lower()}.png"
    if path.exists():
        report.add(FigureSection.from_file(
            f"fig-sensitivity-{chamber.lower()}",
            f"{chamber} Sensitivity Scatter",
            path,
            caption=(
                f"Default vs. sensitivity PC1 scores ({chamber}). "
                "Points near the identity line indicate stable results."
            ),
        ))


def _add_validation_table(report: ReportBuilder, results: dict[str, dict]) -> None:
    """Table: Holdout validation metrics."""
    rows = []
    for chamber, data in results.items():
        rows.append({
            "chamber": chamber,
            "holdout_cells": data["holdout_cells"],
            "base_rate": data["base_rate"],
            "base_accuracy": data["base_accuracy"],
            "accuracy": data["accuracy"],
            "auc_roc": data["auc_roc"],
            "rmse": data["rmse"],
            "n_components": data["n_components"],
        })

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
    report.add(TableSection(
        id="validation", title="Holdout Validation", html=html,
    ))


def _add_analysis_parameters(report: ReportBuilder, n_components: int) -> None:
    """Table: Analysis parameters used in this run."""
    try:
        from analysis.pca import (
            HOLDOUT_FRACTION,
            HOLDOUT_SEED,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            SENSITIVITY_THRESHOLD,
        )
    except ModuleNotFoundError:
        from pca import (  # type: ignore[no-redef]
            HOLDOUT_FRACTION,
            HOLDOUT_SEED,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            SENSITIVITY_THRESHOLD,
        )

    df = pl.DataFrame({
        "Parameter": [
            "N Components",
            "Imputation Method",
            "Minority Threshold (Default)",
            "Minority Threshold (Sensitivity)",
            "Min Substantive Votes",
            "Holdout Fraction",
            "Holdout Random Seed",
        ],
        "Value": [
            str(n_components),
            "Row mean (legislator Yea rate)",
            f"{MINORITY_THRESHOLD:.3f} ({MINORITY_THRESHOLD*100:.1f}%)",
            f"{SENSITIVITY_THRESHOLD:.2f} ({SENSITIVITY_THRESHOLD*100:.0f}%)",
            str(MIN_VOTES),
            f"{HOLDOUT_FRACTION:.2f} ({HOLDOUT_FRACTION*100:.0f}%)",
            str(HOLDOUT_SEED),
        ],
        "Description": [
            "Principal components extracted per chamber",
            "Missing votes filled with each legislator's average Yea rate",
            "Drop votes where minority side < this fraction",
            "Alternative threshold for sensitivity analysis",
            "Drop legislators with fewer substantive votes",
            "Fraction of non-null cells randomly masked for validation",
            "NumPy random seed for reproducible holdout selection",
        ],
    })
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="Changing these values constitutes a sensitivity analysis.",
    )
    report.add(TableSection(
        id="analysis-params", title="Analysis Parameters", html=html,
    ))
