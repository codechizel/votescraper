"""UMAP-specific HTML report builder.

Builds ~12 sections (tables + figures) for the UMAP Ideological Landscape report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from umap_viz.py):
    from analysis.umap_report import build_umap_report
    build_umap_report(ctx.report, results=results, ...)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, make_gt
except ModuleNotFoundError:
    from report import FigureSection, ReportBuilder, TableSection, make_gt


def build_umap_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    sensitivity_findings: dict[str, dict],
    plots_dir: Path,
    n_neighbors: int,
    min_dist: float,
) -> None:
    """Build the full UMAP HTML report by adding ~12 sections to the ReportBuilder."""
    for chamber, result in results.items():
        _add_umap_parameters(report, result, chamber)
        _add_landscape_figure(report, plots_dir, chamber)
        _add_pc1_gradient_figure(report, plots_dir, chamber)
        _add_irt_gradient_figure(report, plots_dir, chamber)
        _add_validation_table(report, result, chamber)
        _add_legislator_embeddings(report, result, chamber)

    if sensitivity_findings:
        for chamber in results:
            _add_sensitivity_figure(report, plots_dir, chamber)
        _add_sensitivity_table(report, sensitivity_findings)

    _add_analysis_config(report, n_neighbors, min_dist)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_umap_parameters(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: UMAP configuration for this chamber."""
    df = pl.DataFrame(
        {
            "parameter": [
                "N Neighbors",
                "Min Distance",
                "Metric",
                "Random State",
                "N Legislators",
                "N Votes",
                "Imputation",
            ],
            "value": [
                str(result["n_neighbors"]),
                str(result["min_dist"]),
                "cosine",
                "42",
                str(result["embedding_df"].height),
                str(len(result["vote_ids"])),
                "Row mean (legislator Yea rate)",
            ],
        }
    )

    html = make_gt(
        df,
        title=f"{chamber} -- UMAP Configuration",
        subtitle=f"{result['embedding_df'].height} legislators x {len(result['vote_ids'])} votes",
        column_labels={"parameter": "Parameter", "value": "Value"},
    )
    report.add(
        TableSection(
            id=f"umap-params-{chamber.lower()}",
            title=f"{chamber} UMAP Parameters",
            html=html,
        )
    )


def _add_landscape_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Primary UMAP visualization: party-colored scatter plot."""
    path = plots_dir / f"umap_landscape_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-landscape-{chamber.lower()}",
                f"{chamber} UMAP Ideological Landscape",
                path,
                caption=(
                    f"A map of voting behavior ({chamber}). Nearby legislators "
                    "vote alike; distance = voting dissimilarity. Red = Republican, "
                    "Blue = Democrat. The axes have no inherent meaning — only "
                    "relative positions and distances matter. Cross-party outliers "
                    "(if any) are labeled with imputation artifact warnings."
                ),
            )
        )


def _add_pc1_gradient_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Validation: UMAP colored by PCA PC1 scores."""
    path = plots_dir / f"umap_pc1_gradient_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-pc1grad-{chamber.lower()}",
                f"{chamber} UMAP Colored by PCA PC1",
                path,
                caption=(
                    f"UMAP embedding colored by PCA PC1 score ({chamber}). A smooth "
                    "red-to-blue gradient validates that UMAP preserves the same "
                    "ideological dimension identified by PCA."
                ),
            )
        )


def _add_irt_gradient_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    """Validation: UMAP colored by IRT ideal points."""
    path = plots_dir / f"umap_irt_gradient_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irtgrad-{chamber.lower()}",
                f"{chamber} UMAP Colored by IRT Ideal Point",
                path,
                caption=(
                    f"UMAP embedding colored by Bayesian IRT ideal point ({chamber}). "
                    "Smooth gradient confirms UMAP aligns with the nonlinear IRT model."
                ),
            )
        )


def _add_validation_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Spearman correlations of UMAP1 vs upstream methods."""
    validation = result.get("validation", {})
    if not validation:
        return

    rows = []
    if "pca_pc1_spearman" in validation:
        rows.append(
            {
                "comparison": "UMAP1 vs PCA PC1",
                "spearman_rho": validation["pca_pc1_spearman"],
                "p_value": validation["pca_pc1_pvalue"],
                "n_shared": validation["pca_n_shared"],
            }
        )
    if "irt_spearman" in validation:
        rows.append(
            {
                "comparison": "UMAP1 vs IRT Ideal Point",
                "spearman_rho": validation["irt_spearman"],
                "p_value": validation["irt_pvalue"],
                "n_shared": validation["irt_n_shared"],
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} -- Validation Correlations",
        subtitle="Spearman rank correlation (not Pearson -- UMAP is nonlinear)",
        column_labels={
            "comparison": "Comparison",
            "spearman_rho": "Spearman rho",
            "p_value": "p-value",
            "n_shared": "N Shared",
        },
        number_formats={
            "spearman_rho": ".4f",
            "p_value": ".2e",
        },
        source_note="Spearman rho > 0.85 indicates strong alignment with upstream methods.",
    )
    report.add(
        TableSection(
            id=f"validation-{chamber.lower()}",
            title=f"{chamber} Validation",
            html=html,
        )
    )


def _add_legislator_embeddings(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators ranked by UMAP1 coordinate."""
    embedding_df = result["embedding_df"].sort("UMAP1", descending=True)

    display_cols = ["full_name", "party", "district", "UMAP1", "UMAP2"]
    available = [c for c in display_cols if c in embedding_df.columns]
    df = embedding_df.select(available)

    html = make_gt(
        df,
        title=f"{chamber} -- Legislator UMAP Coordinates (ranked by UMAP1)",
        subtitle=(
            f"{df.height} legislators. UMAP coordinates are arbitrary — "
            "only relative positions matter, not the numbers themselves."
        ),
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "district": "District",
            "UMAP1": "UMAP1",
            "UMAP2": "UMAP2",
        },
        number_formats={"UMAP1": ".3f", "UMAP2": ".3f"},
    )
    report.add(
        TableSection(
            id=f"embeddings-{chamber.lower()}",
            title=f"{chamber} Legislator Embeddings",
            html=html,
        )
    )


def _add_sensitivity_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Sensitivity grid: 4-panel comparison across n_neighbors values."""
    path = plots_dir / f"umap_sensitivity_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-sensitivity-{chamber.lower()}",
                f"{chamber} UMAP Sensitivity Grid",
                path,
                caption=(
                    f"UMAP embeddings at n_neighbors = 5, 15, 30, 50 ({chamber}). "
                    "Consistent party separation across settings indicates robust "
                    "results. Structure that appears only at one setting is an artifact."
                ),
            )
        )


def _add_sensitivity_table(
    report: ReportBuilder,
    sensitivity_findings: dict[str, dict],
) -> None:
    """Table: Procrustes similarity between n_neighbors settings."""
    rows = []
    for chamber, sweep in sensitivity_findings.items():
        for pair in sweep["pairs"]:
            rows.append(
                {
                    "chamber": chamber,
                    "n_neighbors_a": pair["nn_a"],
                    "n_neighbors_b": pair["nn_b"],
                    "procrustes_similarity": pair["procrustes_similarity"],
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Sensitivity -- Procrustes Similarity",
        subtitle=(
            "Rotation-invariant comparison of embeddings across n_neighbors values. "
            "1.0 = identical structure, > 0.7 = stable."
        ),
        column_labels={
            "chamber": "Chamber",
            "n_neighbors_a": "N Neighbors A",
            "n_neighbors_b": "N Neighbors B",
            "procrustes_similarity": "Procrustes Similarity",
        },
        number_formats={"procrustes_similarity": ".4f"},
        source_note=(
            "Procrustes similarity measures shape agreement after optimal "
            "rotation/reflection/scaling. Values > 0.7 indicate stable structure."
        ),
    )
    report.add(
        TableSection(
            id="sensitivity-procrustes",
            title="Sensitivity Analysis",
            html=html,
        )
    )


def _add_analysis_config(
    report: ReportBuilder,
    n_neighbors: int,
    min_dist: float,
) -> None:
    """Table: Analysis parameters used in this run."""
    try:
        from analysis.umap_viz import (
            DEFAULT_METRIC,
            DEFAULT_MIN_DIST,
            DEFAULT_N_NEIGHBORS,
            RANDOM_STATE,
            SENSITIVITY_N_NEIGHBORS,
        )
    except ModuleNotFoundError:
        from umap_viz import (  # type: ignore[no-redef]
            DEFAULT_METRIC,
            DEFAULT_MIN_DIST,
            DEFAULT_N_NEIGHBORS,
            RANDOM_STATE,
            SENSITIVITY_N_NEIGHBORS,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "N Neighbors",
                "Min Distance",
                "Metric",
                "Random State",
                "Imputation Method",
                "Sensitivity N Neighbors",
            ],
            "Value": [
                str(n_neighbors),
                str(min_dist),
                DEFAULT_METRIC,
                str(RANDOM_STATE),
                "Row mean (legislator Yea rate)",
                str(SENSITIVITY_N_NEIGHBORS),
            ],
            "Description": [
                f"Controls local vs global focus (default: {DEFAULT_N_NEIGHBORS})",
                f"Minimum distance between embedded points (default: {DEFAULT_MIN_DIST})",
                "Cosine distance for binary vote data",
                "Fixed seed for reproducibility",
                "Missing votes filled with each legislator's average Yea rate",
                "Values tested in sensitivity sweep",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="Changing n_neighbors or min_dist constitutes a sensitivity analysis.",
    )
    report.add(
        TableSection(
            id="analysis-config",
            title="Analysis Parameters",
            html=html,
        )
    )
