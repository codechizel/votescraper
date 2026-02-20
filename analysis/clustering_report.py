"""Clustering-specific HTML report builder.

Builds ~35 sections (tables, figures, and text) for the clustering analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from clustering.py):
    from analysis.clustering_report import build_clustering_report
    build_clustering_report(ctx.report, results=results, ...)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    from analysis.report import FigureSection, ReportBuilder, TableSection, TextSection, make_gt
except ModuleNotFoundError:
    from report import (  # type: ignore[no-redef]
        FigureSection,
        ReportBuilder,
        TableSection,
        TextSection,
        make_gt,
    )

# Must match CONTESTED_PARTY_THRESHOLD in clustering.py (circular import prevents direct import)
_CONTESTED_PARTY_THRESHOLD = 0.10


def build_clustering_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    skip_gmm: bool = False,
    skip_sensitivity: bool = False,
) -> None:
    """Build the full clustering HTML report by adding sections to the ReportBuilder."""
    _add_data_summary(report, results)
    _add_interpretation_intro(report)

    for chamber, result in results.items():
        _add_party_loyalty_table(report, result, chamber)

    for chamber in results:
        _add_dendrogram_figure(report, plots_dir, chamber)

    for chamber in results:
        _add_model_selection_figure(report, plots_dir, chamber)

    if not skip_gmm:
        for chamber in results:
            _add_gmm_model_selection_figure(report, plots_dir, chamber)

    _add_model_selection_interpretation(report, results)

    for chamber, result in results.items():
        _add_cluster_assignments_table(report, result, chamber)

    for chamber in results:
        _add_irt_clusters_figure(report, plots_dir, chamber)

    for chamber in results:
        _add_irt_loyalty_figure(report, plots_dir, chamber)

    for chamber, result in results.items():
        _add_cluster_composition_table(report, result, chamber)

    _add_cross_method_agreement(report, results)
    _add_cross_method_interpretation(report)

    _add_flagged_legislators(report, results)

    for chamber in results:
        _add_cluster_box_figure(report, plots_dir, chamber)

    _add_veto_override_table(report, results)

    if not skip_sensitivity:
        _add_sensitivity_table(report, results)

    # Within-party clustering
    _add_within_party_intro(report)
    for chamber in results:
        for party_key in ["republican", "democrat"]:
            _add_within_party_model_selection_figure(report, plots_dir, chamber, party_key)
            _add_within_party_clusters_figure(report, plots_dir, chamber, party_key)
    _add_within_party_summary_table(report, results)

    _add_analysis_parameters(report)
    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_data_summary(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Data dimensions and upstream sources per chamber."""
    rows = []
    for chamber, result in results.items():
        ip = result["ideal_points"]
        vm = result["vote_matrix"]
        rows.append(
            {
                "Chamber": chamber,
                "N Legislators": ip.height,
                "N Votes (filtered)": len(vm.columns) - 1,
                "IRT Source": "irt/latest",
                "Kappa Source": "eda/latest",
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Clustering Data Summary",
        subtitle="Upstream data dimensions per chamber",
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_party_loyalty_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators by party loyalty rate."""
    loyalty = result.get("loyalty")
    if loyalty is None or loyalty.height == 0:
        return

    sorted_loy = loyalty.sort("loyalty_rate")

    display_cols = ["full_name", "party", "loyalty_rate", "n_contested_votes", "n_agree"]
    available = [c for c in display_cols if c in sorted_loy.columns]
    df = sorted_loy.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Party Loyalty ({df.height} legislators)",
        subtitle="Fraction of contested votes agreeing with party median",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "loyalty_rate": "Loyalty Rate",
            "n_contested_votes": "Contested Votes",
            "n_agree": "Agreed",
        },
        number_formats={"loyalty_rate": ".3f"},
        source_note=(
            f"Contested: >= {_CONTESTED_PARTY_THRESHOLD * 100:.0f}% of party dissents on that vote."
        ),
    )
    report.add(
        TableSection(
            id=f"party-loyalty-{chamber.lower()}",
            title=f"{chamber} Party Loyalty",
            html=html,
        )
    )


def _add_dendrogram_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"dendrogram_{chamber.lower()}.png"
    if path.exists():
        truncated = " (truncated)" if chamber == "House" else ""
        report.add(
            FigureSection.from_file(
                f"fig-dendrogram-{chamber.lower()}",
                f"{chamber} Dendrogram{truncated}",
                path,
                caption=(
                    f"Hierarchical clustering dendrogram ({chamber}) using Ward linkage "
                    "on Kappa distance. Leaf labels colored by party."
                ),
            )
        )


def _add_model_selection_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"model_selection_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-model-sel-{chamber.lower()}",
                f"{chamber} K-Means Model Selection",
                path,
                caption=(
                    f"K-Means model selection ({chamber}): inertia (elbow) and "
                    "silhouette score vs number of clusters."
                ),
            )
        )


def _add_gmm_model_selection_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"bic_aic_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-gmm-sel-{chamber.lower()}",
                f"{chamber} GMM Model Selection",
                path,
                caption=(
                    f"GMM model selection ({chamber}): BIC and AIC vs number of components. "
                    "Lower BIC = better model."
                ),
            )
        )


def _add_cluster_assignments_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: All legislators with cluster assignment, IRT, and loyalty."""
    ip = result["ideal_points"]
    km = result.get("kmeans", {})
    labels = km.get("labels")
    if labels is None:
        return

    loyalty = result.get("loyalty")

    df = ip.select("full_name", "party", "district", "xi_mean", "xi_sd").with_columns(
        pl.Series("cluster", labels.tolist())
    )

    if loyalty is not None and loyalty.height > 0:
        df = (
            df.with_columns(
                pl.Series(
                    "legislator_slug",
                    ip["legislator_slug"].to_list(),
                )
            )
            .join(
                loyalty.select("legislator_slug", "loyalty_rate"),
                on="legislator_slug",
                how="left",
            )
            .drop("legislator_slug")
        )

    df = df.sort("cluster", "xi_mean", descending=[False, True])

    labels_dict: dict[str, str] = {
        "full_name": "Legislator",
        "party": "Party",
        "district": "District",
        "xi_mean": "Ideal Point",
        "xi_sd": "Std Dev",
        "cluster": "Cluster",
    }
    formats: dict[str, str] = {
        "xi_mean": ".3f",
        "xi_sd": ".3f",
    }
    if "loyalty_rate" in df.columns:
        labels_dict["loyalty_rate"] = "Loyalty"
        formats["loyalty_rate"] = ".3f"

    html = make_gt(
        df,
        title=f"{chamber} — Cluster Assignments (K-Means)",
        subtitle=f"{df.height} legislators, k={km.get('optimal_k', '?')}",
        column_labels=labels_dict,
        number_formats=formats,
    )
    report.add(
        TableSection(
            id=f"cluster-assign-{chamber.lower()}",
            title=f"{chamber} Cluster Assignments",
            html=html,
        )
    )


def _add_irt_clusters_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"irt_clusters_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-clusters-{chamber.lower()}",
                f"{chamber} IRT Clusters",
                path,
                caption=(
                    f"IRT ideal points colored by K-Means cluster ({chamber}). "
                    "Circles = Republican, Squares = Democrat."
                ),
            )
        )


def _add_irt_loyalty_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"irt_loyalty_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-irt-loyalty-{chamber.lower()}",
                f"{chamber} Ideology vs Party Loyalty",
                path,
                caption=(
                    f"2D view ({chamber}): IRT ideal point (x) vs party loyalty (y). "
                    "Mavericks appear in the low-loyalty region with extreme ideology."
                ),
            )
        )


def _add_cluster_composition_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    """Table: Party breakdown per cluster."""
    summary = result.get("cluster_summary")
    if summary is None:
        return

    display_cols = [
        "cluster",
        "label",
        "n_legislators",
        "n_republican",
        "n_democrat",
        "pct_republican",
        "xi_mean",
        "xi_median",
        "avg_xi_sd",
    ]
    if "avg_loyalty" in summary.columns:
        display_cols.append("avg_loyalty")
    available = [c for c in display_cols if c in summary.columns]

    labels_dict: dict[str, str] = {
        "cluster": "Cluster",
        "label": "Label",
        "n_legislators": "N",
        "n_republican": "Republican",
        "n_democrat": "Democrat",
        "pct_republican": "% Republican",
        "xi_mean": "Mean Ideal Pt",
        "xi_median": "Median Ideal Pt",
        "avg_xi_sd": "Avg Std Dev",
        "avg_loyalty": "Avg Loyalty",
    }
    formats: dict[str, str] = {
        "pct_republican": ".1f",
        "xi_mean": ".3f",
        "xi_median": ".3f",
        "avg_xi_sd": ".3f",
        "avg_loyalty": ".3f",
    }

    html = make_gt(
        summary.select(available),
        title=f"{chamber} — Cluster Composition",
        subtitle="Party composition and IRT statistics per cluster",
        column_labels={k: v for k, v in labels_dict.items() if k in available},
        number_formats={k: v for k, v in formats.items() if k in available},
    )
    report.add(
        TableSection(
            id=f"cluster-comp-{chamber.lower()}",
            title=f"{chamber} Cluster Composition",
            html=html,
        )
    )


def _add_cross_method_agreement(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: ARI matrix between clustering methods."""
    rows = []
    for chamber, result in results.items():
        comparison = result.get("comparison", {})
        ari_matrix = comparison.get("ari_matrix", {})
        for pair, ari_val in ari_matrix.items():
            methods = pair.split("_vs_")
            rows.append(
                {
                    "Chamber": chamber,
                    "Method A": methods[0] if len(methods) > 0 else pair,
                    "Method B": methods[1] if len(methods) > 1 else "",
                    "ARI": ari_val,
                }
            )
        if comparison.get("mean_ari") is not None:
            rows.append(
                {
                    "Chamber": chamber,
                    "Method A": "all methods",
                    "Method B": "mean ARI",
                    "ARI": comparison["mean_ari"],
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Cross-Method Agreement",
        subtitle="Adjusted Rand Index between clustering methods (1.0 = identical)",
        column_labels={
            "Chamber": "Chamber",
            "Method A": "Method A",
            "Method B": "Method B",
            "ARI": "ARI / Rate",
        },
        number_formats={"ARI": ".4f"},
        source_note=(
            "ARI > 0.7 = strong agreement. "
            "Stability = fraction of legislators in same cluster across all methods."
        ),
    )
    report.add(
        TableSection(
            id="cross-method-ari",
            title="Cross-Method Agreement",
            html=html,
        )
    )


def _add_flagged_legislators(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Flagged legislators with assignments and notes."""
    rows = []
    for chamber, result in results.items():
        for entry in result.get("flagged_legislators", []):
            note = ""
            slug = entry["legislator_slug"]
            if "tyson" in slug:
                note = "Extreme IRT from contrarian pattern; check loyalty rate"
            elif "thompson" in slug:
                note = "Milder version of Tyson contrarian pattern"
            elif "miller" in slug:
                note = "Sparse data (30/194 votes); low-confidence cluster"
            elif "hill" in slug:
                note = "Widest HDI in Senate; lowest-confidence cluster"

            loy_val = entry.get("loyalty_rate")
            rows.append(
                {
                    "Chamber": chamber,
                    "Legislator": entry["full_name"],
                    "Party": entry["party"],
                    "Ideal Point": entry["xi_mean"],
                    "Std Dev": entry["xi_sd"],
                    "Cluster": entry["cluster"],
                    "Loyalty": loy_val,
                    "Note": note,
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Flagged Legislators — Cluster Assignments",
        subtitle="Legislators flagged in prior phases with their clustering results",
        column_labels={
            "Chamber": "Chamber",
            "Legislator": "Legislator",
            "Party": "Party",
            "Ideal Point": "Ideal Pt",
            "Std Dev": "SD",
            "Cluster": "Cluster",
            "Loyalty": "Loyalty",
            "Note": "Note",
        },
        number_formats={
            "Ideal Point": ".3f",
            "Std Dev": ".3f",
            "Loyalty": ".3f",
        },
    )
    report.add(
        TableSection(
            id="flagged-legislators",
            title="Flagged Legislators",
            html=html,
        )
    )


def _add_cluster_box_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"cluster_box_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-cluster-box-{chamber.lower()}",
                f"{chamber} Cluster Boxplot",
                path,
                caption=(
                    f"Distribution of IRT ideal points per cluster ({chamber}). "
                    "Box = IQR, whiskers = 1.5x IQR, dots = outliers."
                ),
            )
        )


def _add_veto_override_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Veto override cluster statistics."""
    rows = []
    for chamber, result in results.items():
        veto = result.get("veto_overrides", {})
        if veto.get("skipped"):
            rows.append(
                {
                    "Chamber": chamber,
                    "N Override Votes": veto.get("n_override_votes", 0),
                    "Cluster": "N/A",
                    "Mean Override Yea": None,
                    "N Legislators": None,
                    "Note": "Insufficient override votes for analysis",
                }
            )
            continue

        cluster_stats = veto.get("cluster_stats")
        if cluster_stats is None:
            continue

        for row in cluster_stats.iter_rows(named=True):
            rows.append(
                {
                    "Chamber": chamber,
                    "N Override Votes": veto["n_override_votes"],
                    "Cluster": row["full_cluster"],
                    "Mean Override Yea": row["mean_override_yea_rate"],
                    "N Legislators": row["n_legislators"],
                    "Note": "",
                }
            )

        rows.append(
            {
                "Chamber": chamber,
                "N Override Votes": veto["n_override_votes"],
                "Cluster": "High Yea (>70%)",
                "Mean Override Yea": None,
                "N Legislators": veto.get("n_high_yea_r", 0) + veto.get("n_high_yea_d", 0),
                "Note": f"{veto.get('n_high_yea_r', 0)}R, {veto.get('n_high_yea_d', 0)}D",
            }
        )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Veto Override Voting Patterns by Cluster",
        subtitle="Mean Yea rate on veto override votes per full-dataset cluster",
        column_labels={
            "Chamber": "Chamber",
            "N Override Votes": "Override Votes",
            "Cluster": "Cluster",
            "Mean Override Yea": "Mean Yea Rate",
            "N Legislators": "N Legislators",
            "Note": "Note",
        },
        number_formats={"Mean Override Yea": ".3f"},
        source_note="Veto overrides require 2/3 supermajority, revealing cross-party coalitions.",
    )
    report.add(
        TableSection(
            id="veto-overrides",
            title="Veto Override Clusters",
            html=html,
        )
    )


def _add_sensitivity_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Sensitivity analysis — ARI across k variations."""
    rows = []
    for chamber, result in results.items():
        sensitivity = result.get("sensitivity", {})
        for key, data in sensitivity.items():
            if not isinstance(data, dict):
                continue
            rows.append(
                {
                    "Chamber": chamber,
                    "Comparison": key,
                    "ARI": data.get("ari"),
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Sensitivity Analysis — Cluster Stability",
        subtitle="ARI between default and alternative k values / methods",
        column_labels={
            "Chamber": "Chamber",
            "Comparison": "Comparison",
            "ARI": "ARI",
        },
        number_formats={"ARI": ".4f"},
        source_note="ARI > 0.7 indicates robust cluster structure across k choices.",
    )
    report.add(
        TableSection(
            id="sensitivity",
            title="Sensitivity Analysis",
            html=html,
        )
    )


def _add_interpretation_intro(report: ReportBuilder) -> None:
    """Text block: How to read this report."""
    report.add(
        TextSection(
            id="interpretation-intro",
            title="How to Read This Report",
            html=(
                "<p>This report presents the results of clustering analysis on Kansas "
                "Legislature voting data. Clustering identifies discrete voting blocs "
                "(factions) among legislators using multiple methods for robustness.</p>"
                "<p><strong>Key finding:</strong> k=2 clusters emerged as optimal for both "
                "chambers, corresponding exactly to the party split (all Republicans in one "
                "cluster, all Democrats in the other). The expected k=3 structure "
                "(conservative Rs, moderate Rs, Democrats) was not supported — the "
                "moderate/conservative Republican distinction is continuous, not discrete.</p>"
                "<p>The report is organized as: (1) data summary, (2) party loyalty metric, "
                "(3) dendrograms, (4) model selection, (5) cluster assignments and "
                "characterization, (6) cross-method validation, (7) within-party clustering "
                "to test for finer structure, and (8) veto override subgroup.</p>"
                "<p>Each section includes figures and/or tables. Interpretation guidance "
                "blocks appear between sections to help readers understand what the "
                "results mean.</p>"
            ),
        )
    )


def _add_model_selection_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Text block: Interpreting model selection plots."""
    # Build chamber-specific details
    chamber_details = []
    for chamber, result in results.items():
        km = result.get("kmeans", {})
        km_results = km.get("results", {})
        sil_k2 = km_results.get(2, {}).get("silhouette_1d")
        sil_k3 = km_results.get(3, {}).get("silhouette_1d")
        if sil_k2 is not None and sil_k3 is not None:
            drop = sil_k2 - sil_k3
            chamber_details.append(
                f"<li><strong>{chamber}:</strong> silhouette drops from "
                f"{sil_k2:.2f} (k=2) to {sil_k3:.2f} (k=3) — a decrease of "
                f"{drop:.2f}. This is a substantial drop, not a marginal one.</li>"
            )

    details_html = "<ul>" + "".join(chamber_details) + "</ul>" if chamber_details else ""

    report.add(
        TextSection(
            id="model-sel-interpretation",
            title="Interpreting Model Selection",
            html=(
                "<p><strong>Silhouette score</strong> measures how well-separated clusters "
                "are. The scale:</p>"
                "<ul>"
                "<li>&gt; 0.70: Strong structure — clusters are well-separated</li>"
                "<li>&gt; 0.50: Good structure — meaningful clusters exist</li>"
                "<li>0.25–0.50: Weak structure — clusters overlap substantially</li>"
                "<li>&lt; 0.25: No structure — data is not naturally clustered</li>"
                "</ul>"
                "<p>The <strong>elbow plot</strong> (inertia) shows the rate of decrease in "
                'within-cluster variance. A visible "elbow" suggests the right k, but '
                "silhouette is the primary decision metric because it accounts for "
                "between-cluster separation, not just within-cluster tightness.</p>"
                f"{details_html}"
                "<p>Forcing k=3 would split a continuous Republican distribution at an "
                "arbitrary point, modeling noise rather than genuine factional boundaries. "
                "The silhouette drop from k=2 to k=3 confirms this — it's not marginal.</p>"
                "<p><strong>GMM BIC</strong> selects the number of Gaussian components that "
                "best fits the data distribution. BIC's k=4 reflects the distributional "
                "shape (long right tail, bimodal D/R peaks) rather than discrete factions. "
                "Silhouette and BIC measure different things; silhouette measures cluster "
                "separation, while BIC measures generative model fit.</p>"
            ),
        )
    )


def _add_cross_method_interpretation(report: ReportBuilder) -> None:
    """Text block: What cross-method agreement means."""
    report.add(
        TextSection(
            id="cross-method-interpretation",
            title="What Cross-Method Agreement Means",
            html=(
                "<p>The Adjusted Rand Index (ARI) measures agreement between two sets of "
                "cluster labels, corrected for chance. ARI = 1.0 means identical "
                "assignments; ARI = 0 means no better than random.</p>"
                "<p>High ARI across three fundamentally different methods — hierarchical "
                "(agglomerative, based on pairwise agreement), k-means (centroid-based, on "
                "IRT), and GMM (probabilistic, on IRT) — confirms that the 2-cluster "
                "structure is a real property of the data, not an artifact of any "
                "particular algorithm or input representation.</p>"
            ),
        )
    )


def _add_within_party_intro(report: ReportBuilder) -> None:
    """Text block: Why within-party clustering."""
    report.add(
        TextSection(
            id="within-party-intro",
            title="Within-Party Clustering",
            html=(
                "<p>The whole-chamber clustering found k=2 (party split) as optimal because "
                "the Democrat–Republican gap is so large that it overwhelms any finer "
                "intra-party variation. To search for structure <em>within</em> each party "
                "(e.g., conservative vs. moderate Republicans), we cluster each party "
                "caucus separately, removing the cross-party gap from the analysis.</p>"
                "<p>Within-party clustering uses k-means on two feature spaces: 1D (IRT "
                "ideal point only) and 2D (IRT ideal point + party loyalty rate). If the "
                "best within-party silhouette is below 0.50 for all k &gt; 1, we conclude "
                "that the variation is continuous, not discrete — there are no clean "
                '"subclusters" within the party.</p>'
                "<p><strong>This is a valid and important finding.</strong> Continuous "
                "variation means legislators are spread across a spectrum rather than "
                "forming distinct factions. This affects downstream analyses: network "
                "analysis may reveal gradients rather than communities, and prediction "
                "models should use continuous features (IRT, loyalty) rather than "
                "discrete cluster labels.</p>"
            ),
        )
    )


def _add_within_party_model_selection_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
    party_key: str,
) -> None:
    """Figure: Within-party silhouette vs k."""
    path = plots_dir / f"within_party_model_sel_{party_key}_{chamber.lower()}.png"
    if path.exists():
        party_label = party_key.title()
        report.add(
            FigureSection.from_file(
                f"fig-wp-model-sel-{party_key}-{chamber.lower()}",
                f"{chamber} Within-{party_label} Model Selection",
                path,
                caption=(
                    f"Within-{party_label} silhouette scores ({chamber}): 1D (IRT only) "
                    f"and 2D (IRT + loyalty) vs k. Dashed line = 0.50 good-structure "
                    f"threshold."
                ),
            )
        )


def _add_within_party_clusters_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
    party_key: str,
) -> None:
    """Figure: Within-party 2D scatter."""
    path = plots_dir / f"within_party_clusters_{party_key}_{chamber.lower()}.png"
    if path.exists():
        party_label = party_key.title()
        report.add(
            FigureSection.from_file(
                f"fig-wp-clusters-{party_key}-{chamber.lower()}",
                f"{chamber} Within-{party_label} Clusters",
                path,
                caption=(
                    f"Within-{party_label} clustering ({chamber}): IRT ideal point (x) "
                    f"vs party loyalty (y), colored by within-party cluster. A red "
                    f"banner indicates no discrete subclusters were found."
                ),
            )
        )


def _add_within_party_summary_table(
    report: ReportBuilder,
    results: dict[str, dict],
) -> None:
    """Table: Within-party clustering summary per chamber and party."""
    rows = []
    for chamber, result in results.items():
        wp = result.get("within_party", {})
        for party_key in ["republican", "democrat"]:
            pd = wp.get(party_key, {})
            if not isinstance(pd, dict):
                continue
            if pd.get("skipped"):
                rows.append(
                    {
                        "Chamber": chamber,
                        "Party": party_key.title(),
                        "N Legislators": pd.get("n_legislators", 0),
                        "Optimal k (1D)": "N/A",
                        "Best Silhouette (1D)": None,
                        "Optimal k (2D)": "N/A",
                        "Best Silhouette (2D)": None,
                        "Structure Found": "Skipped (too small)",
                    }
                )
            else:
                struct = "Yes" if pd.get("structure_found") else "No — continuous"
                rows.append(
                    {
                        "Chamber": chamber,
                        "Party": party_key.title(),
                        "N Legislators": pd.get("n_legislators", 0),
                        "Optimal k (1D)": pd.get("optimal_k_1d"),
                        "Best Silhouette (1D)": pd.get("best_silhouette_1d"),
                        "Optimal k (2D)": pd.get("optimal_k_2d"),
                        "Best Silhouette (2D)": pd.get("best_silhouette_2d"),
                        "Structure Found": struct,
                    }
                )

    if not rows:
        return

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Within-Party Clustering Summary",
        subtitle="Per-party k-means results after removing the cross-party gap",
        number_formats={
            "Best Silhouette (1D)": ".3f",
            "Best Silhouette (2D)": ".3f",
        },
        source_note=(
            "Structure found if best silhouette >= 0.50. "
            '"No — continuous" means intra-party variation is a spectrum, not factions.'
        ),
    )
    report.add(
        TableSection(
            id="within-party-summary",
            title="Within-Party Summary",
            html=html,
        )
    )


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: All constants and settings used in this run."""
    try:
        from analysis.clustering import (
            CLUSTER_CMAP,
            CONTESTED_PARTY_THRESHOLD,
            COPHENETIC_THRESHOLD,
            DEFAULT_K,
            GMM_COVARIANCE,
            GMM_N_INIT,
            K_RANGE,
            LINKAGE_METHOD,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            RANDOM_SEED,
            SENSITIVITY_THRESHOLD,
            SILHOUETTE_GOOD,
            WITHIN_PARTY_MIN_SIZE,
        )
    except ModuleNotFoundError:
        from clustering import (  # type: ignore[no-redef]
            CLUSTER_CMAP,
            CONTESTED_PARTY_THRESHOLD,
            COPHENETIC_THRESHOLD,
            DEFAULT_K,
            GMM_COVARIANCE,
            GMM_N_INIT,
            K_RANGE,
            LINKAGE_METHOD,
            MIN_VOTES,
            MINORITY_THRESHOLD,
            RANDOM_SEED,
            SENSITIVITY_THRESHOLD,
            SILHOUETTE_GOOD,
            WITHIN_PARTY_MIN_SIZE,
        )

    df = pl.DataFrame(
        {
            "Parameter": [
                "Random Seed",
                "K Range",
                "Default K",
                "Linkage Method",
                "Cophenetic Threshold",
                "Silhouette 'Good' Threshold",
                "GMM Covariance Type",
                "GMM N Initializations",
                "Cluster Colormap",
                "Minority Threshold (Default)",
                "Minority Threshold (Sensitivity)",
                "Min Substantive Votes",
                "Contested Party Threshold",
                "Within-Party Min Size",
            ],
            "Value": [
                str(RANDOM_SEED),
                str(list(K_RANGE)),
                str(DEFAULT_K),
                LINKAGE_METHOD,
                str(COPHENETIC_THRESHOLD),
                str(SILHOUETTE_GOOD),
                GMM_COVARIANCE,
                str(GMM_N_INIT),
                CLUSTER_CMAP,
                f"{MINORITY_THRESHOLD:.3f} ({MINORITY_THRESHOLD * 100:.1f}%)",
                f"{SENSITIVITY_THRESHOLD:.2f} ({SENSITIVITY_THRESHOLD * 100:.0f}%)",
                str(MIN_VOTES),
                f"{CONTESTED_PARTY_THRESHOLD:.2f} ({CONTESTED_PARTY_THRESHOLD * 100:.0f}%)",
                str(WITHIN_PARTY_MIN_SIZE),
            ],
            "Description": [
                "For reproducible k-means/GMM initialization",
                "Range of k values evaluated for model selection",
                "Expected optimal k (conservative R, moderate R, Democrat)",
                "Ward minimizes within-cluster variance",
                "Minimum cophenetic correlation for valid dendrogram",
                "Silhouette > this indicates good cluster structure",
                "Full covariance allows elliptical clusters",
                "Multiple GMM restarts for stability",
                "Matplotlib colormap for cluster visualization",
                "Inherited from EDA; votes with minority < this are filtered",
                "Alternative threshold for sensitivity analysis",
                "Inherited from EDA; legislators with < this filtered",
                "A vote is contested for a party if >= this fraction dissents",
                "Minimum caucus size for within-party clustering",
            ],
        }
    )
    html = make_gt(
        df,
        title="Analysis Parameters",
        source_note="See analysis/design/clustering.md for justification.",
    )
    report.add(
        TableSection(
            id="analysis-params",
            title="Analysis Parameters",
            html=html,
        )
    )
