"""Network-specific HTML report builder.

Builds ~29 sections (tables, figures, and text) for the network analysis report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from network.py):
    from analysis.network_report import build_network_report
    build_network_report(ctx.report, results=results, ...)
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


def build_network_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    kappa_threshold: float = 0.40,
    skip_cross_chamber: bool = False,
    skip_high_disc: bool = False,
) -> None:
    """Build the full network HTML report by adding sections to the ReportBuilder."""
    chambers = [c for c in ["House", "Senate"] if c in results and "summary" in results[c]]

    _add_data_summary(report, results, chambers)
    _add_how_to_read(report)

    # Per-chamber core sections
    for chamber in chambers:
        _add_network_summary(report, results[chamber], chamber)

    for chamber in chambers:
        _add_edge_weight_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_network_party_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_network_irt_figure(report, plots_dir, chamber)

    # Centrality
    for chamber in chambers:
        _add_centrality_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_bridge_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_party_centrality_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_centrality_scatter_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_centrality_vs_irt_figure(report, plots_dir, chamber, results[chamber])

    for chamber in chambers:
        _add_centrality_ranking_figure(report, plots_dir, chamber)

    _add_centrality_interpretation(report, results, chambers, kappa_threshold)

    # Community detection
    for chamber in chambers:
        _add_multi_resolution_table(report, results[chamber], chamber)

    for chamber in chambers:
        _add_multi_resolution_figure(report, plots_dir, chamber)

    for chamber in chambers:
        _add_community_composition(report, results[chamber], chamber)

    _add_community_vs_party(report, results, chambers)
    _add_community_vs_clusters(report, results, chambers)

    for chamber in chambers:
        _add_community_network_figure(report, plots_dir, chamber)

    _add_community_interpretation(report)

    # Within-party
    for chamber in chambers:
        _add_within_party_communities(report, results[chamber], chamber)

    _add_extreme_edge_analysis(report, results, chambers)

    # Threshold sensitivity
    for chamber in chambers:
        _add_threshold_sweep_figure(report, plots_dir, chamber)

    # High-disc subnetwork
    if not skip_high_disc:
        for chamber in chambers:
            _add_high_disc_summary(report, results[chamber], chamber)
        for chamber in chambers:
            _add_high_disc_figure(report, plots_dir, chamber)

    # Veto override
    for chamber in chambers:
        _add_veto_override(report, results[chamber], chamber)

    # Cross-chamber
    if not skip_cross_chamber and "cross_chamber" in results:
        _add_cross_chamber_figure(report, plots_dir)

    # Flagged and downstream
    _add_flagged_legislators(report, results, chambers)
    _add_downstream_findings(report, results, chambers, kappa_threshold)
    _add_analysis_parameters(report, kappa_threshold, skip_cross_chamber, skip_high_disc)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ─────────────────────────────────────────────────


def _add_data_summary(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    rows = []
    for chamber in chambers:
        r = results[chamber]
        ip = r["ideal_points"]
        vm = r["vote_matrix"]
        s = r.get("summary", {})
        rows.append(
            {
                "Chamber": chamber,
                "N Legislators": ip.height,
                "N Votes (filtered)": len(vm.columns) - 1,
                "N Edges": s.get("n_edges", 0),
                "Density": s.get("density", 0.0),
            }
        )
    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Network Data Summary",
        subtitle="Upstream data dimensions and graph statistics per chamber",
        number_formats={"Density": ".4f"},
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_how_to_read(report: ReportBuilder) -> None:
    html = """
    <p>This report presents the network analysis of Kansas legislative co-voting
    patterns. Key concepts:</p>
    <ul>
    <li><strong>Edge weight (Kappa):</strong> Cohen's Kappa agreement between two
    legislators, corrected for the 82% Yea base rate. Higher = more similar voting.</li>
    <li><strong>Threshold:</strong> Only pairs with Kappa above the threshold are
    connected by an edge. Default 0.40 = "substantial" agreement.</li>
    <li><strong>Centrality:</strong> Measures of structural importance. High betweenness
    = legislator lies on many shortest paths between others (a "bridge").</li>
    <li><strong>Community:</strong> Louvain algorithm finds groups with dense internal
    connections. Compared to party labels and clustering assignments.</li>
    <li><strong>Resolution:</strong> Louvain parameter controlling granularity.
    Low resolution = fewer communities; high = more communities.</li>
    </ul>
    """
    report.add(TextSection(id="how-to-read", title="How to Read This Report", html=html))


def _add_network_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    s = result.get("summary", {})
    rows = [
        {"Metric": "Nodes", "Value": str(s.get("n_nodes", 0))},
        {"Metric": "Edges", "Value": str(s.get("n_edges", 0))},
        {"Metric": "Density", "Value": f"{s.get('density', 0):.4f}"},
        {"Metric": "Avg Clustering Coeff", "Value": f"{s.get('avg_clustering', 0):.4f}"},
        {"Metric": "Transitivity", "Value": f"{s.get('transitivity', 0):.4f}"},
        {"Metric": "Connected Components", "Value": str(s.get("n_components", 0))},
    ]
    if s.get("assortativity_party") is not None:
        rows.append({"Metric": "Party Assortativity", "Value": f"{s['assortativity_party']:.4f}"})
    df = pl.DataFrame(rows)
    html = make_gt(df, title=f"{chamber} — Network Summary")
    report.add(
        TableSection(
            id=f"network-summary-{chamber.lower()}",
            title=f"{chamber} Network Summary",
            html=html,
        )
    )


def _add_edge_weight_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"edge_weights_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-edge-weights-{chamber.lower()}",
                f"{chamber} Edge Weight Distribution",
                path,
                caption=(
                    f"Distribution of edge weights (Kappa) for {chamber}, separated by "
                    "within-party (R-R, D-D) and cross-party edges. "
                    "Blue dashed line marks the default threshold."
                ),
            )
        )


def _add_network_party_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"network_party_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-network-party-{chamber.lower()}",
                f"{chamber} Network (Party Colors)",
                path,
                caption=(
                    f"Spring layout of {chamber} co-voting network. Node color = party. "
                    "Node size proportional to degree. Red-ringed nodes = top 3 bridge "
                    "legislators (highest betweenness centrality)."
                ),
            )
        )


def _add_network_irt_figure(report: ReportBuilder, plots_dir: Path, chamber: str) -> None:
    path = plots_dir / f"network_irt_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-network-irt-{chamber.lower()}",
                f"{chamber} Network (IRT Gradient)",
                path,
                caption=(
                    "Same layout as party plot, but node color = IRT ideal point "
                    "(blue = liberal, red = conservative)."
                ),
            )
        )


def _add_centrality_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    centralities = result.get("centralities")
    if centralities is None or centralities.height == 0:
        return

    df = centralities.sort("betweenness", descending=True)
    display_cols = [
        "full_name",
        "party",
        "xi_mean",
        "degree",
        "weighted_degree",
        "betweenness",
        "eigenvector",
        "closeness",
        "pagerank",
    ]
    available = [c for c in display_cols if c in df.columns]
    df = df.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Centrality Measures ({df.height} legislators)",
        subtitle="Sorted by betweenness centrality (descending)",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "xi_mean": "Ideal Point",
            "degree": "Degree",
            "weighted_degree": "W. Degree",
            "betweenness": "Betweenness",
            "eigenvector": "Eigenvector",
            "closeness": "Closeness",
            "pagerank": "PageRank",
        },
        number_formats={
            "xi_mean": ".3f",
            "degree": ".4f",
            "weighted_degree": ".2f",
            "betweenness": ".4f",
            "eigenvector": ".4f",
            "closeness": ".4f",
            "pagerank": ".4f",
        },
    )
    report.add(
        TableSection(
            id=f"centrality-table-{chamber.lower()}",
            title=f"{chamber} Centrality Table",
            html=html,
        )
    )


def _add_bridge_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    bridges = result.get("bridges")
    if bridges is None or bridges.height == 0:
        return

    display_cols = [
        "full_name",
        "party",
        "xi_mean",
        "betweenness",
        "cross_party_edges",
        "total_edges",
    ]
    available = [c for c in display_cols if c in bridges.columns]
    df = bridges.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Bridge Legislators (Top {df.height})",
        subtitle="Highest betweenness centrality with cross-party edge counts",
        column_labels={
            "full_name": "Legislator",
            "party": "Party",
            "xi_mean": "Ideal Point",
            "betweenness": "Betweenness",
            "cross_party_edges": "Cross-Party Edges",
            "total_edges": "Total Edges",
        },
        number_formats={"xi_mean": ".3f", "betweenness": ".4f"},
    )
    report.add(
        TableSection(
            id=f"bridge-{chamber.lower()}",
            title=f"{chamber} Bridge Legislators",
            html=html,
        )
    )


def _add_party_centrality_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    party_cent = result.get("party_centrality")
    if party_cent is None or party_cent.height == 0:
        return

    display_cols = [
        "party",
        "n",
        "betweenness_mean",
        "betweenness_median",
        "eigenvector_mean",
        "eigenvector_median",
        "pagerank_mean",
        "pagerank_median",
    ]
    available = [c for c in display_cols if c in party_cent.columns]
    df = party_cent.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Centrality by Party",
        subtitle="Mean and median centrality measures per party",
        column_labels={
            "party": "Party",
            "n": "N",
            "betweenness_mean": "Between. Mean",
            "betweenness_median": "Between. Median",
            "eigenvector_mean": "Eigenvec. Mean",
            "eigenvector_median": "Eigenvec. Median",
            "pagerank_mean": "PageRank Mean",
            "pagerank_median": "PageRank Median",
        },
        number_formats={
            "betweenness_mean": ".4f",
            "betweenness_median": ".4f",
            "eigenvector_mean": ".4f",
            "eigenvector_median": ".4f",
            "pagerank_mean": ".4f",
            "pagerank_median": ".4f",
        },
    )
    report.add(
        TableSection(
            id=f"party-centrality-{chamber.lower()}",
            title=f"{chamber} Centrality by Party",
            html=html,
        )
    )


def _add_centrality_scatter_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"centrality_scatter_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-centrality-scatter-{chamber.lower()}",
                f"{chamber} Centrality Scatter",
                path,
                caption=f"Betweenness vs eigenvector centrality for {chamber}.",
            )
        )


def _add_centrality_vs_irt_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
    result: dict,
) -> None:
    path = plots_dir / f"centrality_vs_irt_{chamber.lower()}.png"
    if path.exists():
        n_components = result.get("summary", {}).get("n_components", 0)
        if n_components >= 2:
            qualifier = "With disconnected party components, betweenness is within-party only."
        else:
            qualifier = "Graph is connected; betweenness reflects cross-party structure."
        report.add(
            FigureSection.from_file(
                f"fig-centrality-irt-{chamber.lower()}",
                f"{chamber} Centrality vs IRT",
                path,
                caption=f"IRT ideal point vs betweenness centrality for {chamber}. {qualifier}",
            )
        )


def _add_centrality_ranking_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"centrality_ranking_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-centrality-ranking-{chamber.lower()}",
                f"{chamber} Centrality Ranking",
                path,
                caption=(
                    f"All {chamber} legislators ranked by betweenness centrality. "
                    "Higher betweenness = more influence as a connector between "
                    "otherwise-separate voting blocs."
                ),
            )
        )


def _add_centrality_interpretation(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
    kappa_threshold: float,
) -> None:
    # Determine if any chamber has cross-party edges
    all_disconnected = all(
        results[c].get("summary", {}).get("n_components", 1) >= 2 for c in chambers
    )

    if all_disconnected:
        context = (
            f"<p><em>Important:</em> At Kappa threshold {kappa_threshold}, there are zero "
            "cross-party edges. The graph is two disconnected components (one per party). All "
            "centrality measures below are <strong>within-party only</strong> — they measure "
            "structural importance within a legislator's own caucus, not bipartisan bridging.</p>"
        )
    else:
        context = (
            f"<p>At Kappa threshold {kappa_threshold}, some chambers have cross-party edges. "
            "Centrality measures reflect both within-party and cross-party structure.</p>"
        )

    html = f"""
    <p><strong>Centrality interpretation guide:</strong></p>
    {context}
    <ul>
    <li><strong>Betweenness:</strong> Legislators on many shortest paths between
    others. High betweenness = central hub or bridge. Low betweenness + few
    edges = isolated within own caucus.</li>
    <li><strong>Eigenvector:</strong> Legislators connected to other well-connected
    legislators. Identifies core members vs. periphery.</li>
    <li><strong>PageRank:</strong> Similar to eigenvector but more robust to disconnected
    graphs. Useful when the network has multiple components.</li>
    <li><strong>Degree:</strong> Count of legislators with whom voting agreement
    exceeds the Kappa threshold.</li>
    </ul>
    """
    report.add(
        TextSection(id="centrality-interpretation", title="Centrality Interpretation", html=html)
    )


def _add_multi_resolution_table(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    res_df = result.get("resolution_df")
    if res_df is None or res_df.height == 0:
        return

    html = make_gt(
        res_df,
        title=f"{chamber} — Multi-Resolution Louvain",
        subtitle="Community count and modularity at each resolution parameter",
        column_labels={
            "resolution": "Resolution",
            "n_communities": "N Communities",
            "modularity": "Modularity",
        },
        number_formats={"resolution": ".2f", "modularity": ".4f"},
    )
    report.add(
        TableSection(
            id=f"multi-res-{chamber.lower()}",
            title=f"{chamber} Multi-Resolution Table",
            html=html,
        )
    )


def _add_multi_resolution_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"multi_resolution_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-multi-res-{chamber.lower()}",
                f"{chamber} Multi-Resolution",
                path,
                caption=(
                    f"Number of communities and modularity vs Louvain resolution for {chamber}."
                ),
            )
        )


def _add_community_composition(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    comp = result.get("community_composition")
    if comp is None or comp.height == 0:
        return

    display_cols = [
        "community",
        "n_legislators",
        "n_republican",
        "n_democrat",
        "pct_republican",
        "mean_xi",
        "std_xi",
        "mean_loyalty",
    ]
    available = [c for c in display_cols if c in comp.columns]
    df = comp.select(available)

    html = make_gt(
        df,
        title=f"{chamber} — Community Composition",
        subtitle="Party breakdown and IRT summary per community (best resolution)",
        column_labels={
            "community": "Community",
            "n_legislators": "N",
            "n_republican": "N Rep",
            "n_democrat": "N Dem",
            "pct_republican": "% Rep",
            "mean_xi": "Mean XI",
            "std_xi": "SD XI",
            "mean_loyalty": "Mean Loyalty",
        },
        number_formats={
            "pct_republican": ".1f",
            "mean_xi": ".3f",
            "std_xi": ".3f",
            "mean_loyalty": ".3f",
        },
    )
    report.add(
        TableSection(
            id=f"community-comp-{chamber.lower()}",
            title=f"{chamber} Community Composition",
            html=html,
        )
    )


def _add_community_vs_party(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    rows = []
    for chamber in chambers:
        cvp = results[chamber].get("community_vs_party", {})
        rows.append(
            {
                "Chamber": chamber,
                "NMI": cvp.get("nmi"),
                "ARI": cvp.get("ari"),
                "N Misclassified": len(cvp.get("misclassified", [])),
            }
        )
    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Communities vs Party Labels",
        subtitle="NMI and ARI comparing Louvain communities to party membership",
        number_formats={"NMI": ".4f", "ARI": ".4f"},
    )
    report.add(
        TableSection(id="community-vs-party", title="Community vs Party (NMI/ARI)", html=html)
    )


def _add_community_vs_clusters(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    rows = []
    for chamber in chambers:
        cvc = results[chamber].get("community_vs_clusters", {})
        rows.append(
            {
                "Chamber": chamber,
                "NMI": cvc.get("nmi"),
                "ARI": cvc.get("ari"),
            }
        )
    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Communities vs Clustering Assignments",
        subtitle="NMI and ARI comparing Louvain communities to k-means clusters",
        number_formats={"NMI": ".4f", "ARI": ".4f"},
    )
    report.add(
        TableSection(
            id="community-vs-clusters",
            title="Community vs Clustering (NMI/ARI)",
            html=html,
        )
    )


def _add_community_network_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"community_network_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-community-{chamber.lower()}",
                f"{chamber} Community Network",
                path,
                caption=(
                    f"Side-by-side: party coloring (left) vs Louvain community coloring "
                    f"(right) for {chamber}."
                ),
            )
        )


def _add_community_interpretation(report: ReportBuilder) -> None:
    html = """
    <p><strong>Community detection interpretation:</strong></p>
    <ul>
    <li>If NMI ≈ 1.0 at low resolution, communities match party perfectly —
    party is the dominant network structure.</li>
    <li>If higher resolutions split within-party groups with modularity > 0.3,
    those subcommunities have meaningful internal cohesion.</li>
    <li>Misclassified legislators (party ≠ community majority) are the most
    analytically interesting — they vote more like the opposite party.</li>
    <li>ARI > 0.7 between communities and k-means clusters confirms that
    network-based and centroid-based groupings agree.</li>
    </ul>
    """
    report.add(
        TextSection(
            id="community-interpretation", title="Community Detection Interpretation", html=html
        )
    )


def _add_within_party_communities(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    wp = result.get("within_party", {})
    rows = []
    for party_key in ["republican", "democrat"]:
        pd = wp.get(party_key, {})
        if pd.get("skipped", True):
            rows.append(
                {
                    "Chamber": chamber,
                    "Party": party_key.title(),
                    "Best Resolution": None,
                    "N Communities": None,
                    "Modularity": None,
                    "Note": pd.get("reason", "Skipped"),
                }
            )
        else:
            rows.append(
                {
                    "Chamber": chamber,
                    "Party": party_key.title(),
                    "Best Resolution": pd.get("best_resolution"),
                    "N Communities": pd.get("n_communities"),
                    "Modularity": pd.get("best_modularity"),
                    "Note": "",
                }
            )

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title=f"{chamber} — Within-Party Communities",
            subtitle="Louvain community detection run separately on each party caucus",
            number_formats={"Best Resolution": ".2f", "Modularity": ".4f"},
        )
        report.add(
            TableSection(
                id=f"within-party-{chamber.lower()}",
                title=f"{chamber} Within-Party Communities",
                html=html,
            )
        )


def _add_extreme_edge_analysis(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    for chamber in chambers:
        ee = results[chamber].get("extreme_edge_weights")
        if ee is None:
            continue

        maj = ee.get("majority_party", "Majority")
        rows = []
        for leg in ee.get("legislators", []):
            rows.append(
                {
                    "Legislator": leg["name"],
                    "Ideal Point": leg.get("xi_mean"),
                    f"N {maj[0]}-{maj[0]} Edges": leg["n_r_edges"],
                    f"Mean {maj[0]} Weight": leg["mean_r_weight"],
                    f"Median {maj[0]} Weight": leg["median_r_weight"],
                    f"Min {maj[0]} Weight": leg["min_r_weight"],
                    f"vs {maj} Median": leg["vs_r_median"],
                }
            )

        if rows:
            df = pl.DataFrame(rows)
            source = (
                f"{maj} median edge weight: {ee['r_median_edge_weight']:.4f} "
                f"(mean: {ee['r_mean_edge_weight']:.4f}, N edges: {ee['r_n_edges']})"
            )
            html = make_gt(
                df,
                title=f"{chamber} — Most Extreme {maj} Edge Weights",
                subtitle=(
                    f"Within-{maj} edge weights for the most ideologically extreme legislators"
                ),
                number_formats={
                    "Ideal Point": ".3f",
                    f"Mean {maj[0]} Weight": ".4f",
                    f"Median {maj[0]} Weight": ".4f",
                    f"Min {maj[0]} Weight": ".4f",
                    f"vs {maj} Median": ".4f",
                },
                source_note=source,
            )
            report.add(
                TableSection(
                    id=f"extreme-edge-{chamber.lower()}",
                    title=f"{chamber} Extreme Edge Weights",
                    html=html,
                )
            )


def _add_threshold_sweep_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"threshold_sweep_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-threshold-{chamber.lower()}",
                f"{chamber} Threshold Sensitivity",
                path,
                caption=(
                    f"Network statistics at different Kappa thresholds for {chamber}. "
                    "Blue dashed line = default threshold; red dashed line = party split point. "
                    "Green shading = range where the number of groups stays constant."
                ),
            )
        )


def _add_high_disc_summary(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    hd = result.get("high_disc_summary")
    if hd is None:
        return

    rows = [
        {"Metric": "Nodes", "Value": str(hd.get("n_nodes", 0))},
        {"Metric": "Edges", "Value": str(hd.get("n_edges", 0))},
        {"Metric": "Density", "Value": f"{hd.get('density', 0):.4f}"},
        {"Metric": "Components", "Value": str(hd.get("n_components", 0))},
        {"Metric": "Avg Clustering", "Value": f"{hd.get('avg_clustering', 0):.4f}"},
    ]
    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — High-Discrimination Subnetwork",
        subtitle="Network built from |beta| > 1.5 bills only",
    )
    report.add(
        TableSection(
            id=f"high-disc-{chamber.lower()}",
            title=f"{chamber} High-Disc Summary",
            html=html,
        )
    )


def _add_high_disc_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    path = plots_dir / f"high_disc_network_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                f"fig-high-disc-{chamber.lower()}",
                f"{chamber} High-Disc Network",
                path,
                caption=(
                    f"Network built from high-discrimination bills (|beta| > 1.5) for {chamber}."
                ),
            )
        )


def _add_veto_override(
    report: ReportBuilder,
    result: dict,
    chamber: str,
) -> None:
    vs = result.get("veto_summary")
    if vs is None:
        report.add(
            TextSection(
                id=f"veto-override-{chamber.lower()}",
                title=f"{chamber} Veto Override",
                html=f"<p>{chamber}: Fewer than 5 veto override votes — subnetwork skipped.</p>",
            )
        )
        return

    rows = [
        {"Metric": "Nodes", "Value": str(vs.get("n_nodes", 0))},
        {"Metric": "Edges", "Value": str(vs.get("n_edges", 0))},
        {"Metric": "Density", "Value": f"{vs.get('density', 0):.4f}"},
        {"Metric": "Components", "Value": str(vs.get("n_components", 0))},
    ]
    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} — Veto Override Subnetwork",
        subtitle="Network built from veto override votes only",
    )
    report.add(
        TableSection(
            id=f"veto-override-{chamber.lower()}",
            title=f"{chamber} Veto Override Summary",
            html=html,
        )
    )


def _add_cross_chamber_figure(report: ReportBuilder, plots_dir: Path) -> None:
    path = plots_dir / "cross_chamber_network.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                "fig-cross-chamber",
                "Cross-Chamber Network",
                path,
                caption=(
                    "Combined House and Senate network positioned by equated IRT ideal points. "
                    "Circles = House, squares = Senate. No cross-chamber edges."
                ),
            )
        )


def _add_flagged_legislators(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    rows = []
    for chamber in chambers:
        bridges = results[chamber].get("bridges")
        if bridges is None or bridges.height == 0:
            continue
        # Top 5 bridge legislators per chamber
        top = bridges.head(5)
        for row in top.iter_rows(named=True):
            rows.append(
                {
                    "Chamber": chamber,
                    "Legislator": row["full_name"],
                    "Party": row["party"],
                    "Betweenness": row["betweenness"],
                    "Cross-Party Edges": row.get("cross_party_edges", 0),
                    "Flag": "High within-party betweenness",
                }
            )

        # Misclassified legislators
        cvp = results[chamber].get("community_vs_party", {})
        for m in cvp.get("misclassified", []):
            rows.append(
                {
                    "Chamber": chamber,
                    "Legislator": m["full_name"],
                    "Party": m["party"],
                    "Betweenness": None,
                    "Cross-Party Edges": None,
                    "Flag": f"Community {m['community']} (majority {m['community_majority']})",
                }
            )

    if rows:
        df = pl.DataFrame(rows)
        html = make_gt(
            df,
            title="Flagged Legislators",
            subtitle="High within-party betweenness and community-party mismatches",
            number_formats={"Betweenness": ".4f"},
        )
        report.add(TableSection(id="flagged-legislators", title="Flagged Legislators", html=html))


def _add_downstream_findings(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
    kappa_threshold: float,
) -> None:
    findings = []

    # Check cross-party connectivity
    all_disconnected = all(
        results[c].get("summary", {}).get("n_components", 1) >= 2 for c in chambers
    )
    if all_disconnected:
        findings.append(
            f"<strong>Zero cross-party edges at κ={kappa_threshold}.</strong> The network is "
            "two disconnected party cliques. Community detection, betweenness, and all "
            "centrality measures operate strictly within each party. This adds no information "
            "beyond party membership itself."
        )
    else:
        connected = [
            c for c in chambers if results[c].get("summary", {}).get("n_components", 1) == 1
        ]
        findings.append(
            f"At κ={kappa_threshold}, {', '.join(connected)} "
            f"{'has' if len(connected) == 1 else 'have'} cross-party edges. "
            "Centrality measures in connected chambers reflect genuine bipartisan structure."
        )

    # Check within-party modularity
    max_mod = 0.0
    for c in chambers:
        wp = results[c].get("within_party", {})
        for party_key in ["republican", "democrat"]:
            pd = wp.get(party_key, {})
            if not pd.get("skipped", True):
                max_mod = max(max_mod, pd.get("best_modularity", 0.0))

    if max_mod < 0.1:
        findings.append(
            f"<strong>Within-party modularity ≈ {max_mod:.2f}.</strong> No meaningful subcaucus "
            "structure. Within-party community labels should not be used as prediction features."
        )
    else:
        findings.append(
            f"<strong>Within-party modularity up to {max_mod:.2f}.</strong> "
            "Some subcaucus structure detected; within-party community labels may be useful."
        )

    findings.append(
        "Within-party centrality measures (betweenness, eigenvector) capture "
        "<em>intra-caucus</em> structural position and may provide marginal predictive "
        "value over IRT ideal points alone — particularly for identifying legislators "
        "who are central vs. peripheral within their party."
    )

    items = "\n".join(f"    <li>{f}</li>" for f in findings)
    html = f"""
    <p><strong>Key findings for downstream analysis (Prediction phase):</strong></p>
    <ul>
{items}
    </ul>
    """
    report.add(TextSection(id="downstream-findings", title="Downstream Findings", html=html))


def _add_analysis_parameters(
    report: ReportBuilder,
    kappa_threshold: float,
    skip_cross_chamber: bool,
    skip_high_disc: bool,
) -> None:
    rows = [
        {"Parameter": "Kappa Threshold", "Value": str(kappa_threshold)},
        {"Parameter": "Louvain Resolutions", "Value": "0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0"},
        {"Parameter": "High-Disc Beta Threshold", "Value": "1.5"},
        {"Parameter": "Top Bridge N", "Value": "15"},
        {"Parameter": "Random Seed", "Value": "42"},
        {"Parameter": "Skip Cross-Chamber", "Value": str(skip_cross_chamber)},
        {"Parameter": "Skip High-Disc", "Value": str(skip_high_disc)},
    ]
    df = pl.DataFrame(rows)
    html = make_gt(df, title="Analysis Parameters")
    report.add(TableSection(id="analysis-params", title="Analysis Parameters", html=html))
