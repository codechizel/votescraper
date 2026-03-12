"""Bipartite network-specific HTML report builder.

Builds ~20 sections (tables, figures, and text) for the bipartite network report.
Each section is a small function that slices/aggregates polars DataFrames and calls
make_gt() or FigureSection.from_file().

Usage (called from bipartite.py):
    from analysis.bipartite_report import build_bipartite_report
    build_bipartite_report(ctx.report, results=results, plots_dir=plots_dir)
"""

from pathlib import Path

import networkx as nx
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


def build_bipartite_report(
    report: ReportBuilder,
    *,
    results: dict[str, dict],
    plots_dir: Path,
    skip_phase6: bool = False,
) -> None:
    """Build the full bipartite network HTML report."""
    chambers = [c for c in ["House", "Senate"] if c in results and "summary" in results[c]]

    findings = _generate_bipartite_key_findings(results)
    if findings:
        report.add(KeyFindingsSection(findings=findings))

    _add_data_summary(report, results, chambers)
    _add_how_to_read(report)

    for chamber in chambers:
        _add_bipartite_summary(report, results[chamber], chamber)
        _add_degree_dist_figure(report, plots_dir, chamber)
        _add_polarization_table(report, results[chamber], chamber)
        _add_polarization_figure(report, plots_dir, chamber)
        _add_bridge_bills_table(report, results[chamber], chamber)
        _add_bridge_vs_beta_figure(report, plots_dir, chamber)
        _add_bill_communities(report, results[chamber], chamber)
        _add_bill_community_sweep(report, results[chamber], chamber)
        _add_bill_cluster_heatmap_figure(report, plots_dir, chamber)
        _add_backbone_summary(report, results[chamber], chamber)
        _add_backbone_sparsity_caveat(report, results[chamber], chamber)
        _add_backbone_layout_figure(report, plots_dir, chamber)

        if not skip_phase6:
            _add_backbone_comparison(report, results[chamber], chamber, plots_dir)
            _add_hidden_alliances(report, results[chamber], chamber)

        _add_backbone_communities(report, results[chamber], chamber)
        _add_bipartite_layout_figure(report, plots_dir, chamber)

    if not skip_phase6:
        _add_cross_method_summary(report, results, chambers)

    _add_downstream_findings(report, results, chambers)
    _add_analysis_parameters(report)

    print(f"  Report: {len(report._sections)} sections added")


# ── Private section builders ──────────────────────────────────────────────────


def _add_data_summary(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    """Table: Bipartite graph dimensions per chamber."""
    rows = []
    for chamber in chambers:
        s = results[chamber]["summary"]
        rows.append(
            {
                "Chamber": chamber,
                "Legislators": s["n_legislators"],
                "Bills": s["n_bills"],
                "Yea Edges": s["n_edges"],
                "Density": s["density"],
                "Avg Legislator Degree": s["avg_legislator_degree"],
                "Avg Bill Degree": s["avg_bill_degree"],
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Bipartite Graph Summary",
        subtitle="Two-mode graph dimensions per chamber (legislators × bills, Yea = edge)",
    )
    report.add(TableSection(id="data-summary", title="Data Summary", html=html))


def _add_how_to_read(report: ReportBuilder) -> None:
    """Text: How to read this report."""
    html = """<div style="padding: 1em; background: #f8f9fa; border-radius: 6px;">
<h4>How to Read This Report</h4>
<p>This report analyzes the <strong>bipartite</strong> (two-mode) network connecting
legislators to bills via Yea votes. Unlike Phase 6's legislator-only network, this
preserves the bill side — enabling bill-centric questions.</p>
<ul>
<li><strong>Bill Polarization:</strong> |%R_Yea − %D_Yea|. High = party-line, Low = bipartisan.</li>
<li><strong>Bridge Bills:</strong> High bipartite betweenness + low polarization — bills that
connect otherwise separate partisan blocs.</li>
<li><strong>Bill Communities:</strong> Bills grouped by which legislators vote Yea together
(Leiden on Newman-weighted projection). <em>Not</em> grouped by topic.</li>
<li><strong>BiCM Backbone:</strong> Statistically validated legislator co-voting edges.
Maximum-entropy null model preserves degree sequences;
only edges with p &lt; 0.01 are retained.</li>
</ul>
</div>"""
    report.add(TextSection(id="how-to-read", title="How to Read This Report", html=html))


def _add_bipartite_summary(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: Bipartite graph summary statistics."""
    s = chamber_results["summary"]
    rows = [
        {"Metric": "Legislators", "Value": str(s["n_legislators"])},
        {"Metric": "Bills", "Value": str(s["n_bills"])},
        {"Metric": "Yea Edges", "Value": str(s["n_edges"])},
        {"Metric": "Density", "Value": f"{s['density']:.4f}"},
        {"Metric": "Avg Legislator Degree", "Value": str(s["avg_legislator_degree"])},
        {"Metric": "Max Legislator Degree", "Value": str(s["max_legislator_degree"])},
        {"Metric": "Avg Bill Degree", "Value": str(s["avg_bill_degree"])},
        {"Metric": "Max Bill Degree", "Value": str(s["max_bill_degree"])},
    ]
    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} Bipartite Graph",
        subtitle="Graph structure statistics",
    )
    report.add(
        TableSection(
            id=f"bipartite-summary-{chamber.lower()}",
            title=f"{chamber} — Bipartite Graph Summary",
            html=html,
        )
    )


def _add_degree_dist_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Figure: Legislator and bill degree distributions."""
    path = plots_dir / f"degree_dist_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"degree-dist-{chamber.lower()}",
                title=f"{chamber} — Degree Distributions",
                path=path,
                caption=(
                    "Left: how many bills each legislator voted Yea on. "
                    "Right: how many Yea votes each bill received."
                ),
                alt_text=(
                    f"Side-by-side histograms of {chamber} bipartite degree "
                    "distributions. Left panel shows legislator Yea-vote counts; "
                    "right panel shows bill Yea-vote counts."
                ),
            )
        )


def _add_polarization_table(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: ALL bill polarization scores (never truncated)."""
    pol = chamber_results.get("polarization")
    if pol is None or pol.height == 0:
        return

    from analysis.phase_utils import drop_empty_optional_columns

    pol = drop_empty_optional_columns(pol, ["short_title"])

    # Select display columns
    display_cols = [
        "vote_id",
        "bill_number",
        "polarization",
        "pct_r_yea",
        "pct_d_yea",
        "n_r",
        "n_d",
    ]
    if "short_title" in pol.columns:
        display_cols.insert(2, "short_title")
    if "beta_mean" in pol.columns:
        display_cols.append("beta_mean")
    existing = [c for c in display_cols if c in pol.columns]
    df = pol.select(existing)

    html = make_gt(
        df,
        title=f"{chamber} Bill Polarization",
        subtitle=f"All {df.height} bills scored — |%R_Yea − %D_Yea|, sorted by polarization (desc)",
        column_labels={
            "vote_id": "Vote ID",
            "bill_number": "Bill",
            "short_title": "Title",
            "polarization": "Polarization",
            "pct_r_yea": "R %Yea",
            "pct_d_yea": "D %Yea",
            "n_r": "N(R)",
            "n_d": "N(D)",
            "beta_mean": "IRT β",
        },
        number_formats={
            "polarization": ".3f",
            "pct_r_yea": ".3f",
            "pct_d_yea": ".3f",
            "beta_mean": ".3f",
        },
    )
    report.add(
        TableSection(
            id=f"polarization-{chamber.lower()}",
            title=f"{chamber} — Bill Polarization (All Bills)",
            html=html,
            caption="Never truncated — shows all bills with ≥10 Yea+Nay votes.",
        )
    )


def _add_polarization_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Figure: Bill polarization distribution histogram."""
    path = plots_dir / f"polarization_hist_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"polarization-hist-{chamber.lower()}",
                title=f"{chamber} — Polarization Distribution",
                path=path,
                caption=(
                    "0 = both parties vote identically, 1 = perfect party-line vote. "
                    "Bills above 0.8 are highly partisan."
                ),
                alt_text=(
                    f"Histogram of {chamber} bill polarization scores ranging from 0 (bipartisan) "
                    "to 1 (perfect party-line). Distribution shape reveals the prevalence of "
                    "partisan versus bipartisan legislation."
                ),
            )
        )


def _add_bridge_bills_table(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: Top bridge bills by bipartite betweenness."""
    from analysis.phase_utils import drop_empty_optional_columns

    bridge = chamber_results.get("bridge_bills")
    if bridge is None or bridge.height == 0:
        return

    bridge = drop_empty_optional_columns(bridge, ["short_title"])

    display_cols = [
        "vote_id",
        "bill_number",
        "betweenness",
        "polarization",
        "pct_r_yea",
        "pct_d_yea",
        "degree",
        "beta_mean",
    ]
    if "short_title" in bridge.columns:
        display_cols.insert(2, "short_title")
    existing = [c for c in display_cols if c in bridge.columns]
    df = bridge.select(existing)

    html = make_gt(
        df,
        title=f"{chamber} Bridge Bills",
        subtitle=f"Top {df.height} bills by bipartite betweenness centrality",
        column_labels={
            "vote_id": "Vote ID",
            "bill_number": "Bill",
            "short_title": "Title",
            "betweenness": "Betweenness",
            "polarization": "Polarization",
            "pct_r_yea": "R %Yea",
            "pct_d_yea": "D %Yea",
            "degree": "Yea Votes",
            "beta_mean": "IRT β",
        },
        number_formats={
            "betweenness": ".4f",
            "polarization": ".3f",
            "pct_r_yea": ".3f",
            "pct_d_yea": ".3f",
            "beta_mean": ".3f",
        },
        source_note=(
            "Bridge bills connect otherwise separate partisan blocs. "
            "Low polarization + high betweenness = genuine bridge."
        ),
    )
    report.add(
        TableSection(
            id=f"bridge-bills-{chamber.lower()}",
            title=f"{chamber} — Bridge Bills (Top {df.height})",
            html=html,
        )
    )


def _add_bridge_vs_beta_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Figure: Bridge betweenness vs IRT discrimination scatter."""
    path = plots_dir / f"bridge_vs_beta_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"bridge-beta-{chamber.lower()}",
                title=f"{chamber} — Bridge Betweenness vs IRT Discrimination",
                path=path,
                caption=(
                    "Bridge bills should cluster in the upper-left: high betweenness "
                    "(connecting blocs) + low |β| (not discriminating along the "
                    "ideological dimension)."
                ),
                alt_text=(
                    f"Scatter plot of {chamber} bills with bipartite betweenness on the y-axis "
                    "and IRT discrimination (|beta|) on the x-axis. Bridge bills cluster in the "
                    "upper-left with high betweenness and low discrimination."
                ),
            )
        )


def _add_bill_communities(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: Bill community profiles (size + party support)."""
    profiles = chamber_results.get("bill_community_profiles")
    if profiles is None or profiles.height == 0:
        return

    html = make_gt(
        profiles,
        title=f"{chamber} Bill Communities",
        subtitle=(
            f"Leiden on Newman-weighted bill projection "
            f"(resolution={chamber_results.get('best_bill_resolution', '?')})"
        ),
        column_labels={
            "community": "Community",
            "n_bills": "Bills",
            "mean_pct_r_yea": "R %Yea",
            "mean_pct_d_yea": "D %Yea",
            "mean_polarization": "Polarization",
        },
        number_formats={
            "mean_pct_r_yea": ".3f",
            "mean_pct_d_yea": ".3f",
            "mean_polarization": ".3f",
        },
        source_note="Communities group bills by coalition support, not by topic.",
    )
    report.add(
        TableSection(
            id=f"bill-communities-{chamber.lower()}",
            title=f"{chamber} — Bill Community Profiles",
            html=html,
        )
    )

    # Modularity quality gate
    best_modularity = chamber_results.get("best_bill_modularity")
    if best_modularity is not None and best_modularity < 0.10:
        report.add(
            TextSection(
                id=f"bill-community-modularity-{chamber.lower()}",
                title=f"{chamber} Bill Community Structure Note",
                html=(
                    f"<p><strong>Note:</strong> Best modularity for {chamber} bill communities "
                    f"is <strong>{best_modularity:.3f}</strong> (below 0.10), indicating weak "
                    f"community structure. The bill projection is nearly homogeneous — bill "
                    f"communities largely mirror the known party divide rather than revealing "
                    f"independent legislative coalitions.</p>"
                ),
            )
        )


def _add_bill_community_sweep(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: Bill community resolution sweep."""
    sweep = chamber_results.get("bill_sweep")
    if sweep is None or sweep.height == 0:
        return

    html = make_gt(
        sweep,
        title=f"{chamber} Resolution Sweep",
        subtitle="Leiden modularity optimization on bill projection",
        column_labels={
            "resolution": "Resolution",
            "n_communities": "Communities",
            "modularity": "Modularity",
        },
        number_formats={"modularity": ".4f"},
    )
    report.add(
        TableSection(
            id=f"bill-sweep-{chamber.lower()}",
            title=f"{chamber} — Bill Community Resolution Sweep",
            html=html,
        )
    )


def _add_bill_cluster_heatmap_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Figure: Party support heatmap per bill community."""
    path = plots_dir / f"bill_cluster_heatmap_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"bill-heatmap-{chamber.lower()}",
                title=f"{chamber} — Bill Community Party Support",
                path=path,
                caption=(
                    "Each row is a bill community. Colors show mean %Yea by party. "
                    "Divergent rows indicate communities with different partisan support."
                ),
                alt_text=(
                    f"Heatmap of {chamber} bill communities showing mean percent Yea by party "
                    "per community. Divergent color patterns across rows reveal communities with "
                    "different partisan support profiles."
                ),
            )
        )


def _add_backbone_summary(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: BiCM backbone summary statistics."""
    backbone_G = chamber_results.get("backbone_graph")
    if backbone_G is None:
        return

    rows = [
        {"Metric": "Nodes", "Value": str(backbone_G.number_of_nodes())},
        {"Metric": "Edges", "Value": str(backbone_G.number_of_edges())},
    ]

    if backbone_G.number_of_nodes() > 0:
        max_possible = backbone_G.number_of_nodes() * (backbone_G.number_of_nodes() - 1) // 2
        density = backbone_G.number_of_edges() / max_possible if max_possible > 0 else 0
        rows.append({"Metric": "Density", "Value": f"{density:.4f}"})

        comps = nx.number_connected_components(backbone_G)
        rows.append({"Metric": "Components", "Value": str(comps)})

    vs_party = chamber_results.get("backbone_vs_party", {})
    if vs_party.get("nmi") is not None:
        rows.append({"Metric": "NMI vs Party", "Value": f"{vs_party['nmi']:.4f}"})
        rows.append({"Metric": "ARI vs Party", "Value": f"{vs_party['ari']:.4f}"})

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} BiCM Backbone",
        subtitle="Statistically validated co-voting edges (p < 0.01)",
    )
    report.add(
        TableSection(
            id=f"backbone-summary-{chamber.lower()}",
            title=f"{chamber} — BiCM Backbone Summary",
            html=html,
        )
    )


def _add_backbone_sparsity_caveat(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Add caveat when backbone is too sparse for legislator-level analysis."""
    backbone_G = chamber_results.get("backbone_graph")
    if backbone_G is None or backbone_G.number_of_nodes() == 0:
        return

    n_nodes = backbone_G.number_of_nodes()
    n_isolated = sum(1 for n in backbone_G.nodes() if backbone_G.degree(n) == 0)
    isolated_frac = n_isolated / n_nodes if n_nodes > 0 else 0

    if isolated_frac <= 0.50:
        return

    report.add(
        TextSection(
            id=f"backbone-sparsity-{chamber.lower()}",
            title=f"{chamber} Backbone Sparsity Note",
            html=(
                f"<p><strong>Caveat:</strong> {isolated_frac:.0%} of {chamber} legislators "
                f"({n_isolated} of {n_nodes}) are isolated in the BiCM backbone — they have "
                f"no statistically validated co-voting edges. This extreme sparsity means "
                f"the BiCM null model (preserving degree sequences) explains nearly all "
                f"{chamber} co-voting patterns.</p>"
                f"<p>Backbone centrality measures are unreliable for this chamber. "
                f"The Phase 6 Kappa-weighted network provides more useful legislator-level "
                f"centrality rankings.</p>"
            ),
        )
    )


def _add_backbone_layout_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Figure: BiCM backbone network layout."""
    path = plots_dir / f"backbone_layout_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"backbone-layout-{chamber.lower()}",
                title=f"{chamber} — BiCM Backbone Layout",
                path=path,
                caption=(
                    "Spring layout of the BiCM-validated legislator network. "
                    "Only statistically significant co-voting edges are shown."
                ),
                alt_text=(
                    f"Network graph of {chamber} legislators using spring layout with only "
                    "BiCM-validated co-voting edges. Nodes are colored by party; clusters "
                    "reveal partisan voting blocs."
                ),
            )
        )


def _add_backbone_comparison(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
    plots_dir: Path,
) -> None:
    """Table: BiCM backbone vs Phase 6 Kappa backbone."""
    comp = chamber_results.get("backbone_comparison")
    if comp is None:
        return

    rows = [
        {"Metric": "Edge Jaccard", "Value": f"{comp['edge_jaccard']:.4f}"},
        {"Metric": "Shared Edges", "Value": str(comp["shared_edges"])},
        {"Metric": "BiCM-Only Edges", "Value": str(comp["bicm_only"])},
        {"Metric": "Kappa-Only Edges", "Value": str(comp["kappa_only"])},
        {"Metric": "BiCM Total Edges", "Value": str(comp["bicm_total"])},
        {"Metric": "Kappa Total Edges", "Value": str(comp["kappa_total"])},
    ]

    comm = comp.get("community_comparison", {})
    if comm.get("nmi") is not None:
        rows.append({"Metric": "Community NMI", "Value": f"{comm['nmi']:.4f}"})
        rows.append({"Metric": "Community ARI", "Value": f"{comm['ari']:.4f}"})

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} Backbone Comparison",
        subtitle="BiCM backbone vs Phase 6 Kappa + disparity filter backbone",
    )
    report.add(
        TableSection(
            id=f"backbone-comparison-{chamber.lower()}",
            title=f"{chamber} — Backbone Comparison (BiCM vs Phase 6)",
            html=html,
        )
    )

    # Add comparison figure
    path = plots_dir / f"backbone_comparison_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"backbone-comparison-fig-{chamber.lower()}",
                title=f"{chamber} — Backbone Comparison Layout",
                path=path,
                caption="Side-by-side: BiCM backbone (left) vs Kappa + disparity backbone (right).",
                alt_text=(
                    f"Side-by-side network layouts comparing {chamber} BiCM backbone (left) with "
                    "Phase 6 Kappa + disparity filter backbone (right). Shared and method-specific "
                    "edges are visually distinguishable."
                ),
            )
        )


def _add_hidden_alliances(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: Cross-party edges found by BiCM but not by Phase 6."""
    comp = chamber_results.get("backbone_comparison")
    if comp is None:
        return

    hidden = comp.get("hidden_alliances", [])
    if not hidden:
        report.add(
            TextSection(
                id=f"hidden-alliances-{chamber.lower()}",
                title=f"{chamber} — Hidden Alliances",
                html=f"<p>No cross-party edges found in BiCM backbone that are absent from "
                f"the Phase 6 Kappa backbone for {chamber}.</p>",
            )
        )
        return

    df = pl.DataFrame(hidden)
    html = make_gt(
        df,
        title=f"{chamber} Hidden Alliances",
        subtitle=(
            "Cross-party edges present in BiCM backbone but absent from Phase 6 — "
            "co-voting relationships that Kappa thresholding missed"
        ),
        column_labels={
            "legislator_1": "Legislator 1",
            "party_1": "Party 1",
            "legislator_2": "Legislator 2",
            "party_2": "Party 2",
        },
    )
    report.add(
        TableSection(
            id=f"hidden-alliances-{chamber.lower()}",
            title=f"{chamber} — Hidden Alliances",
            html=html,
        )
    )


def _add_backbone_communities(
    report: ReportBuilder,
    chamber_results: dict,
    chamber: str,
) -> None:
    """Table: BiCM backbone community detection vs party."""
    vs_party = chamber_results.get("backbone_vs_party", {})
    if vs_party.get("nmi") is None:
        return

    backbone_G = chamber_results.get("backbone_graph")
    partition = chamber_results.get("backbone_partition", {})
    if not partition or backbone_G is None:
        return

    # Count party composition per community
    comm_party: dict[int, dict[str, int]] = {}
    for node, comm_id in partition.items():
        party = backbone_G.nodes[node].get("party", "Unknown")
        if comm_id not in comm_party:
            comm_party[comm_id] = {}
        comm_party[comm_id][party] = comm_party[comm_id].get(party, 0) + 1

    rows = []
    for comm_id in sorted(comm_party.keys()):
        party_counts = comm_party[comm_id]
        total = sum(party_counts.values())
        row: dict = {"Community": comm_id, "Total": total}
        for party in ["Republican", "Democrat", "Independent"]:
            row[party] = party_counts.get(party, 0)
        rows.append(row)

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title=f"{chamber} Backbone Communities vs Party",
        subtitle=f"NMI={vs_party['nmi']:.4f}, ARI={vs_party['ari']:.4f}",
    )
    report.add(
        TableSection(
            id=f"backbone-communities-{chamber.lower()}",
            title=f"{chamber} — Backbone Communities vs Party",
            html=html,
        )
    )


def _add_bipartite_layout_figure(
    report: ReportBuilder,
    plots_dir: Path,
    chamber: str,
) -> None:
    """Figure: Bipartite layout showing bridge bills and their voters."""
    path = plots_dir / f"bipartite_layout_{chamber.lower()}.png"
    if path.exists():
        report.add(
            FigureSection.from_file(
                id=f"bipartite-layout-{chamber.lower()}",
                title=f"{chamber} — Bridge Bill Bipartite Layout",
                path=path,
                caption=(
                    "Two-column layout: legislators (left, colored by party) connected "
                    "to top bridge bills (right, gold squares) via Yea votes."
                ),
                alt_text=(
                    f"Two-column bipartite network layout for {chamber} with legislators on the "
                    "left colored by party and top bridge bills on the right as gold squares, "
                    "connected by Yea-vote edges."
                ),
            )
        )


def _add_cross_method_summary(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    """Table: Cross-method community agreement summary."""
    rows = []
    for chamber in chambers:
        comp = results[chamber].get("backbone_comparison", {})
        comm = comp.get("community_comparison", {})
        vs_party = results[chamber].get("backbone_vs_party", {})

        rows.append(
            {
                "Chamber": chamber,
                "BiCM vs Party NMI": vs_party.get("nmi"),
                "BiCM vs Party ARI": vs_party.get("ari"),
                "BiCM vs Kappa NMI": comm.get("nmi"),
                "BiCM vs Kappa ARI": comm.get("ari"),
                "Edge Jaccard": comp.get("edge_jaccard"),
            }
        )

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Cross-Method Community Agreement",
        subtitle="BiCM backbone vs party labels and Phase 6 Kappa backbone",
        number_formats={
            "BiCM vs Party NMI": ".4f",
            "BiCM vs Party ARI": ".4f",
            "BiCM vs Kappa NMI": ".4f",
            "BiCM vs Kappa ARI": ".4f",
            "Edge Jaccard": ".4f",
        },
    )
    report.add(
        TableSection(
            id="cross-method-summary",
            title="Cross-Method Community Agreement",
            html=html,
        )
    )


def _add_downstream_findings(
    report: ReportBuilder,
    results: dict[str, dict],
    chambers: list[str],
) -> None:
    """Text: Key downstream findings and implications."""
    parts = ["<ul>"]

    for chamber in chambers:
        pol = results[chamber].get("polarization")
        if pol is not None and pol.height > 0:
            median_pol = float(pol["polarization"].median())
            high_pol = pol.filter(pl.col("polarization") > 0.8).height
            pct_high = high_pol / pol.height * 100
            parts.append(
                f"<li><strong>{chamber}:</strong> Median polarization = {median_pol:.2f}. "
                f"{high_pol} bills ({pct_high:.0f}%) are highly polarized (&gt;0.8).</li>"
            )

        bridge = results[chamber].get("bridge_bills")
        if bridge is not None and bridge.height > 0:
            top = bridge.head(1)
            if top.height > 0:
                row = top.row(0, named=True)
                bn = row.get("bill_number", row["vote_id"])
                parts.append(
                    f"<li><strong>{chamber} top bridge:</strong> {bn} "
                    f"(betweenness={row['betweenness']:.4f})</li>"
                )

        backbone_G = results[chamber].get("backbone_graph")
        if backbone_G is not None:
            parts.append(
                f"<li><strong>{chamber} BiCM backbone:</strong> "
                f"{backbone_G.number_of_edges()} validated edges</li>"
            )

    parts.append("</ul>")
    html = "".join(parts)

    report.add(
        TextSection(
            id="downstream-findings",
            title="Downstream Findings",
            html=html,
        )
    )


def _generate_bipartite_key_findings(results: dict[str, dict]) -> list[str]:
    """Generate 2-4 key findings from bipartite network results."""
    findings: list[str] = []
    import polars as pl

    for chamber in ["House", "Senate"]:
        if chamber not in results or "summary" not in results[chamber]:
            continue

        # Polarization
        pol = results[chamber].get("polarization")
        if pol is not None and pol.height > 0:
            median_pol = float(pol["polarization"].median())
            high_pol = pol.filter(pl.col("polarization") > 0.8).height
            pct_high = high_pol / pol.height * 100
            findings.append(
                f"{chamber}: <strong>{pct_high:.0f}%</strong> of bills are highly "
                f"polarized (>0.8), median = {median_pol:.2f}."
            )

        # Bridge bills
        bridge = results[chamber].get("bridge_bills")
        if bridge is not None and bridge.height > 0:
            top = bridge.head(1).row(0, named=True)
            bn = top.get("bill_number", top.get("vote_id", "?"))
            findings.append(
                f"{chamber} top bridge bill: <strong>{bn}</strong> "
                f"(betweenness = {top.get('betweenness', 0):.4f})."
            )

        # Backbone density
        backbone_G = results[chamber].get("backbone_graph")
        if backbone_G is not None and backbone_G.number_of_nodes() > 0:
            n_nodes = backbone_G.number_of_nodes()
            n_edges = backbone_G.number_of_edges()
            max_possible = n_nodes * (n_nodes - 1) // 2
            density = n_edges / max_possible if max_possible > 0 else 0
            findings.append(
                f"{chamber} BiCM backbone: <strong>{n_edges}</strong> validated edges "
                f"(density = {density:.4f})."
            )

        break  # First chamber only

    return findings


def _add_analysis_parameters(report: ReportBuilder) -> None:
    """Table: Analysis parameters used."""
    from analysis.bipartite import (
        BACKBONE_COMPARISON_THRESHOLD,
        BICM_SIGNIFICANCE,
        BICM_SIGNIFICANCE_SENATE,
        BILL_CLUSTER_RESOLUTIONS,
        BILL_POLARIZATION_MIN_VOTERS,
        NEWMAN_PROJECTION,
        RANDOM_SEED,
        TOP_BRIDGE_BILLS,
    )

    rows = [
        {"Parameter": "BiCM Significance (House)", "Value": str(BICM_SIGNIFICANCE)},
        {"Parameter": "BiCM Significance (Senate)", "Value": str(BICM_SIGNIFICANCE_SENATE)},
        {"Parameter": "Min Voters (Polarization)", "Value": str(BILL_POLARIZATION_MIN_VOTERS)},
        {"Parameter": "Newman Projection", "Value": str(NEWMAN_PROJECTION)},
        {"Parameter": "Bill Cluster Resolutions", "Value": str(BILL_CLUSTER_RESOLUTIONS)},
        {"Parameter": "Top Bridge Bills", "Value": str(TOP_BRIDGE_BILLS)},
        {"Parameter": "Backbone Comparison Threshold", "Value": str(BACKBONE_COMPARISON_THRESHOLD)},
        {"Parameter": "Random Seed", "Value": str(RANDOM_SEED)},
    ]

    df = pl.DataFrame(rows)
    html = make_gt(
        df,
        title="Analysis Parameters",
        subtitle="Constants used in this analysis run",
    )
    report.add(TableSection(id="parameters", title="Analysis Parameters", html=html))
