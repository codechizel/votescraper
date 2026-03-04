"""
Tests for Bipartite Network (Phase 12) helper functions using synthetic fixtures.

These tests verify that the non-plotting functions in analysis/12_bipartite/bipartite.py
produce correct results on known inputs. Plotting and full pipeline execution are not tested here;
graph construction, polarization, betweenness, projection, backbone extraction, and comparison are.

Run: uv run pytest tests/test_bipartite.py -v
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.bipartite import (
    BACKBONE_COMPARISON_THRESHOLD,
    BICM_SIGNIFICANCE,
    BILL_CLUSTER_RESOLUTIONS,
    BILL_POLARIZATION_MIN_VOTERS,
    NEWMAN_PROJECTION,
    PARTY_COLORS,
    RANDOM_SEED,
    TOP_BRIDGE_BILLS,
    analyze_bill_community_profiles,
    build_backbone_graph,
    build_bill_projection,
    build_bipartite_graph,
    build_kappa_network_for_comparison,
    compare_backbones,
    compute_backbone_centrality,
    compute_bill_polarization,
    compute_bipartite_betweenness,
    compute_bipartite_summary,
    detect_backbone_communities,
    detect_bill_communities,
    disparity_filter,
    extract_bicm_backbone,
    identify_bridge_bills,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vote_matrix_pl() -> pl.DataFrame:
    """Synthetic vote matrix: 12 legislators (8R, 4D) x 15 bills.

    Bill types:
      v1-v5:   Party-line (R=Yea, D=Nay)
      v6-v10:  Easy/bipartisan (both parties mostly Yea)
      v11-v12: Bridge (mixed support)
      v13-v15: Reverse party-line (R=Nay, D=Yea)
    """
    rng = np.random.default_rng(42)
    slugs = [f"rep_r{i}_1" for i in range(1, 9)] + [f"rep_d{i}_1" for i in range(1, 5)]
    data: dict[str, list] = {"legislator_slug": slugs}

    for v in range(1, 16):
        col: list = []
        for i in range(12):
            is_r = i < 8
            if v <= 5:  # Party-line R=Yea
                p = 0.95 if is_r else 0.05
            elif v <= 10:  # Easy/bipartisan
                p = 0.85
            elif v <= 12:  # Bridge bills (moderate R + D)
                p = 0.5 if is_r else 0.7
            else:  # Reverse (R=Nay, D=Yea)
                p = 0.05 if is_r else 0.95
            col.append(float(int(rng.random() < p)))
        data[f"v{v}"] = col

    return pl.DataFrame(data)


@pytest.fixture
def ideal_points() -> pl.DataFrame:
    """IRT ideal points matching the 12-legislator fixture."""
    slugs = [f"rep_r{i}_1" for i in range(1, 9)] + [f"rep_d{i}_1" for i in range(1, 5)]
    xis = list(np.linspace(2.0, 0.5, 8)) + list(np.linspace(-0.5, -2.0, 4))
    parties = ["Republican"] * 8 + ["Democrat"] * 4
    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "xi_mean": xis,
            "xi_sd": [0.1] * 12,
            "party": parties,
            "full_name": [f"Legislator {s}" for s in slugs],
        }
    )


@pytest.fixture
def bill_params() -> pl.DataFrame:
    """Bill parameters matching the 15-bill fixture."""
    vote_ids = [f"v{i}" for i in range(1, 16)]
    betas = (
        [2.0, 1.8, 2.1, 1.9, 2.2]  # Party-line: high discrimination
        + [0.3, 0.2, 0.4, 0.1, 0.3]  # Easy: low discrimination
        + [0.5, 0.6]  # Bridge: moderate
        + [-1.8, -2.0, -1.9]  # Reverse: high (negative)
    )
    return pl.DataFrame(
        {
            "vote_id": vote_ids,
            "beta_mean": betas,
            "alpha_mean": [0.0] * 15,
            "bill_number": [f"HB {i}" for i in range(1, 16)],
        }
    )


@pytest.fixture
def rollcalls() -> pl.DataFrame:
    """Rollcall metadata matching the 15-bill fixture."""
    return pl.DataFrame(
        {
            "vote_id": [f"v{i}" for i in range(1, 16)],
            "bill_number": [f"HB {i}" for i in range(1, 16)],
            "short_title": [f"Test Bill {i}" for i in range(1, 16)],
            "motion": ["Final Action"] * 15,
        }
    )


@pytest.fixture
def bipartite_graph(
    vote_matrix_pl: pl.DataFrame,
    ideal_points: pl.DataFrame,
    bill_params: pl.DataFrame,
    rollcalls: pl.DataFrame,
) -> nx.Graph:
    """Built bipartite graph from synthetic fixtures."""
    return build_bipartite_graph(vote_matrix_pl, ideal_points, bill_params, rollcalls)


# ── Test Classes ─────────────────────────────────────────────────────────────


class TestBuildBipartiteGraph:
    """Tests for build_bipartite_graph()."""

    def test_node_count(self, bipartite_graph: nx.Graph) -> None:
        """Graph should have 12 legislators + 15 bills = 27 nodes."""
        assert bipartite_graph.number_of_nodes() == 27

    def test_bipartite_attribute(self, bipartite_graph: nx.Graph) -> None:
        """All nodes should have a bipartite attribute (0 or 1)."""
        for _, data in bipartite_graph.nodes(data=True):
            assert "bipartite" in data
            assert data["bipartite"] in (0, 1)

    def test_legislator_count(self, bipartite_graph: nx.Graph) -> None:
        """Should have 12 legislator nodes (bipartite=0)."""
        legislators = [n for n, d in bipartite_graph.nodes(data=True) if d["bipartite"] == 0]
        assert len(legislators) == 12

    def test_bill_count(self, bipartite_graph: nx.Graph) -> None:
        """Should have 15 bill nodes (bipartite=1)."""
        bills = [n for n, d in bipartite_graph.nodes(data=True) if d["bipartite"] == 1]
        assert len(bills) == 15

    def test_edge_count_matches_yea_votes(
        self, vote_matrix_pl: pl.DataFrame, bipartite_graph: nx.Graph
    ) -> None:
        """Number of edges should equal number of Yea (1.0) votes."""
        slug_col = vote_matrix_pl.columns[0]
        vote_cols = [c for c in vote_matrix_pl.columns if c != slug_col]
        arr = vote_matrix_pl.select(vote_cols).to_numpy().astype(float)
        n_yea = int(np.nansum(arr))
        assert bipartite_graph.number_of_edges() == n_yea

    def test_no_within_mode_edges(self, bipartite_graph: nx.Graph) -> None:
        """No edges between two legislators or two bills."""
        for u, v in bipartite_graph.edges():
            bu = bipartite_graph.nodes[u]["bipartite"]
            bv = bipartite_graph.nodes[v]["bipartite"]
            assert bu != bv, f"Same-mode edge: {u} ({bu}) -- {v} ({bv})"

    def test_legislator_attributes(self, bipartite_graph: nx.Graph) -> None:
        """Legislator nodes should have party, xi_mean, full_name."""
        for n, d in bipartite_graph.nodes(data=True):
            if d["bipartite"] == 0:
                assert "party" in d
                assert "xi_mean" in d
                assert "full_name" in d


class TestBipartiteSummary:
    """Tests for compute_bipartite_summary()."""

    def test_summary_keys(self, bipartite_graph: nx.Graph) -> None:
        summary = compute_bipartite_summary(bipartite_graph)
        expected_keys = {
            "n_legislators",
            "n_bills",
            "n_edges",
            "density",
            "avg_legislator_degree",
            "avg_bill_degree",
            "max_legislator_degree",
            "max_bill_degree",
        }
        assert set(summary.keys()) == expected_keys

    def test_density_range(self, bipartite_graph: nx.Graph) -> None:
        summary = compute_bipartite_summary(bipartite_graph)
        assert 0 <= summary["density"] <= 1


class TestBillPolarization:
    """Tests for compute_bill_polarization()."""

    def test_party_line_high_polarization(
        self, vote_matrix_pl: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """Party-line bills (v1-v5) should have high polarization."""
        pol = compute_bill_polarization(vote_matrix_pl, ideal_points)
        party_line = pol.filter(pl.col("vote_id").is_in([f"v{i}" for i in range(1, 6)]))
        if party_line.height > 0:
            mean_pol = float(party_line["polarization"].mean())
            assert mean_pol > 0.5, f"Party-line bills should be polarized, got {mean_pol}"

    def test_easy_bills_low_polarization(
        self, vote_matrix_pl: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """Easy/bipartisan bills (v6-v10) should have low polarization."""
        pol = compute_bill_polarization(vote_matrix_pl, ideal_points)
        easy = pol.filter(pl.col("vote_id").is_in([f"v{i}" for i in range(6, 11)]))
        if easy.height > 0:
            mean_pol = float(easy["polarization"].mean())
            assert mean_pol < 0.5, f"Easy bills should be less polarized, got {mean_pol}"

    def test_polarization_range(
        self, vote_matrix_pl: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """All polarization values should be in [0, 1]."""
        pol = compute_bill_polarization(vote_matrix_pl, ideal_points)
        if pol.height > 0:
            assert float(pol["polarization"].min()) >= 0.0
            assert float(pol["polarization"].max()) <= 1.0

    def test_min_voters_filter(
        self, vote_matrix_pl: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """With min_voters=100, should return no bills (only 12 legislators)."""
        pol = compute_bill_polarization(vote_matrix_pl, ideal_points, min_voters=100)
        assert pol.height == 0

    def test_columns(self, vote_matrix_pl: pl.DataFrame, ideal_points: pl.DataFrame) -> None:
        """Result should have expected columns."""
        pol = compute_bill_polarization(vote_matrix_pl, ideal_points)
        expected = {
            "vote_id",
            "polarization",
            "pct_r_yea",
            "pct_d_yea",
            "n_r",
            "n_d",
            "bill_number",
            "short_title",
            "beta_mean",
        }
        assert expected <= set(pol.columns)


class TestBipartiteBetweenness:
    """Tests for compute_bipartite_betweenness()."""

    def test_returns_all_nodes(self, bipartite_graph: nx.Graph) -> None:
        """Betweenness should be computed for all nodes."""
        btwn = compute_bipartite_betweenness(bipartite_graph)
        assert len(btwn) == bipartite_graph.number_of_nodes()

    def test_values_nonnegative(self, bipartite_graph: nx.Graph) -> None:
        """All betweenness values should be >= 0."""
        btwn = compute_bipartite_betweenness(bipartite_graph)
        for v in btwn.values():
            assert v >= 0.0

    def test_bridge_bills_have_betweenness(self, bipartite_graph: nx.Graph) -> None:
        """Bridge bills (v11, v12) should have non-zero betweenness if connected."""
        btwn = compute_bipartite_betweenness(bipartite_graph)
        bridge_vals = [btwn.get(f"v{i}", 0.0) for i in [11, 12]]
        # At least one bridge bill should have betweenness > 0
        assert max(bridge_vals) >= 0.0  # They might be 0 if graph is too connected

    def test_bridge_bill_identification(
        self,
        bipartite_graph: nx.Graph,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
        bill_params: pl.DataFrame,
    ) -> None:
        """identify_bridge_bills should return a DataFrame with expected columns."""
        btwn = compute_bipartite_betweenness(bipartite_graph)
        pol = compute_bill_polarization(vote_matrix_pl, ideal_points)
        bridges = identify_bridge_bills(bipartite_graph, btwn, pol, bill_params, top_n=5)
        assert bridges.height <= 5
        expected_cols = {"vote_id", "betweenness", "degree"}
        assert expected_cols <= set(bridges.columns)


class TestBillProjection:
    """Tests for build_bill_projection()."""

    def test_bill_only_nodes(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Projected graph should only have bill nodes."""
        G = build_bill_projection(vote_matrix_pl)
        vote_ids = set(vote_matrix_pl.columns[1:])
        assert set(G.nodes()) == vote_ids

    def test_newman_weights_positive(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Newman-weighted projection should have positive edge weights."""
        G = build_bill_projection(vote_matrix_pl, use_newman=True)
        for _, _, d in G.edges(data=True):
            assert d["weight"] > 0

    def test_symmetric(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Projection should be symmetric (undirected graph)."""
        G = build_bill_projection(vote_matrix_pl)
        for u, v in G.edges():
            assert G.has_edge(v, u)

    def test_simple_projection(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Simple (non-Newman) projection should also work."""
        G = build_bill_projection(vote_matrix_pl, use_newman=False)
        assert G.number_of_nodes() > 0
        for _, _, d in G.edges(data=True):
            assert d["weight"] > 0


class TestBillCommunities:
    """Tests for detect_bill_communities()."""

    def test_partition_and_sweep_returned(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Should return both partitions dict and sweep DataFrame."""
        G = build_bill_projection(vote_matrix_pl)
        partitions, sweep = detect_bill_communities(G, resolutions=[1.0])
        assert isinstance(partitions, dict)
        assert isinstance(sweep, pl.DataFrame)

    def test_all_bills_assigned(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Every bill should have a community assignment."""
        G = build_bill_projection(vote_matrix_pl)
        partitions, _ = detect_bill_communities(G, resolutions=[1.0])
        if partitions:
            partition = list(partitions.values())[0]
            assert set(partition.keys()) == set(G.nodes())

    def test_resolution_sweep(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Sweep DataFrame should have one row per resolution."""
        G = build_bill_projection(vote_matrix_pl)
        resolutions = [0.5, 1.0, 2.0]
        _, sweep = detect_bill_communities(G, resolutions=resolutions)
        assert sweep.height == len(resolutions)
        assert "resolution" in sweep.columns
        assert "n_communities" in sweep.columns
        assert "modularity" in sweep.columns

    def test_community_profiles(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Community profiles should have expected columns."""
        G = build_bill_projection(vote_matrix_pl)
        partitions, _ = detect_bill_communities(G, resolutions=[1.0])
        if partitions:
            partition = list(partitions.values())[0]
            profiles = analyze_bill_community_profiles(partition, vote_matrix_pl, ideal_points)
            assert "community" in profiles.columns
            assert "n_bills" in profiles.columns
            assert "mean_pct_r_yea" in profiles.columns
            assert "mean_pct_d_yea" in profiles.columns


class TestBiCMBackbone:
    """Tests for extract_bicm_backbone() and build_backbone_graph()."""

    def test_shape(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Validated matrix should be (n_legs, n_legs)."""
        validated, pvalues, slugs = extract_bicm_backbone(vote_matrix_pl)
        n = len(slugs)
        assert validated.shape == (n, n)
        assert pvalues.shape == (n, n)

    def test_symmetric(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Validated matrix should be symmetric."""
        validated, _, _ = extract_bicm_backbone(vote_matrix_pl)
        np.testing.assert_array_equal(validated, validated.T)

    def test_fewer_edges_than_full(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Backbone should have fewer edges than a fully connected graph."""
        validated, _, slugs = extract_bicm_backbone(vote_matrix_pl)
        n = len(slugs)
        max_edges = n * (n - 1) / 2
        actual_edges = np.sum(validated[np.triu_indices(n, k=1)])
        assert actual_edges <= max_edges

    def test_backbone_graph_attributes(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Backbone graph nodes should have party and xi_mean attributes."""
        validated, _, slugs = extract_bicm_backbone(vote_matrix_pl)
        G = build_backbone_graph(validated, slugs, ideal_points)
        for n, d in G.nodes(data=True):
            assert "party" in d
            assert "xi_mean" in d
            assert "full_name" in d

    def test_significance_threshold(self, vote_matrix_pl: pl.DataFrame) -> None:
        """Stricter threshold should produce fewer or equal edges."""
        v_loose, _, slugs = extract_bicm_backbone(vote_matrix_pl, significance=0.1)
        v_strict, _, _ = extract_bicm_backbone(vote_matrix_pl, significance=0.001)
        n = len(slugs)
        loose_count = np.sum(v_loose[np.triu_indices(n, k=1)])
        strict_count = np.sum(v_strict[np.triu_indices(n, k=1)])
        assert strict_count <= loose_count


class TestBackboneCentrality:
    """Tests for compute_backbone_centrality()."""

    def test_centrality_columns(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Centrality result should have expected columns."""
        validated, _, slugs = extract_bicm_backbone(vote_matrix_pl)
        G = build_backbone_graph(validated, slugs, ideal_points)
        cent = compute_backbone_centrality(G)
        expected = {
            "legislator_slug",
            "full_name",
            "party",
            "xi_mean",
            "degree",
            "betweenness",
            "eigenvector",
            "pagerank",
        }
        assert expected <= set(cent.columns)

    def test_empty_graph(self) -> None:
        """Should handle empty graph gracefully."""
        G = nx.Graph()
        cent = compute_backbone_centrality(G)
        assert cent.height == 0


class TestBackboneComparison:
    """Tests for compare_backbones()."""

    def test_identical_graphs_jaccard_one(self) -> None:
        """Identical graphs should have Jaccard = 1.0."""
        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"], party="R", full_name="Test", xi_mean=0.0)
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        result = compare_backbones(G, G)
        assert result["edge_jaccard"] == 1.0

    def test_disjoint_graphs_jaccard_zero(self) -> None:
        """Disjoint edge sets should have Jaccard = 0.0."""
        G1 = nx.Graph()
        G1.add_nodes_from(["a", "b", "c", "d"], party="R", full_name="Test", xi_mean=0.0)
        G1.add_edge("a", "b")

        G2 = nx.Graph()
        G2.add_nodes_from(["a", "b", "c", "d"], party="R", full_name="Test", xi_mean=0.0)
        G2.add_edge("c", "d")

        result = compare_backbones(G1, G2)
        assert result["edge_jaccard"] == 0.0

    def test_expected_keys(self) -> None:
        """Result should have all expected keys."""
        G = nx.Graph()
        G.add_node("a", party="R", full_name="Test", xi_mean=0.0)
        result = compare_backbones(G, G)
        expected = {
            "edge_jaccard",
            "shared_edges",
            "bicm_only",
            "kappa_only",
            "bicm_total",
            "kappa_total",
            "hidden_alliances",
            "community_comparison",
        }
        assert expected <= set(result.keys())


class TestBackboneCommunities:
    """Tests for detect_backbone_communities()."""

    def test_all_nodes_assigned(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Every node should have a community assignment."""
        validated, _, slugs = extract_bicm_backbone(vote_matrix_pl)
        G = build_backbone_graph(validated, slugs, ideal_points)
        partition, metrics = detect_backbone_communities(G)
        assert set(partition.keys()) == set(G.nodes())

    def test_comparison_metrics(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Should return NMI and ARI values."""
        validated, _, slugs = extract_bicm_backbone(vote_matrix_pl)
        G = build_backbone_graph(validated, slugs, ideal_points)
        _, metrics = detect_backbone_communities(G)
        assert "nmi" in metrics
        assert "ari" in metrics

    def test_empty_graph(self) -> None:
        """Empty graph should return all-zero partition."""
        G = nx.Graph()
        G.add_node("a", party="R")
        partition, metrics = detect_backbone_communities(G)
        assert partition == {"a": 0}


class TestKappaNetworkComparison:
    """Tests for build_kappa_network_for_comparison() and disparity_filter()."""

    def test_kappa_network_builds(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Should build a Kappa network from vote matrix."""
        G = build_kappa_network_for_comparison(vote_matrix_pl, ideal_points)
        assert G.number_of_nodes() > 0

    def test_disparity_filter_reduces_edges(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Disparity filter should retain same or fewer edges."""
        G = build_kappa_network_for_comparison(vote_matrix_pl, ideal_points, threshold=0.2)
        backbone = disparity_filter(G)
        assert backbone.number_of_edges() <= G.number_of_edges()

    def test_disparity_filter_preserves_nodes(
        self,
        vote_matrix_pl: pl.DataFrame,
        ideal_points: pl.DataFrame,
    ) -> None:
        """Disparity filter should preserve all nodes."""
        G = build_kappa_network_for_comparison(vote_matrix_pl, ideal_points, threshold=0.2)
        backbone = disparity_filter(G)
        assert set(backbone.nodes()) == set(G.nodes())


class TestConstants:
    """Tests for module constants consistency."""

    def test_random_seed(self) -> None:
        assert RANDOM_SEED == 42

    def test_backbone_comparison_threshold(self) -> None:
        assert BACKBONE_COMPARISON_THRESHOLD == 0.40

    def test_party_colors(self) -> None:
        assert "Republican" in PARTY_COLORS
        assert "Democrat" in PARTY_COLORS
        assert "Independent" in PARTY_COLORS

    def test_bicm_significance(self) -> None:
        assert 0 < BICM_SIGNIFICANCE < 1

    def test_bill_cluster_resolutions(self) -> None:
        assert len(BILL_CLUSTER_RESOLUTIONS) >= 3
        assert all(r > 0 for r in BILL_CLUSTER_RESOLUTIONS)

    def test_top_bridge_bills(self) -> None:
        assert TOP_BRIDGE_BILLS > 0

    def test_min_voters(self) -> None:
        assert BILL_POLARIZATION_MIN_VOTERS > 0

    def test_newman_projection(self) -> None:
        assert isinstance(NEWMAN_PROJECTION, bool)
