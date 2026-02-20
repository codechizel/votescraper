"""
Tests for clustering helper functions using synthetic fixtures.

These tests verify that the non-plotting functions in analysis/clustering.py
produce correct results on known inputs. Plotting and full pipeline execution
are not tested here (integration-level), but data preparation, party loyalty
computation, within-party clustering, and cross-method comparison logic are.

Run: uv run pytest tests/test_clustering.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path so we can import analysis.clustering
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.clustering import (
    CONTESTED_PARTY_THRESHOLD,
    RANDOM_SEED,
    SILHOUETTE_GOOD,
    WITHIN_PARTY_MIN_SIZE,
    compare_methods,
    compute_party_loyalty,
    run_within_party_clustering,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vote_matrix() -> pl.DataFrame:
    """Synthetic vote matrix: 8 legislators x 6 votes.

    Layout (1=Yea, 0=Nay, None=absent):
      R1-R4 are Republicans, D1-D4 are Democrats.
      v1: party-line (Rs=Yea, Ds=Nay)
      v2: party-line (Rs=Yea, Ds=Nay)
      v3: near-unanimous (all Yea except D4=Nay)
      v4: party-line reversed (Rs=Nay, Ds=Yea)
      v5: R1 dissents from R (party-line otherwise)
      v6: D1 dissents from D (party-line otherwise)
    """
    return pl.DataFrame(
        {
            "legislator_slug": [
                "sen_r1_1",
                "sen_r2_1",
                "sen_r3_1",
                "sen_r4_1",
                "sen_d1_1",
                "sen_d2_1",
                "sen_d3_1",
                "sen_d4_1",
            ],
            "v1": [1, 1, 1, 1, 0, 0, 0, 0],
            "v2": [1, 1, 1, 1, 0, 0, 0, 0],
            "v3": [1, 1, 1, 1, 1, 1, 1, 0],
            "v4": [0, 0, 0, 0, 1, 1, 1, 1],
            "v5": [0, 1, 1, 1, 0, 0, 0, 0],  # R1 dissents
            "v6": [1, 1, 1, 1, 1, 0, 0, 0],  # D1 dissents
        }
    )


@pytest.fixture
def ideal_points() -> pl.DataFrame:
    """IRT ideal points matching vote_matrix legislators.

    R1-R4 are positive (conservative), D1-D4 are negative (liberal).
    R1 is the most extreme Republican.
    """
    return pl.DataFrame(
        {
            "legislator_slug": [
                "sen_r1_1",
                "sen_r2_1",
                "sen_r3_1",
                "sen_r4_1",
                "sen_d1_1",
                "sen_d2_1",
                "sen_d3_1",
                "sen_d4_1",
            ],
            "xi_mean": [3.0, 2.5, 2.0, 1.5, -1.5, -2.0, -2.5, -3.0],
            "xi_sd": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            "party": ["Republican"] * 4 + ["Democrat"] * 4,
            "full_name": ["R1", "R2", "R3", "R4", "D1", "D2", "D3", "D4"],
        }
    )


# ── compute_party_loyalty tests ──────────────────────────────────────────────


class TestComputePartyLoyalty:
    """Test the party loyalty metric computation."""

    def test_returns_all_legislators(
        self, vote_matrix: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """All legislators with party info should appear in results."""
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        assert loyalty.height == 8

    def test_required_columns(self, vote_matrix: pl.DataFrame, ideal_points: pl.DataFrame) -> None:
        """Output should have the expected columns."""
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        expected = {"legislator_slug", "loyalty_rate", "n_contested_votes", "n_agree", "party"}
        assert expected.issubset(set(loyalty.columns))

    def test_loyalty_rate_range(
        self, vote_matrix: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """Loyalty rates should be between 0 and 1."""
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        rates = loyalty["loyalty_rate"].to_list()
        assert all(0.0 <= r <= 1.0 for r in rates)

    def test_dissenter_has_lower_loyalty(
        self, vote_matrix: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """R1 dissents on v5, so R1 should have lower loyalty than R2-R4."""
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        r1 = loyalty.filter(pl.col("legislator_slug") == "sen_r1_1")["loyalty_rate"][0]
        r2 = loyalty.filter(pl.col("legislator_slug") == "sen_r2_1")["loyalty_rate"][0]
        assert r1 < r2

    def test_d1_dissenter_lower_than_d2(
        self, vote_matrix: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """D1 dissents on v6, so D1 should have lower loyalty than D2."""
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        d1 = loyalty.filter(pl.col("legislator_slug") == "sen_d1_1")["loyalty_rate"][0]
        d2 = loyalty.filter(pl.col("legislator_slug") == "sen_d2_1")["loyalty_rate"][0]
        assert d1 < d2

    def test_n_agree_plus_disagree_equals_contested(
        self, vote_matrix: pl.DataFrame, ideal_points: pl.DataFrame
    ) -> None:
        """n_agree / n_contested_votes should equal loyalty_rate."""
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        for row in loyalty.iter_rows(named=True):
            expected = row["n_agree"] / row["n_contested_votes"]
            assert abs(row["loyalty_rate"] - expected) < 1e-10

    def test_unanimous_votes_produce_empty_loyalty(self) -> None:
        """If all party members vote identically, no votes are contested → empty result."""
        vm = pl.DataFrame(
            {
                "legislator_slug": ["sen_a_1", "sen_b_1", "sen_c_1"],
                "v1": [1, 1, 1],
                "v2": [0, 0, 0],
                "v3": [1, 1, 1],
            }
        )
        ip = pl.DataFrame(
            {
                "legislator_slug": ["sen_a_1", "sen_b_1", "sen_c_1"],
                "xi_mean": [1.0, 0.5, 0.0],
                "xi_sd": [0.3, 0.3, 0.3],
                "party": ["Republican"] * 3,
                "full_name": ["A", "B", "C"],
            }
        )
        loyalty = compute_party_loyalty(vm, ip, "Senate")
        # All votes are unanimous → 0% dissent → no contested votes → empty result
        assert loyalty.height == 0

    def test_contested_threshold_respected(self) -> None:
        """Votes where minority < CONTESTED_PARTY_THRESHOLD should NOT be counted."""
        # 10 Rs, 1 dissents = 10% dissent (exactly at threshold → contested)
        n_rs = 10
        slugs = [f"sen_r{i}_1" for i in range(n_rs)]
        # v1: 9 Yea, 1 Nay (10% dissent = threshold → contested)
        # v2: all Yea (0% dissent → not contested)
        vm = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "v1": [1] * 9 + [0],
                "v2": [1] * 10,
            }
        )
        ip = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "xi_mean": [float(i) for i in range(n_rs)],
                "xi_sd": [0.3] * n_rs,
                "party": ["Republican"] * n_rs,
                "full_name": [f"R{i}" for i in range(n_rs)],
            }
        )
        loyalty = compute_party_loyalty(vm, ip, "Senate")
        # v2 is not contested (unanimous), so only v1 counts
        for row in loyalty.iter_rows(named=True):
            assert row["n_contested_votes"] == 1


# ── compare_methods tests ────────────────────────────────────────────────────


class TestCompareMethods:
    """Test the cross-method ARI comparison."""

    def test_identical_labels_ari_one(self) -> None:
        """Two identical label sets should have ARI = 1.0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        assignments = {"method_a": labels, "method_b": labels.copy()}
        result = compare_methods(assignments, "TestChamber")
        assert result["mean_ari"] == pytest.approx(1.0)

    def test_permuted_labels_ari_one(self) -> None:
        """ARI should be 1.0 even if cluster labels are permuted (0↔1)."""
        a = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([1, 1, 1, 0, 0, 0])  # same partition, different labels
        assignments = {"method_a": a, "method_b": b}
        result = compare_methods(assignments, "TestChamber")
        assert result["mean_ari"] == pytest.approx(1.0)

    def test_random_labels_low_ari(self) -> None:
        """Random labels should have ARI near 0 (not 1)."""
        rng = np.random.default_rng(42)
        a = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        b = rng.integers(0, 2, size=10)
        assignments = {"method_a": a, "method_b": b}
        result = compare_methods(assignments, "TestChamber")
        assert result["mean_ari"] < 0.5

    def test_three_methods_pairwise(self) -> None:
        """With 3 methods, should compute 3 pairwise ARI values."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        assignments = {
            "method_a": labels,
            "method_b": labels.copy(),
            "method_c": labels.copy(),
        }
        result = compare_methods(assignments, "TestChamber")
        assert len(result["ari_matrix"]) == 3  # C(3,2) = 3 pairs

    def test_k_values_returned(self) -> None:
        """k_values should report the number of unique clusters per method."""
        assignments = {
            "method_a": np.array([0, 0, 1, 1]),
            "method_b": np.array([0, 1, 2, 2]),
        }
        result = compare_methods(assignments, "TestChamber")
        assert result["k_values"]["method_a"] == 2
        assert result["k_values"]["method_b"] == 3

    def test_different_length_truncates(self) -> None:
        """Methods with different lengths should be truncated to the shortest."""
        a = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([0, 0, 0, 1])
        assignments = {"method_a": a, "method_b": b}
        result = compare_methods(assignments, "TestChamber")
        assert result["n_common"] == 4


# ── run_within_party_clustering tests ────────────────────────────────────────


class TestWithinPartyClustering:
    """Test within-party clustering logic."""

    def test_small_caucus_skipped(
        self, ideal_points: pl.DataFrame, vote_matrix: pl.DataFrame
    ) -> None:
        """Caucuses smaller than WITHIN_PARTY_MIN_SIZE should be skipped."""
        # Our fixture has 4 per party, well below the 15 threshold
        loyalty = compute_party_loyalty(vote_matrix, ideal_points, "Senate")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_within_party_clustering(
                ideal_points, loyalty, range(2, 4), "Senate", Path(tmpdir)
            )
        assert results["republican"]["skipped"] is True
        assert results["democrat"]["skipped"] is True

    def test_large_caucus_not_skipped(self) -> None:
        """Caucuses >= WITHIN_PARTY_MIN_SIZE should be processed."""
        n = WITHIN_PARTY_MIN_SIZE + 5
        slugs = [f"sen_r{i}_1" for i in range(n)]
        rng = np.random.default_rng(RANDOM_SEED)
        ip = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "xi_mean": rng.normal(2.0, 0.5, n).tolist(),
                "xi_sd": [0.3] * n,
                "party": ["Republican"] * n,
                "full_name": [f"R{i}" for i in range(n)],
            }
        )
        # Simple vote matrix for loyalty computation
        vm = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "v1": rng.integers(0, 2, n).tolist(),
                "v2": rng.integers(0, 2, n).tolist(),
                "v3": rng.integers(0, 2, n).tolist(),
            }
        )
        loyalty = compute_party_loyalty(vm, ip, "Senate")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_within_party_clustering(ip, loyalty, range(2, 5), "Senate", Path(tmpdir))
        assert results["republican"]["skipped"] is False
        assert "optimal_k_1d" in results["republican"]
        # Democrats not present → skipped
        assert results["democrat"]["skipped"] is True

    def test_structure_found_flag(self) -> None:
        """With well-separated clusters, structure_found should be True."""
        # Create 2 clearly separated groups within Republicans
        n_per_group = 10
        n = n_per_group * 2
        slugs = [f"sen_r{i}_1" for i in range(n)]
        xi = [4.0 + np.random.default_rng(42).normal(0, 0.1) for _ in range(n_per_group)]
        xi += [1.0 + np.random.default_rng(43).normal(0, 0.1) for _ in range(n_per_group)]

        ip = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "xi_mean": xi,
                "xi_sd": [0.3] * n,
                "party": ["Republican"] * n,
                "full_name": [f"R{i}" for i in range(n)],
            }
        )
        # Loyalty: first group = high loyalty, second group = low loyalty
        loyalty = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "loyalty_rate": [0.95] * n_per_group + [0.50] * n_per_group,
                "n_contested_votes": [50] * n,
                "n_agree": [48] * n_per_group + [25] * n_per_group,
                "party": ["Republican"] * n,
                "full_name": [f"R{i}" for i in range(n)],
            }
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_within_party_clustering(ip, loyalty, range(2, 5), "Senate", Path(tmpdir))
        # With 3-unit separation and tight clusters, silhouette should be high
        assert results["republican"]["best_silhouette_1d"] > SILHOUETTE_GOOD
        assert results["republican"]["structure_found"] is True

    def test_labels_length_matches_caucus(self) -> None:
        """Cluster labels should have one entry per caucus member."""
        n = 20
        slugs = [f"sen_r{i}_1" for i in range(n)]
        rng = np.random.default_rng(RANDOM_SEED)
        ip = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "xi_mean": rng.normal(2.0, 1.0, n).tolist(),
                "xi_sd": [0.3] * n,
                "party": ["Republican"] * n,
                "full_name": [f"R{i}" for i in range(n)],
            }
        )
        vm = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "v1": rng.integers(0, 2, n).tolist(),
                "v2": rng.integers(0, 2, n).tolist(),
            }
        )
        loyalty = compute_party_loyalty(vm, ip, "Senate")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_within_party_clustering(ip, loyalty, range(2, 5), "Senate", Path(tmpdir))
        labels = results["republican"]["labels"]
        assert len(labels) == n


# ── Constants tests ──────────────────────────────────────────────────────────


class TestConstants:
    """Verify critical constants have expected values."""

    def test_random_seed(self) -> None:
        assert RANDOM_SEED == 42

    def test_silhouette_good(self) -> None:
        assert SILHOUETTE_GOOD == 0.50

    def test_within_party_min_size(self) -> None:
        assert WITHIN_PARTY_MIN_SIZE == 15

    def test_contested_threshold(self) -> None:
        assert CONTESTED_PARTY_THRESHOLD == 0.10


# ── Report constant consistency ──────────────────────────────────────────────


class TestReportConstantConsistency:
    """Verify the report module's mirrored constant matches clustering.py."""

    def test_contested_threshold_matches(self) -> None:
        """The report's _CONTESTED_PARTY_THRESHOLD must match clustering.py."""
        from analysis.clustering_report import _CONTESTED_PARTY_THRESHOLD

        assert _CONTESTED_PARTY_THRESHOLD == CONTESTED_PARTY_THRESHOLD
