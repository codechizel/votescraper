"""
Tests for clustering helper functions using synthetic fixtures.

These tests verify that the non-plotting functions in analysis/clustering.py
produce correct results on known inputs. Plotting and full pipeline execution
are not tested here (integration-level), but data preparation, party loyalty
computation, within-party clustering, and cross-method comparison logic are.

Run: uv run pytest tests/test_clustering.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path so we can import analysis.clustering
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.clustering import (
    COMPARISON_K,
    CONTESTED_PARTY_THRESHOLD,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    LINKAGE_METHOD,
    RANDOM_SEED,
    SILHOUETTE_GOOD,
    WITHIN_PARTY_MIN_SIZE,
    _build_display_labels,
    _kappa_to_distance,
    _standardize_2d,
    characterize_clusters,
    compare_methods,
    compute_party_loyalty,
    find_optimal_k_hierarchical,
    run_hdbscan_pca,
    run_hierarchical,
    run_kmeans_irt,
    run_spectral_clustering,
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


# ── Kappa matrix fixture ────────────────────────────────────────────────────


def _make_kappa_matrix(n: int = 20, separation: float = 0.7) -> pl.DataFrame:
    """Build a synthetic Kappa agreement matrix with 2-cluster structure.

    First half = cluster A (high intra-agreement), second half = cluster B.
    Cross-cluster agreement is lower by `separation`.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    slugs = [f"sen_leg{i}_1" for i in range(n)]
    kappa = np.ones((n, n))

    half = n // 2
    for i in range(n):
        for j in range(i + 1, n):
            same_cluster = (i < half and j < half) or (i >= half and j >= half)
            if same_cluster:
                val = 0.80 + rng.normal(0, 0.03)
            else:
                val = 0.80 - separation + rng.normal(0, 0.03)
            kappa[i, j] = val
            kappa[j, i] = val

    data = {"legislator_slug": slugs}
    for j, slug in enumerate(slugs):
        data[slug] = kappa[:, j].tolist()
    return pl.DataFrame(data)


@pytest.fixture
def kappa_matrix_20() -> pl.DataFrame:
    """20-legislator Kappa matrix with clear 2-cluster structure."""
    return _make_kappa_matrix(20)


def _make_pca_scores(n: int = 20) -> pl.DataFrame:
    """Synthetic PCA scores with 2-cluster structure."""
    rng = np.random.default_rng(RANDOM_SEED)
    half = n // 2
    pc1 = np.concatenate([rng.normal(2.0, 0.3, half), rng.normal(-2.0, 0.3, n - half)])
    pc2 = rng.normal(0, 0.5, n)
    pc3 = rng.normal(0, 0.3, n)
    return pl.DataFrame(
        {
            "legislator_slug": [f"sen_leg{i}_1" for i in range(n)],
            "PC1": pc1.tolist(),
            "PC2": pc2.tolist(),
            "PC3": pc3.tolist(),
        }
    )


def _make_ideal_points_20(n: int = 20) -> pl.DataFrame:
    """Ideal points for 20 legislators matching kappa/pca fixtures."""
    rng = np.random.default_rng(RANDOM_SEED)
    half = n // 2
    xi = np.concatenate([rng.normal(2.0, 0.5, half), rng.normal(-2.0, 0.5, n - half)])
    parties = ["Republican"] * half + ["Democrat"] * (n - half)
    return pl.DataFrame(
        {
            "legislator_slug": [f"sen_leg{i}_1" for i in range(n)],
            "xi_mean": xi.tolist(),
            "xi_sd": [0.3] * n,
            "party": parties,
            "full_name": [f"Leg{i}" for i in range(n)],
        }
    )


# ── _kappa_to_distance tests ────────────────────────────────────────────────


class TestKappaToDistance:
    """Test the Kappa-to-distance conversion helper."""

    def test_returns_correct_shape(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Distance matrix should be n×n."""
        dist, slugs = _kappa_to_distance(kappa_matrix_20)
        assert dist.shape == (20, 20)
        assert len(slugs) == 20

    def test_diagonal_is_zero(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Self-distance should be zero."""
        dist, _ = _kappa_to_distance(kappa_matrix_20)
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)

    def test_symmetric(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Distance matrix should be symmetric."""
        dist, _ = _kappa_to_distance(kappa_matrix_20)
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_non_negative(self, kappa_matrix_20: pl.DataFrame) -> None:
        """All distances should be non-negative."""
        dist, _ = _kappa_to_distance(kappa_matrix_20)
        assert np.all(dist >= 0)

    def test_nan_filled_with_max(self) -> None:
        """NaN entries in the Kappa matrix should be filled with max distance."""
        data = {
            "legislator_slug": ["a", "b", "c"],
            "a": [1.0, 0.8, None],
            "b": [0.8, 1.0, 0.6],
            "c": [None, 0.6, 1.0],
        }
        kappa = pl.DataFrame(data)
        dist, _ = _kappa_to_distance(kappa, chamber="Test")
        assert not np.any(np.isnan(dist))
        # The NaN pairs should be filled with max finite distance
        max_d = dist.max()
        assert dist[0, 2] == pytest.approx(max_d)

    def test_distance_equals_one_minus_kappa(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Distance should be approximately 1 - kappa for well-behaved inputs."""
        dist, slugs = _kappa_to_distance(kappa_matrix_20)
        # Pick an off-diagonal element
        kappa_val = kappa_matrix_20[slugs[1]][0]
        assert dist[0, 1] == pytest.approx(1.0 - kappa_val, abs=0.01)


# ── _standardize_2d tests ────────────────────────────────────────────────────


class TestStandardize2D:
    """Test the 2D standardization helper."""

    def test_output_shape(self) -> None:
        """Should return an n×2 array."""
        xi = np.array([1.0, 2.0, 3.0, 4.0])
        loyalty = np.array([0.9, 0.8, 0.7, 0.6])
        result = _standardize_2d(xi, loyalty)
        assert result.shape == (4, 2)

    def test_zero_mean(self) -> None:
        """Each column should have approximately zero mean."""
        rng = np.random.default_rng(42)
        xi = rng.normal(5.0, 2.0, 50)
        loyalty = rng.uniform(0.5, 1.0, 50)
        result = _standardize_2d(xi, loyalty)
        assert abs(result[:, 0].mean()) < 1e-10
        assert abs(result[:, 1].mean()) < 1e-10

    def test_unit_std(self) -> None:
        """Each column should have approximately unit standard deviation."""
        rng = np.random.default_rng(42)
        xi = rng.normal(5.0, 2.0, 50)
        loyalty = rng.uniform(0.5, 1.0, 50)
        result = _standardize_2d(xi, loyalty)
        assert result[:, 0].std() == pytest.approx(1.0, abs=0.05)
        assert result[:, 1].std() == pytest.approx(1.0, abs=0.05)


# ── _build_display_labels tests ─────────────────────────────────────────────


class TestBuildDisplayLabels:
    """Test the display label builder for plot annotations."""

    def test_unique_last_names(self) -> None:
        """Unique last names should use last name only."""
        names = ["Joseph Claeys", "Mike Thompson", "Jane Smith"]
        labels = _build_display_labels(names)
        assert labels["Joseph Claeys"] == "Claeys"
        assert labels["Mike Thompson"] == "Thompson"

    def test_duplicate_last_names_disambiguated(self) -> None:
        """Duplicate last names should include abbreviated first name."""
        names = ["Joseph Smith", "Jane Smith", "Alice Jones"]
        labels = _build_display_labels(names)
        # Both Smiths should have first-name disambiguation
        assert "Smith" in labels["Joseph Smith"]
        assert "Smith" in labels["Jane Smith"]
        assert labels["Joseph Smith"] != labels["Jane Smith"]
        # Jones is unique → last name only
        assert labels["Alice Jones"] == "Jones"

    def test_leadership_suffix_stripped(self) -> None:
        """Leadership suffix '- Vice President of the Senate' should be stripped."""
        names = ["Tim Shallenburger - Vice President of the Senate", "Jane Doe"]
        labels = _build_display_labels(names)
        assert "Shallenburger" in labels[names[0]]
        assert "Vice" not in labels[names[0]]

    def test_empty_list(self) -> None:
        """Empty input should return empty dict."""
        assert _build_display_labels([]) == {}


# ── run_hierarchical tests ──────────────────────────────────────────────────


class TestRunHierarchical:
    """Test hierarchical clustering on Kappa distance matrix."""

    def test_returns_linkage_matrix(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Should return a valid scipy linkage matrix."""
        Z, coph, slugs, dist = run_hierarchical(kappa_matrix_20, "Senate")
        # Linkage matrix is (n-1) x 4
        assert Z.shape == (19, 4)
        assert Z.shape[1] == 4

    def test_cophenetic_positive(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Cophenetic correlation should be positive for structured data."""
        _, coph, _, _ = run_hierarchical(kappa_matrix_20, "Senate")
        assert coph > 0.0

    def test_returns_distance_array(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Should return the pre-computed distance array."""
        _, _, _, dist = run_hierarchical(kappa_matrix_20, "Senate")
        assert dist.shape == (20, 20)
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)

    def test_uses_average_linkage(self) -> None:
        """LINKAGE_METHOD should be average (not ward)."""
        assert LINKAGE_METHOD == "average"


# ── find_optimal_k_hierarchical tests ───────────────────────────────────────


class TestFindOptimalKHierarchical:
    """Test hierarchical silhouette-based k selection."""

    def test_returns_scores_for_all_k(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Should return a silhouette score for each k in range."""
        Z, _, _, dist = run_hierarchical(kappa_matrix_20, "Senate")
        scores, optimal_k = find_optimal_k_hierarchical(Z, dist, range(2, 6), "Senate")
        assert set(scores.keys()) == {2, 3, 4, 5}

    def test_optimal_k_has_best_silhouette(self, kappa_matrix_20: pl.DataFrame) -> None:
        """The returned optimal_k should have the highest silhouette score."""
        Z, _, _, dist = run_hierarchical(kappa_matrix_20, "Senate")
        scores, optimal_k = find_optimal_k_hierarchical(Z, dist, range(2, 6), "Senate")
        assert scores[optimal_k] == max(scores.values())

    def test_two_cluster_data_finds_k2(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Clear 2-cluster data should prefer k=2."""
        Z, _, _, dist = run_hierarchical(kappa_matrix_20, "Senate")
        _, optimal_k = find_optimal_k_hierarchical(Z, dist, range(2, 6), "Senate")
        assert optimal_k == 2

    def test_silhouette_values_valid_range(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Silhouette scores should be between -1 and 1."""
        Z, _, _, dist = run_hierarchical(kappa_matrix_20, "Senate")
        scores, _ = find_optimal_k_hierarchical(Z, dist, range(2, 6), "Senate")
        for s in scores.values():
            assert -1.0 <= s <= 1.0


# ── run_spectral_clustering tests ────────────────────────────────────────────


class TestRunSpectralClustering:
    """Test spectral clustering on Kappa agreement matrix."""

    def test_returns_results_for_all_k(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Should return results for each k in range."""
        results, optimal_k = run_spectral_clustering(kappa_matrix_20, range(2, 5), "Senate")
        assert set(results.keys()) == {2, 3, 4}

    def test_labels_correct_length(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Labels should have one entry per legislator."""
        results, optimal_k = run_spectral_clustering(kappa_matrix_20, range(2, 5), "Senate")
        for k in range(2, 5):
            assert len(results[k]["labels"]) == 20

    def test_silhouette_included(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Each k result should include a silhouette score."""
        results, _ = run_spectral_clustering(kappa_matrix_20, range(2, 5), "Senate")
        for k in range(2, 5):
            assert "silhouette" in results[k]
            assert -1.0 <= results[k]["silhouette"] <= 1.0

    def test_two_cluster_data_finds_k2(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Clear 2-cluster data should prefer k=2."""
        _, optimal_k = run_spectral_clustering(kappa_matrix_20, range(2, 5), "Senate")
        assert optimal_k == 2

    def test_labels_have_k_unique_values(self, kappa_matrix_20: pl.DataFrame) -> None:
        """Labels for k=2 should have exactly 2 unique values."""
        results, _ = run_spectral_clustering(kappa_matrix_20, range(2, 4), "Senate")
        assert len(set(results[2]["labels"])) == 2


# ── run_hdbscan_pca tests ───────────────────────────────────────────────────


class TestRunHDBSCAN:
    """Test HDBSCAN density-based clustering on PCA scores."""

    def test_returns_required_keys(self) -> None:
        """Result dict should contain all required keys."""
        pca = _make_pca_scores(20)
        ip = _make_ideal_points_20(20)
        result = run_hdbscan_pca(pca, ip, "Senate")
        assert "labels" in result
        assert "n_clusters" in result
        assert "n_noise" in result
        assert "slugs" in result

    def test_labels_length_matches_common_slugs(self) -> None:
        """Labels should have one entry per common legislator."""
        pca = _make_pca_scores(20)
        ip = _make_ideal_points_20(20)
        result = run_hdbscan_pca(pca, ip, "Senate")
        assert len(result["labels"]) == len(result["slugs"])

    def test_noise_count_non_negative(self) -> None:
        """Noise count should be non-negative."""
        pca = _make_pca_scores(20)
        ip = _make_ideal_points_20(20)
        result = run_hdbscan_pca(pca, ip, "Senate")
        assert result["n_noise"] >= 0

    def test_noise_slugs_have_minus_one_label(self) -> None:
        """Noise legislators should have label -1."""
        pca = _make_pca_scores(20)
        ip = _make_ideal_points_20(20)
        result = run_hdbscan_pca(pca, ip, "Senate")
        labels = result["labels"]
        slugs = result["slugs"]
        for slug in result["noise_slugs"]:
            idx = slugs.index(slug)
            assert labels[idx] == -1

    def test_n_pcs_used_reported(self) -> None:
        """Should report how many PCs were used."""
        pca = _make_pca_scores(20)
        ip = _make_ideal_points_20(20)
        result = run_hdbscan_pca(pca, ip, "Senate")
        assert result["n_pcs_used"] == 3  # Our fixture has 3 PCs

    def test_constants_defined(self) -> None:
        """HDBSCAN constants should be defined."""
        assert HDBSCAN_MIN_CLUSTER_SIZE >= 2
        assert HDBSCAN_MIN_SAMPLES >= 1


# ── characterize_clusters tests ─────────────────────────────────────────────


class TestCharacterizeClusters:
    """Test cluster characterization logic."""

    def test_returns_one_row_per_cluster(self) -> None:
        """Summary should have k rows."""
        ip = _make_ideal_points_20(20)
        labels = np.array([0] * 10 + [1] * 10)
        summary = characterize_clusters(ip, labels, None, 2, "Senate")
        assert summary.height == 2

    def test_n_legislators_sums_correctly(self) -> None:
        """Total n_legislators across clusters should equal total legislators."""
        ip = _make_ideal_points_20(20)
        labels = np.array([0] * 10 + [1] * 10)
        summary = characterize_clusters(ip, labels, None, 2, "Senate")
        total = summary["n_legislators"].sum()
        assert total == 20

    def test_party_counts_correct(self) -> None:
        """Party counts should match fixture data: R in cluster 0, D in cluster 1."""
        ip = _make_ideal_points_20(20)
        labels = np.array([0] * 10 + [1] * 10)
        summary = characterize_clusters(ip, labels, None, 2, "Senate")
        # Cluster 0 = first 10 = Rs, Cluster 1 = last 10 = Ds
        c0 = summary.filter(pl.col("cluster") == 0)
        assert c0["n_republican"][0] == 10
        assert c0["n_democrat"][0] == 0

    def test_n_other_counted(self) -> None:
        """n_other should be present and correct for non-R/D legislators."""
        n = 12
        ip = pl.DataFrame(
            {
                "legislator_slug": [f"sen_l{i}_1" for i in range(n)],
                "xi_mean": [float(i) for i in range(n)],
                "xi_sd": [0.3] * n,
                "party": ["Republican"] * 5 + ["Democrat"] * 5 + ["Independent"] * 2,
                "full_name": [f"L{i}" for i in range(n)],
            }
        )
        labels = np.array([0] * 6 + [1] * 6)
        summary = characterize_clusters(ip, labels, None, 2, "Senate")
        assert "n_other" in summary.columns
        total_other = summary["n_other"].sum()
        assert total_other == 2

    def test_loyalty_included_when_provided(self) -> None:
        """avg_loyalty should be non-null when loyalty data is provided."""
        ip = _make_ideal_points_20(20)
        loyalty = pl.DataFrame(
            {
                "legislator_slug": [f"sen_leg{i}_1" for i in range(20)],
                "loyalty_rate": [0.9] * 10 + [0.8] * 10,
            }
        )
        labels = np.array([0] * 10 + [1] * 10)
        summary = characterize_clusters(ip, labels, loyalty, 2, "Senate")
        for row in summary.iter_rows(named=True):
            assert row["avg_loyalty"] is not None
            assert 0.0 <= row["avg_loyalty"] <= 1.0

    def test_heuristic_label_assigned(self) -> None:
        """Clusters should get heuristic labels based on composition."""
        ip = _make_ideal_points_20(20)
        labels = np.array([0] * 10 + [1] * 10)
        summary = characterize_clusters(ip, labels, None, 2, "Senate")
        # Cluster 0 (Rs with xi > 0) should be labeled something with R
        label0 = summary.filter(pl.col("cluster") == 0)["label"][0]
        assert "R" in label0 or "Republican" in label0


# ── run_kmeans_irt tests ────────────────────────────────────────────────────


class TestRunKMeansIRT:
    """Test k-means on IRT ideal points."""

    def test_returns_results_for_all_k(self) -> None:
        """Should return results for each k in range."""
        ip = _make_ideal_points_20(20)
        results, optimal_k = run_kmeans_irt(ip, None, range(2, 5), "Senate")
        assert set(results.keys()) == {2, 3, 4}

    def test_labels_have_correct_k(self) -> None:
        """Labels for k=2 should have 2 unique values."""
        ip = _make_ideal_points_20(20)
        results, _ = run_kmeans_irt(ip, None, range(2, 5), "Senate")
        assert len(set(results[2]["labels_1d"])) == 2

    def test_two_cluster_data_finds_k2(self) -> None:
        """Clear 2-cluster data should prefer k=2."""
        ip = _make_ideal_points_20(20)
        _, optimal_k = run_kmeans_irt(ip, None, range(2, 5), "Senate")
        assert optimal_k == 2

    def test_inertia_decreases_with_k(self) -> None:
        """Inertia should monotonically decrease as k increases."""
        ip = _make_ideal_points_20(20)
        results, _ = run_kmeans_irt(ip, None, range(2, 6), "Senate")
        inertias = [results[k]["inertia"] for k in range(2, 6)]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_2d_kmeans_with_loyalty(self) -> None:
        """2D k-means should run when loyalty data is provided."""
        ip = _make_ideal_points_20(20)
        loyalty = pl.DataFrame(
            {
                "legislator_slug": [f"sen_leg{i}_1" for i in range(20)],
                "loyalty_rate": np.linspace(0.5, 1.0, 20).tolist(),
                "n_contested_votes": [50] * 20,
                "n_agree": [40] * 20,
                "party": ["Republican"] * 10 + ["Democrat"] * 10,
                "full_name": [f"Leg{i}" for i in range(20)],
            }
        )
        results, _ = run_kmeans_irt(ip, loyalty, range(2, 5), "Senate")
        assert "labels_2d" in results[2]
        assert "silhouette_2d" in results[2]


# ── Additional constants tests ───────────────────────────────────────────────


class TestNewConstants:
    """Verify new and renamed constants have expected values."""

    def test_comparison_k(self) -> None:
        """COMPARISON_K should be 3 (k=3 for downstream comparison)."""
        assert COMPARISON_K == 3

    def test_linkage_method_average(self) -> None:
        """LINKAGE_METHOD should be 'average' (not 'ward')."""
        assert LINKAGE_METHOD == "average"

    def test_hdbscan_min_cluster_size(self) -> None:
        assert HDBSCAN_MIN_CLUSTER_SIZE == 5

    def test_hdbscan_min_samples(self) -> None:
        assert HDBSCAN_MIN_SAMPLES == 3
