"""
Tests for LCA (Latent Class Analysis) helper functions using synthetic fixtures.

These tests verify that the non-plotting functions in analysis/10_lca/lca.py
produce correct results on known inputs. Plotting and full pipeline execution
are not tested here; data preparation, class enumeration, Salsa effect
detection, IRT cross-validation, and ARI comparison are.

Run: uv run pytest tests/test_lca.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.lca import (
    K_MAX,
    MIN_CLASS_FRACTION,
    MIN_VOTES,
    CONTESTED_THRESHOLD,
    N_INIT,
    RANDOM_SEED,
    SALSA_THRESHOLD,
    build_vote_matrix,
    compare_with_clustering,
    cross_validate_irt,
    detect_salsa_effect,
    enumerate_classes,
    find_discriminating_bills,
    fit_final_model,
    select_optimal_k,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vote_matrix_pl() -> pl.DataFrame:
    """Synthetic vote matrix: 20 legislators x 10 votes.

    Two clear clusters: R1-R10 (mostly Yea on v1-v5, Nay on v6-v10)
    and D1-D10 (opposite pattern).
    """
    rng = np.random.default_rng(42)
    slugs = [f"rep_r{i}_1" for i in range(1, 11)] + [f"rep_d{i}_1" for i in range(1, 11)]
    data: dict[str, list] = {"legislator_slug": slugs}

    for v in range(1, 11):
        col = []
        for i in range(20):
            if i < 10:  # Republicans
                p = 0.9 if v <= 5 else 0.1
            else:  # Democrats
                p = 0.1 if v <= 5 else 0.9
            col.append(int(rng.random() < p))
        data[f"v{v}"] = col

    return pl.DataFrame(data)


@pytest.fixture
def vote_array_2class(vote_matrix_pl: pl.DataFrame) -> np.ndarray:
    """Numpy array version of 2-class vote matrix."""
    arr, _, _ = build_vote_matrix(vote_matrix_pl)
    return arr


@pytest.fixture
def ideal_points() -> pl.DataFrame:
    """IRT ideal points matching the 20-legislator fixture."""
    slugs = [f"rep_r{i}_1" for i in range(1, 11)] + [f"rep_d{i}_1" for i in range(1, 11)]
    xis = list(np.linspace(2.0, 0.5, 10)) + list(np.linspace(-0.5, -2.0, 10))
    parties = ["Republican"] * 10 + ["Democrat"] * 10
    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "xi_mean": xis,
            "xi_sd": [0.1] * 20,
            "party": parties,
            "full_name": [f"Legislator {s}" for s in slugs],
        }
    )


@pytest.fixture
def legislators() -> pl.DataFrame:
    """Legislator metadata matching the fixture."""
    slugs = [f"rep_r{i}_1" for i in range(1, 11)] + [f"rep_d{i}_1" for i in range(1, 11)]
    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "full_name": [f"Legislator {s}" for s in slugs],
            "party": ["Republican"] * 10 + ["Democrat"] * 10,
        }
    )


# ── Vote Matrix Construction ────────────────────────────────────────────────


class TestBuildVoteMatrix:
    """Tests for build_vote_matrix()."""

    def test_shape(self, vote_matrix_pl: pl.DataFrame) -> None:
        arr, slugs, vote_ids = build_vote_matrix(vote_matrix_pl)
        assert arr.shape == (20, 10)
        assert len(slugs) == 20
        assert len(vote_ids) == 10

    def test_slug_order(self, vote_matrix_pl: pl.DataFrame) -> None:
        _, slugs, _ = build_vote_matrix(vote_matrix_pl)
        assert slugs[0] == "rep_r1_1"
        assert slugs[10] == "rep_d1_1"

    def test_vote_ids(self, vote_matrix_pl: pl.DataFrame) -> None:
        _, _, vote_ids = build_vote_matrix(vote_matrix_pl)
        assert vote_ids == [f"v{i}" for i in range(1, 11)]

    def test_binary_values(self, vote_matrix_pl: pl.DataFrame) -> None:
        arr, _, _ = build_vote_matrix(vote_matrix_pl)
        valid = arr[~np.isnan(arr)]
        assert set(valid.astype(int)) <= {0, 1}

    def test_nan_handling(self) -> None:
        """Missing values should become NaN in the numpy array."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a", "b"],
                "v1": [1, None],
                "v2": [0, 1],
            }
        )
        arr, _, _ = build_vote_matrix(df)
        assert np.isnan(arr[1, 0])
        assert arr[0, 0] == 1.0


# ── Class Enumeration ────────────────────────────────────────────────────────


class TestEnumerateClasses:
    """Tests for enumerate_classes() and select_optimal_k()."""

    def test_returns_all_k(self, vote_array_2class: np.ndarray) -> None:
        results = enumerate_classes(vote_array_2class, k_max=4, n_init=5)
        assert len(results) == 4
        ks = [r["k"] for r in results]
        assert ks == [1, 2, 3, 4]

    def test_has_required_fields(self, vote_array_2class: np.ndarray) -> None:
        results = enumerate_classes(vote_array_2class, k_max=2, n_init=5)
        for r in results:
            assert "k" in r
            assert "bic" in r
            assert "aic" in r
            assert "log_likelihood" in r
            assert "entropy" in r
            assert "converged" in r

    def test_bic_selects_2_for_two_clusters(self, vote_array_2class: np.ndarray) -> None:
        """With two clearly separated groups, BIC should select K=2."""
        results = enumerate_classes(vote_array_2class, k_max=5, n_init=10)
        optimal_k, _ = select_optimal_k(results)
        assert optimal_k == 2, f"Expected K=2, got K={optimal_k}"

    def test_select_optimal_k_returns_rationale(self, vote_array_2class: np.ndarray) -> None:
        results = enumerate_classes(vote_array_2class, k_max=3, n_init=5)
        _, rationale = select_optimal_k(results)
        assert "BIC" in rationale
        assert isinstance(rationale, str)

    def test_entropy_k1_is_one(self, vote_array_2class: np.ndarray) -> None:
        """K=1 should have entropy=1.0 (perfect certainty)."""
        results = enumerate_classes(vote_array_2class, k_max=1, n_init=5)
        assert results[0]["entropy"] == 1.0


# ── Fit Final Model ──────────────────────────────────────────────────────────


class TestFitFinalModel:
    """Tests for fit_final_model()."""

    def test_returns_correct_shapes(self, vote_array_2class: np.ndarray) -> None:
        _, labels, probs, profiles = fit_final_model(vote_array_2class, k=2)
        assert labels.shape == (20,)
        assert probs.shape == (20, 2)
        assert profiles.shape[0] == 2
        assert profiles.shape[1] == vote_array_2class.shape[1]

    def test_labels_are_integers(self, vote_array_2class: np.ndarray) -> None:
        _, labels, _, _ = fit_final_model(vote_array_2class, k=2)
        assert set(labels) <= {0, 1}

    def test_probabilities_sum_to_one(self, vote_array_2class: np.ndarray) -> None:
        _, _, probs, _ = fit_final_model(vote_array_2class, k=2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_profiles_bounded(self, vote_array_2class: np.ndarray) -> None:
        _, _, _, profiles = fit_final_model(vote_array_2class, k=2)
        assert np.all(profiles >= 0)
        assert np.all(profiles <= 1)


# ── Salsa Effect Detection ───────────────────────────────────────────────────


class TestDetectSalsaEffect:
    """Tests for detect_salsa_effect()."""

    def test_parallel_profiles_flagged(self) -> None:
        """Two profiles that are scaled versions should trigger Salsa."""
        p1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        p2 = np.array([0.7, 0.6, 0.5, 0.4, 0.3])
        profiles = np.vstack([p1, p2])
        result = detect_salsa_effect(profiles)
        assert result["is_salsa"]
        assert result["mean_correlation"] > SALSA_THRESHOLD

    def test_distinct_profiles_not_flagged(self) -> None:
        """Profiles with different patterns should not trigger Salsa."""
        p1 = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
        p2 = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        profiles = np.vstack([p1, p2])
        result = detect_salsa_effect(profiles)
        assert not result["is_salsa"]
        assert result["mean_correlation"] < 0

    def test_single_class(self) -> None:
        """K=1 should return is_salsa=False and 'not applicable'."""
        profiles = np.array([[0.5, 0.6, 0.7]])
        result = detect_salsa_effect(profiles)
        assert not result["is_salsa"]
        assert "not applicable" in result["verdict"].lower()

    def test_three_class_mixed(self) -> None:
        """Three profiles: two similar, one different → not universal Salsa."""
        p1 = np.array([0.9, 0.8, 0.7, 0.6])
        p2 = np.array([0.7, 0.6, 0.5, 0.4])
        p3 = np.array([0.1, 0.9, 0.1, 0.9])
        profiles = np.vstack([p1, p2, p3])
        result = detect_salsa_effect(profiles)
        # min_correlation should be low due to p3 being different
        assert not result["is_salsa"]
        assert result["correlation_matrix"].shape == (3, 3)


# ── IRT Cross-Validation ────────────────────────────────────────────────────


class TestCrossValidateIrt:
    """Tests for cross_validate_irt()."""

    def test_monotonicity_with_known_ordering(self, ideal_points: pl.DataFrame) -> None:
        """Labels that match IRT ordering should be monotonic."""
        slugs = ideal_points["legislator_slug"].to_list()
        # Assign labels: 0 for high xi (Republicans), 1 for low xi (Democrats)
        labels = np.array([0] * 10 + [1] * 10)
        probs = np.zeros((20, 2))
        for i in range(20):
            probs[i, labels[i]] = 0.95
            probs[i, 1 - labels[i]] = 0.05

        result = cross_validate_irt(labels, probs, slugs, ideal_points)
        assert result["is_monotonic"]
        assert len(result["class_stats"]) == 2

    def test_straddler_detection(self, ideal_points: pl.DataFrame) -> None:
        """Legislators with max P < 0.7 should be identified as straddlers."""
        slugs = ideal_points["legislator_slug"].to_list()
        labels = np.array([0] * 10 + [1] * 10)
        probs = np.full((20, 2), 0.5)  # All uncertain → all straddlers
        probs[:, 0] = 0.5
        probs[:, 1] = 0.5

        result = cross_validate_irt(labels, probs, slugs, ideal_points)
        assert result["n_straddlers"] == 20

    def test_class_stats_fields(self, ideal_points: pl.DataFrame) -> None:
        """Each class stat dict should have required fields."""
        slugs = ideal_points["legislator_slug"].to_list()
        labels = np.array([0] * 10 + [1] * 10)
        probs = np.zeros((20, 2))
        for i in range(20):
            probs[i, labels[i]] = 0.9
            probs[i, 1 - labels[i]] = 0.1

        result = cross_validate_irt(labels, probs, slugs, ideal_points)
        for cs in result["class_stats"]:
            assert "class" in cs
            assert "n" in cs
            assert "mean_xi" in cs
            assert "median_xi" in cs
            assert "sd_xi" in cs


# ── Clustering Agreement ────────────────────────────────────────────────────


class TestCompareWithClustering:
    """Tests for compare_with_clustering()."""

    def test_identical_labels(self) -> None:
        """Identical labels should produce ARI=1.0."""
        labels = np.array([0, 0, 1, 1, 0, 1])
        slugs = [f"leg_{i}" for i in range(6)]
        cl_labels = {"hierarchical": labels.copy()}
        result = compare_with_clustering(labels, slugs, cl_labels)
        assert "lca_vs_hierarchical" in result
        assert abs(result["lca_vs_hierarchical"] - 1.0) < 1e-10

    def test_different_labels(self) -> None:
        """Random vs structured labels should produce low ARI."""
        lca_labels = np.array([0, 0, 0, 1, 1, 1])
        cl_labels = {"kmeans": np.array([1, 0, 1, 0, 1, 0])}
        slugs = [f"leg_{i}" for i in range(6)]
        result = compare_with_clustering(lca_labels, slugs, cl_labels)
        assert result["lca_vs_kmeans"] < 0.5

    def test_none_clustering_returns_empty(self) -> None:
        """None clustering labels should return empty dict."""
        labels = np.array([0, 1])
        slugs = ["a", "b"]
        result = compare_with_clustering(labels, slugs, None)
        assert result == {}


# ── Within-Party Edge Cases ──────────────────────────────────────────────────


class TestWithinPartyEdgeCases:
    """Tests for within-party LCA edge cases."""

    def test_small_party_skipped(self, legislators: pl.DataFrame) -> None:
        """Parties with < 10 legislators should be skipped."""
        from analysis.lca import run_within_party_lca

        # Create a tiny vote array (only 5 R legislators)
        small_array = np.random.default_rng(42).integers(0, 2, size=(5, 10)).astype(float)
        small_slugs = [f"rep_r{i}_1" for i in range(1, 6)]

        # Only 5 Republicans → should skip
        result = run_within_party_lca(small_array, small_slugs, legislators, "House", k_max=3)
        assert result["Republican"]["skipped"]
        assert "Too few" in result["Republican"].get("reason", "")


# ── Discriminating Bills ─────────────────────────────────────────────────────


class TestFindDiscriminatingBills:
    """Tests for find_discriminating_bills()."""

    def test_returns_correct_count(self) -> None:
        profiles = np.array(
            [
                [0.9, 0.5, 0.1],
                [0.1, 0.5, 0.9],
            ]
        )
        vote_ids = ["v1", "v2", "v3"]
        result = find_discriminating_bills(profiles, vote_ids, n_top=2)
        assert len(result) == 2

    def test_highest_range_first(self) -> None:
        profiles = np.array(
            [
                [0.9, 0.5, 0.1],
                [0.1, 0.5, 0.9],
            ]
        )
        vote_ids = ["v1", "v2", "v3"]
        result = find_discriminating_bills(profiles, vote_ids, n_top=3)
        # v1 and v3 have range 0.8, v2 has range 0.0
        assert result[0]["range"] > result[-1]["range"]

    def test_profiles_included(self) -> None:
        profiles = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        )
        vote_ids = ["v1", "v2"]
        result = find_discriminating_bills(profiles, vote_ids, n_top=2)
        assert "profiles" in result[0]
        assert len(result[0]["profiles"]) == 2


# ── Constants Consistency ────────────────────────────────────────────────────


class TestConstants:
    """Verify constants match project-wide values."""

    def test_random_seed(self) -> None:
        assert RANDOM_SEED == 42

    def test_min_votes(self) -> None:
        assert MIN_VOTES == 20

    def test_minority_threshold(self) -> None:
        assert CONTESTED_THRESHOLD == 0.025

    def test_salsa_threshold(self) -> None:
        assert SALSA_THRESHOLD == 0.80

    def test_min_class_fraction(self) -> None:
        assert MIN_CLASS_FRACTION == 0.05

    def test_k_max(self) -> None:
        assert K_MAX == 8

    def test_n_init(self) -> None:
        assert N_INIT == 50


class TestConstantsConsistency:
    """Verify shared constants match between lca.py and clustering.py."""

    def test_minority_threshold_matches_clustering(self) -> None:
        from analysis.clustering import CONTESTED_THRESHOLD as CLUSTERING_MINORITY

        assert CONTESTED_THRESHOLD == CLUSTERING_MINORITY

    def test_random_seed_matches_clustering(self) -> None:
        from analysis.clustering import RANDOM_SEED as CLUSTERING_SEED

        assert RANDOM_SEED == CLUSTERING_SEED
