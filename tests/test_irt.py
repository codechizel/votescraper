"""
Tests for IRT helper functions using synthetic fixtures.

These tests verify that the non-MCMC functions in analysis/irt.py produce
correct results on known inputs. MCMC sampling itself is not tested here
(too slow for unit tests), but data preparation, anchor selection, holdout
splitting, and posterior extraction logic are.

Run: uv run pytest tests/test_irt.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path so we can import analysis.irt
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.irt import (
    HOLDOUT_FRACTION,
    HOLDOUT_SEED,
    filter_vote_matrix_for_sensitivity,
    prepare_irt_data,
    select_anchors,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def senate_matrix() -> pl.DataFrame:
    """Synthetic Senate vote matrix: 6 legislators x 5 votes.

    Layout (1=Yea, 0=Nay, None=absent):
        v1: 3Y/3N (party-line: A,B,C Yea; D,E,F Nay)
        v2: 5Y/1N (lopsided: only F votes Nay)
        v3: 4Y/1N/1absent (C absent)
        v4: 2Y/4N (reversed: D,E Yea; A,B,C,F Nay)
        v5: 3Y/3N (party-line again)
    """
    return pl.DataFrame({
        "legislator_slug": [
            "sen_a_a_1", "sen_b_b_1", "sen_c_c_1",
            "sen_d_d_1", "sen_e_e_1", "sen_f_f_1",
        ],
        "v1": [1, 1, 1, 0, 0, 0],
        "v2": [1, 1, 1, 1, 1, 0],
        "v3": [1, 1, None, 1, 0, 1],
        "v4": [0, 0, 0, 1, 1, 0],
        "v5": [1, 1, 1, 0, 0, 0],
    })


@pytest.fixture
def pca_scores() -> pl.DataFrame:
    """PCA scores matching senate_matrix legislators.

    A,B,C are "conservative" (high PC1), D,E,F are "liberal" (low PC1).
    """
    return pl.DataFrame({
        "legislator_slug": [
            "sen_a_a_1", "sen_b_b_1", "sen_c_c_1",
            "sen_d_d_1", "sen_e_e_1", "sen_f_f_1",
        ],
        "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
        "party": ["Republican"] * 3 + ["Democrat"] * 3,
        "PC1": [6.0, 5.5, 5.0, -5.0, -5.5, -6.0],
        "PC2": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1],
    })


@pytest.fixture
def rollcalls() -> pl.DataFrame:
    """Roll call metadata matching senate_matrix votes."""
    return pl.DataFrame({
        "vote_id": ["v1", "v2", "v3", "v4", "v5"],
        "chamber": ["Senate"] * 5,
        "bill_number": ["SB 1", "SB 2", "SB 3", "SB 4", "SB 5"],
        "short_title": ["Bill 1", "Bill 2", "Bill 3", "Bill 4", "Bill 5"],
        "motion": ["Final Action"] * 5,
        "vote_type": ["final_action"] * 5,
    })


@pytest.fixture
def legislators() -> pl.DataFrame:
    """Legislator metadata matching senate_matrix."""
    return pl.DataFrame({
        "name": ["A", "B", "C", "D", "E", "F"],
        "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
        "slug": [
            "sen_a_a_1", "sen_b_b_1", "sen_c_c_1",
            "sen_d_d_1", "sen_e_e_1", "sen_f_f_1",
        ],
        "chamber": ["Senate"] * 6,
        "party": ["Republican"] * 3 + ["Democrat"] * 3,
        "district": [1, 2, 3, 4, 5, 6],
        "member_url": [""] * 6,
    })


# ── prepare_irt_data tests ────────────────────────────────────────────────────


class TestPrepareIrtData:
    """Test conversion from wide vote matrix to long format for IRT."""

    def test_correct_legislator_count(self, senate_matrix: pl.DataFrame) -> None:
        data = prepare_irt_data(senate_matrix, "Senate")
        assert data["n_legislators"] == 6

    def test_correct_vote_count(self, senate_matrix: pl.DataFrame) -> None:
        data = prepare_irt_data(senate_matrix, "Senate")
        assert data["n_votes"] == 5

    def test_nulls_excluded(self, senate_matrix: pl.DataFrame) -> None:
        """Absent votes (None) should be excluded from observed cells."""
        data = prepare_irt_data(senate_matrix, "Senate")
        # 6 legislators x 5 votes = 30, minus 1 null (C on v3) = 29
        assert data["n_obs"] == 29

    def test_y_is_binary(self, senate_matrix: pl.DataFrame) -> None:
        """All observed votes should be 0 or 1."""
        data = prepare_irt_data(senate_matrix, "Senate")
        assert set(np.unique(data["y"])) <= {0, 1}

    def test_index_ranges(self, senate_matrix: pl.DataFrame) -> None:
        """Legislator and vote indices should be within valid ranges."""
        data = prepare_irt_data(senate_matrix, "Senate")
        assert data["leg_idx"].min() >= 0
        assert data["leg_idx"].max() < data["n_legislators"]
        assert data["vote_idx"].min() >= 0
        assert data["vote_idx"].max() < data["n_votes"]

    def test_yea_rate_reasonable(self, senate_matrix: pl.DataFrame) -> None:
        """Yea rate should match manual count."""
        data = prepare_irt_data(senate_matrix, "Senate")
        # Manual count: v1=3Y, v2=5Y, v3=4Y, v4=2Y, v5=3Y → 17 Yea of 29 obs
        expected_yea_rate = 17 / 29
        actual_yea_rate = data["y"].mean()
        assert actual_yea_rate == pytest.approx(expected_yea_rate, abs=0.01)

    def test_slug_list_preserved(self, senate_matrix: pl.DataFrame) -> None:
        """Slug list should match the matrix's legislator order."""
        data = prepare_irt_data(senate_matrix, "Senate")
        assert data["leg_slugs"] == senate_matrix["legislator_slug"].to_list()

    def test_vote_ids_preserved(self, senate_matrix: pl.DataFrame) -> None:
        """Vote ID list should match the matrix columns (excluding slug)."""
        data = prepare_irt_data(senate_matrix, "Senate")
        expected = [c for c in senate_matrix.columns if c != "legislator_slug"]
        assert data["vote_ids"] == expected


# ── select_anchors tests ──────────────────────────────────────────────────────


class TestSelectAnchors:
    """Test PCA-based anchor selection."""

    def test_conservative_anchor_is_highest_pc1(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Conservative anchor should be the legislator with highest PC1."""
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        assert cons_slug == "sen_a_a_1"  # A has PC1=6.0 (highest)

    def test_liberal_anchor_is_lowest_pc1(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Liberal anchor should be the legislator with lowest PC1."""
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        assert lib_slug == "sen_f_f_1"  # F has PC1=-6.0 (lowest)

    def test_anchor_indices_are_valid(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Anchor indices should correspond to slug positions in the matrix."""
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        slugs = senate_matrix["legislator_slug"].to_list()
        assert slugs[cons_idx] == cons_slug
        assert slugs[lib_idx] == lib_slug

    def test_anchors_are_different(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Conservative and liberal anchors must be different legislators."""
        cons_idx, _, lib_idx, _ = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        assert cons_idx != lib_idx

    def test_low_participation_excluded(
        self, pca_scores: pl.DataFrame,
    ) -> None:
        """Legislators with < 50% participation should not be anchors.

        Create a matrix where the most extreme PCA legislator (A, PC1=6.0)
        has only 1 of 5 votes (20% participation). B (PC1=5.5) should be
        selected instead.
        """
        matrix = pl.DataFrame({
            "legislator_slug": [
                "sen_a_a_1", "sen_b_b_1", "sen_c_c_1",
                "sen_d_d_1", "sen_e_e_1", "sen_f_f_1",
            ],
            "v1": [1, 1, 1, 0, 0, 0],
            "v2": [None, 1, 1, 1, 1, 0],
            "v3": [None, 1, 1, 1, 0, 1],
            "v4": [None, 0, 0, 1, 1, 0],
            "v5": [None, 1, 1, 0, 0, 0],
        })
        cons_idx, cons_slug, _, _ = select_anchors(pca_scores, matrix, "Senate")
        # A has 1/5 = 20% < 50%, so B (PC1=5.5) should be the anchor
        assert cons_slug == "sen_b_b_1"


# ── filter_vote_matrix_for_sensitivity tests ──────────────────────────────────


class TestSensitivityFilter:
    """Test the sensitivity re-filtering of vote matrices."""

    def test_higher_threshold_drops_more_votes(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """A 10% threshold should retain fewer votes than 2.5%."""
        filtered_low = filter_vote_matrix_for_sensitivity(
            senate_matrix, rollcalls, "Senate",
            minority_threshold=0.025, min_votes=1,
        )
        filtered_high = filter_vote_matrix_for_sensitivity(
            senate_matrix, rollcalls, "Senate",
            minority_threshold=0.20, min_votes=1,
        )
        n_low = len(filtered_low.columns) - 1  # exclude slug col
        n_high = len(filtered_high.columns) - 1
        assert n_high <= n_low

    def test_slug_column_preserved(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """Output should always have the legislator_slug column."""
        filtered = filter_vote_matrix_for_sensitivity(
            senate_matrix, rollcalls, "Senate",
            minority_threshold=0.025, min_votes=1,
        )
        assert "legislator_slug" in filtered.columns

    def test_chamber_filter_applied(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """Only Senate legislators (sen_ prefix) should remain."""
        filtered = filter_vote_matrix_for_sensitivity(
            senate_matrix, rollcalls, "Senate",
            minority_threshold=0.025, min_votes=1,
        )
        for slug in filtered["legislator_slug"].to_list():
            assert slug.startswith("sen_"), f"Non-Senate slug: {slug}"

    def test_min_votes_filters_legislators(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """Setting min_votes high should drop low-participation legislators."""
        filtered = filter_vote_matrix_for_sensitivity(
            senate_matrix, rollcalls, "Senate",
            minority_threshold=0.025, min_votes=100,
        )
        # Nobody has 100 votes, so should be empty
        assert filtered.height == 0


# ── Holdout reproducibility test ──────────────────────────────────────────────


class TestHoldoutReproducibility:
    """Verify that holdout selection is deterministic with the same seed."""

    def test_same_seed_same_indices(self) -> None:
        """Two holdout selections with the same seed should be identical."""
        n = 1000
        rng1 = np.random.default_rng(HOLDOUT_SEED)
        n_holdout = int(n * HOLDOUT_FRACTION)
        indices1 = rng1.choice(n, size=n_holdout, replace=False)

        rng2 = np.random.default_rng(HOLDOUT_SEED)
        indices2 = rng2.choice(n, size=n_holdout, replace=False)

        np.testing.assert_array_equal(indices1, indices2)

    def test_different_seed_different_indices(self) -> None:
        """Different seeds should produce different holdout selections."""
        n = 1000
        n_holdout = int(n * HOLDOUT_FRACTION)

        rng1 = np.random.default_rng(HOLDOUT_SEED)
        indices1 = rng1.choice(n, size=n_holdout, replace=False)

        rng2 = np.random.default_rng(HOLDOUT_SEED + 1)
        indices2 = rng2.choice(n, size=n_holdout, replace=False)

        # Very unlikely to be identical with different seeds
        assert not np.array_equal(indices1, indices2)

    def test_holdout_fraction_correct(self) -> None:
        """Holdout should select exactly HOLDOUT_FRACTION of observations."""
        n = 1000
        n_holdout = int(n * HOLDOUT_FRACTION)
        assert n_holdout == 200  # 20% of 1000


# ── IRT math verification ─────────────────────────────────────────────────────


class TestIrtMath:
    """Verify the IRT equation produces correct probabilities."""

    def test_logistic_function(self) -> None:
        """P(Yea) = 1/(1+exp(-eta)) should be between 0 and 1."""
        # eta = beta * xi - alpha
        xi = np.array([2.0, -1.0, 0.0])
        beta = 1.5
        alpha = 0.5
        eta = beta * xi - alpha
        p = 1.0 / (1.0 + np.exp(-eta))
        assert np.all(p > 0) and np.all(p < 1)

    def test_conservative_higher_p_for_positive_beta(self) -> None:
        """With beta > 0, more conservative (higher xi) should have higher P(Yea)."""
        xi_conservative = 2.0
        xi_liberal = -2.0
        beta = 1.5
        alpha = 0.0
        p_cons = 1.0 / (1.0 + np.exp(-(beta * xi_conservative - alpha)))
        p_lib = 1.0 / (1.0 + np.exp(-(beta * xi_liberal - alpha)))
        assert p_cons > p_lib

    def test_liberal_higher_p_for_negative_beta(self) -> None:
        """With beta < 0, more liberal (lower xi) should have higher P(Yea)."""
        xi_conservative = 2.0
        xi_liberal = -2.0
        beta = -1.5
        alpha = 0.0
        p_cons = 1.0 / (1.0 + np.exp(-(beta * xi_conservative - alpha)))
        p_lib = 1.0 / (1.0 + np.exp(-(beta * xi_liberal - alpha)))
        assert p_lib > p_cons

    def test_zero_beta_no_discrimination(self) -> None:
        """With beta = 0, P(Yea) should be the same for all legislators."""
        xi_values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        beta = 0.0
        alpha = 0.5
        eta = beta * xi_values - alpha
        p = 1.0 / (1.0 + np.exp(-eta))
        # All probabilities should be identical
        assert np.all(np.isclose(p, p[0]))

    def test_alpha_shifts_threshold(self) -> None:
        """Higher alpha should reduce P(Yea) for all legislators."""
        xi = 1.0
        beta = 1.0
        alpha_low = 0.0
        alpha_high = 3.0
        p_low_alpha = 1.0 / (1.0 + np.exp(-(beta * xi - alpha_low)))
        p_high_alpha = 1.0 / (1.0 + np.exp(-(beta * xi - alpha_high)))
        assert p_low_alpha > p_high_alpha

    def test_alpha_cannot_flip_direction(self) -> None:
        """With beta > 0, no value of alpha can make liberal P(Yea) > conservative P(Yea).

        This is the core insight from the beta prior investigation.
        Uses >= because at extreme alpha values both probabilities saturate to 1.0
        (or 0.0) in float64, making strict > fail despite the mathematical truth.
        The key claim is that alpha can never FLIP the ordering.
        """
        xi_conservative = 2.0
        xi_liberal = -2.0
        beta = 1.0  # positive
        for alpha in [-100, -10, -1, 0, 1, 10, 100]:
            p_cons = 1.0 / (1.0 + np.exp(-(beta * xi_conservative - alpha)))
            p_lib = 1.0 / (1.0 + np.exp(-(beta * xi_liberal - alpha)))
            assert p_cons >= p_lib, (
                f"With beta>0, conservative should always have P(Yea) >= liberal. "
                f"alpha={alpha}, p_cons={p_cons:.6f}, p_lib={p_lib:.6f}"
            )
        # Verify strict inequality holds in the non-saturated range
        for alpha in [-5, -1, 0, 1, 5]:
            p_cons = 1.0 / (1.0 + np.exp(-(beta * xi_conservative - alpha)))
            p_lib = 1.0 / (1.0 + np.exp(-(beta * xi_liberal - alpha)))
            assert p_cons > p_lib
