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

import arviz as az
import xarray as xr
from analysis.irt import (
    HOLDOUT_FRACTION,
    HOLDOUT_SEED,
    MAX_DIVERGENCES,
    PARADOX_YEA_GAP,
    RHAT_THRESHOLD,
    _detect_forest_highlights,
    build_irt_graph,
    build_joint_vote_matrix,
    check_convergence,
    equate_chambers,
    extract_bill_parameters,
    extract_ideal_points,
    filter_vote_matrix_for_sensitivity,
    find_paradox_legislator,
    prepare_irt_data,
    select_anchors,
    unmerge_bridging_legislators,
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
    return pl.DataFrame(
        {
            "legislator_slug": [
                "sen_a_a_1",
                "sen_b_b_1",
                "sen_c_c_1",
                "sen_d_d_1",
                "sen_e_e_1",
                "sen_f_f_1",
            ],
            "v1": [1, 1, 1, 0, 0, 0],
            "v2": [1, 1, 1, 1, 1, 0],
            "v3": [1, 1, None, 1, 0, 1],
            "v4": [0, 0, 0, 1, 1, 0],
            "v5": [1, 1, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def pca_scores() -> pl.DataFrame:
    """PCA scores matching senate_matrix legislators.

    A,B,C are "conservative" (high PC1), D,E,F are "liberal" (low PC1).
    """
    return pl.DataFrame(
        {
            "legislator_slug": [
                "sen_a_a_1",
                "sen_b_b_1",
                "sen_c_c_1",
                "sen_d_d_1",
                "sen_e_e_1",
                "sen_f_f_1",
            ],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
            "party": ["Republican"] * 3 + ["Democrat"] * 3,
            "PC1": [6.0, 5.5, 5.0, -5.0, -5.5, -6.0],
            "PC2": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1],
        }
    )


@pytest.fixture
def rollcalls() -> pl.DataFrame:
    """Roll call metadata matching senate_matrix votes."""
    return pl.DataFrame(
        {
            "vote_id": ["v1", "v2", "v3", "v4", "v5"],
            "chamber": ["Senate"] * 5,
            "bill_number": ["SB 1", "SB 2", "SB 3", "SB 4", "SB 5"],
            "short_title": ["Bill 1", "Bill 2", "Bill 3", "Bill 4", "Bill 5"],
            "motion": ["Final Action"] * 5,
            "vote_type": ["final_action"] * 5,
        }
    )


@pytest.fixture
def legislators() -> pl.DataFrame:
    """Legislator metadata matching senate_matrix."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "D", "E", "F"],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
            "legislator_slug": [
                "sen_a_a_1",
                "sen_b_b_1",
                "sen_c_c_1",
                "sen_d_d_1",
                "sen_e_e_1",
                "sen_f_f_1",
            ],
            "chamber": ["Senate"] * 6,
            "party": ["Republican"] * 3 + ["Democrat"] * 3,
            "district": [1, 2, 3, 4, 5, 6],
            "member_url": [""] * 6,
        }
    )


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
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(pca_scores, senate_matrix, "Senate")
        assert cons_slug == "sen_a_a_1"  # A has PC1=6.0 (highest)

    def test_liberal_anchor_is_lowest_pc1(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Liberal anchor should be the legislator with lowest PC1."""
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(pca_scores, senate_matrix, "Senate")
        assert lib_slug == "sen_f_f_1"  # F has PC1=-6.0 (lowest)

    def test_anchor_indices_are_valid(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Anchor indices should correspond to slug positions in the matrix."""
        cons_idx, cons_slug, lib_idx, lib_slug = select_anchors(pca_scores, senate_matrix, "Senate")
        slugs = senate_matrix["legislator_slug"].to_list()
        assert slugs[cons_idx] == cons_slug
        assert slugs[lib_idx] == lib_slug

    def test_anchors_are_different(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Conservative and liberal anchors must be different legislators."""
        cons_idx, _, lib_idx, _ = select_anchors(pca_scores, senate_matrix, "Senate")
        assert cons_idx != lib_idx

    def test_low_participation_excluded(
        self,
        pca_scores: pl.DataFrame,
    ) -> None:
        """Legislators with < 50% participation should not be anchors.

        Create a matrix where the most extreme PCA legislator (A, PC1=6.0)
        has only 1 of 5 votes (20% participation). B (PC1=5.5) should be
        selected instead.
        """
        matrix = pl.DataFrame(
            {
                "legislator_slug": [
                    "sen_a_a_1",
                    "sen_b_b_1",
                    "sen_c_c_1",
                    "sen_d_d_1",
                    "sen_e_e_1",
                    "sen_f_f_1",
                ],
                "v1": [1, 1, 1, 0, 0, 0],
                "v2": [None, 1, 1, 1, 1, 0],
                "v3": [None, 1, 1, 1, 0, 1],
                "v4": [None, 0, 0, 1, 1, 0],
                "v5": [None, 1, 1, 0, 0, 0],
            }
        )
        cons_idx, cons_slug, _, _ = select_anchors(pca_scores, matrix, "Senate")
        # A has 1/5 = 20% < 50%, so B (PC1=5.5) should be the anchor
        assert cons_slug == "sen_b_b_1"


# ── build_irt_graph tests ─────────────────────────────────────────────────────


class TestBuildIrtGraph:
    """Tests for build_irt_graph model construction.

    Run: uv run pytest tests/test_irt.py::TestBuildIrtGraph -v
    """

    def test_build_irt_graph_returns_model(self, senate_matrix: pl.DataFrame) -> None:
        """build_irt_graph should return a PyMC model with expected free RVs."""
        import pymc as pm

        data = prepare_irt_data(senate_matrix, "Senate")
        anchors = [(0, 1.0), (5, -1.0)]
        model = build_irt_graph(data, anchors)
        assert isinstance(model, pm.Model)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "xi_free" in rv_names
        assert "alpha" in rv_names
        assert "beta" in rv_names

    def test_build_irt_graph_coords(self, senate_matrix: pl.DataFrame) -> None:
        """build_irt_graph model should have legislator, vote, obs_id coords."""
        data = prepare_irt_data(senate_matrix, "Senate")
        anchors = [(0, 1.0), (5, -1.0)]
        model = build_irt_graph(data, anchors)
        assert "legislator" in model.coords
        assert "vote" in model.coords
        assert "obs_id" in model.coords
        assert len(model.coords["legislator"]) == data["n_legislators"]
        assert len(model.coords["vote"]) == data["n_votes"]

    def test_build_irt_graph_xi_free_shape(self, senate_matrix: pl.DataFrame) -> None:
        """xi_free should have shape n_legislators - n_anchors."""
        data = prepare_irt_data(senate_matrix, "Senate")
        anchors = [(0, 1.0), (5, -1.0)]
        model = build_irt_graph(data, anchors)
        xi_free_rv = [rv for rv in model.free_RVs if rv.name == "xi_free"][0]
        assert xi_free_rv.type.shape[0] == data["n_legislators"] - len(anchors)


# ── filter_vote_matrix_for_sensitivity tests ──────────────────────────────────


class TestSensitivityFilter:
    """Test the sensitivity re-filtering of vote matrices."""

    def test_higher_threshold_drops_more_votes(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """A 10% threshold should retain fewer votes than 2.5%."""
        filtered_low = filter_vote_matrix_for_sensitivity(
            senate_matrix,
            rollcalls,
            "Senate",
            minority_threshold=0.025,
            min_votes=1,
        )
        filtered_high = filter_vote_matrix_for_sensitivity(
            senate_matrix,
            rollcalls,
            "Senate",
            minority_threshold=0.20,
            min_votes=1,
        )
        n_low = len(filtered_low.columns) - 1  # exclude slug col
        n_high = len(filtered_high.columns) - 1
        assert n_high <= n_low

    def test_slug_column_preserved(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """Output should always have the legislator_slug column."""
        filtered = filter_vote_matrix_for_sensitivity(
            senate_matrix,
            rollcalls,
            "Senate",
            minority_threshold=0.025,
            min_votes=1,
        )
        assert "legislator_slug" in filtered.columns

    def test_chamber_filter_applied(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """Only Senate legislators (sen_ prefix) should remain."""
        filtered = filter_vote_matrix_for_sensitivity(
            senate_matrix,
            rollcalls,
            "Senate",
            minority_threshold=0.025,
            min_votes=1,
        )
        for slug in filtered["legislator_slug"].to_list():
            assert slug.startswith("sen_"), f"Non-Senate slug: {slug}"

    def test_min_votes_filters_legislators(
        self, senate_matrix: pl.DataFrame, rollcalls: pl.DataFrame
    ) -> None:
        """Setting min_votes high should drop low-participation legislators."""
        filtered = filter_vote_matrix_for_sensitivity(
            senate_matrix,
            rollcalls,
            "Senate",
            minority_threshold=0.025,
            min_votes=100,
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


# ── Joint model fixtures ────────────────────────────────────────────────────


@pytest.fixture
def house_matrix() -> pl.DataFrame:
    """Synthetic House vote matrix: 4 legislators x 3 votes.

    rep_x_x_1 is a bridging legislator (also appears as sen_x_x_1 in Senate).
    """
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a_a_1", "rep_b_b_1", "rep_c_c_1", "rep_x_x_1"],
            "hv1": [1, 1, 0, 1],
            "hv2": [1, 0, 0, 1],
            "hv3": [0, 1, 1, 0],
        }
    )


@pytest.fixture
def joint_senate_matrix() -> pl.DataFrame:
    """Synthetic Senate vote matrix: 3 legislators x 3 votes.

    sen_x_x_1 is a bridging legislator (also rep_x_x_1 in House).
    sv1 and hv1 are the same bill (SB 1).
    """
    return pl.DataFrame(
        {
            "legislator_slug": ["sen_d_d_1", "sen_e_e_1", "sen_x_x_1"],
            "sv1": [1, 0, 1],
            "sv2": [0, 1, 0],
            "sv3": [1, 1, 1],
        }
    )


@pytest.fixture
def joint_rollcalls() -> pl.DataFrame:
    """Roll call metadata: hv1 and sv1 share bill SB 1."""
    return pl.DataFrame(
        {
            "vote_id": ["hv1", "hv2", "hv3", "sv1", "sv2", "sv3"],
            "chamber": ["House", "House", "House", "Senate", "Senate", "Senate"],
            "bill_number": ["SB 1", "HB 2", "HB 3", "SB 1", "SB 4", "SB 5"],
            "short_title": ["Bill 1", "Bill 2", "Bill 3", "Bill 1", "Bill 4", "Bill 5"],
            "motion": [
                "Final Action",
                "Final Action",
                "Final Action",
                "Final Action",
                "Final Action",
                "Final Action",
            ],
            "vote_type": ["final_action"] * 6,
        }
    )


@pytest.fixture
def joint_legislators() -> pl.DataFrame:
    """Legislator metadata with one bridging legislator (X X)."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "X_H", "D", "E", "X_S"],
            "full_name": ["A A", "B B", "C C", "X X", "D D", "E E", "X X"],
            "legislator_slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_x_x_1",
                "sen_d_d_1",
                "sen_e_e_1",
                "sen_x_x_1",
            ],
            "chamber": ["House", "House", "House", "House", "Senate", "Senate", "Senate"],
            "party": [
                "Republican",
                "Republican",
                "Democrat",
                "Republican",
                "Democrat",
                "Democrat",
                "Republican",
            ],
            "district": [1, 2, 3, 4, 5, 6, 7],
            "member_url": [""] * 7,
        }
    )


# ── build_joint_vote_matrix tests ────────────────────────────────────────────


class TestBuildJointVoteMatrix:
    """Test joint vote matrix construction."""

    def test_shared_bill_creates_matched_column(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """SB 1 appears in both chambers, so should create one matched column."""
        joint, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        assert len(info["matched_bills"]) == 1
        assert info["matched_bills"][0]["bill_number"] == "SB 1"

    def test_matched_column_in_matrix(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """The matched column should exist in the joint matrix."""
        joint, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        matched_cols = [c for c in joint.columns if c.startswith("matched_")]
        assert len(matched_cols) == 1

    def test_bridging_legislator_detected(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """X X should be detected as a bridging legislator."""
        _, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        assert len(info["bridging_legislators"]) == 1
        assert info["bridging_legislators"][0]["full_name"] == "X X"

    def test_bridging_legislator_merged(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """Bridging legislator should appear once (House slug), not twice."""
        joint, _ = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        slugs = joint["legislator_slug"].to_list()
        assert "rep_x_x_1" in slugs
        assert "sen_x_x_1" not in slugs

    def test_total_legislators(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """4 House + 3 Senate - 1 bridging = 6 unique legislators."""
        joint, _ = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        assert joint.height == 6

    def test_house_only_and_senate_only_columns(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """House-only and senate-only vote_ids should be tracked in mapping_info."""
        _, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        # hv1 is matched (SB 1), so hv2 and hv3 are house-only
        assert set(info["house_only_vote_ids"]) == {"hv2", "hv3"}
        # sv1 is matched (SB 1), so sv2 and sv3 are senate-only
        assert set(info["senate_only_vote_ids"]) == {"sv2", "sv3"}

    def test_bridging_legislator_has_both_chambers_votes(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """Bridging legislator should have votes from both chambers."""
        joint, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        x_row = joint.filter(pl.col("legislator_slug") == "rep_x_x_1")
        # Should have House votes (hv2, hv3) AND Senate votes (sv2, sv3)
        assert x_row["hv2"][0] is not None  # House vote
        assert x_row["sv2"][0] is not None  # Senate vote


# ── unmerge_bridging_legislators tests ───────────────────────────────────────


class TestUnmergeBridgingLegislators:
    """Test expanding bridging legislators back to per-chamber slugs."""

    def test_bridging_legislator_duplicated(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """Bridging legislator should appear twice after unmerging."""
        joint, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        # Create mock ideal points from joint matrix slugs
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": joint["legislator_slug"].to_list(),
                "xi_mean": [1.0, 0.5, -0.5, 0.8, -1.0, -0.3],
                "xi_sd": [0.1] * 6,
                "xi_hdi_2.5": [0.8, 0.3, -0.7, 0.6, -1.2, -0.5],
                "xi_hdi_97.5": [1.2, 0.7, -0.3, 1.0, -0.8, -0.1],
                "full_name": ["A A", "B B", "C C", "X X", "D D", "E E"],
                "party": [
                    "Republican",
                    "Republican",
                    "Democrat",
                    "Republican",
                    "Democrat",
                    "Democrat",
                ],
                "district": [1, 2, 3, 4, 5, 6],
                "chamber": ["House", "House", "House", "House", "Senate", "Senate"],
            }
        )

        unmerged = unmerge_bridging_legislators(
            ideal_points,
            info,
            joint_legislators,
        )
        assert unmerged.height == 7  # 6 + 1 duplicated bridging

    def test_both_slugs_present_after_unmerge(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """Both rep_x_x_1 and sen_x_x_1 should exist after unmerging."""
        joint, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": joint["legislator_slug"].to_list(),
                "xi_mean": [1.0, 0.5, -0.5, 0.8, -1.0, -0.3],
                "xi_sd": [0.1] * 6,
                "xi_hdi_2.5": [0.8, 0.3, -0.7, 0.6, -1.2, -0.5],
                "xi_hdi_97.5": [1.2, 0.7, -0.3, 1.0, -0.8, -0.1],
                "full_name": ["A A", "B B", "C C", "X X", "D D", "E E"],
                "party": [
                    "Republican",
                    "Republican",
                    "Democrat",
                    "Republican",
                    "Democrat",
                    "Democrat",
                ],
                "district": [1, 2, 3, 4, 5, 6],
                "chamber": ["House", "House", "House", "House", "Senate", "Senate"],
            }
        )

        unmerged = unmerge_bridging_legislators(
            ideal_points,
            info,
            joint_legislators,
        )
        slugs = unmerged["legislator_slug"].to_list()
        assert "rep_x_x_1" in slugs
        assert "sen_x_x_1" in slugs

    def test_ideal_point_same_for_both_slugs(
        self,
        house_matrix: pl.DataFrame,
        joint_senate_matrix: pl.DataFrame,
        joint_rollcalls: pl.DataFrame,
        joint_legislators: pl.DataFrame,
    ) -> None:
        """Both versions of bridging legislator should have the same ideal point."""
        joint, info = build_joint_vote_matrix(
            house_matrix,
            joint_senate_matrix,
            joint_rollcalls,
            joint_legislators,
        )
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": joint["legislator_slug"].to_list(),
                "xi_mean": [1.0, 0.5, -0.5, 0.8, -1.0, -0.3],
                "xi_sd": [0.1] * 6,
                "xi_hdi_2.5": [0.8, 0.3, -0.7, 0.6, -1.2, -0.5],
                "xi_hdi_97.5": [1.2, 0.7, -0.3, 1.0, -0.8, -0.1],
                "full_name": ["A A", "B B", "C C", "X X", "D D", "E E"],
                "party": [
                    "Republican",
                    "Republican",
                    "Democrat",
                    "Republican",
                    "Democrat",
                    "Democrat",
                ],
                "district": [1, 2, 3, 4, 5, 6],
                "chamber": ["House", "House", "House", "House", "Senate", "Senate"],
            }
        )

        unmerged = unmerge_bridging_legislators(
            ideal_points,
            info,
            joint_legislators,
        )
        rep_x = unmerged.filter(pl.col("legislator_slug") == "rep_x_x_1")
        sen_x = unmerged.filter(pl.col("legislator_slug") == "sen_x_x_1")
        assert rep_x["xi_mean"][0] == sen_x["xi_mean"][0]

    def test_no_bridging_returns_unchanged(self) -> None:
        """If no bridging legislators, return the DataFrame unchanged."""
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "sen_b_b_1"],
                "xi_mean": [1.0, -1.0],
                "xi_sd": [0.1, 0.1],
                "xi_hdi_2.5": [0.8, -1.2],
                "xi_hdi_97.5": [1.2, -0.8],
                "full_name": ["A A", "B B"],
                "party": ["Republican", "Democrat"],
                "district": [1, 2],
                "chamber": ["House", "Senate"],
            }
        )
        info = {"bridging_legislators": []}
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "sen_b_b_1"],
                "full_name": ["A A", "B B"],
                "chamber": ["House", "Senate"],
                "party": ["Republican", "Democrat"],
                "district": [1, 2],
            }
        )

        result = unmerge_bridging_legislators(ideal_points, info, legislators)
        assert result.height == 2


# ── _detect_forest_highlights tests ──────────────────────────────────────────


class TestDetectForestHighlights:
    """Test data-driven detection of notable legislators for forest plot annotation."""

    @pytest.fixture
    def highlights_ideal_points(self) -> pl.DataFrame:
        """10 legislators with varying xi_mean and xi_sd."""
        return pl.DataFrame(
            {
                "legislator_slug": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(10)],
                "xi_mean": [3.0, 2.0, 1.5, 1.0, 0.5, 0.1, -0.5, -1.0, -1.5, -2.5],
                "xi_sd": [0.15, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.10, 0.50],
                "xi_hdi_2.5": [2.7, 1.8, 1.3, 0.8, 0.3, -0.1, -0.7, -1.2, -1.7, -3.5],
                "xi_hdi_97.5": [3.3, 2.2, 1.7, 1.2, 0.7, 0.3, -0.3, -0.8, -1.3, -1.5],
                "full_name": [f"{chr(65 + i)} {chr(65 + i)}" for i in range(10)],
                "party": ["Republican"] * 6 + ["Democrat"] * 4,
                "district": list(range(1, 11)),
                "chamber": ["House"] * 10,
            }
        )

    def test_detects_most_extreme(self, highlights_ideal_points: pl.DataFrame) -> None:
        """Legislator with highest |xi_mean| should be highlighted."""
        result = _detect_forest_highlights(highlights_ideal_points, "House")
        # rep_a_a_1 has xi_mean=3.0, the most extreme
        assert "rep_a_a_1" in result
        assert "Most conservative" in result["rep_a_a_1"]

    def test_detects_widest_hdi(self, highlights_ideal_points: pl.DataFrame) -> None:
        """Legislator with xi_sd > 2x median should be highlighted."""
        result = _detect_forest_highlights(highlights_ideal_points, "House")
        # rep_j_j_1 has xi_sd=0.50, median is ~0.10, so 0.50 > 2*0.10
        assert "rep_j_j_1" in result
        assert "uncertainty" in result["rep_j_j_1"].lower()

    def test_max_five_highlights(self) -> None:
        """Should never return more than 5 highlights."""
        # Create 20 legislators where many would trigger highlights
        df = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}_x_1" for i in range(20)],
                "xi_mean": [float(i) for i in range(20)],
                "xi_sd": [0.5 * (i + 1) for i in range(20)],
                "xi_hdi_2.5": [float(i - 1) for i in range(20)],
                "xi_hdi_97.5": [float(i + 1) for i in range(20)],
                "full_name": [f"Leg {i}" for i in range(20)],
                "party": ["Republican"] * 10 + ["Democrat"] * 10,
                "district": list(range(1, 21)),
                "chamber": ["House"] * 20,
            }
        )
        result = _detect_forest_highlights(df, "House")
        assert len(result) <= 5

    def test_empty_for_small_df(self) -> None:
        """With fewer than 3 legislators, should return empty dict."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "rep_b_b_1"],
                "xi_mean": [1.0, -1.0],
                "xi_sd": [0.1, 0.1],
                "xi_hdi_2.5": [0.8, -1.2],
                "xi_hdi_97.5": [1.2, -0.8],
                "full_name": ["A A", "B B"],
                "party": ["Republican", "Democrat"],
                "district": [1, 2],
                "chamber": ["House"] * 2,
            }
        )
        result = _detect_forest_highlights(df, "House")
        assert result == {}


# ── find_paradox_legislator tests ────────────────────────────────────────────


class TestFindParadoxLegislator:
    """Test detection of ideologically extreme but contrarian legislators."""

    @pytest.fixture
    def paradox_ideal_points(self) -> pl.DataFrame:
        """6 legislators: 4 R, 2 D. rep_a is most extreme R."""
        return pl.DataFrame(
            {
                "legislator_slug": [
                    "rep_a_a_1",
                    "rep_b_b_1",
                    "rep_c_c_1",
                    "rep_d_d_1",
                    "rep_e_e_1",
                    "rep_f_f_1",
                ],
                "xi_mean": [3.0, 2.0, 1.5, 1.0, -1.0, -2.0],
                "xi_sd": [0.1] * 6,
                "xi_hdi_2.5": [2.8, 1.8, 1.3, 0.8, -1.2, -2.2],
                "xi_hdi_97.5": [3.2, 2.2, 1.7, 1.2, -0.8, -1.8],
                "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
                "party": ["Republican"] * 4 + ["Democrat"] * 2,
                "district": list(range(1, 7)),
                "chamber": ["House"] * 6,
            }
        )

    @pytest.fixture
    def paradox_bill_params(self) -> pl.DataFrame:
        """8 bills: 4 high-disc, 4 low-disc."""
        return pl.DataFrame(
            {
                "vote_id": [f"v{i}" for i in range(1, 9)],
                "beta_mean": [2.0, 2.5, -2.0, 1.8, 0.3, 0.2, -0.1, 0.4],
                "alpha_mean": [0.0] * 8,
            }
        )

    @pytest.fixture
    def paradox_data_with_gap(self) -> dict:
        """Raw vote data where rep_a votes differently on high-disc vs low-disc bills.

        rep_a votes Yea on all 4 high-disc bills but Nay on 3 of 4 low-disc bills.
        Other Rs vote consistently Yea on all bills.
        """
        slugs = [
            "rep_a_a_1",
            "rep_b_b_1",
            "rep_c_c_1",
            "rep_d_d_1",
            "rep_e_e_1",
            "rep_f_f_1",
        ]
        vote_ids = [f"v{i}" for i in range(1, 9)]

        # Build vote matrix:
        # Rows: 6 legislators, Cols: 8 votes
        # rep_a: high-disc=[1,1,1,1], low-disc=[0,0,0,1]
        # others: all 1s (Yea) on everything
        votes = {
            "rep_a_a_1": [1, 1, 1, 1, 0, 0, 0, 1],
            "rep_b_b_1": [1, 1, 1, 1, 1, 1, 1, 1],
            "rep_c_c_1": [1, 1, 1, 1, 1, 1, 1, 1],
            "rep_d_d_1": [1, 1, 1, 1, 1, 1, 1, 1],
            "rep_e_e_1": [0, 0, 0, 0, 1, 1, 1, 0],
            "rep_f_f_1": [0, 0, 0, 0, 1, 1, 1, 0],
        }

        leg_idx_list = []
        vote_idx_list = []
        y_list = []
        for s_idx, slug in enumerate(slugs):
            for v_idx, vote in enumerate(votes[slug]):
                leg_idx_list.append(s_idx)
                vote_idx_list.append(v_idx)
                y_list.append(vote)

        return {
            "leg_slugs": slugs,
            "vote_ids": vote_ids,
            "leg_idx": np.array(leg_idx_list, dtype=np.int64),
            "vote_idx": np.array(vote_idx_list, dtype=np.int64),
            "y": np.array(y_list, dtype=np.int64),
            "n_legislators": len(slugs),
            "n_votes": len(vote_ids),
            "n_obs": len(y_list),
        }

    def test_finds_paradox_when_gap_exists(
        self,
        paradox_ideal_points: pl.DataFrame,
        paradox_bill_params: pl.DataFrame,
        paradox_data_with_gap: dict,
    ) -> None:
        """Should return dict with correct slug when high/low-disc Yea rate gap > threshold."""
        result = find_paradox_legislator(
            paradox_ideal_points,
            paradox_bill_params,
            paradox_data_with_gap,
        )
        assert result is not None
        assert result["legislator_slug"] == "rep_a_a_1"
        assert result["gap"] > PARADOX_YEA_GAP

    def test_no_paradox_when_consistent(
        self,
        paradox_ideal_points: pl.DataFrame,
        paradox_bill_params: pl.DataFrame,
    ) -> None:
        """Should return None when the most extreme legislator votes consistently."""
        # All Yea on everything — no gap
        slugs = paradox_ideal_points["legislator_slug"].to_list()
        vote_ids = paradox_bill_params["vote_id"].to_list()
        leg_idx_list = []
        vote_idx_list = []
        y_list = []
        for s_idx in range(len(slugs)):
            for v_idx in range(len(vote_ids)):
                leg_idx_list.append(s_idx)
                vote_idx_list.append(v_idx)
                y_list.append(1)

        data = {
            "leg_slugs": slugs,
            "vote_ids": vote_ids,
            "leg_idx": np.array(leg_idx_list, dtype=np.int64),
            "vote_idx": np.array(vote_idx_list, dtype=np.int64),
            "y": np.array(y_list, dtype=np.int64),
            "n_legislators": len(slugs),
            "n_votes": len(vote_ids),
            "n_obs": len(y_list),
        }
        result = find_paradox_legislator(paradox_ideal_points, paradox_bill_params, data)
        assert result is None

    def test_returns_none_for_small_n(
        self,
        paradox_ideal_points: pl.DataFrame,
    ) -> None:
        """Should return None when fewer than PARADOX_MIN_BILLS high-disc or low-disc bills."""
        # Only 2 bills total — not enough
        bill_params = pl.DataFrame(
            {
                "vote_id": ["v1", "v2"],
                "beta_mean": [2.0, 0.3],
                "alpha_mean": [0.0, 0.0],
            }
        )
        slugs = paradox_ideal_points["legislator_slug"].to_list()
        data = {
            "leg_slugs": slugs,
            "vote_ids": ["v1", "v2"],
            "leg_idx": np.array([0, 0, 1, 1], dtype=np.int64),
            "vote_idx": np.array([0, 1, 0, 1], dtype=np.int64),
            "y": np.array([1, 0, 1, 1], dtype=np.int64),
            "n_legislators": len(slugs),
            "n_votes": 2,
            "n_obs": 4,
        }
        result = find_paradox_legislator(paradox_ideal_points, bill_params, data)
        assert result is None

    def test_paradox_info_has_required_keys(
        self,
        paradox_ideal_points: pl.DataFrame,
        paradox_bill_params: pl.DataFrame,
        paradox_data_with_gap: dict,
    ) -> None:
        """Returned dict should have all expected keys."""
        result = find_paradox_legislator(
            paradox_ideal_points,
            paradox_bill_params,
            paradox_data_with_gap,
        )
        assert result is not None
        expected_keys = {
            "legislator_slug",
            "full_name",
            "party",
            "xi_mean",
            "high_disc_yea_rate",
            "low_disc_yea_rate",
            "party_high_disc_yea_rate",
            "party_low_disc_yea_rate",
            "n_high_disc",
            "n_low_disc",
            "gap",
        }
        assert set(result.keys()) == expected_keys


# ── Mock InferenceData helper ────────────────────────────────────────────────


def _make_irt_idata(
    n_legislators: int = 6,
    n_votes: int = 5,
    n_chains: int = 2,
    n_draws: int = 100,
    xi_values: np.ndarray | None = None,
    alpha_values: np.ndarray | None = None,
    beta_values: np.ndarray | None = None,
    leg_slugs: list[str] | None = None,
    vote_ids: list[str] | None = None,
    n_divergences: int = 0,
) -> az.InferenceData:
    """Create synthetic InferenceData for testing convergence and extraction."""
    if leg_slugs is None:
        leg_slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(n_legislators)]
    if vote_ids is None:
        vote_ids = [f"v{i + 1}" for i in range(n_votes)]

    rng = np.random.default_rng(42)

    if xi_values is None:
        base = np.linspace(2.0, -2.0, n_legislators)
        xi_values = np.tile(base, (n_chains, n_draws, 1)) + rng.normal(
            0, 0.05, (n_chains, n_draws, n_legislators)
        )
    if alpha_values is None:
        alpha_values = rng.normal(0, 1, (n_chains, n_draws, n_votes))
    if beta_values is None:
        # Mix of positive and negative discrimination
        base_betas = np.array([1.5, -1.2, 0.8, -0.5, 2.0])[:n_votes]
        beta_values = np.tile(base_betas, (n_chains, n_draws, 1)) + rng.normal(
            0, 0.1, (n_chains, n_draws, n_votes)
        )

    posterior = xr.Dataset(
        {
            "xi": xr.DataArray(
                xi_values,
                dims=["chain", "draw", "legislator"],
                coords={"legislator": leg_slugs},
            ),
            "alpha": xr.DataArray(
                alpha_values,
                dims=["chain", "draw", "vote"],
                coords={"vote": vote_ids},
            ),
            "beta": xr.DataArray(
                beta_values,
                dims=["chain", "draw", "vote"],
                coords={"vote": vote_ids},
            ),
        }
    )

    # Sample stats with divergences
    diverging = np.zeros((n_chains, n_draws), dtype=bool)
    if n_divergences > 0:
        flat_indices = rng.choice(n_chains * n_draws, size=n_divergences, replace=False)
        for idx in flat_indices:
            c, d = divmod(idx, n_draws)
            diverging[c, d] = True

    # Energy values for E-BFMI
    energy = rng.normal(0, 1, (n_chains, n_draws))

    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray(diverging, dims=["chain", "draw"]),
            "energy": xr.DataArray(energy, dims=["chain", "draw"]),
        }
    )

    return az.InferenceData(posterior=posterior, sample_stats=sample_stats)


# ── check_convergence tests ──────────────────────────────────────────────────


class TestCheckConvergence:
    """Test MCMC convergence diagnostics with synthetic InferenceData."""

    def test_well_converged_passes(self) -> None:
        """Well-behaved chains should pass all convergence checks."""
        idata = _make_irt_idata(n_draws=500)
        diag = check_convergence(idata, "Senate")
        assert diag["all_ok"] is True

    def test_rhat_reported(self) -> None:
        """Diagnostics should include R-hat for all parameter types."""
        idata = _make_irt_idata()
        diag = check_convergence(idata, "Senate")
        assert "xi_rhat_max" in diag
        assert "alpha_rhat_max" in diag
        assert "beta_rhat_max" in diag

    def test_bulk_ess_reported(self) -> None:
        """Diagnostics should include bulk ESS for all parameter types."""
        idata = _make_irt_idata()
        diag = check_convergence(idata, "Senate")
        assert "xi_ess_min" in diag
        assert "alpha_ess_min" in diag
        assert "beta_ess_min" in diag

    def test_tail_ess_reported(self) -> None:
        """Diagnostics should include tail ESS for all parameter types."""
        idata = _make_irt_idata()
        diag = check_convergence(idata, "Senate")
        assert "xi_tail_ess_min" in diag
        assert "alpha_tail_ess_min" in diag
        assert "beta_tail_ess_min" in diag

    def test_divergences_counted(self) -> None:
        """Divergences should be counted correctly."""
        idata = _make_irt_idata(n_divergences=5)
        diag = check_convergence(idata, "Senate")
        assert diag["divergences"] == 5

    def test_high_divergences_fails(self) -> None:
        """More than MAX_DIVERGENCES should fail convergence."""
        idata = _make_irt_idata(n_divergences=MAX_DIVERGENCES + 5)
        diag = check_convergence(idata, "Senate")
        assert diag["all_ok"] is False

    def test_ebfmi_reported(self) -> None:
        """E-BFMI values should be reported per chain."""
        idata = _make_irt_idata(n_chains=2)
        diag = check_convergence(idata, "Senate")
        assert "ebfmi" in diag
        assert len(diag["ebfmi"]) == 2

    def test_mode_split_detected(self) -> None:
        """Chains in different modes should produce high R-hat."""
        n_leg = 6
        n_chains = 2
        n_draws = 100
        rng = np.random.default_rng(42)

        # Chain 0: positive values. Chain 1: negative (mirror image)
        xi_vals = np.zeros((n_chains, n_draws, n_leg))
        base = np.linspace(2, -2, n_leg)
        xi_vals[0] = base + rng.normal(0, 0.05, (n_draws, n_leg))
        xi_vals[1] = -base + rng.normal(0, 0.05, (n_draws, n_leg))

        idata = _make_irt_idata(xi_values=xi_vals)
        diag = check_convergence(idata, "Senate")
        # R-hat should be very high due to mode-splitting
        assert diag["xi_rhat_max"] > RHAT_THRESHOLD
        assert diag["all_ok"] is False

    def test_rhat_threshold_applied(self) -> None:
        """R-hat below threshold should be OK."""
        idata = _make_irt_idata(n_draws=500)
        diag = check_convergence(idata, "Senate")
        assert diag["xi_rhat_max"] < RHAT_THRESHOLD


# ── extract_ideal_points tests ───────────────────────────────────────────────


class TestExtractIdealPoints:
    """Test posterior extraction for legislator ideal points."""

    @pytest.fixture
    def extraction_legislators(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "name": ["A", "B", "C", "D", "E", "F"],
                "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
                "legislator_slug": [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)],
                "chamber": ["Senate"] * 6,
                "party": ["Republican"] * 3 + ["Democrat"] * 3,
                "district": list(range(1, 7)),
                "member_url": [""] * 6,
            }
        )

    def test_output_schema(self, extraction_legislators: pl.DataFrame) -> None:
        """Output should have all required columns."""
        idata = _make_irt_idata()
        slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)]
        data = {"leg_slugs": slugs}
        df = extract_ideal_points(idata, data, extraction_legislators)
        required = {"legislator_slug", "xi_mean", "xi_sd", "xi_hdi_2.5", "xi_hdi_97.5"}
        assert required.issubset(set(df.columns))

    def test_correct_legislator_count(self, extraction_legislators: pl.DataFrame) -> None:
        """Should return one row per legislator."""
        idata = _make_irt_idata()
        slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)]
        data = {"leg_slugs": slugs}
        df = extract_ideal_points(idata, data, extraction_legislators)
        assert df.height == 6

    def test_sorted_descending(self, extraction_legislators: pl.DataFrame) -> None:
        """Output should be sorted by xi_mean descending."""
        idata = _make_irt_idata()
        slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)]
        data = {"leg_slugs": slugs}
        df = extract_ideal_points(idata, data, extraction_legislators)
        means = df["xi_mean"].to_list()
        assert means == sorted(means, reverse=True)

    def test_hdi_contains_mean(self, extraction_legislators: pl.DataFrame) -> None:
        """HDI bounds should contain the posterior mean."""
        idata = _make_irt_idata()
        slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)]
        data = {"leg_slugs": slugs}
        df = extract_ideal_points(idata, data, extraction_legislators)
        for row in df.iter_rows(named=True):
            assert row["xi_hdi_2.5"] <= row["xi_mean"] <= row["xi_hdi_97.5"]

    def test_positive_sd(self, extraction_legislators: pl.DataFrame) -> None:
        """Standard deviations should be positive."""
        idata = _make_irt_idata()
        slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)]
        data = {"leg_slugs": slugs}
        df = extract_ideal_points(idata, data, extraction_legislators)
        assert (df["xi_sd"] > 0).all()

    def test_metadata_joined(self, extraction_legislators: pl.DataFrame) -> None:
        """Legislator metadata (party, name) should be joined."""
        idata = _make_irt_idata()
        slugs = [f"sen_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(6)]
        data = {"leg_slugs": slugs}
        df = extract_ideal_points(idata, data, extraction_legislators)
        assert "full_name" in df.columns
        assert "party" in df.columns
        parties = set(df["party"].to_list())
        assert "Republican" in parties
        assert "Democrat" in parties


# ── extract_bill_parameters tests ────────────────────────────────────────────


class TestExtractBillParameters:
    """Test posterior extraction for bill parameters."""

    def test_output_schema(self, rollcalls: pl.DataFrame) -> None:
        """Output should have all required columns."""
        idata = _make_irt_idata()
        data = {"vote_ids": [f"v{i + 1}" for i in range(5)]}
        df = extract_bill_parameters(idata, data, rollcalls)
        required = {"vote_id", "alpha_mean", "alpha_sd", "beta_mean", "beta_sd"}
        assert required.issubset(set(df.columns))

    def test_correct_vote_count(self, rollcalls: pl.DataFrame) -> None:
        """Should return one row per vote."""
        idata = _make_irt_idata()
        data = {"vote_ids": [f"v{i + 1}" for i in range(5)]}
        df = extract_bill_parameters(idata, data, rollcalls)
        assert df.height == 5

    def test_positive_sd(self, rollcalls: pl.DataFrame) -> None:
        """Standard deviations should be positive."""
        idata = _make_irt_idata()
        data = {"vote_ids": [f"v{i + 1}" for i in range(5)]}
        df = extract_bill_parameters(idata, data, rollcalls)
        assert (df["alpha_sd"] > 0).all()
        assert (df["beta_sd"] > 0).all()

    def test_negative_beta_allowed(self, rollcalls: pl.DataFrame) -> None:
        """With unconstrained beta, some bills should have negative discrimination."""
        idata = _make_irt_idata()
        data = {"vote_ids": [f"v{i + 1}" for i in range(5)]}
        df = extract_bill_parameters(idata, data, rollcalls)
        # Our fixture has mix of positive and negative betas
        has_negative = (df["beta_mean"] < 0).any()
        has_positive = (df["beta_mean"] > 0).any()
        assert has_negative, "Should have some negative discrimination (D-Yea bills)"
        assert has_positive, "Should have some positive discrimination (R-Yea bills)"

    def test_sorted_by_beta_descending(self, rollcalls: pl.DataFrame) -> None:
        """Output should be sorted by beta_mean descending."""
        idata = _make_irt_idata()
        data = {"vote_ids": [f"v{i + 1}" for i in range(5)]}
        df = extract_bill_parameters(idata, data, rollcalls)
        betas = df["beta_mean"].to_list()
        assert betas == sorted(betas, reverse=True)

    def test_rollcall_metadata_joined(self, rollcalls: pl.DataFrame) -> None:
        """Roll call metadata should be joined when available."""
        idata = _make_irt_idata()
        data = {"vote_ids": [f"v{i + 1}" for i in range(5)]}
        df = extract_bill_parameters(idata, data, rollcalls)
        assert "bill_number" in df.columns


# ── equate_chambers tests ────────────────────────────────────────────────────


class TestEquateChambers:
    """Test cross-chamber test equating."""

    @pytest.fixture
    def equating_data(self, tmp_path: Path) -> dict:
        """Create per-chamber results, mapping info, and legislators for equating."""
        house_ip = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "rep_b_b_1", "rep_c_c_1", "rep_x_x_1"],
                "xi_mean": [2.0, 1.0, -1.0, 0.5],
                "xi_sd": [0.1, 0.1, 0.1, 0.1],
                "xi_hdi_2.5": [1.8, 0.8, -1.2, 0.3],
                "xi_hdi_97.5": [2.2, 1.2, -0.8, 0.7],
                "full_name": ["A A", "B B", "C C", "X X"],
                "party": ["Republican", "Republican", "Democrat", "Republican"],
                "district": [1, 2, 3, 4],
                "chamber": ["House"] * 4,
            }
        )
        senate_ip = pl.DataFrame(
            {
                "legislator_slug": ["sen_d_d_1", "sen_e_e_1", "sen_x_x_1"],
                "xi_mean": [1.5, -1.5, 0.4],
                "xi_sd": [0.1, 0.1, 0.1],
                "xi_hdi_2.5": [1.3, -1.7, 0.2],
                "xi_hdi_97.5": [1.7, -1.3, 0.6],
                "full_name": ["D D", "E E", "X X"],
                "party": ["Republican", "Democrat", "Republican"],
                "district": [5, 6, 7],
                "chamber": ["Senate"] * 3,
            }
        )
        house_bp = pl.DataFrame(
            {
                "vote_id": ["hv1", "hv2", "hv3"],
                "beta_mean": [1.5, -1.0, 0.8],
                "alpha_mean": [0.0, 0.0, 0.0],
            }
        )
        senate_bp = pl.DataFrame(
            {
                "vote_id": ["sv1", "sv2", "sv3"],
                "beta_mean": [2.0, -1.3, 1.1],
                "alpha_mean": [0.0, 0.0, 0.0],
            }
        )
        per_chamber_results = {
            "House": {"ideal_points": house_ip, "bill_params": house_bp},
            "Senate": {"ideal_points": senate_ip, "bill_params": senate_bp},
        }
        mapping_info = {
            "matched_bills": [
                {"bill_number": "SB 1", "house_vote_id": "hv1", "senate_vote_id": "sv1"},
                {"bill_number": "SB 2", "house_vote_id": "hv2", "senate_vote_id": "sv2"},
                {"bill_number": "SB 3", "house_vote_id": "hv3", "senate_vote_id": "sv3"},
            ],
            "bridging_legislators": [
                {
                    "full_name": "X X",
                    "house_slug": "rep_x_x_1",
                    "senate_slug": "sen_x_x_1",
                }
            ],
        }
        legislators = pl.DataFrame(
            {
                "legislator_slug": [
                    "rep_a_a_1",
                    "rep_b_b_1",
                    "rep_c_c_1",
                    "rep_x_x_1",
                    "sen_d_d_1",
                    "sen_e_e_1",
                    "sen_x_x_1",
                ],
                "full_name": ["A A", "B B", "C C", "X X", "D D", "E E", "X X"],
                "chamber": ["House"] * 4 + ["Senate"] * 3,
                "party": [
                    "Republican",
                    "Republican",
                    "Democrat",
                    "Republican",
                    "Republican",
                    "Democrat",
                    "Republican",
                ],
                "district": list(range(1, 8)),
            }
        )
        return {
            "per_chamber_results": per_chamber_results,
            "mapping_info": mapping_info,
            "legislators": legislators,
            "out_dir": tmp_path,
        }

    def test_returns_equated_ideal_points(self, equating_data: dict) -> None:
        """Should return equated ideal points DataFrame."""
        result = equate_chambers(**equating_data)
        assert "equated_ideal_points" in result
        df = result["equated_ideal_points"]
        assert df.height == 7  # 4 House + 3 Senate

    def test_transformation_has_a_and_b(self, equating_data: dict) -> None:
        """Transformation dict should have A (scale) and B (location)."""
        result = equate_chambers(**equating_data)
        t = result["transformation"]
        assert "A" in t
        assert "B" in t
        assert isinstance(t["A"], float)
        assert isinstance(t["B"], float)

    def test_a_is_positive_for_concordant_betas(self, equating_data: dict) -> None:
        """A should be positive when shared bills have concordant discrimination signs."""
        result = equate_chambers(**equating_data)
        assert result["transformation"]["A"] > 0

    def test_bridging_method_used(self, equating_data: dict) -> None:
        """When bridging legislators exist, method should be bridging_legislators."""
        result = equate_chambers(**equating_data)
        assert result["transformation"]["b_method"] == "bridging_legislators"

    def test_no_bridging_fallback(self, equating_data: dict) -> None:
        """Without bridging legislators, B should fall back to 0.0."""
        equating_data["mapping_info"]["bridging_legislators"] = []
        result = equate_chambers(**equating_data)
        assert result["transformation"]["B"] == 0.0
        assert result["transformation"]["b_method"] == "fallback_zero"

    def test_house_ideal_points_unchanged(self, equating_data: dict) -> None:
        """House ideal points should remain unchanged (reference scale)."""
        result = equate_chambers(**equating_data)
        equated = result["equated_ideal_points"]
        house_orig = equating_data["per_chamber_results"]["House"]["ideal_points"]
        for row in house_orig.iter_rows(named=True):
            equated_row = equated.filter(pl.col("legislator_slug") == row["legislator_slug"])
            assert equated_row["xi_mean"][0] == pytest.approx(row["xi_mean"], abs=1e-6)

    def test_correlations_computed(self, equating_data: dict) -> None:
        """Should compute Pearson correlations for each chamber."""
        result = equate_chambers(**equating_data)
        assert "correlations" in result
        assert "House" in result["correlations"]
