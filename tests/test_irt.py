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
    CONTESTED_VOTE_THRESHOLD,
    HOLDOUT_FRACTION,
    HOLDOUT_SEED,
    HORSESHOE_DEM_WRONG_SIDE_FRAC,
    MAX_DIVERGENCES,
    MIN_CONTESTED_FOR_AGREEMENT,
    MIN_CONTESTED_FOR_REFIT,
    MIN_CONTESTED_VOTES_PER_LEG,
    MIN_PC2_VOTES_FOR_REFIT,
    PARADOX_YEA_GAP,
    PC2_PRIOR_SIGMA,
    PROMOTE_2D_RANK_SHIFT,
    RHAT_THRESHOLD,
    SUPERMAJORITY_THRESHOLD,
    IdentificationStrategy,
    RobustnessFlag,
    RobustnessFlags,
    _detect_forest_highlights,
    build_irt_graph,
    build_joint_vote_matrix,
    check_convergence,
    compute_cross_party_agreement,
    cross_reference_2d,
    detect_horseshoe,
    detect_supermajority,
    equate_chambers,
    extract_bill_parameters,
    extract_ideal_points,
    filter_contested_votes,
    filter_pc2_dominant_votes,
    filter_vote_matrix_for_sensitivity,
    find_paradox_legislator,
    prepare_irt_data,
    select_anchors,
    select_identification_strategy,
    unmerge_bridging_legislators,
    validate_sign,
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
    """Test anchor selection (agreement-based primary, PCA fallback).

    Run: uv run pytest tests/test_irt.py::TestSelectAnchors -v
    """

    def test_pca_fallback_conservative_anchor(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """PCA fallback: conservative anchor should be highest PC1 Republican."""
        # No legislators passed → PCA fallback
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        assert cons_slug == "sen_a_a_1"  # A has PC1=6.0 (highest R)

    def test_pca_fallback_liberal_anchor(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """PCA fallback: liberal anchor should be lowest PC1 Democrat."""
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        assert lib_slug == "sen_f_f_1"  # F has PC1=-6.0 (lowest D)

    def test_anchor_indices_are_valid(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Anchor indices should correspond to slug positions in the matrix."""
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, senate_matrix, "Senate"
        )
        slugs = senate_matrix["legislator_slug"].to_list()
        assert slugs[cons_idx] == cons_slug
        assert slugs[lib_idx] == lib_slug

    def test_anchors_are_different(
        self, pca_scores: pl.DataFrame, senate_matrix: pl.DataFrame
    ) -> None:
        """Conservative and liberal anchors must be different legislators."""
        cons_idx, _, lib_idx, _, _ = select_anchors(pca_scores, senate_matrix, "Senate")
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
        cons_idx, cons_slug, _, _, _ = select_anchors(pca_scores, matrix, "Senate")
        # A has 1/5 = 20% < 50%, so B (PC1=5.5) should be the anchor
        assert cons_slug == "sen_b_b_1"

    def test_falls_back_to_pca_with_few_contested_votes(
        self,
        pca_scores: pl.DataFrame,
        senate_matrix: pl.DataFrame,
        legislators: pl.DataFrame,
    ) -> None:
        """With too few contested votes, should fall back to PCA anchors."""
        # The 5-vote fixture won't have ≥10 contested votes → PCA fallback
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, senate_matrix, "Senate", legislators=legislators
        )
        assert cons_slug == "sen_a_a_1"  # PCA fallback: highest PC1 R
        assert lib_slug == "sen_f_f_1"  # PCA fallback: lowest PC1 D


class TestComputeCrossPartyAgreement:
    """Test cross-party contested vote agreement computation.

    Run: uv run pytest tests/test_irt.py::TestComputeCrossPartyAgreement -v
    """

    @pytest.fixture
    def agreement_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Synthetic data with clear partisan structure and contested votes.

        10 Rs + 5 Ds, 40 votes. Rs have ideal points from +0.5 to +2.5,
        Ds from -2.5 to -0.5. Moderate Rs should have higher D-agreement.
        """
        n_r, n_d = 10, 5
        n_legs = n_r + n_d
        n_votes = 40
        rng = np.random.default_rng(42)

        slugs = [f"sen_r{i}_1" for i in range(n_r)] + [f"sen_d{i}_1" for i in range(n_d)]
        parties = ["Republican"] * n_r + ["Democrat"] * n_d

        xi_true = np.concatenate(
            [
                np.linspace(0.5, 2.5, n_r),
                np.linspace(-2.5, -0.5, n_d),
            ]
        )

        vote_matrix = np.full((n_legs, n_votes), np.nan)
        alphas = rng.normal(0, 0.5, n_votes)
        for j in range(n_votes):
            p = 1 / (1 + np.exp(-(xi_true - alphas[j])))
            vote_matrix[:, j] = (rng.random(n_legs) < p).astype(float)

        matrix_data: dict = {"legislator_slug": slugs}
        for j in range(n_votes):
            matrix_data[f"v{j}"] = vote_matrix[:, j].tolist()
        matrix = pl.DataFrame(matrix_data)

        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": [f"R{i}" for i in range(n_r)] + [f"D{i}" for i in range(n_d)],
                "party": parties,
                "district": list(range(1, n_legs + 1)),
                "chamber": ["Senate"] * n_legs,
            }
        )

        return matrix, legislators

    def test_returns_agreement_rates(self, agreement_data: tuple) -> None:
        """Should return non-empty agreement rates for both parties."""
        matrix, legislators = agreement_data
        rates, n_contested = compute_cross_party_agreement(matrix, legislators)
        assert len(rates) > 0
        assert n_contested >= MIN_CONTESTED_FOR_AGREEMENT

    def test_agreement_rates_in_range(self, agreement_data: tuple) -> None:
        """Agreement rates should be between 0 and 1."""
        matrix, legislators = agreement_data
        rates, _ = compute_cross_party_agreement(matrix, legislators)
        for rate in rates.values():
            assert 0.0 <= rate <= 1.0

    def test_moderates_have_higher_agreement(self, agreement_data: tuple) -> None:
        """Moderate Rs (lower index) should have higher D-agreement than extreme Rs."""
        matrix, legislators = agreement_data
        rates, _ = compute_cross_party_agreement(matrix, legislators)
        # r0 is most moderate R (xi=0.5), r9 is most extreme (xi=2.5)
        r0_rate = rates.get("sen_r0_1")
        r9_rate = rates.get("sen_r9_1")
        if r0_rate is not None and r9_rate is not None:
            assert r0_rate > r9_rate, (
                f"Moderate R0 ({r0_rate:.3f}) should agree with Ds more than extreme R9 "
                f"({r9_rate:.3f})"
            )

    def test_empty_with_too_few_parties(self) -> None:
        """Should return empty when one party has < 3 legislators."""
        slugs = ["sen_r0_1", "sen_r1_1", "sen_r2_1", "sen_d0_1", "sen_d1_1"]
        matrix = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "v1": [1, 1, 1, 0, 0],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": slugs,
                "party": ["Republican"] * 3 + ["Democrat"] * 2,
                "district": list(range(1, 6)),
                "chamber": ["Senate"] * 5,
            }
        )
        rates, _ = compute_cross_party_agreement(matrix, legislators)
        assert rates == {}

    def test_constants(self) -> None:
        """Agreement constants should have expected values."""
        assert MIN_CONTESTED_FOR_AGREEMENT == 10
        assert MIN_CONTESTED_VOTES_PER_LEG == 5


class TestSelectAnchorsAgreement:
    """Test agreement-based anchor selection (primary method).

    Run: uv run pytest tests/test_irt.py::TestSelectAnchorsAgreement -v
    """

    @pytest.fixture
    def supermajority_data(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Simulate a supermajority chamber where PCA horseshoe creates bad anchors.

        15 Rs + 5 Ds, 40 votes. Rs span from moderate (xi=0.2) to extreme
        conservative (xi=3.0). Ds from -3.0 to -0.2. PCA scores deliberately
        inverted for the most extreme R (horseshoe effect): the most extreme R
        has the highest PCA PC1, but the most MODERATE R also has high PC1
        (establishment-aligned). The agreement-based method should pick the
        genuine extreme, not the PCA artifact.
        """
        n_r, n_d = 15, 5
        n_legs = n_r + n_d
        n_votes = 40
        rng = np.random.default_rng(42)

        r_slugs = [f"sen_r{i}_1" for i in range(n_r)]
        d_slugs = [f"sen_d{i}_1" for i in range(n_d)]
        slugs = r_slugs + d_slugs
        parties = ["Republican"] * n_r + ["Democrat"] * n_d

        # True ideology: r0 most moderate, r14 most conservative
        xi_true = np.concatenate(
            [
                np.linspace(0.2, 3.0, n_r),
                np.linspace(-3.0, -0.2, n_d),
            ]
        )

        # Generate votes based on true ideology
        vote_matrix = np.full((n_legs, n_votes), np.nan)
        alphas = rng.normal(0, 0.5, n_votes)
        for j in range(n_votes):
            p = 1 / (1 + np.exp(-(xi_true - alphas[j])))
            vote_matrix[:, j] = (rng.random(n_legs) < p).astype(float)

        matrix_data: dict = {"legislator_slug": slugs}
        for j in range(n_votes):
            matrix_data[f"v{j}"] = vote_matrix[:, j].tolist()
        matrix = pl.DataFrame(matrix_data)

        # PCA scores: horseshoe effect — most moderate R (r0) gets highest PC1
        # (establishment-aligned), most extreme R (r14) gets mid-range PC1
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": [f"R{i}" for i in range(n_r)] + [f"D{i}" for i in range(n_d)],
                "party": parties,
                "PC1": (
                    list(np.linspace(8.0, 2.0, n_r))  # r0=8.0 (highest), r14=2.0
                    + list(np.linspace(-6.0, -2.0, n_d))
                ),
                "PC2": [0.0] * n_legs,
            }
        )

        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": [f"R{i}" for i in range(n_r)] + [f"D{i}" for i in range(n_d)],
                "party": parties,
                "district": list(range(1, n_legs + 1)),
                "chamber": ["Senate"] * n_legs,
            }
        )

        return matrix, pca_scores, legislators

    def test_agreement_picks_genuine_extreme_r(self, supermajority_data: tuple) -> None:
        """Agreement-based selection should pick the most partisan R, not PCA extreme."""
        matrix, pca_scores, legislators = supermajority_data
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, matrix, "Senate", legislators=legislators
        )
        # PCA would pick r0 (PC1=8.0, most moderate). Agreement should pick
        # an extreme R (high index) since they have lowest D-agreement.
        cons_r_idx = int(cons_slug.split("_")[1][1:])  # extract number from "sen_rN_1"
        assert cons_r_idx >= 7, (
            f"Expected agreement to pick an extreme R (index ≥7), got r{cons_r_idx} ({cons_slug})"
        )

    def test_agreement_picks_genuine_extreme_d(self, supermajority_data: tuple) -> None:
        """Agreement-based selection should pick the most partisan D."""
        matrix, pca_scores, legislators = supermajority_data
        _, _, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, matrix, "Senate", legislators=legislators
        )
        # Most extreme D should be d0 (xi=-3.0, lowest R-agreement)
        lib_d_idx = int(lib_slug.split("_")[1][1:])
        assert lib_d_idx <= 2, (
            f"Expected agreement to pick an extreme D (index ≤2), got d{lib_d_idx} ({lib_slug})"
        )

    def test_pca_fallback_when_no_legislators(self, supermajority_data: tuple) -> None:
        """Without legislators kwarg, should fall back to PCA."""
        matrix, pca_scores, _ = supermajority_data
        cons_idx, cons_slug, _, _, _ = select_anchors(pca_scores, matrix, "Senate")
        # PCA picks r0 (highest PC1=8.0, the moderate — horseshoe artifact)
        assert cons_slug == "sen_r0_1"

    def test_agreement_anchor_indices_valid(self, supermajority_data: tuple) -> None:
        """Agreement-based anchor indices should match slug positions."""
        matrix, pca_scores, legislators = supermajority_data
        cons_idx, cons_slug, lib_idx, lib_slug, _ = select_anchors(
            pca_scores, matrix, "Senate", legislators=legislators
        )
        slugs = matrix["legislator_slug"].to_list()
        assert slugs[cons_idx] == cons_slug
        assert slugs[lib_idx] == lib_slug
        assert cons_idx != lib_idx


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


# ── validate_sign tests ──────────────────────────────────────────────────────


class TestValidateSign:
    """Test post-hoc sign validation using cross-party contested vote agreement."""

    @pytest.fixture
    def correctly_signed_idata(self) -> tuple[az.InferenceData, pl.DataFrame, pl.DataFrame, dict]:
        """Synthetic IRT output where sign is correct.

        10 Rs and 5 Ds. Ideal points: Rs positive, Ds negative.
        Vote matrix: contested votes where moderate Rs agree with Ds.
        """
        n_r, n_d = 10, 5
        n_legs = n_r + n_d
        n_votes = 40
        rng = np.random.default_rng(42)

        slugs = [f"sen_r{i}_1" for i in range(n_r)] + [f"sen_d{i}_1" for i in range(n_d)]
        parties = ["Republican"] * n_r + ["Democrat"] * n_d

        # Ideal points: Rs from +0.5 to +2.5, Ds from -2.5 to -0.5
        xi_true = np.concatenate(
            [
                np.linspace(0.5, 2.5, n_r),  # R: moderate to conservative
                np.linspace(-2.5, -0.5, n_d),  # D: liberal to moderate
            ]
        )

        # Generate votes based on ideal points (correct sign)
        vote_matrix = np.full((n_legs, n_votes), np.nan)
        alphas = rng.normal(0, 0.5, n_votes)  # centered for contested splits
        for j in range(n_votes):
            p = 1 / (1 + np.exp(-(xi_true - alphas[j])))
            vote_matrix[:, j] = (rng.random(n_legs) < p).astype(float)

        # Build polars matrix
        matrix_data: dict = {"legislator_slug": slugs}
        vote_ids = [f"v{j}" for j in range(n_votes)]
        for j, vid in enumerate(vote_ids):
            matrix_data[vid] = vote_matrix[:, j].tolist()
        matrix = pl.DataFrame(matrix_data)

        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": [f"R{i}" for i in range(n_r)] + [f"D{i}" for i in range(n_d)],
                "party": parties,
                "district": list(range(1, n_legs + 1)),
                "chamber": ["Senate"] * n_legs,
            }
        )

        # Build fake idata with correct sign
        xi_posterior = xi_true[np.newaxis, np.newaxis, :] + rng.normal(0, 0.1, (2, 100, n_legs))
        beta_posterior = rng.normal(1, 0.3, (2, 100, n_votes))
        xi_free_posterior = xi_posterior[:, :, :n_legs]  # simplified

        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "xi": xr.DataArray(xi_posterior, dims=["chain", "draw", "legislator"]),
                    "xi_free": xr.DataArray(
                        xi_free_posterior, dims=["chain", "draw", "xi_free_dim_0"]
                    ),
                    "beta": xr.DataArray(beta_posterior, dims=["chain", "draw", "vote"]),
                }
            )
        )

        data = {"leg_slugs": slugs, "vote_ids": vote_ids}
        return idata, matrix, legislators, data

    @pytest.fixture
    def flipped_idata(self) -> tuple[az.InferenceData, pl.DataFrame, pl.DataFrame, dict]:
        """Synthetic IRT output where sign is FLIPPED.

        Simulates horseshoe effect: extreme Rs placed at negative end.
        Vote matrix uses true ideology, but ideal points are negated.
        """
        n_r, n_d = 10, 5
        n_legs = n_r + n_d
        n_votes = 40
        rng = np.random.default_rng(42)

        slugs = [f"sen_r{i}_1" for i in range(n_r)] + [f"sen_d{i}_1" for i in range(n_d)]
        parties = ["Republican"] * n_r + ["Democrat"] * n_d

        # True ideology: Rs positive, Ds negative
        xi_true = np.concatenate(
            [
                np.linspace(0.5, 2.5, n_r),
                np.linspace(-2.5, -0.5, n_d),
            ]
        )

        # Generate votes from true ideology — center alphas to create contested votes
        vote_matrix = np.full((n_legs, n_votes), np.nan)
        alphas = rng.normal(0, 0.5, n_votes)  # centered near 0 for more contested splits
        for j in range(n_votes):
            p = 1 / (1 + np.exp(-(xi_true - alphas[j])))
            vote_matrix[:, j] = (rng.random(n_legs) < p).astype(float)

        matrix_data: dict = {"legislator_slug": slugs}
        vote_ids = [f"v{j}" for j in range(n_votes)]
        for j, vid in enumerate(vote_ids):
            matrix_data[vid] = vote_matrix[:, j].tolist()
        matrix = pl.DataFrame(matrix_data)

        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": [f"R{i}" for i in range(n_r)] + [f"D{i}" for i in range(n_d)],
                "party": parties,
                "district": list(range(1, n_legs + 1)),
                "chamber": ["Senate"] * n_legs,
            }
        )

        # Build fake idata with FLIPPED sign (negate ideal points)
        xi_flipped = -xi_true
        xi_posterior = xi_flipped[np.newaxis, np.newaxis, :] + rng.normal(0, 0.1, (2, 100, n_legs))
        beta_posterior = rng.normal(-1, 0.3, (2, 100, n_votes))  # also flipped
        xi_free_posterior = xi_posterior[:, :, :n_legs]

        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "xi": xr.DataArray(xi_posterior, dims=["chain", "draw", "legislator"]),
                    "xi_free": xr.DataArray(
                        xi_free_posterior, dims=["chain", "draw", "xi_free_dim_0"]
                    ),
                    "beta": xr.DataArray(beta_posterior, dims=["chain", "draw", "vote"]),
                }
            )
        )

        data = {"leg_slugs": slugs, "vote_ids": vote_ids}
        return idata, matrix, legislators, data

    def test_correct_sign_not_flipped(
        self,
        correctly_signed_idata: tuple,
    ) -> None:
        """Correctly signed ideal points should not be flipped."""
        idata, matrix, legislators, data = correctly_signed_idata
        xi_before = idata.posterior["xi"].mean(dim=["chain", "draw"]).values.copy()
        result_idata, was_flipped = validate_sign(
            idata,
            matrix,
            legislators,
            data,
            "Senate",
        )
        xi_after = result_idata.posterior["xi"].mean(dim=["chain", "draw"]).values
        assert not was_flipped
        np.testing.assert_array_almost_equal(xi_before, xi_after)

    def test_flipped_sign_detected_and_corrected(
        self,
        flipped_idata: tuple,
    ) -> None:
        """Flipped ideal points should be detected and negated."""
        idata, matrix, legislators, data = flipped_idata
        xi_before = idata.posterior["xi"].mean(dim=["chain", "draw"]).values.copy()
        result_idata, was_flipped = validate_sign(
            idata,
            matrix,
            legislators,
            data,
            "Senate",
        )
        xi_after = result_idata.posterior["xi"].mean(dim=["chain", "draw"]).values
        assert was_flipped
        # After flip, xi should be negated
        np.testing.assert_array_almost_equal(xi_after, -xi_before, decimal=5)

    def test_beta_also_negated_on_flip(
        self,
        flipped_idata: tuple,
    ) -> None:
        """Beta posteriors should also be negated when sign is flipped."""
        idata, matrix, legislators, data = flipped_idata
        beta_before = idata.posterior["beta"].mean(dim=["chain", "draw"]).values.copy()
        result_idata, was_flipped = validate_sign(
            idata,
            matrix,
            legislators,
            data,
            "Senate",
        )
        beta_after = result_idata.posterior["beta"].mean(dim=["chain", "draw"]).values
        assert was_flipped
        np.testing.assert_array_almost_equal(beta_after, -beta_before, decimal=5)

    def test_skips_with_too_few_legislators(self, legislators: pl.DataFrame) -> None:
        """Should skip when one party has fewer than 3 legislators."""
        # Only 3 Rs and 3 Ds in base fixture — need < 3 of one
        small_legs = legislators.head(4)  # 3 R + 1 D
        matrix = pl.DataFrame(
            {
                "legislator_slug": small_legs["legislator_slug"].to_list(),
                "v1": [1, 1, 0, 0],
            }
        )
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "xi": xr.DataArray(np.zeros((1, 10, 4)), dims=["chain", "draw", "legislator"]),
                }
            )
        )
        data = {"leg_slugs": small_legs["legislator_slug"].to_list()}
        result_idata, was_flipped = validate_sign(
            idata,
            matrix,
            small_legs,
            data,
            "Senate",
        )
        assert not was_flipped

    def test_skips_with_too_few_contested_votes(self) -> None:
        """Should skip when fewer than 10 contested votes."""
        n_r, n_d = 5, 5
        slugs = [f"sen_r{i}_1" for i in range(n_r)] + [f"sen_d{i}_1" for i in range(n_d)]
        # 3 votes, all party-line (not contested)
        matrix = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "v1": [1] * n_r + [0] * n_d,
                "v2": [1] * n_r + [0] * n_d,
                "v3": [1] * n_r + [0] * n_d,
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": slugs,
                "party": ["Republican"] * n_r + ["Democrat"] * n_d,
                "district": list(range(1, n_r + n_d + 1)),
                "chamber": ["Senate"] * (n_r + n_d),
            }
        )
        idata = az.InferenceData(
            posterior=xr.Dataset(
                {
                    "xi": xr.DataArray(
                        np.zeros((1, 10, n_r + n_d)), dims=["chain", "draw", "legislator"]
                    ),
                }
            )
        )
        data = {"leg_slugs": slugs}
        _, was_flipped = validate_sign(idata, matrix, legislators, data, "Senate")
        assert not was_flipped

    def test_contested_vote_threshold_constant(self) -> None:
        """Contested vote threshold should be 10%."""
        assert CONTESTED_VOTE_THRESHOLD == 0.10


# -- Identification Strategy Tests -------------------------------------------


class TestIdentificationStrategy:
    """Test IdentificationStrategy enumeration and constants.

    Run: uv run pytest tests/test_irt.py::TestIdentificationStrategy -v
    """

    def test_all_strategies_list_completeness(self) -> None:
        """ALL_STRATEGIES should contain exactly 7 strategies (not AUTO)."""
        IS = IdentificationStrategy
        assert len(IS.ALL_STRATEGIES) == 7
        assert IS.AUTO not in IS.ALL_STRATEGIES

    def test_all_strategies_have_descriptions(self) -> None:
        """Every strategy (including AUTO) should have a description."""
        IS = IdentificationStrategy
        for s in IS.ALL_STRATEGIES + [IS.AUTO]:
            assert s in IS.DESCRIPTIONS, f"Missing description for {s}"
            assert len(IS.DESCRIPTIONS[s]) > 10

    def test_all_strategies_have_references(self) -> None:
        """Every strategy should have a literature reference."""
        IS = IdentificationStrategy
        for s in IS.ALL_STRATEGIES:
            assert s in IS.REFERENCES, f"Missing reference for {s}"
            assert len(IS.REFERENCES[s]) > 5

    def test_strategy_string_values_match_cli(self) -> None:
        """Strategy string values should be valid CLI choices."""
        IS = IdentificationStrategy
        expected_cli = {
            "anchor-pca",
            "anchor-agreement",
            "sort-constraint",
            "positive-beta",
            "hierarchical-prior",
            "unconstrained",
            "external-prior",
        }
        assert set(IS.ALL_STRATEGIES) == expected_cli

    def test_auto_is_separate(self) -> None:
        """AUTO should not be in ALL_STRATEGIES but should exist."""
        IS = IdentificationStrategy
        assert IS.AUTO == "auto"
        assert IS.AUTO not in IS.ALL_STRATEGIES


class TestDetectSupermajority:
    """Test supermajority detection logic.

    Run: uv run pytest tests/test_irt.py::TestDetectSupermajority -v
    """

    def test_balanced_chamber(self) -> None:
        """50/50 split should not be supermajority."""
        legs = pl.DataFrame(
            {
                "legislator_slug": [f"s{i}" for i in range(10)],
                "party": ["Republican"] * 5 + ["Democrat"] * 5,
                "chamber": ["Senate"] * 10,
            }
        )
        is_super, frac = detect_supermajority(legs, "Senate")
        assert not is_super
        assert frac == pytest.approx(0.5)

    def test_supermajority_detected(self) -> None:
        """75% R should trigger supermajority."""
        legs = pl.DataFrame(
            {
                "legislator_slug": [f"s{i}" for i in range(20)],
                "party": ["Republican"] * 15 + ["Democrat"] * 5,
                "chamber": ["Senate"] * 20,
            }
        )
        is_super, frac = detect_supermajority(legs, "Senate")
        assert is_super
        assert frac == pytest.approx(0.75)

    def test_exactly_at_threshold(self) -> None:
        """Exactly 70% should trigger supermajority."""
        legs = pl.DataFrame(
            {
                "legislator_slug": [f"s{i}" for i in range(10)],
                "party": ["Republican"] * 7 + ["Democrat"] * 3,
                "chamber": ["Senate"] * 10,
            }
        )
        is_super, frac = detect_supermajority(legs, "Senate")
        assert is_super
        assert frac == pytest.approx(0.7)

    def test_just_below_threshold(self) -> None:
        """69% should NOT trigger supermajority."""
        # 69/100 = 0.69
        n_r, n_d = 69, 31
        legs = pl.DataFrame(
            {
                "legislator_slug": [f"s{i}" for i in range(100)],
                "party": ["Republican"] * n_r + ["Democrat"] * n_d,
                "chamber": ["Senate"] * 100,
            }
        )
        is_super, frac = detect_supermajority(legs, "Senate")
        assert not is_super
        assert frac == pytest.approx(0.69)

    def test_empty_chamber(self) -> None:
        """Empty chamber should return False, 0.0."""
        legs = pl.DataFrame(
            {
                "legislator_slug": pl.Series([], dtype=pl.Utf8),
                "party": pl.Series([], dtype=pl.Utf8),
                "chamber": pl.Series([], dtype=pl.Utf8),
            }
        )
        is_super, frac = detect_supermajority(legs, "Senate")
        assert not is_super
        assert frac == 0.0

    def test_wrong_chamber_filtered(self) -> None:
        """Should only consider legislators from the specified chamber."""
        legs = pl.DataFrame(
            {
                "legislator_slug": [f"s{i}" for i in range(10)],
                "party": ["Republican"] * 8 + ["Republican", "Democrat"],
                "chamber": ["House"] * 8 + ["Senate"] * 2,
            }
        )
        is_super, frac = detect_supermajority(legs, "Senate")
        # 2 Senate legislators, 1 R + 1 D → 50%
        assert not is_super

    def test_threshold_constant(self) -> None:
        """SUPERMAJORITY_THRESHOLD should be 0.70."""
        assert SUPERMAJORITY_THRESHOLD == 0.70


class TestSelectIdentificationStrategy:
    """Test auto-detection logic for identification strategy selection.

    Run: uv run pytest tests/test_irt.py::TestSelectIdentificationStrategy -v
    """

    def _make_chamber_data(
        self, n_r: int, n_d: int, n_contested: int = 20
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create legislators and vote matrix for strategy selection tests."""
        n = n_r + n_d
        slugs = [f"sen_r{i}_1" for i in range(n_r)] + [f"sen_d{i}_1" for i in range(n_d)]
        parties = ["Republican"] * n_r + ["Democrat"] * n_d

        legs = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": slugs,
                "party": parties,
                "district": list(range(1, n + 1)),
                "chamber": ["Senate"] * n,
            }
        )

        # Generate vote matrix with contested votes
        rng = np.random.default_rng(42)
        xi_true = np.concatenate(
            [
                np.linspace(0.5, 2.0, n_r),
                np.linspace(-2.0, -0.5, n_d),
            ]
        )
        matrix_data: dict = {"legislator_slug": slugs}
        for j in range(max(n_contested + 5, 30)):
            alpha = rng.normal(0, 0.3)
            p = 1 / (1 + np.exp(-(xi_true - alpha)))
            votes = (rng.random(n) < p).astype(float)
            matrix_data[f"v{j}"] = votes.tolist()
        return legs, pl.DataFrame(matrix_data)

    def test_auto_balanced_selects_pca(self) -> None:
        """Balanced chamber should auto-select anchor-pca."""
        legs, matrix = self._make_chamber_data(n_r=20, n_d=20)
        strategy, rationale = select_identification_strategy(
            "auto",
            legs,
            matrix,
            "Senate",
        )
        assert strategy == IdentificationStrategy.ANCHOR_PCA
        assert "SELECTED" in rationale[strategy]

    def test_auto_supermajority_selects_agreement(self) -> None:
        """Supermajority with contested votes should auto-select anchor-agreement."""
        legs, matrix = self._make_chamber_data(n_r=30, n_d=10, n_contested=30)
        strategy, rationale = select_identification_strategy(
            "auto",
            legs,
            matrix,
            "Senate",
        )
        assert strategy == IdentificationStrategy.ANCHOR_AGREEMENT
        assert "SELECTED" in rationale[strategy]

    def test_auto_external_scores_selects_external(self) -> None:
        """External scores available should auto-select external-prior."""
        legs, matrix = self._make_chamber_data(n_r=20, n_d=20)
        strategy, rationale = select_identification_strategy(
            "auto",
            legs,
            matrix,
            "Senate",
            external_scores_available=True,
        )
        assert strategy == IdentificationStrategy.EXTERNAL_PRIOR

    def test_user_override(self) -> None:
        """Explicit strategy should override auto-detection."""
        legs, matrix = self._make_chamber_data(n_r=20, n_d=20)
        strategy, rationale = select_identification_strategy(
            "sort-constraint",
            legs,
            matrix,
            "Senate",
        )
        assert strategy == IdentificationStrategy.SORT_CONSTRAINT
        assert "user override" in rationale[strategy]

    def test_rationale_covers_all_strategies(self) -> None:
        """Rationale dict should have an entry for every strategy."""
        legs, matrix = self._make_chamber_data(n_r=20, n_d=20)
        _, rationale = select_identification_strategy(
            "auto",
            legs,
            matrix,
            "Senate",
        )
        for s in IdentificationStrategy.ALL_STRATEGIES:
            assert s in rationale, f"Missing rationale for {s}"

    def test_exactly_one_selected(self) -> None:
        """Exactly one strategy should be marked SELECTED in the rationale."""
        legs, matrix = self._make_chamber_data(n_r=20, n_d=20)
        _, rationale = select_identification_strategy(
            "auto",
            legs,
            matrix,
            "Senate",
        )
        selected_count = sum(1 for v in rationale.values() if v.startswith("SELECTED"))
        assert selected_count == 1

    def test_non_selected_prefixed(self) -> None:
        """Non-selected strategies should be prefixed with 'Not selected.'."""
        legs, matrix = self._make_chamber_data(n_r=20, n_d=20)
        strategy, rationale = select_identification_strategy(
            "auto",
            legs,
            matrix,
            "Senate",
        )
        for s, desc in rationale.items():
            if s != strategy:
                assert desc.startswith("Not selected."), f"Strategy {s}: {desc}"


class TestBuildIrtGraphStrategies:
    """Test that build_irt_graph() produces valid PyMC models for each strategy.

    Run: uv run pytest tests/test_irt.py::TestBuildIrtGraphStrategies -v
    """

    @pytest.fixture
    def irt_data(self) -> dict:
        """Minimal IRT data dict for graph construction."""
        n_leg, n_votes = 10, 20
        rng = np.random.default_rng(42)
        n_obs = n_leg * n_votes
        return {
            "leg_idx": np.repeat(np.arange(n_leg), n_votes),
            "vote_idx": np.tile(np.arange(n_votes), n_leg),
            "y": rng.integers(0, 2, size=n_obs).astype(np.float64),
            "n_legislators": n_leg,
            "n_votes": n_votes,
            "n_obs": n_obs,
            "leg_slugs": [f"s{i}" for i in range(n_leg)],
            "vote_ids": [f"v{j}" for j in range(n_votes)],
        }

    def test_anchor_pca_model(self, irt_data: dict) -> None:
        """Anchor-PCA strategy should build a model with anchored xi."""
        anchors = [(0, 1.0), (9, -1.0)]
        model = build_irt_graph(
            irt_data,
            anchors,
            strategy=IdentificationStrategy.ANCHOR_PCA,
        )
        assert "xi" in model.named_vars
        assert "xi_free" in model.named_vars
        assert "alpha" in model.named_vars
        assert "beta" in model.named_vars
        # xi_free has n_leg - n_anchors shape
        assert model.named_vars["xi_free"].shape.eval().item() == 8

    def test_anchor_agreement_model(self, irt_data: dict) -> None:
        """Anchor-agreement should produce the same graph structure as anchor-pca."""
        anchors = [(2, 1.0), (7, -1.0)]
        model = build_irt_graph(
            irt_data,
            anchors,
            strategy=IdentificationStrategy.ANCHOR_AGREEMENT,
        )
        assert "xi_free" in model.named_vars
        assert model.named_vars["xi_free"].shape.eval().item() == 8

    def test_sort_constraint_model(self, irt_data: dict) -> None:
        """Sort-constraint should have party_order potential and all-free xi."""
        party_indices = {"Republican": [0, 1, 2, 3, 4], "Democrat": [5, 6, 7, 8, 9]}
        model = build_irt_graph(
            irt_data,
            [],
            strategy=IdentificationStrategy.SORT_CONSTRAINT,
            party_indices=party_indices,
        )
        assert "xi_free" in model.named_vars
        assert model.named_vars["xi_free"].shape.eval().item() == 10
        # Check that the Potential exists
        potential_names = [p.name for p in model.potentials]
        assert "party_order" in potential_names

    def test_positive_beta_model(self, irt_data: dict) -> None:
        """Positive-beta should use HalfNormal for beta."""

        model = build_irt_graph(
            irt_data,
            [],
            strategy=IdentificationStrategy.POSITIVE_BETA,
        )
        # HalfNormal is a distribution — check that it's bounded positive
        assert "beta" in model.named_vars
        # xi_free should be fully free (no anchors)
        assert model.named_vars["xi_free"].shape.eval().item() == 10

    def test_hierarchical_prior_model(self, irt_data: dict) -> None:
        """Hierarchical-prior should set party-aware mu for xi_free."""
        party_indices = {"Republican": [0, 1, 2, 3, 4], "Democrat": [5, 6, 7, 8, 9]}
        model = build_irt_graph(
            irt_data,
            [],
            strategy=IdentificationStrategy.HIERARCHICAL_PRIOR,
            party_indices=party_indices,
        )
        assert "xi_free" in model.named_vars
        assert model.named_vars["xi_free"].shape.eval().item() == 10
        # No potential for sort constraint
        potential_names = [p.name for p in model.potentials]
        assert "party_order" not in potential_names

    def test_unconstrained_model(self, irt_data: dict) -> None:
        """Unconstrained should have all-free xi with no potentials."""
        model = build_irt_graph(
            irt_data,
            [],
            strategy=IdentificationStrategy.UNCONSTRAINED,
        )
        assert "xi_free" in model.named_vars
        assert model.named_vars["xi_free"].shape.eval().item() == 10
        potential_names = [p.name for p in model.potentials]
        assert "party_order" not in potential_names

    def test_external_prior_model(self, irt_data: dict) -> None:
        """External-prior should use provided prior means."""
        priors = np.linspace(1.0, -1.0, 10)
        model = build_irt_graph(
            irt_data,
            [],
            strategy=IdentificationStrategy.EXTERNAL_PRIOR,
            external_priors=priors,
        )
        assert "xi_free" in model.named_vars
        assert model.named_vars["xi_free"].shape.eval().item() == 10

    def test_external_prior_defaults_to_zero(self, irt_data: dict) -> None:
        """External-prior with None priors should default to zero means."""
        model = build_irt_graph(
            irt_data,
            [],
            strategy=IdentificationStrategy.EXTERNAL_PRIOR,
            external_priors=None,
        )
        assert "xi_free" in model.named_vars

    def test_all_strategies_build_successfully(self, irt_data: dict) -> None:
        """Every strategy should produce a valid PyMC model without error."""
        party_indices = {"Republican": [0, 1, 2, 3, 4], "Democrat": [5, 6, 7, 8, 9]}
        anchors = [(0, 1.0), (9, -1.0)]
        priors = np.linspace(1.0, -1.0, 10)

        for strategy in IdentificationStrategy.ALL_STRATEGIES:
            a = anchors if strategy.startswith("anchor") else []
            pi = party_indices if strategy in ("sort-constraint", "hierarchical-prior") else None
            ep = priors if strategy == "external-prior" else None
            model = build_irt_graph(
                irt_data, a, strategy=strategy, party_indices=pi, external_priors=ep
            )
            assert "xi" in model.named_vars, f"Strategy {strategy} missing xi"
            assert "obs" in model.named_vars, f"Strategy {strategy} missing obs"


# -- Robustness Flags (ADR-0104) --


class TestRobustnessFlag:
    """Test RobustnessFlag dataclass and RobustnessFlags registry.

    Run: uv run pytest tests/test_irt.py::TestRobustnessFlag -v
    """

    def test_flag_is_frozen(self) -> None:
        """RobustnessFlag should be immutable."""
        flag = RobustnessFlag(
            name="test",
            label="Test",
            description="A test flag",
            enabled=True,
        )
        with pytest.raises(AttributeError):
            flag.enabled = False  # type: ignore[misc]

    def test_build_flags_all_off(self) -> None:
        """With no CLI flags set, all robustness flags should be OFF."""
        import argparse

        args = argparse.Namespace(
            contested_only=False,
            horseshoe_diagnostic=False,
            horseshoe_remediate=False,
            promote_2d=False,
        )
        flags = RobustnessFlags.build_flags(args)
        assert len(flags) == 4
        assert all(not f.enabled for f in flags)

    def test_build_flags_selective(self) -> None:
        """Enabling one flag should not affect others."""
        import argparse

        args = argparse.Namespace(
            contested_only=True,
            horseshoe_diagnostic=False,
            horseshoe_remediate=False,
            promote_2d=False,
        )
        flags = RobustnessFlags.build_flags(args)
        by_name = {f.name: f for f in flags}
        assert by_name["contested-only"].enabled is True
        assert by_name["horseshoe-diagnostic"].enabled is False
        assert by_name["horseshoe-remediate"].enabled is False
        assert by_name["promote-2d"].enabled is False

    def test_all_flags_have_labels_and_descriptions(self) -> None:
        """Every flag in the registry must have a label and description."""
        for name in RobustnessFlags.ALL_FLAGS:
            assert name in RobustnessFlags.LABELS
            assert name in RobustnessFlags.DESCRIPTIONS
            assert len(RobustnessFlags.LABELS[name]) > 0
            assert len(RobustnessFlags.DESCRIPTIONS[name]) > 0

    def test_constants_exist(self) -> None:
        """Robustness flag constants should be defined."""
        assert MIN_CONTESTED_FOR_REFIT == 50
        assert HORSESHOE_DEM_WRONG_SIDE_FRAC == 0.20
        assert PROMOTE_2D_RANK_SHIFT == 10
        assert MIN_PC2_VOTES_FOR_REFIT == 50
        assert PC2_PRIOR_SIGMA == 1.0


class TestFilterContestedVotes:
    """Test filter_contested_votes() function.

    Run: uv run pytest tests/test_irt.py::TestFilterContestedVotes -v
    """

    @pytest.fixture
    def partisan_matrix(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Matrix with clear party-line votes + some near-unanimous votes."""
        matrix = pl.DataFrame(
            {
                "legislator_slug": [
                    "rep_a_1",
                    "rep_b_1",
                    "rep_c_1",
                    "dem_d_1",
                    "dem_e_1",
                    "dem_f_1",
                ],
                # v1: contested (3R Yea / 3D Nay)
                "v1": [1, 1, 1, 0, 0, 0],
                # v2: near-unanimous (5 Yea / 1 Nay)
                "v2": [1, 1, 1, 1, 1, 0],
                # v3: contested (2R Yea, 1R Nay / 2D Nay, 1D Yea)
                "v3": [1, 1, 0, 0, 0, 1],
                # v4: unanimous (all Yea)
                "v4": [1, 1, 1, 1, 1, 1],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": [
                    "rep_a_1",
                    "rep_b_1",
                    "rep_c_1",
                    "dem_d_1",
                    "dem_e_1",
                    "dem_f_1",
                ],
                "party": [
                    "Republican",
                    "Republican",
                    "Republican",
                    "Democrat",
                    "Democrat",
                    "Democrat",
                ],
            }
        )
        return matrix, legislators

    def test_filters_to_contested_only(self, partisan_matrix) -> None:
        """Should keep only contested votes."""
        matrix, legislators = partisan_matrix
        filtered, n_contested, n_total = filter_contested_votes(matrix, legislators)
        assert n_total == 4
        # v1 and v3 are contested; v2 and v4 are not
        assert n_contested >= 1
        assert "legislator_slug" in filtered.columns
        # Filtered should have fewer vote columns
        vote_cols = [c for c in filtered.columns if c != "legislator_slug"]
        assert len(vote_cols) == n_contested

    def test_returns_original_when_no_parties(self) -> None:
        """With only one party, should return full matrix."""
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1", "rep_b_1"],
                "v1": [1, 0],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_1", "rep_b_1"],
                "party": ["Republican", "Republican"],
            }
        )
        filtered, n_contested, n_total = filter_contested_votes(matrix, legislators)
        assert n_contested == 0
        assert n_total == 1


class TestFilterPC2DominantVotes:
    """Test filter_pc2_dominant_votes() function.

    Run: uv run pytest tests/test_irt.py::TestFilterPC2DominantVotes -v
    """

    def test_keeps_pc2_dominant_votes(self) -> None:
        """Should keep votes where |PC2| > |PC1| and drop others."""
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "d1"],
                "v1": [1, 1, 0],
                "v2": [1, 0, 1],
                "v3": [0, 1, 0],
            }
        )
        loadings = pl.DataFrame(
            {
                "vote_id": ["v1", "v2", "v3"],
                "PC1": [0.8, 0.2, 0.5],  # v1 PC1-dominant, v2 PC2-dominant, v3 tie
                "PC2": [0.3, 0.9, 0.5],
            }
        )
        filtered, n_kept, n_total = filter_pc2_dominant_votes(matrix, loadings)
        assert n_total == 3
        assert n_kept == 1
        assert "v2" in filtered.columns
        assert "v1" not in filtered.columns

    def test_returns_all_when_no_pc2_dominant(self) -> None:
        """Should return all votes if no votes are PC2-dominant."""
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["r1", "d1"],
                "v1": [1, 0],
            }
        )
        loadings = pl.DataFrame(
            {
                "vote_id": ["v1"],
                "PC1": [0.9],
                "PC2": [0.1],
            }
        )
        filtered, n_kept, n_total = filter_pc2_dominant_votes(matrix, loadings)
        assert n_kept == n_total == 1

    def test_uses_absolute_values(self) -> None:
        """Negative loadings should be compared by absolute value."""
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["r1"],
                "v1": [1],
                "v2": [0],
            }
        )
        loadings = pl.DataFrame(
            {
                "vote_id": ["v1", "v2"],
                "PC1": [0.3, -0.8],
                "PC2": [-0.7, 0.2],
            }
        )
        filtered, n_kept, n_total = filter_pc2_dominant_votes(matrix, loadings)
        assert n_kept == 1
        assert "v1" in filtered.columns  # |PC2|=0.7 > |PC1|=0.3


class TestDetectHorseshoe:
    """Test detect_horseshoe() function.

    Run: uv run pytest tests/test_irt.py::TestDetectHorseshoe -v
    """

    def test_no_horseshoe_in_balanced_chamber(self) -> None:
        """Balanced chamber with proper separation should not detect horseshoe."""
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "r3", "d1", "d2", "d3"],
                "xi_mean": [2.0, 1.5, 1.0, -1.0, -1.5, -2.0],
                "party": ["Republican"] * 3 + ["Democrat"] * 3,
                "full_name": ["R1", "R2", "R3", "D1", "D2", "D3"],
            }
        )
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "r3", "d1", "d2", "d3"],
                "PC1": [2.0, 1.5, 1.0, -1.0, -1.5, -2.0],
                "PC2": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1],
            }
        )
        result = detect_horseshoe(ideal_points, pca_scores, "Senate")
        assert result["detected"] is False
        assert result["dem_wrong_side_frac"] == 0.0
        assert not result["r_more_liberal_than_d_mean"]

    def test_detects_horseshoe_when_dems_wrong_side(self) -> None:
        """Should detect horseshoe when Democrats are on the conservative side."""
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "r3", "d1", "d2", "d3"],
                "xi_mean": [-3.0, -2.0, 1.0, 0.5, 0.3, -0.5],
                "party": ["Republican"] * 3 + ["Democrat"] * 3,
                "full_name": ["R1-rebel", "R2-rebel", "R3-moderate", "D1", "D2", "D3"],
            }
        )
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "r3", "d1", "d2", "d3"],
                "PC1": [-3.0, -2.0, 1.0, 0.5, 0.3, -0.5],
                "PC2": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1],
            }
        )
        result = detect_horseshoe(ideal_points, pca_scores, "Senate")
        assert result["detected"] is True
        # At least one of the detection criteria should trigger
        assert (
            result["dem_wrong_side_frac"] > 0.20
            or result["r_more_liberal_than_d_mean"]
            or result["overlap_frac"] > 0.30
        )

    def test_detects_r_more_liberal_than_d_mean(self) -> None:
        """Should flag when a Republican is more liberal than Democrat mean."""
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "d1", "d2"],
                "xi_mean": [1.0, -2.0, -0.5, -1.0],
                "party": ["Republican", "Republican", "Democrat", "Democrat"],
                "full_name": ["R-normal", "R-rebel", "D1", "D2"],
            }
        )
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2", "d1", "d2"],
                "PC1": [1.0, -2.0, -0.5, -1.0],
                "PC2": [0.1, -0.1, 0.2, -0.2],
            }
        )
        result = detect_horseshoe(ideal_points, pca_scores, "Senate")
        assert result["r_more_liberal_than_d_mean"] is True
        assert result["most_neg_r_name"] == "R-rebel"


class TestCrossReference2D:
    """Test cross_reference_2d() function.

    Run: uv run pytest tests/test_irt.py::TestCrossReference2D -v
    """

    def test_returns_none_when_no_2d_results(self, tmp_path) -> None:
        """Should return None when 2D results don't exist."""
        ideal_points = pl.DataFrame(
            {
                "legislator_slug": ["s1"],
                "xi_mean": [1.0],
                "full_name": ["S1"],
                "party": ["R"],
            }
        )
        result = cross_reference_2d(ideal_points, tmp_path, "Senate")
        assert result is None

    def test_cross_references_matching_legislators(self, tmp_path) -> None:
        """Should compute rank shifts between 1D and 2D models."""
        # Create 1D ideal points
        ideal_points_1d = pl.DataFrame(
            {
                "legislator_slug": ["s1", "s2", "s3", "s4"],
                "xi_mean": [3.0, 2.0, -1.0, -2.0],
                "full_name": ["S1", "S2", "S3", "S4"],
                "party": ["R", "R", "D", "D"],
            }
        )
        # Create 2D results with different ordering on Dim1
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        ip_2d = pl.DataFrame(
            {
                "legislator_slug": ["s1", "s2", "s3", "s4"],
                "xi_dim1_mean": [1.0, 3.0, -2.0, -1.0],  # s1 and s2 swap, s3 and s4 swap
            }
        )
        ip_2d.write_parquet(data_dir / "ideal_points_2d_senate.parquet")

        result = cross_reference_2d(ideal_points_1d, tmp_path, "Senate")
        assert result is not None
        assert result["n_matched"] == 4
        assert isinstance(result["correlation"], float)
        # s1 was rank 1 in 1D, rank 2 in 2D → shift of 1. No big shifts here.
        assert result["n_flagged"] == 0  # shifts are small (1 position)

    def test_flags_large_rank_shifts(self, tmp_path) -> None:
        """Should flag legislators with rank shifts > PROMOTE_2D_RANK_SHIFT."""
        n = 20
        slugs = [f"s{i}" for i in range(n)]
        # 1D ordering: s0 is most conservative
        xi_1d = list(np.linspace(3.0, -3.0, n))
        # 2D: s0 drops from rank 1 to rank 15 (shift=14)
        xi_2d = list(np.linspace(3.0, -3.0, n))
        xi_2d[0] = -2.0  # s0 drops to near the bottom

        ideal_1d = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "xi_mean": xi_1d,
                "full_name": [f"Leg {i}" for i in range(n)],
                "party": ["Republican"] * (n // 2) + ["Democrat"] * (n // 2),
            }
        )
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        ip_2d = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "xi_dim1_mean": xi_2d,
            }
        )
        ip_2d.write_parquet(data_dir / "ideal_points_2d_senate.parquet")

        result = cross_reference_2d(ideal_1d, tmp_path, "Senate")
        assert result is not None
        assert result["n_flagged"] >= 1  # s0 should be flagged
