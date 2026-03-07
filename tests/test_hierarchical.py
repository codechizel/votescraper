"""
Tests for hierarchical Bayesian IRT helper functions.

These tests verify the non-MCMC functions in analysis/hierarchical.py:
data preparation, result extraction, variance decomposition, and shrinkage
comparison. MCMC sampling is not tested (too slow for unit tests).

Run: uv run pytest tests/test_hierarchical.py -v
"""

import sys
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pytest
import xarray as xr

# Add project root to path so we can import analysis.hierarchical
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytensor.tensor as pt
from analysis.hierarchical import (
    MIN_GROUP_SIZE_WARN,
    PARTY_NAMES,
    SMALL_GROUP_SIGMA_SCALE,
    SMALL_GROUP_THRESHOLD,
    _match_bills_across_chambers,
    build_joint_graph,
    build_per_chamber_graph,
    compute_flat_hier_correlation,
    compute_variance_decomposition,
    extract_group_params,
    extract_hierarchical_ideal_points,
    fix_joint_sign_convention,
    prepare_hierarchical_data,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def house_matrix() -> pl.DataFrame:
    """Synthetic House vote matrix: 8 legislators x 4 votes.

    Layout: 5 Republicans (A-E), 3 Democrats (F-H).
    v1: party-line (R=Yea, D=Nay)
    v2: bipartisan (all Yea except H absent)
    v3: mixed (A,B,C,F Yea; D,E,G,H Nay)
    v4: reverse party-line (D=Yea, R=Nay)
    """
    return pl.DataFrame(
        {
            "legislator_slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_d_d_1",
                "rep_e_e_1",
                "rep_f_f_1",
                "rep_g_g_1",
                "rep_h_h_1",
            ],
            "v1": [1, 1, 1, 1, 1, 0, 0, 0],
            "v2": [1, 1, 1, 1, 1, 1, 1, None],
            "v3": [1, 1, 1, 0, 0, 1, 0, 0],
            "v4": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def legislators() -> pl.DataFrame:
    """Legislator metadata matching house_matrix."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F", "G G", "H H"],
            "legislator_slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_d_d_1",
                "rep_e_e_1",
                "rep_f_f_1",
                "rep_g_g_1",
                "rep_h_h_1",
            ],
            "chamber": ["House"] * 8,
            "party": ["Republican"] * 5 + ["Democrat"] * 3,
            "district": [1, 2, 3, 4, 5, 6, 7, 8],
            "member_url": [""] * 8,
        }
    )


@pytest.fixture
def flat_ideal_points() -> pl.DataFrame:
    """Flat IRT ideal points matching house_matrix legislators."""
    return pl.DataFrame(
        {
            "legislator_slug": [
                "rep_a_a_1",
                "rep_b_b_1",
                "rep_c_c_1",
                "rep_d_d_1",
                "rep_e_e_1",
                "rep_f_f_1",
                "rep_g_g_1",
                "rep_h_h_1",
            ],
            "xi_mean": [1.5, 1.2, 0.8, 0.5, 0.3, -0.8, -1.2, -1.5],
            "xi_sd": [0.15, 0.12, 0.14, 0.13, 0.11, 0.16, 0.14, 0.18],
            "xi_hdi_2.5": [1.2, 0.9, 0.5, 0.2, 0.1, -1.1, -1.5, -1.8],
            "xi_hdi_97.5": [1.8, 1.5, 1.1, 0.8, 0.5, -0.5, -0.9, -1.2],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F", "G G", "H H"],
            "party": ["Republican"] * 5 + ["Democrat"] * 3,
            "district": [1, 2, 3, 4, 5, 6, 7, 8],
            "chamber": ["House"] * 8,
        }
    )


def _make_fake_idata(
    n_legislators: int = 8,
    n_votes: int = 4,
    n_parties: int = 2,
    n_chains: int = 2,
    n_draws: int = 100,
    xi_values: np.ndarray | None = None,
    mu_party_values: np.ndarray | None = None,
    sigma_within_values: np.ndarray | None = None,
    leg_slugs: list[str] | None = None,
    vote_ids: list[str] | None = None,
    party_names: list[str] | None = None,
) -> "xr.Dataset":
    """Create a fake ArviZ InferenceData-like object for testing extraction."""
    import arviz as az

    if leg_slugs is None:
        leg_slugs = [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(n_legislators)]
    if vote_ids is None:
        vote_ids = [f"v{i + 1}" for i in range(n_votes)]
    if party_names is None:
        party_names = PARTY_NAMES

    # Default ideal points: Republicans positive, Democrats negative
    if xi_values is None:
        xi_values = np.tile(
            np.linspace(1.5, -1.5, n_legislators), (n_chains, n_draws, 1)
        ) + np.random.default_rng(42).normal(0, 0.05, (n_chains, n_draws, n_legislators))

    if mu_party_values is None:
        # D mean = -1.0, R mean = +1.0
        mu_party_values = np.zeros((n_chains, n_draws, n_parties))
        mu_party_values[:, :, 0] = -1.0 + np.random.default_rng(42).normal(
            0, 0.05, (n_chains, n_draws)
        )
        mu_party_values[:, :, 1] = 1.0 + np.random.default_rng(42).normal(
            0, 0.05, (n_chains, n_draws)
        )

    if sigma_within_values is None:
        sigma_within_values = np.ones((n_chains, n_draws, n_parties)) * 0.5
        sigma_within_values += np.random.default_rng(42).normal(
            0, 0.02, (n_chains, n_draws, n_parties)
        )
        sigma_within_values = np.abs(sigma_within_values)

    alpha_values = np.random.default_rng(42).normal(0, 1, (n_chains, n_draws, n_votes))
    beta_values = np.random.default_rng(42).normal(0, 0.5, (n_chains, n_draws, n_votes))

    posterior = xr.Dataset(
        {
            "xi": xr.DataArray(
                xi_values,
                dims=["chain", "draw", "legislator"],
                coords={"legislator": leg_slugs},
            ),
            "mu_party": xr.DataArray(
                mu_party_values,
                dims=["chain", "draw", "party"],
                coords={"party": party_names},
            ),
            "sigma_within": xr.DataArray(
                sigma_within_values,
                dims=["chain", "draw", "party"],
                coords={"party": party_names},
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
    idata = az.InferenceData(posterior=posterior)
    return idata


# ── TestPrepareHierarchicalData ──────────────────────────────────────────────


class TestPrepareHierarchicalData:
    """Test extension of flat IRT data with party indices."""

    def test_party_idx_shape(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """party_idx should have one entry per legislator."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["party_idx"].shape == (8,)

    def test_party_idx_values(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """Republicans should map to 1, Democrats to 0."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        # First 5 are Republican (1), last 3 are Democrat (0)
        assert list(data["party_idx"][:5]) == [1, 1, 1, 1, 1]
        assert list(data["party_idx"][5:]) == [0, 0, 0]

    def test_both_parties_present(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Both party indices (0 and 1) should appear."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        unique = set(data["party_idx"])
        assert 0 in unique
        assert 1 in unique

    def test_n_parties(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """n_parties should be 2."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["n_parties"] == 2

    def test_party_names(self, house_matrix: pl.DataFrame, legislators: pl.DataFrame) -> None:
        """party_names should match PARTY_NAMES constant."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["party_names"] == PARTY_NAMES

    def test_preserves_irt_data_fields(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Should preserve all fields from prepare_irt_data."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert "leg_idx" in data
        assert "vote_idx" in data
        assert "y" in data
        assert "n_legislators" in data
        assert "n_votes" in data
        assert "n_obs" in data
        assert "leg_slugs" in data
        assert "vote_ids" in data

    def test_single_party_matrix(self, legislators: pl.DataFrame) -> None:
        """Matrix with only one party should still work."""
        single_party_matrix = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "rep_b_b_1", "rep_c_c_1"],
                "v1": [1, 1, 0],
                "v2": [1, 0, 1],
            }
        )
        # All 3 slugs are Republican in the legislators fixture
        data = prepare_hierarchical_data(single_party_matrix, legislators, "House")
        assert all(data["party_idx"] == 1)  # All Republican


# ── TestBuildHierarchicalModel ───────────────────────────────────────────────


class TestBuildHierarchicalModel:
    """Test PyMC model structure (no sampling)."""

    def test_model_has_mu_party(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Model should have mu_party variable."""
        import pymc as pm

        data = prepare_hierarchical_data(house_matrix, legislators, "House")

        with pm.Model() as model:
            mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=2)
            pm.Deterministic("mu_party", pt.sort(mu_party_raw))
            sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=2)
            xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=data["n_legislators"])
            pm.Deterministic(
                "xi",
                mu_party_raw[data["party_idx"]]  # Use raw for structural test
                + sigma_within[data["party_idx"]] * xi_offset,
            )

        assert "mu_party" in model.named_vars
        assert "sigma_within" in model.named_vars
        assert "xi" in model.named_vars

    def test_xi_shape_matches_legislators(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """xi should have shape (n_legislators,)."""
        import pymc as pm

        data = prepare_hierarchical_data(house_matrix, legislators, "House")

        with pm.Model():
            mu_party_raw = pm.Normal("mu_party_raw", mu=0, sigma=2, shape=2)
            sigma_within = pm.HalfNormal("sigma_within", sigma=1, shape=2)
            xi_offset = pm.Normal("xi_offset", mu=0, sigma=1, shape=data["n_legislators"])
            xi = pm.Deterministic(
                "xi",
                mu_party_raw[data["party_idx"]] + sigma_within[data["party_idx"]] * xi_offset,
            )

        assert xi.eval().shape == (data["n_legislators"],)

    def test_ordering_constraint(self) -> None:
        """pt.sort should enforce mu_party[0] <= mu_party[1]."""
        raw = np.array([2.0, -1.0])
        sorted_vals = pt.sort(pt.as_tensor_variable(raw)).eval()
        assert sorted_vals[0] < sorted_vals[1]

    def test_ordering_already_sorted(self) -> None:
        """pt.sort on already-sorted input should be identity."""
        raw = np.array([-1.0, 2.0])
        sorted_vals = pt.sort(pt.as_tensor_variable(raw)).eval()
        np.testing.assert_array_almost_equal(sorted_vals, raw)

    def test_build_per_chamber_graph_returns_model(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """build_per_chamber_graph should return a PyMC model with expected RVs."""
        import pymc as pm

        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        model = build_per_chamber_graph(data)
        assert isinstance(model, pm.Model)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "xi_offset" in rv_names
        assert "alpha" in rv_names
        assert "mu_party_raw" in rv_names
        assert "sigma_within" in rv_names

    def test_build_per_chamber_graph_coords(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """build_per_chamber_graph model should have legislator and vote coords."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        model = build_per_chamber_graph(data)
        assert "legislator" in model.coords
        assert "vote" in model.coords
        assert "party" in model.coords
        assert len(model.coords["legislator"]) == data["n_legislators"]
        assert len(model.coords["vote"]) == data["n_votes"]


# ── TestBuildJointGraph ─────────────────────────────────────────────────────


class TestBuildJointGraph:
    """Tests for build_joint_graph model construction.

    Run: uv run pytest tests/test_hierarchical.py::TestBuildJointGraph -v
    """

    @pytest.fixture
    def senate_matrix(self) -> pl.DataFrame:
        """Synthetic Senate vote matrix: 4 legislators x 3 votes."""
        return pl.DataFrame(
            {
                "legislator_slug": [
                    "sen_i_i_1",
                    "sen_j_j_1",
                    "sen_k_k_1",
                    "sen_l_l_1",
                ],
                "v5": [1, 1, 0, 0],
                "v6": [1, 0, 1, 0],
                "v7": [0, 1, 0, 1],
            }
        )

    @pytest.fixture
    def senate_legislators(self) -> pl.DataFrame:
        """Legislator metadata matching senate_matrix."""
        return pl.DataFrame(
            {
                "name": ["I", "J", "K", "L"],
                "full_name": ["I I", "J J", "K K", "L L"],
                "legislator_slug": ["sen_i_i_1", "sen_j_j_1", "sen_k_k_1", "sen_l_l_1"],
                "chamber": ["Senate"] * 4,
                "party": ["Republican"] * 2 + ["Democrat"] * 2,
                "district": [9, 10, 11, 12],
                "member_url": [""] * 4,
            }
        )

    def test_build_joint_graph_returns_model_and_data(
        self,
        house_matrix: pl.DataFrame,
        legislators: pl.DataFrame,
        senate_matrix: pl.DataFrame,
        senate_legislators: pl.DataFrame,
    ) -> None:
        """build_joint_graph should return (pm.Model, dict) with expected free RVs."""
        import pymc as pm

        house_data = prepare_hierarchical_data(house_matrix, legislators, "House")
        senate_data = prepare_hierarchical_data(senate_matrix, senate_legislators, "Senate")
        model, combined_data = build_joint_graph(house_data, senate_data)
        assert isinstance(model, pm.Model)
        assert isinstance(combined_data, dict)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "mu_global" in rv_names
        assert "sigma_chamber" in rv_names
        assert "chamber_offset" in rv_names
        assert "sigma_party" in rv_names
        assert "group_offset_raw" in rv_names
        assert "sigma_within" in rv_names
        assert "xi_offset" in rv_names
        assert "alpha" in rv_names

    def test_build_joint_graph_coords(
        self,
        house_matrix: pl.DataFrame,
        legislators: pl.DataFrame,
        senate_matrix: pl.DataFrame,
        senate_legislators: pl.DataFrame,
    ) -> None:
        """build_joint_graph model should have legislator, vote, group, chamber coords."""
        house_data = prepare_hierarchical_data(house_matrix, legislators, "House")
        senate_data = prepare_hierarchical_data(senate_matrix, senate_legislators, "Senate")
        model, combined_data = build_joint_graph(house_data, senate_data)
        assert "legislator" in model.coords
        assert "vote" in model.coords
        assert "group" in model.coords
        assert "chamber" in model.coords
        assert len(model.coords["legislator"]) == combined_data["n_legislators"]
        assert len(model.coords["vote"]) == combined_data["n_votes"]
        assert len(model.coords["group"]) == 4
        assert len(model.coords["chamber"]) == 2

    def test_build_joint_graph_combined_data(
        self,
        house_matrix: pl.DataFrame,
        legislators: pl.DataFrame,
        senate_matrix: pl.DataFrame,
        senate_legislators: pl.DataFrame,
    ) -> None:
        """combined_data should have correct legislator counts."""
        house_data = prepare_hierarchical_data(house_matrix, legislators, "House")
        senate_data = prepare_hierarchical_data(senate_matrix, senate_legislators, "Senate")
        _, combined_data = build_joint_graph(house_data, senate_data)
        assert combined_data["n_house"] == house_data["n_legislators"]
        assert combined_data["n_senate"] == senate_data["n_legislators"]
        expected = house_data["n_legislators"] + senate_data["n_legislators"]
        assert combined_data["n_legislators"] == expected


# ── TestExtractHierarchicalResults ───────────────────────────────────────────


class TestExtractHierarchicalResults:
    """Test posterior extraction and shrinkage computation."""

    def test_output_columns(self, legislators: pl.DataFrame) -> None:
        """Output should have required columns."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators)
        required = {
            "legislator_slug",
            "xi_mean",
            "xi_sd",
            "xi_hdi_2.5",
            "xi_hdi_97.5",
            "party_mean",
        }
        assert required.issubset(set(df.columns))

    def test_shrinkage_columns_with_flat(
        self, legislators: pl.DataFrame, flat_ideal_points: pl.DataFrame
    ) -> None:
        """With flat_ip provided, should have shrinkage columns including flat_xi_rescaled."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ideal_points)
        assert "delta_from_flat" in df.columns
        assert "toward_party_mean" in df.columns
        assert "flat_xi_mean" in df.columns
        assert "flat_xi_rescaled" in df.columns

    def test_party_mean_assignment(self, legislators: pl.DataFrame) -> None:
        """Republicans should have positive party_mean, Democrats negative."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators)

        r_mean = df.filter(pl.col("party") == "Republican")["party_mean"].to_list()
        d_mean = df.filter(pl.col("party") == "Democrat")["party_mean"].to_list()

        # All R party means should be the same value, and > all D party means
        assert len(set(round(x, 3) for x in r_mean)) == 1  # Same for all R
        assert len(set(round(x, 3) for x in d_mean)) == 1  # Same for all D
        assert r_mean[0] > d_mean[0]

    def test_sort_order(self, legislators: pl.DataFrame) -> None:
        """Output should be sorted by xi_mean descending."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators)
        means = df["xi_mean"].to_list()
        assert means == sorted(means, reverse=True)


# ── TestExtractGroupParams ───────────────────────────────────────────────────


class TestExtractGroupParams:
    """Test extraction of party-level parameters."""

    def test_schema(self) -> None:
        """Output should have all expected columns."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_group_params(idata, data)
        required = {
            "party",
            "n_legislators",
            "mu_mean",
            "mu_sd",
            "mu_hdi_2.5",
            "mu_hdi_97.5",
            "sigma_within_mean",
            "sigma_within_sd",
        }
        assert required.issubset(set(df.columns))

    def test_two_parties(self) -> None:
        """Should have exactly 2 rows (one per party)."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_group_params(idata, data)
        assert df.height == 2

    def test_legislator_counts(self) -> None:
        """N legislators should match the party_idx counts."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_group_params(idata, data)
        d_row = df.filter(pl.col("party") == "Democrat")
        r_row = df.filter(pl.col("party") == "Republican")
        assert d_row["n_legislators"][0] == 3
        assert r_row["n_legislators"][0] == 5


# ── TestVarianceDecomposition ────────────────────────────────────────────────


class TestVarianceDecomposition:
    """Test ICC computation from posterior samples."""

    def test_icc_bounded(self) -> None:
        """ICC should be between 0 and 1."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert 0 <= icc <= 1

    def test_high_icc_when_separated(self) -> None:
        """When party means are far apart and within-party SD is small, ICC should be high."""
        n_chains, n_draws, n_parties = 2, 100, 2
        # Very separated means: D=-5, R=+5
        mu_values = np.zeros((n_chains, n_draws, n_parties))
        mu_values[:, :, 0] = -5.0
        mu_values[:, :, 1] = 5.0
        # Very small within-party SD
        sigma_values = np.ones((n_chains, n_draws, n_parties)) * 0.1

        idata = _make_fake_idata(
            mu_party_values=mu_values,
            sigma_within_values=sigma_values,
        )
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert icc > 0.9, f"ICC should be high when parties are well separated, got {icc}"

    def test_low_icc_when_overlapping(self) -> None:
        """When party means are identical and within-party SD is large, ICC should be low."""
        n_chains, n_draws, n_parties = 2, 100, 2
        # Same means for both parties
        mu_values = np.zeros((n_chains, n_draws, n_parties))
        # Large within-party SD
        sigma_values = np.ones((n_chains, n_draws, n_parties)) * 5.0

        idata = _make_fake_idata(
            mu_party_values=mu_values,
            sigma_within_values=sigma_values,
        )
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert icc < 0.1, f"ICC should be low when parties overlap, got {icc}"

    def test_icc_schema(self) -> None:
        """Output should have icc_mean, icc_sd, icc_ci_* columns."""
        idata = _make_fake_idata()
        data = {
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = compute_variance_decomposition(idata, data)
        assert "icc_mean" in df.columns
        assert "icc_sd" in df.columns
        assert "icc_ci_2.5" in df.columns
        assert "icc_ci_97.5" in df.columns
        assert df.height == 1


# ── TestCompareWithFlat ──────────────────────────────────────────────────────


class TestCompareWithFlat:
    """Test shrinkage comparison between hierarchical and flat IRT."""

    def test_correlation_high_when_similar(self) -> None:
        """Correlated ideal points should produce high Pearson r."""
        hier_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.5, 0.5, -0.5, -1.5],
            }
        )
        flat_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d"],
                "xi_mean": [1.4, 0.6, -0.4, -1.4],
            }
        )
        r = compute_flat_hier_correlation(hier_ip, flat_ip, "House")
        assert r > 0.99

    def test_correlation_handles_missing(self) -> None:
        """With < 3 overlap, should return NaN."""
        hier_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c"],
                "xi_mean": [1.5, 0.5, -0.5],
            }
        )
        flat_ip = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "d"],
                "xi_mean": [1.4, 0.6, -1.0],
            }
        )
        r = compute_flat_hier_correlation(hier_ip, flat_ip, "House")
        # Only 2 overlap — code returns NaN when < 3 matched
        assert np.isnan(r)

    def test_shrinkage_toward_party_mean(
        self, legislators: pl.DataFrame, flat_ideal_points: pl.DataFrame
    ) -> None:
        """toward_party_mean should be computed for legislators with sufficient flat_dist."""
        # Build hierarchical IPs that are explicitly closer to party means than the flat IPs.
        # Party means: D=-1.0, R=+1.0. Move each legislator halfway toward their party mean.
        flat_means = [1.5, 1.2, 0.8, 0.5, 0.3, -0.8, -1.2, -1.5]
        party_means = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]  # R=1.0, D=-1.0
        # Shrunk = halfway between flat and party mean
        shrunk = [(f + p) / 2 for f, p in zip(flat_means, party_means)]

        xi_values = np.tile(shrunk, (2, 100, 1))  # (chains, draws, legislators)
        xi_values += np.random.default_rng(42).normal(0, 0.01, xi_values.shape)

        idata = _make_fake_idata(xi_values=xi_values)
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ideal_points)

        non_null = df.drop_nulls(subset=["toward_party_mean"])
        assert non_null.height > 0
        # With explicitly shrunk values, majority should move toward party mean
        toward_count = non_null.filter(pl.col("toward_party_mean"))["toward_party_mean"].len()
        assert toward_count >= non_null.height // 2, (
            f"Expected majority to shrink toward party mean, got {toward_count}/{non_null.height}"
        )

    def test_delta_sign(self, legislators: pl.DataFrame, flat_ideal_points: pl.DataFrame) -> None:
        """delta_from_flat should be hier - rescaled_flat (not raw flat).

        uv run pytest tests/test_hierarchical.py::TestCompareWithFlat::test_delta_sign -v
        """
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ideal_points)
        # delta_from_flat uses rescaled flat values, so residuals should be small
        # (the rescaling is a best-fit linear transform)
        deltas = df.drop_nulls(subset=["delta_from_flat"])["delta_from_flat"]
        # Mean absolute delta should be small relative to the spread of ideal points
        xi_range = df["xi_mean"].max() - df["xi_mean"].min()
        mean_abs_delta = deltas.abs().mean()
        assert mean_abs_delta < xi_range * 0.5, (
            f"Mean |delta| = {mean_abs_delta:.3f} too large relative to range {xi_range:.3f}"
        )


# ── TestSmallGroupWarning ──────────────────────────────────────────────────


class TestSmallGroupWarning:
    """Test small-group warnings in prepare_hierarchical_data.

    Run: uv run pytest tests/test_hierarchical.py::TestSmallGroupWarning -v
    """

    def test_small_group_triggers_warning(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame, capsys: pytest.CaptureFixture
    ) -> None:
        """A party with fewer than MIN_GROUP_SIZE_WARN legislators should print WARNING."""
        # The fixture has 5 R + 3 D, both below MIN_GROUP_SIZE_WARN (15)
        prepare_hierarchical_data(house_matrix, legislators, "House")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "hierarchical shrinkage may be unreliable" in captured.out.lower()

    def test_large_groups_no_warning(
        self, legislators: pl.DataFrame, capsys: pytest.CaptureFixture
    ) -> None:
        """Groups at or above MIN_GROUP_SIZE_WARN should not trigger a warning."""
        # Create a matrix with enough legislators to avoid the warning
        n_r = MIN_GROUP_SIZE_WARN + 1
        n_d = MIN_GROUP_SIZE_WARN + 1
        slugs_r = [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(n_r)]
        slugs_d = [f"rep_d{i}_d{i}_1" for i in range(n_d)]
        all_slugs = slugs_r + slugs_d

        # Build matching legislator metadata
        big_legislators = pl.DataFrame(
            {
                "name": [f"L{i}" for i in range(n_r + n_d)],
                "full_name": [f"L{i} L{i}" for i in range(n_r + n_d)],
                "legislator_slug": all_slugs,
                "chamber": ["House"] * (n_r + n_d),
                "party": ["Republican"] * n_r + ["Democrat"] * n_d,
                "district": list(range(1, n_r + n_d + 1)),
                "member_url": [""] * (n_r + n_d),
            }
        )

        # Build a simple vote matrix
        rng = np.random.default_rng(42)
        matrix = pl.DataFrame(
            {
                "legislator_slug": all_slugs,
                "v1": rng.integers(0, 2, n_r + n_d).tolist(),
                "v2": rng.integers(0, 2, n_r + n_d).tolist(),
            }
        )

        prepare_hierarchical_data(matrix, big_legislators, "House")
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


# ── TestJointModelOrdering ─────────────────────────────────────────────────


class TestJointModelOrdering:
    """Test joint model per-chamber-pair ordering constraint.

    Run: uv run pytest tests/test_hierarchical.py::TestJointModelOrdering -v
    """

    def test_joint_ordering_per_chamber(self) -> None:
        """Joint model should sort each chamber's pair independently."""
        # Simulate unsorted group offsets: [House-D, House-R, Senate-D, Senate-R]
        raw = np.array([1.5, -0.5, 0.8, -1.2])
        house_pair = pt.sort(pt.as_tensor_variable(raw[:2])).eval()
        senate_pair = pt.sort(pt.as_tensor_variable(raw[2:])).eval()
        assert house_pair[0] < house_pair[1], "House pair should be sorted D < R"
        assert senate_pair[0] < senate_pair[1], "Senate pair should be sorted D < R"

    def test_joint_ordering_preserves_already_sorted(self) -> None:
        """Already-sorted pairs should pass through unchanged."""
        raw = np.array([-1.0, 2.0, -0.5, 1.5])
        house_pair = pt.sort(pt.as_tensor_variable(raw[:2])).eval()
        senate_pair = pt.sort(pt.as_tensor_variable(raw[2:])).eval()
        np.testing.assert_array_almost_equal(house_pair, raw[:2])
        np.testing.assert_array_almost_equal(senate_pair, raw[2:])


# ── TestShrinkageRescalingFallback ─────────────────────────────────────────


class TestShrinkageRescalingFallback:
    """Test shrinkage rescaling when too few legislators match.

    Run: uv run pytest tests/test_hierarchical.py::TestShrinkageRescalingFallback -v
    """

    def test_fallback_with_two_matches(
        self, legislators: pl.DataFrame, capsys: pytest.CaptureFixture
    ) -> None:
        """With <= 2 matched legislators, should fall back to slope=1.0 with warning."""
        idata = _make_fake_idata()
        data = {
            "leg_slugs": [f"rep_{chr(97 + i)}_{chr(97 + i)}_1" for i in range(8)],
            "party_idx": np.array([1, 1, 1, 1, 1, 0, 0, 0]),
            "party_names": PARTY_NAMES,
        }
        # Flat IPs with only 2 matching slugs
        flat_ip = pl.DataFrame(
            {
                "legislator_slug": ["rep_a_a_1", "rep_b_b_1"],
                "xi_mean": [1.5, 1.2],
                "xi_sd": [0.15, 0.12],
            }
        )
        df = extract_hierarchical_ideal_points(idata, data, legislators, flat_ip=flat_ip)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "slope=1.0" in captured.out
        # Rescaled should equal raw flat values (identity transform)
        matched = df.filter(pl.col("flat_xi_rescaled").is_not_null())
        for row in matched.iter_rows(named=True):
            assert row["flat_xi_rescaled"] == row["flat_xi_mean"]


# ── TestUnequalGroupsICC ──────────────────────────────────────────────────


class TestUnequalGroupsICC:
    """Test ICC with highly unequal group sizes (closer to real Kansas data).

    Run: uv run pytest tests/test_hierarchical.py::TestUnequalGroupsICC -v
    """

    def test_highly_unequal_groups(self) -> None:
        """ICC should still be valid with 20R / 3D (Kansas-like proportions)."""
        n_r, n_d = 20, 3
        n_legislators = n_r + n_d
        n_chains, n_draws, n_parties = 2, 100, 2

        # Separated means
        mu_values = np.zeros((n_chains, n_draws, n_parties))
        mu_values[:, :, 0] = -2.0  # D
        mu_values[:, :, 1] = 1.0  # R
        sigma_values = np.ones((n_chains, n_draws, n_parties)) * 0.5

        idata = _make_fake_idata(
            n_legislators=n_legislators,
            mu_party_values=mu_values,
            sigma_within_values=sigma_values,
        )
        party_idx = np.array([1] * n_r + [0] * n_d)
        data = {"party_idx": party_idx, "party_names": PARTY_NAMES}

        df = compute_variance_decomposition(idata, data)
        icc = float(df["icc_mean"][0])
        assert 0 <= icc <= 1, f"ICC out of bounds: {icc}"
        # With separated means and small sigma, ICC should be high
        assert icc > 0.5, f"ICC unexpectedly low for separated groups: {icc}"


# ── TestExtractGroupParamsBoundary ──────────────────────────────────────────


class TestExtractGroupParamsBoundary:
    """Test extract_group_params boundary conditions.

    Run: uv run pytest tests/test_hierarchical.py::TestExtractGroupParamsBoundary -v
    """

    def test_joint_model_data_raises(self) -> None:
        """extract_group_params should raise ValueError for joint model InferenceData."""
        import arviz as az

        # Create idata with mu_group instead of mu_party
        n_chains, n_draws, n_groups = 2, 100, 4
        mu_group_values = np.random.default_rng(42).normal(0, 1, (n_chains, n_draws, n_groups))
        sigma_values = np.abs(
            np.random.default_rng(42).normal(0.5, 0.1, (n_chains, n_draws, n_groups))
        )

        posterior = xr.Dataset(
            {
                "mu_group": xr.DataArray(
                    mu_group_values,
                    dims=["chain", "draw", "group"],
                    coords={"group": ["House-D", "House-R", "Senate-D", "Senate-R"]},
                ),
                "sigma_within": xr.DataArray(
                    sigma_values,
                    dims=["chain", "draw", "group"],
                    coords={"group": ["House-D", "House-R", "Senate-D", "Senate-R"]},
                ),
            }
        )
        idata = az.InferenceData(posterior=posterior)
        data = {"party_idx": np.array([1, 1, 0, 0]), "party_names": PARTY_NAMES}

        with pytest.raises(ValueError, match="per-chamber models"):
            extract_group_params(idata, data)


# ── TestIndependentExclusion ───────────────────────────────────────────────


class TestIndependentExclusion:
    """Test that Independent legislators are correctly excluded.

    Run: uv run pytest tests/test_hierarchical.py::TestIndependentExclusion -v
    """

    def test_independent_excluded(self) -> None:
        """Independent legislators should be excluded from hierarchical data."""
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["rep_r_1", "rep_r_2", "rep_d_1", "rep_i_1"],
                "v1": [1, 1, 0, 1],
                "v2": [1, 0, 0, 1],
            }
        )
        legislators = pl.DataFrame(
            {
                "name": ["R1", "R2", "D1", "I1"],
                "full_name": ["R R1", "R R2", "D D1", "I I1"],
                "legislator_slug": ["rep_r_1", "rep_r_2", "rep_d_1", "rep_i_1"],
                "chamber": ["House"] * 4,
                "party": ["Republican", "Republican", "Democrat", "Independent"],
                "district": [1, 2, 3, 4],
                "member_url": [""] * 4,
            }
        )
        data = prepare_hierarchical_data(matrix, legislators, "House")
        # Independent should be excluded
        assert data["n_excluded"] == 1
        assert "rep_i_1" not in data["leg_slugs"]
        assert data["n_legislators"] == 3

    def test_no_independents_no_exclusion(
        self, house_matrix: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """When no Independents present, n_excluded should be 0."""
        data = prepare_hierarchical_data(house_matrix, legislators, "House")
        assert data["n_excluded"] == 0


# -- Sign Convention Fix -------------------------------------------------------


class TestFixJointSignConvention:
    """Tests for fix_joint_sign_convention — post-hoc sign correction for joint model."""

    def _make_joint_idata(
        self, xi_house: list[float], xi_senate: list[float], mu_group: list[float]
    ) -> az.InferenceData:
        """Build a minimal InferenceData with xi and mu_group."""
        xi = np.array([[xi_house + xi_senate]])  # (1 chain, 1 draw, N legislators)
        mg = np.array([[mu_group]])  # (1 chain, 1 draw, 4 groups)
        ds = xr.Dataset(
            {
                "xi": xr.DataArray(xi, dims=["chain", "draw", "leg"]),
                "mu_group": xr.DataArray(mg, dims=["chain", "draw", "group"]),
            }
        )
        return az.InferenceData(posterior=ds)

    def _make_per_chamber_results(
        self,
        house_slugs: list[str],
        house_xi: list[float],
        senate_slugs: list[str],
        senate_xi: list[float],
    ) -> dict:
        """Build per-chamber results dict with ideal_points DataFrames."""
        return {
            "House": {
                "ideal_points": pl.DataFrame(
                    {
                        "legislator_slug": house_slugs,
                        "xi_mean": house_xi,
                    }
                )
            },
            "Senate": {
                "ideal_points": pl.DataFrame(
                    {
                        "legislator_slug": senate_slugs,
                        "xi_mean": senate_xi,
                    }
                )
            },
        }

    def test_no_flip_needed(self) -> None:
        """When joint and per-chamber agree, no correction applied."""
        house_slugs = ["rep_d_1", "rep_d_2", "rep_r_1", "rep_r_2"]
        senate_slugs = ["sen_d_1", "sen_r_1", "sen_r_2"]

        idata = self._make_joint_idata(
            xi_house=[-3.0, -2.0, 4.0, 5.0],
            xi_senate=[-6.0, 3.0, 4.0],
            mu_group=[-2.5, 4.5, -6.0, 3.5],
        )
        combined_data = {
            "n_house": 4,
            "leg_slugs": house_slugs + senate_slugs,
            "group_names": [
                "House Democrat",
                "House Republican",
                "Senate Democrat",
                "Senate Republican",
            ],
        }
        per_chamber = self._make_per_chamber_results(
            house_slugs,
            [-3.0, -2.0, 4.0, 5.0],
            senate_slugs,
            [-6.0, 3.0, 4.0],
        )

        result_idata, flipped = fix_joint_sign_convention(idata, combined_data, per_chamber)
        assert flipped == []
        # xi unchanged
        xi = result_idata.posterior["xi"].values[0, 0]
        np.testing.assert_array_almost_equal(xi[4:], [-6.0, 3.0, 4.0])

    def test_senate_sign_flip(self) -> None:
        """When Senate xi is flipped, correction negates Senate xi."""
        house_slugs = ["rep_d_1", "rep_d_2", "rep_r_1", "rep_r_2"]
        senate_slugs = ["sen_d_1", "sen_d_2", "sen_r_1", "sen_r_2"]

        # Joint has Senate flipped: D positive, R negative
        idata = self._make_joint_idata(
            xi_house=[-3.0, -2.0, 4.0, 5.0],
            xi_senate=[6.0, 7.0, -4.0, -3.0],  # flipped!
            mu_group=[-2.5, 4.5, -3.5, -3.0],
        )
        combined_data = {
            "n_house": 4,
            "leg_slugs": house_slugs + senate_slugs,
            "group_names": [
                "House Democrat",
                "House Republican",
                "Senate Democrat",
                "Senate Republican",
            ],
        }
        # Per-chamber has correct sign: D negative, R positive
        per_chamber = self._make_per_chamber_results(
            house_slugs,
            [-3.0, -2.0, 4.0, 5.0],
            senate_slugs,
            [-6.0, -7.0, 4.0, 3.0],
        )

        result_idata, flipped = fix_joint_sign_convention(idata, combined_data, per_chamber)
        assert flipped == ["Senate"]
        xi = result_idata.posterior["xi"].values[0, 0]
        # House unchanged
        np.testing.assert_array_almost_equal(xi[:4], [-3.0, -2.0, 4.0, 5.0])
        # Senate negated
        np.testing.assert_array_almost_equal(xi[4:], [-6.0, -7.0, 4.0, 3.0])

    def test_flipped_chambers_returned(self) -> None:
        """Flipped chambers list is returned for use in extract function."""
        house_slugs = ["rep_d_1", "rep_d_2", "rep_r_1", "rep_r_2"]
        senate_slugs = ["sen_d_1", "sen_d_2", "sen_r_1", "sen_r_2"]

        # Both chambers flipped
        idata = self._make_joint_idata(
            xi_house=[3.0, 2.0, -4.0, -5.0],  # house flipped
            xi_senate=[6.0, 7.0, -4.0, -3.0],  # senate flipped
            mu_group=[2.5, -4.5, 6.5, -3.5],
        )
        combined_data = {
            "n_house": 4,
            "leg_slugs": house_slugs + senate_slugs,
            "group_names": [
                "House Democrat",
                "House Republican",
                "Senate Democrat",
                "Senate Republican",
            ],
        }
        per_chamber = self._make_per_chamber_results(
            house_slugs,
            [-3.0, -2.0, 4.0, 5.0],
            senate_slugs,
            [-6.0, -7.0, 4.0, 3.0],
        )

        _, flipped = fix_joint_sign_convention(idata, combined_data, per_chamber)
        assert sorted(flipped) == ["House", "Senate"]

    def test_extract_uses_empirical_means_for_flipped(self) -> None:
        """party_mean uses empirical xi for flipped chambers, mu_group for non-flipped."""
        n_house = 3
        n_senate = 3
        xi = np.array([[[-3.0, -2.0, 5.0, -7.0, -6.0, 4.0]]])  # correct sign
        mu_group = np.array([[[-2.5, 5.0, -6.0, 3.5]]])  # correct for house, wrong for senate
        ds = xr.Dataset(
            {
                "xi": xr.DataArray(xi, dims=["chain", "draw", "leg"]),
                "mu_group": xr.DataArray(mu_group, dims=["chain", "draw", "group"]),
            }
        )
        idata = az.InferenceData(posterior=ds)

        slugs = ["rep_d_1", "rep_d_2", "rep_r_1", "sen_d_1", "sen_d_2", "sen_r_1"]
        group_idx = np.array([0, 0, 1, 2, 2, 3])

        data = {
            "leg_slugs": slugs,
            "group_idx": group_idx,
            "group_names": [
                "House Democrat",
                "House Republican",
                "Senate Democrat",
                "Senate Republican",
            ],
        }
        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": [f"Leg {i}" for i in range(6)],
                "party": [
                    "Democrat",
                    "Democrat",
                    "Republican",
                    "Democrat",
                    "Democrat",
                    "Republican",
                ],
                "district": list(range(1, 7)),
                "chamber": ["House"] * n_house + ["Senate"] * n_senate,
            }
        )

        result = extract_hierarchical_ideal_points(
            idata, data, legislators, flipped_chambers=["Senate"]
        )

        # House party_mean should come from mu_group
        house_d = result.filter((pl.col("chamber") == "House") & (pl.col("party") == "Democrat"))
        assert house_d["party_mean"][0] == pytest.approx(-2.5)

        # Senate party_mean should be empirical mean of corrected xi
        senate_d = result.filter((pl.col("chamber") == "Senate") & (pl.col("party") == "Democrat"))
        expected_senate_d_mean = (-7.0 + -6.0) / 2
        assert senate_d["party_mean"][0] == pytest.approx(expected_senate_d_mean)

        senate_r = result.filter(
            (pl.col("chamber") == "Senate") & (pl.col("party") == "Republican")
        )
        assert senate_r["party_mean"][0] == pytest.approx(4.0)


# ── TestBillMatchingAcrossChambers ──────────────────────────────────────────


class TestBillMatchingAcrossChambers:
    """Tests for _match_bills_across_chambers — shared bill identification.

    Run: uv run pytest tests/test_hierarchical.py::TestBillMatchingAcrossChambers -v
    """

    @pytest.fixture
    def rollcalls(self) -> pl.DataFrame:
        """Rollcalls with bills appearing in both chambers."""
        return pl.DataFrame(
            {
                "vote_id": [
                    "h_v1",
                    "h_v2",
                    "h_v3",
                    "h_v4",
                    "s_v1",
                    "s_v2",
                    "s_v3",
                ],
                "bill_number": [
                    "HB 2001",
                    "HB 2001",
                    "SB 100",
                    "HB 2002",
                    "HB 2001",
                    "SB 100",
                    "SB 200",
                ],
                "motion": [
                    "Committee Vote",
                    "Final Action",
                    "Emergency Final Action",
                    "Final Action",
                    "Final Action",
                    "Final Action",
                    "Final Action",
                ],
            }
        )

    def test_shared_bill_count(self, rollcalls: pl.DataFrame) -> None:
        """Should find 2 shared bills (HB 2001 and SB 100)."""
        house_vids = ["h_v1", "h_v2", "h_v3"]
        senate_vids = ["s_v1", "s_v2", "s_v3"]
        matched, house_only, senate_only = _match_bills_across_chambers(
            house_vids, senate_vids, rollcalls
        )
        assert len(matched) == 2
        bill_numbers = {m["bill_number"] for m in matched}
        assert bill_numbers == {"HB 2001", "SB 100"}

    def test_prefers_final_action(self, rollcalls: pl.DataFrame) -> None:
        """For HB 2001 in House, should pick h_v2 (Final Action) over h_v1 (Committee Vote)."""
        house_vids = ["h_v1", "h_v2", "h_v3"]
        senate_vids = ["s_v1", "s_v2", "s_v3"]
        matched, _, _ = _match_bills_across_chambers(house_vids, senate_vids, rollcalls)
        hb2001 = next(m for m in matched if m["bill_number"] == "HB 2001")
        assert hb2001["house_vote_id"] == "h_v2"  # Final Action
        assert hb2001["senate_vote_id"] == "s_v1"  # Final Action

    def test_house_only_and_senate_only(self, rollcalls: pl.DataFrame) -> None:
        """Unmatched vote_ids should appear in house_only / senate_only lists."""
        house_vids = ["h_v1", "h_v2", "h_v3", "h_v4"]
        senate_vids = ["s_v1", "s_v2", "s_v3"]
        matched, house_only, senate_only = _match_bills_across_chambers(
            house_vids, senate_vids, rollcalls
        )
        # h_v4 is HB 2002 (house-only), h_v1 is the non-preferred dup for HB 2001
        assert "h_v4" in house_only
        # s_v3 is SB 200 (senate-only)
        assert "s_v3" in senate_only

    def test_no_shared_bills(self) -> None:
        """When no bills overlap, matched should be empty."""
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["h_v1", "s_v1"],
                "bill_number": ["HB 100", "SB 200"],
                "motion": ["Final Action", "Final Action"],
            }
        )
        matched, house_only, senate_only = _match_bills_across_chambers(
            ["h_v1"], ["s_v1"], rollcalls
        )
        assert len(matched) == 0
        assert house_only == ["h_v1"]
        assert senate_only == ["s_v1"]

    def test_emergency_final_action_preferred(self) -> None:
        """Emergency Final Action should be treated same as Final Action."""
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["h_v1", "h_v2", "s_v1"],
                "bill_number": ["HB 1", "HB 1", "HB 1"],
                "motion": ["Committee Vote", "Emergency Final Action", "Final Action"],
            }
        )
        matched, _, _ = _match_bills_across_chambers(["h_v1", "h_v2"], ["s_v1"], rollcalls)
        assert len(matched) == 1
        assert matched[0]["house_vote_id"] == "h_v2"  # Emergency Final Action preferred


# ── TestGroupSizeAdaptiveSigma ──────────────────────────────────────────────


class TestGroupSizeAdaptiveSigma:
    """Tests for group-size-adaptive sigma_within priors.

    Run: uv run pytest tests/test_hierarchical.py::TestGroupSizeAdaptiveSigma -v
    """

    def test_constants_defined(self) -> None:
        """SMALL_GROUP_THRESHOLD and SMALL_GROUP_SIGMA_SCALE should be defined."""
        assert SMALL_GROUP_THRESHOLD == 20
        assert SMALL_GROUP_SIGMA_SCALE == 0.5

    def test_small_group_gets_tighter_prior(self) -> None:
        """Groups below SMALL_GROUP_THRESHOLD get sigma=0.5, others get 1.0."""
        party_counts = np.array([10, 80])  # 10 Democrats, 80 Republicans
        sigma_scale = np.array(
            [
                SMALL_GROUP_SIGMA_SCALE if party_counts[p] < SMALL_GROUP_THRESHOLD else 1.0
                for p in range(2)
            ]
        )
        assert sigma_scale[0] == SMALL_GROUP_SIGMA_SCALE  # Small group → tighter
        assert sigma_scale[1] == 1.0  # Large group → standard

    def test_all_large_groups_standard_prior(self) -> None:
        """When all groups are large enough, all get sigma=1.0."""
        party_counts = np.array([25, 80])
        sigma_scale = np.array(
            [
                SMALL_GROUP_SIGMA_SCALE if party_counts[p] < SMALL_GROUP_THRESHOLD else 1.0
                for p in range(2)
            ]
        )
        assert sigma_scale[0] == 1.0
        assert sigma_scale[1] == 1.0

    def test_joint_model_four_groups(self) -> None:
        """Joint model with 4 groups: only small groups get adaptive prior."""
        # Simulating: House-D=30, House-R=95, Senate-D=11, Senate-R=29
        group_counts = np.array([30, 95, 11, 29])
        sigma_scale = np.array(
            [
                SMALL_GROUP_SIGMA_SCALE if group_counts[g] < SMALL_GROUP_THRESHOLD else 1.0
                for g in range(4)
            ]
        )
        assert sigma_scale[0] == 1.0  # House-D (30) → standard
        assert sigma_scale[1] == 1.0  # House-R (95) → standard
        assert sigma_scale[2] == SMALL_GROUP_SIGMA_SCALE  # Senate-D (11) → tighter
        assert sigma_scale[3] == 1.0  # Senate-R (29) → standard
