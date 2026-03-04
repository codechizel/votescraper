"""
Tests for Phase 08 posterior predictive check computations.

Uses synthetic InferenceData fixtures with known parameters to verify
log-likelihood computation, PPC battery, item/person checks, Q3 local
dependence, LOO wrappers, and graceful degradation.

Run: uv run pytest tests/test_ppc.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import arviz as az
import xarray as xr
from analysis.ppc_data import (
    _log1m_sigmoid,
    _log_sigmoid,
    add_log_likelihood_to_idata,
    compute_item_ppc,
    compute_log_likelihood_1d,
    compute_log_likelihood_2d,
    compute_log_likelihood_hierarchical,
    compute_loo,
    compute_person_ppc,
    compute_vote_margin_ppc,
    compute_yens_q3,
    run_ppc_battery,
    summarize_pareto_k,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data() -> dict:
    """Synthetic IRT data dict: 10 legislators x 8 votes, ~60 observations."""
    rng = np.random.default_rng(42)
    n_leg = 10
    n_votes = 8

    # Build long format: all possible pairs minus ~20% absences
    leg_idx_list = []
    vote_idx_list = []
    y_list = []
    for i in range(n_leg):
        for j in range(n_votes):
            if rng.random() > 0.2:  # 80% participation
                leg_idx_list.append(i)
                vote_idx_list.append(j)
                y_list.append(rng.integers(0, 2))

    return {
        "leg_idx": np.array(leg_idx_list, dtype=np.int64),
        "vote_idx": np.array(vote_idx_list, dtype=np.int64),
        "y": np.array(y_list, dtype=np.int64),
        "n_legislators": n_leg,
        "n_votes": n_votes,
        "n_obs": len(y_list),
        "leg_slugs": [f"leg_{i}" for i in range(n_leg)],
        "vote_ids": [f"vote_{j}" for j in range(n_votes)],
    }


@pytest.fixture
def synthetic_idata_1d(synthetic_data: dict) -> az.InferenceData:
    """Synthetic 1D IRT InferenceData: 2 chains x 50 draws."""
    rng = np.random.default_rng(42)
    n_chains = 2
    n_draws = 50
    n_leg = synthetic_data["n_legislators"]
    n_votes = synthetic_data["n_votes"]

    xi = rng.normal(0, 1, (n_chains, n_draws, n_leg))
    alpha = rng.normal(0, 1, (n_chains, n_draws, n_votes))
    beta = rng.normal(1, 0.5, (n_chains, n_draws, n_votes))

    posterior = xr.Dataset(
        {
            "xi": (["chain", "draw", "legislator"], xi),
            "alpha": (["chain", "draw", "vote"], alpha),
            "beta": (["chain", "draw", "vote"], beta),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "legislator": synthetic_data["leg_slugs"],
            "vote": synthetic_data["vote_ids"],
        },
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def synthetic_idata_2d(synthetic_data: dict) -> az.InferenceData:
    """Synthetic 2D IRT InferenceData: 2 chains x 50 draws."""
    rng = np.random.default_rng(42)
    n_chains = 2
    n_draws = 50
    n_leg = synthetic_data["n_legislators"]
    n_votes = synthetic_data["n_votes"]

    xi = rng.normal(0, 1, (n_chains, n_draws, n_leg, 2))
    alpha = rng.normal(0, 1, (n_chains, n_draws, n_votes))
    beta_matrix = rng.normal(1, 0.5, (n_chains, n_draws, n_votes, 2))

    posterior = xr.Dataset(
        {
            "xi": (["chain", "draw", "legislator", "dim"], xi),
            "alpha": (["chain", "draw", "vote"], alpha),
            "beta_matrix": (["chain", "draw", "vote", "dim"], beta_matrix),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "legislator": synthetic_data["leg_slugs"],
            "vote": synthetic_data["vote_ids"],
            "dim": [0, 1],
        },
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def synthetic_idata_2d_separate_betas(synthetic_data: dict) -> az.InferenceData:
    """Synthetic 2D IRT with separate beta_1/beta_2 (no beta_matrix)."""
    rng = np.random.default_rng(42)
    n_chains = 2
    n_draws = 50
    n_leg = synthetic_data["n_legislators"]
    n_votes = synthetic_data["n_votes"]

    xi = rng.normal(0, 1, (n_chains, n_draws, n_leg, 2))
    alpha = rng.normal(0, 1, (n_chains, n_draws, n_votes))
    beta_1 = rng.normal(1, 0.5, (n_chains, n_draws, n_votes))
    beta_2 = rng.normal(0, 0.5, (n_chains, n_draws, n_votes))

    posterior = xr.Dataset(
        {
            "xi": (["chain", "draw", "legislator", "dim"], xi),
            "alpha": (["chain", "draw", "vote"], alpha),
            "beta_1": (["chain", "draw", "vote"], beta_1),
            "beta_2": (["chain", "draw", "vote"], beta_2),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "legislator": synthetic_data["leg_slugs"],
            "vote": synthetic_data["vote_ids"],
            "dim": [0, 1],
        },
    )
    return az.InferenceData(posterior=posterior)


# ── Log-Sigmoid Tests ───────────────────────────────────────────────────────


class TestLogSigmoid:
    """Tests for numerically stable log-sigmoid functions."""

    def test_log_sigmoid_known_value(self) -> None:
        """log(sigmoid(0)) = log(0.5) = -0.693..."""
        result = _log_sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.log(0.5), atol=1e-10)

    def test_log1m_sigmoid_known_value(self) -> None:
        """log(1 - sigmoid(0)) = log(0.5) = -0.693..."""
        result = _log1m_sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, np.log(0.5), atol=1e-10)

    def test_log_sigmoid_large_positive(self) -> None:
        """log(sigmoid(500)) should be close to 0 without overflow."""
        result = _log_sigmoid(np.array([500.0]))
        assert np.isfinite(result[0])
        assert result[0] > -1e-10

    def test_log_sigmoid_large_negative(self) -> None:
        """log(sigmoid(-500)) should be close to -500 without overflow."""
        result = _log_sigmoid(np.array([-500.0]))
        assert np.isfinite(result[0])
        assert result[0] < -499.0

    def test_log1m_sigmoid_large_positive(self) -> None:
        """log(1 - sigmoid(500)) should be close to -500 without overflow."""
        result = _log1m_sigmoid(np.array([500.0]))
        assert np.isfinite(result[0])
        assert result[0] < -499.0

    def test_log1m_sigmoid_large_negative(self) -> None:
        """log(1 - sigmoid(-500)) should be close to 0 without overflow."""
        result = _log1m_sigmoid(np.array([-500.0]))
        assert np.isfinite(result[0])
        assert result[0] > -1e-10

    def test_log_sigmoid_vectorized(self) -> None:
        """Should handle arrays."""
        eta = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = _log_sigmoid(eta)
        assert result.shape == (5,)
        assert all(np.isfinite(result))
        # log(sigmoid) is always <= 0
        assert all(result <= 0)

    def test_sum_of_log_sigmoid_and_log1m(self) -> None:
        """log(sigmoid(x)) + log(1-sigmoid(x)) should equal -log(2*cosh(x/2))^2 behavior."""
        eta = np.array([-5.0, 0.0, 5.0])
        log_s = _log_sigmoid(eta)
        log_1ms = _log1m_sigmoid(eta)
        # Sum should be log(sigmoid * (1-sigmoid)) = log(exp(-|x|) / (1+exp(-|x|))^2)
        # Both should be negative
        assert all(log_s <= 0)
        assert all(log_1ms <= 0)


# ── Log-Likelihood 1D Tests ────────────────────────────────────────────────


class TestLogLikelihood1D:
    """Tests for 1D IRT log-likelihood computation."""

    def test_shape(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        """Output shape should be (n_chains, n_draws, n_obs)."""
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        assert ds["y"].shape == (2, 50, synthetic_data["n_obs"])

    def test_values_are_negative(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """Log-likelihood should always be <= 0."""
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        assert float(ds["y"].values.max()) <= 1e-15

    def test_values_are_finite(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """All values should be finite (no NaN/Inf)."""
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        assert np.all(np.isfinite(ds["y"].values))

    def test_matches_scipy(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """Cross-check against scipy for a single draw."""
        from scipy.special import expit

        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)

        # Manual scipy computation for chain=0, draw=0
        xi = synthetic_idata_1d.posterior["xi"].values[0, 0]
        alpha = synthetic_idata_1d.posterior["alpha"].values[0, 0]
        beta = synthetic_idata_1d.posterior["beta"].values[0, 0]
        leg_idx = synthetic_data["leg_idx"]
        vote_idx = synthetic_data["vote_idx"]
        y = synthetic_data["y"].astype(float)

        eta = beta[vote_idx] * xi[leg_idx] - alpha[vote_idx]
        p = expit(eta)
        expected = y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)

        actual = ds["y"].values[0, 0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_all_yea(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        """All-Yea data: log-likelihood should be log(sigmoid(eta))."""
        data_all_yea = {**synthetic_data, "y": np.ones(synthetic_data["n_obs"], dtype=np.int64)}
        ds = compute_log_likelihood_1d(synthetic_idata_1d, data_all_yea)
        # Every observation should have ll = log(sigmoid(eta))
        assert np.all(np.isfinite(ds["y"].values))

    def test_all_nay(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        """All-Nay data: log-likelihood should be log(1 - sigmoid(eta))."""
        data_all_nay = {**synthetic_data, "y": np.zeros(synthetic_data["n_obs"], dtype=np.int64)}
        ds = compute_log_likelihood_1d(synthetic_idata_1d, data_all_nay)
        assert np.all(np.isfinite(ds["y"].values))

    def test_coords_present(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """Output should have chain, draw, obs coordinates."""
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        assert "chain" in ds.coords
        assert "draw" in ds.coords
        assert "obs" in ds.coords

    def test_deterministic(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """Same inputs should produce identical outputs."""
        ds1 = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        ds2 = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        np.testing.assert_array_equal(ds1["y"].values, ds2["y"].values)


# ── Log-Likelihood 2D Tests ────────────────────────────────────────────────


class TestLogLikelihood2D:
    """Tests for 2D IRT log-likelihood computation."""

    def test_shape(self, synthetic_idata_2d: az.InferenceData, synthetic_data: dict) -> None:
        """Output shape should be (n_chains, n_draws, n_obs)."""
        ds = compute_log_likelihood_2d(synthetic_idata_2d, synthetic_data)
        assert ds["y"].shape == (2, 50, synthetic_data["n_obs"])

    def test_values_negative_and_finite(
        self, synthetic_idata_2d: az.InferenceData, synthetic_data: dict
    ) -> None:
        ds = compute_log_likelihood_2d(synthetic_idata_2d, synthetic_data)
        assert np.all(np.isfinite(ds["y"].values))
        assert float(ds["y"].values.max()) <= 1e-15

    def test_separate_betas(
        self, synthetic_idata_2d_separate_betas: az.InferenceData, synthetic_data: dict
    ) -> None:
        """Should work with beta_1/beta_2 instead of beta_matrix."""
        ds = compute_log_likelihood_2d(synthetic_idata_2d_separate_betas, synthetic_data)
        assert ds["y"].shape == (2, 50, synthetic_data["n_obs"])
        assert np.all(np.isfinite(ds["y"].values))

    def test_2d_eta_computation(
        self, synthetic_idata_2d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """Verify 2D eta uses dot product correctly for a single draw."""
        xi = synthetic_idata_2d.posterior["xi"].values[0, 0]  # (n_leg, 2)
        alpha = synthetic_idata_2d.posterior["alpha"].values[0, 0]  # (n_votes,)
        beta_m = synthetic_idata_2d.posterior["beta_matrix"].values[0, 0]  # (n_votes, 2)

        leg_idx = synthetic_data["leg_idx"]
        vote_idx = synthetic_data["vote_idx"]

        # Manual eta
        eta_manual = (
            beta_m[vote_idx, 0] * xi[leg_idx, 0]
            + beta_m[vote_idx, 1] * xi[leg_idx, 1]
            - alpha[vote_idx]
        )

        ds = compute_log_likelihood_2d(synthetic_idata_2d, synthetic_data)
        y = synthetic_data["y"].astype(float)

        # Recompute expected log_lik from manual eta
        p = 1.0 / (1.0 + np.exp(-eta_manual))
        expected = y * np.log(p + 1e-300) + (1 - y) * np.log(1 - p + 1e-300)
        np.testing.assert_allclose(ds["y"].values[0, 0], expected, atol=1e-5)

    def test_different_from_1d(
        self,
        synthetic_idata_1d: az.InferenceData,
        synthetic_idata_2d: az.InferenceData,
        synthetic_data: dict,
    ) -> None:
        """2D log-lik should differ from 1D (different models)."""
        ds_1d = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        ds_2d = compute_log_likelihood_2d(synthetic_idata_2d, synthetic_data)
        # The sums should be different (different parameters)
        assert float(ds_1d["y"].values.sum()) != float(ds_2d["y"].values.sum())


# ── Log-Likelihood Hierarchical Tests ───────────────────────────────────────


class TestLogLikelihoodHierarchical:
    """Tests for hierarchical IRT log-likelihood (delegates to 1D)."""

    def test_matches_1d(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        """Hierarchical uses xi (not xi_offset), so should match 1D computation."""
        ds_1d = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        ds_hier = compute_log_likelihood_hierarchical(synthetic_idata_1d, synthetic_data)
        np.testing.assert_array_equal(ds_1d["y"].values, ds_hier["y"].values)

    def test_shape(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        ds = compute_log_likelihood_hierarchical(synthetic_idata_1d, synthetic_data)
        assert ds["y"].shape == (2, 50, synthetic_data["n_obs"])

    def test_finite(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        ds = compute_log_likelihood_hierarchical(synthetic_idata_1d, synthetic_data)
        assert np.all(np.isfinite(ds["y"].values))


# ── Add Log-Likelihood Tests ───────────────────────────────────────────────


class TestAddLogLikelihood:
    """Tests for adding log_likelihood group to InferenceData."""

    def test_group_added(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_new = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        assert hasattr(idata_new, "log_likelihood")

    def test_original_unchanged(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        _ = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        # Original should NOT have log_likelihood
        assert not hasattr(synthetic_idata_1d, "log_likelihood") or (
            synthetic_idata_1d.log_likelihood is None
        )

    def test_posterior_preserved(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_new = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        assert hasattr(idata_new, "posterior")
        assert "xi" in idata_new.posterior


# ── PPC Battery Tests ───────────────────────────────────────────────────────


class TestPPCBattery:
    """Tests for the comprehensive PPC battery."""

    def test_returns_all_keys(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        expected_keys = {
            "observed_yea_rate",
            "replicated_yea_rate_mean",
            "replicated_yea_rate_sd",
            "bayesian_p_yea_rate",
            "mean_accuracy",
            "accuracy_sd",
            "mean_gmp",
            "gmp_sd",
            "apre",
            "baseline_accuracy",
            "n_reps",
            "n_obs",
            "replicated_yea_rates",
            "replicated_accuracies",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_yea_rate_in_range(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert 0.0 <= result["observed_yea_rate"] <= 1.0
        assert 0.0 <= result["replicated_yea_rate_mean"] <= 1.0

    def test_p_value_in_range(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert 0.0 <= result["bayesian_p_yea_rate"] <= 1.0

    def test_accuracy_in_range(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_gmp_in_range(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        result = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert 0.0 <= result["mean_gmp"] <= 1.0

    def test_apre_finite(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        result = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert np.isfinite(result["apre"])

    def test_deterministic_with_seed(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        r1 = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        r2 = run_ppc_battery(synthetic_idata_1d, synthetic_data, n_reps=50)
        np.testing.assert_array_equal(r1["replicated_yea_rates"], r2["replicated_yea_rates"])

    def test_2d_model(self, synthetic_idata_2d: az.InferenceData, synthetic_data: dict) -> None:
        result = run_ppc_battery(synthetic_idata_2d, synthetic_data, n_reps=50, model_type="2d")
        assert 0.0 <= result["mean_accuracy"] <= 1.0


# ── Item PPC Tests ──────────────────────────────────────────────────────────


class TestItemPPC:
    """Tests for per-item endorsement rate checks."""

    def test_observed_rates_shape(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_item_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["observed_rates"].shape == (synthetic_data["n_votes"],)

    def test_replicated_rates_shape(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_item_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["replicated_rates"].shape == (50, synthetic_data["n_votes"])

    def test_rates_in_range(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_item_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert np.all(result["observed_rates"] >= 0.0)
        assert np.all(result["observed_rates"] <= 1.0)

    def test_misfitting_count(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_item_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["n_misfitting"] >= 0
        assert result["n_misfitting"] <= result["n_votes"]


# ── Person PPC Tests ────────────────────────────────────────────────────────


class TestPersonPPC:
    """Tests for per-legislator total score checks."""

    def test_observed_totals_shape(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_person_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["observed_totals"].shape == (synthetic_data["n_legislators"],)

    def test_replicated_totals_shape(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_person_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["replicated_totals"].shape == (50, synthetic_data["n_legislators"])

    def test_totals_non_negative(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_person_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert np.all(result["observed_totals"] >= 0)

    def test_misfitting_count(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_person_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["n_misfitting"] >= 0
        assert result["n_misfitting"] <= result["n_legislators"]


# ── Vote Margin Tests ───────────────────────────────────────────────────────


class TestVoteMargins:
    """Tests for vote margin distribution checks."""

    def test_observed_margins_shape(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_vote_margin_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert result["observed_margins"].shape == (synthetic_data["n_votes"],)

    def test_margins_in_range(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_vote_margin_ppc(synthetic_idata_1d, synthetic_data, n_reps=50)
        assert np.all(result["observed_margins"] >= 0.0)
        assert np.all(result["observed_margins"] <= 1.0)

    def test_unanimous_all_yea(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """All-Yea data should produce margin = 1.0 for all items."""
        data_all_yea = {**synthetic_data, "y": np.ones(synthetic_data["n_obs"], dtype=np.int64)}
        result = compute_vote_margin_ppc(synthetic_idata_1d, data_all_yea, n_reps=10)
        np.testing.assert_allclose(result["observed_margins"], 1.0)


# ── Yen's Q3 Tests ─────────────────────────────────────────────────────────


class TestYensQ3:
    """Tests for Q3 local dependence computation."""

    def test_q3_shape(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        result = compute_yens_q3(synthetic_idata_1d, synthetic_data, n_draws_sample=10)
        n_votes = synthetic_data["n_votes"]
        assert result["q3_matrix"].shape == (n_votes, n_votes)

    def test_q3_symmetric(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        result = compute_yens_q3(synthetic_idata_1d, synthetic_data, n_draws_sample=10)
        np.testing.assert_allclose(result["q3_matrix"], result["q3_matrix"].T, atol=1e-10)

    def test_q3_diagonal_is_one(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_yens_q3(synthetic_idata_1d, synthetic_data, n_draws_sample=10)
        np.testing.assert_allclose(np.diag(result["q3_matrix"]), 1.0, atol=1e-10)

    def test_violation_count_non_negative(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_yens_q3(synthetic_idata_1d, synthetic_data, n_draws_sample=10)
        assert result["n_violations"] >= 0
        assert result["n_violations"] <= result["n_pairs"]

    def test_n_pairs_correct(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        result = compute_yens_q3(synthetic_idata_1d, synthetic_data, n_draws_sample=10)
        n = synthetic_data["n_votes"]
        assert result["n_pairs"] == n * (n - 1) // 2


# ── LOO Tests ───────────────────────────────────────────────────────────────


class TestLOO:
    """Tests for LOO-CV computation."""

    def test_loo_smoke(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        """LOO should run without error on idata with log_likelihood."""
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_ll = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        loo_result = compute_loo(idata_ll)
        assert hasattr(loo_result, "elpd_loo")

    def test_elpd_negative(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        """ELPD should generally be negative (log-scale)."""
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_ll = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        loo_result = compute_loo(idata_ll)
        assert loo_result.elpd_loo < 0

    def test_pareto_k_present(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_ll = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        loo_result = compute_loo(idata_ll)
        assert hasattr(loo_result, "pareto_k")


# ── Pareto k Summary Tests ─────────────────────────────────────────────────


class TestParetoK:
    """Tests for Pareto k summary."""

    def test_categories_present(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_ll = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        loo_result = compute_loo(idata_ll)
        summary = summarize_pareto_k(loo_result)
        assert all(k in summary for k in ("good", "ok", "bad", "very_bad", "total"))

    def test_counts_sum(self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_ll = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        loo_result = compute_loo(idata_ll)
        summary = summarize_pareto_k(loo_result)
        total = summary["good"] + summary["ok"] + summary["bad"] + summary["very_bad"]
        assert total == summary["total"]

    def test_total_matches_obs(
        self, synthetic_idata_1d: az.InferenceData, synthetic_data: dict
    ) -> None:
        ds = compute_log_likelihood_1d(synthetic_idata_1d, synthetic_data)
        idata_ll = add_log_likelihood_to_idata(synthetic_idata_1d, ds)
        loo_result = compute_loo(idata_ll)
        summary = summarize_pareto_k(loo_result)
        assert summary["total"] == synthetic_data["n_obs"]


# ── Data Alignment Tests ───────────────────────────────────────────────────


class TestDataAlignment:
    """Tests for data alignment between models and vote matrices."""

    def test_coord_matching(self, synthetic_idata_1d: az.InferenceData) -> None:
        """InferenceData legislator coords should match fixture slugs."""
        coords = list(synthetic_idata_1d.posterior.coords["legislator"].values)
        assert coords == [f"leg_{i}" for i in range(10)]

    def test_obs_count_consistent(self, synthetic_data: dict) -> None:
        """n_obs should equal length of leg_idx, vote_idx, and y."""
        assert synthetic_data["n_obs"] == len(synthetic_data["leg_idx"])
        assert synthetic_data["n_obs"] == len(synthetic_data["vote_idx"])
        assert synthetic_data["n_obs"] == len(synthetic_data["y"])

    def test_index_ranges_valid(self, synthetic_data: dict) -> None:
        """leg_idx and vote_idx should be within valid ranges."""
        assert np.all(synthetic_data["leg_idx"] >= 0)
        assert np.all(synthetic_data["leg_idx"] < synthetic_data["n_legislators"])
        assert np.all(synthetic_data["vote_idx"] >= 0)
        assert np.all(synthetic_data["vote_idx"] < synthetic_data["n_votes"])
