"""Tests for Beta-Binomial Bayesian Party Loyalty.

Run:
    uv run pytest tests/test_beta_binomial.py -v
"""

import numpy as np
import polars as pl
from analysis.beta_binomial import (
    FALLBACK_ALPHA,
    FALLBACK_BETA,
    compute_bayesian_loyalty,
    estimate_beta_params,
    tarone_test,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_unity_df(rows: list[dict]) -> pl.DataFrame:
    """Build a unity DataFrame matching the indices phase output schema."""
    return pl.DataFrame(rows)


# ── TestEstimateBetaParams ───────────────────────────────────────────────────


class TestEstimateBetaParams:
    """Tests for method-of-moments Beta prior estimation."""

    def test_high_loyalty_party(self) -> None:
        """A party with high loyalty rates should produce alpha >> beta."""
        y = np.array([90, 85, 95, 88, 92, 87, 91, 93, 89, 94])
        n = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        prior_mean = alpha / (alpha + beta)
        assert 0.85 < prior_mean < 0.95, f"Prior mean {prior_mean} not in expected range"
        assert alpha > beta, "Alpha should exceed beta for high-loyalty party"

    def test_low_variance_produces_high_concentration(self) -> None:
        """Near-identical rates should produce high alpha+beta (tight prior)."""
        y = np.array([90, 91, 90, 89, 90, 91, 90, 90])
        n = np.array([100, 100, 100, 100, 100, 100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        concentration = alpha + beta
        assert concentration > 50, f"Concentration {concentration} too low for tight data"

    def test_high_variance_fallback(self) -> None:
        """Variance exceeding Beta maximum should trigger fallback to (1, 1).

        With ddof=1 (sample variance), two rates [0.0, 1.0] give:
          mu=0.5, var=0.5 (ddof=1), mu*(1-mu)=0.25
          var >= mu*(1-mu) → fallback triggered.
        """
        # Two rates at opposite extremes: sample variance exceeds Beta maximum
        y = np.array([0, 100])
        n = np.array([100, 100])
        alpha, beta = estimate_beta_params(y, n)

        # mu=0.5, var(ddof=1)=0.5, mu*(1-mu)=0.25 → var >= limit → fallback
        assert alpha == FALLBACK_ALPHA
        assert beta == FALLBACK_BETA

    def test_all_identical_rates(self) -> None:
        """All legislators with identical rates should produce tight prior around that rate."""
        y = np.array([80, 80, 80, 80, 80])
        n = np.array([100, 100, 100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        prior_mean = alpha / (alpha + beta)
        assert abs(prior_mean - 0.80) < 0.05, f"Prior mean {prior_mean} should be ~0.80"
        # Tight prior = high concentration
        assert (alpha + beta) > 50

    def test_all_perfect_loyalty(self) -> None:
        """All rates = 1.0 — mu=1 triggers fallback to (1, 1)."""
        y = np.array([100, 100, 100])
        n = np.array([100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        # mu=1.0 → boundary fallback
        assert alpha == FALLBACK_ALPHA
        assert beta == FALLBACK_BETA

    def test_all_zero_loyalty(self) -> None:
        """All rates = 0.0 — mu=0 triggers fallback to (1, 1)."""
        y = np.array([0, 0, 0])
        n = np.array([100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        # mu=0 triggers the mu <= 0 branch → fallback
        assert alpha == FALLBACK_ALPHA
        assert beta == FALLBACK_BETA

    def test_near_zero_loyalty(self) -> None:
        """Near-zero rates should produce small alpha, large beta."""
        y = np.array([1, 2, 1, 3, 2])
        n = np.array([100, 100, 100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        prior_mean = alpha / (alpha + beta)
        assert prior_mean < 0.10, f"Prior mean {prior_mean} should be very low"

    def test_single_legislator_fallback(self) -> None:
        """Only 1 legislator — cannot estimate variance, use fallback."""
        y = np.array([85])
        n = np.array([100])
        alpha, beta = estimate_beta_params(y, n)

        assert alpha == FALLBACK_ALPHA
        assert beta == FALLBACK_BETA

    def test_two_legislators_works(self) -> None:
        """Two legislators should be enough for method of moments."""
        y = np.array([80, 90])
        n = np.array([100, 100])
        alpha, beta = estimate_beta_params(y, n)

        prior_mean = alpha / (alpha + beta)
        assert 0.7 < prior_mean < 0.95

    def test_returns_at_least_half(self) -> None:
        """Alpha and beta should always be at least 0.5."""
        y = np.array([50, 60, 70, 80, 90])
        n = np.array([100, 100, 100, 100, 100])
        alpha, beta = estimate_beta_params(y, n)

        assert alpha >= 0.5
        assert beta >= 0.5


# ── TestComputeBayesianLoyalty ───────────────────────────────────────────────


class TestComputeBayesianLoyalty:
    """Tests for the full posterior computation pipeline."""

    def _make_simple_df(self) -> pl.DataFrame:
        """3 Republican legislators: high-n, mid-n, low-n."""
        return _make_unity_df(
            [
                {
                    "legislator_slug": "rep_high",
                    "party": "Republican",
                    "full_name": "High N",
                    "district": "1",
                    "votes_with_party": 180,
                    "party_votes_present": 200,
                    "unity_score": 0.900,
                    "maverick_rate": 0.100,
                },
                {
                    "legislator_slug": "rep_mid",
                    "party": "Republican",
                    "full_name": "Mid N",
                    "district": "2",
                    "votes_with_party": 45,
                    "party_votes_present": 50,
                    "unity_score": 0.900,
                    "maverick_rate": 0.100,
                },
                {
                    "legislator_slug": "rep_low",
                    "party": "Republican",
                    "full_name": "Low N",
                    "district": "3",
                    "votes_with_party": 9,
                    "party_votes_present": 10,
                    "unity_score": 0.900,
                    "maverick_rate": 0.100,
                },
            ]
        )

    def test_basic_output_columns(self) -> None:
        """Output should have all expected columns."""
        df = self._make_simple_df()
        result = compute_bayesian_loyalty(df, "House")

        expected_cols = {
            "legislator_slug",
            "party",
            "full_name",
            "district",
            "raw_loyalty",
            "posterior_mean",
            "posterior_median",
            "ci_lower",
            "ci_upper",
            "ci_width",
            "shrinkage",
            "votes_with_party",
            "n_party_votes",
            "alpha_prior",
            "beta_prior",
            "prior_mean",
            "prior_kappa",
        }
        assert set(result.columns) == expected_cols

    def test_correct_row_count(self) -> None:
        """Should produce one row per legislator with enough votes."""
        df = self._make_simple_df()
        result = compute_bayesian_loyalty(df, "House")
        assert result.height == 3

    def test_ci_ordering(self) -> None:
        """CI lower < posterior mean < CI upper for all legislators."""
        df = self._make_simple_df()
        result = compute_bayesian_loyalty(df, "House")

        for row in result.iter_rows(named=True):
            assert row["ci_lower"] < row["posterior_mean"] < row["ci_upper"], (
                f"CI violation for {row['legislator_slug']}: "
                f"{row['ci_lower']:.3f} < {row['posterior_mean']:.3f} < {row['ci_upper']:.3f}"
            )

    def test_ci_width_positive(self) -> None:
        """CI width should always be positive."""
        df = self._make_simple_df()
        result = compute_bayesian_loyalty(df, "House")

        for row in result.iter_rows(named=True):
            assert row["ci_width"] > 0

    def test_posterior_between_raw_and_prior(self) -> None:
        """Posterior mean should always be between raw rate and prior mean."""
        # Create data where legislators deviate from the party mean
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_loyal",
                    "party": "Republican",
                    "full_name": "Loyal",
                    "district": "1",
                    "votes_with_party": 95,
                    "party_votes_present": 100,
                    "unity_score": 0.95,
                    "maverick_rate": 0.05,
                },
                {
                    "legislator_slug": "rep_maverick",
                    "party": "Republican",
                    "full_name": "Maverick",
                    "district": "2",
                    "votes_with_party": 70,
                    "party_votes_present": 100,
                    "unity_score": 0.70,
                    "maverick_rate": 0.30,
                },
                {
                    "legislator_slug": "rep_avg",
                    "party": "Republican",
                    "full_name": "Average",
                    "district": "3",
                    "votes_with_party": 85,
                    "party_votes_present": 100,
                    "unity_score": 0.85,
                    "maverick_rate": 0.15,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        for row in result.iter_rows(named=True):
            raw = row["raw_loyalty"]
            prior = row["prior_mean"]
            post = row["posterior_mean"]
            lo = min(raw, prior)
            hi = max(raw, prior)
            assert lo - 0.001 <= post <= hi + 0.001, (
                f"Posterior {post:.4f} outside [{lo:.4f}, {hi:.4f}] for {row['legislator_slug']}"
            )

    def test_excludes_below_min_votes(self) -> None:
        """Legislators with fewer than MIN_PARTY_VOTES are excluded."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_ok",
                    "party": "Republican",
                    "full_name": "OK",
                    "district": "1",
                    "votes_with_party": 90,
                    "party_votes_present": 100,
                    "unity_score": 0.90,
                    "maverick_rate": 0.10,
                },
                {
                    "legislator_slug": "rep_too_few",
                    "party": "Republican",
                    "full_name": "Too Few",
                    "district": "2",
                    "votes_with_party": 2,
                    "party_votes_present": 2,
                    "unity_score": 1.0,
                    "maverick_rate": 0.0,
                },
                {
                    "legislator_slug": "rep_another",
                    "party": "Republican",
                    "full_name": "Another",
                    "district": "3",
                    "votes_with_party": 85,
                    "party_votes_present": 100,
                    "unity_score": 0.85,
                    "maverick_rate": 0.15,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        slugs = set(result["legislator_slug"].to_list())
        assert "rep_too_few" not in slugs
        assert "rep_ok" in slugs
        assert "rep_another" in slugs

    def test_both_parties(self) -> None:
        """Should compute posteriors for both Republican and Democrat."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_a",
                    "party": "Republican",
                    "full_name": "Rep A",
                    "district": "1",
                    "votes_with_party": 90,
                    "party_votes_present": 100,
                    "unity_score": 0.90,
                    "maverick_rate": 0.10,
                },
                {
                    "legislator_slug": "rep_b",
                    "party": "Republican",
                    "full_name": "Rep B",
                    "district": "2",
                    "votes_with_party": 85,
                    "party_votes_present": 100,
                    "unity_score": 0.85,
                    "maverick_rate": 0.15,
                },
                {
                    "legislator_slug": "dem_a",
                    "party": "Democrat",
                    "full_name": "Dem A",
                    "district": "10",
                    "votes_with_party": 80,
                    "party_votes_present": 100,
                    "unity_score": 0.80,
                    "maverick_rate": 0.20,
                },
                {
                    "legislator_slug": "dem_b",
                    "party": "Democrat",
                    "full_name": "Dem B",
                    "district": "11",
                    "votes_with_party": 75,
                    "party_votes_present": 100,
                    "unity_score": 0.75,
                    "maverick_rate": 0.25,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        parties = set(result["party"].to_list())
        assert "Republican" in parties
        assert "Democrat" in parties


# ── TestShrinkageProperties ──────────────────────────────────────────────────


class TestShrinkageProperties:
    """Tests for the key statistical property: shrinkage is proportional to sample size."""

    def test_high_n_barely_shrinks(self) -> None:
        """A legislator with many votes should have low shrinkage (< 0.25)."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": f"rep_{i}",
                    "party": "Republican",
                    "full_name": f"Rep {i}",
                    "district": str(i),
                    "votes_with_party": int(170 + i * 3),
                    "party_votes_present": 200,
                    "unity_score": (170 + i * 3) / 200,
                    "maverick_rate": 1 - (170 + i * 3) / 200,
                }
                for i in range(10)
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        for row in result.iter_rows(named=True):
            assert row["shrinkage"] < 0.25, (
                f"Shrinkage {row['shrinkage']:.3f} too high for n={row['n_party_votes']}"
            )

    def test_low_n_shrinks_a_lot(self) -> None:
        """A legislator with few votes should have higher shrinkage."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_small",
                    "party": "Republican",
                    "full_name": "Small N",
                    "district": "1",
                    "votes_with_party": 4,
                    "party_votes_present": 5,
                    "unity_score": 0.80,
                    "maverick_rate": 0.20,
                },
                {
                    "legislator_slug": "rep_big",
                    "party": "Republican",
                    "full_name": "Big N",
                    "district": "2",
                    "votes_with_party": 160,
                    "party_votes_present": 200,
                    "unity_score": 0.80,
                    "maverick_rate": 0.20,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        small = result.filter(pl.col("legislator_slug") == "rep_small").row(0, named=True)
        big = result.filter(pl.col("legislator_slug") == "rep_big").row(0, named=True)

        assert small["shrinkage"] > big["shrinkage"], (
            f"Small-n shrinkage {small['shrinkage']:.3f} should exceed "
            f"big-n shrinkage {big['shrinkage']:.3f}"
        )

    def test_shrinkage_range(self) -> None:
        """Shrinkage should be between 0 and 1."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": f"rep_{i}",
                    "party": "Republican",
                    "full_name": f"Rep {i}",
                    "district": str(i),
                    "votes_with_party": int(80 + i),
                    "party_votes_present": 100,
                    "unity_score": (80 + i) / 100,
                    "maverick_rate": 1 - (80 + i) / 100,
                }
                for i in range(10)
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        for row in result.iter_rows(named=True):
            assert 0 <= row["shrinkage"] <= 1, f"Shrinkage {row['shrinkage']:.3f} outside [0, 1]"


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for unusual data configurations."""

    def test_single_legislator_in_party(self) -> None:
        """A party with only 1 legislator should be skipped (can't estimate prior)."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "dem_solo",
                    "party": "Democrat",
                    "full_name": "Solo Dem",
                    "district": "1",
                    "votes_with_party": 80,
                    "party_votes_present": 100,
                    "unity_score": 0.80,
                    "maverick_rate": 0.20,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "Senate")
        # Should be empty — can't compute prior with 1 legislator
        assert result.height == 0

    def test_all_identical_rates(self) -> None:
        """All legislators with the exact same rate should produce tiny CI widths."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": f"rep_{i}",
                    "party": "Republican",
                    "full_name": f"Rep {i}",
                    "district": str(i),
                    "votes_with_party": 90,
                    "party_votes_present": 100,
                    "unity_score": 0.90,
                    "maverick_rate": 0.10,
                }
                for i in range(5)
            ]
        )
        result = compute_bayesian_loyalty(df, "House")
        assert result.height == 5

        # All posterior means should be near 0.90
        for row in result.iter_rows(named=True):
            assert abs(row["posterior_mean"] - 0.90) < 0.02

    def test_legislator_with_zero_defections(self) -> None:
        """A legislator with 100% loyalty — should still be shrunk slightly toward party mean."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_perfect",
                    "party": "Republican",
                    "full_name": "Perfect",
                    "district": "1",
                    "votes_with_party": 50,
                    "party_votes_present": 50,
                    "unity_score": 1.0,
                    "maverick_rate": 0.0,
                },
                {
                    "legislator_slug": "rep_normal",
                    "party": "Republican",
                    "full_name": "Normal",
                    "district": "2",
                    "votes_with_party": 85,
                    "party_votes_present": 100,
                    "unity_score": 0.85,
                    "maverick_rate": 0.15,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")
        perfect = result.filter(pl.col("legislator_slug") == "rep_perfect").row(0, named=True)

        # Posterior should be slightly below 1.0 (shrunk toward party mean)
        assert perfect["posterior_mean"] < 1.0
        assert perfect["raw_loyalty"] == 1.0

    def test_empty_input(self) -> None:
        """Empty input DataFrame should return empty result."""
        df = _make_unity_df([])
        result = compute_bayesian_loyalty(df, "House")
        assert result.height == 0

    def test_no_qualifying_legislators(self) -> None:
        """All legislators below MIN_PARTY_VOTES should return empty result."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_tiny",
                    "party": "Republican",
                    "full_name": "Tiny",
                    "district": "1",
                    "votes_with_party": 1,
                    "party_votes_present": 1,
                    "unity_score": 1.0,
                    "maverick_rate": 0.0,
                },
                {
                    "legislator_slug": "rep_tiny2",
                    "party": "Republican",
                    "full_name": "Tiny2",
                    "district": "2",
                    "votes_with_party": 2,
                    "party_votes_present": 2,
                    "unity_score": 1.0,
                    "maverick_rate": 0.0,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")
        assert result.height == 0

    def test_mixed_parties_separate_priors(self) -> None:
        """Republicans and Democrats should get different alpha_prior values."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": "rep_a",
                    "party": "Republican",
                    "full_name": "Rep A",
                    "district": "1",
                    "votes_with_party": 95,
                    "party_votes_present": 100,
                    "unity_score": 0.95,
                    "maverick_rate": 0.05,
                },
                {
                    "legislator_slug": "rep_b",
                    "party": "Republican",
                    "full_name": "Rep B",
                    "district": "2",
                    "votes_with_party": 90,
                    "party_votes_present": 100,
                    "unity_score": 0.90,
                    "maverick_rate": 0.10,
                },
                {
                    "legislator_slug": "dem_a",
                    "party": "Democrat",
                    "full_name": "Dem A",
                    "district": "10",
                    "votes_with_party": 70,
                    "party_votes_present": 100,
                    "unity_score": 0.70,
                    "maverick_rate": 0.30,
                },
                {
                    "legislator_slug": "dem_b",
                    "party": "Democrat",
                    "full_name": "Dem B",
                    "district": "11",
                    "votes_with_party": 65,
                    "party_votes_present": 100,
                    "unity_score": 0.65,
                    "maverick_rate": 0.35,
                },
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        r_prior = result.filter(pl.col("party") == "Republican")["prior_mean"][0]
        d_prior = result.filter(pl.col("party") == "Democrat")["prior_mean"][0]

        # Republican prior should be higher (they have higher loyalty)
        assert r_prior > d_prior, (
            f"Republican prior {r_prior:.3f} should exceed Democrat prior {d_prior:.3f}"
        )

    def test_output_includes_votes_with_party(self) -> None:
        """Output should include integer votes_with_party column."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": f"rep_{i}",
                    "party": "Republican",
                    "full_name": f"Rep {i}",
                    "district": str(i),
                    "votes_with_party": 85 + i,
                    "party_votes_present": 100,
                    "unity_score": (85 + i) / 100,
                    "maverick_rate": 1 - (85 + i) / 100,
                }
                for i in range(5)
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        assert "votes_with_party" in result.columns
        assert "prior_kappa" in result.columns
        # votes_with_party should match input
        for row in result.iter_rows(named=True):
            assert isinstance(row["votes_with_party"], int)
            expected_y = round(row["raw_loyalty"] * row["n_party_votes"])
            assert row["votes_with_party"] == expected_y

    def test_prior_kappa_is_alpha_plus_beta(self) -> None:
        """prior_kappa should equal alpha_prior + beta_prior."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": f"rep_{i}",
                    "party": "Republican",
                    "full_name": f"Rep {i}",
                    "district": str(i),
                    "votes_with_party": 80 + i * 2,
                    "party_votes_present": 100,
                    "unity_score": (80 + i * 2) / 100,
                    "maverick_rate": 1 - (80 + i * 2) / 100,
                }
                for i in range(5)
            ]
        )
        result = compute_bayesian_loyalty(df, "House")

        for row in result.iter_rows(named=True):
            expected_kappa = row["alpha_prior"] + row["beta_prior"]
            assert abs(row["prior_kappa"] - expected_kappa) < 1e-10


# ── TestMethodOfMoments ─────────────────────────────────────────────────────


class TestMethodOfMoments:
    """Tests for statistical correctness of method-of-moments estimation."""

    def test_uses_sample_variance(self) -> None:
        """Method of moments should use sample variance (ddof=1), not population variance.

        With 5 rates, sample variance (ddof=1) gives a slightly larger variance
        than population variance (ddof=0), producing a lower concentration.
        """
        y = np.array([80, 85, 90, 95, 100])
        n = np.array([100, 100, 100, 100, 100])
        rates = y / n
        sample_var = float(np.var(rates, ddof=1))
        mu = float(np.mean(rates))
        expected_concentration = mu * (1 - mu) / sample_var - 1

        alpha, beta = estimate_beta_params(y, n)
        actual_concentration = alpha + beta

        assert abs(actual_concentration - expected_concentration) < 0.1, (
            f"Concentration {actual_concentration:.2f} should be near "
            f"{expected_concentration:.2f} (sample variance)"
        )

    def test_variable_sample_sizes(self) -> None:
        """Legislators with different numbers of party votes should all get valid posteriors."""
        df = _make_unity_df(
            [
                {
                    "legislator_slug": f"rep_{i}",
                    "party": "Republican",
                    "full_name": f"Rep {i}",
                    "district": str(i),
                    "votes_with_party": int(n_val * 0.85),
                    "party_votes_present": n_val,
                    "unity_score": int(n_val * 0.85) / n_val,
                    "maverick_rate": 1 - int(n_val * 0.85) / n_val,
                }
                for i, n_val in enumerate([5, 20, 50, 100, 200])
            ]
        )
        result = compute_bayesian_loyalty(df, "House")
        assert result.height == 5

        # Legislators with more votes should have narrower CIs
        sorted_result = result.sort("n_party_votes")
        ci_widths = sorted_result["ci_width"].to_list()
        for i in range(len(ci_widths) - 1):
            assert ci_widths[i] >= ci_widths[i + 1] - 0.001, (
                f"CI width should decrease with sample size: "
                f"n={sorted_result['n_party_votes'][i]} width={ci_widths[i]:.3f} vs "
                f"n={sorted_result['n_party_votes'][i + 1]} width={ci_widths[i + 1]:.3f}"
            )


# ── TestTaroneTest ──────────────────────────────────────────────────────────


class TestTaroneTest:
    """Tests for Tarone's score test for Beta-Binomial overdispersion."""

    def test_overdispersed_data(self) -> None:
        """Known overdispersed data should produce significant Tarone's Z."""
        rng = np.random.default_rng(42)
        # Draw different thetas from Beta(5, 1) — clearly overdispersed
        thetas = rng.beta(5, 1, size=50)
        n = np.full(50, 100)
        y = rng.binomial(n, thetas)
        z, p = tarone_test(y, n)
        assert z > 2.0, f"Z={z:.2f} should be large for overdispersed data"
        assert p < 0.05, f"p={p:.4f} should be significant"

    def test_binomial_data_not_significant(self) -> None:
        """Pure binomial data (no overdispersion) should not be significant."""
        rng = np.random.default_rng(42)
        n = np.full(50, 100)
        y = rng.binomial(n, 0.9)  # Same theta for all — no overdispersion
        z, p = tarone_test(y, n)
        assert p > 0.01, f"p={p:.4f} should not be significant for pure Binomial"

    def test_returns_tuple(self) -> None:
        """Should return (z_statistic, p_value) tuple."""
        y = np.array([80, 90, 70, 85])
        n = np.array([100, 100, 100, 100])
        result = tarone_test(y, n)
        assert len(result) == 2
        z, p = result
        assert isinstance(z, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_degenerate_input(self) -> None:
        """Degenerate data (all same) should not crash."""
        y = np.array([90, 90, 90])
        n = np.array([100, 100, 100])
        z, p = tarone_test(y, n)
        assert isinstance(z, float)
        assert isinstance(p, float)
