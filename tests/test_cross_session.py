"""Tests for cross-session validation data logic.

Run:
    uv run pytest tests/test_cross_session.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from analysis.cross_session_data import (
    SHIFT_THRESHOLD_SD,
    FreshmenAnalysis,
    align_feature_columns,
    align_irt_scales,
    analyze_freshmen_cohort,
    classify_turnover,
    compare_feature_importance,
    compute_bloc_stability,
    compute_icc,
    compute_ideology_shift,
    compute_metric_stability,
    compute_psi,
    compute_turnover_impact,
    fuzzy_match_legislators,
    interpret_icc,
    interpret_psi,
    interpret_stability,
    match_legislators,
    normalize_name,
    standardize_features,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_legislators(
    names: list[str],
    *,
    prefix: str = "rep",
    party: str = "Republican",
    chamber: str = "House",
    start_district: int = 1,
) -> pl.DataFrame:
    """Build a legislators DataFrame matching the CSV schema."""
    return pl.DataFrame(
        {
            "slug": [f"{prefix}_{n.split()[-1].lower()}" for n in names],
            "full_name": names,
            "party": [party] * len(names),
            "chamber": [chamber] * len(names),
            "district": list(range(start_district, start_district + len(names))),
        }
    )


def _make_ideal_points(
    slugs: list[str],
    xi_means: list[float],
    *,
    names: list[str] | None = None,
) -> pl.DataFrame:
    """Build an IRT ideal points DataFrame matching the parquet schema."""
    if names is None:
        names = [s.replace("rep_", "").replace("sen_", "").title() for s in slugs]
    return pl.DataFrame(
        {
            "legislator_slug": slugs,
            "xi_mean": xi_means,
            "xi_sd": [0.1] * len(slugs),
            "full_name": names,
            "party": ["Republican"] * len(slugs),
            "district": list(range(1, len(slugs) + 1)),
            "chamber": ["House"] * len(slugs),
        }
    )


def _make_large_matched(n: int = 25) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create two sessions with n overlapping legislators for alignment tests."""
    names_shared = [f"Member {i}" for i in range(n)]
    names_only_a = [f"Departing D{i}" for i in range(5)]
    names_only_b = [f"Newcomer N{i}" for i in range(5)]

    leg_a = _make_legislators(names_shared + names_only_a)
    leg_b = _make_legislators(names_shared + names_only_b, prefix="rep2")

    matched = match_legislators(leg_a, leg_b)
    return leg_a, leg_b, matched


# ── TestNormalizeName ────────────────────────────────────────────────────────


class TestNormalizeName:
    """Tests for legislator name normalization."""

    def test_lowercase(self) -> None:
        """Names should be lowercased."""
        assert normalize_name("John Smith") == "john smith"

    def test_strip_whitespace(self) -> None:
        """Leading/trailing whitespace should be removed."""
        assert normalize_name("  Jane Doe  ") == "jane doe"

    def test_strip_leadership_suffix(self) -> None:
        """Leadership titles after ' - ' should be removed."""
        assert normalize_name("Bob Jones - House Minority Caucus Chair") == "bob jones"

    def test_strip_complex_suffix(self) -> None:
        """Multiple-word suffixes should be fully removed."""
        assert normalize_name("Alice Brown - Speaker Pro Tem") == "alice brown"

    def test_no_suffix_unchanged(self) -> None:
        """Names without suffixes should pass through normally."""
        assert normalize_name("Simple Name") == "simple name"

    def test_hyphenated_name_preserved(self) -> None:
        """Hyphenated surnames should not be stripped (no space before hyphen)."""
        assert normalize_name("Mary Smith-Jones") == "mary smith-jones"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert normalize_name("") == ""


# ── TestMatchLegislators ─────────────────────────────────────────────────────


class TestMatchLegislators:
    """Tests for cross-session legislator matching."""

    def test_exact_match(self) -> None:
        """Legislators with the same name should match."""
        names = [f"Member {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="rep2")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_case_insensitive_match(self) -> None:
        """Matching should be case-insensitive."""
        leg_a = _make_legislators(["JOHN SMITH"] + [f"M {i}" for i in range(24)])
        leg_b = _make_legislators(["john smith"] + [f"M {i}" for i in range(24)], prefix="x")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_suffix_stripped_match(self) -> None:
        """Leadership suffixes should not prevent matching."""
        names_a = ["Bob Jones - Speaker"] + [f"M {i}" for i in range(24)]
        names_b = ["Bob Jones"] + [f"M {i}" for i in range(24)]
        leg_a = _make_legislators(names_a)
        leg_b = _make_legislators(names_b, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_partial_overlap(self) -> None:
        """Only shared legislators should appear in output."""
        names_shared = [f"Shared {i}" for i in range(22)]
        leg_a = _make_legislators(names_shared + ["Only In A", "Also Only A", "Third Only A"])
        leg_b = _make_legislators(names_shared + ["Only In B", "Also Only B"], prefix="x")
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 22

    def test_too_few_matches_raises(self) -> None:
        """Should raise ValueError if overlap < MIN_OVERLAP."""
        leg_a = _make_legislators([f"A {i}" for i in range(10)])
        leg_b = _make_legislators([f"B {i}" for i in range(10)])
        with pytest.raises(ValueError, match="Only 0 legislators matched"):
            match_legislators(leg_a, leg_b)

    def test_chamber_switch_flagged(self) -> None:
        """A legislator who changed chambers should be flagged."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names, chamber="House")
        leg_b = _make_legislators(names, chamber="House", prefix="x")
        # Change one legislator's chamber in session B
        leg_b = leg_b.with_columns(
            pl.when(pl.col("slug") == "x_0")
            .then(pl.lit("Senate"))
            .otherwise("chamber")
            .alias("chamber")
        )
        matched = match_legislators(leg_a, leg_b)
        switches = matched.filter(pl.col("is_chamber_switch"))
        assert switches.height == 1

    def test_party_switch_flagged(self) -> None:
        """A legislator who changed parties should be flagged."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names, party="Republican")
        leg_b = _make_legislators(names, party="Republican", prefix="x")
        leg_b = leg_b.with_columns(
            pl.when(pl.col("slug") == "x_0")
            .then(pl.lit("Democrat"))
            .otherwise("party")
            .alias("party")
        )
        matched = match_legislators(leg_a, leg_b)
        switches = matched.filter(pl.col("is_party_switch"))
        assert switches.height == 1

    def test_slug_column_name_flexibility(self) -> None:
        """Should handle both 'slug' and 'legislator_slug' column names."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)  # has 'slug' column
        leg_b = _make_legislators(names, prefix="x").rename({"slug": "legislator_slug"})
        matched = match_legislators(leg_a, leg_b)
        assert matched.height == 25

    def test_output_columns(self) -> None:
        """Output should have all expected columns."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        expected = {
            "name_norm",
            "full_name_a",
            "full_name_b",
            "slug_a",
            "slug_b",
            "party_a",
            "party_b",
            "chamber_a",
            "chamber_b",
            "district_a",
            "district_b",
            "is_chamber_switch",
            "is_party_switch",
        }
        assert set(matched.columns) == expected

    def test_sorted_by_name(self) -> None:
        """Output should be sorted by name_norm."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        norms = matched["name_norm"].to_list()
        assert norms == sorted(norms)


# ── TestClassifyTurnover ─────────────────────────────────────────────────────


class TestClassifyTurnover:
    """Tests for turnover classification."""

    def test_counts(self) -> None:
        """Returning + departing should equal session A; returning + new should equal session B."""
        leg_a, leg_b, matched = _make_large_matched(25)

        cohorts = classify_turnover(leg_a, leg_b, matched)
        assert cohorts["returning"].height == 25
        assert cohorts["departing"].height == 5
        assert cohorts["new"].height == 5

    def test_no_departing(self) -> None:
        """If all A legislators are in B, departing should be empty."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names + ["Extra"], prefix="x")
        matched = match_legislators(leg_a, leg_b)
        cohorts = classify_turnover(leg_a, leg_b, matched)
        assert cohorts["departing"].height == 0
        assert cohorts["new"].height == 1

    def test_no_new(self) -> None:
        """If all B legislators were in A, new should be empty."""
        names = [f"M {i}" for i in range(25)]
        leg_a = _make_legislators(names + ["Old Timer"])
        leg_b = _make_legislators(names, prefix="x")
        matched = match_legislators(leg_a, leg_b)
        cohorts = classify_turnover(leg_a, leg_b, matched)
        assert cohorts["departing"].height == 1
        assert cohorts["new"].height == 0


# ── TestAlignIrtScales ───────────────────────────────────────────────────────


class TestAlignIrtScales:
    """Tests for IRT scale alignment."""

    def test_identity_transform(self) -> None:
        """If both sessions have identical xi values, A~1 and B~0."""
        _, _, matched = _make_large_matched(25)
        xi_vals = [float(i) / 10 for i in range(25)]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals,
            names=matched["full_name_b"].to_list(),
        )
        a, b, aligned = align_irt_scales(xi_a, xi_b, matched)
        assert abs(a - 1.0) < 0.05, f"A should be ~1, got {a}"
        assert abs(b) < 0.05, f"B should be ~0, got {b}"

    def test_known_affine_transform(self) -> None:
        """If session B = 2*A + 1, alignment should recover A~2, B~1."""
        _, _, matched = _make_large_matched(25)
        xi_vals_a = [float(i) / 10 for i in range(25)]
        xi_vals_b = [2 * v + 1 for v in xi_vals_a]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals_a)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals_b,
            names=matched["full_name_b"].to_list(),
        )
        a, b, aligned = align_irt_scales(xi_a, xi_b, matched)
        assert abs(a - 2.0) < 0.1, f"A should be ~2, got {a}"
        assert abs(b - 1.0) < 0.1, f"B should be ~1, got {b}"

    def test_aligned_delta_near_zero(self) -> None:
        """After alignment with no real movers, delta_xi should be near zero."""
        _, _, matched = _make_large_matched(25)
        xi_vals = [float(i) / 10 for i in range(25)]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals,
            names=matched["full_name_b"].to_list(),
        )
        _, _, aligned = align_irt_scales(xi_a, xi_b, matched)
        max_delta = aligned["abs_delta_xi"].max()
        assert max_delta < 0.1, f"Max delta should be near 0, got {max_delta}"

    def test_trimming_resists_outliers(self) -> None:
        """A few genuine movers should not distort the alignment."""
        _, _, matched = _make_large_matched(30)
        xi_vals_a = [float(i) / 10 for i in range(30)]
        # Session B matches A except two extreme movers
        xi_vals_b = list(xi_vals_a)
        xi_vals_b[0] += 5.0  # Huge positive shift
        xi_vals_b[1] -= 5.0  # Huge negative shift
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals_a)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals_b,
            names=matched["full_name_b"].to_list(),
        )
        a, b, _ = align_irt_scales(xi_a, xi_b, matched)
        # Should still recover ~identity despite outliers
        assert abs(a - 1.0) < 0.3, f"A should be ~1 despite outliers, got {a}"
        assert abs(b) < 0.3, f"B should be ~0 despite outliers, got {b}"

    def test_too_few_irt_scores_raises(self) -> None:
        """Should raise ValueError if too few legislators have IRT scores."""
        _, _, matched = _make_large_matched(25)
        # Only give IRT scores to 5 legislators
        xi_a = _make_ideal_points(matched["slug_a"].to_list()[:5], [0.1, 0.2, 0.3, 0.4, 0.5])
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list()[:5],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            names=matched["full_name_b"].to_list()[:5],
        )
        with pytest.raises(ValueError, match="Only 5 legislators"):
            align_irt_scales(xi_a, xi_b, matched)

    def test_output_columns(self) -> None:
        """Aligned DataFrame should have all expected columns."""
        _, _, matched = _make_large_matched(25)
        xi_vals = [float(i) / 10 for i in range(25)]
        xi_a = _make_ideal_points(matched["slug_a"].to_list(), xi_vals)
        xi_b = _make_ideal_points(
            matched["slug_b"].to_list(),
            xi_vals,
            names=matched["full_name_b"].to_list(),
        )
        _, _, aligned = align_irt_scales(xi_a, xi_b, matched)
        expected = {
            "name_norm",
            "slug_a",
            "slug_b",
            "party",
            "chamber",
            "xi_a",
            "xi_b",
            "xi_a_aligned",
            "delta_xi",
            "abs_delta_xi",
            "full_name",
        }
        assert set(aligned.columns) == expected


# ── TestComputeIdeologyShift ─────────────────────────────────────────────────


class TestComputeIdeologyShift:
    """Tests for ideology shift classification."""

    def _make_aligned(self, deltas: list[float]) -> pl.DataFrame:
        """Build an aligned DataFrame with known delta values."""
        n = len(deltas)
        xi_a = [float(i) for i in range(n)]
        xi_b = [xi_a[i] + deltas[i] for i in range(n)]
        return pl.DataFrame(
            {
                "name_norm": [f"member {i}" for i in range(n)],
                "slug_a": [f"rep_a_{i}" for i in range(n)],
                "slug_b": [f"rep_b_{i}" for i in range(n)],
                "full_name": [f"Member {i}" for i in range(n)],
                "party": ["Republican"] * n,
                "chamber": ["House"] * n,
                "xi_a": xi_a,
                "xi_b": xi_b,
                "xi_a_aligned": xi_a,
                "delta_xi": deltas,
                "abs_delta_xi": [abs(d) for d in deltas],
            }
        )

    def test_no_movers_all_stable(self) -> None:
        """If all deltas are zero, everyone should be 'stable'."""
        aligned = self._make_aligned([0.0] * 25)
        result = compute_ideology_shift(aligned)
        assert result["is_significant_mover"].sum() == 0
        assert set(result["shift_direction"].to_list()) == {"stable"}

    def test_large_positive_shift_is_rightward(self) -> None:
        """A large positive delta should be classified as rightward."""
        deltas = [0.0] * 24 + [5.0]
        aligned = self._make_aligned(deltas)
        result = compute_ideology_shift(aligned)
        movers = result.filter(pl.col("is_significant_mover"))
        assert movers.height >= 1
        # The legislator with delta=5.0 should be rightward
        big_mover = result.filter(pl.col("delta_xi") == 5.0)
        assert big_mover["shift_direction"][0] == "rightward"

    def test_large_negative_shift_is_leftward(self) -> None:
        """A large negative delta should be classified as leftward."""
        deltas = [0.0] * 24 + [-5.0]
        aligned = self._make_aligned(deltas)
        result = compute_ideology_shift(aligned)
        big_mover = result.filter(pl.col("delta_xi") == -5.0)
        assert big_mover["shift_direction"][0] == "leftward"

    def test_rank_columns_present(self) -> None:
        """Should add rank_a, rank_b, rank_shift columns."""
        aligned = self._make_aligned([0.0] * 25)
        result = compute_ideology_shift(aligned)
        assert "rank_a" in result.columns
        assert "rank_b" in result.columns
        assert "rank_shift" in result.columns

    def test_threshold_sensitivity(self) -> None:
        """With SHIFT_THRESHOLD_SD=1.0, deltas within 1 SD should be stable."""
        # All deltas small and uniform → std is tiny → even small deltas are significant
        # Use a spread of deltas to get a meaningful threshold
        rng = np.random.default_rng(42)
        deltas = rng.normal(0, 0.1, 24).tolist() + [2.0]
        aligned = self._make_aligned(deltas)
        result = compute_ideology_shift(aligned)
        big_mover = result.filter(pl.col("delta_xi") > 1.5)
        assert big_mover["is_significant_mover"].all()


# ── TestComputeMetricStability ───────────────────────────────────────────────


class TestComputeMetricStability:
    """Tests for cross-session metric correlation."""

    def _make_leg_df(
        self, slugs: list[str], metric_vals: list[float], metric_name: str = "unity_score"
    ) -> pl.DataFrame:
        """Build a minimal legislator DataFrame with one metric."""
        return pl.DataFrame({"legislator_slug": slugs, metric_name: metric_vals})

    def test_perfect_correlation(self) -> None:
        """Identical values should give r=1.0."""
        _, _, matched = _make_large_matched(25)
        vals = [float(i) / 25 for i in range(25)]
        df_a = self._make_leg_df(matched["slug_a"].to_list(), vals)
        df_b = self._make_leg_df(matched["slug_b"].to_list(), vals)
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert result.height == 1
        assert result["pearson_r"][0] == pytest.approx(1.0, abs=0.001)

    def test_negative_correlation(self) -> None:
        """Reversed values should give r=-1.0."""
        _, _, matched = _make_large_matched(25)
        vals = [float(i) / 25 for i in range(25)]
        df_a = self._make_leg_df(matched["slug_a"].to_list(), vals)
        df_b = self._make_leg_df(matched["slug_b"].to_list(), list(reversed(vals)))
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert result["pearson_r"][0] == pytest.approx(-1.0, abs=0.001)

    def test_missing_metric_skipped(self) -> None:
        """Metrics not present in one DataFrame should be silently skipped."""
        _, _, matched = _make_large_matched(25)
        vals = [0.5] * 25
        df_a = self._make_leg_df(matched["slug_a"].to_list(), vals)
        df_b = pl.DataFrame({"legislator_slug": matched["slug_b"].to_list()})
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert result.height == 0

    def test_multiple_metrics(self) -> None:
        """Should compute correlations for multiple metrics."""
        _, _, matched = _make_large_matched(25)
        slugs_a = matched["slug_a"].to_list()
        slugs_b = matched["slug_b"].to_list()
        vals = [float(i) / 25 for i in range(25)]
        df_a = pl.DataFrame(
            {"legislator_slug": slugs_a, "unity_score": vals, "maverick_rate": vals}
        )
        df_b = pl.DataFrame(
            {"legislator_slug": slugs_b, "unity_score": vals, "maverick_rate": vals}
        )
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score", "maverick_rate"])
        assert result.height == 2

    def test_default_metrics(self) -> None:
        """Should use STABILITY_METRICS when no metrics specified."""
        _, _, matched = _make_large_matched(25)
        slugs_a = matched["slug_a"].to_list()
        slugs_b = matched["slug_b"].to_list()
        vals = [0.5] * 25
        df_a = pl.DataFrame({"legislator_slug": slugs_a, "unity_score": vals})
        df_b = pl.DataFrame({"legislator_slug": slugs_b, "unity_score": vals})
        result = compute_metric_stability(df_a, df_b, matched)
        # Only unity_score is present in both → 1 row
        assert result.height == 1
        assert result["metric"][0] == "unity_score"

    def test_empty_result_schema(self) -> None:
        """Empty result should have the correct schema."""
        _, _, matched = _make_large_matched(25)
        df_a = pl.DataFrame({"legislator_slug": matched["slug_a"].to_list()})
        df_b = pl.DataFrame({"legislator_slug": matched["slug_b"].to_list()})
        result = compute_metric_stability(df_a, df_b, matched, ["nonexistent"])
        assert result.height == 0
        assert set(result.columns) == {
            "metric",
            "pearson_r",
            "spearman_rho",
            "n_legislators",
            "psi",
            "psi_interpretation",
            "icc",
            "icc_interpretation",
            "stability_interpretation",
        }


# ── TestComputeTurnoverImpact ────────────────────────────────────────────────


class TestComputeTurnoverImpact:
    """Tests for turnover impact analysis."""

    def test_basic_stats(self) -> None:
        """Should compute mean, std, n for each cohort."""
        ret = np.array([1.0, 2.0, 3.0])
        dep = np.array([0.5, 1.5])
        new = np.array([2.5, 3.5])
        result = compute_turnover_impact(ret, dep, new)
        assert result["returning_n"] == 3
        assert result["departing_n"] == 2
        assert result["new_n"] == 2
        assert result["returning_mean"] == pytest.approx(2.0)

    def test_ks_tests_present(self) -> None:
        """KS test results should be present when cohorts are large enough."""
        ret = np.array([1.0, 2.0, 3.0, 4.0])
        dep = np.array([0.5, 1.5, 2.5])
        new = np.array([2.5, 3.5, 4.5])
        result = compute_turnover_impact(ret, dep, new)
        assert "ks_departing_vs_returning" in result
        assert "p_departing_vs_returning" in result
        assert "ks_new_vs_returning" in result
        assert "p_new_vs_returning" in result

    def test_empty_cohort(self) -> None:
        """Empty arrays should produce None for mean/std."""
        ret = np.array([1.0, 2.0])
        dep = np.array([])
        new = np.array([3.0])
        result = compute_turnover_impact(ret, dep, new)
        assert result["departing_mean"] is None
        assert result["departing_std"] is None
        assert result["departing_n"] == 0
        assert "ks_departing_vs_returning" not in result

    def test_single_element_cohort(self) -> None:
        """A single-element cohort should have mean but no std."""
        ret = np.array([1.0, 2.0])
        dep = np.array([1.5])
        new = np.array([2.5])
        result = compute_turnover_impact(ret, dep, new)
        assert result["departing_mean"] == pytest.approx(1.5)
        assert result["departing_std"] is None
        # KS test needs at least 2 elements
        assert "ks_departing_vs_returning" not in result

    def test_identical_distributions(self) -> None:
        """Identical distributions should have high p-value."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_turnover_impact(arr, arr, arr)
        assert result["p_departing_vs_returning"] > 0.9
        assert result["p_new_vs_returning"] > 0.9


# ── TestAlignFeatureColumns ────────────────────────────────────────────────


class TestAlignFeatureColumns:
    """Tests for aligning vote feature columns across sessions.

    Run: uv run pytest tests/test_cross_session.py::TestAlignFeatureColumns -v
    """

    def _make_vote_features(self, n: int, extra_cols: list[str] | None = None) -> pl.DataFrame:
        """Build a minimal vote features DataFrame."""
        data: dict = {
            "legislator_slug": [f"rep_{i}" for i in range(n)],
            "vote_id": [f"je_{i:014d}" for i in range(n)],
            "vote_binary": [1] * n,
            "xi_mean": [float(i) for i in range(n)],
            "beta_mean": [float(i) * 0.5 for i in range(n)],
            "party_binary": [1] * n,
        }
        if extra_cols:
            for col in extra_cols:
                data[col] = [0] * n
        return pl.DataFrame(data)

    def test_shared_columns_kept(self) -> None:
        """Both DataFrames should retain only shared feature columns."""
        df_a = self._make_vote_features(5, extra_cols=["vt_final_action"])
        df_b = self._make_vote_features(5, extra_cols=["vt_conference"])
        a_out, b_out, feats = align_feature_columns(df_a, df_b)
        assert "xi_mean" in feats
        assert "beta_mean" in feats
        assert "vt_final_action" not in feats
        assert "vt_conference" not in feats

    def test_meta_columns_excluded(self) -> None:
        """Meta columns should not appear in feature_cols."""
        df_a = self._make_vote_features(5)
        df_b = self._make_vote_features(5)
        _, _, feats = align_feature_columns(df_a, df_b)
        assert "legislator_slug" not in feats
        assert "vote_id" not in feats
        assert "vote_binary" not in feats

    def test_identical_schemas(self) -> None:
        """If schemas are identical, all features should be shared."""
        df_a = self._make_vote_features(5)
        df_b = self._make_vote_features(5)
        a_out, b_out, feats = align_feature_columns(df_a, df_b)
        assert set(a_out.columns) == set(b_out.columns)
        assert len(feats) == 3  # xi_mean, beta_mean, party_binary

    def test_output_column_order_matches(self) -> None:
        """Both output DataFrames should have columns in the same order."""
        df_a = self._make_vote_features(5, ["vt_a", "vt_shared"])
        df_b = self._make_vote_features(5, ["vt_b", "vt_shared"])
        a_out, b_out, _ = align_feature_columns(df_a, df_b)
        assert a_out.columns == b_out.columns


# ── TestStandardizeFeatures ────────────────────────────────────────────────


class TestStandardizeFeatures:
    """Tests for z-score standardization of vote features.

    Run: uv run pytest tests/test_cross_session.py::TestStandardizeFeatures -v
    """

    def test_continuous_columns_standardized(self) -> None:
        """Continuous columns should have mean~0, std~1 after standardization."""
        df = pl.DataFrame(
            {
                "xi_mean": [1.0, 2.0, 3.0, 4.0, 5.0],
                "beta_mean": [10.0, 20.0, 30.0, 40.0, 50.0],
                "party_binary": [1, 0, 1, 0, 1],
            }
        )
        result = standardize_features(df, ["xi_mean", "beta_mean", "party_binary"])
        assert result["xi_mean"].mean() == pytest.approx(0.0, abs=0.01)
        assert result["xi_mean"].std() == pytest.approx(1.0, abs=0.1)

    def test_binary_columns_unchanged(self) -> None:
        """Binary columns should not be standardized."""
        df = pl.DataFrame(
            {
                "xi_mean": [1.0, 2.0, 3.0, 4.0, 5.0],
                "party_binary": [1, 0, 1, 0, 1],
            }
        )
        result = standardize_features(df, ["xi_mean", "party_binary"])
        assert result["party_binary"].to_list() == [1, 0, 1, 0, 1]

    def test_constant_column_unchanged(self) -> None:
        """A column with zero variance should not be modified."""
        df = pl.DataFrame(
            {
                "xi_mean": [1.0, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0],
            }
        )
        result = standardize_features(df, ["xi_mean", "constant"])
        # constant has 1 unique value → treated as binary → unchanged
        assert result["constant"].to_list() == [5.0, 5.0, 5.0]

    def test_no_numeric_columns(self) -> None:
        """Empty numeric_cols should return unchanged DataFrame."""
        df = pl.DataFrame({"party_binary": [1, 0, 1]})
        result = standardize_features(df, [])
        assert result.equals(df)


# ── TestCompareFeatureImportance ───────────────────────────────────────────


class TestCompareFeatureImportance:
    """Tests for SHAP importance comparison across sessions.

    Run: uv run pytest tests/test_cross_session.py::TestCompareFeatureImportance -v
    """

    def test_identical_importance_high_tau(self) -> None:
        """Identical SHAP values should produce tau=1.0."""
        rng = np.random.default_rng(42)
        shap_vals = rng.random((100, 5))
        names = ["f1", "f2", "f3", "f4", "f5"]
        df, tau = compare_feature_importance(shap_vals, shap_vals, names)
        assert tau == pytest.approx(1.0, abs=0.01)

    def test_reversed_importance_low_tau(self) -> None:
        """Reversed SHAP importance should produce negative tau."""
        n_features = 5
        # Session A: feature 0 most important, feature 4 least
        shap_a = np.zeros((100, n_features))
        for i in range(n_features):
            shap_a[:, i] = (n_features - i) * 0.1
        # Session B: reversed
        shap_b = np.zeros((100, n_features))
        for i in range(n_features):
            shap_b[:, i] = (i + 1) * 0.1
        names = [f"f{i}" for i in range(n_features)]
        df, tau = compare_feature_importance(shap_a, shap_b, names)
        assert tau < 0

    def test_output_columns(self) -> None:
        """Comparison DataFrame should have expected columns."""
        rng = np.random.default_rng(42)
        shap_vals = rng.random((50, 4))
        names = ["a", "b", "c", "d"]
        df, _ = compare_feature_importance(shap_vals, shap_vals, names)
        expected = {"feature", "importance_a", "importance_b", "rank_a", "rank_b"}
        assert set(df.columns) == expected
        assert df.height == 4

    def test_top_k_limits_tau_calculation(self) -> None:
        """top_k should limit how many features are compared."""
        rng = np.random.default_rng(42)
        shap_vals = rng.random((50, 10))
        names = [f"f{i}" for i in range(10)]
        df, tau = compare_feature_importance(shap_vals, shap_vals, names, top_k=3)
        # Should still return all features in the DataFrame
        assert df.height == 10
        # But tau is computed on top 3 only
        assert tau == pytest.approx(1.0, abs=0.01)

    def test_single_feature(self) -> None:
        """Single feature should produce NaN tau (can't rank 1 item)."""
        shap_vals = np.array([[1.0], [2.0], [3.0]])
        df, tau = compare_feature_importance(shap_vals, shap_vals, ["only"])
        assert np.isnan(tau)
        assert df.height == 1

    def test_asymmetry_swapping_sessions(self) -> None:
        """Swapping session A/B SHAP values should produce different tau.

        The function selects top-K features by session A's ranking, then
        computes tau against session B's ranks for those features.  If
        the two sessions have asymmetrically overlapping top features,
        swapping which session is "A" changes the feature selection and
        therefore the tau.

        Run: uv run pytest tests/test_cross_session.py \
             ::TestCompareFeatureImportance::test_asymmetry_swapping_sessions -v
        """
        n_features = 10
        # Session A: first 5 features important, last 5 negligible
        shap_a = np.zeros((200, n_features))
        for i in range(5):
            shap_a[:, i] = np.random.default_rng(i).normal(0, (5 - i) * 1.0, size=200)
        for i in range(5, n_features):
            shap_a[:, i] = np.random.default_rng(i).normal(0, 0.01, size=200)

        # Session B: last 5 features important, first 5 less so (partial overlap)
        shap_b = np.zeros((200, n_features))
        for i in range(5):
            shap_b[:, i] = np.random.default_rng(i + 10).normal(0, 0.1 * (i + 1), size=200)
        for i in range(5, n_features):
            shap_b[:, i] = np.random.default_rng(i + 10).normal(0, (i - 4) * 1.0, size=200)

        names = [f"f{i}" for i in range(n_features)]
        # top_k=5: when A is "source", picks features 0-4; when B is "source", picks 5-9
        _, tau_ab = compare_feature_importance(shap_a, shap_b, names, top_k=5)
        _, tau_ba = compare_feature_importance(shap_b, shap_a, names, top_k=5)
        # The two tau values should differ because different features are selected
        assert tau_ab != pytest.approx(tau_ba, abs=0.05)


# ── TestNormalizeNameEdgeCases ──────────────────────────────────────────────


class TestNormalizeNameEdgeCases:
    """Additional edge case tests for normalize_name.

    Run: uv run pytest tests/test_cross_session.py::TestNormalizeNameEdgeCases -v
    """

    def test_hyphen_dash_distinction(self) -> None:
        """Hyphenated surname preserved; leadership suffix stripped."""
        # Hyphenated name (no space before hyphen) — should be preserved
        assert normalize_name("Mary Smith-Jones") == "mary smith-jones"
        # Leadership suffix (space-dash-space) — should be stripped
        assert normalize_name("Mary Smith - Chair") == "mary smith"

    def test_multiple_dashes_only_first_stripped(self) -> None:
        """Suffix regex matches from the first ' - ' to end of string."""
        result = normalize_name("Bob Jones - Vice Chair - Emeritus")
        assert result == "bob jones"


# ── TestTurnoverImpactScale ────────────────────────────────────────────────


class TestTurnoverImpactScale:
    """Tests ensuring turnover impact analysis uses consistent scales.

    Run: uv run pytest tests/test_cross_session.py::TestTurnoverImpactScale -v
    """

    def test_affine_transformed_departing_matches_returning_scale(self) -> None:
        """Departing xi values should be transformed before comparison.

        If we have xi_a values and a known affine transform (A=2, B=1),
        the transformed departing values should be 2*xi + 1.
        """
        # Simulate: session A has xi values, affine transform is A=2, B=1
        xi_dep_raw = np.array([0.0, 0.5, 1.0])
        a_coef, b_coef = 2.0, 1.0
        xi_dep_aligned = xi_dep_raw * a_coef + b_coef

        # After transform, values should be [1.0, 2.0, 3.0]
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(xi_dep_aligned, expected)

        # Now compare against returning legislators on session B scale
        xi_ret = np.array([1.5, 2.5, 3.5])
        result = compute_turnover_impact(xi_ret, xi_dep_aligned, np.array([2.0, 3.0]))
        assert result["departing_mean"] == pytest.approx(2.0)
        assert result["returning_mean"] == pytest.approx(2.5)

    def test_untransformed_departing_gives_wrong_answer(self) -> None:
        """Without affine transform, departing mean would be wrong."""
        xi_dep_raw = np.array([0.0, 0.5, 1.0])  # Session A scale
        xi_ret = np.array([1.5, 2.5, 3.5])  # Session B scale
        a_coef, b_coef = 2.0, 1.0

        # Wrong: using raw values
        result_wrong = compute_turnover_impact(xi_ret, xi_dep_raw, np.array([]))
        # Correct: using transformed values
        xi_dep_aligned = xi_dep_raw * a_coef + b_coef
        result_correct = compute_turnover_impact(xi_ret, xi_dep_aligned, np.array([]))

        # The raw departing mean (0.5) is very different from the aligned (2.0)
        assert result_wrong["departing_mean"] == pytest.approx(0.5)
        assert result_correct["departing_mean"] == pytest.approx(2.0)


# ── TestShiftThresholdConsistency ──────────────────────────────────────────


class TestShiftThresholdConsistency:
    """Tests verifying threshold consistency between classification and display.

    Run: uv run pytest tests/test_cross_session.py::TestShiftThresholdConsistency -v
    """

    def test_ddof_consistency(self) -> None:
        """Classification and visualization thresholds should use same ddof."""
        rng = np.random.default_rng(42)
        n = 50
        deltas = rng.normal(0, 0.5, n).tolist()

        aligned = pl.DataFrame(
            {
                "name_norm": [f"m {i}" for i in range(n)],
                "slug_a": [f"a_{i}" for i in range(n)],
                "slug_b": [f"b_{i}" for i in range(n)],
                "full_name": [f"Member {i}" for i in range(n)],
                "party": ["Republican"] * n,
                "chamber": ["House"] * n,
                "xi_a": [float(i) for i in range(n)],
                "xi_b": [float(i) + d for i, d in enumerate(deltas)],
                "xi_a_aligned": [float(i) for i in range(n)],
                "delta_xi": deltas,
                "abs_delta_xi": [abs(d) for d in deltas],
            }
        )

        # Classification threshold (Polars std, ddof=1)
        result = compute_ideology_shift(aligned)
        polars_std = aligned["delta_xi"].std()
        classification_threshold = SHIFT_THRESHOLD_SD * polars_std

        # Visualization threshold should match (np.std with ddof=1)
        numpy_std = float(np.std(np.array(deltas), ddof=1))
        viz_threshold = SHIFT_THRESHOLD_SD * numpy_std

        assert classification_threshold == pytest.approx(viz_threshold, rel=1e-10)

        # Verify mover count matches between the two thresholds
        movers_classification = int(result["is_significant_mover"].sum())
        movers_viz = int(np.sum(np.abs(np.array(deltas)) > viz_threshold))
        assert movers_classification == movers_viz


# ── TestValidateDetection ──────────────────────────────────────────────────


class TestValidateDetection:
    """Tests for the detection validation helper.

    Run: uv run pytest tests/test_cross_session.py::TestValidateDetection -v
    """

    def test_basic_detection_returns_dict(self) -> None:
        """validate_detection should return a dict with all expected keys."""
        from analysis.cross_session import validate_detection

        # Build minimal leg_df with required columns for detection functions
        n = 30
        leg_df = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}" for i in range(n)],
                "full_name": [f"Member {i}" for i in range(n)],
                "party": ["Republican"] * 20 + ["Democrat"] * 10,
                "chamber": ["House"] * n,
                "district": list(range(1, n + 1)),
                "xi_mean": [float(i) / n for i in range(n)],
                "unity_score": [0.9] * 19 + [0.3] + [0.85] * 10,
                "maverick_rate": [0.1] * 19 + [0.7] + [0.15] * 10,
                "weighted_maverick": [0.05] * 19 + [0.6] + [0.1] * 10,
                "betweenness": [0.01] * 19 + [0.5] + [0.02] * 10,
                "eigenvector": [0.1] * n,
                "loyalty_rate": [0.9] * 19 + [0.4] + [0.85] * 10,
            }
        )

        result = validate_detection(leg_df, leg_df, "House")
        expected_keys = {
            "maverick_a",
            "maverick_b",
            "bridge_a",
            "bridge_b",
            "paradox_a",
            "paradox_b",
        }
        assert set(result.keys()) == expected_keys


# ── TestMajorityParty ──────────────────────────────────────────────────────


class TestMajorityParty:
    """Tests for the _majority_party helper.

    Run: uv run pytest tests/test_cross_session.py::TestMajorityParty -v
    """

    def test_returns_largest_party(self) -> None:
        """Should return the party with the most members."""
        from analysis.cross_session import _majority_party

        df = pl.DataFrame({"party": ["Republican"] * 5 + ["Democrat"] * 3})
        assert _majority_party(df) == "Republican"

    def test_empty_dataframe_returns_none(self) -> None:
        """Empty DataFrame should return None."""
        from analysis.cross_session import _majority_party

        df = pl.DataFrame({"party": []}, schema={"party": pl.Utf8})
        assert _majority_party(df) is None


# ── TestExtractName ────────────────────────────────────────────────────────


class TestExtractName:
    """Tests for the _extract_name helper.

    Run: uv run pytest tests/test_cross_session.py::TestExtractName -v
    """

    def test_strips_suffix(self) -> None:
        """Should extract last name, ignoring leadership suffix."""
        from analysis.cross_session import _extract_name

        assert _extract_name("Bob Jones - Speaker") == "Jones"

    def test_single_word_name(self) -> None:
        """Single-word name should be returned as-is."""
        from analysis.cross_session import _extract_name

        assert _extract_name("Cher") == "Cher"


# ── TestPlotSmoke ──────────────────────────────────────────────────────────


class TestPlotSmoke:
    """Smoke tests for all cross-session plot functions.

    These verify plots don't crash and produce output files.
    They do NOT verify visual correctness.

    Run: uv run pytest tests/test_cross_session.py::TestPlotSmoke -v
    """

    @staticmethod
    def _make_shifted(n: int = 30) -> pl.DataFrame:
        """Build a shifted DataFrame for plot tests."""
        rng = np.random.default_rng(42)
        xi_a = rng.normal(0, 1, n)
        deltas = rng.normal(0, 0.3, n)
        xi_b = xi_a + deltas
        return pl.DataFrame(
            {
                "full_name": [f"Member {i}" for i in range(n)],
                "party": ["Republican"] * (n // 2) + ["Democrat"] * (n - n // 2),
                "xi_a_aligned": xi_a.tolist(),
                "xi_b": xi_b.tolist(),
                "delta_xi": deltas.tolist(),
                "abs_delta_xi": np.abs(deltas).tolist(),
                "is_significant_mover": ([True] * 3 + [False] * (n - 3)),
                "shift_direction": (["rightward"] * 2 + ["leftward"] + ["stable"] * (n - 3)),
                "rank_a": list(range(1, n + 1)),
                "rank_b": list(range(1, n + 1)),
                "rank_shift": [0] * n,
            }
        )

    def test_plot_ideology_shift_scatter(self, tmp_path: Path) -> None:
        """Ideology scatter plot should produce a PNG."""
        from analysis.cross_session import plot_ideology_shift_scatter

        shifted = self._make_shifted()
        plot_ideology_shift_scatter(shifted, "House", tmp_path, "2023-24", "2025-26")
        assert (tmp_path / "ideology_shift_scatter_house.png").exists()

    def test_plot_biggest_movers(self, tmp_path: Path) -> None:
        """Biggest movers bar chart should produce a PNG."""
        from analysis.cross_session import plot_biggest_movers

        shifted = self._make_shifted()
        plot_biggest_movers(shifted, "House", tmp_path)
        assert (tmp_path / "biggest_movers_house.png").exists()

    def test_plot_shift_distribution(self, tmp_path: Path) -> None:
        """Shift distribution histogram should produce a PNG."""
        from analysis.cross_session import plot_shift_distribution

        shifted = self._make_shifted()
        plot_shift_distribution(shifted, "House", tmp_path)
        assert (tmp_path / "shift_distribution_house.png").exists()

    def test_plot_turnover_impact(self, tmp_path: Path) -> None:
        """Turnover strip plot should produce a PNG."""
        from analysis.cross_session import plot_turnover_impact

        rng = np.random.default_rng(42)
        plot_turnover_impact(
            rng.normal(0, 1, 20),
            rng.normal(-0.5, 1, 10),
            rng.normal(0.5, 1, 10),
            "House",
            tmp_path,
            "2023-24",
            "2025-26",
        )
        assert (tmp_path / "turnover_impact_house.png").exists()

    def test_plot_prediction_comparison(self, tmp_path: Path) -> None:
        """Prediction AUC bar chart should produce a PNG."""
        from analysis.cross_session import plot_prediction_comparison

        plot_prediction_comparison(0.98, 0.97, 0.90, 0.88, "House", tmp_path, "2023-24", "2025-26")
        assert (tmp_path / "prediction_comparison_house.png").exists()

    def test_plot_feature_importance_comparison(self, tmp_path: Path) -> None:
        """Feature importance side-by-side chart should produce a PNG."""
        from analysis.cross_session import plot_feature_importance_comparison

        comp_df = pl.DataFrame(
            {
                "feature": ["xi_mean", "beta_mean", "party_binary"],
                "importance_a": [0.5, 0.3, 0.2],
                "importance_b": [0.4, 0.35, 0.25],
                "rank_a": [1, 2, 3],
                "rank_b": [2, 1, 3],
            }
        )
        plot_feature_importance_comparison(comp_df, "House", tmp_path, "2023-24", "2025-26")
        assert (tmp_path / "feature_importance_comparison_house.png").exists()


# ── TestBuildCrossSessionReport ────────────────────────────────────────────


class TestBuildCrossSessionReport:
    """Tests for the cross-session report builder.

    Run: uv run pytest tests/test_cross_session.py::TestBuildCrossSessionReport -v
    """

    def test_report_adds_sections(self, tmp_path: Path) -> None:
        """Building a report should add sections without crashing."""
        from analysis.cross_session_report import build_cross_session_report

        from analysis.report import ReportBuilder

        rng = np.random.default_rng(42)
        n = 25
        shifted = pl.DataFrame(
            {
                "full_name": [f"Member {i}" for i in range(n)],
                "party": ["Republican"] * n,
                "xi_a_aligned": rng.normal(0, 1, n).tolist(),
                "xi_b": rng.normal(0, 1, n).tolist(),
                "delta_xi": rng.normal(0, 0.3, n).tolist(),
                "abs_delta_xi": np.abs(rng.normal(0, 0.3, n)).tolist(),
                "shift_direction": ["stable"] * n,
                "is_significant_mover": [False] * n,
                "rank_shift": [0] * n,
            }
        )
        stability = pl.DataFrame(
            {
                "metric": ["unity_score"],
                "pearson_r": [0.85],
                "spearman_rho": [0.82],
                "n_legislators": [25],
                "psi": [0.05],
                "psi_interpretation": ["stable"],
                "icc": [0.88],
                "icc_interpretation": ["good"],
                "stability_interpretation": ["good"],
            }
        )
        matched = pl.DataFrame(
            {
                "name_norm": [f"member {i}" for i in range(n)],
                "is_chamber_switch": [False] * n,
                "is_party_switch": [False] * n,
                "chamber_b": ["House"] * n,
            }
        )

        results = {
            "matched": matched,
            "n_departing": 5,
            "n_new": 5,
            "n_matched": n,
            "chambers": ["house"],
            "alignment_coefficients": {"House": {"A": 1.0, "B": 0.0}},
            "house": {
                "shifted": shifted,
                "stability": stability,
                "turnover": {},
                "detection": {
                    "maverick_a": "Alice",
                    "maverick_b": "Bob",
                    "bridge_a": None,
                    "bridge_b": None,
                    "paradox_a": None,
                    "paradox_b": None,
                },
                "r_value": 0.92,
                "prediction": None,
            },
        }

        report = ReportBuilder("Test Cross-Session Report")
        build_cross_session_report(
            report,
            results=results,
            plots_dir=tmp_path,
            session_a_label="90th_2023-2024",
            session_b_label="91st_2025-2026",
        )
        assert len(report._sections) > 0


# ── TestComputePSI ──────────────────────────────────────────────────────────


class TestComputePSI:
    """Tests for Population Stability Index computation.

    Run: uv run pytest tests/test_cross_session.py::TestComputePSI -v
    """

    def test_identical_distributions(self) -> None:
        """Identical arrays should have PSI near 0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        psi = compute_psi(a, a)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_shifted_distribution(self) -> None:
        """Shifted distribution should have higher PSI."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(2, 1, 200)
        psi = compute_psi(a, b)
        assert psi > 0.10  # Meaningful drift

    def test_empty_array_returns_nan(self) -> None:
        """Arrays with < 2 elements should return NaN."""
        assert np.isnan(compute_psi(np.array([1.0]), np.array([1.0, 2.0])))
        assert np.isnan(compute_psi(np.array([]), np.array([])))

    def test_psi_is_non_negative(self) -> None:
        """PSI is always >= 0 for valid inputs."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0.5, 1.2, 100)
        psi = compute_psi(a, b)
        assert psi >= 0

    def test_interpret_psi_thresholds(self) -> None:
        """PSI interpretation should follow standard thresholds."""
        assert interpret_psi(0.05) == "stable"
        assert interpret_psi(0.15) == "investigate"
        assert interpret_psi(0.30) == "significant drift"
        assert interpret_psi(float("nan")) == "insufficient data"


# ── TestComputeICC ──────────────────────────────────────────────────────────


class TestComputeICC:
    """Tests for Intraclass Correlation Coefficient computation.

    Run: uv run pytest tests/test_cross_session.py::TestComputeICC -v
    """

    def test_perfect_agreement(self) -> None:
        """Identical arrays should have ICC = 1.0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        icc = compute_icc(a, a)
        assert icc == pytest.approx(1.0, abs=0.001)

    def test_no_agreement(self) -> None:
        """Random uncorrelated arrays should have ICC near 0."""
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 1, 100)
        icc = compute_icc(a, b)
        assert abs(icc) < 0.3  # Allow some sampling noise

    def test_known_value(self) -> None:
        """Known linear transform should produce high ICC."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        b = a + 0.1  # Small constant shift
        icc = compute_icc(a, b)
        assert icc > 0.95  # Very high consistency

    def test_too_few_subjects(self) -> None:
        """ICC with < 3 subjects should return NaN."""
        assert np.isnan(compute_icc(np.array([1.0, 2.0]), np.array([1.0, 2.0])))

    def test_interpret_icc_thresholds(self) -> None:
        """ICC interpretation should follow Koo & Li 2016 thresholds."""
        assert interpret_icc(0.30) == "poor"
        assert interpret_icc(0.60) == "moderate"
        assert interpret_icc(0.80) == "good"
        assert interpret_icc(0.95) == "excellent"
        assert interpret_icc(float("nan")) == "insufficient data"


# ── TestFuzzyMatchLegislators ───────────────────────────────────────────────


class TestFuzzyMatchLegislators:
    """Tests for fuzzy name matching between sessions.

    Run: uv run pytest tests/test_cross_session.py::TestFuzzyMatchLegislators -v
    """

    def test_catches_typo(self) -> None:
        """A small typo should be matched above the default threshold."""
        result = fuzzy_match_legislators(
            ["john smithe"],
            ["john smith"],
            threshold=0.85,
        )
        assert result.height == 1
        assert result["name_a"][0] == "john smithe"
        assert result["name_b"][0] == "john smith"
        assert result["similarity"][0] > 0.85

    def test_threshold_respected(self) -> None:
        """Names below the threshold should not match."""
        result = fuzzy_match_legislators(
            ["alice jones"],
            ["bob williams"],
            threshold=0.85,
        )
        assert result.height == 0

    def test_empty_unmatched(self) -> None:
        """Empty input lists should return empty DataFrame."""
        result = fuzzy_match_legislators([], ["bob smith"])
        assert result.height == 0
        assert set(result.columns) == {"name_a", "name_b", "similarity"}

    def test_no_false_positives(self) -> None:
        """Very different names should not match even at low threshold."""
        result = fuzzy_match_legislators(
            ["zachary"],
            ["alexander"],
            threshold=0.50,
        )
        assert result.height == 0

    def test_best_match_selected(self) -> None:
        """When multiple candidates exist, the best match should be selected."""
        result = fuzzy_match_legislators(
            ["john smith"],
            ["john smithe", "jane doe"],
            threshold=0.80,
        )
        assert result.height == 1
        assert result["name_b"][0] == "john smithe"


# ── TestMatchLegislatorsFuzzy ───────────────────────────────────────────────


class TestMatchLegislatorsFuzzy:
    """Tests for fuzzy_threshold parameter in match_legislators.

    Run: uv run pytest tests/test_cross_session.py::TestMatchLegislatorsFuzzy -v
    """

    def test_fuzzy_finds_extra_match(self) -> None:
        """Fuzzy matching should find names that differ slightly."""
        names_shared = [f"Member {i}" for i in range(20)]
        # Add a typo pair
        names_a = names_shared + ["John Smithe"]
        names_b = names_shared + ["John Smith"]

        leg_a = _make_legislators(names_a)
        leg_b = _make_legislators(names_b, prefix="rep2")

        exact = match_legislators(leg_a, leg_b)
        fuzzy = match_legislators(leg_a, leg_b, fuzzy_threshold=0.85)

        # Fuzzy should find one more match
        assert fuzzy.height == exact.height + 1

    def test_none_threshold_no_change(self) -> None:
        """fuzzy_threshold=None should behave identically to default."""
        names = [f"Member {i}" for i in range(25)]
        leg_a = _make_legislators(names)
        leg_b = _make_legislators(names, prefix="rep2")

        default = match_legislators(leg_a, leg_b)
        explicit_none = match_legislators(leg_a, leg_b, fuzzy_threshold=None)

        assert default.height == explicit_none.height


# ── TestStabilityInterpretation ─────────────────────────────────────────────


class TestStabilityInterpretation:
    """Tests for stability interpretation based on Spearman rho.

    Run: uv run pytest tests/test_cross_session.py::TestStabilityInterpretation -v
    """

    def test_interpret_stability_thresholds(self) -> None:
        """Interpretation should follow Koo & Li 2016 thresholds."""
        assert interpret_stability(0.30) == "poor"
        assert interpret_stability(0.60) == "moderate"
        assert interpret_stability(0.80) == "good"
        assert interpret_stability(0.95) == "excellent"
        assert interpret_stability(float("nan")) == "insufficient data"

    def test_negative_rho_uses_absolute(self) -> None:
        """Negative rho should use abs() for interpretation."""
        assert interpret_stability(-0.85) == "good"
        assert interpret_stability(-0.95) == "excellent"

    def test_stability_column_in_output(self) -> None:
        """compute_metric_stability output should include stability_interpretation."""
        _, _, matched = _make_large_matched(25)
        vals = [float(i) / 25 for i in range(25)]
        df_a = pl.DataFrame({"legislator_slug": matched["slug_a"].to_list(), "unity_score": vals})
        df_b = pl.DataFrame({"legislator_slug": matched["slug_b"].to_list(), "unity_score": vals})
        result = compute_metric_stability(df_a, df_b, matched, ["unity_score"])
        assert "stability_interpretation" in result.columns
        assert "psi" in result.columns
        assert "icc" in result.columns
        assert result["stability_interpretation"][0] == "excellent"

    def test_empty_stability_df_schema(self) -> None:
        """Empty stability DataFrame should include new columns."""
        from analysis.cross_session_data import _empty_stability_df

        df = _empty_stability_df()
        expected_cols = {
            "metric",
            "pearson_r",
            "spearman_rho",
            "n_legislators",
            "psi",
            "psi_interpretation",
            "icc",
            "icc_interpretation",
            "stability_interpretation",
        }
        assert set(df.columns) == expected_cols


# ── Tests: Analyze Freshmen Cohort ───────────────────────────────────────


class TestAnalyzeFreshmenCohort:
    """Tests for analyze_freshmen_cohort() function."""

    @pytest.fixture
    def freshmen_fixture(self):
        """Create turnover, leg_df_b, and irt_b for freshmen cohort tests."""
        # Session B legislators: 4 returning, 4 new
        leg_df_b = pl.DataFrame(
            {
                "legislator_slug": [
                    "rep_a", "rep_b", "rep_c", "rep_d",  # returning
                    "rep_e", "rep_f", "rep_g", "rep_h",  # new
                ],
                "full_name": [
                    "Alice A", "Bob B", "Carol C", "Dave D",
                    "Eve E", "Frank F", "Grace G", "Hank H",
                ],
                "party": ["Republican"] * 4 + ["Democrat"] * 2 + ["Republican"] * 2,
                "chamber": ["house"] * 8,
                "unity_score": [0.95, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.65],
                "maverick_rate": [0.05, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35],
            }
        )
        irt_b = pl.DataFrame(
            {
                "legislator_slug": leg_df_b["legislator_slug"],
                "xi_mean": [2.0, 1.5, 1.0, 0.5, -1.0, -2.0, 0.8, 0.3],
            }
        )
        turnover = {
            "returning": pl.DataFrame(
                {
                    "slug_a": ["rep_a_old", "rep_b_old", "rep_c_old", "rep_d_old"],
                    "slug_b": ["rep_a", "rep_b", "rep_c", "rep_d"],
                }
            ),
            "new": pl.DataFrame(
                {"legislator_slug": ["rep_e", "rep_f", "rep_g", "rep_h"]}
            ),
            "departed": pl.DataFrame(
                {"legislator_slug": ["rep_x", "rep_y"]}
            ),
        }
        return turnover, leg_df_b, irt_b

    def test_returns_freshmen_analysis(self, freshmen_fixture):
        turnover, leg_df_b, irt_b = freshmen_fixture
        result = analyze_freshmen_cohort(turnover, leg_df_b, irt_b)
        assert result is not None
        assert isinstance(result, FreshmenAnalysis)

    def test_counts_correct(self, freshmen_fixture):
        turnover, leg_df_b, irt_b = freshmen_fixture
        result = analyze_freshmen_cohort(turnover, leg_df_b, irt_b)
        assert result.n_new == 4
        assert result.n_returning == 4

    def test_ideology_comparison(self, freshmen_fixture):
        turnover, leg_df_b, irt_b = freshmen_fixture
        result = analyze_freshmen_cohort(turnover, leg_df_b, irt_b)
        assert result.ideology_new_mean is not None
        assert result.ideology_returning_mean is not None
        assert result.ideology_ks_stat is not None
        assert result.ideology_ks_p is not None
        assert 0.0 <= result.ideology_ks_p <= 1.0

    def test_unity_comparison(self, freshmen_fixture):
        turnover, leg_df_b, irt_b = freshmen_fixture
        result = analyze_freshmen_cohort(turnover, leg_df_b, irt_b)
        assert result.unity_new_mean is not None
        assert result.unity_returning_mean is not None
        # New members have lower unity in our fixture
        assert result.unity_new_mean < result.unity_returning_mean

    def test_maverick_comparison(self, freshmen_fixture):
        turnover, leg_df_b, irt_b = freshmen_fixture
        result = analyze_freshmen_cohort(turnover, leg_df_b, irt_b)
        assert result.maverick_new_mean is not None
        assert result.maverick_returning_mean is not None

    def test_cohort_df_has_is_new(self, freshmen_fixture):
        turnover, leg_df_b, irt_b = freshmen_fixture
        result = analyze_freshmen_cohort(turnover, leg_df_b, irt_b)
        assert "is_new" in result.cohort_df.columns
        assert result.cohort_df.filter(pl.col("is_new")).height == 4

    def test_returns_none_for_insufficient_data(self):
        """Should return None when fewer than 3 new or returning legislators."""
        turnover = {
            "returning": pl.DataFrame({"slug_a": ["a"], "slug_b": ["b"]}),
            "new": pl.DataFrame({"legislator_slug": ["c", "d"]}),
            "departed": pl.DataFrame({"legislator_slug": []}),
        }
        leg_df = pl.DataFrame(
            {
                "legislator_slug": ["b", "c", "d"],
                "full_name": ["B", "C", "D"],
                "party": ["Republican"] * 3,
                "chamber": ["house"] * 3,
            }
        )
        irt = pl.DataFrame(
            {
                "legislator_slug": ["b", "c", "d"],
                "xi_mean": [1.0, 0.5, -0.5],
            }
        )
        result = analyze_freshmen_cohort(turnover, leg_df, irt)
        assert result is None


# ── Tests: Compute Bloc Stability ────────────────────────────────────────


class TestComputeBlocStability:
    """Tests for compute_bloc_stability() function."""

    @pytest.fixture
    def bloc_fixture(self):
        """Create km_a, km_b, matched for bloc stability tests."""
        # 8 legislators, 6 stay in same cluster, 2 switch
        km_a = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}" for i in range(8)],
                "cluster": [0, 0, 0, 0, 1, 1, 1, 1],
            }
        )
        km_b = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}_new" for i in range(8)],
                "cluster": [0, 0, 0, 1, 1, 1, 1, 0],  # rep_3 and rep_7 switch
            }
        )
        matched = pl.DataFrame(
            {
                "slug_a": [f"rep_{i}" for i in range(8)],
                "slug_b": [f"rep_{i}_new" for i in range(8)],
            }
        )
        return km_a, km_b, matched

    def test_returns_dict(self, bloc_fixture):
        km_a, km_b, matched = bloc_fixture
        result = compute_bloc_stability(km_a, km_b, matched)
        assert result is not None
        assert isinstance(result, dict)
        assert "ari" in result
        assert "n_paired" in result
        assert "transition_df" in result
        assert "switchers" in result

    def test_ari_between_0_and_1(self, bloc_fixture):
        km_a, km_b, matched = bloc_fixture
        result = compute_bloc_stability(km_a, km_b, matched)
        assert -1.0 <= result["ari"] <= 1.0

    def test_perfect_stability(self):
        """ARI should be 1.0 when clusters are identical."""
        km_a = pl.DataFrame(
            {
                "legislator_slug": [f"r{i}" for i in range(6)],
                "cluster": [0, 0, 0, 1, 1, 1],
            }
        )
        km_b = pl.DataFrame(
            {
                "legislator_slug": [f"r{i}_b" for i in range(6)],
                "cluster": [0, 0, 0, 1, 1, 1],
            }
        )
        matched = pl.DataFrame(
            {
                "slug_a": [f"r{i}" for i in range(6)],
                "slug_b": [f"r{i}_b" for i in range(6)],
            }
        )
        result = compute_bloc_stability(km_a, km_b, matched)
        assert result["ari"] == pytest.approx(1.0)
        assert result["switchers"].height == 0

    def test_switchers_detected(self, bloc_fixture):
        km_a, km_b, matched = bloc_fixture
        result = compute_bloc_stability(km_a, km_b, matched)
        # rep_3 (0->1) and rep_7 (1->0) should be switchers
        assert result["switchers"].height == 2

    def test_transition_df_columns(self, bloc_fixture):
        km_a, km_b, matched = bloc_fixture
        result = compute_bloc_stability(km_a, km_b, matched)
        trans = result["transition_df"]
        assert "cluster_a" in trans.columns
        assert "cluster_b" in trans.columns
        assert "count" in trans.columns

    def test_n_paired_correct(self, bloc_fixture):
        km_a, km_b, matched = bloc_fixture
        result = compute_bloc_stability(km_a, km_b, matched)
        assert result["n_paired"] == 8

    def test_returns_none_for_insufficient_pairs(self):
        """Should return None when fewer than 5 matched legislators."""
        km_a = pl.DataFrame(
            {"legislator_slug": ["a", "b", "c"], "cluster": [0, 0, 1]}
        )
        km_b = pl.DataFrame(
            {"legislator_slug": ["a_b", "b_b", "c_b"], "cluster": [0, 1, 1]}
        )
        matched = pl.DataFrame(
            {"slug_a": ["a", "b", "c"], "slug_b": ["a_b", "b_b", "c_b"]}
        )
        result = compute_bloc_stability(km_a, km_b, matched)
        assert result is None

    def test_with_leg_df_b_adds_names(self, bloc_fixture):
        km_a, km_b, matched = bloc_fixture
        leg_df_b = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}_new" for i in range(8)],
                "full_name": [f"Name {i}" for i in range(8)],
                "party": ["Republican"] * 4 + ["Democrat"] * 4,
            }
        )
        result = compute_bloc_stability(km_a, km_b, matched, leg_df_b=leg_df_b)
        switchers = result["switchers"]
        assert "full_name" in switchers.columns
        assert "party" in switchers.columns

    def test_handles_slug_column_name(self):
        """Should work with 'slug' column (not 'legislator_slug')."""
        km_a = pl.DataFrame(
            {"slug": [f"r{i}" for i in range(6)], "cluster": [0, 0, 0, 1, 1, 1]}
        )
        km_b = pl.DataFrame(
            {
                "legislator_slug": [f"r{i}_b" for i in range(6)],
                "cluster": [0, 0, 0, 1, 1, 1],
            }
        )
        matched = pl.DataFrame(
            {"slug_a": [f"r{i}" for i in range(6)], "slug_b": [f"r{i}_b" for i in range(6)]}
        )
        result = compute_bloc_stability(km_a, km_b, matched)
        assert result is not None
        assert result["ari"] == pytest.approx(1.0)
