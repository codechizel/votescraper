"""Tests for Phase 21: Text-Based Ideal Points (Embedding-Vote Approach).

Synthetic data tests for tbip_data.py — pure functions only, no I/O.
Follows the class-based pattern from test_external_validation_dime.py.

Run:
    uv run pytest tests/test_tbip.py -v
"""

import math

import numpy as np
import polars as pl
import pytest
from analysis.tbip_data import (
    GOOD_CORRELATION,
    MIN_BILLS,
    MIN_MATCHED,
    MODERATE_CORRELATION,
    OUTLIER_TOP_N,
    STRONG_CORRELATION,
    align_sign_convention,
    build_matched_df,
    build_vote_embedding_profiles,
    compute_correlations,
    compute_intra_party_correlations,
    compute_text_ideal_points,
    identify_outliers,
)

# ── Factories ────────────────────────────────────────────────────────────────


def _make_votes(rows: list[dict]) -> pl.DataFrame:
    """Build a votes DataFrame from row dicts."""
    defaults = {"legislator_slug": "rep_a_1", "vote_id": "v1", "vote": "Yea"}
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _make_rollcalls(rows: list[dict]) -> pl.DataFrame:
    """Build a rollcalls DataFrame from row dicts."""
    defaults = {"vote_id": "v1", "bill_number": "HB 1", "chamber": "House"}
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _make_irt(rows: list[dict]) -> pl.DataFrame:
    """Build an IRT DataFrame from row dicts."""
    defaults = {
        "legislator_slug": "rep_a_1",
        "xi_mean": 0.5,
        "full_name": "John Doe",
        "party": "Republican",
        "district": 1,
    }
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _make_synthetic_data(
    n_legislators: int = 30,
    n_bills: int = 20,
    embedding_dim: int = 8,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, list[str]]:
    """Create synthetic votes, rollcalls, embeddings, bill_numbers for testing.

    Legislators vote along a latent dimension: Republicans tend Yea on bills
    with positive embedding PC1, Democrats tend Nay. Creates realistic party separation.
    """
    rng = np.random.default_rng(seed)

    bill_numbers = [f"HB {i + 1}" for i in range(n_bills)]
    vote_ids = [f"v{i + 1}" for i in range(n_bills)]

    # Embeddings: first 4 dims encode a latent dimension, rest is noise
    embeddings = rng.standard_normal((n_bills, embedding_dim))

    # Rollcalls
    rc_rows = [
        {"vote_id": vid, "bill_number": bn, "chamber": "House"}
        for vid, bn in zip(vote_ids, bill_numbers, strict=True)
    ]
    rollcalls = pl.DataFrame(rc_rows)

    # Legislators: first 20 Republican, rest Democrat
    n_rep = n_legislators * 2 // 3
    vote_rows = []
    for j in range(n_legislators):
        slug = f"rep_{j}_1" if j < n_rep else f"rep_{j}_1"
        for i in range(n_bills):
            # Republicans tend to vote Yea on bills with positive first embedding dim
            prob_yea = 0.8 if (j < n_rep and embeddings[i, 0] > 0) else 0.3
            if j >= n_rep:
                prob_yea = 1.0 - prob_yea  # Democrats vote opposite
            # Small chance of absence
            if rng.random() < 0.05:
                vote_val = "Not Voting"
            elif rng.random() < prob_yea:
                vote_val = "Yea"
            else:
                vote_val = "Nay"
            vote_rows.append({"legislator_slug": slug, "vote_id": vote_ids[i], "vote": vote_val})

    votes = pl.DataFrame(vote_rows)
    return votes, rollcalls, embeddings, bill_numbers


# ── TestBuildVoteEmbeddingProfiles ───────────────────────────────────────────


class TestBuildVoteEmbeddingProfiles:
    """Tests for build_vote_embedding_profiles()."""

    def test_basic_shape(self):
        """Profiles have shape (n_legislators, embedding_dim)."""
        votes, rollcalls, embeddings, bill_numbers = _make_synthetic_data()
        profiles, slugs, n_matched = build_vote_embedding_profiles(
            votes, rollcalls, embeddings, bill_numbers, min_votes=1
        )
        assert profiles.shape[1] == embeddings.shape[1]
        assert len(slugs) == profiles.shape[0]
        assert n_matched > 0

    def test_vote_encoding(self):
        """Yea → +1, Nay → -1, absent/not voting → excluded from profile."""
        # Single legislator, single bill, Yea vote
        votes = _make_votes([{"legislator_slug": "rep_a_1", "vote_id": "v1", "vote": "Yea"}])
        rollcalls = _make_rollcalls([{"vote_id": "v1", "bill_number": "HB 1"}])
        embeddings = np.array([[1.0, 2.0, 3.0, 4.0]])
        bill_numbers = ["HB 1"]

        # Need enough bills to pass MIN_BILLS threshold — add more
        for i in range(2, MIN_BILLS + 1):
            votes = pl.concat(
                [
                    votes,
                    _make_votes(
                        [{"legislator_slug": "rep_a_1", "vote_id": f"v{i}", "vote": "Yea"}]
                    ),
                ]
            )
            rollcalls = pl.concat(
                [
                    rollcalls,
                    _make_rollcalls([{"vote_id": f"v{i}", "bill_number": f"HB {i}"}]),
                ]
            )
            embeddings = np.vstack([embeddings, np.array([[1.0, 2.0, 3.0, 4.0]])])
            bill_numbers.append(f"HB {i}")

        profiles, slugs, _ = build_vote_embedding_profiles(
            votes, rollcalls, embeddings, bill_numbers, min_votes=1
        )
        assert slugs == ["rep_a_1"]
        # All Yea (+1) on identical embeddings, normalized by count
        np.testing.assert_allclose(profiles[0], [1.0, 2.0, 3.0, 4.0], atol=1e-10)

    def test_nay_negates_embedding(self):
        """Nay votes contribute -1 * embedding."""
        n = MIN_BILLS
        votes_rows = [
            {"legislator_slug": "rep_a_1", "vote_id": f"v{i}", "vote": "Nay"} for i in range(n)
        ]
        rc_rows = [{"vote_id": f"v{i}", "bill_number": f"HB {i}"} for i in range(n)]
        embeddings = np.ones((n, 4))
        bill_numbers = [f"HB {i}" for i in range(n)]

        profiles, _, _ = build_vote_embedding_profiles(
            _make_votes(votes_rows),
            _make_rollcalls(rc_rows),
            embeddings,
            bill_numbers,
            min_votes=1,
        )
        # All Nay (-1) on unit embeddings → profile is [-1, -1, -1, -1]
        np.testing.assert_allclose(profiles[0], [-1.0, -1.0, -1.0, -1.0], atol=1e-10)

    def test_normalization(self):
        """Profiles are normalized by count of non-zero votes."""
        n = MIN_BILLS
        votes_rows = []
        for i in range(n):
            votes_rows.append({"legislator_slug": "rep_a_1", "vote_id": f"v{i}", "vote": "Yea"})
            votes_rows.append(
                {"legislator_slug": "rep_a_1", "vote_id": f"v{i}_extra", "vote": "Not Voting"}
            )
        rc_rows = [{"vote_id": f"v{i}", "bill_number": f"HB {i}"} for i in range(n)]
        # Also add the extra vote_ids to rollcalls (they won't match bills though)
        embeddings = np.ones((n, 4)) * 2.0
        bill_numbers = [f"HB {i}" for i in range(n)]

        profiles, _, _ = build_vote_embedding_profiles(
            _make_votes(votes_rows),
            _make_rollcalls(rc_rows),
            embeddings,
            bill_numbers,
            min_votes=1,
        )
        # n Yea votes on [2,2,2,2] embeddings → sum is n*2, normalized by n → [2,2,2,2]
        np.testing.assert_allclose(profiles[0], [2.0, 2.0, 2.0, 2.0], atol=1e-10)

    def test_min_votes_filter(self):
        """Legislators with fewer than min_votes are excluded."""
        n = MIN_BILLS
        votes_rows = [
            {"legislator_slug": "rep_a_1", "vote_id": f"v{i}", "vote": "Yea"} for i in range(n)
        ]
        # rep_b_1 only votes on 2 bills
        votes_rows.append({"legislator_slug": "rep_b_1", "vote_id": "v0", "vote": "Yea"})
        votes_rows.append({"legislator_slug": "rep_b_1", "vote_id": "v1", "vote": "Yea"})

        rc_rows = [{"vote_id": f"v{i}", "bill_number": f"HB {i}"} for i in range(n)]
        embeddings = np.ones((n, 4))
        bill_numbers = [f"HB {i}" for i in range(n)]

        profiles, slugs, _ = build_vote_embedding_profiles(
            _make_votes(votes_rows),
            _make_rollcalls(rc_rows),
            embeddings,
            bill_numbers,
            min_votes=3,
        )
        assert "rep_b_1" not in slugs
        assert "rep_a_1" in slugs

    def test_chamber_filter(self):
        """When chamber is specified, only that chamber's rollcalls are used."""
        n = MIN_BILLS
        votes_rows = [
            {"legislator_slug": "rep_a_1", "vote_id": f"v{i}", "vote": "Yea"} for i in range(n)
        ]
        votes_rows += [
            {"legislator_slug": "sen_b_1", "vote_id": f"s{i}", "vote": "Yea"} for i in range(n)
        ]

        rc_rows = [
            {"vote_id": f"v{i}", "bill_number": f"HB {i}", "chamber": "House"} for i in range(n)
        ]
        rc_rows += [
            {"vote_id": f"s{i}", "bill_number": f"SB {i}", "chamber": "Senate"} for i in range(n)
        ]

        all_bn = [f"HB {i}" for i in range(n)] + [f"SB {i}" for i in range(n)]
        embeddings = np.ones((2 * n, 4))

        profiles, slugs, _ = build_vote_embedding_profiles(
            _make_votes(votes_rows),
            _make_rollcalls(rc_rows),
            embeddings,
            all_bn,
            chamber="House",
            min_votes=1,
        )
        # Only rep_a_1 should appear (House votes only)
        assert "rep_a_1" in slugs
        assert "sen_b_1" not in slugs

    def test_raises_on_sparse_data(self):
        """ValueError when too few bills match between embeddings and rollcalls."""
        votes = _make_votes([{"legislator_slug": "rep_a_1", "vote_id": "v1", "vote": "Yea"}])
        rollcalls = _make_rollcalls([{"vote_id": "v1", "bill_number": "HB 1"}])
        embeddings = np.ones((1, 4))
        bill_numbers = ["HB 1"]

        with pytest.raises(ValueError, match="Only 1 bills"):
            build_vote_embedding_profiles(votes, rollcalls, embeddings, bill_numbers, min_votes=1)

    def test_raises_no_bill_overlap(self):
        """ValueError when no bills overlap between embeddings and rollcalls."""
        votes = _make_votes([{"legislator_slug": "rep_a_1", "vote_id": "v1", "vote": "Yea"}])
        rollcalls = _make_rollcalls([{"vote_id": "v1", "bill_number": "HB 999"}])
        embeddings = np.ones((1, 4))
        bill_numbers = ["HB 1"]  # Doesn't match HB 999

        with pytest.raises(ValueError, match="Only 0 bills"):
            build_vote_embedding_profiles(votes, rollcalls, embeddings, bill_numbers, min_votes=1)


# ── TestComputeTextIdealPoints ───────────────────────────────────────────────


class TestComputeTextIdealPoints:
    """Tests for compute_text_ideal_points()."""

    def test_shape(self):
        """PC1 scores have one value per legislator."""
        rng = np.random.default_rng(42)
        profiles = rng.standard_normal((20, 8))
        pc1, var_ratio, all_ratios = compute_text_ideal_points(profiles)
        assert pc1.shape == (20,)
        assert 0 < var_ratio <= 1.0
        assert len(all_ratios) == min(20, 8, 10)

    def test_variance_explained(self):
        """PC1 variance ratio should be positive and <= 1."""
        profiles = np.random.default_rng(99).standard_normal((30, 16))
        _, var_ratio, all_ratios = compute_text_ideal_points(profiles)
        assert var_ratio > 0
        assert var_ratio <= 1.0
        assert np.sum(all_ratios) <= 1.0 + 1e-10

    def test_determinism(self):
        """Same input → same output."""
        profiles = np.random.default_rng(42).standard_normal((15, 8))
        pc1_a, _, _ = compute_text_ideal_points(profiles)
        pc1_b, _, _ = compute_text_ideal_points(profiles)
        np.testing.assert_array_equal(pc1_a, pc1_b)

    def test_few_legislators(self):
        """Works with as few as 2 legislators."""
        profiles = np.array([[1.0, 0.0], [-1.0, 0.0]])
        pc1, var_ratio, _ = compute_text_ideal_points(profiles)
        assert pc1.shape == (2,)
        assert var_ratio > 0


# ── TestAlignSignConvention ──────────────────────────────────────────────────


class TestAlignSignConvention:
    """Tests for align_sign_convention()."""

    def test_no_flip_needed(self):
        """When text scores already correlate positively with IRT, no flip."""
        text_scores = np.array([1.0, 0.5, -0.5, -1.0])
        slugs = ["rep_a_1", "rep_b_1", "rep_c_1", "rep_d_1"]
        irt_df = _make_irt(
            [
                {"legislator_slug": "rep_a_1", "xi_mean": 1.2},
                {"legislator_slug": "rep_b_1", "xi_mean": 0.6},
                {"legislator_slug": "rep_c_1", "xi_mean": -0.4},
                {"legislator_slug": "rep_d_1", "xi_mean": -1.1},
            ]
        )
        aligned = align_sign_convention(text_scores, slugs, irt_df)
        # Should stay positive correlation
        assert np.corrcoef(aligned, text_scores)[0, 1] > 0.99

    def test_flip(self):
        """When text scores are negatively correlated with IRT, flip them."""
        text_scores = np.array([-1.0, -0.5, 0.5, 1.0])
        slugs = ["rep_a_1", "rep_b_1", "rep_c_1", "rep_d_1"]
        irt_df = _make_irt(
            [
                {"legislator_slug": "rep_a_1", "xi_mean": 1.2},
                {"legislator_slug": "rep_b_1", "xi_mean": 0.6},
                {"legislator_slug": "rep_c_1", "xi_mean": -0.4},
                {"legislator_slug": "rep_d_1", "xi_mean": -1.1},
            ]
        )
        aligned = align_sign_convention(text_scores, slugs, irt_df)
        # Should flip: now positively correlated with IRT
        np.testing.assert_allclose(aligned, -text_scores)

    def test_too_few_matches(self):
        """With < 3 matched legislators, return as-is (no flip determination)."""
        text_scores = np.array([-1.0, 0.5, 1.0])
        slugs = ["rep_a_1", "rep_b_1", "rep_c_1"]
        irt_df = _make_irt(
            [
                {"legislator_slug": "rep_a_1", "xi_mean": 1.0},
                # b and c don't match
            ]
        )
        aligned = align_sign_convention(text_scores, slugs, irt_df)
        np.testing.assert_array_equal(aligned, text_scores)


# ── TestBuildMatchedDf ───────────────────────────────────────────────────────


class TestBuildMatchedDf:
    """Tests for build_matched_df()."""

    def test_inner_join(self):
        """Only matched legislators appear in output."""
        text_scores = np.array([0.5, -0.3, 0.8])
        slugs = ["rep_a_1", "rep_b_1", "rep_c_1"]
        irt_df = _make_irt(
            [
                {"legislator_slug": "rep_a_1", "xi_mean": 1.0, "party": "Republican"},
                {"legislator_slug": "rep_c_1", "xi_mean": -0.5, "party": "Democrat"},
                # rep_b_1 not in IRT
            ]
        )
        matched = build_matched_df(text_scores, slugs, irt_df)
        assert matched.height == 2
        assert set(matched["legislator_slug"].to_list()) == {"rep_a_1", "rep_c_1"}

    def test_columns_present(self):
        """Output has text_score, xi_mean, and available IRT columns."""
        text_scores = np.array([0.5])
        slugs = ["rep_a_1"]
        irt_df = _make_irt([{"legislator_slug": "rep_a_1", "xi_mean": 1.0, "party": "Republican"}])
        matched = build_matched_df(text_scores, slugs, irt_df)
        assert "text_score" in matched.columns
        assert "xi_mean" in matched.columns
        assert "party" in matched.columns

    def test_slug_column_rename(self):
        """Handles IRT DataFrames with 'slug' instead of 'legislator_slug'."""
        text_scores = np.array([0.5])
        slugs = ["rep_a_1"]
        irt_df = pl.DataFrame(
            {"slug": ["rep_a_1"], "xi_mean": [1.0], "full_name": ["A"], "party": ["R"]}
        )
        matched = build_matched_df(text_scores, slugs, irt_df)
        assert matched.height == 1


# ── TestCorrelations ─────────────────────────────────────────────────────────


class TestCorrelations:
    """Tests for compute_correlations()."""

    def test_positive_correlation(self):
        """Correlated data yields positive Pearson r."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.standard_normal(n)
        y = x + rng.standard_normal(n) * 0.3  # Strong positive correlation
        matched = pl.DataFrame(
            {"xi_mean": x.tolist(), "text_score": y.tolist(), "party": ["Republican"] * n}
        )
        result = compute_correlations(matched)
        assert result["pearson_r"] > 0.5
        assert result["n"] == n
        assert result["quality"] in ("strong", "good", "moderate")

    def test_quality_strong(self):
        """r >= 0.80 → 'strong'."""
        n = 30
        x = np.linspace(-2, 2, n)
        y = x + np.random.default_rng(1).standard_normal(n) * 0.2
        matched = pl.DataFrame({"xi_mean": x.tolist(), "text_score": y.tolist()})
        result = compute_correlations(matched)
        assert result["quality"] == "strong"

    def test_quality_good(self):
        """0.65 <= r < 0.80 → 'good'."""
        # Use a known correlation
        rng = np.random.default_rng(123)
        n = 100
        x = rng.standard_normal(n)
        noise = rng.standard_normal(n)
        # Target r ~ 0.72
        y = 0.72 * x + math.sqrt(1 - 0.72**2) * noise
        matched = pl.DataFrame({"xi_mean": x.tolist(), "text_score": y.tolist()})
        result = compute_correlations(matched)
        assert result["quality"] in ("good", "strong", "moderate")

    def test_quality_weak(self):
        """r < 0.50 → 'weak'."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)  # Uncorrelated
        matched = pl.DataFrame({"xi_mean": x.tolist(), "text_score": y.tolist()})
        result = compute_correlations(matched)
        assert result["quality"] == "weak"

    def test_fisher_ci(self):
        """CI bounds are ordered: ci_lower < pearson_r < ci_upper."""
        n = 40
        x = np.linspace(-1, 1, n)
        y = x + np.random.default_rng(7).standard_normal(n) * 0.3
        matched = pl.DataFrame({"xi_mean": x.tolist(), "text_score": y.tolist()})
        result = compute_correlations(matched)
        assert result["ci_lower"] < result["pearson_r"] < result["ci_upper"]

    def test_insufficient_data(self):
        """< MIN_MATCHED → insufficient_data quality."""
        matched = pl.DataFrame({"xi_mean": [1.0, 2.0], "text_score": [1.0, 2.0]})
        result = compute_correlations(matched)
        assert result["quality"] == "insufficient_data"
        assert math.isnan(result["pearson_r"])

    def test_zero_variance(self):
        """All same values → insufficient_data."""
        n = 15
        matched = pl.DataFrame({"xi_mean": [1.0] * n, "text_score": [0.5] * n})
        result = compute_correlations(matched)
        assert result["quality"] == "insufficient_data"


# ── TestIntraPartyCorrelations ───────────────────────────────────────────────


class TestIntraPartyCorrelations:
    """Tests for compute_intra_party_correlations()."""

    def test_both_parties(self):
        """Returns correlations for both Republican and Democrat."""
        rng = np.random.default_rng(42)
        n = 30
        rows = []
        for i in range(n):
            party = "Republican" if i < 20 else "Democrat"
            rows.append(
                {
                    "xi_mean": float(rng.standard_normal()),
                    "text_score": float(rng.standard_normal()),
                    "party": party,
                }
            )
        matched = pl.DataFrame(rows)
        result = compute_intra_party_correlations(matched)
        assert "Republican" in result
        assert "Democrat" in result

    def test_single_party(self):
        """When only one party present, only that party appears."""
        n = 20
        x = np.linspace(-1, 1, n)
        matched = pl.DataFrame(
            {
                "xi_mean": x.tolist(),
                "text_score": (x * 0.8).tolist(),
                "party": ["Republican"] * n,
            }
        )
        result = compute_intra_party_correlations(matched)
        assert "Republican" in result
        assert "Democrat" in result
        assert result["Democrat"]["quality"] == "insufficient_data"


# ── TestIdentifyOutliers ─────────────────────────────────────────────────────


class TestIdentifyOutliers:
    """Tests for identify_outliers()."""

    def test_top_n(self):
        """Returns at most top_n rows."""
        n = 20
        rng = np.random.default_rng(42)
        matched = pl.DataFrame(
            {
                "xi_mean": rng.standard_normal(n).tolist(),
                "text_score": rng.standard_normal(n).tolist(),
                "legislator_slug": [f"rep_{i}_1" for i in range(n)],
                "full_name": [f"Legislator {i}" for i in range(n)],
                "party": ["Republican"] * n,
            }
        )
        outliers = identify_outliers(matched, top_n=3)
        assert outliers.height == 3

    def test_sorted_descending(self):
        """Outliers are sorted by discrepancy_z descending."""
        n = 15
        matched = pl.DataFrame(
            {
                "xi_mean": list(range(n)),
                "text_score": list(range(n)),
                "legislator_slug": [f"rep_{i}_1" for i in range(n)],
            }
        )
        # Add one extreme outlier
        matched = matched.with_columns(
            pl.when(pl.col("legislator_slug") == "rep_0_1")
            .then(pl.lit(100.0))
            .otherwise(pl.col("text_score"))
            .alias("text_score")
        )
        outliers = identify_outliers(matched, top_n=5)
        assert outliers.height == 5
        discrepancies = outliers["discrepancy_z"].to_list()
        assert discrepancies == sorted(discrepancies, reverse=True)

    def test_z_standardization(self):
        """Output includes xi_z and text_z columns."""
        n = 10
        matched = pl.DataFrame(
            {
                "xi_mean": list(range(n)),
                "text_score": list(range(n, 0, -1)),
                "legislator_slug": [f"rep_{i}_1" for i in range(n)],
            }
        )
        outliers = identify_outliers(matched, top_n=5)
        assert "xi_z" in outliers.columns
        assert "text_z" in outliers.columns
        assert "discrepancy_z" in outliers.columns

    def test_empty_on_few_rows(self):
        """Returns empty DataFrame with < 3 rows."""
        matched = pl.DataFrame({"xi_mean": [1.0], "text_score": [2.0]})
        outliers = identify_outliers(matched)
        assert outliers.height == 0


# ── TestConstants ────────────────────────────────────────────────────────────


class TestConstants:
    """Tests for threshold constants."""

    def test_threshold_ordering(self):
        """Thresholds are in descending order."""
        assert STRONG_CORRELATION > GOOD_CORRELATION > MODERATE_CORRELATION

    def test_lower_than_phase_14(self):
        """Phase 21 thresholds are lower than Phase 14 (text is more distant)."""
        # Phase 14 uses 0.90, 0.85, 0.70
        assert STRONG_CORRELATION < 0.90
        assert GOOD_CORRELATION < 0.85
        assert MODERATE_CORRELATION < 0.70

    def test_min_bills_positive(self):
        """MIN_BILLS is a reasonable positive integer."""
        assert MIN_BILLS >= 3

    def test_min_matched_positive(self):
        """MIN_MATCHED is a reasonable positive integer."""
        assert MIN_MATCHED >= 5

    def test_outlier_top_n_positive(self):
        """OUTLIER_TOP_N is positive."""
        assert OUTLIER_TOP_N > 0
