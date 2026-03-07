"""
Tests for PCA imputation, orientation, and diagnostics in analysis/pca.py.

Covers row-mean imputation (NaN handling), PC1 sign convention (Republicans
positive), vote matrix filtering for sensitivity, parallel analysis,
reconstruction error, and extreme PC2 detection.

Run: uv run pytest tests/test_pca.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.pca import (
    ExtremePC2Legislator,
    build_scores_df,
    compute_reconstruction_error,
    detect_extreme_pc2,
    filter_vote_matrix_for_sensitivity,
    fit_pca,
    impute_vote_matrix,
    orient_pc1,
    parallel_analysis,
)

# ── impute_vote_matrix() ────────────────────────────────────────────────────


class TestImputeVoteMatrix:
    """Row-mean imputation: fill NaN with each legislator's Yea rate."""

    def test_no_missing_values(self):
        """Matrix with no nulls passes through unchanged."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a", "b"],
                "v1": [1.0, 0.0],
                "v2": [1.0, 1.0],
            }
        )
        X, slugs, vote_ids = impute_vote_matrix(df)
        assert slugs == ["a", "b"]
        assert vote_ids == ["v1", "v2"]
        np.testing.assert_array_equal(X[0], [1.0, 1.0])
        np.testing.assert_array_equal(X[1], [0.0, 1.0])

    def test_single_null_imputed_with_row_mean(self):
        """One missing vote filled with that legislator's mean."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a"],
                "v1": [1.0],
                "v2": [None],
                "v3": [0.0],
            }
        )
        X, _, _ = impute_vote_matrix(df)
        # Row mean of [1.0, 0.0] = 0.5
        assert X[0, 1] == pytest.approx(0.5)

    def test_all_yea_imputes_one(self):
        """Legislator who always votes Yea -> missing filled with 1.0."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a"],
                "v1": [1.0],
                "v2": [1.0],
                "v3": [None],
            }
        )
        X, _, _ = impute_vote_matrix(df)
        assert X[0, 2] == pytest.approx(1.0)

    def test_all_nay_imputes_zero(self):
        """Legislator who always votes Nay -> missing filled with 0.0."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a"],
                "v1": [0.0],
                "v2": [0.0],
                "v3": [None],
            }
        )
        X, _, _ = impute_vote_matrix(df)
        assert X[0, 2] == pytest.approx(0.0)

    def test_all_null_imputes_half(self):
        """Legislator with no votes -> filled with 0.5 (uninformative)."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a"],
                "v1": [None],
                "v2": [None],
            }
        )
        X, _, _ = impute_vote_matrix(df)
        np.testing.assert_array_almost_equal(X[0], [0.5, 0.5])

    def test_shape_preserved(self):
        """Output shape matches input (legislators x votes)."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c"],
                "v1": [1.0, 0.0, None],
                "v2": [None, 1.0, 0.0],
                "v3": [1.0, None, 1.0],
            }
        )
        X, slugs, vote_ids = impute_vote_matrix(df)
        assert X.shape == (3, 3)
        assert len(slugs) == 3
        assert len(vote_ids) == 3

    def test_no_nans_in_output(self):
        """Output should never contain NaN."""
        df = pl.DataFrame(
            {
                "legislator_slug": ["a", "b"],
                "v1": [1.0, None],
                "v2": [None, 0.0],
                "v3": [None, None],
            }
        )
        X, _, _ = impute_vote_matrix(df)
        assert not np.isnan(X).any()


# ── orient_pc1() ────────────────────────────────────────────────────────────


class TestOrientPC1:
    """Flip PC1 sign so Republicans have positive mean scores."""

    def test_flips_when_republicans_negative(self):
        """Republicans average negative -> flip both scores and loadings."""
        scores = np.array([[-1.0, 0.5], [-1.5, 0.3], [1.0, -0.2], [1.5, -0.1]])
        loadings = np.array([[-0.5, 0.3, 0.2], [0.1, -0.4, 0.3]])
        slugs = ["rep_a", "rep_b", "dem_x", "dem_y"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "dem_x", "dem_y"],
                "party": ["Republican", "Republican", "Democrat", "Democrat"],
            }
        )

        new_scores, new_loadings = orient_pc1(scores, loadings, slugs, legislators)

        # Republicans should now be positive
        rep_mean = np.mean([new_scores[0, 0], new_scores[1, 0]])
        assert rep_mean > 0

        # PC1 scores should be negated
        assert new_scores[0, 0] == pytest.approx(1.0)
        # PC1 loadings should be negated
        assert new_loadings[0, 0] == pytest.approx(0.5)
        # PC2 should be unchanged
        assert new_scores[0, 1] == pytest.approx(0.5)
        assert new_loadings[1, 0] == pytest.approx(0.1)

    def test_no_flip_when_republicans_positive(self):
        """Republicans already positive -> no change."""
        scores = np.array([[1.0, 0.5], [1.5, 0.3], [-1.0, -0.2]])
        loadings = np.array([[0.5, 0.3], [0.1, -0.4]])
        slugs = ["rep_a", "rep_b", "dem_x"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "dem_x"],
                "party": ["Republican", "Republican", "Democrat"],
            }
        )

        new_scores, new_loadings = orient_pc1(scores, loadings, slugs, legislators)

        # Unchanged
        assert new_scores[0, 0] == pytest.approx(1.0)
        assert new_loadings[0, 0] == pytest.approx(0.5)

    def test_handles_unknown_party(self):
        """Legislators not in party lookup are ignored for orientation."""
        scores = np.array([[-1.0], [1.0], [0.0]])
        loadings = np.array([[-0.5]])
        slugs = ["rep_a", "dem_x", "unknown_z"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "dem_x"],
                "party": ["Republican", "Democrat"],
            }
        )

        new_scores, _ = orient_pc1(scores, loadings, slugs, legislators)
        # rep_a was -1.0, should be flipped to +1.0
        assert new_scores[0, 0] == pytest.approx(1.0)


# ── detect_extreme_pc2() ──────────────────────────────────────────────────


class TestDetectExtremePC2:
    """Detect the most extreme PC2 legislator if >3 sigma from the pack."""

    def _make_scores_df(
        self,
        pc2_values: list[float],
        slugs: list[str] | None = None,
        full_names: list[str | None] | None = None,
        parties: list[str] | None = None,
    ) -> pl.DataFrame:
        """Helper: build a minimal scores DataFrame for testing."""
        n = len(pc2_values)
        if slugs is None:
            slugs = [f"rep_{chr(97 + i)}" for i in range(n)]
        if full_names is None:
            full_names = [f"Test Person{i}" for i in range(n)]
        if parties is None:
            parties = ["Republican"] * n
        return pl.DataFrame(
            {
                "legislator_slug": slugs,
                "PC1": [float(i) for i in range(n)],
                "PC2": pc2_values,
                "full_name": full_names,
                "party": parties,
            }
        )

    def test_returns_none_when_no_extreme(self):
        """All PC2 values within 3 sigma -> returns None."""
        # Tight cluster: std ~ 0.82, 3s ~ 2.45, max |val| = 2.0
        df = self._make_scores_df([0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
        result = detect_extreme_pc2(df)
        assert result is None

    def test_detects_extreme_outlier(self):
        """One value >3 sigma below median -> detected."""
        # 9 legislators near 0, one at -50
        pc2 = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.15, -0.15, -50.0]
        df = self._make_scores_df(pc2)
        result = detect_extreme_pc2(df)
        assert result is not None
        assert result.pc2 == pytest.approx(-50.0)

    def test_fields_populated_correctly(self):
        """All fields on the returned dataclass are correct."""
        pc2 = [0.0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.0, 0.0, -50.0]
        slugs = [f"rep_{chr(97 + i)}" for i in range(10)]
        names = [f"Person {chr(65 + i)}" for i in range(10)]
        parties = ["Republican"] * 9 + ["Democrat"]
        df = self._make_scores_df(pc2, slugs=slugs, full_names=names, parties=parties)

        result = detect_extreme_pc2(df)
        assert result is not None
        assert result.slug == "rep_j"
        assert result.full_name == "Person J"
        assert result.party == "Democrat"
        assert result.pc1 == pytest.approx(9.0)
        assert result.pc2 == pytest.approx(-50.0)
        assert result.pc2_std > 0

    def test_null_full_name_uses_slug(self):
        """Null full_name falls back to the legislator slug."""
        pc2 = [0.0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.0, 0.0, -50.0]
        names: list[str | None] = [f"Person {i}" for i in range(9)] + [None]
        df = self._make_scores_df(pc2, full_names=names)
        result = detect_extreme_pc2(df)
        assert result is not None
        assert result.full_name == "rep_j"  # falls back to slug

    def test_frozen_dataclass(self):
        """ExtremePC2Legislator is immutable."""
        obj = ExtremePC2Legislator(
            slug="rep_a",
            full_name="Test Person",
            party="Republican",
            pc1=1.0,
            pc2=-25.0,
            pc2_std=5.0,
        )
        with pytest.raises(AttributeError):
            obj.slug = "rep_b"  # type: ignore[misc]

    def test_leadership_suffix_stripped(self):
        """Leadership suffixes like ' - House Majority Leader' are stripped."""
        pc2 = [0.0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.0, 0.0, -50.0]
        names: list[str | None] = [f"Person {i}" for i in range(9)] + [
            "Mark Tyson - House Minority Caucus Chair"
        ]
        df = self._make_scores_df(pc2, full_names=names)
        result = detect_extreme_pc2(df)
        assert result is not None
        assert result.full_name == "Mark Tyson"


# ── fit_pca() ──────────────────────────────────────────────────────────────


class TestFitPCA:
    """PCA fitting: component capping and output shapes."""

    def test_components_capped_at_n_rows(self):
        """When n_rows < n_components, caller must cap n_components."""
        # 3 legislators x 100 votes -> max 3 components
        # fit_pca doesn't cap; run_pca_for_chamber does. Test with capped value.
        rng = np.random.default_rng(42)
        X = rng.standard_normal((3, 100))
        n_comp = min(5, X.shape[0], X.shape[1])
        scores, loadings, pca, scaler = fit_pca(X, n_components=n_comp)
        assert scores.shape == (3, 3)
        assert loadings.shape == (3, 100)
        assert pca.n_components_ == 3

    def test_components_capped_at_n_cols(self):
        """When n_cols < n_components, caller must cap n_components."""
        # 100 legislators x 3 votes -> max 3 components
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        n_comp = min(5, X.shape[0], X.shape[1])
        scores, loadings, pca, scaler = fit_pca(X, n_components=n_comp)
        assert scores.shape == (100, 3)
        assert loadings.shape == (3, 3)
        assert pca.n_components_ == 3

    def test_output_shapes(self):
        """Standard case: scores and loadings have expected shapes."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 50))
        scores, loadings, pca, scaler = fit_pca(X, n_components=5)
        assert scores.shape == (20, 5)
        assert loadings.shape == (5, 50)

    def test_standardized_input(self):
        """After StandardScaler, data should have mean ~0 and std ~1."""
        rng = np.random.default_rng(42)
        X = rng.random((50, 10)) * 100  # Unstandardized
        _, _, _, scaler = fit_pca(X, n_components=3)
        X_scaled = scaler.transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0, ddof=0), 1.0, atol=1e-10)


# ── build_scores_df() ──────────────────────────────────────────────────────


class TestBuildScoresDf:
    """Build legislator PC scores DataFrame with metadata."""

    def test_missing_metadata_produces_nulls(self):
        """Legislators not in metadata get null metadata columns, not crash."""
        scores = np.array([[1.0, 0.5], [-1.0, -0.3]])
        slugs = ["rep_a", "rep_unknown"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a"],
                "full_name": ["Alice"],
                "party": ["Republican"],
                "district": ["1"],
                "chamber": ["House"],
            }
        )
        df = build_scores_df(scores, slugs, 2, legislators)
        assert df.height == 2
        # Known legislator has metadata
        row_a = df.filter(pl.col("legislator_slug") == "rep_a")
        assert row_a["full_name"][0] == "Alice"
        # Unknown legislator has null metadata
        row_unknown = df.filter(pl.col("legislator_slug") == "rep_unknown")
        assert row_unknown["full_name"][0] is None

    def test_pc_scores_preserved(self):
        """PC scores are correctly mapped to columns."""
        scores = np.array([[1.5, -0.3], [-2.0, 0.7]])
        slugs = ["rep_a", "dem_b"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "dem_b"],
                "full_name": ["Alice", "Bob"],
                "party": ["Republican", "Democrat"],
                "district": ["1", "2"],
                "chamber": ["House", "House"],
            }
        )
        df = build_scores_df(scores, slugs, 2, legislators)
        assert df["PC1"][0] == pytest.approx(1.5)
        assert df["PC2"][1] == pytest.approx(0.7)


# ── filter_vote_matrix_for_sensitivity() ───────────────────────────────────


class TestFilterVoteMatrixForSensitivity:
    """Re-filter vote matrix at alternative minority threshold."""

    def _make_test_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Build a synthetic vote matrix and rollcalls for testing.

        Creates 20 House legislators with 5 votes at different minority levels:
        - v1: 100% Yea (unanimous, filtered by both thresholds)
        - v2: 5% minority (passes 2.5%, fails 10%)
        - v3: 40% minority (contested, passes both)
        - v4: 50% minority (contested, passes both)
        - v5: 15% minority (passes both)
        """
        slugs = [f"rep_{chr(97 + i)}" if i < 26 else f"rep_{i}" for i in range(20)]
        matrix = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "v1": [1.0] * 20,  # 100% yea -> filtered by both
                "v2": [1.0] * 19 + [0.0],  # 5% minority -> passes 2.5%, fails 10%
                "v3": [1.0] * 12 + [0.0] * 8,  # 40% minority -> passes both
                "v4": [1.0] * 10 + [0.0] * 10,  # 50% minority -> passes both
                "v5": [1.0] * 17 + [0.0] * 3,  # 15% minority -> passes both
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1", "v2", "v3", "v4", "v5"],
                "chamber": ["House"] * 5,
                "bill_number": ["HB1", "HB2", "HB3", "HB4", "HB5"],
            }
        )
        return matrix, rollcalls

    def test_default_threshold_keeps_contested(self):
        """At 2.5%, only the unanimous vote is filtered."""
        matrix, rollcalls = self._make_test_data()
        result = filter_vote_matrix_for_sensitivity(
            matrix,
            rollcalls,
            "House",
            minority_threshold=0.025,
            min_votes=1,
        )
        vote_cols = [c for c in result.columns if c != "legislator_slug"]
        # v1 (0% minority) filtered, v2-v5 kept
        assert "v1" not in vote_cols
        assert "v2" in vote_cols
        assert "v3" in vote_cols

    def test_10pct_threshold_filters_more(self):
        """At 10%, near-unanimous votes with <10% minority are also filtered."""
        matrix, rollcalls = self._make_test_data()
        result = filter_vote_matrix_for_sensitivity(
            matrix,
            rollcalls,
            "House",
            minority_threshold=0.10,
            min_votes=1,
        )
        vote_cols = [c for c in result.columns if c != "legislator_slug"]
        # v1 (0% minority) filtered, v2 (5% minority) filtered
        assert "v1" not in vote_cols
        assert "v2" not in vote_cols
        # v3 (40%), v4 (50%), v5 (10%) kept
        assert "v3" in vote_cols
        assert "v4" in vote_cols
        assert "v5" in vote_cols

    def test_low_participation_filtered(self):
        """Legislators with fewer than min_votes are dropped."""
        matrix, rollcalls = self._make_test_data()
        # Set min_votes very high -> most legislators filtered
        result = filter_vote_matrix_for_sensitivity(
            matrix,
            rollcalls,
            "House",
            minority_threshold=0.025,
            min_votes=100,
        )
        assert result.height == 0

    def test_chamber_restriction(self):
        """Only votes matching the specified chamber are included."""
        matrix, rollcalls = self._make_test_data()
        # Change rollcalls chamber to Senate
        rollcalls_senate = rollcalls.with_columns(pl.lit("Senate").alias("chamber"))
        result = filter_vote_matrix_for_sensitivity(
            matrix,
            rollcalls_senate,
            "House",
            minority_threshold=0.025,
            min_votes=1,
        )
        # No House votes in rollcalls -> empty result
        vote_cols = [c for c in result.columns if c != "legislator_slug"]
        assert len(vote_cols) == 0


# ── parallel_analysis() ───────────────────────────────────────────────────


class TestParallelAnalysis:
    """Horn's parallel analysis: eigenvalue thresholds from random data."""

    def test_returns_correct_shape(self):
        """Output has n_components elements."""
        thresholds = parallel_analysis(50, 100, n_components=5, n_iter=10)
        assert thresholds.shape == (5,)

    def test_thresholds_positive(self):
        """All threshold eigenvalues are positive."""
        thresholds = parallel_analysis(50, 100, n_components=5, n_iter=10)
        assert (thresholds > 0).all()

    def test_thresholds_decrease(self):
        """Threshold eigenvalues are monotonically decreasing."""
        thresholds = parallel_analysis(50, 100, n_components=5, n_iter=20)
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1]

    def test_deterministic(self):
        """Same inputs produce same outputs (seeded RNG)."""
        t1 = parallel_analysis(30, 50, n_components=3, n_iter=10)
        t2 = parallel_analysis(30, 50, n_components=3, n_iter=10)
        np.testing.assert_array_equal(t1, t2)

    def test_structured_data_exceeds_thresholds(self):
        """Real structure should produce eigenvalues above random thresholds."""
        # Create highly structured data: two groups with opposite patterns
        rng = np.random.default_rng(42)
        n, p = 40, 100
        X = np.zeros((n, p))
        X[:20, :] = rng.binomial(1, 0.9, (20, p))  # Group 1: mostly yea
        X[20:, :] = rng.binomial(1, 0.1, (20, p))  # Group 2: mostly nay

        from sklearn.preprocessing import StandardScaler

        X_scaled = StandardScaler().fit_transform(X.astype(float))
        corr = np.corrcoef(X_scaled, rowvar=False)
        actual_evals = np.sort(np.linalg.eigvalsh(corr))[::-1]

        thresholds = parallel_analysis(n, p, n_components=3, n_iter=20)
        # First eigenvalue should exceed threshold (real structure)
        assert actual_evals[0] > thresholds[0]


# ── compute_reconstruction_error() ────────────────────────────────────────


class TestComputeReconstructionError:
    """Per-legislator reconstruction RMSE from PCA model."""

    def _fit_pca_for_test(
        self, n_legislators: int = 20, n_votes: int = 50
    ) -> tuple[np.ndarray, np.ndarray, object, object, list[str]]:
        """Helper: create synthetic data and fit PCA."""
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        X = rng.random((n_legislators, n_votes))
        slugs = [f"rep_{i}" for i in range(n_legislators)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=5)
        scores = pca.fit_transform(X_scaled)
        return X, scores, pca, scaler, slugs

    def test_returns_correct_shape(self):
        """Output has one row per legislator."""
        X, scores, pca, scaler, slugs = self._fit_pca_for_test()
        df = compute_reconstruction_error(X, scores, pca, scaler, slugs)
        assert df.height == len(slugs)

    def test_columns_present(self):
        """Output has expected columns."""
        X, scores, pca, scaler, slugs = self._fit_pca_for_test()
        df = compute_reconstruction_error(X, scores, pca, scaler, slugs)
        assert "legislator_slug" in df.columns
        assert "reconstruction_rmse" in df.columns
        assert "high_error" in df.columns

    def test_rmse_non_negative(self):
        """All RMSE values are non-negative."""
        X, scores, pca, scaler, slugs = self._fit_pca_for_test()
        df = compute_reconstruction_error(X, scores, pca, scaler, slugs)
        assert (df["reconstruction_rmse"] >= 0).all()

    def test_high_error_flag_correct(self):
        """High error flag matches threshold computation."""
        X, scores, pca, scaler, slugs = self._fit_pca_for_test()
        df = compute_reconstruction_error(X, scores, pca, scaler, slugs)
        rmse_vals = df["reconstruction_rmse"].to_numpy()
        mean_rmse = rmse_vals.mean()
        std_rmse = rmse_vals.std()
        threshold = mean_rmse + 2.0 * std_rmse
        expected_flags = [bool(v > threshold) for v in rmse_vals]
        actual_flags = df["high_error"].to_list()
        assert actual_flags == expected_flags

    def test_perfect_reconstruction_zero_error(self):
        """With n_components = min(n, p), reconstruction is perfect."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(42)
        X = rng.random((5, 10))
        slugs = [f"rep_{i}" for i in range(5)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=5)  # Full rank
        scores = pca.fit_transform(X_scaled)
        df = compute_reconstruction_error(X, scores, pca, scaler, slugs)
        # All RMSE should be ~0
        assert df["reconstruction_rmse"].max() < 1e-10
