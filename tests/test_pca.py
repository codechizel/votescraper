"""
Tests for PCA imputation and orientation in analysis/pca.py.

Covers row-mean imputation (NaN handling), PC1 sign convention (Republicans
positive), and vote matrix filtering for sensitivity analysis.

Run: uv run pytest tests/test_pca.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.pca import impute_vote_matrix, orient_pc1

# ── impute_vote_matrix() ────────────────────────────────────────────────────


class TestImputeVoteMatrix:
    """Row-mean imputation: fill NaN with each legislator's Yea rate."""

    def test_no_missing_values(self):
        """Matrix with no nulls passes through unchanged."""
        df = pl.DataFrame({
            "legislator_slug": ["a", "b"],
            "v1": [1.0, 0.0],
            "v2": [1.0, 1.0],
        })
        X, slugs, vote_ids = impute_vote_matrix(df)
        assert slugs == ["a", "b"]
        assert vote_ids == ["v1", "v2"]
        np.testing.assert_array_equal(X[0], [1.0, 1.0])
        np.testing.assert_array_equal(X[1], [0.0, 1.0])

    def test_single_null_imputed_with_row_mean(self):
        """One missing vote filled with that legislator's mean."""
        df = pl.DataFrame({
            "legislator_slug": ["a"],
            "v1": [1.0],
            "v2": [None],
            "v3": [0.0],
        })
        X, _, _ = impute_vote_matrix(df)
        # Row mean of [1.0, 0.0] = 0.5
        assert X[0, 1] == pytest.approx(0.5)

    def test_all_yea_imputes_one(self):
        """Legislator who always votes Yea → missing filled with 1.0."""
        df = pl.DataFrame({
            "legislator_slug": ["a"],
            "v1": [1.0],
            "v2": [1.0],
            "v3": [None],
        })
        X, _, _ = impute_vote_matrix(df)
        assert X[0, 2] == pytest.approx(1.0)

    def test_all_nay_imputes_zero(self):
        """Legislator who always votes Nay → missing filled with 0.0."""
        df = pl.DataFrame({
            "legislator_slug": ["a"],
            "v1": [0.0],
            "v2": [0.0],
            "v3": [None],
        })
        X, _, _ = impute_vote_matrix(df)
        assert X[0, 2] == pytest.approx(0.0)

    def test_all_null_imputes_half(self):
        """Legislator with no votes → filled with 0.5 (uninformative)."""
        df = pl.DataFrame({
            "legislator_slug": ["a"],
            "v1": [None],
            "v2": [None],
        })
        X, _, _ = impute_vote_matrix(df)
        np.testing.assert_array_almost_equal(X[0], [0.5, 0.5])

    def test_shape_preserved(self):
        """Output shape matches input (legislators x votes)."""
        df = pl.DataFrame({
            "legislator_slug": ["a", "b", "c"],
            "v1": [1.0, 0.0, None],
            "v2": [None, 1.0, 0.0],
            "v3": [1.0, None, 1.0],
        })
        X, slugs, vote_ids = impute_vote_matrix(df)
        assert X.shape == (3, 3)
        assert len(slugs) == 3
        assert len(vote_ids) == 3

    def test_no_nans_in_output(self):
        """Output should never contain NaN."""
        df = pl.DataFrame({
            "legislator_slug": ["a", "b"],
            "v1": [1.0, None],
            "v2": [None, 0.0],
            "v3": [None, None],
        })
        X, _, _ = impute_vote_matrix(df)
        assert not np.isnan(X).any()


# ── orient_pc1() ────────────────────────────────────────────────────────────


class TestOrientPC1:
    """Flip PC1 sign so Republicans have positive mean scores."""

    def test_flips_when_republicans_negative(self):
        """Republicans average negative → flip both scores and loadings."""
        scores = np.array([[-1.0, 0.5], [-1.5, 0.3], [1.0, -0.2], [1.5, -0.1]])
        loadings = np.array([[-0.5, 0.3, 0.2], [0.1, -0.4, 0.3]])
        slugs = ["rep_a", "rep_b", "dem_x", "dem_y"]
        legislators = pl.DataFrame({
            "slug": ["rep_a", "rep_b", "dem_x", "dem_y"],
            "party": ["Republican", "Republican", "Democrat", "Democrat"],
        })

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
        """Republicans already positive → no change."""
        scores = np.array([[1.0, 0.5], [1.5, 0.3], [-1.0, -0.2]])
        loadings = np.array([[0.5, 0.3], [0.1, -0.4]])
        slugs = ["rep_a", "rep_b", "dem_x"]
        legislators = pl.DataFrame({
            "slug": ["rep_a", "rep_b", "dem_x"],
            "party": ["Republican", "Republican", "Democrat"],
        })

        new_scores, new_loadings = orient_pc1(scores, loadings, slugs, legislators)

        # Unchanged
        assert new_scores[0, 0] == pytest.approx(1.0)
        assert new_loadings[0, 0] == pytest.approx(0.5)

    def test_handles_unknown_party(self):
        """Legislators not in party lookup are ignored for orientation."""
        scores = np.array([[-1.0], [1.0], [0.0]])
        loadings = np.array([[-0.5]])
        slugs = ["rep_a", "dem_x", "unknown_z"]
        legislators = pl.DataFrame({
            "slug": ["rep_a", "dem_x"],
            "party": ["Republican", "Democrat"],
        })

        new_scores, _ = orient_pc1(scores, loadings, slugs, legislators)
        # rep_a was -1.0, should be flipped to +1.0
        assert new_scores[0, 0] == pytest.approx(1.0)
