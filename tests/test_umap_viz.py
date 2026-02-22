"""
Tests for UMAP ideological landscape in analysis/umap_viz.py.

Covers imputation, orientation, embedding construction, Procrustes similarity,
validation correlations, and (optionally) the UMAP computation itself.

Run: uv run pytest tests/test_umap_viz.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.umap_viz import (
    build_embedding_df,
    compute_procrustes_similarity,
    compute_validation_correlations,
    impute_vote_matrix,
    orient_umap1,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_vote_matrix() -> pl.DataFrame:
    """3 legislators, 4 votes, some nulls."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c"],
            "v1": [1.0, 0.0, 1.0],
            "v2": [1.0, 1.0, None],
            "v3": [0.0, None, 1.0],
            "v4": [1.0, 0.0, 0.0],
        }
    )


@pytest.fixture()
def legislators_df() -> pl.DataFrame:
    """Minimal legislator metadata for 3 reps."""
    return pl.DataFrame(
        {
            "slug": ["rep_a", "rep_b", "rep_c"],
            "full_name": ["Alice Smith", "Bob Jones", "Carol Davis"],
            "party": ["Republican", "Democrat", "Republican"],
            "district": ["1", "2", "3"],
            "chamber": ["House", "House", "House"],
        }
    )


@pytest.fixture()
def all_null_matrix() -> pl.DataFrame:
    """Legislator with no votes at all."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_x"],
            "v1": [None],
            "v2": [None],
        },
        schema={
            "legislator_slug": pl.Utf8,
            "v1": pl.Float64,
            "v2": pl.Float64,
        },
    )


# ── TestImputeVoteMatrix ────────────────────────────────────────────────────


class TestImputeVoteMatrix:
    """Row-mean imputation of binary vote matrix."""

    def test_shape_preserved(self, simple_vote_matrix: pl.DataFrame) -> None:
        """Output array has same shape as input (3 legislators x 4 votes)."""
        X, slugs, vote_ids = impute_vote_matrix(simple_vote_matrix)
        assert X.shape == (3, 4)

    def test_no_nans_after_imputation(self, simple_vote_matrix: pl.DataFrame) -> None:
        """All NaN values are replaced."""
        X, _, _ = impute_vote_matrix(simple_vote_matrix)
        assert not np.isnan(X).any()

    def test_non_null_values_unchanged(self, simple_vote_matrix: pl.DataFrame) -> None:
        """Non-null values in the original matrix are preserved."""
        X, _, _ = impute_vote_matrix(simple_vote_matrix)
        assert X[0, 0] == 1.0  # rep_a, v1
        assert X[1, 0] == 0.0  # rep_b, v1

    def test_row_mean_imputation_value(self, simple_vote_matrix: pl.DataFrame) -> None:
        """Imputed value equals the legislator's non-null mean Yea rate."""
        X, slugs, _ = impute_vote_matrix(simple_vote_matrix)
        # rep_b: v1=0, v2=1, v3=NaN, v4=0 -> mean of [0,1,0] = 0.333...
        b_idx = slugs.index("rep_b")
        assert X[b_idx, 2] == pytest.approx(1 / 3, abs=1e-10)

    def test_all_null_legislator_gets_half(self, all_null_matrix: pl.DataFrame) -> None:
        """Legislator with no votes gets 0.5 fill."""
        X, _, _ = impute_vote_matrix(all_null_matrix)
        assert X[0, 0] == 0.5
        assert X[0, 1] == 0.5

    def test_slugs_extracted(self, simple_vote_matrix: pl.DataFrame) -> None:
        """Slug list matches input order."""
        _, slugs, _ = impute_vote_matrix(simple_vote_matrix)
        assert slugs == ["rep_a", "rep_b", "rep_c"]

    def test_vote_ids_extracted(self, simple_vote_matrix: pl.DataFrame) -> None:
        """Vote ID list excludes the slug column."""
        _, _, vote_ids = impute_vote_matrix(simple_vote_matrix)
        assert vote_ids == ["v1", "v2", "v3", "v4"]


# ── TestOrientUmap1 ─────────────────────────────────────────────────────────


class TestOrientUmap1:
    """Flip UMAP1 so Republicans have positive mean."""

    def test_flip_when_republicans_negative(self, legislators_df: pl.DataFrame) -> None:
        """If Republicans have negative UMAP1 mean, flip the sign."""
        # rep_a (R) = -2, rep_b (D) = 3, rep_c (R) = -1 -> R mean = -1.5
        embedding = np.array([[-2.0, 0.0], [3.0, 0.0], [-1.0, 0.0]])
        slugs = ["rep_a", "rep_b", "rep_c"]
        result = orient_umap1(embedding, slugs, legislators_df)
        # After flip: rep_a = +2, rep_b = -3, rep_c = +1
        assert result[0, 0] > 0
        assert result[1, 0] < 0

    def test_no_flip_when_republicans_positive(self, legislators_df: pl.DataFrame) -> None:
        """If Republicans already positive, no flip."""
        embedding = np.array([[2.0, 0.0], [-3.0, 0.0], [1.0, 0.0]])
        slugs = ["rep_a", "rep_b", "rep_c"]
        result = orient_umap1(embedding, slugs, legislators_df)
        assert result[0, 0] > 0  # Still positive

    def test_unknown_party_ignored(self) -> None:
        """Legislators with unknown party don't affect orientation."""
        legislators = pl.DataFrame(
            {
                "slug": ["rep_a", "rep_b"],
                "full_name": ["A", "B"],
                "party": ["Republican", "Democrat"],
                "district": ["1", "2"],
                "chamber": ["House", "House"],
            }
        )
        # rep_a (R) = -1, rep_b (D) = 2, rep_unknown = 5
        embedding = np.array([[-1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
        slugs = ["rep_a", "rep_b", "rep_unknown"]
        result = orient_umap1(embedding, slugs, legislators)
        # R mean = -1 < D mean = 2 -> flip
        assert result[0, 0] > 0


# ── TestBuildEmbeddingDf ────────────────────────────────────────────────────


class TestBuildEmbeddingDf:
    """Build polars DataFrame from UMAP embedding + metadata."""

    def test_columns_present(self, legislators_df: pl.DataFrame) -> None:
        """Output has UMAP1, UMAP2, and metadata columns."""
        embedding = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        slugs = ["rep_a", "rep_b", "rep_c"]
        df = build_embedding_df(embedding, slugs, legislators_df)
        assert "UMAP1" in df.columns
        assert "UMAP2" in df.columns
        assert "full_name" in df.columns
        assert "party" in df.columns

    def test_shape_matches(self, legislators_df: pl.DataFrame) -> None:
        """Number of rows matches number of legislators."""
        embedding = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        slugs = ["rep_a", "rep_b", "rep_c"]
        df = build_embedding_df(embedding, slugs, legislators_df)
        assert df.height == 3

    def test_metadata_joined(self, legislators_df: pl.DataFrame) -> None:
        """Legislator names are correctly joined."""
        embedding = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        slugs = ["rep_a", "rep_b", "rep_c"]
        df = build_embedding_df(embedding, slugs, legislators_df)
        row_a = df.filter(pl.col("legislator_slug") == "rep_a")
        assert row_a["full_name"][0] == "Alice Smith"
        assert row_a["party"][0] == "Republican"


# ── TestProcrusteSimilarity ─────────────────────────────────────────────────


class TestProcrusteSimilarity:
    """Procrustes similarity between 2D embeddings."""

    def test_identical_embeddings(self) -> None:
        """Identical embeddings give similarity = 1.0."""
        a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        sim = compute_procrustes_similarity(a, a.copy())
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_rotated_embedding(self) -> None:
        """90-degree rotation gives high similarity (same shape)."""
        a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # Rotate 90 degrees: (x,y) -> (-y, x)
        b = np.column_stack([-a[:, 1], a[:, 0]])
        sim = compute_procrustes_similarity(a, b)
        assert sim > 0.99

    def test_scaled_embedding(self) -> None:
        """Scaled embedding gives high similarity (Procrustes normalizes scale)."""
        a = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = a * 3.0  # Scale by 3x
        sim = compute_procrustes_similarity(a, b)
        assert sim > 0.99

    def test_random_embedding_low_similarity(self) -> None:
        """Random embeddings have lower similarity."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal((20, 2))
        b = rng.standard_normal((20, 2))
        sim = compute_procrustes_similarity(a, b)
        # Random embeddings should have low but not necessarily zero similarity
        assert sim < 0.9


# ── TestValidationCorrelations ──────────────────────────────────────────────


class TestValidationCorrelations:
    """Spearman correlations of UMAP1 vs PCA/IRT."""

    def test_no_upstream_returns_empty(self) -> None:
        """When no upstream data available, return empty dict."""
        embedding_df = pl.DataFrame(
            {"legislator_slug": ["a", "b"], "UMAP1": [1.0, 2.0], "UMAP2": [0.0, 0.0]}
        )
        result = compute_validation_correlations(embedding_df, None, None)
        assert result == {}

    def test_pca_correlation_computed(self) -> None:
        """Spearman rho is computed when PCA scores are available."""
        slugs = [f"rep_{i}" for i in range(10)]
        embedding_df = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "UMAP1": list(range(10)),
                "UMAP2": [0.0] * 10,
            }
        )
        pca_scores = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "PC1": list(range(10)),  # Perfect rank correlation
            }
        )
        result = compute_validation_correlations(embedding_df, pca_scores, None)
        assert "pca_pc1_spearman" in result
        assert result["pca_pc1_spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_too_few_shared_legislators(self) -> None:
        """With fewer than 5 shared legislators, skip correlation."""
        embedding_df = pl.DataFrame(
            {"legislator_slug": ["a", "b", "c"], "UMAP1": [1.0, 2.0, 3.0], "UMAP2": [0.0] * 3}
        )
        pca_scores = pl.DataFrame({"legislator_slug": ["a", "b", "d"], "PC1": [1.0, 2.0, 3.0]})
        result = compute_validation_correlations(embedding_df, pca_scores, None)
        # Only 2 shared slugs (a, b) < 5, so no correlation
        assert "pca_pc1_spearman" not in result


# ── TestComputeUmap (requires umap-learn) ───────────────────────────────────


class TestComputeUmap:
    """UMAP computation tests (require umap-learn installed)."""

    @pytest.fixture(autouse=True)
    def _require_umap(self) -> None:
        pytest.importorskip("umap")

    def test_output_shape(self) -> None:
        """UMAP output has shape (n_samples, 2)."""
        from analysis.umap_viz import compute_umap

        rng = np.random.default_rng(42)
        X = rng.random((20, 10))
        embedding = compute_umap(X, n_neighbors=5)
        assert embedding.shape == (20, 2)

    def test_deterministic_with_seed(self) -> None:
        """Same input + same seed = same output."""
        from analysis.umap_viz import compute_umap

        rng = np.random.default_rng(42)
        X = rng.random((20, 10))
        emb1 = compute_umap(X, n_neighbors=5, random_state=42)
        emb2 = compute_umap(X, n_neighbors=5, random_state=42)
        np.testing.assert_allclose(emb1, emb2, atol=1e-10)

    def test_cosine_metric_accepted(self) -> None:
        """UMAP runs without error with cosine metric."""
        from analysis.umap_viz import compute_umap

        rng = np.random.default_rng(42)
        X = rng.random((20, 10))
        embedding = compute_umap(X, n_neighbors=5, metric="cosine")
        assert embedding.shape == (20, 2)
        assert not np.isnan(embedding).any()
