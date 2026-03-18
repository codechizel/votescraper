"""
Tests for MCA categorical vote matrix, fitting, orientation, contributions,
horseshoe detection, and PCA validation in analysis/03_mca/mca.py.

Run: uv run pytest tests/test_mca.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from factories import make_rollcalls, make_votes

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.mca import (
    ABSENT_LABEL,
    CATEGORY_COLORS,
    HORSESHOE_R2_THRESHOLD,
    MIN_VOTES,
    CONTESTED_THRESHOLD,
    PARTY_COLORS,
    build_categorical_vote_matrix,
    detect_horseshoe,
    extract_contributions,
    extract_cos2,
    extract_eigenvalues,
    fit_mca,
    orient_dim1,
    polars_to_pandas_categorical,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_categorical_matrix() -> pl.DataFrame:
    """Create a small categorical vote matrix for unit tests."""
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d"],
            "v1": ["Yea", "Yea", "Nay", "Nay"],
            "v2": ["Nay", "Yea", "Yea", "Nay"],
            "v3": ["Yea", "Nay", "Yea", "Absent"],
            "v4": ["Yea", "Yea", "Nay", "Yea"],
        }
    )


# ── build_categorical_vote_matrix() ─────────────────────────────────────────


class TestBuildCategoricalVoteMatrix:
    """Build categorical vote matrix from raw votes preserving Yea/Nay/Absent."""

    def test_returns_string_categories(self):
        """Matrix values should be categorical strings, not numeric."""
        votes = make_votes(n_votes=30, slug_column="legislator_slug", include_absence=True)
        rollcalls = make_rollcalls(n_votes=30)
        matrix, stats = build_categorical_vote_matrix(
            votes,
            rollcalls,
            "House",
            minority_threshold=0.0,
            min_votes=1,
        )
        vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
        # All values should be strings
        for col in vote_cols[:3]:
            values = set(matrix[col].unique().to_list())
            assert values <= {"Yea", "Nay", ABSENT_LABEL}, f"Unexpected values in {col}: {values}"

    def test_absent_categories_mapped(self):
        """Absence-type categories should be mapped to canonical Absent label."""
        votes = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_a", "rep_a"],
                "vote_id": ["v1", "v2", "v3"],
                "vote": ["Yea", "Absent and Not Voting", "Present and Passing"],
            }
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v1", "v2", "v3"],
                "chamber": ["House", "House", "House"],
            }
        )
        matrix, _ = build_categorical_vote_matrix(
            votes,
            rollcalls,
            "House",
            minority_threshold=0.0,
            min_votes=1,
        )
        vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
        all_values = set()
        for col in vote_cols:
            all_values.update(matrix[col].unique().to_list())
        assert "Absent and Not Voting" not in all_values
        assert "Present and Passing" not in all_values

    def test_filter_stats_populated(self):
        """Filter stats dict should contain all expected keys."""
        votes = make_votes(n_votes=30, slug_column="legislator_slug", include_absence=True)
        rollcalls = make_rollcalls(n_votes=30)
        _, stats = build_categorical_vote_matrix(
            votes,
            rollcalls,
            "House",
            minority_threshold=0.0,
            min_votes=1,
        )
        assert "n_votes_before" in stats
        assert "n_votes_after" in stats
        assert "n_legislators_before" in stats
        assert "n_legislators_after" in stats

    def test_minority_filter_removes_unanimous(self):
        """Near-unanimous votes should be filtered out."""
        # Create votes where one bill is 100% Yea
        rows = []
        for i in range(10):
            rows.append({"legislator_slug": f"rep_leg{i}", "vote_id": "v_unanimous", "vote": "Yea"})
            vote = "Yea" if i < 5 else "Nay"
            rows.append({"legislator_slug": f"rep_leg{i}", "vote_id": "v_contested", "vote": vote})
        votes = pl.DataFrame(rows)
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["v_unanimous", "v_contested"],
                "chamber": ["House", "House"],
            }
        )
        matrix, stats = build_categorical_vote_matrix(
            votes,
            rollcalls,
            "House",
            minority_threshold=0.1,
            min_votes=1,
        )
        vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
        assert "v_unanimous" not in vote_cols
        assert "v_contested" in vote_cols

    def test_min_votes_filter(self):
        """Legislators with too few substantive votes should be dropped."""
        rows = []
        # Legislator with many votes
        for j in range(25):
            vote = "Yea" if j % 2 == 0 else "Nay"
            rows.append(
                {
                    "legislator_slug": "rep_active",
                    "vote_id": f"v{j}",
                    "vote": vote,
                }
            )
        # Legislator with only absences
        for j in range(25):
            rows.append(
                {
                    "legislator_slug": "rep_absent",
                    "vote_id": f"v{j}",
                    "vote": "Absent and Not Voting",
                }
            )
        votes = pl.DataFrame(rows)
        rollcalls = pl.DataFrame(
            {
                "vote_id": [f"v{j}" for j in range(25)],
                "chamber": ["House"] * 25,
            }
        )
        matrix, _ = build_categorical_vote_matrix(
            votes,
            rollcalls,
            "House",
            minority_threshold=0.0,
            min_votes=5,
        )
        slugs = matrix["legislator_slug"].to_list()
        assert "rep_active" in slugs
        assert "rep_absent" not in slugs


# ── polars_to_pandas_categorical() ──────────────────────────────────────────


class TestPolarsToPandas:
    """Convert Polars categorical matrix to pandas for prince."""

    def test_returns_pandas_dataframe(self):
        """Output should be a pandas DataFrame."""
        matrix = _make_categorical_matrix()
        pdf = polars_to_pandas_categorical(matrix)
        assert isinstance(pdf, pd.DataFrame)

    def test_index_is_slugs(self):
        """Index should be legislator slugs."""
        matrix = _make_categorical_matrix()
        pdf = polars_to_pandas_categorical(matrix)
        assert list(pdf.index) == ["rep_a", "rep_b", "rep_c", "rep_d"]

    def test_no_slug_column(self):
        """legislator_slug should not appear as a column."""
        matrix = _make_categorical_matrix()
        pdf = polars_to_pandas_categorical(matrix)
        assert "legislator_slug" not in pdf.columns

    def test_string_typed(self):
        """All values should be string type for prince."""
        matrix = _make_categorical_matrix()
        pdf = polars_to_pandas_categorical(matrix)
        for col in pdf.columns:
            assert pd.api.types.is_string_dtype(pdf[col])


# ── fit_mca() ───────────────────────────────────────────────────────────────


class TestFitMCA:
    """Fit MCA on categorical data using prince."""

    def _make_pdf(self, n: int = 20, p: int = 10, seed: int = 42) -> pd.DataFrame:
        """Create a synthetic pandas DataFrame for MCA fitting."""
        rng = np.random.default_rng(seed)
        data = {}
        for j in range(p):
            data[f"v{j}"] = rng.choice(["Yea", "Nay", "Absent"], size=n)
        return pd.DataFrame(data)

    def test_returns_prince_mca(self):
        """fit_mca should return a prince.MCA object."""
        import prince

        pdf = self._make_pdf()
        mca = fit_mca(pdf, 3, "greenacre")
        assert isinstance(mca, prince.MCA)

    def test_component_capping_by_rows(self):
        """Components should be capped at min(n_rows-1, n_cols-1)."""
        pdf = self._make_pdf(n=4, p=10)
        mca = fit_mca(pdf, 10, "greenacre")
        # Should have at most 3 components (4 rows - 1)
        coords = mca.row_coordinates(pdf)
        assert coords.shape[1] <= 3

    def test_correction_none(self):
        """correction='none' should work without error."""
        pdf = self._make_pdf()
        mca = fit_mca(pdf, 3, "none")
        assert len(mca.eigenvalues_) >= 1

    def test_correction_benzecri(self):
        """correction='benzecri' should work without error."""
        pdf = self._make_pdf()
        mca = fit_mca(pdf, 3, "benzecri")
        assert len(mca.eigenvalues_) >= 1

    def test_row_coordinates_shape(self):
        """Row coordinates should have shape (n_observations, n_components)."""
        pdf = self._make_pdf(n=20, p=10)
        mca = fit_mca(pdf, 3, "greenacre")
        coords = mca.row_coordinates(pdf)
        assert coords.shape[0] == 20
        assert coords.shape[1] == 3


# ── orient_dim1() ───────────────────────────────────────────────────────────


class TestOrientDim1:
    """Flip Dim1 sign so Republicans have positive mean scores."""

    def test_flips_when_republicans_negative(self):
        """Republicans average negative → flip."""
        row_coords = pd.DataFrame(
            {
                0: [-1.0, -1.5, 1.0, 1.5],
                1: [0.5, 0.3, -0.2, -0.1],
            }
        )
        slugs = ["rep_a", "rep_b", "dem_x", "dem_y"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "dem_x", "dem_y"],
                "party": ["Republican", "Republican", "Democrat", "Democrat"],
            }
        )
        oriented, flipped = orient_dim1(row_coords, slugs, legislators)
        assert flipped
        assert oriented.iloc[0, 0] > 0  # Republican should now be positive

    def test_no_flip_when_republicans_positive(self):
        """Republicans already positive → no flip."""
        row_coords = pd.DataFrame(
            {
                0: [1.0, 1.5, -1.0, -1.5],
                1: [0.5, 0.3, -0.2, -0.1],
            }
        )
        slugs = ["rep_a", "rep_b", "dem_x", "dem_y"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "dem_x", "dem_y"],
                "party": ["Republican", "Republican", "Democrat", "Democrat"],
            }
        )
        oriented, flipped = orient_dim1(row_coords, slugs, legislators)
        assert not flipped

    def test_unknown_party_handled(self):
        """Legislators with unknown party should not crash."""
        row_coords = pd.DataFrame(
            {
                0: [1.0, -1.0],
                1: [0.5, -0.5],
            }
        )
        slugs = ["rep_a", "unknown_b"]
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a"],
                "party": ["Republican"],
            }
        )
        oriented, _ = orient_dim1(row_coords, slugs, legislators)
        assert oriented.shape == (2, 2)


# ── extract_eigenvalues() ───────────────────────────────────────────────────


class TestExtractEigenvalues:
    """Extract eigenvalue/inertia DataFrame from MCA."""

    def test_shape(self):
        """Should have one row per component."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_eigenvalues(mca, 3)
        assert df.height == 3

    def test_columns(self):
        """Should have dimension, eigenvalue, inertia_pct, cumulative_pct."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_eigenvalues(mca, 3)
        assert set(df.columns) == {"dimension", "eigenvalue", "inertia_pct", "cumulative_pct"}

    def test_cumulative_increases(self):
        """Cumulative inertia should be monotonically increasing."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_eigenvalues(mca, 3)
        cumulative = df["cumulative_pct"].to_list()
        for i in range(1, len(cumulative)):
            assert cumulative[i] >= cumulative[i - 1]


# ── extract_contributions() ─────────────────────────────────────────────────


class TestExtractContributions:
    """Extract category contributions to MCA dimensions."""

    def test_returns_dataframe(self):
        """Should return a polars DataFrame."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_contributions(mca, pdf, 3)
        assert isinstance(df, pl.DataFrame)

    def test_has_category_column(self):
        """Should have a 'category' column."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_contributions(mca, pdf, 3)
        assert "category" in df.columns

    def test_contributions_non_negative(self):
        """Contributions should be non-negative."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_contributions(mca, pdf, 3)
        for col in df.columns:
            if col.endswith("_ctr"):
                assert df[col].min() >= 0


# ── extract_cos2() ──────────────────────────────────────────────────────────


class TestExtractCos2:
    """Extract squared cosines for representation quality."""

    def test_returns_dataframe(self):
        """Should return a polars DataFrame."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_cos2(mca, pdf, 3)
        assert isinstance(df, pl.DataFrame)

    def test_has_slug_column(self):
        """Should have a 'legislator_slug' column."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        pdf.index = [f"rep_leg{i}" for i in range(20)]
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_cos2(mca, pdf, 3)
        assert "legislator_slug" in df.columns

    def test_cos2_bounded(self):
        """cos² values should be between 0 and 1."""
        rng = np.random.default_rng(42)
        pdf = pd.DataFrame({f"v{j}": rng.choice(["Yea", "Nay"], size=20) for j in range(10)})
        mca = fit_mca(pdf, 3, "greenacre")
        df = extract_cos2(mca, pdf, 3)
        for col in df.columns:
            if col.endswith("_cos2"):
                assert df[col].min() >= -0.001  # small float tolerance
                assert df[col].max() <= 1.001


# ── detect_horseshoe() ──────────────────────────────────────────────────────


class TestDetectHorseshoe:
    """Detect horseshoe/arch effect in MCA dimensions."""

    def test_detects_clear_horseshoe(self):
        """Quadratic Dim2 = f(Dim1) should be detected."""
        x = np.linspace(-3, 3, 50)
        y = x**2 + np.random.default_rng(42).normal(0, 0.1, 50)
        row_coords = pd.DataFrame({0: x, 1: y})
        result = detect_horseshoe(row_coords)
        assert result["detected"]
        assert result["r2"] > HORSESHOE_R2_THRESHOLD

    def test_no_horseshoe_random(self):
        """Random scatter should not be detected as horseshoe."""
        rng = np.random.default_rng(42)
        row_coords = pd.DataFrame({0: rng.normal(size=50), 1: rng.normal(size=50)})
        result = detect_horseshoe(row_coords)
        assert not result["detected"]

    def test_insufficient_data(self):
        """Fewer than 5 observations should return not detected."""
        row_coords = pd.DataFrame({0: [1, 2, 3], 1: [1, 4, 9]})
        result = detect_horseshoe(row_coords)
        assert not result["detected"]


# ── Constants ───────────────────────────────────────────────────────────────


class TestConstants:
    """Verify MCA constants are consistent with PCA conventions."""

    def test_party_colors_all_three(self):
        """Should have colors for Republican, Democrat, Independent."""
        assert "Republican" in PARTY_COLORS
        assert "Democrat" in PARTY_COLORS
        assert "Independent" in PARTY_COLORS

    def test_category_colors(self):
        """Should have colors for Yea, Nay, Absent."""
        assert "Yea" in CATEGORY_COLORS
        assert "Nay" in CATEGORY_COLORS
        assert "Absent" in CATEGORY_COLORS

    def test_minority_threshold_matches_pca(self):
        """MCA minority threshold should match PCA for consistency."""
        assert CONTESTED_THRESHOLD == 0.025

    def test_min_votes_matches_pca(self):
        """MCA min votes should match PCA for consistency."""
        assert MIN_VOTES == 20

    def test_absent_label(self):
        """Canonical absent label should be 'Absent'."""
        assert ABSENT_LABEL == "Absent"
