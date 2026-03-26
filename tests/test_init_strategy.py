"""Tests for analysis/init_strategy.py — shared MCMC initialization strategies.

Run: uv run pytest tests/test_init_strategy.py -v
"""

import numpy as np
import polars as pl
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_irt_scores(slugs: list[str], values: list[float]) -> pl.DataFrame:
    """Build a minimal 1D IRT ideal points DataFrame."""
    return pl.DataFrame({"legislator_slug": slugs, "xi_mean": values})


def _make_pca_scores(slugs: list[str], pc1: list[float], pc2: list[float]) -> pl.DataFrame:
    """Build a minimal PCA scores DataFrame."""
    return pl.DataFrame({"legislator_slug": slugs, "PC1": pc1, "PC2": pc2})


def _make_2d_scores(slugs: list[str], dim1: list[float]) -> pl.DataFrame:
    """Build a minimal 2D IRT ideal points DataFrame."""
    return pl.DataFrame({"legislator_slug": slugs, "xi_dim1_mean": dim1})


def _make_canonical_scores(slugs: list[str], values: list[float]) -> pl.DataFrame:
    """Build a minimal canonical routing output DataFrame."""
    return pl.DataFrame({
        "legislator_slug": slugs,
        "xi_mean": values,
        "source": ["2d_dim1"] * len(slugs),
    })


SLUGS = ["sen_a", "sen_b", "sen_c", "sen_d"]
IRT_SCORES = _make_irt_scores(SLUGS, [2.0, 1.0, -1.0, -2.0])
PCA_SCORES = _make_pca_scores(SLUGS, [5.0, 2.5, -2.5, -5.0], [1.0, -1.0, 0.5, -0.5])
IRT_2D_SCORES = _make_2d_scores(SLUGS, [1.5, 0.5, -0.5, -1.5])
CANONICAL_SCORES = _make_canonical_scores(SLUGS, [1.8, 0.6, -0.6, -1.8])


# ── InitStrategy constants ────────────────────────────────────────────────────


class TestInitStrategyConstants:
    """Verify strategy constants are coherent and Django-ready."""

    def test_all_strategies_excludes_auto(self) -> None:
        from analysis.init_strategy import InitStrategy

        IS = InitStrategy
        assert IS.AUTO not in IS.ALL_STRATEGIES
        assert IS.AUTO == "auto"

    def test_all_strategies_are_lowercase(self) -> None:
        from analysis.init_strategy import InitStrategy

        for s in InitStrategy.ALL_STRATEGIES:
            assert s == s.lower(), f"Strategy {s!r} should be lowercase"

    def test_descriptions_cover_all_strategies(self) -> None:
        from analysis.init_strategy import InitStrategy

        for s in InitStrategy.ALL_STRATEGIES:
            assert s in InitStrategy.DESCRIPTIONS, f"Missing description for {s}"

    def test_references_cover_all_strategies(self) -> None:
        from analysis.init_strategy import InitStrategy

        for s in InitStrategy.ALL_STRATEGIES:
            assert s in InitStrategy.REFERENCES, f"Missing reference for {s}"

    def test_choices_tuple_django_ready(self) -> None:
        from analysis.init_strategy import InitStrategy

        choices = InitStrategy.CHOICES
        assert isinstance(choices, list)
        for db_val, label in choices:
            assert isinstance(db_val, str)
            assert isinstance(label, str)
        # AUTO should be first choice (default)
        assert choices[0][0] == "auto"

    def test_cli_choices_match_constants(self) -> None:
        """CLI choices list should match ALL_STRATEGIES + AUTO."""
        from analysis.init_strategy import InitStrategy

        expected = {InitStrategy.AUTO} | set(InitStrategy.ALL_STRATEGIES)
        actual = {db_val for db_val, _ in InitStrategy.CHOICES}
        assert actual == expected


# ── resolve_init_source ───────────────────────────────────────────────────────


class TestResolveInitSource:
    """Test the main strategy resolution function."""

    def test_irt_informed_returns_standardized(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="irt-informed", slugs=SLUGS, irt_scores=IRT_SCORES
        )
        assert vals.shape == (4,)
        assert strategy == "irt-informed"
        assert "1D IRT" in source
        # Standardized: mean ≈ 0, std ≈ 1
        assert abs(vals.mean()) < 1e-10
        assert abs(vals.std() - 1.0) < 1e-10

    def test_pca_informed_returns_standardized(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="pca-informed", slugs=SLUGS, pca_scores=PCA_SCORES
        )
        assert vals.shape == (4,)
        assert strategy == "pca-informed"
        assert "PCA PC1" in source
        assert abs(vals.mean()) < 1e-10

    def test_pca_pc2_column(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, _, source = resolve_init_source(
            strategy="pca-informed", slugs=SLUGS, pca_scores=PCA_SCORES, pca_column="PC2"
        )
        assert "PC2" in source
        assert vals.shape == (4,)

    def test_auto_prefers_irt_without_canonical(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="auto", slugs=SLUGS, irt_scores=IRT_SCORES, pca_scores=PCA_SCORES
        )
        assert strategy == "irt-informed"
        assert "1D IRT" in source

    def test_auto_falls_back_to_pca(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="auto", slugs=SLUGS, irt_scores=None, pca_scores=PCA_SCORES
        )
        assert strategy == "pca-informed"
        assert "PCA" in source

    def test_auto_pc2_uses_pca_not_irt(self) -> None:
        """When pca_column is PC2, auto should use PCA even if IRT is available."""
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="auto",
            slugs=SLUGS,
            irt_scores=IRT_SCORES,
            pca_scores=PCA_SCORES,
            pca_column="PC2",
        )
        # IRT only has PC1-equivalent; for PC2, auto should fall back to PCA
        assert strategy == "pca-informed"
        assert "PC2" in source

    def test_auto_no_data_returns_zeros(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="auto", slugs=SLUGS, irt_scores=None, pca_scores=None
        )
        assert strategy == "none"
        assert "zeros" in source
        np.testing.assert_array_equal(vals, np.zeros(4))

    def test_irt_informed_raises_without_data(self) -> None:
        from analysis.init_strategy import resolve_init_source

        with pytest.raises(ValueError, match="irt-informed.*requires"):
            resolve_init_source(strategy="irt-informed", slugs=SLUGS, irt_scores=None)

    def test_pca_informed_raises_without_data(self) -> None:
        from analysis.init_strategy import resolve_init_source

        with pytest.raises(ValueError, match="pca-informed.*requires"):
            resolve_init_source(strategy="pca-informed", slugs=SLUGS, pca_scores=None)

    def test_unknown_strategy_raises(self) -> None:
        from analysis.init_strategy import resolve_init_source

        with pytest.raises(ValueError, match="Unknown init strategy"):
            resolve_init_source(strategy="magic", slugs=SLUGS)

    def test_missing_slugs_get_zero(self) -> None:
        """Legislators not in upstream data get 0.0 before standardization."""
        from analysis.init_strategy import resolve_init_source

        extra_slugs = SLUGS + ["sen_unknown"]
        vals, _, _ = resolve_init_source(
            strategy="irt-informed", slugs=extra_slugs, irt_scores=IRT_SCORES
        )
        assert vals.shape == (5,)

    def test_matched_count_in_source(self) -> None:
        from analysis.init_strategy import resolve_init_source

        _, _, source = resolve_init_source(
            strategy="irt-informed", slugs=SLUGS, irt_scores=IRT_SCORES
        )
        assert "4/4 matched" in source

    def test_partial_match_count(self) -> None:
        from analysis.init_strategy import resolve_init_source

        extra_slugs = SLUGS + ["sen_unknown"]
        _, _, source = resolve_init_source(
            strategy="irt-informed", slugs=extra_slugs, irt_scores=IRT_SCORES
        )
        assert "4/5 matched" in source

    def test_order_preserved(self) -> None:
        """Values should follow slug order, not upstream DataFrame order."""
        from analysis.init_strategy import resolve_init_source

        reversed_slugs = list(reversed(SLUGS))
        vals_fwd, _, _ = resolve_init_source(
            strategy="irt-informed", slugs=SLUGS, irt_scores=IRT_SCORES
        )
        vals_rev, _, _ = resolve_init_source(
            strategy="irt-informed", slugs=reversed_slugs, irt_scores=IRT_SCORES
        )
        np.testing.assert_array_almost_equal(vals_fwd, vals_rev[::-1])

    def test_constant_values_not_divided_by_zero(self) -> None:
        """If all upstream scores are identical, std=0 → return as-is."""
        from analysis.init_strategy import resolve_init_source

        uniform_irt = _make_irt_scores(SLUGS, [1.0, 1.0, 1.0, 1.0])
        vals, _, _ = resolve_init_source(
            strategy="irt-informed", slugs=SLUGS, irt_scores=uniform_irt
        )
        # Should not raise; values should be unchanged (all same)
        assert vals.shape == (4,)


# ── 2d-dim1 strategy ─────────────────────────────────────────────────────────


class TestResolve2dDim1:
    """Test the 2d-dim1 strategy for iterative refinement."""

    def test_2d_dim1_returns_standardized(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="2d-dim1", slugs=SLUGS, irt_2d_scores=IRT_2D_SCORES
        )
        assert vals.shape == (4,)
        assert strategy == "2d-dim1"
        assert "2D IRT Dim 1" in source
        assert abs(vals.mean()) < 1e-10
        assert abs(vals.std() - 1.0) < 1e-10

    def test_2d_dim1_raises_without_data(self) -> None:
        from analysis.init_strategy import resolve_init_source

        with pytest.raises(ValueError, match="2d-dim1.*requires"):
            resolve_init_source(strategy="2d-dim1", slugs=SLUGS, irt_2d_scores=None)

    def test_2d_dim1_preserves_order(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals_fwd, _, _ = resolve_init_source(
            strategy="2d-dim1", slugs=SLUGS, irt_2d_scores=IRT_2D_SCORES
        )
        vals_rev, _, _ = resolve_init_source(
            strategy="2d-dim1", slugs=list(reversed(SLUGS)), irt_2d_scores=IRT_2D_SCORES
        )
        np.testing.assert_array_almost_equal(vals_fwd, vals_rev[::-1])

    def test_2d_dim1_match_count(self) -> None:
        from analysis.init_strategy import resolve_init_source

        _, _, source = resolve_init_source(
            strategy="2d-dim1", slugs=SLUGS, irt_2d_scores=IRT_2D_SCORES
        )
        assert "4/4 matched" in source

    def test_auto_does_not_select_2d_dim1(self) -> None:
        """Auto should never select 2d-dim1 — it's an explicit user choice."""
        from analysis.init_strategy import resolve_init_source

        _, strategy, _ = resolve_init_source(
            strategy="auto",
            slugs=SLUGS,
            irt_scores=None,
            pca_scores=PCA_SCORES,
            irt_2d_scores=IRT_2D_SCORES,
        )
        assert strategy != "2d-dim1"


# ── canonical strategy (ADR-0111) ───────────────────────────────────────────


class TestResolveCanonical:
    """Test the canonical strategy for horseshoe-corrected initialization."""

    def test_canonical_returns_standardized(self) -> None:
        from analysis.init_strategy import resolve_init_source

        vals, strategy, source = resolve_init_source(
            strategy="canonical", slugs=SLUGS, canonical_scores=CANONICAL_SCORES
        )
        assert vals.shape == (4,)
        assert strategy == "canonical"
        assert "canonical" in source
        assert abs(vals.mean()) < 1e-10
        assert abs(vals.std() - 1.0) < 1e-10

    def test_canonical_raises_without_data(self) -> None:
        from analysis.init_strategy import resolve_init_source

        with pytest.raises(ValueError, match="canonical.*requires"):
            resolve_init_source(strategy="canonical", slugs=SLUGS, canonical_scores=None)

    def test_canonical_includes_source_type(self) -> None:
        from analysis.init_strategy import resolve_init_source

        _, _, source = resolve_init_source(
            strategy="canonical", slugs=SLUGS, canonical_scores=CANONICAL_SCORES
        )
        assert "2d_dim1" in source  # source column value

    def test_canonical_match_count(self) -> None:
        from analysis.init_strategy import resolve_init_source

        _, _, source = resolve_init_source(
            strategy="canonical", slugs=SLUGS, canonical_scores=CANONICAL_SCORES
        )
        assert "4/4 matched" in source

    def test_auto_prefers_canonical_over_irt(self) -> None:
        """Auto with canonical available should prefer canonical."""
        from analysis.init_strategy import resolve_init_source

        _, strategy, _ = resolve_init_source(
            strategy="auto",
            slugs=SLUGS,
            irt_scores=IRT_SCORES,
            pca_scores=PCA_SCORES,
            canonical_scores=CANONICAL_SCORES,
        )
        assert strategy == "canonical"

    def test_auto_falls_back_to_irt_without_canonical(self) -> None:
        """Auto without canonical should fall back to IRT."""
        from analysis.init_strategy import resolve_init_source

        _, strategy, _ = resolve_init_source(
            strategy="auto",
            slugs=SLUGS,
            irt_scores=IRT_SCORES,
            pca_scores=PCA_SCORES,
            canonical_scores=None,
        )
        assert strategy == "irt-informed"


# ── load_2d_scores ────────────────────────────────────────────────────────────


class TestLoad2dScores:
    """Test 2D IRT score loading utility."""

    def test_returns_none_for_missing(self, tmp_path) -> None:
        from analysis.init_strategy import load_2d_scores

        result = load_2d_scores(tmp_path, "senate")
        assert result is None

    def test_loads_parquet(self, tmp_path) -> None:
        from analysis.init_strategy import load_2d_scores

        IRT_2D_SCORES.write_parquet(tmp_path / "ideal_points_2d_senate.parquet")
        result = load_2d_scores(tmp_path, "senate")
        assert result is not None
        assert result.height == 4
        assert "xi_dim1_mean" in result.columns


# ── build_init_rationale ──────────────────────────────────────────────────────


class TestBuildInitRationale:
    """Test rationale generation for logging/reports."""

    def test_selected_marked(self) -> None:
        from analysis.init_strategy import InitStrategy, build_init_rationale

        rationale = build_init_rationale(
            irt_available=True, pca_available=True, selected="irt-informed", auto=True
        )
        assert rationale[InitStrategy.IRT_INFORMED].startswith("SELECTED (auto)")
        assert rationale[InitStrategy.PCA_INFORMED].startswith("Not selected")

    def test_user_override_marked(self) -> None:
        from analysis.init_strategy import InitStrategy, build_init_rationale

        rationale = build_init_rationale(
            irt_available=True, pca_available=True, selected="pca-informed", auto=False
        )
        assert rationale[InitStrategy.PCA_INFORMED].startswith("SELECTED (user override)")
        assert rationale[InitStrategy.IRT_INFORMED].startswith("Not selected")

    def test_unavailable_irt_noted(self) -> None:
        from analysis.init_strategy import InitStrategy, build_init_rationale

        rationale = build_init_rationale(
            irt_available=False, pca_available=True, selected="pca-informed", auto=True
        )
        assert "not found" in rationale[InitStrategy.IRT_INFORMED]

    def test_all_strategies_covered(self) -> None:
        from analysis.init_strategy import InitStrategy, build_init_rationale

        rationale = build_init_rationale(
            irt_available=True, pca_available=True, selected="irt-informed"
        )
        for s in InitStrategy.ALL_STRATEGIES:
            assert s in rationale


# ── load_canonical_scores ────────────────────────────────────────────────────


class TestLoadCanonicalScores:
    """Test canonical routing score loading utility."""

    def test_returns_none_for_missing(self, tmp_path) -> None:
        from analysis.init_strategy import load_canonical_scores

        result = load_canonical_scores(tmp_path, "senate")
        assert result is None

    def test_loads_parquet(self, tmp_path) -> None:
        from analysis.init_strategy import load_canonical_scores

        CANONICAL_SCORES.write_parquet(tmp_path / "canonical_ideal_points_senate.parquet")
        result = load_canonical_scores(tmp_path, "senate")
        assert result is not None
        assert result.height == 4
        assert "xi_mean" in result.columns
        assert "source" in result.columns


# ── load_irt_scores ───────────────────────────────────────────────────────────


class TestLoadIrtScores:
    """Test IRT score loading utility."""

    def test_returns_none_for_missing(self, tmp_path) -> None:
        from analysis.init_strategy import load_irt_scores

        result = load_irt_scores(tmp_path, "senate")
        assert result is None

    def test_loads_parquet(self, tmp_path) -> None:
        from analysis.init_strategy import load_irt_scores

        IRT_SCORES.write_parquet(tmp_path / "ideal_points_senate.parquet")
        result = load_irt_scores(tmp_path, "senate")
        assert result is not None
        assert result.height == 4
        assert "xi_mean" in result.columns


# ── PCA override ─────────────────────────────────────────────────────────────


class TestPcaOverride:
    """Tests for manual PCA dimension override loading."""

    def test_load_override_known_session(self) -> None:
        """79th Senate should return PC2 from the production override file."""
        from analysis.init_strategy import load_pca_override

        result = load_pca_override("79th_2001-2002", "senate")
        assert result == "PC2"

    def test_load_override_known_session_title_case(self) -> None:
        """Chamber normalization: 'Senate' should match '79th_Senate' key."""
        from analysis.init_strategy import load_pca_override

        result = load_pca_override("79th_2001-2002", "Senate")
        assert result == "PC2"

    def test_load_override_unknown_session(self) -> None:
        """91st Senate has no override — should return None."""
        from analysis.init_strategy import load_pca_override

        result = load_pca_override("91st_2025-2026", "senate")
        assert result is None

    def test_load_override_house_not_overridden(self) -> None:
        """79th House has no axis instability — should return None."""
        from analysis.init_strategy import load_pca_override

        result = load_pca_override("79th_2001-2002", "house")
        assert result is None

    def test_load_override_none_session(self) -> None:
        from analysis.init_strategy import load_pca_override

        assert load_pca_override(None, "senate") is None

    def test_load_override_none_chamber(self) -> None:
        from analysis.init_strategy import load_pca_override

        assert load_pca_override("79th_2001-2002", None) is None

    def test_load_override_all_known_entries(self) -> None:
        """Verify all 8 entries in pca_overrides.yaml are loadable."""
        from analysis.init_strategy import load_pca_override

        expected = [
            ("78th_1999-2000", "senate", "PC2"),
            ("79th_2001-2002", "senate", "PC2"),
            ("80th_2003-2004", "senate", "PC2"),
            ("81st_2005-2006", "senate", "PC2"),
            ("82nd_2007-2008", "senate", "PC2"),
            ("83rd_2009-2010", "senate", "PC2"),
            ("84th_2011-2012", "senate", "PC2"),
            ("88th_2019-2020", "senate", "PC2"),
        ]
        for session, chamber, expected_pc in expected:
            result = load_pca_override(session, chamber)
            assert result == expected_pc, f"Expected {expected_pc} for {session} {chamber}, got {result}"

    def test_resolve_init_source_uses_override(self) -> None:
        """When override exists, resolve_init_source should use the override PC."""
        from analysis.init_strategy import resolve_init_source

        pca = pl.DataFrame({
            "legislator_slug": ["r1", "r2", "d1", "d2"],
            "party": ["Republican", "Republican", "Democrat", "Democrat"],
            "PC1": [0.5, 0.3, -0.2, -0.4],
            "PC2": [2.0, 1.5, -1.5, -2.0],
        })
        vals, _, source = resolve_init_source(
            strategy="pca-informed",
            slugs=["r1", "r2", "d1", "d2"],
            pca_scores=pca,
            pca_column="PC1",
            session="79th_2001-2002",
            chamber="senate",
        )
        assert "PC2" in source
        assert vals is not None
        assert len(vals) == 4

    def test_resolve_override_skipped_when_explicit_pc2(self) -> None:
        """When caller explicitly requests PC2, override should not apply."""
        from analysis.init_strategy import resolve_init_source

        pca = pl.DataFrame({
            "legislator_slug": ["r1", "r2", "d1", "d2"],
            "party": ["Republican", "Republican", "Democrat", "Democrat"],
            "PC1": [0.5, 0.3, -0.2, -0.4],
            "PC2": [2.0, 1.5, -1.5, -2.0],
        })
        # Explicit pca_column="PC2" — override should not fire
        vals, _, source = resolve_init_source(
            strategy="pca-informed",
            slugs=["r1", "r2", "d1", "d2"],
            pca_scores=pca,
            pca_column="PC2",
            session="79th_2001-2002",
            chamber="senate",
        )
        # Source should say PC2 directly (not "swapped from PC1")
        assert "PC2" in source
        assert "swapped" not in source
