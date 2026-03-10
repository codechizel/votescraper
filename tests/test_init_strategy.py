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


SLUGS = ["sen_a", "sen_b", "sen_c", "sen_d"]
IRT_SCORES = _make_irt_scores(SLUGS, [2.0, 1.0, -1.0, -2.0])
PCA_SCORES = _make_pca_scores(SLUGS, [5.0, 2.5, -2.5, -5.0], [1.0, -1.0, 0.5, -0.5])


# ── InitStrategy constants ────────────────────────────────────────────────────


class TestInitStrategyConstants:
    """Verify strategy constants are coherent and Django-ready."""

    def test_all_strategies_excludes_auto(self) -> None:
        from analysis.init_strategy import InitStrategy

        IS = InitStrategy
        assert IS.AUTO not in IS.ALL_STRATEGIES
        assert IS.AUTO == "auto"

    def test_all_strategies_are_kebab_case(self) -> None:
        from analysis.init_strategy import InitStrategy

        for s in InitStrategy.ALL_STRATEGIES:
            assert "-" in s, f"Strategy {s!r} should be kebab-case"
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

    def test_auto_prefers_irt(self) -> None:
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
