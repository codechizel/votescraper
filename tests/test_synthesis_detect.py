"""
Tests for synthesis detection logic in analysis/synthesis_detect.py.

Verifies maverick, bridge-builder, and metric paradox detection using
synthetic polars DataFrames with known properties. Edge cases: empty
frames, missing columns, all-unity > 0.95, single party, rank gap < 0.5.

Run: uv run pytest tests/test_synthesis_detect.py -v
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.synthesis_detect import (
    NotableLegislator,
    ParadoxCase,
    _majority_party,
    detect_annotation_slugs,
    detect_bridge_builder,
    detect_chamber_maverick,
    detect_metric_paradox,
    ideology_label,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _leg_df(overrides: list[dict] | None = None) -> pl.DataFrame:
    """Build a synthetic legislator DataFrame for detection tests.

    Default: 8 Republicans + 3 Democrats with plausible metrics.
    """
    base = [
        {"legislator_slug": "rep_a", "full_name": "Alice A", "party": "Republican",
         "district": "1", "unity_score": 0.95, "weighted_maverick": 0.03,
         "xi_mean": 1.5, "loyalty_rate": 0.90, "betweenness": 0.01},
        {"legislator_slug": "rep_b", "full_name": "Bob B", "party": "Republican",
         "district": "2", "unity_score": 0.70, "weighted_maverick": 0.20,
         "xi_mean": 0.3, "loyalty_rate": 0.75, "betweenness": 0.15},
        {"legislator_slug": "rep_c", "full_name": "Carol C", "party": "Republican",
         "district": "3", "unity_score": 0.98, "weighted_maverick": 0.01,
         "xi_mean": 2.0, "loyalty_rate": 0.95, "betweenness": 0.005},
        {"legislator_slug": "rep_d", "full_name": "Dave D", "party": "Republican",
         "district": "4", "unity_score": 0.92, "weighted_maverick": 0.05,
         "xi_mean": 1.0, "loyalty_rate": 0.88, "betweenness": 0.02},
        {"legislator_slug": "rep_e", "full_name": "Eve E", "party": "Republican",
         "district": "5", "unity_score": 0.60, "weighted_maverick": 0.30,
         "xi_mean": 3.5, "loyalty_rate": 0.40, "betweenness": 0.03},
        {"legislator_slug": "rep_f", "full_name": "Frank F", "party": "Republican",
         "district": "6", "unity_score": 0.88, "weighted_maverick": 0.08,
         "xi_mean": 1.2, "loyalty_rate": 0.85, "betweenness": 0.01},
        {"legislator_slug": "rep_g", "full_name": "Grace G", "party": "Republican",
         "district": "7", "unity_score": 0.93, "weighted_maverick": 0.04,
         "xi_mean": 1.8, "loyalty_rate": 0.92, "betweenness": 0.008},
        {"legislator_slug": "rep_h", "full_name": "Hank H", "party": "Republican",
         "district": "8", "unity_score": 0.85, "weighted_maverick": 0.10,
         "xi_mean": 0.8, "loyalty_rate": 0.80, "betweenness": 0.05},
        {"legislator_slug": "dem_x", "full_name": "Xena X", "party": "Democrat",
         "district": "20", "unity_score": 0.90, "weighted_maverick": 0.07,
         "xi_mean": -1.5, "loyalty_rate": 0.88, "betweenness": 0.02},
        {"legislator_slug": "dem_y", "full_name": "Yuri Y", "party": "Democrat",
         "district": "21", "unity_score": 0.85, "weighted_maverick": 0.10,
         "xi_mean": -0.5, "loyalty_rate": 0.80, "betweenness": 0.10},
        {"legislator_slug": "dem_z", "full_name": "Zara Z", "party": "Democrat",
         "district": "22", "unity_score": 0.95, "weighted_maverick": 0.03,
         "xi_mean": -2.0, "loyalty_rate": 0.93, "betweenness": 0.005},
    ]

    if overrides:
        for override in overrides:
            slug = override.get("legislator_slug")
            for item in base:
                if item["legislator_slug"] == slug:
                    item.update(override)
                    break
            else:
                base.append(override)

    return pl.DataFrame(base)


# ── ideology_label() ─────────────────────────────────────────────────────────


class TestIdeologyLabel:
    """Human-readable ideological direction labels."""

    def test_republican_positive(self):
        assert ideology_label("Republican", 1.5) == "conservative"

    def test_democrat_negative(self):
        assert ideology_label("Democrat", -1.5) == "liberal"

    def test_republican_negative_is_moderate(self):
        assert ideology_label("Republican", -0.5) == "moderate"

    def test_democrat_positive_is_moderate(self):
        assert ideology_label("Democrat", 0.5) == "moderate"


# ── _majority_party() ────────────────────────────────────────────────────────


class TestMajorityParty:
    """Determine majority party from legislator DataFrame."""

    def test_republican_majority(self):
        df = _leg_df()
        assert _majority_party(df) == "Republican"

    def test_empty_dataframe(self):
        df = pl.DataFrame({"party": []}, schema={"party": pl.Utf8})
        assert _majority_party(df) is None


# ── detect_chamber_maverick() ────────────────────────────────────────────────


class TestDetectChamberMaverick:
    """Maverick detection: lowest unity_score in party."""

    def test_detects_lowest_unity(self):
        df = _leg_df()
        result = detect_chamber_maverick(df, "Republican", "house")
        assert result is not None
        # rep_e has lowest unity (0.60)
        assert result.slug == "rep_e"
        assert result.reason == "maverick"
        assert result.role == "House Maverick"

    def test_returns_none_if_all_high_unity(self):
        """All unity > 0.95 means no maverick."""
        df = pl.DataFrame({
            "legislator_slug": ["a", "b", "c"],
            "full_name": ["A", "B", "C"],
            "party": ["Republican"] * 3,
            "district": ["1", "2", "3"],
            "unity_score": [0.96, 0.97, 0.98],
            "weighted_maverick": [0.02, 0.01, 0.01],
        })
        assert detect_chamber_maverick(df, "Republican", "house") is None

    def test_returns_none_if_missing_column(self):
        df = pl.DataFrame({"legislator_slug": ["a"], "party": ["Republican"]})
        assert detect_chamber_maverick(df, "Republican", "house") is None

    def test_returns_none_if_party_empty(self):
        df = _leg_df()
        assert detect_chamber_maverick(df, "Independent", "house") is None

    def test_tiebreaker_by_weighted_maverick(self):
        """When unity ties, highest weighted_maverick wins."""
        df = pl.DataFrame({
            "legislator_slug": ["a", "b"],
            "full_name": ["A", "B"],
            "party": ["Republican"] * 2,
            "district": ["1", "2"],
            "unity_score": [0.80, 0.80],
            "weighted_maverick": [0.10, 0.25],
        })
        result = detect_chamber_maverick(df, "Republican", "senate")
        assert result is not None
        assert result.slug == "b"

    def test_title_format(self):
        df = _leg_df()
        result = detect_chamber_maverick(df, "Republican", "house")
        assert result.title == "Eve E (R-5)"


# ── detect_bridge_builder() ──────────────────────────────────────────────────


class TestDetectBridgeBuilder:
    """Bridge-builder: highest betweenness near cross-party midpoint."""

    def test_detects_centrist_with_betweenness(self):
        df = _leg_df()
        result = detect_bridge_builder(df, "house")
        assert result is not None
        assert result.reason == "bridge"
        # dem_y (xi=-0.5) and rep_b (xi=0.3) are near midpoint
        # rep_b has betweenness=0.15, dem_y has 0.10
        assert result.slug == "rep_b"

    def test_returns_none_without_betweenness(self):
        df = pl.DataFrame({
            "legislator_slug": ["a"], "full_name": ["A"],
            "party": ["Republican"], "district": ["1"], "xi_mean": [1.0],
        })
        assert detect_bridge_builder(df, "house") is None

    def test_returns_none_single_party(self):
        df = pl.DataFrame({
            "legislator_slug": ["a", "b"],
            "full_name": ["A", "B"],
            "party": ["Republican", "Republican"],
            "district": ["1", "2"],
            "xi_mean": [1.0, 2.0],
            "betweenness": [0.1, 0.2],
        })
        assert detect_bridge_builder(df, "house") is None

    def test_subtitle_mentions_other_party(self):
        df = _leg_df()
        result = detect_bridge_builder(df, "house")
        assert result is not None
        assert "Democrat" in result.subtitle or "Republican" in result.subtitle


# ── detect_metric_paradox() ──────────────────────────────────────────────────


class TestDetectMetricParadox:
    """Metric paradox: largest gap between IRT rank and loyalty rank."""

    def test_detects_paradox(self):
        """rep_e has extreme xi (3.5) but low loyalty (0.40) — large rank gap."""
        df = _leg_df()
        result = detect_metric_paradox(df, "house")
        assert result is not None
        assert isinstance(result, ParadoxCase)
        assert result.slug == "rep_e"
        assert result.party == "Republican"

    def test_returns_none_if_missing_columns(self):
        df = pl.DataFrame({"legislator_slug": ["a"], "party": ["Republican"]})
        assert detect_metric_paradox(df, "house") is None

    def test_returns_none_if_too_few_legislators(self):
        """Needs >= 5 in majority party."""
        df = pl.DataFrame({
            "legislator_slug": ["a", "b", "c"],
            "full_name": ["A", "B", "C"],
            "party": ["Republican"] * 3,
            "district": ["1", "2", "3"],
            "xi_mean": [1.0, 2.0, 3.0],
            "loyalty_rate": [0.9, 0.8, 0.7],
        })
        assert detect_metric_paradox(df, "house") is None

    def test_returns_none_if_gap_below_threshold(self):
        """All aligned — rank gap < 0.5 everywhere."""
        n = 6
        df = pl.DataFrame({
            "legislator_slug": [f"rep_{i}" for i in range(n)],
            "full_name": [f"Leg {i}" for i in range(n)],
            "party": ["Republican"] * n,
            "district": [str(i) for i in range(n)],
            "xi_mean": [float(i) for i in range(n)],
            "loyalty_rate": [0.5 + i * 0.08 for i in range(n)],  # correlated with xi
        })
        result = detect_metric_paradox(df, "house")
        # With correlated xi and loyalty, gap should be small
        assert result is None

    def test_direction_rightward_for_republican(self):
        df = _leg_df()
        result = detect_metric_paradox(df, "house")
        assert result is not None
        assert result.direction == "rightward"


# ── detect_annotation_slugs() ────────────────────────────────────────────────


class TestDetectAnnotationSlugs:
    """Collect slugs for dashboard scatter annotations."""

    def test_includes_notable_slugs(self):
        df = _leg_df()
        notable = NotableLegislator(
            slug="rep_e", full_name="Eve E", party="Republican",
            district="5", chamber="house", role="Maverick",
            title="Eve E (R-5)", subtitle="...", reason="maverick",
        )
        slugs = detect_annotation_slugs(df, [notable])
        assert "rep_e" in slugs

    def test_includes_extreme_per_party(self):
        df = _leg_df()
        slugs = detect_annotation_slugs(df, [])
        # Most extreme R (highest xi) = rep_e (3.5), most extreme D (lowest xi) = dem_z (-2.0)
        assert "rep_e" in slugs or "dem_z" in slugs

    def test_caps_at_max_n(self):
        df = _leg_df()
        notables = [
            NotableLegislator(
                slug=f"rep_{c}", full_name=f"{c}", party="Republican",
                district="1", chamber="house", role="Test",
                title="T", subtitle="S", reason="test",
            )
            for c in "abcdefg"
        ]
        slugs = detect_annotation_slugs(df, notables, max_n=3)
        assert len(slugs) <= 3

    def test_deduplicates(self):
        df = _leg_df()
        notable = NotableLegislator(
            slug="rep_e", full_name="Eve E", party="Republican",
            district="5", chamber="house", role="Maverick",
            title="Eve E (R-5)", subtitle="...", reason="maverick",
        )
        # rep_e is also the most extreme R by xi_mean
        slugs = detect_annotation_slugs(df, [notable])
        assert slugs.count("rep_e") == 1
