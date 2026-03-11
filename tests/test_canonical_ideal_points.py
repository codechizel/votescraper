"""Tests for canonical ideal point routing.

Covers horseshoe detection, 1D/2D loading, routing logic, and file output.

Run: uv run pytest tests/test_canonical_ideal_points.py -v
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.canonical_ideal_points import (  # noqa: E402
    DIM1_ESS_THRESHOLD,
    DIM1_RHAT_THRESHOLD,
    HORSESHOE_DEM_WRONG_SIDE_FRAC,
    HORSESHOE_OVERLAP_FRAC,
    check_2d_convergence_quality,
    detect_horseshoe_from_ideal_points,
    load_1d_ideal_points,
    load_2d_dim1_ideal_points,
    route_canonical_ideal_points,
    write_canonical_ideal_points,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def balanced_1d() -> pl.DataFrame:
    """1D ideal points for a balanced chamber (no horseshoe)."""
    return pl.DataFrame(
        {
            "legislator_slug": ["r1", "r2", "r3", "d1", "d2", "d3"],
            "full_name": ["Rep A", "Rep B", "Rep C", "Dem A", "Dem B", "Dem C"],
            "party": ["Republican", "Republican", "Republican", "Democrat", "Democrat", "Democrat"],
            "xi_mean": [1.5, 0.8, 0.5, -1.2, -0.8, -0.5],
            "xi_sd": [0.2, 0.15, 0.18, 0.22, 0.14, 0.19],
            "xi_hdi_2.5": [1.1, 0.5, 0.14, -1.64, -1.08, -0.88],
            "xi_hdi_97.5": [1.9, 1.1, 0.86, -0.76, -0.52, -0.12],
            "district": [1, 2, 3, 4, 5, 6],
            "chamber": ["Senate"] * 6,
        }
    )


@pytest.fixture
def horseshoe_1d() -> pl.DataFrame:
    """1D ideal points exhibiting horseshoe: Democrats on wrong side."""
    return pl.DataFrame(
        {
            "legislator_slug": ["r1", "r2", "r3", "r4", "r5", "d1", "d2"],
            "full_name": ["R A", "R B", "R C", "R D", "R Rebel", "D A", "D B"],
            "party": [
                "Republican",
                "Republican",
                "Republican",
                "Republican",
                "Republican",
                "Democrat",
                "Democrat",
            ],
            # D1 and D2 placed at positive xi (wrong side) due to horseshoe
            "xi_mean": [1.5, 0.8, 0.5, 0.3, -1.0, 0.5, 0.3],
            "xi_sd": [0.2] * 7,
            "xi_hdi_2.5": [1.1, 0.4, 0.1, -0.1, -1.4, 0.1, -0.1],
            "xi_hdi_97.5": [1.9, 1.2, 0.9, 0.7, -0.6, 0.9, 0.7],
            "district": list(range(1, 8)),
            "chamber": ["Senate"] * 7,
        }
    )


@pytest.fixture
def ideal_2d_parquet(tmp_path) -> Path:
    """Create a mock 2D IRT output directory with parquet files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df = pl.DataFrame(
        {
            "legislator_slug": ["r1", "r2", "r3", "d1", "d2"],
            "full_name": ["R A", "R B", "R C", "D A", "D B"],
            "party": ["Republican", "Republican", "Republican", "Democrat", "Democrat"],
            "xi_dim1_mean": [1.5, 0.8, 0.5, -1.2, -0.8],
            "xi_dim1_hdi_3%": [1.1, 0.4, 0.1, -1.7, -1.2],
            "xi_dim1_hdi_97%": [1.9, 1.2, 0.9, -0.7, -0.4],
            "xi_dim2_mean": [0.1, -0.2, 0.3, -0.1, 0.2],
            "xi_dim2_hdi_3%": [-0.4, -0.7, -0.2, -0.6, -0.3],
            "xi_dim2_hdi_97%": [0.6, 0.3, 0.8, 0.4, 0.7],
        }
    )
    df.write_parquet(data_dir / "ideal_points_2d_senate.parquet")

    # Convergence summary
    summary = {
        "chambers": {
            "Senate": {
                "convergence": {"xi_rhat_max": 1.01, "xi_ess_min": 500.0},
            }
        }
    }
    (data_dir / "convergence_summary.json").write_text(json.dumps(summary))

    return tmp_path


@pytest.fixture
def ideal_1d_dir(tmp_path, balanced_1d) -> Path:
    """Create a mock 1D IRT output directory."""
    data_dir = tmp_path / "1d" / "data"
    data_dir.mkdir(parents=True)
    balanced_1d.write_parquet(data_dir / "ideal_points_senate.parquet")
    return tmp_path / "1d"


@pytest.fixture
def horseshoe_1d_dir(tmp_path, horseshoe_1d) -> Path:
    """Create a mock 1D IRT output directory with horseshoe data."""
    data_dir = tmp_path / "1d_hs" / "data"
    data_dir.mkdir(parents=True)
    horseshoe_1d.write_parquet(data_dir / "ideal_points_senate.parquet")
    return tmp_path / "1d_hs"


# ── detect_horseshoe_from_ideal_points ────────────────────────────────────────


class TestDetectHorseshoe:
    """Horseshoe detection from 1D ideal points."""

    def test_no_horseshoe_balanced(self, balanced_1d):
        result = detect_horseshoe_from_ideal_points(balanced_1d)
        assert result["detected"] is False

    def test_horseshoe_detected_wrong_side(self, horseshoe_1d):
        result = detect_horseshoe_from_ideal_points(horseshoe_1d)
        assert result["detected"] is True
        assert "dem_wrong_side" in result["reason"]

    def test_single_party_no_detection(self):
        df = pl.DataFrame(
            {
                "legislator_slug": ["r1", "r2"],
                "full_name": ["R A", "R B"],
                "party": ["Republican", "Republican"],
                "xi_mean": [1.0, 0.5],
            }
        )
        result = detect_horseshoe_from_ideal_points(df)
        assert result["detected"] is False
        assert result["reason"] == "single_party"

    def test_returns_metrics(self, balanced_1d):
        result = detect_horseshoe_from_ideal_points(balanced_1d)
        assert "dem_wrong_side_frac" in result["metrics"]
        assert "overlap_frac" in result["metrics"]
        assert "r_mean" in result["metrics"]
        assert "d_mean" in result["metrics"]


# ── load_1d_ideal_points ──────────────────────────────────────────────────────


class TestLoad1d:
    def test_loads_existing(self, ideal_1d_dir):
        df = load_1d_ideal_points(ideal_1d_dir, "Senate")
        assert df is not None
        assert df.height == 6

    def test_returns_none_missing(self, tmp_path):
        result = load_1d_ideal_points(tmp_path, "Senate")
        assert result is None


# ── load_2d_dim1_ideal_points ─────────────────────────────────────────────────


class TestLoad2dDim1:
    def test_loads_and_maps_columns(self, ideal_2d_parquet):
        df = load_2d_dim1_ideal_points(ideal_2d_parquet, "Senate")
        assert df is not None
        assert "xi_mean" in df.columns
        assert "xi_sd" in df.columns
        assert "xi_hdi_2.5" in df.columns
        assert "xi_hdi_97.5" in df.columns

    def test_xi_sd_computed(self, ideal_2d_parquet):
        df = load_2d_dim1_ideal_points(ideal_2d_parquet, "Senate")
        # xi_sd should be approximately (hdi_high - hdi_low) / 3.92
        for row in df.iter_rows(named=True):
            expected_sd = (row["xi_hdi_97.5"] - row["xi_hdi_2.5"]) / 3.92
            assert row["xi_sd"] == pytest.approx(expected_sd, rel=0.01)

    def test_returns_none_missing(self, tmp_path):
        result = load_2d_dim1_ideal_points(tmp_path, "Senate")
        assert result is None


# ── check_2d_convergence_quality ──────────────────────────────────────────────


class TestCheck2dConvergence:
    def test_passes_good_convergence(self, ideal_2d_parquet):
        assert check_2d_convergence_quality(ideal_2d_parquet, "Senate") is True

    def test_fails_bad_rhat(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        summary = {
            "chambers": {
                "Senate": {"convergence": {"xi_rhat_max": 1.10, "xi_ess_min": 500.0}}
            }
        }
        (data_dir / "convergence_summary.json").write_text(json.dumps(summary))
        assert check_2d_convergence_quality(tmp_path, "Senate") is False

    def test_fails_low_ess(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        summary = {
            "chambers": {
                "Senate": {"convergence": {"xi_rhat_max": 1.01, "xi_ess_min": 50.0}}
            }
        }
        (data_dir / "convergence_summary.json").write_text(json.dumps(summary))
        assert check_2d_convergence_quality(tmp_path, "Senate") is False

    def test_fails_missing_file(self, tmp_path):
        assert check_2d_convergence_quality(tmp_path, "Senate") is False


# ── route_canonical_ideal_points ──────────────────────────────────────────────


class TestRouting:
    def test_balanced_uses_1d(self, ideal_1d_dir, ideal_2d_parquet):
        ip, source, meta = route_canonical_ideal_points(ideal_1d_dir, ideal_2d_parquet, "Senate")
        assert source == "1d_irt"
        assert "source" in ip.columns
        assert ip["source"][0] == "1d_irt"

    def test_horseshoe_uses_2d(self, horseshoe_1d_dir, ideal_2d_parquet):
        ip, source, meta = route_canonical_ideal_points(
            horseshoe_1d_dir, ideal_2d_parquet, "Senate"
        )
        assert source == "2d_dim1"
        assert ip["source"][0] == "2d_dim1"
        assert "xi_mean" in ip.columns

    def test_horseshoe_fallback_no_2d(self, horseshoe_1d_dir, tmp_path):
        ip, source, meta = route_canonical_ideal_points(horseshoe_1d_dir, tmp_path, "Senate")
        assert source == "1d_irt"

    def test_no_1d_uses_2d(self, tmp_path, ideal_2d_parquet):
        empty_1d = tmp_path / "empty_1d"
        empty_1d.mkdir()
        ip, source, meta = route_canonical_ideal_points(empty_1d, ideal_2d_parquet, "Senate")
        assert source == "2d_dim1"

    def test_no_data_raises(self, tmp_path):
        empty1 = tmp_path / "e1"
        empty2 = tmp_path / "e2"
        empty1.mkdir()
        empty2.mkdir()
        with pytest.raises(FileNotFoundError):
            route_canonical_ideal_points(empty1, empty2, "Senate")


# ── write_canonical_ideal_points ──────────────────────────────────────────────


class TestWriteCanonical:
    def test_writes_parquet_and_manifest(self, ideal_1d_dir, ideal_2d_parquet, tmp_path):
        output_dir = tmp_path / "canonical"
        sources = write_canonical_ideal_points(
            ideal_1d_dir, ideal_2d_parquet, output_dir, chambers=["Senate"]
        )
        assert "Senate" in sources
        assert (output_dir / "canonical_ideal_points_senate.parquet").exists()
        assert (output_dir / "routing_manifest.json").exists()

    def test_manifest_contents(self, ideal_1d_dir, ideal_2d_parquet, tmp_path):
        output_dir = tmp_path / "canonical"
        write_canonical_ideal_points(
            ideal_1d_dir, ideal_2d_parquet, output_dir, chambers=["Senate"]
        )
        manifest = json.loads((output_dir / "routing_manifest.json").read_text())
        assert "sources" in manifest
        assert "thresholds" in manifest
        assert manifest["thresholds"]["horseshoe_dem_wrong_side_frac"] == HORSESHOE_DEM_WRONG_SIDE_FRAC

    def test_horseshoe_routes_to_2d(self, horseshoe_1d_dir, ideal_2d_parquet, tmp_path):
        output_dir = tmp_path / "canonical"
        sources = write_canonical_ideal_points(
            horseshoe_1d_dir, ideal_2d_parquet, output_dir, chambers=["Senate"]
        )
        assert sources["Senate"] == "2d_dim1"

        # Verify the parquet has standard columns
        df = pl.read_parquet(output_dir / "canonical_ideal_points_senate.parquet")
        assert "xi_mean" in df.columns
        assert "source" in df.columns
        assert df["source"][0] == "2d_dim1"


# ── Constants ─────────────────────────────────────────────────────────────────


class TestConstants:
    def test_thresholds_match_phase_05(self):
        assert HORSESHOE_DEM_WRONG_SIDE_FRAC == 0.20
        assert HORSESHOE_OVERLAP_FRAC == 0.30

    def test_convergence_thresholds(self):
        assert DIM1_RHAT_THRESHOLD == 1.05
        assert DIM1_ESS_THRESHOLD == 200
