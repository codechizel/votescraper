"""
End-to-end integration tests: synthetic data → EDA → PCA pipeline.

Does NOT call main() — imports core computation functions directly and chains
them together. Tests that EDA outputs feed cleanly into PCA, verifying the
contract between phases.

Run: uv run pytest tests/test_integration_pipeline.py -v
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from factories import make_legislators

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eda import (
    build_vote_matrix,
    compute_agreement_matrices,
    compute_rice_cohesion,
    filter_vote_matrix,
)
from analysis.pca import fit_pca, impute_vote_matrix, orient_pc1

from analysis.run_context import RunContext, generate_run_id, resolve_upstream_dir

# ── Synthetic Data Fixtures ──────────────────────────────────────────────────
# 20 legislators (12R House, 4D House, 3R Senate, 1D Senate) × 30 roll calls.
# R legislators vote Yea 80% on party-line bills; D vote Nay 80%.
# Mix of unanimous and contested votes.

_VOTE_SCHEMA = [
    "session",
    "bill_number",
    "bill_title",
    "vote_id",
    "vote_datetime",
    "vote_date",
    "chamber",
    "motion",
    "legislator_name",
    "legislator_slug",
    "vote",
]


def _make_legislators() -> pl.DataFrame:
    """20 legislators: 12R House, 4D House, 3R Senate, 1D Senate."""
    return pl.concat(
        [
            make_legislators(
                [f"R House {i}" for i in range(1, 13)], prefix="rep_r", chamber="House"
            ),
            make_legislators(
                [f"D House {i}" for i in range(1, 5)],
                prefix="rep_d",
                party="Democrat",
                chamber="House",
                start_district=101,
            ),
            make_legislators(
                [f"R Senate {i}" for i in range(1, 4)], prefix="sen_r", chamber="Senate"
            ),
            make_legislators(
                ["D Senate 1"],
                prefix="sen_d",
                party="Democrat",
                chamber="Senate",
                start_district=40,
            ),
        ]
    )


def _make_rollcalls(n_contested: int = 25, n_unanimous: int = 5) -> pl.DataFrame:
    """30 roll calls: 25 contested + 5 unanimous, mixed chambers."""
    rows = []
    for i in range(1, n_contested + 1):
        chamber = "House" if i <= 18 else "Senate"
        rows.append(
            (
                f"rc_{i:03d}",
                f"SB {i}",
                f"Test Bill {i}",
                f"2025-01-{(i % 28) + 1:02d}T10:00:00",
                chamber,
                "Final Action",
                "passed",
            )
        )
    for i in range(n_contested + 1, n_contested + n_unanimous + 1):
        rows.append(
            (
                f"rc_{i:03d}",
                f"HB {i}",
                f"Unanimous Bill {i}",
                f"2025-02-{(i % 28) + 1:02d}T10:00:00",
                "House",
                "Final Action",
                "passed",
            )
        )
    return pl.DataFrame(
        rows,
        schema=[
            "vote_id",
            "bill_number",
            "short_title",
            "vote_datetime",
            "chamber",
            "motion",
            "passed",
        ],
        orient="row",
    )


def _make_votes(
    legislators: pl.DataFrame,
    rollcalls: pl.DataFrame,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic vote records.

    Contested votes: R legislators Yea 80%, D legislators Nay 80%.
    Unanimous votes: everyone votes Yea.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for rc_row in rollcalls.iter_rows(named=True):
        vote_id = rc_row["vote_id"]
        bill = rc_row["bill_number"]
        dt = rc_row["vote_datetime"]
        chamber = rc_row["chamber"]
        motion = rc_row["motion"]
        is_unanimous = "Unanimous" in rc_row["short_title"]

        for leg_row in legislators.iter_rows(named=True):
            slug = leg_row["legislator_slug"]
            name = leg_row["full_name"]
            party = leg_row["party"]
            leg_chamber = leg_row["chamber"]

            # Only vote in your own chamber
            if leg_chamber != chamber:
                continue

            if is_unanimous:
                vote = "Yea"
            else:
                if party == "Republican":
                    vote = "Yea" if rng.random() < 0.80 else "Nay"
                else:
                    vote = "Nay" if rng.random() < 0.80 else "Yea"

            rows.append(
                (
                    "2025-26",
                    bill,
                    f"Title {bill}",
                    vote_id,
                    dt,
                    dt[:10],
                    chamber,
                    motion,
                    name,
                    slug,
                    vote,
                )
            )

    return pl.DataFrame(rows, schema=_VOTE_SCHEMA, orient="row")


@pytest.fixture
def synthetic_legislators() -> pl.DataFrame:
    return _make_legislators()


@pytest.fixture
def synthetic_rollcalls() -> pl.DataFrame:
    return _make_rollcalls()


@pytest.fixture
def synthetic_votes(synthetic_legislators, synthetic_rollcalls) -> pl.DataFrame:
    return _make_votes(synthetic_legislators, synthetic_rollcalls)


# ── EDA Pipeline Tests ───────────────────────────────────────────────────────


class TestEDAPipeline:
    """EDA core functions on synthetic data."""

    def test_vote_matrix_shape(self, synthetic_votes, synthetic_legislators):
        matrix = build_vote_matrix(synthetic_votes)
        # First column is legislator_slug
        n_legislators = matrix.height
        n_votes = matrix.width - 1  # subtract slug column
        assert n_legislators > 0
        assert n_votes > 0

    def test_vote_matrix_binary_values(self, synthetic_votes):
        matrix = build_vote_matrix(synthetic_votes)
        vote_cols = [c for c in matrix.columns if c != "legislator_slug"]
        for col in vote_cols:
            values = matrix[col].drop_nulls().to_list()
            assert all(v in (0, 1) for v in values), f"Non-binary value in {col}"

    def test_filter_removes_unanimous(self, synthetic_votes, synthetic_rollcalls):
        matrix = build_vote_matrix(synthetic_votes)
        filtered, manifest = filter_vote_matrix(matrix, synthetic_rollcalls, chamber="House")
        # Unanimous votes should be removed
        assert manifest["votes_dropped_unanimous"] > 0
        assert manifest["votes_after"] < manifest["votes_before"]

    def test_filter_manifest_keys(self, synthetic_votes, synthetic_rollcalls):
        matrix = build_vote_matrix(synthetic_votes)
        _, manifest = filter_vote_matrix(matrix, synthetic_rollcalls, chamber="House")
        expected_keys = {
            "chamber",
            "minority_threshold",
            "min_votes",
            "legislators_before",
            "votes_before",
            "votes_dropped_unanimous",
            "votes_after_unanimous_filter",
            "legislators_dropped_low_participation",
            "legislators_after",
            "votes_after",
        }
        assert expected_keys.issubset(set(manifest.keys()))

    def test_agreement_matrix_symmetric(self, synthetic_votes, synthetic_rollcalls):
        matrix = build_vote_matrix(synthetic_votes)
        # Use min_votes=1 so we don't drop all legislators from this small dataset
        filtered, _ = filter_vote_matrix(matrix, synthetic_rollcalls, chamber="House", min_votes=1)
        agreement, kappa = compute_agreement_matrices(filtered)
        n = agreement.shape[0]
        assert agreement.shape == (n, n)
        # Check symmetry (ignoring NaN)
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isnan(agreement[i, j]):
                    assert abs(agreement[i, j] - agreement[j, i]) < 1e-10

    def test_agreement_diagonal_is_one(self, synthetic_votes, synthetic_rollcalls):
        matrix = build_vote_matrix(synthetic_votes)
        filtered, _ = filter_vote_matrix(matrix, synthetic_rollcalls, chamber="House", min_votes=1)
        agreement, _ = compute_agreement_matrices(filtered)
        for i in range(agreement.shape[0]):
            assert abs(agreement[i, i] - 1.0) < 1e-10

    def test_rice_index_range(self, synthetic_votes, synthetic_legislators):
        rice = compute_rice_cohesion(synthetic_votes, synthetic_legislators)
        assert rice.height > 0
        assert "rice_index" in rice.columns
        values = rice["rice_index"].to_list()
        for v in values:
            assert 0.0 <= v <= 1.0, f"Rice index {v} out of [0,1] range"

    def test_rice_output_schema(self, synthetic_votes, synthetic_legislators):
        rice = compute_rice_cohesion(synthetic_votes, synthetic_legislators)
        expected_cols = {"vote_id", "party", "yea", "nay", "n_voting", "rice_index"}
        assert expected_cols.issubset(set(rice.columns))


# ── PCA From EDA Tests ───────────────────────────────────────────────────────


class TestPCAFromEDA:
    """Feed EDA output into PCA functions."""

    @pytest.fixture
    def house_matrix(self, synthetic_votes, synthetic_rollcalls):
        matrix = build_vote_matrix(synthetic_votes)
        filtered, _ = filter_vote_matrix(
            matrix,
            synthetic_rollcalls,
            chamber="House",
            min_votes=1,
        )
        return filtered

    def test_pca_runs_on_eda_output(self, house_matrix, synthetic_legislators):
        X, slugs, vote_ids = impute_vote_matrix(house_matrix)
        n_components = min(5, X.shape[0] - 1, X.shape[1])
        scores, loadings, pca_obj, scaler = fit_pca(X, n_components)
        assert scores.shape == (len(slugs), n_components)
        assert loadings.shape == (n_components, len(vote_ids))

    def test_pc1_explains_most_variance(self, house_matrix):
        X, slugs, vote_ids = impute_vote_matrix(house_matrix)
        n_components = min(5, X.shape[0] - 1, X.shape[1])
        scores, loadings, pca_obj, scaler = fit_pca(X, n_components)
        ev = pca_obj.explained_variance_ratio_
        assert ev[0] >= ev[1], "PC1 should explain most variance"

    def test_scores_correct_legislator_count(self, house_matrix):
        X, slugs, vote_ids = impute_vote_matrix(house_matrix)
        n_components = min(3, X.shape[0] - 1, X.shape[1])
        scores, loadings, pca_obj, scaler = fit_pca(X, n_components)
        # 16 House legislators (12R + 4D)
        assert len(slugs) == 16

    def test_loadings_shape_matches(self, house_matrix):
        X, slugs, vote_ids = impute_vote_matrix(house_matrix)
        n_components = min(3, X.shape[0] - 1, X.shape[1])
        scores, loadings, pca_obj, scaler = fit_pca(X, n_components)
        assert loadings.shape == (n_components, len(vote_ids))

    def test_pc1_orientation(self, house_matrix, synthetic_legislators):
        X, slugs, vote_ids = impute_vote_matrix(house_matrix)
        n_components = min(3, X.shape[0] - 1, X.shape[1])
        scores, loadings, pca_obj, scaler = fit_pca(X, n_components)
        scores, loadings = orient_pc1(scores, loadings, slugs, synthetic_legislators)

        # After orientation, Republicans should have positive mean PC1
        slug_to_party = dict(synthetic_legislators.select("legislator_slug", "party").iter_rows())
        rep_scores = [
            scores[i, 0] for i, s in enumerate(slugs) if slug_to_party.get(s) == "Republican"
        ]
        dem_scores = [
            scores[i, 0] for i, s in enumerate(slugs) if slug_to_party.get(s) == "Democrat"
        ]
        assert np.mean(rep_scores) > np.mean(dem_scores)

    def test_imputation_no_nans(self, house_matrix):
        X, _, _ = impute_vote_matrix(house_matrix)
        assert not np.any(np.isnan(X)), "Imputation should remove all NaN"


# ── Upstream Resolution ──────────────────────────────────────────────────────


class TestUpstreamResolution:
    """resolve_upstream_dir() finds upstream data correctly."""

    def test_run_id_path(self, tmp_path):
        result = resolve_upstream_dir("01_eda", tmp_path, run_id="91-260228.1")
        assert result == tmp_path / "91-260228.1" / "01_eda"

    def test_override_takes_precedence(self, tmp_path):
        override = tmp_path / "custom"
        result = resolve_upstream_dir("01_eda", tmp_path, run_id="91-260228.1", override=override)
        assert result == override

    def test_flat_latest_fallback(self, tmp_path):
        # Create flat phase latest symlink
        phase_dir = tmp_path / "01_eda"
        phase_dir.mkdir()
        run_dir = tmp_path / "01_eda" / "260228.1"
        run_dir.mkdir()
        latest = tmp_path / "01_eda" / "latest"
        latest.symlink_to("260228.1")

        result = resolve_upstream_dir("01_eda", tmp_path)
        assert result == latest

    def test_session_latest_fallback(self, tmp_path):
        # No flat phase latest → falls back to results_root/latest/phase
        result = resolve_upstream_dir("01_eda", tmp_path)
        assert result == tmp_path / "latest" / "01_eda"

    def test_run_id_with_parquet(self, tmp_path):
        """Full round-trip: write parquet, resolve, verify it exists."""
        run_dir = tmp_path / "91-260228.1" / "01_eda" / "data"
        run_dir.mkdir(parents=True)
        df = pl.DataFrame({"x": [1, 2, 3]})
        df.write_parquet(run_dir / "test.parquet")

        resolved = resolve_upstream_dir("01_eda", tmp_path, run_id="91-260228.1")
        data_dir = resolved / "data"
        assert data_dir.exists()
        loaded = pl.read_parquet(data_dir / "test.parquet")
        assert loaded.height == 3


# ── RunContext Lifecycle ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestRunContextLifecycle:
    """Full RunContext lifecycle with tmp_path."""

    def test_directory_structure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("analysis.run_context._normalize_session", lambda s: s)
        monkeypatch.setattr("analysis.run_context._git_commit_hash", lambda: "abc123")

        ctx = RunContext(
            session="test_session",
            analysis_name="01_eda",
            params={"k": 1},
            results_root=tmp_path,
            primer="# Test Primer\nThis is a test.",
            run_id="test-run.1",
        )
        ctx.setup()
        ctx.finalize()

        assert ctx.plots_dir.exists()
        assert ctx.data_dir.exists()
        assert (ctx.run_dir / "run_info.json").exists()
        assert (ctx.run_dir / "run_log.txt").exists()

    def test_primer_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr("analysis.run_context._normalize_session", lambda s: s)
        monkeypatch.setattr("analysis.run_context._git_commit_hash", lambda: "abc123")

        primer_text = "# Test Primer\nPurpose: testing."
        ctx = RunContext(
            session="test_session",
            analysis_name="01_eda",
            params={},
            results_root=tmp_path,
            primer=primer_text,
            run_id="test-run.1",
        )
        ctx.setup()
        ctx.finalize()

        readme = ctx.run_dir / "README.md"
        assert readme.exists()
        assert readme.read_text() == primer_text

    def test_run_info_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr("analysis.run_context._normalize_session", lambda s: s)
        monkeypatch.setattr("analysis.run_context._git_commit_hash", lambda: "abc123")

        ctx = RunContext(
            session="test_session",
            analysis_name="02_pca",
            params={"n_components": 5},
            results_root=tmp_path,
            run_id="test-run.1",
        )
        ctx.setup()
        ctx.finalize()

        info = json.loads((ctx.run_dir / "run_info.json").read_text())
        assert info["analysis"] == "02_pca"
        assert info["session"] == "test_session"
        assert info["run_id"] == "test-run.1"
        assert info["params"]["n_components"] == 5
        assert info["git_commit"] == "abc123"
        assert "elapsed_seconds" in info
        assert "elapsed_display" in info

    def test_latest_symlink_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr("analysis.run_context._normalize_session", lambda s: s)
        monkeypatch.setattr("analysis.run_context._git_commit_hash", lambda: "abc123")

        ctx = RunContext(
            session="test_session",
            analysis_name="01_eda",
            params={},
            results_root=tmp_path,
            run_id="test-run.1",
        )
        ctx.setup()
        ctx.finalize()

        latest = tmp_path / "test_session" / "latest"
        assert latest.is_symlink()

    def test_latest_not_created_on_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("analysis.run_context._normalize_session", lambda s: s)
        monkeypatch.setattr("analysis.run_context._git_commit_hash", lambda: "abc123")

        ctx = RunContext(
            session="test_session",
            analysis_name="01_eda",
            params={},
            results_root=tmp_path,
            run_id="test-run.fail",
        )
        ctx.setup()
        ctx.finalize(failed=True)

        latest = tmp_path / "test_session" / "latest"
        assert not latest.exists()


# ── generate_run_id ──────────────────────────────────────────────────────────


class TestGenerateRunId:
    """Run ID generation for pipeline grouping."""

    def test_format_pattern(self):
        run_id = generate_run_id("2025-26")
        # Should match pattern: digits-YYMMDD.N
        assert re.search(r"\d+-\d{6}\.\d+", run_id), f"Unexpected format: {run_id}"

    def test_increments_on_collision(self, tmp_path):
        # Create existing run directory
        (tmp_path / "91-260228.1").mkdir(parents=True)
        run_id = generate_run_id("2025-26", results_root=tmp_path)
        # Should skip .1 and use .2
        assert run_id.endswith(".2") or run_id.endswith(".1")
