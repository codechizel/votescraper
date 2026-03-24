"""Tests for Phase 28 — Common Space Ideal Points."""

import numpy as np
import polars as pl
from analysis.common_space_data import (
    PARTY_D_MIN,
    build_global_roster,
    compute_bridge_matrix,
    compute_polarization_trajectory,
    compute_quality_gates,
    solve_simultaneous_alignment,
    transform_scores,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Simple normalize for tests."""
    return name.strip().lower()


def _make_scores(
    *,
    sessions: list[str],
    n_per_session: int = 20,
    n_bridge: int = 15,
    scale_a: dict[str, float] | None = None,
    shift_b: dict[str, float] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, pl.DataFrame]]:
    """Generate synthetic canonical scores with known affine relationships.

    Bridge legislators are the first `n_bridge` legislators, shared across
    all sessions. Remaining legislators are session-specific.
    """
    rng = np.random.default_rng(seed)
    scale_a = scale_a or {}
    shift_b = shift_b or {}

    all_scores: dict[str, dict[str, pl.DataFrame]] = {}
    for session in sessions:
        A = scale_a.get(session, 1.0)
        B = shift_b.get(session, 0.0)

        names = []
        slugs = []
        parties = []
        xi_vals = []

        for i in range(n_per_session):
            if i < n_bridge:
                name = f"Legislator {i}"
            else:
                name = f"Legislator {session}_{i}"
            names.append(name)
            slugs.append(name.replace(" ", "_").lower())

            party = "Republican" if i % 3 != 0 else "Democrat"
            parties.append(party)

            # True score on reference scale
            true_xi = -2.0 + 4.0 * i / n_per_session
            if party == "Republican":
                true_xi += 1.0
            else:
                true_xi -= 1.0

            # Transform to session-specific scale with noise
            session_xi = (true_xi - B) / A if A != 0 else true_xi
            session_xi += rng.normal(0, 0.05)
            xi_vals.append(session_xi)

        df = pl.DataFrame(
            {
                "legislator_slug": slugs,
                "full_name": names,
                "party": parties,
                "xi_mean": xi_vals,
            }
        )
        all_scores[session] = {"House": df}

    return all_scores


# ---------------------------------------------------------------------------
# TestGlobalRoster
# ---------------------------------------------------------------------------


class TestGlobalRoster:
    def test_basic_construction(self):
        scores = _make_scores(sessions=["A", "B"], n_per_session=10)
        roster = build_global_roster(scores, _normalize)
        assert roster.height == 20
        assert set(roster.columns) >= {"name_norm", "session", "chamber", "xi_canonical"}

    def test_bridge_detection(self):
        scores = _make_scores(sessions=["A", "B"], n_per_session=10, n_bridge=5)
        roster = build_global_roster(scores, _normalize)
        # Bridge legislators appear in both sessions
        bridge_names = (
            roster.group_by("name_norm")
            .agg(pl.col("session").n_unique().alias("n"))
            .filter(pl.col("n") >= 2)
        )
        assert bridge_names.height == 5

    def test_unique_per_session(self):
        scores = _make_scores(sessions=["A", "B", "C"], n_per_session=15, n_bridge=10)
        roster = build_global_roster(scores, _normalize)
        # No duplicate (name_norm, session) pairs
        deduped = roster.unique(subset=["name_norm", "session"])
        assert deduped.height == roster.height


# ---------------------------------------------------------------------------
# TestBridgeMatrix
# ---------------------------------------------------------------------------


class TestBridgeMatrix:
    def test_pairwise_counts(self):
        scores = _make_scores(sessions=["A", "B", "C"], n_per_session=20, n_bridge=15)
        roster = build_global_roster(scores, _normalize)
        matrix = compute_bridge_matrix(roster, ["A", "B", "C"])
        assert matrix.height == 3  # 3 pairs: A-B, A-C, B-C

    def test_bridge_count_correct(self):
        scores = _make_scores(sessions=["A", "B"], n_per_session=10, n_bridge=7)
        roster = build_global_roster(scores, _normalize)
        matrix = compute_bridge_matrix(roster, ["A", "B"])
        assert matrix.height == 1
        assert matrix["n_bridges"][0] == 7


# ---------------------------------------------------------------------------
# TestSimultaneousAlignment
# ---------------------------------------------------------------------------


class TestSimultaneousAlignment:
    def test_reference_is_identity(self):
        scores = _make_scores(sessions=["A", "B"], n_per_session=30, n_bridge=25)
        roster = build_global_roster(scores, _normalize)
        coefs = solve_simultaneous_alignment(roster, ["A", "B"], "House", "B")
        A_ref, B_ref = coefs["B"]
        assert A_ref == 1.0
        assert B_ref == 0.0

    def test_recovers_known_transform(self):
        """When scores are related by A=2.0, B=1.0, solver should recover these."""
        scores = _make_scores(
            sessions=["ref", "shifted"],
            n_per_session=50,
            n_bridge=40,
            scale_a={"ref": 1.0, "shifted": 2.0},
            shift_b={"ref": 0.0, "shifted": 1.0},
        )
        roster = build_global_roster(scores, _normalize)
        coefs = solve_simultaneous_alignment(
            roster,
            ["ref", "shifted"],
            "House",
            "ref",
            trim_pct=0,
        )
        A, B = coefs["shifted"]
        # Should approximately recover A≈2, B≈1
        assert abs(A - 2.0) < 0.3, f"Expected A≈2.0, got {A}"
        assert abs(B - 1.0) < 0.5, f"Expected B≈1.0, got {B}"

    def test_single_session_returns_identity(self):
        scores = _make_scores(sessions=["A"], n_per_session=10)
        roster = build_global_roster(scores, _normalize)
        coefs = solve_simultaneous_alignment(roster, ["A"], "House", "A")
        assert coefs["A"] == (1.0, 0.0)

    def test_three_session_chain(self):
        """Three sessions with shared bridges should all get aligned."""
        scores = _make_scores(
            sessions=["A", "B", "C"],
            n_per_session=30,
            n_bridge=20,
        )
        roster = build_global_roster(scores, _normalize)
        coefs = solve_simultaneous_alignment(roster, ["A", "B", "C"], "House", "C")
        # All sessions should have valid (non-zero) A
        for s in ["A", "B", "C"]:
            A, _ = coefs[s]
            assert A != 0, f"Session {s} has A=0"


# ---------------------------------------------------------------------------
# TestTransformScores
# ---------------------------------------------------------------------------


class TestTransformScores:
    def test_reference_unchanged(self):
        scores = _make_scores(sessions=["A", "B"], n_per_session=10, n_bridge=8)
        roster = build_global_roster(scores, _normalize)
        coefs = {"A": (1.5, 0.5), "B": (1.0, 0.0)}
        transformed = transform_scores(roster, coefs)

        ref_rows = transformed.filter(pl.col("session") == "B")
        for row in ref_rows.iter_rows(named=True):
            assert abs(row["xi_common"] - row["xi_canonical"]) < 1e-10

    def test_affine_applied_correctly(self):
        roster = pl.DataFrame(
            {
                "name_norm": ["a"],
                "legislator_slug": ["a"],
                "full_name": ["A"],
                "party": ["Republican"],
                "session": ["X"],
                "chamber": ["House"],
                "xi_canonical": [2.0],
            }
        )
        transformed = transform_scores(roster, {"X": (3.0, -1.0)})
        assert abs(transformed["xi_common"][0] - 5.0) < 1e-10  # 3*2 + (-1) = 5

    def test_bootstrap_cis_widen(self):
        roster = pl.DataFrame(
            {
                "name_norm": ["a"],
                "legislator_slug": ["a"],
                "full_name": ["A"],
                "party": ["Republican"],
                "session": ["X"],
                "chamber": ["House"],
                "xi_canonical": [1.0],
            }
        )
        cis = {"X": (0.8, 1.2, -0.5, 0.5)}
        transformed = transform_scores(roster, {"X": (1.0, 0.0)}, cis)
        row = transformed.row(0, named=True)
        assert row["xi_common_lo"] < row["xi_common"]
        assert row["xi_common_hi"] > row["xi_common"]


# ---------------------------------------------------------------------------
# TestQualityGates
# ---------------------------------------------------------------------------


class TestQualityGates:
    def _make_transformed(self, r_mean: float, d_mean: float) -> pl.DataFrame:
        rows = []
        rng = np.random.default_rng(42)
        for i in range(30):
            rows.append(
                {
                    "name_norm": f"r_{i}",
                    "session": "S1",
                    "chamber": "House",
                    "party": "Republican",
                    "xi_common": r_mean + rng.normal(0, 0.3),
                }
            )
        for i in range(10):
            rows.append(
                {
                    "name_norm": f"d_{i}",
                    "session": "S1",
                    "chamber": "House",
                    "party": "Democrat",
                    "xi_common": d_mean + rng.normal(0, 0.3),
                }
            )
        return pl.DataFrame(rows)

    def test_passes_with_good_separation(self):
        df = self._make_transformed(r_mean=2.0, d_mean=-2.0)
        gates = compute_quality_gates(df, ["S1"], "House")
        assert len(gates) == 1
        assert gates[0].passed
        assert gates[0].sign_ok

    def test_fails_sign_flip(self):
        df = self._make_transformed(r_mean=-2.0, d_mean=2.0)
        gates = compute_quality_gates(df, ["S1"], "House")
        assert not gates[0].sign_ok
        assert not gates[0].passed

    def test_fails_low_separation(self):
        df = self._make_transformed(r_mean=0.1, d_mean=-0.1)
        gates = compute_quality_gates(df, ["S1"], "House")
        assert gates[0].party_d < PARTY_D_MIN
        assert not gates[0].passed


# ---------------------------------------------------------------------------
# TestPolarizationTrajectory
# ---------------------------------------------------------------------------


class TestPolarizationTrajectory:
    def test_computes_party_means(self):
        rows = []
        for session in ["A", "B"]:
            for i in range(10):
                rows.append(
                    {
                        "name_norm": f"r_{session}_{i}",
                        "session": session,
                        "chamber": "House",
                        "party": "Republican",
                        "xi_common": 1.5 + i * 0.1,
                    }
                )
                rows.append(
                    {
                        "name_norm": f"d_{session}_{i}",
                        "session": session,
                        "chamber": "House",
                        "party": "Democrat",
                        "xi_common": -1.5 - i * 0.1,
                    }
                )
        df = pl.DataFrame(rows)
        traj = compute_polarization_trajectory(df, ["A", "B"], "House")
        assert traj.height == 2
        assert traj["party_gap"][0] > 0  # R mean > D mean


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_bridge_observations(self):
        """Sessions with no shared legislators produce identity transforms."""
        scores = _make_scores(sessions=["A", "B"], n_per_session=10, n_bridge=0)
        roster = build_global_roster(scores, _normalize)
        coefs = solve_simultaneous_alignment(roster, ["A", "B"], "House", "B")
        # Should not crash; reference should be identity
        assert coefs["B"] == (1.0, 0.0)

    def test_missing_chamber_in_session(self):
        """If a session is missing a chamber, it should be silently skipped."""
        scores = {
            "A": {
                "House": pl.DataFrame(
                    {
                        "legislator_slug": ["a"],
                        "full_name": ["A"],
                        "party": ["Republican"],
                        "xi_mean": [1.0],
                    }
                )
            }
        }
        roster = build_global_roster(scores, _normalize)
        # Senate should produce empty bridge matrix
        matrix = compute_bridge_matrix(roster, ["A"])
        assert matrix.height == 0
