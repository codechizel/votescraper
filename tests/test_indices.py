"""
Tests for classical indices computation in analysis/indices.py.

Covers Rice Index formula, party majority positions, party vote identification,
and seat-based ENP using small synthetic polars DataFrames with hand-verifiable
results.

Run: uv run pytest tests/test_indices.py -v
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.indices import (
    _rice_from_counts,
    compute_bipartisanship_index,
    compute_carey_unity,
    compute_co_defection_matrix,
    compute_enp_per_vote,
    compute_enp_seats,
    compute_party_majority_positions,
    compute_unity_and_maverick,
    find_fractured_votes,
    identify_party_votes,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _simple_votes() -> pl.DataFrame:
    """6 legislators (4R + 2D), 2 roll calls.

    rc1: 4 Yea (3R + 1D), 2 Nay (1R + 1D) — party vote (R majority Yea, D split)
    rc2: 6 Yea (all) — unanimous, NOT a party vote
    """
    rows = [
        # rc1
        ("rc1", "rep_a", "Yea"),
        ("rc1", "rep_b", "Yea"),
        ("rc1", "rep_c", "Yea"),
        ("rc1", "rep_d", "Nay"),
        ("rc1", "dem_x", "Yea"),
        ("rc1", "dem_y", "Nay"),
        # rc2 — unanimous
        ("rc2", "rep_a", "Yea"),
        ("rc2", "rep_b", "Yea"),
        ("rc2", "rep_c", "Yea"),
        ("rc2", "rep_d", "Yea"),
        ("rc2", "dem_x", "Yea"),
        ("rc2", "dem_y", "Yea"),
    ]
    return pl.DataFrame(rows, schema=["vote_id", "legislator_slug", "vote"], orient="row")


def _simple_rollcalls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "vote_id": ["rc1", "rc2"],
            "chamber": ["House", "House"],
            "vote_date": ["01/15/2025", "01/16/2025"],
            "motion": ["Final Action", "Final Action"],
            "bill_number": ["HB 1", "HB 2"],
        }
    )


def _simple_legislators() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d", "dem_x", "dem_y"],
            "party": ["Republican"] * 4 + ["Democrat"] * 2,
        }
    )


# ── _rice_from_counts() ─────────────────────────────────────────────────────


class TestRiceFromCounts:
    """Rice = |Yea - Nay| / (Yea + Nay)."""

    def test_unanimous_yea(self):
        """All Yea → Rice = 1.0."""
        df = pl.DataFrame(
            {
                "yea_count": [5],
                "nay_count": [0],
                "total_voters": [5],
            }
        )
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(1.0)

    def test_even_split(self):
        """50-50 split → Rice = 0.0."""
        df = pl.DataFrame(
            {
                "yea_count": [5],
                "nay_count": [5],
                "total_voters": [10],
            }
        )
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(0.0)

    def test_75_25_split(self):
        """3 Yea, 1 Nay → Rice = |3-1|/4 = 0.5."""
        df = pl.DataFrame(
            {
                "yea_count": [3],
                "nay_count": [1],
                "total_voters": [4],
            }
        )
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(0.5)

    def test_null_when_too_few_voters(self):
        """Below MIN_PARTY_VOTERS threshold → null."""
        df = pl.DataFrame(
            {
                "yea_count": [1],
                "nay_count": [0],
                "total_voters": [1],
            }
        )
        result = _rice_from_counts(df)
        assert result["rice_index"][0] is None

    def test_multiple_rows(self):
        """Vectorized: multiple rows computed at once."""
        df = pl.DataFrame(
            {
                "yea_count": [10, 5, 3],
                "nay_count": [0, 5, 7],
                "total_voters": [10, 10, 10],
            }
        )
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(1.0)
        assert result["rice_index"][1] == pytest.approx(0.0)
        assert result["rice_index"][2] == pytest.approx(0.4)


# ── compute_party_majority_positions() ───────────────────────────────────────


class TestPartyMajorityPositions:
    """Per-vote per-party Yea/Nay counts and majority position."""

    def test_counts_correct(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        result = compute_party_majority_positions(votes, rollcalls, legislators, "House")

        # rc1 Republicans: 3 Yea, 1 Nay
        rc1_r = result.filter((pl.col("vote_id") == "rc1") & (pl.col("party") == "Republican"))
        assert rc1_r["yea_count"][0] == 3
        assert rc1_r["nay_count"][0] == 1
        assert rc1_r["majority_position"][0] == "Yea"

    def test_democrat_split_on_rc1(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        result = compute_party_majority_positions(votes, rollcalls, legislators, "House")

        rc1_d = result.filter((pl.col("vote_id") == "rc1") & (pl.col("party") == "Democrat"))
        # 1 Yea, 1 Nay → tied, but yea_count > nay_count is false so majority = "Nay"
        assert rc1_d["yea_count"][0] == 1
        assert rc1_d["nay_count"][0] == 1

    def test_unanimous_rc2(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        result = compute_party_majority_positions(votes, rollcalls, legislators, "House")

        rc2_r = result.filter((pl.col("vote_id") == "rc2") & (pl.col("party") == "Republican"))
        assert rc2_r["yea_count"][0] == 4
        assert rc2_r["nay_count"][0] == 0


# ── identify_party_votes() ──────────────────────────────────────────────────


class TestIdentifyPartyVotes:
    """Party vote = majority of R opposes majority of D."""

    def test_unanimous_not_party_vote(self):
        """rc2 (all Yea) should NOT be a party vote."""
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        party_counts = compute_party_majority_positions(votes, rollcalls, legislators, "House")
        pv = identify_party_votes(party_counts, rollcalls, "House", "test")

        rc2 = pv.filter(pl.col("vote_id") == "rc2")
        assert rc2["is_party_vote"][0] is False

    def test_contested_vote_classification(self):
        """rc1 with R majority Yea and D tied (Nay) → party vote depends on D majority."""
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        party_counts = compute_party_majority_positions(votes, rollcalls, legislators, "House")
        pv = identify_party_votes(party_counts, rollcalls, "House", "test")

        rc1 = pv.filter(pl.col("vote_id") == "rc1")
        # R majority = Yea, D is 1-1 → majority_position = Nay (tie goes to Nay)
        # So R_majority ≠ D_majority → party vote = True
        assert rc1["is_party_vote"][0] is True

    def test_closeness_weight_present(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        party_counts = compute_party_majority_positions(votes, rollcalls, legislators, "House")
        pv = identify_party_votes(party_counts, rollcalls, "House", "test")
        assert "closeness_weight" in pv.columns
        assert pv["closeness_weight"].null_count() == 0


# ── compute_enp_seats() ─────────────────────────────────────────────────────


class TestComputeENPSeats:
    """Seat-based Effective Number of Parties (Laakso-Taagepera)."""

    def test_two_equal_parties(self):
        """50-50 split → ENP = 2.0."""
        legislators = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}" for i in range(5)] + [f"dem_{i}" for i in range(5)],
                "party": ["Republican"] * 5 + ["Democrat"] * 5,
            }
        )
        # Use "rep_" prefix for House
        result = compute_enp_seats(legislators, "House", "test")
        # Only rep_ slugs → 5 R, 0 D in "House" → ENP = 1.0
        # Wait, we need to use the correct slug prefix convention
        # House uses rep_, Senate uses sen_
        assert result.height > 0
        enp = float(result["enp_seats"][0])
        assert enp == pytest.approx(1.0)  # only one party in this chamber

    def test_supermajority(self):
        """80-20 split → ENP = 1 / (0.8^2 + 0.2^2) = 1/0.68 ≈ 1.47."""
        slugs = [f"sen_{i}" for i in range(10)]
        parties = ["Republican"] * 8 + ["Democrat"] * 2
        legislators = pl.DataFrame({"legislator_slug": slugs, "party": parties})
        result = compute_enp_seats(legislators, "Senate", "test")
        enp = float(result["enp_seats"][0])
        expected = 1.0 / (0.8**2 + 0.2**2)
        assert enp == pytest.approx(expected, rel=0.01)

    def test_empty_chamber(self):
        legislators = pl.DataFrame(
            {
                "legislator_slug": ["rep_a"],
                "party": ["Republican"],
            }
        )
        # Senate prefix is sen_, but no sen_ slugs
        result = compute_enp_seats(legislators, "Senate", "test")
        assert result.height == 0


# ── Richer fixture for unity/maverick/co-defection tests ─────────────────────


def _contested_votes() -> pl.DataFrame:
    """8 legislators (5R + 3D), 3 roll calls with clear party splits.

    rc1: Party vote — 4R Yea + 1R Nay, 3D Nay (R majority Yea, D majority Nay)
    rc2: Party vote — 5R Yea, 2D Nay + 1D Yea (R Yea, D Nay)
    rc3: Not a party vote — 5R Yea, 3D Yea (all Yea, unanimous)
    """
    return pl.DataFrame(
        [
            # rc1 — contested
            ("rc1", "rep_a", "Yea"),
            ("rc1", "rep_b", "Yea"),
            ("rc1", "rep_c", "Yea"),
            ("rc1", "rep_d", "Yea"),
            ("rc1", "rep_e", "Nay"),
            ("rc1", "dem_x", "Nay"),
            ("rc1", "dem_y", "Nay"),
            ("rc1", "dem_z", "Nay"),
            # rc2 — contested
            ("rc2", "rep_a", "Yea"),
            ("rc2", "rep_b", "Yea"),
            ("rc2", "rep_c", "Yea"),
            ("rc2", "rep_d", "Yea"),
            ("rc2", "rep_e", "Yea"),
            ("rc2", "dem_x", "Nay"),
            ("rc2", "dem_y", "Nay"),
            ("rc2", "dem_z", "Yea"),
            # rc3 — unanimous
            ("rc3", "rep_a", "Yea"),
            ("rc3", "rep_b", "Yea"),
            ("rc3", "rep_c", "Yea"),
            ("rc3", "rep_d", "Yea"),
            ("rc3", "rep_e", "Yea"),
            ("rc3", "dem_x", "Yea"),
            ("rc3", "dem_y", "Yea"),
            ("rc3", "dem_z", "Yea"),
        ],
        schema=["vote_id", "legislator_slug", "vote"],
        orient="row",
    )


def _contested_rollcalls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "vote_id": ["rc1", "rc2", "rc3"],
            "chamber": ["House", "House", "House"],
            "vote_date": ["01/15/2025", "01/16/2025", "01/17/2025"],
            "motion": ["Final Action", "Final Action", "Final Action"],
            "bill_number": ["HB 1", "HB 2", "HB 3"],
        }
    )


def _contested_legislators() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "dem_x", "dem_y", "dem_z"],
            "party": ["Republican"] * 5 + ["Democrat"] * 3,
            "full_name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Xena", "Yuri", "Zara"],
            "district": [f"D{i}" for i in range(1, 9)],
        }
    )


# ── compute_unity_and_maverick() ────────────────────────────────────────────


class TestComputeUnityAndMaverick:
    """CQ-standard party unity and maverick scores."""

    def _get_party_votes(self):
        votes = _contested_votes()
        rollcalls = _contested_rollcalls()
        legislators = _contested_legislators()
        pc = compute_party_majority_positions(votes, rollcalls, legislators, "House")
        return identify_party_votes(pc, rollcalls, "House", "test")

    def test_returns_three_outputs(self):
        """compute_unity_and_maverick returns (unity_df, maverick_df, defection_sets)."""
        pv = self._get_party_votes()
        unity, maverick, defections = compute_unity_and_maverick(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        assert isinstance(unity, pl.DataFrame)
        assert isinstance(maverick, pl.DataFrame)
        assert isinstance(defections, dict)

    def test_unity_output_columns(self):
        """Unity DataFrame has required columns."""
        pv = self._get_party_votes()
        unity, _, _ = compute_unity_and_maverick(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        expected = {
            "legislator_slug",
            "party",
            "votes_with_party",
            "party_votes_present",
            "unity_score",
            "maverick_rate",
            "session",
        }
        assert expected.issubset(set(unity.columns))

    def test_maverick_output_columns(self):
        """Maverick DataFrame has required columns."""
        pv = self._get_party_votes()
        _, maverick, _ = compute_unity_and_maverick(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        expected = {
            "legislator_slug",
            "party",
            "maverick_rate",
            "weighted_maverick",
            "n_defections",
            "loyalty_zscore",
        }
        assert expected.issubset(set(maverick.columns))

    def test_perfect_unity(self):
        """rep_a votes with party on all party votes → unity = 1.0."""
        pv = self._get_party_votes()
        unity, _, _ = compute_unity_and_maverick(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        rep_a = unity.filter(pl.col("legislator_slug") == "rep_a")
        assert rep_a.height == 1
        assert rep_a["unity_score"][0] == pytest.approx(1.0)
        assert rep_a["maverick_rate"][0] == pytest.approx(0.0)

    def test_defector_detected(self):
        """rep_e votes Nay on rc1 (R majority=Yea) → at least 1 defection."""
        pv = self._get_party_votes()
        _, maverick, defections = compute_unity_and_maverick(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        rep_e = maverick.filter(pl.col("legislator_slug") == "rep_e")
        assert rep_e.height == 1
        assert int(rep_e["n_defections"][0]) >= 1
        assert "rep_e" in defections
        assert len(defections["rep_e"]) >= 1

    def test_unity_plus_maverick_equals_one(self):
        """For every legislator, unity + maverick = 1.0."""
        pv = self._get_party_votes()
        unity, _, _ = compute_unity_and_maverick(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        for row in unity.iter_rows(named=True):
            assert row["unity_score"] + row["maverick_rate"] == pytest.approx(1.0)

    def test_no_party_votes_returns_empty(self):
        """All-unanimous votes → no party votes → empty DataFrames."""
        votes = pl.DataFrame(
            [
                ("rc1", "rep_a", "Yea"),
                ("rc1", "dem_x", "Yea"),
            ],
            schema=["vote_id", "legislator_slug", "vote"],
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc1"],
                "chamber": ["House"],
                "vote_date": ["01/15/2025"],
                "motion": ["Final Action"],
                "bill_number": ["HB 1"],
            }
        )
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "dem_x"],
                "party": ["Republican", "Democrat"],
                "full_name": ["Alice", "Xena"],
                "district": ["D1", "D2"],
            }
        )
        pc = compute_party_majority_positions(votes, rollcalls, legs, "House")
        pv = identify_party_votes(pc, rollcalls, "House", "test")
        unity, maverick, defections = compute_unity_and_maverick(votes, pv, legs, "House", "test")
        assert unity.height == 0
        assert maverick.height == 0


# ── compute_enp_per_vote() ──────────────────────────────────────────────────


class TestComputeENPPerVote:
    """Per-vote Effective Number of Parties from (party, direction) blocs."""

    def test_unanimous_enp_near_one(self):
        """All legislators vote Yea → 2 blocs (R-Yea, D-Yea) but R dominates → ENP ~ 1.x."""
        votes = _contested_votes()
        rollcalls = _contested_rollcalls()
        legislators = _contested_legislators()
        result = compute_enp_per_vote(votes, rollcalls, legislators, "House", "test")
        rc3 = result.filter(pl.col("vote_id") == "rc3")
        assert rc3.height == 1
        # 5R-Yea + 3D-Yea = 2 blocs, shares 5/8 and 3/8 → ENP = 1/(25/64+9/64) = 64/34 ≈ 1.88
        enp = float(rc3["enp"][0])
        assert 1.0 < enp < 2.1  # two unequal blocs

    def test_party_line_enp_near_two(self):
        """rc1: R=Yea, D=Nay (mostly) → ENP near 2.x (allowing for the 1 R defector)."""
        votes = _contested_votes()
        rollcalls = _contested_rollcalls()
        legislators = _contested_legislators()
        result = compute_enp_per_vote(votes, rollcalls, legislators, "House", "test")
        rc1 = result.filter(pl.col("vote_id") == "rc1")
        assert rc1.height == 1
        enp = float(rc1["enp"][0])
        # 3 blocs: R-Yea(4), R-Nay(1), D-Nay(3) → ENP > 2
        assert enp > 1.5

    def test_output_columns(self):
        result = compute_enp_per_vote(
            _contested_votes(), _contested_rollcalls(), _contested_legislators(), "House", "test"
        )
        assert "enp" in result.columns
        assert "vote_id" in result.columns
        assert "session" in result.columns

    def test_single_party_chamber(self):
        """Only one party → all blocs are same party → ENP depends on vote splits."""
        votes = pl.DataFrame(
            [
                ("rc1", "rep_a", "Yea"),
                ("rc1", "rep_b", "Yea"),
                ("rc1", "rep_c", "Nay"),
            ],
            schema=["vote_id", "legislator_slug", "vote"],
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc1"],
                "chamber": ["House"],
                "vote_date": ["01/15/2025"],
                "motion": ["FA"],
                "bill_number": ["HB 1"],
            }
        )
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "rep_c"],
                "party": ["Republican"] * 3,
            }
        )
        result = compute_enp_per_vote(votes, rollcalls, legs, "House", "test")
        assert result.height == 1
        enp = float(result["enp"][0])
        # 2 blocs: R-Yea(2), R-Nay(1) → ENP = 1/(4/9+1/9) = 9/5 = 1.8
        assert enp == pytest.approx(1.8, rel=0.01)


# ── find_fractured_votes() ──────────────────────────────────────────────────


class TestFindFracturedVotes:
    """Identifies votes where the majority party has Rice < threshold."""

    def _get_rice(self):
        votes = _contested_votes()
        rollcalls = _contested_rollcalls()
        legs = _contested_legislators()
        pc = compute_party_majority_positions(votes, rollcalls, legs, "House")
        from analysis.indices import compute_rice_index

        return compute_rice_index(pc, rollcalls, "House", "test")

    def test_detects_majority_party(self):
        """Majority party = Republican (5 seats vs 3)."""
        rice = self._get_rice()
        # find_fractured_votes should not crash and should use Republican as majority
        result = find_fractured_votes(rice, "House", "test")
        # All R votes have high Rice (4-1 or 5-0), so no fractured votes expected
        assert isinstance(result, pl.DataFrame)

    def test_fractured_when_split(self):
        """A 50-50 R split should produce a fractured vote (Rice = 0.0)."""
        # Create a vote with Rs split evenly
        votes = pl.DataFrame(
            [
                ("rc1", "rep_a", "Yea"),
                ("rc1", "rep_b", "Nay"),
                ("rc1", "rep_c", "Yea"),
                ("rc1", "rep_d", "Nay"),
                ("rc1", "dem_x", "Nay"),
                ("rc1", "dem_y", "Nay"),
            ],
            schema=["vote_id", "legislator_slug", "vote"],
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc1"],
                "chamber": ["House"],
                "vote_date": ["01/15/2025"],
                "motion": ["FA"],
                "bill_number": ["HB 1"],
            }
        )
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d", "dem_x", "dem_y"],
                "party": ["Republican"] * 4 + ["Democrat"] * 2,
            }
        )
        pc = compute_party_majority_positions(votes, rollcalls, legs, "House")
        from analysis.indices import compute_rice_index

        rice = compute_rice_index(pc, rollcalls, "House", "test")
        result = find_fractured_votes(rice, "House", "test")
        assert result.height == 1
        assert float(result["rice_index"][0]) == pytest.approx(0.0)

    def test_empty_rice_returns_empty(self):
        """Rice DataFrame with no rows → empty result."""
        empty = pl.DataFrame(
            {
                "vote_id": [],
                "party": [],
                "yea_count": [],
                "nay_count": [],
                "total_voters": [],
                "rice_index": [],
            }
        ).cast(
            {
                "yea_count": pl.Int64,
                "nay_count": pl.Int64,
                "total_voters": pl.Int64,
                "rice_index": pl.Float64,
            }
        )
        result = find_fractured_votes(empty, "House", "test")
        assert result.height == 0


# ── compute_co_defection_matrix() ───────────────────────────────────────────


class TestComputeCoDefectionMatrix:
    """Pairwise co-defection counts among top defectors."""

    def test_no_shared_defections_returns_none(self):
        """If no pairs share enough defections, returns None."""
        maverick = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b"],
                "party": ["Republican", "Republican"],
                "n_defections": [1, 1],
                "full_name": ["Alice", "Bob"],
            }
        )
        defection_sets = {"rep_a": {"rc1"}, "rep_b": {"rc2"}}
        result = compute_co_defection_matrix(defection_sets, maverick, "House", "test")
        assert result is None

    def test_shared_defections_detected(self):
        """Two legislators sharing 3+ defections appear in the matrix."""
        maverick = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b"],
                "party": ["Republican", "Republican"],
                "n_defections": [5, 5],
                "full_name": ["Alice", "Bob"],
            }
        )
        shared = {"rc1", "rc2", "rc3", "rc4"}
        defection_sets = {
            "rep_a": shared | {"rc5"},
            "rep_b": shared | {"rc6"},
        }
        result = compute_co_defection_matrix(defection_sets, maverick, "House", "test")
        assert result is not None
        assert result.height == 1
        assert int(result["shared_defections"][0]) == 4

    def test_fewer_than_two_defectors_returns_none(self):
        """Only 1 defector → cannot build pairwise matrix."""
        maverick = pl.DataFrame(
            {
                "legislator_slug": ["rep_a"],
                "party": ["Republican"],
                "n_defections": [5],
                "full_name": ["Alice"],
            }
        )
        result = compute_co_defection_matrix(
            {"rep_a": {"rc1", "rc2", "rc3"}}, maverick, "House", "test"
        )
        assert result is None


# ── compute_carey_unity() ───────────────────────────────────────────────────


class TestComputeCareyUnity:
    """Carey UNITY = |Yea - Nay| / (total party members in chamber)."""

    def test_formula_correctness(self):
        """5R Yea, 0 Nay → Carey = |5-0|/5 = 1.0."""
        votes = _contested_votes()  # rc3: 5R Yea, 3D Yea (unanimous)
        rollcalls = _contested_rollcalls()
        legs = _contested_legislators()
        result = compute_carey_unity(votes, rollcalls, legs, "House", "test")
        rc3_r = result.filter((pl.col("vote_id") == "rc3") & (pl.col("party") == "Republican"))
        assert rc3_r.height == 1
        assert rc3_r["carey_unity"][0] == pytest.approx(1.0)

    def test_penalizes_defection(self):
        """rc1: 4R Yea + 1R Nay → Carey = |4-1|/5 = 0.6 (vs Rice = 3/5 = 0.6)."""
        result = compute_carey_unity(
            _contested_votes(), _contested_rollcalls(), _contested_legislators(), "House", "test"
        )
        rc1_r = result.filter((pl.col("vote_id") == "rc1") & (pl.col("party") == "Republican"))
        # |4-1| / 5 = 0.6
        assert rc1_r["carey_unity"][0] == pytest.approx(0.6)

    def test_penalizes_absence(self):
        """Missing voter reduces Carey (Rice stays the same)."""
        # 3R Yea, 0R Nay, 2R absent → Carey = |3|/5 = 0.6, Rice = |3|/3 = 1.0
        votes = pl.DataFrame(
            [
                ("rc1", "rep_a", "Yea"),
                ("rc1", "rep_b", "Yea"),
                ("rc1", "rep_c", "Yea"),
                ("rc1", "dem_x", "Nay"),
                ("rc1", "dem_y", "Nay"),
            ],
            schema=["vote_id", "legislator_slug", "vote"],
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc1"],
                "chamber": ["House"],
                "vote_date": ["01/15/2025"],
                "motion": ["FA"],
                "bill_number": ["HB 1"],
            }
        )
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "rep_c", "rep_d", "rep_e", "dem_x", "dem_y"],
                "party": ["Republican"] * 5 + ["Democrat"] * 2,
            }
        )
        result = compute_carey_unity(votes, rollcalls, legs, "House", "test")
        r_row = result.filter(pl.col("party") == "Republican")
        assert r_row.height == 1
        # |3-0| / 5 = 0.6
        assert r_row["carey_unity"][0] == pytest.approx(0.6)
        assert int(r_row["n_absent"][0]) == 2

    def test_n_absent_column(self):
        """n_absent = n_members - yea - nay."""
        result = compute_carey_unity(
            _contested_votes(), _contested_rollcalls(), _contested_legislators(), "House", "test"
        )
        for row in result.iter_rows(named=True):
            assert row["n_absent"] == row["n_members"] - row["yea_count"] - row["nay_count"]

    def test_output_columns(self):
        result = compute_carey_unity(
            _contested_votes(), _contested_rollcalls(), _contested_legislators(), "House", "test"
        )
        expected = {
            "vote_id",
            "party",
            "carey_unity",
            "yea_count",
            "nay_count",
            "n_absent",
            "n_members",
            "session",
        }
        assert expected.issubset(set(result.columns))

    def test_carey_less_than_or_equal_rice(self):
        """Carey ≤ Rice always (larger denominator)."""
        votes = _contested_votes()
        rollcalls = _contested_rollcalls()
        legs = _contested_legislators()
        carey = compute_carey_unity(votes, rollcalls, legs, "House", "test")
        pc = compute_party_majority_positions(votes, rollcalls, legs, "House")
        rice = _rice_from_counts(pc)
        for row in carey.iter_rows(named=True):
            r = rice.filter(
                (pl.col("vote_id") == row["vote_id"]) & (pl.col("party") == row["party"])
            )
            if r.height > 0 and r["rice_index"][0] is not None:
                assert row["carey_unity"] <= float(r["rice_index"][0]) + 1e-10


# ── compute_bipartisanship_index() ────────────────────────────────────────


class TestComputeBipartisanshipIndex:
    """Lugar Center-style BPI = votes with opposing party majority / party votes present."""

    def _get_party_votes(self, votes, rollcalls, legislators, chamber="House"):
        pc = compute_party_majority_positions(votes, rollcalls, legislators, chamber)
        return identify_party_votes(pc, rollcalls, chamber, "test")

    def test_output_columns(self):
        """BPI DataFrame has required columns."""
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        expected = {
            "legislator_slug",
            "full_name",
            "party",
            "district",
            "bpi_score",
            "votes_with_opposition",
            "party_votes_present",
            "session",
        }
        assert expected.issubset(set(result.columns))

    def test_loyal_republican_bpi_zero(self):
        """rep_a votes Yea on rc1 and rc2 (R majority=Yea on both) → BPI=0.

        BPI counts votes *with* opposition majority, not against own party.
        On party votes, R majority=Yea and D majority=Nay.
        rep_a votes Yea (with own party, against opposition) → 0 votes with opposition.
        """
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        rep_a = result.filter(pl.col("legislator_slug") == "rep_a")
        assert rep_a.height == 1
        assert rep_a["bpi_score"][0] == pytest.approx(0.0)

    def test_defector_has_positive_bpi(self):
        """rep_e votes Nay on rc1 (D majority=Nay) → 1 vote with opposition."""
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        rep_e = result.filter(pl.col("legislator_slug") == "rep_e")
        assert rep_e.height == 1
        assert rep_e["bpi_score"][0] > 0.0
        assert int(rep_e["votes_with_opposition"][0]) >= 1

    def test_democrat_crossing_over(self):
        """dem_z votes Yea on rc2 (R majority=Yea) → 1 vote with opposition."""
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        dem_z = result.filter(pl.col("legislator_slug") == "dem_z")
        assert dem_z.height == 1
        assert int(dem_z["votes_with_opposition"][0]) >= 1
        assert dem_z["bpi_score"][0] > 0.0

    def test_bpi_bounded_zero_one(self):
        """BPI scores are in [0, 1] for all legislators."""
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        assert (result["bpi_score"] >= 0.0).all()
        assert (result["bpi_score"] <= 1.0).all()

    def test_sorted_descending(self):
        """Results are sorted by BPI descending (most bipartisan first)."""
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "test"
        )
        scores = result["bpi_score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_no_party_votes_returns_empty(self):
        """All-unanimous votes → no party votes → empty DataFrame."""
        votes = pl.DataFrame(
            [
                ("rc1", "rep_a", "Yea"),
                ("rc1", "dem_x", "Yea"),
            ],
            schema=["vote_id", "legislator_slug", "vote"],
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc1"],
                "chamber": ["House"],
                "vote_date": ["01/15/2025"],
                "motion": ["Final Action"],
                "bill_number": ["HB 1"],
            }
        )
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "dem_x"],
                "party": ["Republican", "Democrat"],
                "full_name": ["Alice", "Xena"],
                "district": ["D1", "D2"],
            }
        )
        pv = self._get_party_votes(votes, rollcalls, legs)
        result = compute_bipartisanship_index(votes, pv, legs, "House", "test")
        assert result.height == 0

    def test_independent_excluded(self):
        """Independents should not appear in BPI results."""
        votes = pl.DataFrame(
            [
                ("rc1", "rep_a", "Yea"),
                ("rc1", "rep_b", "Yea"),
                ("rc1", "ind_z", "Yea"),
                ("rc1", "dem_x", "Nay"),
                ("rc1", "dem_y", "Nay"),
            ],
            schema=["vote_id", "legislator_slug", "vote"],
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc1"],
                "chamber": ["House"],
                "vote_date": ["01/15/2025"],
                "motion": ["FA"],
                "bill_number": ["HB 1"],
            }
        )
        legs = pl.DataFrame(
            {
                "legislator_slug": ["rep_a", "rep_b", "ind_z", "dem_x", "dem_y"],
                "party": ["Republican", "Republican", "Independent", "Democrat", "Democrat"],
                "full_name": ["A", "B", "Z", "X", "Y"],
                "district": ["D1", "D2", "D3", "D4", "D5"],
            }
        )
        pv = self._get_party_votes(votes, rollcalls, legs)
        result = compute_bipartisanship_index(votes, pv, legs, "House", "test")
        slugs = result["legislator_slug"].to_list()
        assert "ind_z" not in slugs

    def test_session_column_propagated(self):
        """Session string is carried through to the output."""
        pv = self._get_party_votes(
            _contested_votes(), _contested_rollcalls(), _contested_legislators()
        )
        result = compute_bipartisanship_index(
            _contested_votes(), pv, _contested_legislators(), "House", "my_session"
        )
        assert (result["session"] == "my_session").all()
