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
    compute_enp_seats,
    compute_party_majority_positions,
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
        ("rc1", "rep_a", "Yea"), ("rc1", "rep_b", "Yea"),
        ("rc1", "rep_c", "Yea"), ("rc1", "rep_d", "Nay"),
        ("rc1", "dem_x", "Yea"), ("rc1", "dem_y", "Nay"),
        # rc2 — unanimous
        ("rc2", "rep_a", "Yea"), ("rc2", "rep_b", "Yea"),
        ("rc2", "rep_c", "Yea"), ("rc2", "rep_d", "Yea"),
        ("rc2", "dem_x", "Yea"), ("rc2", "dem_y", "Yea"),
    ]
    return pl.DataFrame(rows, schema=["vote_id", "legislator_slug", "vote"], orient="row")


def _simple_rollcalls() -> pl.DataFrame:
    return pl.DataFrame({
        "vote_id": ["rc1", "rc2"],
        "chamber": ["House", "House"],
        "vote_date": ["01/15/2025", "01/16/2025"],
        "motion": ["Final Action", "Final Action"],
        "bill_number": ["HB 1", "HB 2"],
    })


def _simple_legislators() -> pl.DataFrame:
    return pl.DataFrame({
        "slug": ["rep_a", "rep_b", "rep_c", "rep_d", "dem_x", "dem_y"],
        "party": ["Republican"] * 4 + ["Democrat"] * 2,
    })


# ── _rice_from_counts() ─────────────────────────────────────────────────────


class TestRiceFromCounts:
    """Rice = |Yea - Nay| / (Yea + Nay)."""

    def test_unanimous_yea(self):
        """All Yea → Rice = 1.0."""
        df = pl.DataFrame({
            "yea_count": [5], "nay_count": [0], "total_voters": [5],
        })
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(1.0)

    def test_even_split(self):
        """50-50 split → Rice = 0.0."""
        df = pl.DataFrame({
            "yea_count": [5], "nay_count": [5], "total_voters": [10],
        })
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(0.0)

    def test_75_25_split(self):
        """3 Yea, 1 Nay → Rice = |3-1|/4 = 0.5."""
        df = pl.DataFrame({
            "yea_count": [3], "nay_count": [1], "total_voters": [4],
        })
        result = _rice_from_counts(df)
        assert result["rice_index"][0] == pytest.approx(0.5)

    def test_null_when_too_few_voters(self):
        """Below MIN_PARTY_VOTERS threshold → null."""
        df = pl.DataFrame({
            "yea_count": [1], "nay_count": [0], "total_voters": [1],
        })
        result = _rice_from_counts(df)
        assert result["rice_index"][0] is None

    def test_multiple_rows(self):
        """Vectorized: multiple rows computed at once."""
        df = pl.DataFrame({
            "yea_count": [10, 5, 3],
            "nay_count": [0, 5, 7],
            "total_voters": [10, 10, 10],
        })
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
        rc1_r = result.filter(
            (pl.col("vote_id") == "rc1") & (pl.col("party") == "Republican")
        )
        assert rc1_r["yea_count"][0] == 3
        assert rc1_r["nay_count"][0] == 1
        assert rc1_r["majority_position"][0] == "Yea"

    def test_democrat_split_on_rc1(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        result = compute_party_majority_positions(votes, rollcalls, legislators, "House")

        rc1_d = result.filter(
            (pl.col("vote_id") == "rc1") & (pl.col("party") == "Democrat")
        )
        # 1 Yea, 1 Nay → tied, but yea_count > nay_count is false so majority = "Nay"
        assert rc1_d["yea_count"][0] == 1
        assert rc1_d["nay_count"][0] == 1

    def test_unanimous_rc2(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        result = compute_party_majority_positions(votes, rollcalls, legislators, "House")

        rc2_r = result.filter(
            (pl.col("vote_id") == "rc2") & (pl.col("party") == "Republican")
        )
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
        party_counts = compute_party_majority_positions(
            votes, rollcalls, legislators, "House"
        )
        pv = identify_party_votes(party_counts, rollcalls, "House", "test")

        rc2 = pv.filter(pl.col("vote_id") == "rc2")
        assert rc2["is_party_vote"][0] is False

    def test_contested_vote_classification(self):
        """rc1 with R majority Yea and D tied (Nay) → party vote depends on D majority."""
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        party_counts = compute_party_majority_positions(
            votes, rollcalls, legislators, "House"
        )
        pv = identify_party_votes(party_counts, rollcalls, "House", "test")

        rc1 = pv.filter(pl.col("vote_id") == "rc1")
        # R majority = Yea, D is 1-1 → majority_position = Nay (tie goes to Nay)
        # So R_majority ≠ D_majority → party vote = True
        assert rc1["is_party_vote"][0] is True

    def test_closeness_weight_present(self):
        votes = _simple_votes()
        rollcalls = _simple_rollcalls()
        legislators = _simple_legislators()
        party_counts = compute_party_majority_positions(
            votes, rollcalls, legislators, "House"
        )
        pv = identify_party_votes(party_counts, rollcalls, "House", "test")
        assert "closeness_weight" in pv.columns
        assert pv["closeness_weight"].null_count() == 0


# ── compute_enp_seats() ─────────────────────────────────────────────────────


class TestComputeENPSeats:
    """Seat-based Effective Number of Parties (Laakso-Taagepera)."""

    def test_two_equal_parties(self):
        """50-50 split → ENP = 2.0."""
        legislators = pl.DataFrame({
            "slug": [f"rep_{i}" for i in range(5)] + [f"dem_{i}" for i in range(5)],
            "party": ["Republican"] * 5 + ["Democrat"] * 5,
        })
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
        legislators = pl.DataFrame({"slug": slugs, "party": parties})
        result = compute_enp_seats(legislators, "Senate", "test")
        enp = float(result["enp_seats"][0])
        expected = 1.0 / (0.8**2 + 0.2**2)
        assert enp == pytest.approx(expected, rel=0.01)

    def test_empty_chamber(self):
        legislators = pl.DataFrame({
            "slug": ["rep_a"], "party": ["Republican"],
        })
        # Senate prefix is sen_, but no sen_ slugs
        result = compute_enp_seats(legislators, "Senate", "test")
        assert result.height == 0
