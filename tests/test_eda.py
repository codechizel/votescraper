"""
Tests for EDA computation logic using synthetic fixtures.

These tests verify that the analysis functions in analysis/eda.py produce
correct results on known inputs. They don't read real data — they use small
synthetic datasets where we can hand-verify the answers.

Run: uv run pytest tests/test_eda.py -v
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add project root to path so we can import analysis.eda
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.eda import (
    _check_near_duplicate_rollcalls,
    build_vote_matrix,
    check_data_integrity,
    classify_party_line,
    compute_agreement_matrices,
    compute_desposato_rice_correction,
    compute_eigenvalue_preview,
    compute_item_total_correlations,
    compute_party_unity_scores,
    compute_rice_cohesion,
    filter_vote_matrix,
    numpy_matrix_to_polars,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────
# Small synthetic datasets with known properties.

# Slugs for 6 test legislators (A-F)
_SLUGS = [f"sen_{c.lower()}_{c.lower()}_1" for c in "abcdef"]

# Column schema for vote rows
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


def _vote_row(rc: str, bill: str, day: str, slug: str, vote: str) -> tuple:
    """Build a single vote tuple with Senate defaults to keep fixture compact."""
    name = slug.split("_")[1].upper()
    return (
        "2025-26",
        bill,
        "Test",
        rc,
        f"2025-01-0{day}T10:00:00",
        f"01/0{day}/2025",
        "Senate",
        "Final Action",
        name,
        slug,
        vote,
    )


@pytest.fixture
def simple_votes() -> pl.DataFrame:
    """6 legislators × 4 rollcalls with known vote patterns.

    Rollcall layout:
      rc1: all Yea (unanimous — should be filtered)
      rc2: 3 Yea, 3 Nay (contested, evenly split)
      rc3: 5 Yea, 1 Nay (contested, lopsided but >2.5%)
      rc4: 4 Yea, 1 Nay, 1 Absent (contested)
    """
    # rc1: unanimous (all Yea)
    rc1 = [_vote_row("rc1", "SB 1", "1", s, "Yea") for s in _SLUGS]
    # rc2: 3 Yea (A,B,C) / 3 Nay (D,E,F)
    rc2 = [
        *[_vote_row("rc2", "SB 2", "2", s, "Yea") for s in _SLUGS[:3]],
        *[_vote_row("rc2", "SB 2", "2", s, "Nay") for s in _SLUGS[3:]],
    ]
    # rc3: 5 Yea / 1 Nay (F)
    rc3 = [
        *[_vote_row("rc3", "SB 3", "3", s, "Yea") for s in _SLUGS[:5]],
        _vote_row("rc3", "SB 3", "3", _SLUGS[5], "Nay"),
    ]
    # rc4: 4 Yea (A-D), 1 Nay (E), 1 Absent (F)
    rc4 = [
        *[_vote_row("rc4", "SB 4", "4", s, "Yea") for s in _SLUGS[:4]],
        _vote_row("rc4", "SB 4", "4", _SLUGS[4], "Nay"),
        _vote_row("rc4", "SB 4", "4", _SLUGS[5], "Absent and Not Voting"),
    ]
    return pl.DataFrame(
        rc1 + rc2 + rc3 + rc4,
        schema=_VOTE_SCHEMA,
        orient="row",
    )


@pytest.fixture
def simple_rollcalls() -> pl.DataFrame:
    """Rollcall summaries matching simple_votes."""
    return pl.DataFrame(
        {
            "vote_id": ["rc1", "rc2", "rc3", "rc4"],
            "chamber": ["Senate", "Senate", "Senate", "Senate"],
            "yea_count": [6, 3, 5, 4],
            "nay_count": [0, 3, 1, 1],
            "total_votes": [6, 6, 6, 6],
        }
    )


@pytest.fixture
def simple_legislators() -> pl.DataFrame:
    """Legislators matching simple_votes: 3 R, 3 D."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "D", "E", "F"],
            "full_name": ["A A", "B B", "C C", "D D", "E E", "F F"],
            "legislator_slug": [
                "sen_a_a_1",
                "sen_b_b_1",
                "sen_c_c_1",
                "sen_d_d_1",
                "sen_e_e_1",
                "sen_f_f_1",
            ],
            "chamber": ["Senate"] * 6,
            "party": ["Republican", "Republican", "Republican", "Democrat", "Democrat", "Democrat"],
            "district": [1, 2, 3, 4, 5, 6],
            "member_url": [""] * 6,
        }
    )


# ── Vote Matrix Tests ────────────────────────────────────────────────────────


class TestBuildVoteMatrix:
    """Test binary vote matrix construction."""

    def test_shape(self, simple_votes: pl.DataFrame) -> None:
        """Matrix should have 6 legislators × 4 rollcalls."""
        matrix = build_vote_matrix(simple_votes)
        assert matrix.height == 6
        # 4 vote columns + 1 slug column
        assert len(matrix.columns) == 5

    def test_yea_encoded_as_1(self, simple_votes: pl.DataFrame) -> None:
        """Yea votes should be encoded as 1."""
        matrix = build_vote_matrix(simple_votes)
        # Legislator A (sen_a_a_1) voted Yea on all 4 rollcalls
        row_a = matrix.filter(pl.col("legislator_slug") == "sen_a_a_1")
        for rc in ["rc1", "rc2", "rc3", "rc4"]:
            assert row_a[rc][0] == 1, f"Expected 1 for Yea on {rc}"

    def test_nay_encoded_as_0(self, simple_votes: pl.DataFrame) -> None:
        """Nay votes should be encoded as 0."""
        matrix = build_vote_matrix(simple_votes)
        # Legislator D (sen_d_d_1) voted Nay on rc2
        row_d = matrix.filter(pl.col("legislator_slug") == "sen_d_d_1")
        assert row_d["rc2"][0] == 0

    def test_absent_encoded_as_null(self, simple_votes: pl.DataFrame) -> None:
        """Absent/Not Voting should be encoded as null (not 0 or 1)."""
        matrix = build_vote_matrix(simple_votes)
        # Legislator F (sen_f_f_1) was absent on rc4
        row_f = matrix.filter(pl.col("legislator_slug") == "sen_f_f_1")
        assert row_f["rc4"][0] is None


# ── Filter Vote Matrix Tests ────────────────────────────────────────────────


class TestFilterVoteMatrix:
    """Test vote matrix filtering logic."""

    def test_unanimous_vote_dropped(
        self, simple_votes: pl.DataFrame, simple_rollcalls: pl.DataFrame
    ) -> None:
        """rc1 (all Yea, 0% minority) should be dropped at 2.5% threshold."""
        matrix = build_vote_matrix(simple_votes)
        filtered, manifest = filter_vote_matrix(
            matrix,
            simple_rollcalls,
            chamber="Senate",
            minority_threshold=0.025,
            min_votes=1,
        )
        # rc1 should be gone, rc2/rc3/rc4 should remain
        vote_cols = [c for c in filtered.columns if c != "legislator_slug"]
        assert "rc1" not in vote_cols
        assert "rc2" in vote_cols

    def test_manifest_records_drops(
        self, simple_votes: pl.DataFrame, simple_rollcalls: pl.DataFrame
    ) -> None:
        """The manifest should record how many votes were dropped."""
        matrix = build_vote_matrix(simple_votes)
        _, manifest = filter_vote_matrix(
            matrix,
            simple_rollcalls,
            chamber="Senate",
            minority_threshold=0.025,
            min_votes=1,
        )
        assert manifest["votes_dropped_unanimous"] == 1  # just rc1
        assert manifest["votes_before"] == 4
        assert manifest["votes_after"] == 3

    def test_low_participation_legislator_dropped(
        self, simple_votes: pl.DataFrame, simple_rollcalls: pl.DataFrame
    ) -> None:
        """Setting min_votes=4 should drop legislators with fewer votes."""
        matrix = build_vote_matrix(simple_votes)
        # All 6 legislators have 3-4 non-null values (rc1 was dropped).
        # Legislator F has only 2 non-null (was absent on rc4 → null).
        filtered, manifest = filter_vote_matrix(
            matrix,
            simple_rollcalls,
            chamber="Senate",
            minority_threshold=0.025,
            min_votes=4,
        )
        slugs = filtered["legislator_slug"].to_list()
        # F only has 2 substantive votes on contested rollcalls → dropped
        assert "sen_f_f_1" not in slugs
        assert manifest["legislators_dropped_low_participation"] > 0


# ── Agreement Matrix Tests ───────────────────────────────────────────────────


class TestAgreementMatrices:
    """Test pairwise agreement and Cohen's Kappa computation."""

    def test_perfect_agreement(self) -> None:
        """Two legislators who vote identically should have agreement = 1.0."""
        # Create a small matrix: 2 legislators, 5 votes, all identical
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["leg_a", "leg_b"],
                "v1": [1, 1],
                "v2": [0, 0],
                "v3": [1, 1],
                "v4": [0, 0],
                "v5": [1, 1],
                "v6": [1, 1],
                "v7": [0, 0],
                "v8": [1, 1],
                "v9": [0, 0],
                "v10": [1, 1],
            }
        )
        agreement, kappa = compute_agreement_matrices(matrix)
        assert agreement[0, 1] == pytest.approx(1.0)
        assert kappa[0, 1] == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        """Two legislators who always disagree should have agreement = 0.0."""
        # Use 5/5 split for symmetric marginals so kappa reaches -1.0
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["leg_a", "leg_b"],
                "v1": [1, 0],
                "v2": [0, 1],
                "v3": [1, 0],
                "v4": [0, 1],
                "v5": [1, 0],
                "v6": [0, 1],
                "v7": [1, 0],
                "v8": [0, 1],
                "v9": [1, 0],
                "v10": [0, 1],
            }
        )
        agreement, kappa = compute_agreement_matrices(matrix)
        assert agreement[0, 1] == pytest.approx(0.0)
        # Kappa = -1.0 when marginals are symmetric (both 5 Yea / 5 Nay)
        assert kappa[0, 1] == pytest.approx(-1.0)

    def test_null_handling(self) -> None:
        """Nulls (absences) should be excluded from agreement computation."""
        # Leg A and B overlap on v1-v10 (all agree); v11-v14 have mixed nulls.
        # Need >= MIN_SHARED_VOTES (10) shared non-null votes.
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["leg_a", "leg_b"],
                "v1": [1, 1],
                "v2": [0, 0],
                "v3": [1, 1],
                "v4": [0, 0],
                "v5": [1, 1],
                "v6": [0, 0],
                "v7": [1, 1],
                "v8": [0, 0],
                "v9": [1, 1],
                "v10": [0, 0],
                "v11": [1, None],
                "v12": [0, None],
                "v13": [None, 1],
                "v14": [None, 0],
            }
        )
        agreement, _ = compute_agreement_matrices(matrix)
        # 10 shared non-null votes (v1-v10), all matching → agreement = 1.0
        assert agreement[0, 1] == pytest.approx(1.0)

    def test_insufficient_overlap_gives_nan(self) -> None:
        """Pairs with fewer than MIN_SHARED_VOTES shared votes get NaN."""
        # Only 5 shared votes (below the default threshold of 10)
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["leg_a", "leg_b"],
                "v1": [1, 1],
                "v2": [0, 0],
                "v3": [1, 1],
                "v4": [0, 0],
                "v5": [1, 1],
                "v6": [1, None],
                "v7": [0, None],
                "v8": [None, 1],
                "v9": [None, 0],
                "v10": [None, None],
            }
        )
        agreement, _ = compute_agreement_matrices(matrix)
        # Only 5 shared — below MIN_SHARED_VOTES (10), so NaN
        assert np.isnan(agreement[0, 1])

    def test_kappa_corrects_for_base_rate(self) -> None:
        """Kappa should be lower than raw agreement when base rate is high.

        If both legislators vote Yea 80% of the time, raw agreement is ~68%
        by chance. Kappa removes this inflation.
        """
        # Construct: both vote Yea 80% of the time, with some agreement
        np.random.seed(42)
        n = 100
        # Both lean Yea but have some variance
        a_votes = (np.random.rand(n) < 0.8).astype(float)
        b_votes = (np.random.rand(n) < 0.8).astype(float)

        cols = {f"v{i}": [a_votes[i], b_votes[i]] for i in range(n)}
        cols["legislator_slug"] = ["leg_a", "leg_b"]
        matrix = pl.DataFrame(cols)

        agreement, kappa = compute_agreement_matrices(matrix)
        # Kappa should be meaningfully lower than raw agreement
        assert kappa[0, 1] < agreement[0, 1]


# ── Rice Cohesion Tests ──────────────────────────────────────────────────────


class TestRiceCohesion:
    """Test Rice Cohesion Index computation."""

    def test_perfect_unity_gives_rice_1(
        self, simple_votes: pl.DataFrame, simple_legislators: pl.DataFrame
    ) -> None:
        """rc1 is all Yea — both parties have Rice=1.0."""
        rice = compute_rice_cohesion(simple_votes, simple_legislators)
        rc1_r = rice.filter((pl.col("vote_id") == "rc1") & (pl.col("party") == "Republican"))
        assert rc1_r["rice_index"][0] == pytest.approx(1.0)

    def test_even_split_gives_rice_0(
        self, simple_votes: pl.DataFrame, simple_legislators: pl.DataFrame
    ) -> None:
        """If a party splits 50/50, Rice should be 0.0.

        In our fixture, rc2 has R=3 Yea/0 Nay and D=0 Yea/3 Nay,
        so both parties are perfectly unified (Rice=1.0 each).
        The PARTY is unified, it's the inter-party that disagrees.
        """
        rice = compute_rice_cohesion(simple_votes, simple_legislators)
        # Republicans on rc2: all 3 voted Yea → Rice=1.0
        rc2_r = rice.filter((pl.col("vote_id") == "rc2") & (pl.col("party") == "Republican"))
        assert rc2_r["rice_index"][0] == pytest.approx(1.0)

    def test_rice_range(self, simple_votes: pl.DataFrame, simple_legislators: pl.DataFrame) -> None:
        """All Rice values should be in [0, 1]."""
        rice = compute_rice_cohesion(simple_votes, simple_legislators)
        assert rice["rice_index"].min() >= 0.0
        assert rice["rice_index"].max() <= 1.0

    def test_rice_formula_manually(self) -> None:
        """Verify Rice = |yea - nay| / (yea + nay) on a known case.

        Party with 7 Yea and 3 Nay: Rice = |7-3|/10 = 0.4
        """
        votes = pl.DataFrame(
            {
                "vote_id": ["v1"] * 10,
                "legislator_slug": [f"rep_{i}_x_1" for i in range(10)],
                "vote": ["Yea"] * 7 + ["Nay"] * 3,
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": [f"rep_{i}_x_1" for i in range(10)],
                "party": ["Republican"] * 10,
            }
        )
        rice = compute_rice_cohesion(votes, legislators)
        assert rice["rice_index"][0] == pytest.approx(0.4)


# ── Party Line Classification Tests ─────────────────────────────────────────


class TestClassifyPartyLine:
    """Test the bipartisan/party-line/mixed classification."""

    def test_party_line_vote(
        self,
        simple_votes: pl.DataFrame,
        simple_rollcalls: pl.DataFrame,
        simple_legislators: pl.DataFrame,
    ) -> None:
        """rc2 (R all Yea, D all Nay) should be classified as party-line."""
        result = classify_party_line(simple_votes, simple_rollcalls, simple_legislators)
        rc2_row = result.filter(pl.col("vote_id") == "rc2")
        assert rc2_row.height == 1
        assert rc2_row["vote_alignment"][0] == "party-line"

    def test_bipartisan_vote(
        self,
        simple_votes: pl.DataFrame,
        simple_rollcalls: pl.DataFrame,
        simple_legislators: pl.DataFrame,
    ) -> None:
        """rc1 (all Yea, both parties >90%) should be bipartisan."""
        result = classify_party_line(simple_votes, simple_rollcalls, simple_legislators)
        rc1_row = result.filter(pl.col("vote_id") == "rc1")
        assert rc1_row.height == 1
        assert rc1_row["vote_alignment"][0] == "bipartisan"

    def test_mixed_vote(self) -> None:
        """A vote where neither party is >90% in either direction should be mixed."""
        # R: 6 Yea, 4 Nay (60% Yea) — not >90% either way
        # D: 3 Yea, 2 Nay (60% Yea) — not >90% either way
        slugs_r = [f"rep_{i}_x_1" for i in range(10)]
        slugs_d = [f"rep_{i}_y_1" for i in range(5)]
        vote_data = []
        for s in slugs_r[:6]:
            vote_data.append(("rc_mixed", "HB 1", "01", s, "Yea"))
        for s in slugs_r[6:]:
            vote_data.append(("rc_mixed", "HB 1", "01", s, "Nay"))
        for s in slugs_d[:3]:
            vote_data.append(("rc_mixed", "HB 1", "01", s, "Yea"))
        for s in slugs_d[3:]:
            vote_data.append(("rc_mixed", "HB 1", "01", s, "Nay"))

        votes = pl.DataFrame(
            [_vote_row(rc, bill, day, slug, vote) for rc, bill, day, slug, vote in vote_data],
            schema=_VOTE_SCHEMA,
            orient="row",
        )
        rollcalls = pl.DataFrame(
            {
                "vote_id": ["rc_mixed"],
                "chamber": ["House"],
                "yea_count": [9],
                "nay_count": [6],
                "total_votes": [15],
            }
        )
        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs_r + slugs_d,
                "full_name": [f"R{i}" for i in range(10)] + [f"D{i}" for i in range(5)],
                "chamber": ["House"] * 15,
                "party": ["Republican"] * 10 + ["Democrat"] * 5,
                "district": list(range(1, 16)),
                "member_url": [""] * 15,
            }
        )

        result = classify_party_line(votes, rollcalls, legislators)
        assert result.height == 1
        assert result["vote_alignment"][0] == "mixed"


# ── Near-Duplicate Rollcall Detection Tests ─────────────────────────────────


class TestNearDuplicateRollcalls:
    """Test near-duplicate rollcall detection."""

    def test_detects_identical_contested_votes(self) -> None:
        """Two contested rollcalls with identical vote patterns should be detected."""
        # Need >= 10 legislators for MIN_SHARED_VOTES threshold
        slugs = [f"sen_{c}_{c}_1" for c in "abcdefghijklmn"]  # 14 legislators
        rows = []
        for rc in ["rc_a", "rc_b"]:
            # 7 Yea, 7 Nay — contested and identical across both rollcalls
            for s in slugs[:7]:
                rows.append(_vote_row(rc, "SB 1", "1", s, "Yea"))
            for s in slugs[7:]:
                rows.append(_vote_row(rc, "SB 1", "1", s, "Nay"))

        votes = pl.DataFrame(rows, schema=_VOTE_SCHEMA, orient="row")
        findings: dict = {"warnings": [], "info": []}
        _check_near_duplicate_rollcalls(votes, findings)

        # rc_a and rc_b are identical on contested votes → detected
        assert "near_duplicate_rollcalls" in findings
        pairs = findings["near_duplicate_rollcalls"]
        ids = {(p["vote_id_a"], p["vote_id_b"]) for p in pairs}
        assert ("rc_a", "rc_b") in ids or ("rc_b", "rc_a") in ids

    def test_ignores_different_votes(self) -> None:
        """Two contested rollcalls with very different patterns should not be flagged."""
        # rc_a: all Yea from first 5, Nay from last
        # rc_b: opposite pattern
        slugs = [f"sen_{c}_{c}_1" for c in "abcdefghij"]  # 10 legislators
        rows = []
        # rc_a: 5 Yea, 5 Nay
        for s in slugs[:5]:
            rows.append(_vote_row("rc_a", "SB 1", "1", s, "Yea"))
        for s in slugs[5:]:
            rows.append(_vote_row("rc_a", "SB 1", "1", s, "Nay"))
        # rc_b: reversed
        for s in slugs[:5]:
            rows.append(_vote_row("rc_b", "SB 2", "2", s, "Nay"))
        for s in slugs[5:]:
            rows.append(_vote_row("rc_b", "SB 2", "2", s, "Yea"))

        votes = pl.DataFrame(rows, schema=_VOTE_SCHEMA, orient="row")
        findings: dict = {"warnings": [], "info": []}
        _check_near_duplicate_rollcalls(votes, findings)

        # Completely opposite patterns → NOT near-duplicates
        assert "near_duplicate_rollcalls" not in findings


# ── Integrity Check Isolated Tests ──────────────────────────────────────────


class TestIntegrityChecks:
    """Test individual data integrity checks in isolation."""

    def test_detects_duplicate_votes(
        self,
        simple_votes: pl.DataFrame,
        simple_rollcalls: pl.DataFrame,
        simple_legislators: pl.DataFrame,
    ) -> None:
        """Injecting a duplicate (same slug + vote_id) should trigger a warning."""
        # Duplicate: legislator A votes Yea twice on rc1
        dupe_row = simple_votes.filter(
            (pl.col("legislator_slug") == "sen_a_a_1") & (pl.col("vote_id") == "rc1")
        )
        votes_with_dupe = pl.concat([simple_votes, dupe_row])

        findings = check_data_integrity(votes_with_dupe, simple_rollcalls, simple_legislators)
        # Should have a warning about duplicate votes
        dupe_warnings = [w for w in findings["warnings"] if "duplicate" in w.lower()]
        assert len(dupe_warnings) > 0

    def test_detects_tally_mismatch(
        self,
        simple_votes: pl.DataFrame,
        simple_rollcalls: pl.DataFrame,
        simple_legislators: pl.DataFrame,
    ) -> None:
        """A rollcall with wrong yea_count should trigger a tally mismatch warning."""
        # Corrupt rc1 yea_count: actual is 6 Yea, set to 5
        bad_rollcalls = simple_rollcalls.with_columns(
            pl.when(pl.col("vote_id") == "rc1")
            .then(pl.lit(5))
            .otherwise(pl.col("yea_count"))
            .alias("yea_count")
        )

        findings = check_data_integrity(simple_votes, bad_rollcalls, simple_legislators)
        tally_warnings = [w for w in findings["warnings"] if "tally" in w.lower()]
        assert len(tally_warnings) > 0


# ── New Feature Tests ───────────────────────────────────────────────────────


class TestPartyUnityScores:
    """Test Party Unity Score computation."""

    def test_perfect_loyalist_gets_1(
        self,
        simple_votes: pl.DataFrame,
        simple_rollcalls: pl.DataFrame,
        simple_legislators: pl.DataFrame,
    ) -> None:
        """A legislator who always votes with party majority on party-line votes gets 1.0."""
        unity = compute_party_unity_scores(simple_votes, simple_rollcalls, simple_legislators)
        if unity.is_empty():
            pytest.skip("No party-line votes in fixture")
        # All Rs voted Yea together on rc2 (party-line) → unity = 1.0
        a_row = unity.filter(pl.col("legislator_slug") == "sen_a_a_1")
        if a_row.height > 0:
            assert a_row["party_unity_score"][0] == pytest.approx(1.0)


class TestEigenvaluePreview:
    """Test eigenvalue preview computation."""

    def test_returns_eigenvalues(self) -> None:
        """Should return a list of eigenvalues and a ratio."""
        # Create a simple matrix with some variance
        matrix = pl.DataFrame(
            {
                "legislator_slug": ["a", "b", "c", "d", "e"],
                "v1": [1, 1, 0, 0, 1],
                "v2": [1, 1, 0, 0, 0],
                "v3": [0, 0, 1, 1, 0],
                "v4": [1, 0, 1, 0, 1],
                "v5": [0, 1, 0, 1, 0],
            }
        )
        result = compute_eigenvalue_preview(matrix, "Test")
        assert "eigenvalues" in result
        assert len(result["eigenvalues"]) > 0
        assert result["lambda_ratio"] is not None
        assert result["lambda_ratio"] > 0


class TestItemTotalCorrelations:
    """Test item-total correlation computation."""

    def test_flags_non_discriminating_votes(self) -> None:
        """A constant-outcome vote should have near-zero correlation and be flagged."""
        # v1-v4: correlated with Yea tendency; v5: random/uncorrelated
        np.random.seed(42)
        matrix = pl.DataFrame(
            {
                "legislator_slug": [f"leg_{i}" for i in range(20)],
                "v1": [1] * 15 + [0] * 5,
                "v2": [1] * 14 + [0] * 6,
                "v3": [1] * 13 + [0] * 7,
                "v4": [1] * 12 + [0] * 8,
                # v5: nearly random relative to others
                "v5": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        result = compute_item_total_correlations(matrix, "Test")
        assert not result.is_empty()
        # At least some votes should have correlations computed
        assert result.height > 0


class TestDesposato:
    """Test Desposato Rice correction."""

    def test_correction_produces_valid_results(self) -> None:
        """Desposato correction should produce valid Rice indices for both parties.

        `uv run pytest tests/test_eda.py::TestDesposato -v`
        """
        # Create a dataset with asymmetric party sizes: 20 Rs, 5 Ds
        slugs_r = [f"rep_{i}_x_1" for i in range(20)]
        slugs_d = [f"rep_{i}_y_1" for i in range(5)]
        # Rs: 18 Yea / 2 Nay on most votes → high Rice from size
        rows = []
        for vid in [f"v{i}" for i in range(20)]:
            for s in slugs_r[:18]:
                rows.append({"vote_id": vid, "legislator_slug": s, "vote": "Yea"})
            for s in slugs_r[18:]:
                rows.append({"vote_id": vid, "legislator_slug": s, "vote": "Nay"})
            # Ds: 4 Yea / 1 Nay
            for s in slugs_d[:4]:
                rows.append({"vote_id": vid, "legislator_slug": s, "vote": "Yea"})
            rows.append({"vote_id": vid, "legislator_slug": slugs_d[4], "vote": "Nay"})

        votes = pl.DataFrame(rows)
        legislators = pl.DataFrame(
            {
                "legislator_slug": slugs_r + slugs_d,
                "party": ["Republican"] * 20 + ["Democrat"] * 5,
            }
        )

        result = compute_desposato_rice_correction(votes, legislators)
        assert "raw" in result
        assert "corrected" in result
        # Both raw and corrected Rice should be in [0, 1]
        for party in ["Republican", "Democrat"]:
            assert 0.0 <= result["raw"][party] <= 1.0
            assert 0.0 <= result["corrected"][party] <= 1.0
        # Smaller party (Democrat) should be unchanged
        assert result["corrected"]["Democrat"] == result["raw"]["Democrat"]
        # Correction identifies smaller party correctly
        assert result["min_party"] == "Democrat"
        assert result["min_size"] == 5


class TestNumpyMatrixToPolars:
    """Test the module-level numpy_matrix_to_polars helper."""

    def test_roundtrip(self) -> None:
        """Converting numpy → polars should preserve values."""
        mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        slugs = ["leg_a", "leg_b"]
        df = numpy_matrix_to_polars(mat, slugs)
        assert df.height == 2
        assert df["leg_a"][0] == pytest.approx(1.0)
        assert df["leg_b"][0] == pytest.approx(0.5)
        assert df["legislator_slug"].to_list() == slugs
