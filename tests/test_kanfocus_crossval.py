"""
Tests for KanFocus cross-validation (KF vs JE data comparison).

Run: uv run pytest tests/test_kanfocus_crossval.py -v
"""

import pytest

from tallgrass.kanfocus.crossval import (
    CrossValReport,
    RollCallComparison,
    VoteMismatch,
    compare_individual_votes,
    compare_rollcall,
    find_matches,
    format_report,
    normalize_bill_number,
)
from tallgrass.models import IndividualVote, RollCall

pytestmark = pytest.mark.scraper


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_rc(
    bill_number: str = "SB 1",
    chamber: str = "Senate",
    vote_date: str = "03/20/2025",
    vote_id: str = "kf_1_2025_S",
    yea_count: int = 30,
    nay_count: int = 10,
    present_passing_count: int = 0,
    absent_not_voting_count: int = 0,
    not_voting_count: int = 0,
    passed: bool | None = True,
) -> RollCall:
    return RollCall(
        session="91st (2025-2026)",
        bill_number=bill_number,
        bill_title="",
        vote_id=vote_id,
        vote_url="",
        vote_datetime="",
        vote_date=vote_date,
        chamber=chamber,
        motion="Final Action",
        vote_type="",
        result="",
        short_title="",
        sponsor="",
        yea_count=yea_count,
        nay_count=nay_count,
        present_passing_count=present_passing_count,
        absent_not_voting_count=absent_not_voting_count,
        not_voting_count=not_voting_count,
        passed=passed,
    )


def _make_vote(
    vote_id: str = "kf_1_2025_S",
    slug: str = "sen_smith_john_1",
    name: str = "Smith",
    vote: str = "Yea",
) -> IndividualVote:
    return IndividualVote(
        session="91st (2025-2026)",
        bill_number="SB 1",
        bill_title="",
        vote_id=vote_id,
        vote_datetime="",
        vote_date="03/20/2025",
        chamber="Senate",
        motion="Final Action",
        legislator_name=name,
        legislator_slug=slug,
        vote=vote,
    )


# ── TestNormalizeBillNumber ──────────────────────────────────────────────


class TestNormalizeBillNumber:
    """Run: uv run pytest tests/test_kanfocus_crossval.py -k TestNormalizeBillNumber -v"""

    def test_standard_bill(self):
        assert normalize_bill_number("HB 2001") == "HB 2001"

    def test_senate_bill(self):
        assert normalize_bill_number("SB 55") == "SB 55"

    def test_sub_for(self):
        assert normalize_bill_number("Sub for HB 2007") == "HB 2007"

    def test_double_sub_for(self):
        assert normalize_bill_number("Sub for Sub for HB 2007") == "HB 2007"

    def test_s_sub_for(self):
        assert normalize_bill_number("S Sub for Sub for HB 2007") == "HB 2007"

    def test_h_sub_for(self):
        assert normalize_bill_number("H Sub for SB 123") == "SB 123"

    def test_whitespace(self):
        assert normalize_bill_number("  SB   55  ") == "SB 55"

    def test_resolution(self):
        assert normalize_bill_number("HR 6004") == "HR 6004"

    def test_sub_for_case_insensitive(self):
        assert normalize_bill_number("sub for HB 100") == "HB 100"

    def test_empty_string(self):
        assert normalize_bill_number("") == ""


# ── TestFindMatches ──────────────────────────────────────────────────────


class TestFindMatches:
    """Run: uv run pytest tests/test_kanfocus_crossval.py -k TestFindMatches -v"""

    def test_exact_match(self):
        kf = [_make_rc(bill_number="SB 1", vote_id="kf_1_2025_S")]
        je = [_make_rc(bill_number="SB 1", vote_id="je_20250320_1")]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 1
        assert len(unmatched) == 0
        assert matched[0][0].vote_id == "kf_1_2025_S"
        assert matched[0][1].vote_id == "je_20250320_1"

    def test_sub_for_match(self):
        """KF 'Sub for HB 2007' should match JE 'HB 2007'."""
        kf = [_make_rc(bill_number="Sub for HB 2007", vote_id="kf_1_2025_H")]
        je = [_make_rc(bill_number="HB 2007", vote_id="je_20250320_1")]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 1
        assert len(unmatched) == 0

    def test_no_match_different_bill(self):
        kf = [_make_rc(bill_number="SB 1", vote_id="kf_1_2025_S")]
        je = [_make_rc(bill_number="SB 2", vote_id="je_20250320_1")]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 0
        assert len(unmatched) == 1

    def test_no_match_different_chamber(self):
        kf = [_make_rc(bill_number="SB 1", chamber="Senate", vote_id="kf_1_2025_S")]
        je = [_make_rc(bill_number="SB 1", chamber="House", vote_id="je_20250320_1")]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 0
        assert len(unmatched) == 1

    def test_no_match_different_date(self):
        kf = [_make_rc(bill_number="SB 1", vote_date="03/20/2025", vote_id="kf_1")]
        je = [_make_rc(bill_number="SB 1", vote_date="03/21/2025", vote_id="je_1")]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 0
        assert len(unmatched) == 1

    def test_multiple_matches(self):
        kf = [
            _make_rc(bill_number="SB 1", vote_id="kf_1"),
            _make_rc(bill_number="SB 2", vote_date="03/21/2025", vote_id="kf_2"),
        ]
        je = [
            _make_rc(bill_number="SB 1", vote_id="je_1"),
            _make_rc(bill_number="SB 2", vote_date="03/21/2025", vote_id="je_2"),
        ]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 2
        assert len(unmatched) == 0

    def test_empty_kf(self):
        matched, unmatched = find_matches([], [_make_rc()])
        assert len(matched) == 0
        assert len(unmatched) == 0

    def test_empty_je(self):
        matched, unmatched = find_matches([_make_rc()], [])
        assert len(matched) == 0
        assert len(unmatched) == 1

    # -- Multi-motion tally-based matching --

    def test_multi_motion_same_bill_different_tallies(self):
        """Two KF rollcalls on the same bill/day match two JE rollcalls by tally."""
        kf = [
            _make_rc(bill_number="SB 63", vote_id="kf_1", yea_count=25, nay_count=15),
            _make_rc(bill_number="SB 63", vote_id="kf_2", yea_count=30, nay_count=10),
        ]
        je = [
            _make_rc(bill_number="SB 63", vote_id="je_1", yea_count=30, nay_count=10),
            _make_rc(bill_number="SB 63", vote_id="je_2", yea_count=25, nay_count=15),
        ]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 2
        assert len(unmatched) == 0
        # Verify correct pairing by tally (kf_1 25/15 → je_2 25/15)
        match_dict = {m[0].vote_id: m[1].vote_id for m in matched}
        assert match_dict["kf_1"] == "je_2"
        assert match_dict["kf_2"] == "je_1"

    def test_multi_motion_identical_tallies_paired_positionally(self):
        """Duplicate tallies on the same bill/day pair positionally."""
        kf = [
            _make_rc(bill_number="SB 87", vote_id="kf_1", yea_count=30, nay_count=10),
            _make_rc(bill_number="SB 87", vote_id="kf_2", yea_count=30, nay_count=10),
        ]
        je = [
            _make_rc(bill_number="SB 87", vote_id="je_1", yea_count=30, nay_count=10),
            _make_rc(bill_number="SB 87", vote_id="je_2", yea_count=30, nay_count=10),
        ]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 2
        assert len(unmatched) == 0

    def test_multi_motion_kf_extra_unmatched(self):
        """Extra KF rollcall with no matching tally → unmatched."""
        kf = [
            _make_rc(bill_number="SB 1", vote_id="kf_1", yea_count=30, nay_count=10),
            _make_rc(bill_number="SB 1", vote_id="kf_2", yea_count=20, nay_count=20),
        ]
        je = [
            _make_rc(bill_number="SB 1", vote_id="je_1", yea_count=30, nay_count=10),
        ]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 1
        assert len(unmatched) == 1
        assert matched[0][0].vote_id == "kf_1"
        assert unmatched[0].vote_id == "kf_2"

    def test_multi_motion_nv_total_matching(self):
        """JE nv_total = not_voting + absent_not_voting for tally key."""
        kf = [
            _make_rc(
                bill_number="SB 1", vote_id="kf_1",
                yea_count=30, nay_count=5, not_voting_count=5,
            ),
        ]
        je = [
            _make_rc(
                bill_number="SB 1", vote_id="je_1",
                yea_count=30, nay_count=5, not_voting_count=2, absent_not_voting_count=3,
            ),
        ]
        matched, unmatched = find_matches(kf, je)
        assert len(matched) == 1
        assert len(unmatched) == 0


# ── TestCompareTallies ───────────────────────────────────────────────────


class TestCompareTallies:
    """Run: uv run pytest tests/test_kanfocus_crossval.py -k TestCompareTallies -v"""

    def test_identical_tallies(self):
        kf_rc = _make_rc(yea_count=30, nay_count=10, not_voting_count=0, vote_id="kf_1")
        je_rc = _make_rc(yea_count=30, nay_count=10, not_voting_count=0, vote_id="je_1")
        comp = compare_rollcall(kf_rc, je_rc, [], [])
        assert comp.yea_match is True
        assert comp.nay_match is True
        assert comp.nv_match is True
        assert comp.nv_compatible is True

    def test_yea_mismatch(self):
        kf_rc = _make_rc(yea_count=30, vote_id="kf_1")
        je_rc = _make_rc(yea_count=29, vote_id="je_1")
        comp = compare_rollcall(kf_rc, je_rc, [], [])
        assert comp.yea_match is False
        assert comp.nay_match is True

    def test_nv_compatible(self):
        """KF not_voting=5 should be compatible with JE not_voting=2 + anv=3."""
        kf_rc = _make_rc(not_voting_count=5, vote_id="kf_1")
        je_rc = _make_rc(not_voting_count=2, absent_not_voting_count=3, vote_id="je_1")
        comp = compare_rollcall(kf_rc, je_rc, [], [])
        assert comp.nv_match is False  # not exact
        assert comp.nv_compatible is True  # but compatible

    def test_nv_incompatible(self):
        """KF not_voting=5 with JE not_voting=2 + anv=2 is a genuine mismatch."""
        kf_rc = _make_rc(not_voting_count=5, vote_id="kf_1")
        je_rc = _make_rc(not_voting_count=2, absent_not_voting_count=2, vote_id="je_1")
        comp = compare_rollcall(kf_rc, je_rc, [], [])
        assert comp.nv_match is False
        assert comp.nv_compatible is False

    def test_passed_match(self):
        kf_rc = _make_rc(passed=True, vote_id="kf_1")
        je_rc = _make_rc(passed=True, vote_id="je_1")
        comp = compare_rollcall(kf_rc, je_rc, [], [])
        assert comp.passed_match is True

    def test_passed_disagree(self):
        kf_rc = _make_rc(passed=True, vote_id="kf_1")
        je_rc = _make_rc(passed=False, vote_id="je_1")
        comp = compare_rollcall(kf_rc, je_rc, [], [])
        assert comp.passed_match is False


# ── TestCompareIndividualVotes ───────────────────────────────────────────


class TestCompareIndividualVotes:
    """Run: uv run pytest tests/test_kanfocus_crossval.py -k TestCompareIndividualVotes -v"""

    def test_all_agree(self):
        kf = [_make_vote(slug="sen_a_b_1", vote="Yea")]
        je = [_make_vote(slug="sen_a_b_1", vote="Yea")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 0
        assert len(kf_only) == 0
        assert len(je_only) == 0

    def test_genuine_mismatch(self):
        kf = [_make_vote(slug="sen_a_b_1", vote="Yea")]
        je = [_make_vote(slug="sen_a_b_1", vote="Nay")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 1
        assert mismatches[0].compatible is False
        assert mismatches[0].kf_vote == "Yea"
        assert mismatches[0].je_vote == "Nay"

    def test_anv_nv_compatible(self):
        """KF 'Not Voting' vs JE 'Absent and Not Voting' is compatible."""
        kf = [_make_vote(slug="sen_a_b_1", vote="Not Voting")]
        je = [_make_vote(slug="sen_a_b_1", vote="Absent and Not Voting")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 1
        assert mismatches[0].compatible is True

    def test_kf_only_legislator(self):
        kf = [_make_vote(slug="sen_a_b_1"), _make_vote(slug="sen_c_d_1")]
        je = [_make_vote(slug="sen_a_b_1")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert kf_only == ["sen_c_d_1"]
        assert je_only == []

    def test_je_only_legislator(self):
        kf = [_make_vote(slug="sen_a_b_1")]
        je = [_make_vote(slug="sen_a_b_1"), _make_vote(slug="sen_c_d_1")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert kf_only == []
        assert je_only == ["sen_c_d_1"]

    def test_multiple_legislators_mixed(self):
        kf = [
            _make_vote(slug="sen_a_b_1", vote="Yea"),
            _make_vote(slug="sen_c_d_1", vote="Not Voting"),
            _make_vote(slug="sen_e_f_1", vote="Nay"),
        ]
        je = [
            _make_vote(slug="sen_a_b_1", vote="Yea"),
            _make_vote(slug="sen_c_d_1", vote="Absent and Not Voting"),
            _make_vote(slug="sen_e_f_1", vote="Nay"),
        ]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        # Only the NV/ANV pair shows up as a compatible mismatch
        assert len(mismatches) == 1
        assert mismatches[0].slug == "sen_c_d_1"
        assert mismatches[0].compatible is True

    def test_empty_vote_lists(self):
        mismatches, kf_only, je_only = compare_individual_votes([], [])
        assert len(mismatches) == 0
        assert len(kf_only) == 0
        assert len(je_only) == 0

    # -- Name-based fallback --

    def test_name_fallback_matches_different_slugs(self):
        """Different slugs but same name → matched by name, not in kf/je_only."""
        kf = [_make_vote(slug="rep_smith_john_1", name="John Smith", vote="Yea")]
        je = [_make_vote(slug="rep_smith_john_2", name="John Smith", vote="Yea")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 0
        assert len(kf_only) == 0
        assert len(je_only) == 0

    def test_name_fallback_reports_mismatch(self):
        """Name-matched but different votes → reported as mismatch."""
        kf = [_make_vote(slug="rep_smith_john_1", name="John Smith", vote="Yea")]
        je = [_make_vote(slug="rep_smith_john_2", name="John Smith", vote="Nay")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 1
        assert mismatches[0].compatible is False
        assert len(kf_only) == 0
        assert len(je_only) == 0

    def test_name_fallback_anv_compatible(self):
        """Name-matched with ANV/NV difference → compatible mismatch."""
        kf = [_make_vote(slug="kf_slug_1", name="John Smith", vote="Not Voting")]
        je = [_make_vote(slug="je_slug_1", name="John Smith", vote="Absent and Not Voting")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 1
        assert mismatches[0].compatible is True

    def test_name_fallback_no_false_match(self):
        """Different names on different slugs → remain in kf/je_only."""
        kf = [_make_vote(slug="kf_slug_1", name="John Smith", vote="Yea")]
        je = [_make_vote(slug="je_slug_1", name="Jane Doe", vote="Yea")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert kf_only == ["kf_slug_1"]
        assert je_only == ["je_slug_1"]

    def test_name_fallback_with_middle_initial(self):
        """Name normalization strips middle initials for fallback match."""
        kf = [_make_vote(slug="kf_slug_1", name="Stephen R. Morris", vote="Yea")]
        je = [_make_vote(slug="je_slug_1", name="Stephen Morris", vote="Yea")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(kf_only) == 0
        assert len(je_only) == 0

    def test_name_fallback_last_name_match(self):
        """KF full name matches JE last-name-only via last-name fallback."""
        kf = [_make_vote(slug="rep_barrett_brad_1", name="Brad Barrett", vote="Yea")]
        je = [_make_vote(slug="rep_barrett_bradley_1", name="Barrett", vote="Yea")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(kf_only) == 0
        assert len(je_only) == 0

    def test_name_fallback_last_name_mismatch_reported(self):
        """Last-name match with different votes → reported as mismatch."""
        kf = [_make_vote(slug="kf_1", name="Chip VanHouden", vote="Yea")]
        je = [_make_vote(slug="je_1", name="VanHouden", vote="Nay")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert len(mismatches) == 1
        assert mismatches[0].compatible is False
        assert len(kf_only) == 0
        assert len(je_only) == 0

    def test_name_fallback_last_name_no_false_positive(self):
        """Different last names don't produce false matches."""
        kf = [_make_vote(slug="kf_1", name="Brad Barrett", vote="Yea")]
        je = [_make_vote(slug="je_1", name="Smith", vote="Yea")]
        mismatches, kf_only, je_only = compare_individual_votes(kf, je)
        assert kf_only == ["kf_1"]
        assert je_only == ["je_1"]


# ── TestFormatReport ─────────────────────────────────────────────────────


class TestFormatReport:
    """Run: uv run pytest tests/test_kanfocus_crossval.py -k TestFormatReport -v"""

    def test_empty_report(self):
        report = CrossValReport(session_label="91st (2025-2026)")
        md = format_report(report)
        assert "# KanFocus Cross-Validation Report" in md
        assert "No overlapping rollcalls found" in md

    def test_basic_structure(self):
        report = CrossValReport(
            session_label="91st (2025-2026)",
            total_kf_rollcalls=100,
            total_je_rollcalls=200,
            matched_rollcalls=90,
            unmatched_kf_rollcalls=10,
            tally_perfect=85,
            tally_compatible=3,
            tally_mismatch=2,
            passed_agree=88,
            passed_disagree=2,
            individual_perfect=80,
            individual_compatible=5,
            individual_mismatch=5,
            total_genuine_mismatches=8,
            total_compatible_mismatches=12,
        )
        md = format_report(report)
        assert "## Summary" in md
        assert "## Tally Agreement" in md
        assert "## Passed/Failed Agreement" in md
        assert "## Individual Vote Agreement" in md
        assert "90" in md  # matched rollcalls
        assert "85" in md  # tally_perfect

    def test_normalizations_section(self):
        report = CrossValReport(
            session_label="test",
            matched_rollcalls=1,
            tally_perfect=1,
            passed_agree=1,
            individual_perfect=1,
            normalizations=[("Sub for HB 2007", "HB 2007")],
        )
        md = format_report(report)
        assert "## Bill Number Normalizations" in md
        assert "Sub for HB 2007" in md
        assert "HB 2007" in md

    def test_unmatched_section(self):
        report = CrossValReport(
            session_label="test",
            matched_rollcalls=1,
            tally_perfect=1,
            passed_agree=1,
            individual_perfect=1,
            unmatched_kf_bills=["SB 999 (Senate, 03/20/2025)"],
        )
        md = format_report(report)
        assert "## Unmatched KF Rollcalls" in md
        assert "SB 999" in md

    def test_mismatch_detail_sections(self):
        comp = RollCallComparison(
            bill_number="SB 1",
            chamber="Senate",
            vote_date="03/20/2025",
            kf_vote_id="kf_1",
            je_vote_id="je_1",
            yea_match=False,
            nay_match=True,
            present_match=True,
            nv_match=True,
            nv_compatible=True,
            passed_match=True,
            mismatches=(
                VoteMismatch(
                    slug="sen_a_b_1",
                    name="A B",
                    kf_vote="Yea",
                    je_vote="Nay",
                    compatible=False,
                ),
            ),
            kf_only_slugs=(),
            je_only_slugs=(),
        )
        report = CrossValReport(
            session_label="test",
            matched_rollcalls=1,
            tally_mismatch=1,
            individual_mismatch=1,
            total_genuine_mismatches=1,
            comparisons=[comp],
        )
        md = format_report(report)
        assert "## Tally Mismatches (Detail)" in md
        assert "DIFF" in md  # yea_match is False
        assert "## Individual Vote Mismatches (Detail)" in md
        assert "Genuine mismatches" in md
        assert "sen_a_b_1" in md

    def test_deduplicates_normalizations(self):
        report = CrossValReport(
            session_label="test",
            matched_rollcalls=1,
            tally_perfect=1,
            passed_agree=1,
            individual_perfect=1,
            normalizations=[
                ("Sub for HB 100", "HB 100"),
                ("Sub for HB 100", "HB 100"),
            ],
        )
        md = format_report(report)
        # Should only appear once in the table
        assert md.count("Sub for HB 100") == 1
