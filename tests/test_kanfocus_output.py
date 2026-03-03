"""
Tests for KanFocus output conversion and gap-fill merge.

Run: uv run pytest tests/test_kanfocus_output.py -v
"""

import pytest

from tallgrass.kanfocus.models import KanFocusLegislator, KanFocusVoteRecord
from tallgrass.kanfocus.output import (
    _classify_vote_type,
    _convert_date,
    _derive_passed,
    _parse_bool,
    convert_to_standard,
    merge_gap_fill,
)

pytestmark = pytest.mark.scraper


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_record(
    vote_num: int = 33,
    year: int = 2011,
    chamber: str = "S",
    date: str = "02/03/2011",
    bill_number: str = "SB 13",
    question: str = "On final action",
    result: str = "Passed",
    legislators: tuple = (),
) -> KanFocusVoteRecord:
    return KanFocusVoteRecord(
        vote_num=vote_num,
        year=year,
        chamber=chamber,
        date=date,
        bill_number=bill_number,
        question=question,
        result=result,
        yea_count=38,
        nay_count=0,
        present_count=0,
        not_voting_count=1,
        legislators=legislators,
        source_url="https://kanfocus.com/test",
    )


def _make_legislator(
    name: str = "Steve Abrams",
    party: str = "R",
    district: str = "32nd",
    category: str = "Yea",
) -> KanFocusLegislator:
    return KanFocusLegislator(
        name=name,
        party=party,
        district=district,
        vote_category=category,
    )


# ── _convert_date() ───────────────────────────────────────────────────────


class TestConvertDate:
    """Convert KanFocus date format to tallgrass format."""

    def test_standard_date(self):
        dt, d = _convert_date("02/03/2011")
        assert dt == "2011-02-03T00:00:00"
        assert d == "02/03/2011"

    def test_december_date(self):
        dt, d = _convert_date("12/31/2009")
        assert dt == "2009-12-31T00:00:00"
        assert d == "12/31/2009"

    def test_invalid_date(self):
        dt, d = _convert_date("not-a-date")
        assert dt == ""
        assert d == "not-a-date"


# ── _derive_passed() ──────────────────────────────────────────────────────


class TestDerivePassed:
    """Derive passed boolean from result text."""

    def test_passed(self):
        assert _derive_passed("Passed") is True

    def test_failed(self):
        assert _derive_passed("Failed") is False

    def test_adopted(self):
        assert _derive_passed("Adopted") is True

    def test_rejected(self):
        assert _derive_passed("Rejected") is False

    def test_not_passed(self):
        assert _derive_passed("Not Passed") is False

    def test_empty(self):
        assert _derive_passed("") is None

    def test_unknown(self):
        assert _derive_passed("Tabled") is None


# ── _classify_vote_type() ─────────────────────────────────────────────────


class TestClassifyVoteType:
    """Classify question text into vote_type."""

    def test_final_action(self):
        vt, _ = _classify_vote_type("Final Action")
        assert vt == "Final Action"

    def test_emergency_final_action(self):
        vt, _ = _classify_vote_type("Emergency Final Action")
        assert vt == "Emergency Final Action"

    def test_veto_override(self):
        vt, _ = _classify_vote_type("Motion to override veto")
        assert vt == "Veto Override"

    def test_concurrence(self):
        vt, _ = _classify_vote_type("On concurring with amendments")
        assert vt == "Concurrence"

    def test_plain_question(self):
        vt, result = _classify_vote_type("On agreeing to the amendment")
        assert vt == ""
        assert result == "On agreeing to the amendment"

    def test_empty(self):
        assert _classify_vote_type("") == ("", "")


# ── convert_to_standard() ─────────────────────────────────────────────────


class TestConvertToStandard:
    """Convert KanFocus records to standard tallgrass format."""

    def test_produces_rollcall(self):
        record = _make_record()
        votes, rollcalls, legislators = convert_to_standard([record], "84th (2011-2012)", {})
        assert len(rollcalls) == 1
        assert rollcalls[0].vote_id == "kf_33_2011_S"
        assert rollcalls[0].chamber == "Senate"

    def test_produces_individual_votes(self):
        leg = _make_legislator()
        record = _make_record(legislators=(leg,))
        votes, _, _ = convert_to_standard([record], "84th (2011-2012)", {})
        assert len(votes) == 1
        assert votes[0].vote == "Yea"
        assert votes[0].legislator_name == "Steve Abrams"

    def test_vote_datetime_format(self):
        record = _make_record(date="02/03/2011")
        _, rollcalls, _ = convert_to_standard([record], "test", {})
        assert rollcalls[0].vote_datetime == "2011-02-03T00:00:00"
        assert rollcalls[0].vote_date == "02/03/2011"

    def test_passed_derived(self):
        record = _make_record(result="Passed")
        _, rollcalls, _ = convert_to_standard([record], "test", {})
        assert rollcalls[0].passed is True

    def test_failed_derived(self):
        record = _make_record(result="Failed")
        _, rollcalls, _ = convert_to_standard([record], "test", {})
        assert rollcalls[0].passed is False

    def test_legislator_dict_built(self):
        leg = _make_legislator(name="Steve Abrams", party="R", district="32nd")
        record = _make_record(chamber="S", legislators=(leg,))
        _, _, legislators = convert_to_standard([record], "test", {})
        assert len(legislators) == 1
        slug = list(legislators.keys())[0]
        assert legislators[slug]["party"] == "Republican"
        assert legislators[slug]["chamber"] == "Senate"

    def test_house_chamber_name(self):
        record = _make_record(chamber="H")
        _, rollcalls, _ = convert_to_standard([record], "test", {})
        assert rollcalls[0].chamber == "House"

    def test_session_label_preserved(self):
        record = _make_record()
        _, rollcalls, _ = convert_to_standard([record], "84th (2011-2012)", {})
        assert rollcalls[0].session == "84th (2011-2012)"

    def test_source_url_preserved(self):
        record = _make_record()
        _, rollcalls, _ = convert_to_standard([record], "test", {})
        assert rollcalls[0].vote_url == "https://kanfocus.com/test"


# ── _parse_bool() ─────────────────────────────────────────────────────────


class TestParseBool:
    """Parse boolean strings from CSV."""

    def test_true(self):
        assert _parse_bool("True") is True

    def test_false(self):
        assert _parse_bool("False") is False

    def test_empty(self):
        assert _parse_bool("") is None

    def test_none_string(self):
        assert _parse_bool("None") is None


# ── merge_gap_fill() deduplication ───────────────────────────────────────


class TestGapFillDedup:
    """Gap-fill should only add kf_ rollcalls that don't overlap with existing je_ data."""

    def _write_csv(self, path, header, rows):
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in rows:
                w.writerow(row)

    def _rollcall_header(self):
        return [
            "session", "bill_number", "bill_title", "vote_id", "vote_url",
            "vote_datetime", "vote_date", "chamber", "motion", "vote_type",
            "result", "short_title", "sponsor", "sponsor_slugs",
            "yea_count", "nay_count", "present_passing_count",
            "absent_not_voting_count", "not_voting_count", "total_votes", "passed",
        ]

    def _vote_header(self):
        return [
            "session", "bill_number", "bill_title", "vote_id",
            "vote_datetime", "vote_date", "chamber", "motion",
            "legislator_name", "legislator_slug", "vote",
        ]

    def _leg_header(self):
        return ["name", "full_name", "slug", "chamber", "party", "district", "member_url", "ocd_id"]

    def test_skips_duplicate_rollcalls(self, tmp_path):
        """kf_ rollcalls matching existing je_ by bill+chamber+date are skipped."""
        # Existing je_ rollcall for SB 13 / Senate / 02/03/2011
        self._write_csv(
            tmp_path / "test_rollcalls.csv",
            self._rollcall_header(),
            [["test", "SB 13", "", "je_123", "", "", "02/03/2011", "Senate",
              "", "", "", "", "", "", 38, 0, 0, 0, 1, 38, "True"]],
        )
        self._write_csv(
            tmp_path / "test_votes.csv",
            self._vote_header(),
            [["test", "SB 13", "", "je_123", "", "02/03/2011", "Senate", "",
              "Abrams", "sen_abrams_steve_1", "Yea"]],
        )
        self._write_csv(tmp_path / "test_legislators.csv", self._leg_header(), [])

        # New kf_ rollcall for same bill+chamber+date — should be skipped
        leg = _make_legislator()
        record = _make_record(bill_number="SB 13", chamber="S", date="02/03/2011")
        new_votes, new_rollcalls, new_legs = convert_to_standard([record], "test", {})

        merge_gap_fill(tmp_path, "test", new_votes, new_rollcalls, new_legs)

        # Read merged rollcalls — should only have the original je_
        import csv

        with open(tmp_path / "test_rollcalls.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["vote_id"] == "je_123"

    def test_keeps_genuinely_new_rollcalls(self, tmp_path):
        """kf_ rollcalls for bills NOT in existing data are kept."""
        # Existing je_ rollcall for SB 13
        self._write_csv(
            tmp_path / "test_rollcalls.csv",
            self._rollcall_header(),
            [["test", "SB 13", "", "je_123", "", "", "02/03/2011", "Senate",
              "", "", "", "", "", "", 38, 0, 0, 0, 1, 38, "True"]],
        )
        self._write_csv(
            tmp_path / "test_votes.csv",
            self._vote_header(),
            [["test", "SB 13", "", "je_123", "", "02/03/2011", "Senate", "",
              "Abrams", "sen_abrams_steve_1", "Yea"]],
        )
        self._write_csv(tmp_path / "test_legislators.csv", self._leg_header(), [])

        # New kf_ rollcall for SB 99 — different bill, should be kept
        record = _make_record(vote_num=50, bill_number="SB 99", chamber="S", date="03/15/2011")
        new_votes, new_rollcalls, new_legs = convert_to_standard([record], "test", {})

        merge_gap_fill(tmp_path, "test", new_votes, new_rollcalls, new_legs)

        import csv

        with open(tmp_path / "test_rollcalls.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        vote_ids = {r["vote_id"] for r in rows}
        assert "je_123" in vote_ids
        assert "kf_50_2011_S" in vote_ids

    def test_idempotent_rerun(self, tmp_path):
        """Running gap-fill twice produces the same result."""
        self._write_csv(
            tmp_path / "test_rollcalls.csv",
            self._rollcall_header(),
            [["test", "SB 13", "", "je_123", "", "", "02/03/2011", "Senate",
              "", "", "", "", "", "", 38, 0, 0, 0, 1, 38, "True"]],
        )
        self._write_csv(
            tmp_path / "test_votes.csv",
            self._vote_header(),
            [["test", "SB 13", "", "je_123", "", "02/03/2011", "Senate", "",
              "Abrams", "sen_abrams_steve_1", "Yea"]],
        )
        self._write_csv(tmp_path / "test_legislators.csv", self._leg_header(), [])

        record = _make_record(vote_num=50, bill_number="SB 99", chamber="S", date="03/15/2011")
        new_votes, new_rollcalls, new_legs = convert_to_standard([record], "test", {})

        # Run twice
        merge_gap_fill(tmp_path, "test", new_votes, new_rollcalls, new_legs)
        merge_gap_fill(tmp_path, "test", new_votes, new_rollcalls, new_legs)

        import csv

        with open(tmp_path / "test_rollcalls.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
