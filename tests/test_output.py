"""
Tests for CSV export in output.py.

Verifies that save_csvs() writes three correctly structured CSV files with
the right filenames, headers, row counts, and field ordering.

Run: uv run pytest tests/test_output.py -v
"""

import csv

from ks_vote_scraper.models import IndividualVote, RollCall
from ks_vote_scraper.output import save_csvs

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_vote(slug: str = "sen_doe_john_1", vote: str = "Yea") -> IndividualVote:
    return IndividualVote(
        session="91st (2025-2026)",
        bill_number="SB 1",
        bill_title="AN ACT concerning taxation",
        vote_id="je_20250320203513",
        vote_datetime="2025-03-20T20:35:13",
        vote_date="03/20/2025",
        chamber="Senate",
        motion="Emergency Final Action",
        legislator_name="Doe, John",
        legislator_slug=slug,
        vote=vote,
    )


def _make_rollcall(vote_id: str = "je_20250320203513") -> RollCall:
    return RollCall(
        session="91st (2025-2026)",
        bill_number="SB 1",
        bill_title="AN ACT concerning taxation",
        vote_id=vote_id,
        vote_url="https://example.com/vote",
        vote_datetime="2025-03-20T20:35:13",
        vote_date="03/20/2025",
        chamber="Senate",
        motion="Emergency Final Action",
        vote_type="Emergency Final Action",
        result="Passed as amended",
        short_title="Taxation",
        sponsor="Senator Steffen",
        yea_count=33,
        nay_count=5,
        total_votes=40,
        passed=True,
    )


def _make_legislator(slug: str, name: str, party: str = "Republican") -> dict:
    return {
        "name": name.split()[-1],
        "full_name": name,
        "slug": slug,
        "chamber": "Senate",
        "party": party,
        "district": "1",
        "member_url": f"https://example.com/members/{slug}/",
    }


# ── save_csvs() ─────────────────────────────────────────────────────────────


class TestSaveCsvs:
    """CSV export creates correct files with expected structure."""

    def test_creates_three_files(self, tmp_path):
        save_csvs(
            output_dir=tmp_path,
            output_name="test_session",
            individual_votes=[_make_vote()],
            rollcalls=[_make_rollcall()],
            legislators={"sen_doe_john_1": _make_legislator("sen_doe_john_1", "John Doe")},
        )
        assert (tmp_path / "test_session_votes.csv").exists()
        assert (tmp_path / "test_session_rollcalls.csv").exists()
        assert (tmp_path / "test_session_legislators.csv").exists()

    def test_votes_csv_headers(self, tmp_path):
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[_make_vote()],
            rollcalls=[],
            legislators={},
        )
        with open(tmp_path / "test_votes.csv") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
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

    def test_rollcalls_csv_headers(self, tmp_path):
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[_make_rollcall()],
            legislators={},
        )
        with open(tmp_path / "test_rollcalls.csv") as f:
            reader = csv.DictReader(f)
            assert "yea_count" in reader.fieldnames
            assert "passed" in reader.fieldnames
            assert reader.fieldnames[0] == "session"

    def test_legislators_csv_headers(self, tmp_path):
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={"sen_doe_john_1": _make_legislator("sen_doe_john_1", "John Doe")},
        )
        with open(tmp_path / "test_legislators.csv") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "name",
                "full_name",
                "slug",
                "chamber",
                "party",
                "district",
                "member_url",
            ]

    def test_row_counts(self, tmp_path):
        votes = [_make_vote("sen_a_a_1", "Yea"), _make_vote("sen_b_b_1", "Nay")]
        rcs = [_make_rollcall("rc1"), _make_rollcall("rc2")]
        legs = {
            "sen_a_a_1": _make_legislator("sen_a_a_1", "Alice A"),
            "sen_b_b_1": _make_legislator("sen_b_b_1", "Bob B"),
        }
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=votes,
            rollcalls=rcs,
            legislators=legs,
        )
        with open(tmp_path / "test_votes.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2

        with open(tmp_path / "test_rollcalls.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2

        with open(tmp_path / "test_legislators.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 2

    def test_legislators_sorted_by_slug(self, tmp_path):
        legs = {
            "sen_z_z_1": _make_legislator("sen_z_z_1", "Zara Z"),
            "sen_a_a_1": _make_legislator("sen_a_a_1", "Alice A"),
            "sen_m_m_1": _make_legislator("sen_m_m_1", "Mike M"),
        }
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators=legs,
        )
        with open(tmp_path / "test_legislators.csv") as f:
            rows = list(csv.DictReader(f))
            slugs = [r["slug"] for r in rows]
            assert slugs == ["sen_a_a_1", "sen_m_m_1", "sen_z_z_1"]

    def test_empty_data(self, tmp_path):
        """Empty inputs still produce CSVs with headers only."""
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators={},
        )
        with open(tmp_path / "test_votes.csv") as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 0

    def test_vote_data_roundtrip(self, tmp_path):
        """Vote field values survive the CSV write/read cycle."""
        vote = _make_vote()
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[vote],
            rollcalls=[],
            legislators={},
        )
        with open(tmp_path / "test_votes.csv") as f:
            row = next(csv.DictReader(f))
            assert row["bill_number"] == "SB 1"
            assert row["vote"] == "Yea"
            assert row["legislator_slug"] == "sen_doe_john_1"

    def test_rollcall_passed_values(self, tmp_path):
        """Boolean and None passed values written correctly."""
        rc_true = _make_rollcall("rc1")
        rc_none = RollCall(
            session="s", bill_number="SB 2", bill_title="t", vote_id="rc2",
            vote_url="u", vote_datetime="dt", vote_date="d", chamber="Senate",
            motion="m", vote_type="vt", result="unknown", short_title="st",
            sponsor="sp", passed=None,
        )
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[rc_true, rc_none],
            legislators={},
        )
        with open(tmp_path / "test_rollcalls.csv") as f:
            rows = list(csv.DictReader(f))
            assert rows[0]["passed"] == "True"
            assert rows[1]["passed"] == ""

    def test_missing_legislator_fields_default_empty(self, tmp_path):
        """Legislators with missing keys get empty strings."""
        legs = {"sen_x_x_1": {"name": "X", "slug": "sen_x_x_1"}}
        save_csvs(
            output_dir=tmp_path,
            output_name="test",
            individual_votes=[],
            rollcalls=[],
            legislators=legs,
        )
        with open(tmp_path / "test_legislators.csv") as f:
            row = next(csv.DictReader(f))
            assert row["party"] == ""
            assert row["full_name"] == ""
