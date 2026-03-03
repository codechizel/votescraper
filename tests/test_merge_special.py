"""
Tests for merge_special.py — merging special session CSVs into parent bienniums.

Covers: correct row counts, idempotency, column alignment (sponsor_slugs
mismatch), legislator dedup, parent_session property, CLI --merge-special flag.

Run: uv run pytest tests/test_merge_special.py -v
"""

import polars as pl
import pytest

from tallgrass.merge_special import _merge_csv, merge_special_into_parent
from tallgrass.session import KSSession

pytestmark = pytest.mark.scraper

# ── parent_session property ─────────────────────────────────────────────────


class TestParentSession:
    """KSSession.parent_session maps special years to correct parent bienniums."""

    def test_2024_special_parent(self):
        s = KSSession(start_year=2024, special=True)
        assert s.parent_session == KSSession(start_year=2023)

    def test_2021_special_parent(self):
        s = KSSession(start_year=2021, special=True)
        assert s.parent_session == KSSession(start_year=2021)

    def test_2020_special_parent(self):
        s = KSSession(start_year=2020, special=True)
        assert s.parent_session == KSSession(start_year=2019)

    def test_2016_special_parent(self):
        s = KSSession(start_year=2016, special=True)
        assert s.parent_session == KSSession(start_year=2015)

    def test_2013_special_parent(self):
        s = KSSession(start_year=2013, special=True)
        assert s.parent_session == KSSession(start_year=2013)

    def test_regular_session_returns_self(self):
        s = KSSession(start_year=2025)
        assert s.parent_session is s

    def test_parent_is_not_special(self):
        s = KSSession(start_year=2020, special=True)
        assert s.parent_session.special is False

    def test_all_five_specials(self):
        """Verify the full mapping table from the design."""
        expected = {
            2013: 2013,
            2016: 2015,
            2020: 2019,
            2021: 2021,
            2024: 2023,
        }
        for special_year, parent_start in expected.items():
            s = KSSession(start_year=special_year, special=True)
            assert s.parent_session.start_year == parent_start


# ── _merge_csv internals ────────────────────────────────────────────────────


class TestMergeCsvBasic:
    """Low-level merge of individual CSV files."""

    def test_votes_merged_correctly(self, tmp_path):
        """Parent + special votes concat without dedup."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame(
            {
                "session": ["88th (2019-2020)"] * 3,
                "vote_id": ["v1", "v2", "v3"],
                "legislator_slug": ["rep_a", "rep_b", "rep_c"],
                "vote": ["Yea", "Nay", "Yea"],
            }
        )
        parent_df.write_csv(parent_dir / "parent_votes.csv")

        special_df = pl.DataFrame(
            {
                "session": ["2020 Special"] * 2,
                "vote_id": ["v4", "v5"],
                "legislator_slug": ["rep_a", "rep_b"],
                "vote": ["Nay", "Yea"],
            }
        )
        special_df.write_csv(special_dir / "special_votes.csv")

        added = _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="votes",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        assert added == 2
        result = pl.read_csv(parent_dir / "parent_votes.csv")
        assert result.height == 5

    def test_idempotent_merge(self, tmp_path):
        """Running merge twice produces identical output."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame(
            {
                "session": ["88th (2019-2020)"] * 2,
                "vote_id": ["v1", "v2"],
                "vote": ["Yea", "Nay"],
            }
        )
        parent_df.write_csv(parent_dir / "parent_votes.csv")

        special_df = pl.DataFrame(
            {
                "session": ["2020 Special"] * 2,
                "vote_id": ["v3", "v4"],
                "vote": ["Yea", "Yea"],
            }
        )
        special_df.write_csv(special_dir / "special_votes.csv")

        kwargs = dict(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="votes",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        # First merge
        _merge_csv(**kwargs)
        first_result = pl.read_csv(parent_dir / "parent_votes.csv")

        # Second merge (idempotent)
        added = _merge_csv(**kwargs)
        second_result = pl.read_csv(parent_dir / "parent_votes.csv")

        assert first_result.shape == second_result.shape
        assert first_result.to_dicts() == second_result.to_dicts()
        assert added == 2  # Same count each time

    def test_column_alignment_adds_missing(self, tmp_path):
        """Parent lacks sponsor_slugs -> gets added with empty string."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame(
            {
                "session": ["88th (2019-2020)"],
                "vote_id": ["v1"],
                "sponsor": ["Smith"],
            }
        )
        parent_df.write_csv(parent_dir / "parent_rollcalls.csv")

        special_df = pl.DataFrame(
            {
                "session": ["2020 Special"],
                "vote_id": ["v2"],
                "sponsor": ["Jones"],
                "sponsor_slugs": ["rep_jones_1"],
            }
        )
        special_df.write_csv(special_dir / "special_rollcalls.csv")

        _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="rollcalls",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        result = pl.read_csv(parent_dir / "parent_rollcalls.csv")
        assert "sponsor_slugs" in result.columns
        assert result.height == 2
        # Parent row gets empty sponsor_slugs
        parent_row = result.filter(pl.col("session") == "88th (2019-2020)")
        assert parent_row["sponsor_slugs"][0] == ""

    def test_special_gets_parent_only_columns(self, tmp_path):
        """Special lacks a column that parent has -> gets added."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame(
            {
                "session": ["88th (2019-2020)"],
                "vote_id": ["v1"],
                "extra_col": ["data"],
            }
        )
        parent_df.write_csv(parent_dir / "parent_votes.csv")

        special_df = pl.DataFrame(
            {
                "session": ["2020 Special"],
                "vote_id": ["v2"],
            }
        )
        special_df.write_csv(special_dir / "special_votes.csv")

        _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="votes",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        result = pl.read_csv(parent_dir / "parent_votes.csv")
        assert "extra_col" in result.columns
        special_row = result.filter(pl.col("session") == "2020 Special")
        assert special_row["extra_col"][0] == ""

    def test_legislator_dedup_parent_wins(self, tmp_path):
        """Overlapping slugs: parent row kept, not special's."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame(
            {
                "slug": ["rep_a", "rep_b"],
                "name": ["Alice", "Bob"],
                "member_url": ["/parent/a", "/parent/b"],
            }
        )
        parent_df.write_csv(parent_dir / "parent_legislators.csv")

        special_df = pl.DataFrame(
            {
                "slug": ["rep_a", "rep_c"],
                "name": ["Alice", "Charlie"],
                "member_url": ["/special/a", "/special/c"],
            }
        )
        special_df.write_csv(special_dir / "special_legislators.csv")

        added = _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="legislators",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=["slug"],
        )

        assert added == 1  # Only rep_c is new
        result = pl.read_csv(parent_dir / "parent_legislators.csv")
        assert result.height == 3
        # rep_a should have parent's member_url
        rep_a = result.filter(pl.col("slug") == "rep_a")
        assert rep_a["member_url"][0] == "/parent/a"

    def test_missing_special_csv_returns_zero(self, tmp_path):
        """If special CSV doesn't exist, return 0 and don't touch parent."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame({"session": ["test"], "vote_id": ["v1"]})
        parent_df.write_csv(parent_dir / "parent_votes.csv")

        added = _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="votes",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        assert added == 0

    def test_missing_parent_csv_returns_zero(self, tmp_path):
        """If parent CSV doesn't exist, return 0."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        special_df = pl.DataFrame({"session": ["2020 Special"], "vote_id": ["v1"]})
        special_df.write_csv(special_dir / "special_votes.csv")

        added = _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="votes",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        assert added == 0

    def test_empty_special_csv_returns_zero(self, tmp_path):
        """If special CSV is header-only (no data rows), return 0."""
        parent_dir = tmp_path / "parent"
        special_dir = tmp_path / "special"
        parent_dir.mkdir()
        special_dir.mkdir()

        parent_df = pl.DataFrame({"session": ["test"], "vote_id": ["v1"]})
        parent_df.write_csv(parent_dir / "parent_votes.csv")

        # Write header-only CSV
        (special_dir / "special_votes.csv").write_text("session,vote_id\n")

        added = _merge_csv(
            parent_dir=parent_dir,
            special_dir=special_dir,
            filename_stem="votes",
            parent_name="parent",
            special_name="special",
            special_label="2020 Special",
            dedup_subset=None,
        )

        assert added == 0


# ── merge_special_into_parent ───────────────────────────────────────────────


class TestMergeSpecialIntoParent:
    """End-to-end merge of a special session using real KSSession paths."""

    def test_full_merge(self, tmp_path, monkeypatch):
        """Simulates 2020 special merge with tmp data dirs."""
        # Create parent dir structure
        parent = KSSession(start_year=2019)
        special = KSSession(start_year=2020, special=True)

        parent_dir = tmp_path / "data" / "kansas" / parent.output_name
        special_dir = tmp_path / "data" / "kansas" / special.output_name
        parent_dir.mkdir(parents=True)
        special_dir.mkdir(parents=True)

        # Monkeypatch data_dir to use tmp_path
        monkeypatch.setattr(
            KSSession,
            "data_dir",
            property(lambda self: tmp_path / "data" / "kansas" / self.output_name),
        )

        # Write parent CSVs
        pl.DataFrame(
            {
                "session": ["88th (2019-2020)"] * 3,
                "vote_id": ["v1", "v2", "v3"],
                "legislator_slug": ["rep_a", "rep_b", "rep_c"],
                "vote": ["Yea", "Nay", "Yea"],
            }
        ).write_csv(parent_dir / f"{parent.output_name}_votes.csv")

        pl.DataFrame(
            {
                "session": ["88th (2019-2020)"] * 2,
                "vote_id": ["v1", "v2"],
                "chamber": ["House", "Senate"],
            }
        ).write_csv(parent_dir / f"{parent.output_name}_rollcalls.csv")

        pl.DataFrame(
            {
                "slug": ["rep_a", "rep_b", "rep_c"],
                "name": ["A", "B", "C"],
            }
        ).write_csv(parent_dir / f"{parent.output_name}_legislators.csv")

        # Write special CSVs
        pl.DataFrame(
            {
                "session": ["2020 Special"] * 2,
                "vote_id": ["v4", "v5"],
                "legislator_slug": ["rep_a", "rep_d"],
                "vote": ["Nay", "Yea"],
            }
        ).write_csv(special_dir / f"{special.output_name}_votes.csv")

        pl.DataFrame(
            {
                "session": ["2020 Special"],
                "vote_id": ["v4"],
                "chamber": ["House"],
            }
        ).write_csv(special_dir / f"{special.output_name}_rollcalls.csv")

        pl.DataFrame(
            {
                "slug": ["rep_a", "rep_d"],
                "name": ["A", "D"],
            }
        ).write_csv(special_dir / f"{special.output_name}_legislators.csv")

        stats = merge_special_into_parent(2020)

        assert stats["votes_added"] == 2
        assert stats["rollcalls_added"] == 1
        assert stats["legislators_added"] == 1  # rep_d is new; rep_a deduped

        # Verify merged votes
        votes = pl.read_csv(parent_dir / f"{parent.output_name}_votes.csv")
        assert votes.height == 5
        assert votes.filter(pl.col("session") == "2020 Special").height == 2

        # Verify merged legislators
        legs = pl.read_csv(parent_dir / f"{parent.output_name}_legislators.csv")
        assert legs.height == 4  # rep_a, rep_b, rep_c, rep_d


# ── CLI --merge-special ─────────────────────────────────────────────────────


class TestCliMergeSpecial:
    """CLI --merge-special flag dispatches to merge logic."""

    def test_merge_special_single(self, monkeypatch, capsys):
        """--merge-special 2020 calls merge_special_into_parent(2020)."""
        calls = []

        def fake_merge(year):
            calls.append(year)
            return {"votes_added": 10, "rollcalls_added": 2, "legislators_added": 0}

        monkeypatch.setattr(
            "tallgrass.merge_special.merge_special_into_parent",
            fake_merge,
        )

        from tallgrass.cli import main

        main(["--merge-special", "2020"])
        assert calls == [2020]
        output = capsys.readouterr().out
        assert "2020 Special" in output
        assert "10" in output

    def test_merge_special_all(self, monkeypatch, capsys):
        """--merge-special all calls merge_all_specials()."""
        called = []

        def fake_merge_all():
            called.append(True)
            return {2020: {"votes_added": 5, "rollcalls_added": 1, "legislators_added": 0}}

        monkeypatch.setattr(
            "tallgrass.merge_special.merge_all_specials",
            fake_merge_all,
        )

        from tallgrass.cli import main

        main(["--merge-special", "all"])
        assert called == [True]

    def test_merge_special_unknown_year(self, capsys):
        """--merge-special with unknown year prints error."""
        from tallgrass.cli import main

        main(["--merge-special", "1999"])
        output = capsys.readouterr().out
        assert "not a known special session year" in output

    def test_merge_special_skips_scraper(self, monkeypatch):
        """--merge-special does NOT instantiate a scraper."""
        instances = []

        class FakeScraper:
            def __init__(self, **kwargs):
                instances.append(self)

        monkeypatch.setattr("tallgrass.cli.KSVoteScraper", FakeScraper)

        def fake_merge(year):
            return {"votes_added": 0, "rollcalls_added": 0, "legislators_added": 0}

        monkeypatch.setattr(
            "tallgrass.merge_special.merge_special_into_parent",
            fake_merge,
        )

        from tallgrass.cli import main

        main(["--merge-special", "2020"])
        assert len(instances) == 0
