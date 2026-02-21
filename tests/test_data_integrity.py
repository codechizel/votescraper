"""
Data integrity tests for scraped Kansas Legislature CSVs.

These tests run against the REAL data in data/91st_2025-2026/ and verify
structural correctness. They catch scraping bugs, not analysis logic.

Run: uv run pytest tests/test_data_integrity.py -v
"""

from pathlib import Path

import polars as pl
import pytest

from ks_vote_scraper.session import KSSession

# ── Constants ────────────────────────────────────────────────────────────────

_SESSION = KSSession.from_year(2025)
DATA_DIR = Path("data") / _SESSION.output_name
_PREFIX = _SESSION.output_name
HOUSE_SEATS = 125
SENATE_SEATS = 40
VALID_VOTE_CATEGORIES = {"Yea", "Nay", "Present and Passing", "Absent and Not Voting", "Not Voting"}
VALID_CHAMBERS = {"House", "Senate"}
VALID_PARTIES = {"Republican", "Democrat"}

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def votes() -> pl.DataFrame:
    """Load the individual votes CSV."""
    return pl.read_csv(DATA_DIR / f"{_PREFIX}_votes.csv")


@pytest.fixture(scope="module")
def rollcalls() -> pl.DataFrame:
    """Load the rollcall summaries CSV."""
    return pl.read_csv(DATA_DIR / f"{_PREFIX}_rollcalls.csv")


@pytest.fixture(scope="module")
def legislators() -> pl.DataFrame:
    """Load the legislators CSV."""
    return pl.read_csv(DATA_DIR / f"{_PREFIX}_legislators.csv")


# ── Structural Tests ─────────────────────────────────────────────────────────


class TestCSVStructure:
    """Verify that each CSV has the expected columns and basic shape."""

    def test_votes_columns(self, votes: pl.DataFrame) -> None:
        expected = {
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
        }
        assert set(votes.columns) == expected

    def test_rollcalls_columns(self, rollcalls: pl.DataFrame) -> None:
        expected = {
            "session",
            "bill_number",
            "bill_title",
            "vote_id",
            "vote_url",
            "vote_datetime",
            "vote_date",
            "chamber",
            "motion",
            "vote_type",
            "result",
            "short_title",
            "sponsor",
            "yea_count",
            "nay_count",
            "present_passing_count",
            "absent_not_voting_count",
            "not_voting_count",
            "total_votes",
            "passed",
        }
        assert set(rollcalls.columns) == expected

    def test_legislators_columns(self, legislators: pl.DataFrame) -> None:
        expected = {
            "name",
            "full_name",
            "slug",
            "chamber",
            "party",
            "district",
            "member_url",
        }
        assert set(legislators.columns) == expected

    def test_votes_not_empty(self, votes: pl.DataFrame) -> None:
        assert votes.height > 0, "Votes CSV is empty"

    def test_rollcalls_not_empty(self, rollcalls: pl.DataFrame) -> None:
        assert rollcalls.height > 0, "Rollcalls CSV is empty"

    def test_legislators_not_empty(self, legislators: pl.DataFrame) -> None:
        assert legislators.height > 0, "Legislators CSV is empty"


# ── Referential Integrity ────────────────────────────────────────────────────


class TestReferentialIntegrity:
    """Verify foreign-key relationships between the three CSVs."""

    def test_all_vote_slugs_in_legislators(
        self, votes: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Every legislator_slug in votes.csv must exist in legislators.csv."""
        vote_slugs = set(votes["legislator_slug"].unique().to_list())
        leg_slugs = set(legislators["slug"].to_list())
        unknown = vote_slugs - leg_slugs
        assert not unknown, f"Vote slugs not in legislators: {unknown}"

    def test_all_legislators_have_votes(
        self, votes: pl.DataFrame, legislators: pl.DataFrame
    ) -> None:
        """Every legislator in legislators.csv should have at least one vote."""
        leg_slugs = set(legislators["slug"].to_list())
        vote_slugs = set(votes["legislator_slug"].unique().to_list())
        no_votes = leg_slugs - vote_slugs
        assert not no_votes, f"Legislators with zero votes: {no_votes}"

    def test_all_vote_ids_in_rollcalls(self, votes: pl.DataFrame, rollcalls: pl.DataFrame) -> None:
        """Every vote_id in votes.csv must exist in rollcalls.csv."""
        vote_ids = set(votes["vote_id"].unique().to_list())
        rc_ids = set(rollcalls["vote_id"].to_list())
        orphans = vote_ids - rc_ids
        assert not orphans, f"Vote IDs not in rollcalls: {len(orphans)} orphans"

    def test_all_rollcall_ids_in_votes(self, votes: pl.DataFrame, rollcalls: pl.DataFrame) -> None:
        """Every vote_id in rollcalls.csv should have individual votes."""
        rc_ids = set(rollcalls["vote_id"].to_list())
        vote_ids = set(votes["vote_id"].unique().to_list())
        missing = rc_ids - vote_ids
        assert not missing, f"Rollcalls with no individual votes: {missing}"


# ── Value Validation ─────────────────────────────────────────────────────────


class TestValueValidation:
    """Verify that column values are within expected domains."""

    def test_vote_categories_valid(self, votes: pl.DataFrame) -> None:
        """Every vote value must be one of the 5 known categories."""
        actual = set(votes["vote"].unique().to_list())
        unexpected = actual - VALID_VOTE_CATEGORIES
        assert not unexpected, f"Unexpected vote categories: {unexpected}"

    def test_chambers_valid(self, rollcalls: pl.DataFrame) -> None:
        """Chamber must be either 'House' or 'Senate'."""
        actual = set(rollcalls["chamber"].unique().to_list())
        unexpected = actual - VALID_CHAMBERS
        assert not unexpected, f"Unexpected chambers: {unexpected}"

    def test_parties_valid(self, legislators: pl.DataFrame) -> None:
        """Party must be 'Republican' or 'Democrat'."""
        actual = set(legislators["party"].unique().to_list())
        unexpected = actual - VALID_PARTIES
        assert not unexpected, f"Unexpected parties: {unexpected}"

    def test_vote_ids_unique_in_rollcalls(self, rollcalls: pl.DataFrame) -> None:
        """Each vote_id should appear exactly once in rollcalls.csv."""
        n_unique = rollcalls["vote_id"].n_unique()
        assert n_unique == rollcalls.height, (
            f"{rollcalls.height - n_unique} duplicate vote_ids in rollcalls"
        )

    def test_no_null_vote_ids(self, votes: pl.DataFrame) -> None:
        assert votes["vote_id"].null_count() == 0, "Null vote_ids in votes"

    def test_no_null_slugs(self, votes: pl.DataFrame) -> None:
        assert votes["legislator_slug"].null_count() == 0, "Null slugs in votes"

    def test_no_null_vote_values(self, votes: pl.DataFrame) -> None:
        assert votes["vote"].null_count() == 0, "Null vote values"


# ── Tally Consistency ────────────────────────────────────────────────────────


class TestTallyConsistency:
    """Verify that rollcall summary counts match individual vote details."""

    def test_yea_counts_match(self, votes: pl.DataFrame, rollcalls: pl.DataFrame) -> None:
        """Yea count in rollcalls.csv must match count of 'Yea' votes."""
        detail = (
            votes.filter(pl.col("vote") == "Yea")
            .group_by("vote_id")
            .agg(pl.len().alias("detail_yea"))
        )
        merged = rollcalls.join(detail, on="vote_id", how="left").fill_null(0)
        mismatches = merged.filter(pl.col("yea_count") != pl.col("detail_yea"))
        assert mismatches.height == 0, f"{mismatches.height} rollcalls with yea count mismatch"

    def test_nay_counts_match(self, votes: pl.DataFrame, rollcalls: pl.DataFrame) -> None:
        """Nay count in rollcalls.csv must match count of 'Nay' votes."""
        detail = (
            votes.filter(pl.col("vote") == "Nay")
            .group_by("vote_id")
            .agg(pl.len().alias("detail_nay"))
        )
        merged = rollcalls.join(detail, on="vote_id", how="left").fill_null(0)
        mismatches = merged.filter(pl.col("nay_count") != pl.col("detail_nay"))
        assert mismatches.height == 0, f"{mismatches.height} rollcalls with nay count mismatch"

    def test_total_votes_match_sum(self, rollcalls: pl.DataFrame) -> None:
        """total_votes should equal yea + nay + present + absent + not_voting."""
        computed = rollcalls.with_columns(
            (
                pl.col("yea_count")
                + pl.col("nay_count")
                + pl.col("present_passing_count")
                + pl.col("absent_not_voting_count")
                + pl.col("not_voting_count")
            ).alias("computed_total")
        )
        mismatches = computed.filter(pl.col("total_votes") != pl.col("computed_total"))
        assert mismatches.height == 0, (
            f"{mismatches.height} rollcalls where total != sum of categories"
        )


# ── Chamber Bounds ───────────────────────────────────────────────────────────


class TestChamberBounds:
    """Verify that vote counts don't exceed chamber seat counts."""

    def test_no_rollcall_exceeds_chamber_size(self, rollcalls: pl.DataFrame) -> None:
        """No rollcall should have more total votes than seats in the chamber."""
        overcount = rollcalls.with_columns(
            pl.when(pl.col("chamber") == "Senate")
            .then(pl.lit(SENATE_SEATS))
            .otherwise(pl.lit(HOUSE_SEATS))
            .alias("max_seats")
        ).filter(pl.col("total_votes") > pl.col("max_seats"))

        assert overcount.height == 0, f"{overcount.height} rollcalls exceed chamber seat count"

    def test_chamber_slug_consistency(self, votes: pl.DataFrame) -> None:
        """sen_* slugs should only appear in Senate votes, rep_* in House."""
        cross_chamber = votes.filter(
            (pl.col("legislator_slug").str.starts_with("sen_") & (pl.col("chamber") == "House"))
            | (pl.col("legislator_slug").str.starts_with("rep_") & (pl.col("chamber") == "Senate"))
        )
        assert cross_chamber.height == 0, f"{cross_chamber.height} votes with chamber-slug mismatch"


# ── No Duplicate Votes ──────────────────────────────────────────────────────


class TestNoDuplicates:
    """Verify no legislator voted twice on the same rollcall."""

    def test_no_duplicate_individual_votes(self, votes: pl.DataFrame) -> None:
        """Each (vote_id, legislator_slug) pair must be unique."""
        dupes = votes.group_by("vote_id", "legislator_slug").agg(pl.len()).filter(pl.col("len") > 1)
        assert dupes.height == 0, f"{dupes.height} duplicate (vote_id, legislator_slug) pairs"

    def test_legislator_slugs_unique(self, legislators: pl.DataFrame) -> None:
        """Each slug in legislators.csv must be unique."""
        n_unique = legislators["slug"].n_unique()
        assert n_unique == legislators.height, (
            f"{legislators.height - n_unique} duplicate slugs in legislators"
        )
