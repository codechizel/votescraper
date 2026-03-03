"""
Tests for KanFocus vote tally page parsing.

Uses inline text fixtures matching the real KanFocus page structure observed
during planning (session 112, vote #33 Senate and vote #1 House).

Run: uv run pytest tests/test_kanfocus_parser.py -v
"""

import pytest

from tallgrass.kanfocus.parser import (
    CATEGORY_MAP,
    _parse_counts,
    _parse_legislators_by_category,
    _parse_metadata,
    is_empty_page,
    parse_legislator_entry,
    parse_vote_page,
)

pytestmark = pytest.mark.scraper


# ── Fixtures ──────────────────────────────────────────────────────────────

_JS = "if (acell == 4) {document.write('</tr><tr>'); acell=1;} else {++acell;} ;"
_VAR = "var acell = 1, x; x=acell;"

_SENATE_META = (
    "Vote #: 33 Date: 02/03/2011 Bill Number: SB 13 Question: On final action Result: Passed"
)
SENATE_VOTE_TEXT = "\n".join(
    [
        _SENATE_META,
        "All Members Republicans Democrats",
        "For 38 100% 31 100% 7 100%",
        "Against 0 0% 0 0% 0 0%",
        "Present 0 0% 0 0% 0 0%",
        "Not Voting 1 N/A 1 N/A 0 N/A",
        _VAR,
        "Yea (38)",
        "Steve Abrams, R-32nd",
        _JS,
        "Pat Apple, R-12th",
        _JS,
        "Oletha Faust-Goudeau, D-29th",
        _JS,
        "Thomas C. (Tim) Owens, R-8th",
        _JS,
        "Mary Pilcher Cook, R-10th",
        _JS,
        _VAR,
        "Not Voting (1)",
        "Les Donovan, R-27th",
        _JS,
    ]
)

_HOUSE_META = (
    "Vote #: 1 Date: 02/04/2011 Bill Number: Sub for HR 6004"
    " Question: On agreeing to the amendment Result: Failed"
)
HOUSE_VOTE_TEXT = "\n".join(
    [
        _HOUSE_META,
        "All Members Republicans Democrats",
        "For 42 39% 12 16% 30 100%",
        "Against 65 61% 65 84% 0 0%",
        "Present 0 0% 0 0% 0 0%",
        "Not Voting 18 N/A 15 N/A 3 N/A",
        _VAR,
        "Yea (42)",
        "Barbara Ballard, D-44th",
        _JS,
        "Ramon Gonzalez Jr., R-47th",
        _JS,
        "Melody McCray Miller, D-89th",
        _JS,
        _VAR,
        "Nay (65)",
        "Tom Arpke, R-69th",
        _JS,
        "Robert (Bob) Montgomery, R-26th",
        _JS,
        "Virgil Peck Jr., R-11th",
        _JS,
        _VAR,
        "Not Voting (18)",
        "J. Stephen Alford, R-124th",
        _JS,
        "Gail Finney, D-84th",
        _JS,
    ]
)

EMPTY_PAGE_TEXT = "\n".join(
    [
        "Vote #: Date: Bill Number: Question: Result:",
        "All Members Republicans Democrats",
        "For 0 0% 0 0% 0 0%",
        "Against 0 0% 0 0% 0 0%",
        "Present 0 0% 0 0% 0 0%",
        "Not Voting 0 N/A 0 N/A 0 N/A",
    ]
)

_PRESENT_META = (
    "Vote #: 5 Date: 03/15/2011 Bill Number: HB 2000 Question: On final action Result: Passed"
)
PRESENT_VOTE_TEXT = "\n".join(
    [
        _PRESENT_META,
        "All Members Republicans Democrats",
        "For 100 80% 70 90% 30 100%",
        "Against 15 12% 5 6% 10 0%",
        "Present 2 2% 1 1% 1 3%",
        "Not Voting 8 N/A 1 N/A 0 N/A",
        _VAR,
        "Yea (100)",
        "John Smith, R-1st",
        _JS,
        _VAR,
        "Nay (15)",
        "Jane Doe, D-2nd",
        _JS,
        _VAR,
        "Present (2)",
        "Mike Johnson, R-3rd",
        _JS,
        "Alice Brown, D-4th",
        _JS,
        _VAR,
        "Not Voting (8)",
        "Bob Wilson, R-5th",
        _JS,
    ]
)


# ── parse_vote_page() ─────────────────────────────────────────────────────


class TestParseVotePage:
    """Full page parsing into KanFocusVoteRecord."""

    def test_senate_vote_parsed(self):
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://example.com")
        assert record is not None
        assert record.vote_num == 33
        assert record.year == 2011
        assert record.chamber == "S"

    def test_senate_metadata(self):
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://example.com")
        assert record.date == "02/03/2011"
        assert record.bill_number == "SB 13"
        assert record.question == "On final action"
        assert record.result == "Passed"

    def test_senate_counts(self):
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://example.com")
        assert record.yea_count == 38
        assert record.nay_count == 0
        assert record.present_count == 0
        assert record.not_voting_count == 1

    def test_senate_legislators(self):
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://example.com")
        names = [leg.name for leg in record.legislators]
        assert "Steve Abrams" in names
        assert "Thomas C. (Tim) Owens" in names
        assert "Les Donovan" in names

    def test_senate_categories(self):
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://example.com")
        yea_names = [leg.name for leg in record.legislators if leg.vote_category == "Yea"]
        nv_names = [leg.name for leg in record.legislators if leg.vote_category == "Not Voting"]
        assert "Steve Abrams" in yea_names
        assert "Les Donovan" in nv_names

    def test_house_vote_parsed(self):
        record = parse_vote_page(HOUSE_VOTE_TEXT, 1, 2011, "H", "https://example.com")
        assert record is not None
        assert record.bill_number == "Sub for HR 6004"

    def test_house_counts(self):
        record = parse_vote_page(HOUSE_VOTE_TEXT, 1, 2011, "H", "https://example.com")
        assert record.yea_count == 42
        assert record.nay_count == 65
        assert record.not_voting_count == 18

    def test_house_has_three_categories(self):
        record = parse_vote_page(HOUSE_VOTE_TEXT, 1, 2011, "H", "https://example.com")
        categories = {leg.vote_category for leg in record.legislators}
        assert "Yea" in categories
        assert "Nay" in categories
        assert "Not Voting" in categories

    def test_empty_page_returns_none(self):
        record = parse_vote_page(EMPTY_PAGE_TEXT, 999, 2011, "H", "https://example.com")
        assert record is None

    def test_blank_string_returns_none(self):
        record = parse_vote_page("", 1, 2011, "H", "https://example.com")
        assert record is None

    def test_source_url_preserved(self):
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://kanfocus.com/test")
        assert record.source_url == "https://kanfocus.com/test"

    def test_legislators_are_tuple(self):
        """Frozen dataclass requires tuple, not list."""
        record = parse_vote_page(SENATE_VOTE_TEXT, 33, 2011, "S", "https://example.com")
        assert isinstance(record.legislators, tuple)

    def test_present_category_parsed(self):
        """Present votes map to 'Present and Passing'."""
        record = parse_vote_page(PRESENT_VOTE_TEXT, 5, 2011, "H", "https://example.com")
        assert record is not None
        present = [leg for leg in record.legislators if leg.vote_category == "Present and Passing"]
        assert len(present) == 2
        assert present[0].name == "Mike Johnson"


# ── is_empty_page() ───────────────────────────────────────────────────────


class TestIsEmptyPage:
    """Detect empty/nonexistent vote pages."""

    def test_empty_string(self):
        assert is_empty_page("") is True

    def test_whitespace_only(self):
        assert is_empty_page("   \n\n   ") is True

    def test_empty_page_text(self):
        assert is_empty_page(EMPTY_PAGE_TEXT) is True

    def test_valid_page_not_empty(self):
        assert is_empty_page(SENATE_VOTE_TEXT) is False

    def test_house_page_not_empty(self):
        assert is_empty_page(HOUSE_VOTE_TEXT) is False


# ── _parse_metadata() ─────────────────────────────────────────────────────


class TestParseMetadata:
    """Extract structured metadata from page text."""

    def test_senate_metadata(self):
        meta = _parse_metadata(SENATE_VOTE_TEXT)
        assert meta is not None
        assert meta["vote_num_str"] == "33"
        assert meta["date"] == "02/03/2011"
        assert meta["bill_number"] == "SB 13"
        assert meta["question"] == "On final action"
        assert meta["result"] == "Passed"

    def test_house_metadata(self):
        meta = _parse_metadata(HOUSE_VOTE_TEXT)
        assert meta is not None
        assert meta["vote_num_str"] == "1"
        assert meta["bill_number"] == "Sub for HR 6004"
        assert meta["question"] == "On agreeing to the amendment"
        assert meta["result"] == "Failed"

    def test_empty_page_metadata(self):
        """Empty page has no vote number — returns None."""
        meta = _parse_metadata(EMPTY_PAGE_TEXT)
        assert meta is None


# ── _parse_counts() ───────────────────────────────────────────────────────


class TestParseCounts:
    """Extract vote count totals."""

    def test_senate_counts(self):
        counts = _parse_counts(SENATE_VOTE_TEXT)
        assert counts["for"] == 38
        assert counts["against"] == 0
        assert counts["present"] == 0
        assert counts["not_voting"] == 1

    def test_house_counts(self):
        counts = _parse_counts(HOUSE_VOTE_TEXT)
        assert counts["for"] == 42
        assert counts["against"] == 65
        assert counts["present"] == 0
        assert counts["not_voting"] == 18

    def test_present_counts(self):
        counts = _parse_counts(PRESENT_VOTE_TEXT)
        assert counts["for"] == 100
        assert counts["against"] == 15
        assert counts["present"] == 2
        assert counts["not_voting"] == 8


# ── _parse_legislators_by_category() ──────────────────────────────────────


class TestParseLegislatorsByCategory:
    """Parse legislators grouped by vote category."""

    def test_senate_legislator_count(self):
        legislators = _parse_legislators_by_category(SENATE_VOTE_TEXT)
        # 5 Yea + 1 Not Voting in fixture
        assert len(legislators) == 6

    def test_house_legislator_count(self):
        legislators = _parse_legislators_by_category(HOUSE_VOTE_TEXT)
        # 3 Yea + 3 Nay + 2 Not Voting in fixture
        assert len(legislators) == 8

    def test_category_assignment(self):
        legislators = _parse_legislators_by_category(SENATE_VOTE_TEXT)
        by_cat = {}
        for leg in legislators:
            by_cat.setdefault(leg.vote_category, []).append(leg)
        assert len(by_cat["Yea"]) == 5
        assert len(by_cat["Not Voting"]) == 1

    def test_party_extraction(self):
        legislators = _parse_legislators_by_category(SENATE_VOTE_TEXT)
        abrams = next(leg for leg in legislators if leg.name == "Steve Abrams")
        assert abrams.party == "R"
        assert abrams.district == "32nd"

    def test_democrat_party(self):
        legislators = _parse_legislators_by_category(SENATE_VOTE_TEXT)
        faust = next(leg for leg in legislators if "Faust-Goudeau" in leg.name)
        assert faust.party == "D"
        assert faust.district == "29th"

    def test_nickname_preserved(self):
        legislators = _parse_legislators_by_category(SENATE_VOTE_TEXT)
        owens = next(leg for leg in legislators if "Owens" in leg.name)
        assert owens.name == "Thomas C. (Tim) Owens"

    def test_multi_word_name(self):
        legislators = _parse_legislators_by_category(SENATE_VOTE_TEXT)
        pilcher = next(leg for leg in legislators if "Pilcher Cook" in leg.name)
        assert pilcher.name == "Mary Pilcher Cook"

    def test_suffix_preserved(self):
        legislators = _parse_legislators_by_category(HOUSE_VOTE_TEXT)
        gonzalez = next(leg for leg in legislators if "Gonzalez" in leg.name)
        assert gonzalez.name == "Ramon Gonzalez Jr."

    def test_parenthetical_name(self):
        legislators = _parse_legislators_by_category(HOUSE_VOTE_TEXT)
        montgomery = next(leg for leg in legislators if "Montgomery" in leg.name)
        assert montgomery.name == "Robert (Bob) Montgomery"


# ── parse_legislator_entry() ──────────────────────────────────────────────


class TestParseLegislatorEntry:
    """Parse single 'Name, Party-District' entries."""

    def test_standard(self):
        assert parse_legislator_entry("Steve Abrams, R-32nd") == (
            "Steve Abrams",
            "R",
            "32nd",
        )

    def test_democrat(self):
        assert parse_legislator_entry("David Haley, D-4th") == ("David Haley", "D", "4th")

    def test_suffix(self):
        result = parse_legislator_entry("Ramon Gonzalez Jr., R-47th")
        assert result == ("Ramon Gonzalez Jr.", "R", "47th")

    def test_nickname(self):
        result = parse_legislator_entry("Thomas C. (Tim) Owens, R-8th")
        assert result == ("Thomas C. (Tim) Owens", "R", "8th")

    def test_multi_word(self):
        result = parse_legislator_entry("Mary Pilcher Cook, R-10th")
        assert result == ("Mary Pilcher Cook", "R", "10th")

    def test_hyphenated_name(self):
        result = parse_legislator_entry("Oletha Faust-Goudeau, D-29th")
        assert result == ("Oletha Faust-Goudeau", "D", "29th")

    def test_three_digit_district(self):
        result = parse_legislator_entry("Bob Bethell, R-113th")
        assert result == ("Bob Bethell", "R", "113th")

    def test_invalid_returns_none(self):
        assert parse_legislator_entry("not a legislator") is None

    def test_empty_returns_none(self):
        assert parse_legislator_entry("") is None


# ── CATEGORY_MAP ──────────────────────────────────────────────────────────


class TestCategoryMap:
    """KanFocus → tallgrass vote category mapping."""

    def test_yea(self):
        assert CATEGORY_MAP["Yea"] == "Yea"

    def test_nay(self):
        assert CATEGORY_MAP["Nay"] == "Nay"

    def test_present(self):
        assert CATEGORY_MAP["Present"] == "Present and Passing"

    def test_not_voting(self):
        assert CATEGORY_MAP["Not Voting"] == "Not Voting"

    def test_exactly_four_categories(self):
        assert len(CATEGORY_MAP) == 4
