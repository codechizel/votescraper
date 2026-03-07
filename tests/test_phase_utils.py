"""Tests for phase_utils sponsor matching utilities.

Run: uv run pytest tests/test_phase_utils.py -v
"""

import polars as pl
import pytest

from analysis.phase_utils import match_sponsor_to_party, match_sponsor_to_slug, parse_sponsor_name

# ── Tests: parse_sponsor_name ───────────────────────────────────────────────


class TestParseSponsorName:
    """Tests for parse_sponsor_name().

    Run: uv run pytest tests/test_phase_utils.py::TestParseSponsorName -v
    """

    def test_senator(self):
        name, chamber = parse_sponsor_name("Senator Tyson")
        assert name == "Tyson"
        assert chamber == "Senate"

    def test_representative(self):
        name, chamber = parse_sponsor_name("Representative Smith")
        assert name == "Smith"
        assert chamber == "House"

    def test_comma_initial(self):
        """Name with comma-separated initials."""
        name, chamber = parse_sponsor_name("Senator Claeys, J.R.")
        assert name == "Claeys, J.R."
        assert chamber == "Senate"

    def test_committee_returns_none(self):
        """Committee sponsors produce (None, None)."""
        name, chamber = parse_sponsor_name("Committee on Taxation")
        assert name is None
        assert chamber is None

    def test_empty_string(self):
        name, chamber = parse_sponsor_name("")
        assert name is None
        assert chamber is None

    def test_none_input(self):
        name, chamber = parse_sponsor_name(None)
        assert name is None
        assert chamber is None

    def test_no_title_prefix(self):
        """Raw name without Senator/Representative returns None."""
        name, chamber = parse_sponsor_name("John Smith")
        assert name is None
        assert chamber is None


# ── Tests: match_sponsor_to_party ───────────────────────────────────────────


class TestMatchSponsorToParty:
    """Tests for match_sponsor_to_party().

    Run: uv run pytest tests/test_phase_utils.py::TestMatchSponsorToParty -v
    """

    @pytest.fixture
    def legislators(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "legislator_slug": [
                    "sen_tyson_caryn_1",
                    "sen_schreiber_tom_1",
                    "rep_smith_john_1",
                ],
                "full_name": ["Tyson", "Schreiber", "Smith"],
                "chamber": ["Senate", "Senate", "House"],
                "party": ["Republican", "Democrat", "Republican"],
            }
        )

    def test_republican_match(self, legislators):
        result = match_sponsor_to_party("Senator Tyson", legislators)
        assert result == "Republican"

    def test_democrat_match(self, legislators):
        result = match_sponsor_to_party("Senator Schreiber", legislators)
        assert result == "Democrat"

    def test_house_match(self, legislators):
        result = match_sponsor_to_party("Representative Smith", legislators)
        assert result == "Republican"

    def test_committee_returns_none(self, legislators):
        result = match_sponsor_to_party("Committee on Taxation", legislators)
        assert result is None

    def test_empty_returns_none(self, legislators):
        result = match_sponsor_to_party("", legislators)
        assert result is None

    def test_multi_sponsor_uses_first(self, legislators):
        """Multi-sponsor string uses the first entry."""
        result = match_sponsor_to_party("Senator Tyson; Senator Schreiber", legislators)
        assert result == "Republican"


# ── Tests: match_sponsor_to_slug ──────────────────────────────────────────


class TestMatchSponsorToSlug:
    """Tests for match_sponsor_to_slug().

    Run: uv run pytest tests/test_phase_utils.py::TestMatchSponsorToSlug -v
    """

    @pytest.fixture
    def legislators(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "legislator_slug": [
                    "sen_tyson_caryn_1",
                    "sen_schreiber_tom_1",
                    "rep_smith_john_1",
                ],
                "full_name": ["Tyson", "Schreiber", "Smith"],
                "chamber": ["Senate", "Senate", "House"],
                "party": ["Republican", "Democrat", "Republican"],
            }
        )

    def test_returns_slug(self, legislators):
        result = match_sponsor_to_slug("Senator Tyson", legislators)
        assert result == "sen_tyson_caryn_1"

    def test_house_slug(self, legislators):
        result = match_sponsor_to_slug("Representative Smith", legislators)
        assert result == "rep_smith_john_1"

    def test_committee_returns_none(self, legislators):
        result = match_sponsor_to_slug("Committee on Taxation", legislators)
        assert result is None

    def test_no_match_returns_none(self, legislators):
        result = match_sponsor_to_slug("Senator Unknown", legislators)
        assert result is None

    def test_empty_returns_none(self, legislators):
        result = match_sponsor_to_slug("", legislators)
        assert result is None

    def test_legislator_slug_column(self):
        """Works with 'legislator_slug' column."""
        legs = pl.DataFrame(
            {
                "legislator_slug": ["sen_tyson_caryn_1"],
                "full_name": ["Tyson"],
                "chamber": ["Senate"],
                "party": ["Republican"],
            }
        )
        result = match_sponsor_to_slug("Senator Tyson", legs)
        assert result == "sen_tyson_caryn_1"
