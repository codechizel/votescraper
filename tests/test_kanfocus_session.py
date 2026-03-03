"""
Tests for KanFocus session ID mapping and URL construction.

Run: uv run pytest tests/test_kanfocus_session.py -v
"""

import pytest

from tallgrass.kanfocus.session import (
    biennium_streams,
    generate_vote_id,
    session_id_for_biennium,
    vote_tally_url,
)

pytestmark = pytest.mark.scraper


# ── session_id_for_biennium() ─────────────────────────────────────────────


class TestSessionIdForBiennium:
    """Map biennium start years to KanFocus session IDs."""

    def test_78th_1999(self):
        assert session_id_for_biennium(1999) == 106

    def test_79th_2001(self):
        assert session_id_for_biennium(2001) == 107

    def test_80th_2003(self):
        assert session_id_for_biennium(2003) == 108

    def test_81st_2005(self):
        assert session_id_for_biennium(2005) == 109

    def test_82nd_2007(self):
        assert session_id_for_biennium(2007) == 110

    def test_83rd_2009(self):
        assert session_id_for_biennium(2009) == 111

    def test_84th_2011(self):
        assert session_id_for_biennium(2011) == 112

    def test_85th_2013(self):
        assert session_id_for_biennium(2013) == 113

    def test_86th_2015(self):
        assert session_id_for_biennium(2015) == 114

    def test_87th_2017(self):
        assert session_id_for_biennium(2017) == 115

    def test_88th_2019(self):
        assert session_id_for_biennium(2019) == 116

    def test_89th_2021(self):
        assert session_id_for_biennium(2021) == 117

    def test_90th_2023(self):
        assert session_id_for_biennium(2023) == 118

    def test_91st_2025(self):
        assert session_id_for_biennium(2025) == 119

    def test_even_year_raises(self):
        with pytest.raises(ValueError, match="odd year"):
            session_id_for_biennium(2000)

    def test_too_early_raises(self):
        with pytest.raises(ValueError, match="1999"):
            session_id_for_biennium(1997)


# ── vote_tally_url() ──────────────────────────────────────────────────────


class TestVoteTallyUrl:
    """Construct KanFocus tally page URLs."""

    def test_senate_vote(self):
        url = vote_tally_url(112, 33, 2011, "S")
        assert url == "https://kanfocus.com/Tally_House_Alpha_112.shtml?&Unique_VoteID=332011S"

    def test_house_vote(self):
        url = vote_tally_url(112, 1, 2011, "H")
        assert url == "https://kanfocus.com/Tally_House_Alpha_112.shtml?&Unique_VoteID=12011H"

    def test_pre_2011_senate(self):
        url = vote_tally_url(111, 1, 2009, "S")
        assert url == "https://kanfocus.com/Tally_House_Alpha_111.shtml?&Unique_VoteID=12009S"

    def test_high_vote_number(self):
        url = vote_tally_url(119, 500, 2025, "H")
        assert url == "https://kanfocus.com/Tally_House_Alpha_119.shtml?&Unique_VoteID=5002025H"


# ── generate_vote_id() ────────────────────────────────────────────────────


class TestGenerateVoteId:
    """Generate deterministic vote IDs with kf_ prefix."""

    def test_senate_id(self):
        assert generate_vote_id(33, 2011, "S") == "kf_33_2011_S"

    def test_house_id(self):
        assert generate_vote_id(1, 2011, "H") == "kf_1_2011_H"

    def test_high_number(self):
        assert generate_vote_id(500, 2025, "H") == "kf_500_2025_H"


# ── biennium_streams() ────────────────────────────────────────────────────


class TestBienniumStreams:
    """Return the 4 (year, chamber) streams for a biennium."""

    def test_2011_streams(self):
        streams = biennium_streams(2011)
        assert streams == [(2011, "H"), (2011, "S"), (2012, "H"), (2012, "S")]

    def test_1999_streams(self):
        streams = biennium_streams(1999)
        assert streams == [(1999, "H"), (1999, "S"), (2000, "H"), (2000, "S")]

    def test_2025_streams(self):
        streams = biennium_streams(2025)
        assert streams == [(2025, "H"), (2025, "S"), (2026, "H"), (2026, "S")]
