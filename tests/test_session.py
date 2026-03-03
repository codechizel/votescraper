"""
Tests for KSSession URL resolution and biennium logic.

CLAUDE.md calls session.py "the single trickiest part of the scraper." These
tests verify every property and factory method for the three session types:
current biennium, historical biennium, and special session.

Run: uv run pytest tests/test_session.py -v
"""

from pathlib import Path

import pytest

from tallgrass.session import KSSession, _ordinal

pytestmark = pytest.mark.scraper

# ── _ordinal() ───────────────────────────────────────────────────────────────


class TestOrdinal:
    """Ordinal suffix formatting (1st, 2nd, 3rd, 11th, etc.)."""

    def test_first(self):
        assert _ordinal(1) == "1st"

    def test_second(self):
        assert _ordinal(2) == "2nd"

    def test_third(self):
        assert _ordinal(3) == "3rd"

    def test_fourth(self):
        assert _ordinal(4) == "4th"

    def test_teens_use_th(self):
        """11th, 12th, 13th — NOT 11st, 12nd, 13rd."""
        assert _ordinal(11) == "11th"
        assert _ordinal(12) == "12th"
        assert _ordinal(13) == "13th"

    def test_twenty_first(self):
        assert _ordinal(21) == "21st"

    def test_ninety_first(self):
        """91st — the current Kansas Legislature."""
        assert _ordinal(91) == "91st"


# ── KSSession properties ────────────────────────────────────────────────────


class TestKSSessionProperties:
    """Properties computed from start_year and special flag."""

    # -- Current biennium (2025-2026) --

    def test_end_year_current(self, current_session: KSSession):
        assert current_session.end_year == 2026

    def test_biennium_code_current(self, current_session: KSSession):
        assert current_session.biennium_code == "b2025_26"

    def test_is_current_true(self, current_session: KSSession):
        assert current_session.is_current is True

    def test_legislature_number_91(self, current_session: KSSession):
        assert current_session.legislature_number == 91

    def test_legislature_name_91st(self, current_session: KSSession):
        assert current_session.legislature_name == "91st"

    def test_li_prefix_current(self, current_session: KSSession):
        assert current_session.li_prefix == "/li/b2025_26"

    def test_bills_path_current(self, current_session: KSSession):
        assert current_session.bills_path == "/li/b2025_26/measures/bills/"

    def test_label_current(self, current_session: KSSession):
        assert current_session.label == "91st (2025-2026)"

    def test_output_name_current(self, current_session: KSSession):
        assert current_session.output_name == "91st_2025-2026"

    def test_api_path_current(self, current_session: KSSession):
        assert current_session.api_path == "/li/api/v13/rev-1"

    # -- Historical biennium (2023-2024) --

    def test_end_year_historical(self, historical_session: KSSession):
        assert historical_session.end_year == 2024

    def test_biennium_code_historical(self, historical_session: KSSession):
        assert historical_session.biennium_code == "b2023_24"

    def test_is_current_false_historical(self, historical_session: KSSession):
        assert historical_session.is_current is False

    def test_legislature_number_90(self, historical_session: KSSession):
        assert historical_session.legislature_number == 90

    def test_legislature_name_90th(self, historical_session: KSSession):
        assert historical_session.legislature_name == "90th"

    def test_li_prefix_historical(self, historical_session: KSSession):
        assert historical_session.li_prefix == "/li_2024/b2023_24"

    def test_bills_path_historical(self, historical_session: KSSession):
        assert historical_session.bills_path == "/li_2024/b2023_24/measures/bills/"

    def test_label_historical(self, historical_session: KSSession):
        assert historical_session.label == "90th (2023-2024)"

    def test_output_name_historical(self, historical_session: KSSession):
        assert historical_session.output_name == "90th_2023-2024"

    def test_api_path_historical(self, historical_session: KSSession):
        assert historical_session.api_path == "/li_2024/api/v13/rev-1"

    # -- Special session (2024) --

    def test_is_current_false_special(self, special_session: KSSession):
        assert special_session.is_current is False

    def test_li_prefix_special(self, special_session: KSSession):
        assert special_session.li_prefix == "/li_2024s/b2023_24"

    def test_li_prefix_special_all(self):
        """Each special session uses its empirically verified biennium code."""
        assert KSSession(2024, special=True).li_prefix == "/li_2024s/b2023_24"
        assert KSSession(2021, special=True).li_prefix == "/li_2021s/b2021s"
        assert KSSession(2020, special=True).li_prefix == "/li_2020s/b2020s"
        assert KSSession(2016, special=True).li_prefix == "/li_2016s/b2015_16"
        assert KSSession(2013, special=True).li_prefix == "/li_2013s/b2013_14"

    def test_label_special(self, special_session: KSSession):
        assert special_session.label == "2024 Special"

    def test_output_name_special(self, special_session: KSSession):
        assert special_session.output_name == "2024s"

    def test_api_path_special(self, special_session: KSSession):
        assert special_session.api_path == "/li_2024s/api/v13/rev-1"

    # -- Frozen --

    def test_frozen(self, current_session: KSSession):
        import dataclasses

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            current_session.start_year = 2027  # type: ignore[misc]


# ── from_year() ──────────────────────────────────────────────────────────────


class TestFromYear:
    """Factory that normalizes even years to the biennium start."""

    def test_odd_year_passthrough(self):
        s = KSSession.from_year(2025)
        assert s.start_year == 2025

    def test_even_year_normalized(self):
        """2026 → start_year 2025."""
        s = KSSession.from_year(2026)
        assert s.start_year == 2025

    def test_historical_odd(self):
        s = KSSession.from_year(2023)
        assert s.start_year == 2023

    def test_historical_even(self):
        s = KSSession.from_year(2024)
        assert s.start_year == 2023

    def test_special_keeps_year(self):
        """Special sessions use the year as-is, no normalization."""
        s = KSSession.from_year(2024, special=True)
        assert s.start_year == 2024
        assert s.special is True


# ── from_session_string() ────────────────────────────────────────────────────


class TestFromSessionString:
    """Parse CLI-style session strings like '2025-26'."""

    def test_dash_short(self):
        s = KSSession.from_session_string("2025-26")
        assert s.start_year == 2025

    def test_underscore_short(self):
        s = KSSession.from_session_string("2025_26")
        assert s.start_year == 2025

    def test_dash_full(self):
        s = KSSession.from_session_string("2025-2026")
        assert s.start_year == 2025

    def test_historical(self):
        s = KSSession.from_session_string("2023-24")
        assert s.start_year == 2023

    def test_special_session(self):
        s = KSSession.from_session_string("2024s")
        assert s.start_year == 2024
        assert s.special is True

    def test_special_session_output_name(self):
        s = KSSession.from_session_string("2024s")
        assert s.output_name == "2024s"

    def test_special_session_label(self):
        s = KSSession.from_session_string("2024s")
        assert s.label == "2024 Special"


# ── data_dir_for_session() ───────────────────────────────────────────────────


class TestDataDirForSession:
    """Static method that converts CLI strings to data directory Paths."""

    def test_current(self):
        assert KSSession.data_dir_for_session("2025-26") == Path("data/kansas/91st_2025-2026")

    def test_historical(self):
        assert KSSession.data_dir_for_session("2023-24") == Path("data/kansas/90th_2023-2024")

    def test_special(self):
        assert KSSession.data_dir_for_session("2024", special=True) == Path("data/kansas/2024s")

    def test_special_string(self):
        assert KSSession.data_dir_for_session("2024s") == Path("data/kansas/2024s")


# ── uses_odt ────────────────────────────────────────────────────────────────


class TestUsesOdt:
    """ODT vote format detection for pre-2015 sessions."""

    def test_2013_uses_odt(self):
        s = KSSession(start_year=2013)
        assert s.uses_odt is True

    def test_2011_uses_odt(self):
        s = KSSession(start_year=2011)
        assert s.uses_odt is True

    def test_2015_does_not_use_odt(self):
        s = KSSession(start_year=2015)
        assert s.uses_odt is False

    def test_current_does_not_use_odt(self, current_session: KSSession):
        assert current_session.uses_odt is False

    def test_special_does_not_use_odt(self):
        """Special sessions (even 2013) don't use ODT."""
        s = KSSession(start_year=2013, special=True)
        assert s.uses_odt is False


# ── js_data_paths ───────────────────────────────────────────────────────────


class TestJsDataPaths:
    """JavaScript data file paths for bill discovery fallback."""

    def test_2019_has_paths(self):
        s = KSSession(start_year=2019)
        paths = s.js_data_paths
        assert len(paths) == 2
        assert "/li_2020/s/js/data/bills_li_2020.js" in paths
        assert "/li_2020/m/js/data/bills_li_2020.js" in paths

    def test_2013_has_paths(self):
        s = KSSession(start_year=2013)
        paths = s.js_data_paths
        assert len(paths) == 2
        assert "/li_2014/s/js/data/bills_li_2014.js" in paths

    def test_2025_empty(self, current_session: KSSession):
        """Current session uses HTML listing, no JS fallback needed."""
        assert current_session.js_data_paths == []

    def test_2023_empty(self, historical_session: KSSession):
        """2023 (start_year=2023) is >= 2021, so no JS fallback."""
        assert historical_session.js_data_paths == []

    def test_special_session_path(self):
        """Special sessions try both /s/ and /m/ JS data paths."""
        s = KSSession(start_year=2016, special=True)
        paths = s.js_data_paths
        assert len(paths) == 2
        assert "/li_2016s/s/js/data/bills_li_2016s.js" in paths
        assert "/li_2016s/m/js/data/bills_li_2016s.js" in paths

    def test_special_session_2024_no_js(self):
        """2024 special uses HTML bill listing, no JS fallback needed."""
        s = KSSession(start_year=2024, special=True)
        assert s.js_data_paths == []

    def test_special_session_2021_has_js(self):
        """2021 special needs JS fallback despite being >= 2021."""
        s = KSSession(start_year=2021, special=True)
        paths = s.js_data_paths
        assert len(paths) == 2
        assert "/li_2021s/s/js/data/bills_li_2021s.js" in paths
