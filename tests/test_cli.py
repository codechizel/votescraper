"""
Tests for CLI argument parsing in cli.py.

Uses monkeypatch to intercept KSVoteScraper construction and method calls,
verifying that argument combinations produce the correct session, output dir,
delay, and flag settings without performing any actual HTTP requests.

Run: uv run pytest tests/test_cli.py -v
"""

from pathlib import Path

import pytest

from ks_vote_scraper.cli import main
from ks_vote_scraper.session import CURRENT_BIENNIUM_START

# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_scraper(monkeypatch):
    """Patch KSVoteScraper to capture construction args without running."""
    instances = []

    class FakeScraper:
        def __init__(self, session, output_dir=None, delay=0.15):
            self.session = session
            self.output_dir = output_dir
            self.delay = delay
            self.cache_cleared = False
            self.run_called = False
            self.run_enrich = None
            instances.append(self)

        def clear_cache(self):
            self.cache_cleared = True

        def run(self, enrich=True):
            self.run_called = True
            self.run_enrich = enrich

    monkeypatch.setattr("ks_vote_scraper.cli.KSVoteScraper", FakeScraper)
    return instances


# ── Default arguments ────────────────────────────────────────────────────────


class TestDefaultArgs:
    """CLI with no arguments uses current biennium defaults."""

    def test_default_year(self, mock_scraper):
        main([])
        assert mock_scraper[0].session.start_year == CURRENT_BIENNIUM_START

    def test_default_runs_scraper(self, mock_scraper):
        main([])
        assert mock_scraper[0].run_called is True

    def test_default_enriches(self, mock_scraper):
        main([])
        assert mock_scraper[0].run_enrich is True

    def test_default_output_dir_none(self, mock_scraper):
        """Output dir defaults to None (scraper picks data/{output_name}/)."""
        main([])
        assert mock_scraper[0].output_dir is None


# ── Explicit year ────────────────────────────────────────────────────────────


class TestExplicitYear:
    """CLI with an explicit year argument."""

    def test_historical_year(self, mock_scraper):
        main(["2023"])
        assert mock_scraper[0].session.start_year == 2023
        assert mock_scraper[0].session.special is False

    def test_even_year_normalized(self, mock_scraper):
        """2024 normalizes to start_year 2023."""
        main(["2024"])
        assert mock_scraper[0].session.start_year == 2023


# ── --special flag ───────────────────────────────────────────────────────────


class TestSpecialSession:
    """CLI --special flag creates a special session."""

    def test_special_flag(self, mock_scraper):
        main(["2024", "--special"])
        assert mock_scraper[0].session.special is True
        assert mock_scraper[0].session.start_year == 2024

    def test_special_output_name(self, mock_scraper):
        main(["2024", "--special"])
        assert mock_scraper[0].session.output_name == "2024s"


# ── --output flag ────────────────────────────────────────────────────────────


class TestOutputDir:
    """CLI --output / -o flag sets custom output directory."""

    def test_long_flag(self, mock_scraper):
        main(["2025", "--output", "/tmp/custom"])
        assert mock_scraper[0].output_dir == Path("/tmp/custom")

    def test_short_flag(self, mock_scraper):
        main(["2025", "-o", "/tmp/short"])
        assert mock_scraper[0].output_dir == Path("/tmp/short")


# ── --delay flag ─────────────────────────────────────────────────────────────


class TestDelay:
    """CLI --delay flag sets request delay."""

    def test_custom_delay(self, mock_scraper):
        main(["2025", "--delay", "0.5"])
        assert mock_scraper[0].delay == 0.5


# ── --no-enrich flag ─────────────────────────────────────────────────────────


class TestNoEnrich:
    """CLI --no-enrich skips legislator enrichment."""

    def test_no_enrich(self, mock_scraper):
        main(["2025", "--no-enrich"])
        assert mock_scraper[0].run_enrich is False


# ── --clear-cache flag ───────────────────────────────────────────────────────


class TestClearCache:
    """CLI --clear-cache clears cached pages before running."""

    def test_clear_cache_called(self, mock_scraper):
        main(["2025", "--clear-cache"])
        assert mock_scraper[0].cache_cleared is True

    def test_no_clear_cache_by_default(self, mock_scraper):
        main(["2025"])
        assert mock_scraper[0].cache_cleared is False


# ── --list-sessions flag ─────────────────────────────────────────────────────


class TestListSessions:
    """CLI --list-sessions prints sessions and exits without scraping."""

    def test_list_sessions_no_scraper(self, mock_scraper, capsys):
        main(["--list-sessions"])
        assert len(mock_scraper) == 0  # No scraper instantiated
        output = capsys.readouterr().out
        assert "Known Kansas Legislature sessions" in output

    def test_list_sessions_includes_current(self, capsys, mock_scraper):
        main(["--list-sessions"])
        output = capsys.readouterr().out
        assert str(CURRENT_BIENNIUM_START) in output

    def test_list_sessions_includes_special(self, capsys, mock_scraper):
        main(["--list-sessions"])
        output = capsys.readouterr().out
        assert "Special" in output
