"""
Tests for KanFocus HTTP fetcher, caching, and vote enumeration.

Run: uv run pytest tests/test_kanfocus_fetcher.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tallgrass.kanfocus.fetcher import DEFAULT_DELAY, DEFAULT_MAX_EMPTY, KanFocusFetcher

pytestmark = pytest.mark.scraper


# ── Fixtures ──────────────────────────────────────────────────────────────

# Minimal valid page text for testing
_META = "Vote #: 1 Date: 02/03/2011 Bill Number: SB 1 Question: On final action Result: Passed"
_JS = "if (acell == 4) {document.write('</tr><tr>'); acell=1;} else {++acell;} ;"
VALID_PAGE = "\n".join(
    [
        _META,
        "All Members Republicans Democrats",
        "For 30 100% 20 100% 10 100%",
        "Against 0 0% 0 0% 0 0%",
        "Present 0 0% 0 0% 0 0%",
        "Not Voting 0 N/A 0 N/A 0 N/A",
        "var acell = 1, x; x=acell;",
        "Yea (30)",
        "John Smith, R-1st",
        _JS,
    ]
)

EMPTY_PAGE = "\n".join(
    [
        "Vote #: Date: Bill Number: Question: Result:",
        "All Members Republicans Democrats",
        "For 0 0% 0 0% 0 0%",
        "Against 0 0% 0 0% 0 0%",
        "Present 0 0% 0 0% 0 0%",
        "Not Voting 0 N/A 0 N/A 0 N/A",
    ]
)


# ── Caching ───────────────────────────────────────────────────────────────


class TestFetcherCaching:
    """File-based page cache."""

    def test_creates_cache_dir(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        KanFocusFetcher(cache_dir=cache_dir, delay=0)
        assert cache_dir.exists()

    @patch("tallgrass.kanfocus.fetcher.requests.Session")
    def test_caches_fetched_pages(self, mock_session_cls, tmp_path: Path):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = VALID_PAGE
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        fetcher = KanFocusFetcher(cache_dir=tmp_path, delay=0)
        fetcher.http = mock_session

        url = "https://kanfocus.com/test"
        result1 = fetcher.fetch_page(url)
        result2 = fetcher.fetch_page(url)

        assert result1 == VALID_PAGE
        assert result2 == VALID_PAGE
        # Second call should read from cache, not HTTP
        assert mock_session.get.call_count == 1

    def test_cache_files_exist(self, tmp_path: Path):
        """Manually write a cache file and verify it's read."""
        fetcher = KanFocusFetcher(cache_dir=tmp_path, delay=0)
        # Pre-populate cache
        import hashlib

        url = "https://kanfocus.com/cached"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_file = tmp_path / f"{url_hash}.html"
        cache_file.write_text("cached content", encoding="utf-8")

        result = fetcher.fetch_page(url)
        assert result == "cached content"

    def test_clear_cache(self, tmp_path: Path):
        # Create some cache files
        (tmp_path / "abc123.html").write_text("test")
        (tmp_path / "def456.html").write_text("test")

        fetcher = KanFocusFetcher(cache_dir=tmp_path, delay=0)
        fetcher.clear_cache()

        assert len(list(tmp_path.iterdir())) == 0


# ── Vote Enumeration ──────────────────────────────────────────────────────


class TestEnumerateVotes:
    """Enumerate votes by incrementing vote numbers."""

    def test_stops_after_consecutive_empty(self, tmp_path: Path):
        """Should stop after max_empty consecutive empty pages."""
        fetcher = KanFocusFetcher(cache_dir=tmp_path, delay=0, max_empty=3)

        # Mock fetch_page to return empty pages
        fetcher.fetch_page = MagicMock(return_value=EMPTY_PAGE)

        records = fetcher.enumerate_votes(112, 2011, "S")
        assert len(records) == 0
        # Should have tried 3 pages (max_empty=3) before stopping
        assert fetcher.fetch_page.call_count == 3

    def test_collects_valid_records(self, tmp_path: Path):
        """Should collect records from valid pages."""
        fetcher = KanFocusFetcher(cache_dir=tmp_path, delay=0, max_empty=2)

        call_count = 0

        def mock_fetch(url):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Return valid page for first 2 calls
                return VALID_PAGE.replace("Vote #: 1", f"Vote #: {call_count}")
            return EMPTY_PAGE

        fetcher.fetch_page = mock_fetch

        records = fetcher.enumerate_votes(112, 2011, "S")
        assert len(records) == 2

    def test_resets_empty_counter_on_valid(self, tmp_path: Path):
        """Consecutive empty counter resets when a valid page is found."""
        fetcher = KanFocusFetcher(cache_dir=tmp_path, delay=0, max_empty=3)

        responses = [EMPTY_PAGE, EMPTY_PAGE, VALID_PAGE, EMPTY_PAGE, EMPTY_PAGE, EMPTY_PAGE]
        idx = 0

        def mock_fetch(url):
            nonlocal idx
            if idx < len(responses):
                result = responses[idx]
                idx += 1
                return result
            return EMPTY_PAGE

        fetcher.fetch_page = mock_fetch

        records = fetcher.enumerate_votes(112, 2011, "S")
        # Should find 1 valid record (the VALID_PAGE in position 3)
        assert len(records) == 1


# ── Defaults ──────────────────────────────────────────────────────────────


class TestFetcherDefaults:
    """Conservative default settings."""

    def test_default_delay(self):
        assert DEFAULT_DELAY == 7.0

    def test_default_max_empty(self):
        assert DEFAULT_MAX_EMPTY == 20
