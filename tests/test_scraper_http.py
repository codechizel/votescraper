"""
Tests for the HTTP fetch layer: _get() retries, error classification, caching,
_fetch_many() retry waves, and _rate_limit() thread safety.

Covers the previously untested HTTP and concurrency infrastructure that underpins
the entire scraping pipeline.  All network calls are monkeypatched.

Run: uv run pytest tests/test_scraper_http.py -v
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

import hashlib

from tallgrass.config import (
    BILL_TITLE_MAX_LENGTH,
    MAX_RETRIES,
)
from tallgrass.scraper import FetchResult, KSVoteScraper
from tallgrass.session import KSSession

pytestmark = pytest.mark.scraper

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def scraper(tmp_path: Path) -> KSVoteScraper:
    """Scraper instance with temp output dir and no real HTTP."""
    session = KSSession(start_year=2025)
    s = KSVoteScraper(session, output_dir=tmp_path, delay=0.0)
    return s


# HTML responses must be > 200 chars to avoid the error page heuristic in _get(),
# which flags short HTML pages starting with "<" as error pages (Bug #9 defense).
_VALID_HTML = "<html><body>" + ("x" * 200) + "</body></html>"


def _mock_response(
    status_code: int = 200,
    text: str = _VALID_HTML,
    content: bytes | None = None,
) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.content = content or text.encode()
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        http_error = requests.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_error
    return resp


# ── _get() success paths ────────────────────────────────────────────────────


class TestGetSuccess:
    """Successful fetches: HTML, binary, and cache hits."""

    def test_html_success(self, scraper: KSVoteScraper):
        html = _VALID_HTML
        scraper.http.get = MagicMock(return_value=_mock_response(text=html))

        result = scraper._get("https://example.com/vote_view/je_123/")

        assert result.ok
        assert result.html == html
        assert result.error_type is None

    def test_binary_success(self, scraper: KSVoteScraper):
        odt_bytes = b"PK\x03\x04" + b"\x00" * 100  # ZIP magic bytes
        scraper.http.get = MagicMock(return_value=_mock_response(content=odt_bytes, text=""))

        result = scraper._get("https://example.com/odt_view/je_123/", binary=True)

        assert result.ok
        assert result.content_bytes == odt_bytes
        assert result.html is None

    def test_cache_hit_html(self, scraper: KSVoteScraper):
        url = "https://example.com/page"
        cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_file = scraper.cache_dir / f"{cache_hash}.html"
        cache_file.write_text("cached content", encoding="utf-8")

        # No HTTP call should happen
        scraper.http.get = MagicMock(side_effect=AssertionError("should not be called"))

        result = scraper._get(url)

        assert result.ok
        assert result.html == "cached content"

    def test_cache_hit_binary(self, scraper: KSVoteScraper):
        url = "https://example.com/odt"
        cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_file = scraper.cache_dir / f"{cache_hash}.bin"
        cache_file.write_bytes(b"cached binary")

        scraper.http.get = MagicMock(side_effect=AssertionError("should not be called"))

        result = scraper._get(url, binary=True)

        assert result.ok
        assert result.content_bytes == b"cached binary"

    def test_successful_fetch_writes_cache(self, scraper: KSVoteScraper):
        html = _VALID_HTML
        scraper.http.get = MagicMock(return_value=_mock_response(text=html))

        scraper._get("https://example.com/vote")

        cache_files = list(scraper.cache_dir.glob("*.html"))
        assert len(cache_files) == 1
        assert cache_files[0].read_text(encoding="utf-8") == html

    def test_binary_fetch_writes_bin_cache(self, scraper: KSVoteScraper):
        content = b"PK\x03\x04binary"
        scraper.http.get = MagicMock(return_value=_mock_response(content=content, text=""))

        scraper._get("https://example.com/odt", binary=True)

        cache_files = list(scraper.cache_dir.glob("*.bin"))
        assert len(cache_files) == 1
        assert cache_files[0].read_bytes() == content


# ── _get() error classification ──────────────────────────────────────────────


@pytest.mark.slow
class TestGetErrorClassification:
    """Error types are correctly classified for retry logic."""

    def test_404_is_permanent(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=404))

        result = scraper._get("https://example.com/missing")

        assert not result.ok
        assert result.error_type == "permanent"

    def test_500_is_transient(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=500))

        result = scraper._get("https://example.com/error")

        assert not result.ok
        assert result.error_type == "transient"

    def test_502_is_transient(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=502))

        result = scraper._get("https://example.com/gateway")

        assert not result.ok
        assert result.error_type == "transient"

    def test_403_is_permanent(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=403))

        result = scraper._get("https://example.com/forbidden")

        assert not result.ok
        assert result.error_type == "permanent"

    def test_timeout_error(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(side_effect=requests.Timeout("timed out"))

        result = scraper._get("https://example.com/slow")

        assert not result.ok
        assert result.error_type == "timeout"

    def test_connection_error(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(side_effect=requests.ConnectionError("connection refused"))

        result = scraper._get("https://example.com/down")

        assert not result.ok
        assert result.error_type == "connection"

    def test_generic_request_exception(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(side_effect=requests.RequestException("unknown error"))

        result = scraper._get("https://example.com/unknown")

        assert not result.ok
        assert result.error_type == "connection"


# ── _get() error page detection ──────────────────────────────────────────────


class TestGetErrorPageDetection:
    """HTTP 200 responses that are actually error pages."""

    def test_html_error_page_not_found(self, scraper: KSVoteScraper):
        error_html = "<html><head><title>Page Not Found</title></head><body></body></html>"
        scraper.http.get = MagicMock(return_value=_mock_response(text=error_html))

        result = scraper._get("https://example.com/bad")

        assert not result.ok
        assert result.error_type == "permanent"
        assert result.status_code == 200

    def test_html_error_page_title_error(self, scraper: KSVoteScraper):
        error_html = "<html><head><title>Error</title></head><body></body></html>"
        scraper.http.get = MagicMock(return_value=_mock_response(text=error_html))

        result = scraper._get("https://example.com/error")

        assert not result.ok
        assert result.error_type == "permanent"

    def test_short_html_page_is_error(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(text="<html>"))

        result = scraper._get("https://example.com/short")

        assert not result.ok
        assert result.error_type == "permanent"

    def test_empty_response_is_error(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(text=""))

        result = scraper._get("https://example.com/empty")

        assert not result.ok
        assert result.error_type == "permanent"

    def test_binary_html_error_page(self, scraper: KSVoteScraper):
        """Bug #9: KS Legislature returns HTML error pages for binary ODT URLs."""
        error_html = b"<html><body>Page not available</body></html>"
        scraper.http.get = MagicMock(return_value=_mock_response(content=error_html, text=""))

        result = scraper._get("https://example.com/odt", binary=True)

        assert not result.ok
        assert result.error_type == "permanent"

    def test_valid_long_html_is_not_error(self, scraper: KSVoteScraper):
        """Long HTML pages with actual content should not be flagged as errors."""
        valid_html = "<html><body>" + "x" * 500 + "</body></html>"
        scraper.http.get = MagicMock(return_value=_mock_response(text=valid_html))

        result = scraper._get("https://example.com/good")

        assert result.ok

    def test_json_api_response_is_not_error(self, scraper: KSVoteScraper):
        """JSON API responses should not be flagged as error pages."""
        json_text = '{"content": [{"BILLNO": "SB 1"}]}'
        scraper.http.get = MagicMock(return_value=_mock_response(text=json_text))

        result = scraper._get("https://example.com/api")

        assert result.ok
        assert result.html == json_text


# ── _get() retry behavior ───────────────────────────────────────────────────


@pytest.mark.slow
class TestGetRetries:
    """Retry counts and backoff for different error types."""

    def test_404_retries_at_most_twice(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=404))

        scraper._get("https://example.com/missing")

        assert scraper.http.get.call_count == 2  # initial + 1 retry

    def test_500_retries_max_retries_times(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=500))

        scraper._get("https://example.com/error")

        assert scraper.http.get.call_count == MAX_RETRIES

    def test_timeout_retries_max_retries_times(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(side_effect=requests.Timeout("timeout"))

        scraper._get("https://example.com/slow")

        assert scraper.http.get.call_count == MAX_RETRIES

    def test_connection_error_retries(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(side_effect=requests.ConnectionError("refused"))

        scraper._get("https://example.com/down")

        assert scraper.http.get.call_count == MAX_RETRIES

    def test_success_after_transient_failure(self, scraper: KSVoteScraper):
        """Recovery: first request fails, second succeeds."""
        fail_resp = _mock_response(status_code=500)
        ok_resp = _mock_response(text=_VALID_HTML)
        scraper.http.get = MagicMock(side_effect=[fail_resp, ok_resp])

        result = scraper._get("https://example.com/flaky")

        assert result.ok
        assert scraper.http.get.call_count == 2

    def test_failed_fetch_not_cached(self, scraper: KSVoteScraper):
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=500))

        scraper._get("https://example.com/fail")

        cache_files = list(scraper.cache_dir.glob("*"))
        assert len(cache_files) == 0


# ── _fetch_many() ───────────────────────────────────────────────────────────


class TestFetchMany:
    """Concurrent fetch with retry waves."""

    def test_all_succeed(self, scraper: KSVoteScraper):
        urls = ["https://example.com/a", "https://example.com/b"]
        html = _VALID_HTML
        scraper.http.get = MagicMock(return_value=_mock_response(text=html))

        results = scraper._fetch_many(urls, max_waves=0)

        assert len(results) == 2
        assert all(r.ok for r in results.values())

    def test_returns_all_urls(self, scraper: KSVoteScraper):
        urls = ["https://example.com/a", "https://example.com/b"]
        html = _VALID_HTML
        scraper.http.get = MagicMock(return_value=_mock_response(text=html))

        results = scraper._fetch_many(urls, max_waves=0)

        assert set(results.keys()) == set(urls)

    @pytest.mark.slow
    def test_permanent_failures_not_retried_in_waves(self, scraper: KSVoteScraper):
        """404 failures should not be retried in waves."""
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=404))

        results = scraper._fetch_many(
            ["https://example.com/gone"],
            max_waves=2,
            wave_cooldown=0,
        )

        assert not results["https://example.com/gone"].ok
        # Only initial pass calls (404 retries at most 2x in _get), no wave calls
        assert scraper.http.get.call_count == 2

    @pytest.mark.slow
    def test_delay_restored_after_waves(self, scraper: KSVoteScraper):
        """self.delay must be restored to normal after retry waves."""
        original_delay = scraper.delay
        scraper.http.get = MagicMock(return_value=_mock_response(status_code=500))

        scraper._fetch_many(
            ["https://example.com/fail"],
            max_waves=1,
            wave_cooldown=0,
        )

        assert scraper.delay == original_delay

    def test_delay_restored_on_exception(self, scraper: KSVoteScraper):
        """self.delay must be restored even if a wave raises an exception."""
        original_delay = scraper.delay

        def fail_then_explode(url, binary=False):
            return FetchResult(url=url, html=None, error_type="transient", error_message="fail")

        scraper._get = fail_then_explode

        scraper._fetch_many(
            ["https://example.com/fail"],
            max_waves=1,
            wave_cooldown=0,
        )

        assert scraper.delay == original_delay


# ── _rate_limit() ───────────────────────────────────────────────────────────


class TestRateLimit:
    """Thread-safe rate limiting."""

    def test_enforces_minimum_delay(self, tmp_path: Path):
        session = KSSession(start_year=2025)
        scraper = KSVoteScraper(session, output_dir=tmp_path, delay=0.05)

        t0 = time.monotonic()
        scraper._rate_limit()
        scraper._rate_limit()
        elapsed = time.monotonic() - t0

        # Second call should wait at least the delay
        assert elapsed >= 0.04  # small margin for timing precision

    def test_concurrent_calls_are_serialized(self, tmp_path: Path):
        """Multiple threads calling _rate_limit() should not skip the delay."""
        session = KSSession(start_year=2025)
        scraper = KSVoteScraper(session, output_dir=tmp_path, delay=0.02)

        timestamps: list[float] = []
        lock = threading.Lock()

        def record_rate_limit():
            scraper._rate_limit()
            with lock:
                timestamps.append(time.monotonic())

        threads = [threading.Thread(target=record_rate_limit) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 4 timestamps should be recorded
        assert len(timestamps) == 4

        # Timestamps should span at least 3 delays (4 calls, 3 gaps)
        timestamps.sort()
        total_span = timestamps[-1] - timestamps[0]
        assert total_span >= 0.05  # 3 * 0.02 with margin


# ── _filter_bills_with_votes() ──────────────────────────────────────────────


class TestFilterBillsWithVotes:
    """KLISS API pre-filter: JSON parsing, both response formats, fallback."""

    def test_list_format(self, scraper: KSVoteScraper):
        """API returns a raw JSON list."""
        api_json = """[
            {"BILLNO": "SB 1", "HISTORY": [{"status": "Yea: 33 Nay: 7"}],
             "SHORTTITLE": "Daylight saving", "ORIGINAL_SPONSOR": ["Senator Titus"]},
            {"BILLNO": "SB 2", "HISTORY": [{"status": "Introduced"}]}
        ]"""
        scraper._get = MagicMock(return_value=FetchResult(url="api", html=api_json))

        bill_urls = ["/li/b2025_26/measures/sb1/", "/li/b2025_26/measures/sb2/"]
        filtered, metadata = scraper._filter_bills_with_votes(bill_urls)

        assert len(filtered) == 1
        assert "/sb1/" in filtered[0]
        assert metadata["sb1"]["short_title"] == "Daylight saving"

    def test_content_wrapper_format(self, scraper: KSVoteScraper):
        """API returns {"content": [...]} wrapper."""
        api_json = """{"content": [
            {"BILLNO": "HB 2001", "HISTORY": [{"status": "Yea: 100 Nay: 25"}],
             "SHORTTITLE": "Tax reform", "ORIGINAL_SPONSOR": ["Rep Smith"]}
        ]}"""
        scraper._get = MagicMock(return_value=FetchResult(url="api", html=api_json))

        bill_urls = ["/li/b2025_26/measures/hb2001/"]
        filtered, metadata = scraper._filter_bills_with_votes(bill_urls)

        assert len(filtered) == 1

    def test_api_failure_falls_back(self, scraper: KSVoteScraper):
        """When API call fails, return all bills unfiltered."""
        scraper._get = MagicMock(
            return_value=FetchResult(
                url="api", html=None, error_type="transient", error_message="500"
            )
        )

        bill_urls = ["/li/b2025_26/measures/sb1/", "/li/b2025_26/measures/sb2/"]
        filtered, metadata = scraper._filter_bills_with_votes(bill_urls)

        assert filtered == bill_urls
        assert metadata == {}

    def test_invalid_json_falls_back(self, scraper: KSVoteScraper):
        scraper._get = MagicMock(return_value=FetchResult(url="api", html="not json at all"))

        bill_urls = ["/li/b2025_26/measures/sb1/"]
        filtered, metadata = scraper._filter_bills_with_votes(bill_urls)

        assert filtered == bill_urls
        assert metadata == {}

    def test_empty_html_falls_back(self, scraper: KSVoteScraper):
        """Defensive check: result.html is None even though result.ok is True."""
        scraper._get = MagicMock(
            return_value=FetchResult(url="api", html=None, content_bytes=b"binary")
        )

        bill_urls = ["/li/b2025_26/measures/sb1/"]
        filtered, metadata = scraper._filter_bills_with_votes(bill_urls)

        assert filtered == bill_urls

    def test_no_bills_with_votes_falls_back(self, scraper: KSVoteScraper):
        api_json = '[{"BILLNO": "SB 1", "HISTORY": [{"status": "Introduced"}]}]'
        scraper._get = MagicMock(return_value=FetchResult(url="api", html=api_json))

        bill_urls = ["/li/b2025_26/measures/sb1/"]
        filtered, metadata = scraper._filter_bills_with_votes(bill_urls)

        assert filtered == bill_urls

    def test_sponsor_list_joined(self, scraper: KSVoteScraper):
        """Multiple sponsors are joined with '; '."""
        api_json = """[{"BILLNO": "SB 1", "HISTORY": [{"status": "Yea: 33"}],
            "SHORTTITLE": "Test", "ORIGINAL_SPONSOR": ["Senator A", "Senator B"]}]"""
        scraper._get = MagicMock(return_value=FetchResult(url="api", html=api_json))

        _, metadata = scraper._filter_bills_with_votes(["/li/b2025_26/measures/sb1/"])

        assert metadata["sb1"]["sponsor"] == "Senator A; Senator B"


# ── Cache behavior ──────────────────────────────────────────────────────────


class TestCacheBehavior:
    """Cache write, read, and clear_cache()."""

    def test_clear_cache(self, scraper: KSVoteScraper):
        # Create some cache files
        (scraper.cache_dir / "test.html").write_text("cached")
        (scraper.cache_dir / "test.bin").write_bytes(b"binary")
        assert len(list(scraper.cache_dir.iterdir())) == 2

        scraper.clear_cache()

        assert scraper.cache_dir.exists()
        assert len(list(scraper.cache_dir.iterdir())) == 0

    def test_cache_write_failure_is_nonfatal(self, scraper: KSVoteScraper):
        """If cache directory is unwritable, fetch still succeeds."""
        html = _VALID_HTML
        scraper.http.get = MagicMock(return_value=_mock_response(text=html))

        # Make cache dir read-only
        scraper.cache_dir.chmod(0o444)
        try:
            result = scraper._get("https://example.com/page")
            assert result.ok
            assert result.html == html
        finally:
            scraper.cache_dir.chmod(0o755)


# ── Config constants ────────────────────────────────────────────────────────


class TestConfigConstants:
    """Verify config constants are importable and have expected types/values."""

    def test_bill_title_max_length(self):
        assert isinstance(BILL_TITLE_MAX_LENGTH, int)
        assert BILL_TITLE_MAX_LENGTH > 0
