"""KanFocusFetcher — conservative HTTP client for KanFocus vote pages.

Very conservative defaults: 7-second delay between requests, single-threaded.
KanFocus is a shared paid service — we must not degrade performance for other
users. A biennium with ~2000 votes takes ~4 hours. Cache ensures re-runs
skip already-fetched pages.
"""

import hashlib
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from tallgrass.config import MAX_RETRIES, REQUEST_TIMEOUT, USER_AGENT
from tallgrass.kanfocus.models import KanFocusVoteRecord
from tallgrass.kanfocus.parser import parse_vote_page
from tallgrass.kanfocus.session import (
    biennium_streams,
    session_id_for_biennium,
    vote_tally_url,
)

# Conservative defaults for a paid subscription service
DEFAULT_DELAY = 7.0  # seconds between requests
DEFAULT_MAX_EMPTY = 20  # consecutive empty pages before stopping a stream


class KanFocusFetcher:
    """HTTP client for fetching KanFocus vote tally pages.

    Strictly sequential: one page at a time with configurable delay.
    File-based cache in ``{data_dir}/.cache/kanfocus/`` keyed by URL hash.
    """

    def __init__(
        self,
        cache_dir: Path,
        delay: float = DEFAULT_DELAY,
        max_empty: int = DEFAULT_MAX_EMPTY,
    ):
        self.cache_dir = cache_dir
        self.delay = delay
        self.max_empty = max_empty

        self.http = requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT})
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self) -> None:
        """Thread-safe rate limiting."""
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_request_time = time.monotonic()

    def _cache_path(self, url: str) -> Path:
        """SHA-256 hash-keyed cache file path."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{url_hash}.html"

    def fetch_page(self, url: str) -> str | None:
        """Fetch a single page, using cache when available.

        Returns page text on success, None on failure.
        """
        cache_file = self._cache_path(url)
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                resp = self.http.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()

                text = resp.text
                try:
                    cache_file.write_text(text, encoding="utf-8")
                except OSError:
                    pass
                return text

            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
                continue

        return None

    def enumerate_votes(
        self,
        session_id: int,
        year: int,
        chamber: str,
    ) -> list[KanFocusVoteRecord]:
        """Enumerate all votes for one stream (year + chamber).

        Iterates vote numbers from 1 upward. Stops after ``max_empty``
        consecutive empty/nonexistent pages.
        """
        records: list[KanFocusVoteRecord] = []
        consecutive_empty = 0
        vote_num = 1
        chamber_name = "Senate" if chamber == "S" else "House"

        while consecutive_empty < self.max_empty:
            url = vote_tally_url(session_id, vote_num, year, chamber)
            text = self.fetch_page(url)

            if text is None:
                consecutive_empty += 1
                vote_num += 1
                continue

            record = parse_vote_page(text, vote_num, year, chamber, url)

            if record is None:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
                records.append(record)

            vote_num += 1

        if records:
            print(f"    {year} {chamber_name}: {len(records)} votes (scanned {vote_num - 1} pages)")
        else:
            print(f"    {year} {chamber_name}: 0 votes (scanned {vote_num - 1} pages)")

        return records

    def fetch_biennium(self, start_year: int) -> list[KanFocusVoteRecord]:
        """Fetch all votes for a biennium across all 4 streams.

        Iterates House/Senate × odd_year/even_year sequentially.
        """
        session_id = session_id_for_biennium(start_year)
        all_records: list[KanFocusVoteRecord] = []

        print(f"\n  Fetching votes (session ID {session_id})...")
        for year, chamber in biennium_streams(start_year):
            records = self.enumerate_votes(session_id, year, chamber)
            all_records.extend(records)

        print(f"\n  Total: {len(all_records)} votes across all streams")
        return all_records

    def clear_cache(self) -> None:
        """Remove all cached pages."""
        if self.cache_dir.exists():
            count = 0
            for f in self.cache_dir.iterdir():
                if f.is_file():
                    f.unlink()
                    count += 1
            print(f"  Cleared {count} cached pages from {self.cache_dir}")
