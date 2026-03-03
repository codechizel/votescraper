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


def load_chrome_cookies(domain: str = "kanfocus.com") -> dict[str, str]:
    """Extract Chrome cookies for a domain on macOS.

    Reads Chrome's encrypted cookie database using the Keychain-stored
    encryption key. Returns ``{name: value}`` dict for ``requests.Session``.
    Requires ``cryptography`` package (already a transitive dependency).
    """
    import shutil
    import sqlite3
    import subprocess
    import tempfile

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    # Chrome Safe Storage key from macOS Keychain
    chrome_key = subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage", "-a", "Chrome"],
        stderr=subprocess.DEVNULL,
    ).strip()

    kdf = PBKDF2HMAC(algorithm=hashes.SHA1(), length=16, salt=b"saltysalt", iterations=1003)
    aes_key = kdf.derive(chrome_key)

    # Chrome locks the DB — copy to temp file
    cookie_db = Path.home() / "Library/Application Support/Google/Chrome/Default/Cookies"
    if not cookie_db.exists():
        for profile in ["Profile 1", "Profile 2", "Profile 3"]:
            alt = Path.home() / f"Library/Application Support/Google/Chrome/{profile}/Cookies"
            if alt.exists():
                cookie_db = alt
                break

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copy2(cookie_db, tmp_path)

    try:
        conn = sqlite3.connect(str(tmp_path))
        rows = conn.execute(
            "SELECT name, encrypted_value FROM cookies WHERE host_key LIKE ?",
            (f"%{domain}%",),
        ).fetchall()
        conn.close()
    finally:
        tmp_path.unlink(missing_ok=True)

    cookies: dict[str, str] = {}
    for name, enc in rows:
        if not enc or len(enc) < 3:
            continue
        if enc[:3] == b"v10":
            data = enc[3:]
            iv = b" " * 16
            cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            dec = cipher.decryptor().update(data) + cipher.decryptor().finalize()
            # Wait — need single decryptor instance
            cipher2 = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
            d = cipher2.decryptor()
            dec = d.update(data) + d.finalize()
            pad = dec[-1]
            if isinstance(pad, int) and 1 <= pad <= 16:
                dec = dec[:-pad]
            # Skip 32-byte app-bound encryption prefix (Chrome 127+)
            if len(dec) > 32:
                dec = dec[32:]
            value = dec.decode("utf-8", errors="replace")
            if value.isascii() and value:
                cookies[name] = value

    return cookies


class KanFocusFetcher:
    """HTTP client for fetching KanFocus vote tally pages.

    Strictly sequential: one page at a time with configurable delay.
    File-based cache in ``{data_dir}/.cache/kanfocus/`` keyed by URL hash.

    Uses Chrome cookies for authentication (KanFocus is a paid subscription
    service). Cookies are extracted from Chrome's encrypted cookie database
    on macOS using the Keychain-stored encryption key.
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

        # Load Chrome session cookies for authentication
        try:
            cookies = load_chrome_cookies()
            self.http.cookies.update(cookies)
            print(f"  Loaded {len(cookies)} Chrome cookies for kanfocus.com")
        except Exception as e:
            print(f"  Warning: could not load Chrome cookies ({e})")
            print("  Requests may fail — ensure you're logged into KanFocus in Chrome")

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

                # Detect redirect page (unauthenticated / expired session)
                if "MM_goToURL" in text and len(text) < 500:
                    if attempt == 0:
                        print("  Warning: got redirect page — session may have expired")
                    continue

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
