"""OpenStates multi-state bill text adapter.

Implements the StateAdapter Protocol for any US state via the
OpenStates API v3 (https://v3.openstates.org/docs).  Discovers
bills and their PDF version URLs; text download + extraction is
handled by ``BillTextFetcher`` (state-agnostic).

API rate limits: 10 requests/second (free tier, no key required).
With API key (OPENSTATES_API_KEY): higher limits.
"""

import os
import re
import threading
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

from tallgrass.config import USER_AGENT
from tallgrass.text.models import BillDocumentRef

# ── Constants ────────────────────────────────────────────────────────────────

OPENSTATES_API_URL = "https://v3.openstates.org"
OPENSTATES_DELAY = 0.12  # ~8 req/sec, under the 10/sec free tier limit
OPENSTATES_TIMEOUT = 30
OPENSTATES_PAGE_SIZE = 20  # API default
MAX_PAGES = 200  # safety cap

# State abbreviation to OpenStates jurisdiction mapping
STATE_JURISDICTIONS: dict[str, str] = {
    "mo": "ocd-jurisdiction/country:us/state:mo/government",
    "ok": "ocd-jurisdiction/country:us/state:ok/government",
    "ne": "ocd-jurisdiction/country:us/state:ne/government",
    "co": "ocd-jurisdiction/country:us/state:co/government",
    "ks": "ocd-jurisdiction/country:us/state:ks/government",
}

STATE_NAMES: dict[str, str] = {
    "mo": "missouri",
    "ok": "oklahoma",
    "ne": "nebraska",
    "co": "colorado",
    "ks": "kansas",
}


class OpenStatesAdapter:
    """StateAdapter implementation for any state via OpenStates API v3.

    Discovers bills and their document version URLs.  Prefers
    "Introduced" version (closest to model legislation source);
    falls back to the earliest available version.
    """

    def __init__(self, state: str, api_key: str | None = None):
        """Initialize adapter for a specific state.

        Args:
            state: Two-letter state abbreviation (e.g., "mo", "ok").
            api_key: OpenStates API key.  Falls back to
                     OPENSTATES_API_KEY env var, then unauthenticated.
        """
        self.state = state.lower()
        self.state_name = STATE_NAMES.get(self.state, self.state)
        self._jurisdiction = STATE_JURISDICTIONS.get(self.state)
        if not self._jurisdiction:
            msg = f"Unknown state: {state!r}. Known: {sorted(STATE_JURISDICTIONS)}"
            raise ValueError(msg)

        self._api_key = api_key or os.environ.get("OPENSTATES_API_KEY", "")

        self._http = requests.Session()
        self._http.headers.update({"User-Agent": USER_AGENT})
        if self._api_key:
            self._http.headers.update({"X-API-KEY": self._api_key})
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=4)
        self._http.mount("https://", adapter)

        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Thread-safe rate limiting for API calls."""
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < OPENSTATES_DELAY:
                time.sleep(OPENSTATES_DELAY - elapsed)
            self._last_request_time = time.monotonic()

    def _api_get(self, path: str, params: dict | None = None) -> dict | None:
        """Make a rate-limited GET request to the OpenStates API.

        Returns parsed JSON on success, None on failure.
        """
        url = f"{OPENSTATES_API_URL}{path}"
        self._rate_limit()

        try:
            resp = self._http.get(url, params=params, timeout=OPENSTATES_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  OpenStates API error: {e}")
            return None

    def _extract_pdf_url(self, versions: list[dict]) -> tuple[str, str]:
        """Find the best PDF URL from bill version data.

        Prefers "Introduced" version.  Falls back to earliest version
        with a PDF link.

        Returns (url, version_label) or ("", "") if no PDF found.
        """
        if not versions:
            return "", ""

        # Sort by date ascending (earliest first)
        sorted_versions = sorted(versions, key=lambda v: v.get("date", "") or "")

        # First pass: look for "Introduced" version
        for version in sorted_versions:
            note = (version.get("note") or "").lower()
            if "introduced" in note or "filed" in note or "prefiled" in note:
                for link in version.get("links", []):
                    url = link.get("url", "")
                    media = (link.get("media_type") or "").lower()
                    if url and ("pdf" in media or url.lower().endswith(".pdf")):
                        return url, version.get("note", "introduced")

        # Second pass: take earliest version with a PDF
        for version in sorted_versions:
            for link in version.get("links", []):
                url = link.get("url", "")
                media = (link.get("media_type") or "").lower()
                if url and ("pdf" in media or url.lower().endswith(".pdf")):
                    return url, version.get("note", "")

        # Third pass: take any URL from earliest version (may be HTML)
        for version in sorted_versions:
            for link in version.get("links", []):
                url = link.get("url", "")
                if url:
                    return url, version.get("note", "")

        return "", ""

    def _normalize_bill_number(self, identifier: str) -> str:
        """Normalize bill identifier to 'HB 1234' or 'SB 567' format.

        OpenStates uses formats like 'HB 1234', 'SB 567', 'LB 1' (Nebraska).
        Collapses multiple spaces and strips outer whitespace.
        """
        return re.sub(r"\s+", " ", identifier.strip())

    def discover_bills(self, session_id: str) -> list[BillDocumentRef]:
        """Discover all bills for a session via OpenStates API.

        Args:
            session_id: Session identifier (e.g., "2025", "2025-2026").
                       Format varies by state; OpenStates normalizes.

        Returns:
            List of BillDocumentRef with PDF URLs for text download.
        """
        refs: list[BillDocumentRef] = []
        seen_bills: set[str] = set()
        page = 1

        while page <= MAX_PAGES:
            params = {
                "jurisdiction": self._jurisdiction,
                "session": session_id,
                "include": "versions",
                "page": page,
                "per_page": OPENSTATES_PAGE_SIZE,
            }

            data = self._api_get("/bills", params)
            if not data:
                break

            results = data.get("results", [])
            if not results:
                break

            for bill in results:
                identifier = bill.get("identifier", "")
                bill_number = self._normalize_bill_number(identifier)

                if not bill_number or bill_number in seen_bills:
                    continue
                seen_bills.add(bill_number)

                versions = bill.get("versions", [])
                pdf_url, version_label = self._extract_pdf_url(versions)

                if pdf_url:
                    refs.append(
                        BillDocumentRef(
                            bill_number=bill_number,
                            document_type="introduced",
                            url=pdf_url,
                            version=version_label,
                            session=f"{self.state.upper()}-{session_id}",
                        )
                    )

            # Check for more pages
            pagination = data.get("pagination", {})
            total_pages = pagination.get("max_page", 1)
            if page >= total_pages:
                break
            page += 1

        return refs

    def data_dir(self, session_id: str) -> Path:
        """Return the data directory for this state + session."""
        return Path(f"data/{self.state_name}/{session_id}")

    def cache_dir(self, session_id: str) -> Path:
        """Return the cache directory for downloaded documents."""
        return self.data_dir(session_id) / ".cache" / "text"
