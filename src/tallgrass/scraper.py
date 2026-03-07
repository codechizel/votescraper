"""Core scraper class for Kansas Legislature roll call votes."""

import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from tallgrass.bills import (
    BILL_URL_RE as _BILL_URL_RE,
)
from tallgrass.bills import (
    bill_sort_key,
    discover_bill_urls,
    parse_js_array,
    parse_js_bill_data,
)
from tallgrass.config import (
    BASE_URL,
    BILL_TITLE_MAX_LENGTH,
    MAX_RETRIES,
    MAX_WORKERS,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    RETRY_WAVES,
    USER_AGENT,
    WAVE_COOLDOWN,
    WAVE_DELAY,
    WAVE_WORKERS,
)
from tallgrass.models import BillAction, IndividualVote, RollCall
from tallgrass.output import save_csvs
from tallgrass.session import KSSession

# The 5 vote categories used by the KS Legislature (in display order)
VOTE_CATEGORIES = ("Yea", "Nay", "Present and Passing", "Absent and Not Voting", "Not Voting")


def _normalize_bill_code(text: str) -> str:
    """Normalize a bill identifier to a compact lowercase code.

    "SB 1" -> "sb1", "HB 2124" -> "hb2124"
    """
    return re.sub(r"\s+", "", text).lower()


def _clean_text(element: BeautifulSoup) -> str:
    """Extract text from a BeautifulSoup element, preserving spaces around inline tags.

    get_text(strip=True) strips each text node then concatenates WITHOUT separators,
    which drops spaces around inline elements like <a>.  For example, an <h3> containing
    ``Amendment by <a>Senator Francisco</a> was rejected`` becomes the mangled
    ``Amendment bySenator Franciscowas rejected``.

    Using separator=" " inserts a space between text nodes, then we collapse any
    resulting multiple spaces into one.
    """
    return " ".join(element.get_text(separator=" ", strip=True).split())


@dataclass(frozen=True)
class FetchResult:
    """Result of an HTTP fetch attempt."""

    url: str
    html: str | None
    status_code: int | None = None
    error_type: str | None = None  # permanent, transient, timeout, connection
    error_message: str | None = None
    content_bytes: bytes | None = None

    @property
    def ok(self) -> bool:
        return self.html is not None or self.content_bytes is not None


@dataclass(frozen=True)
class FetchFailure:
    """Record of a failed vote page fetch with bill context."""

    bill_number: str
    vote_text: str
    vote_url: str
    bill_path: str
    status_code: int | None
    error_type: str
    error_message: str
    timestamp: str


@dataclass(frozen=True)
class VoteLink:
    """A link to a roll call vote page, discovered on a bill page."""

    bill_number: str
    bill_path: str
    vote_url: str
    vote_text: str
    is_odt: bool = False


class KSVoteScraper:
    """Scrapes Kansas Legislature roll call votes from kslegislature.gov."""

    def __init__(
        self,
        session: KSSession,
        output_dir: Path | None = None,
        delay: float = REQUEST_DELAY,
    ):
        self.session = session
        self.output_dir = output_dir or session.data_dir
        self.cache_dir = self.output_dir / ".cache"
        self.delay = delay
        self._normal_delay = delay
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": USER_AGENT})

        # Connection pool sized to match worker count
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=MAX_WORKERS)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

        # Thread-safe rate limiting
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

        self.individual_votes: list[IndividualVote] = []
        self.rollcalls: list[RollCall] = []
        self.legislators: dict[str, dict] = {}  # slug -> info
        self.bill_metadata: dict[str, dict] = {}  # normalized code -> API data
        self.bill_actions: list[BillAction] = []
        self.failures: list[FetchFailure] = []
        self._member_directory: dict[tuple[str, str], dict] | None = None

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- HTTP helpers ----------------------------------------------------------

    def _rate_limit(self) -> None:
        """Apply thread-safe rate limiting before an HTTP request."""
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_request_time = time.monotonic()

    def _get(self, url: str, binary: bool = False) -> FetchResult:
        """Fetch a URL with retries, caching, and rate limiting.

        Retry strategy varies by error type:
        - 404: max 2 attempts (one retry), no backoff
        - 5xx: exponential backoff (5s, 10s, 20s)
        - Timeout: exponential backoff
        - Connection error: fixed 5s delay

        When ``binary=True``, stores ``response.content`` (bytes) in
        ``content_bytes`` instead of decoding to text.  Skips HTML
        error-page detection.  Caches with a ``.bin`` extension.
        """
        # Check cache first (no rate limiting needed)
        cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_ext = ".bin" if binary else ".html"
        cache_file = self.cache_dir / f"{cache_hash}{cache_ext}"
        if cache_file.exists():
            if binary:
                return FetchResult(url=url, html=None, content_bytes=cache_file.read_bytes())
            return FetchResult(url=url, html=cache_file.read_text(encoding="utf-8"))

        last_error = ""
        last_status: int | None = None
        last_error_type: str | None = None
        max_attempts = MAX_RETRIES
        retry_delay = RETRY_DELAY
        attempt = 0

        while attempt < max_attempts:
            try:
                # Rate limit only actual network requests
                self._rate_limit()

                resp = self.http.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()

                if binary:
                    content = resp.content
                    # Detect HTML error pages served as binary (e.g., 404 page
                    # returned with HTTP 200 for .odt URLs)
                    if content[:5].startswith(b"<html") or content[:5].startswith(b"<HTML"):
                        return FetchResult(
                            url=url,
                            html=None,
                            status_code=200,
                            error_type="permanent",
                            error_message="HTML error page served for binary request",
                        )
                    try:
                        cache_file.write_bytes(content)
                    except OSError:
                        pass
                    return FetchResult(url=url, html=None, content_bytes=content)

                html = resp.text

                # Guard against HTML error pages served with HTTP 200
                # (only applies to HTML responses, not JSON API responses)
                if not html or (
                    "<" in html[:10]
                    and (
                        len(html) < 200
                        or "<title>Page Not Found</title>" in html
                        or "<title>Error</title>" in html
                    )
                ):
                    return FetchResult(
                        url=url,
                        html=None,
                        status_code=200,
                        error_type="permanent",
                        error_message="Error page served with HTTP 200",
                    )

                try:
                    cache_file.write_text(html, encoding="utf-8")
                except OSError:
                    pass  # cache write failure is non-fatal
                return FetchResult(url=url, html=html)

            except requests.HTTPError as e:
                last_status = e.response.status_code if e.response is not None else None
                last_error = str(e)
                if last_status == 404:
                    last_error_type = "permanent"
                    max_attempts = min(max_attempts, 2)
                    retry_delay = RETRY_DELAY
                elif last_status is not None and last_status >= 500:
                    last_error_type = "transient"
                    retry_delay = RETRY_DELAY * (2**attempt) * (1 + random.uniform(0, 0.5))
                else:
                    # Other 4xx — don't retry
                    last_error_type = "permanent"
                    print(f"  Failed: {url}: {e}")
                    break

            except requests.Timeout as e:
                last_error = str(e)
                last_error_type = "timeout"
                last_status = None
                retry_delay = RETRY_DELAY * (2**attempt) * (1 + random.uniform(0, 0.5))

            except requests.ConnectionError as e:
                last_error = str(e)
                last_error_type = "connection"
                last_status = None
                retry_delay = RETRY_DELAY

            except requests.RequestException as e:
                last_error = str(e)
                last_error_type = "connection"
                last_status = None
                retry_delay = RETRY_DELAY

            attempt += 1
            if attempt < max_attempts:
                print(f"  Retry {attempt}/{max_attempts} for {url}: {last_error}")
                time.sleep(retry_delay)
            else:
                print(f"  Failed after {max_attempts} attempts: {url}: {last_error}")

        return FetchResult(
            url=url,
            html=None,
            status_code=last_status,
            error_type=last_error_type,
            error_message=last_error,
        )

    def _fetch_many(
        self,
        urls: list[str],
        desc: str = "Fetching",
        max_waves: int = RETRY_WAVES,
        wave_cooldown: float = WAVE_COOLDOWN,
        binary: bool = False,
    ) -> dict[str, FetchResult]:
        """Fetch multiple URLs concurrently using a thread pool.

        After the initial pass, transient failures (5xx, timeout, connection) are
        retried in up to ``max_waves`` additional passes with reduced concurrency
        and a cooldown between waves.  This lets the server recover from sustained
        load without the thundering-herd effect of all workers retrying at once.

        When ``binary=True``, all fetches use binary mode (for ODT downloads).

        Returns a dict mapping each URL to its FetchResult.
        """
        # --- initial pass (full concurrency) ---
        results: dict[str, FetchResult] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(self._get, url, binary): url for url in urls}
            for future in tqdm(
                as_completed(future_to_url),
                total=len(future_to_url),
                desc=desc,
                unit="page",
            ):
                url = future_to_url[future]
                results[url] = future.result()

        # --- retry waves for transient failures ---
        _RETRIABLE = {"transient", "timeout", "connection"}

        for wave in range(1, max_waves + 1):
            failed_urls = [
                url for url, r in results.items() if not r.ok and r.error_type in _RETRIABLE
            ]
            if not failed_urls:
                break

            print(
                f"\n  {len(failed_urls)} transient failure(s)"
                f" — waiting {int(wave_cooldown)}s before retry wave {wave}/{max_waves}..."
            )
            time.sleep(wave_cooldown)

            # Reduce concurrency and slow rate limit for the retry wave.
            # Safe to mutate self.delay here: the initial ThreadPoolExecutor has
            # fully completed (all futures resolved), so no concurrent threads are
            # reading self.delay via _rate_limit().  The wave executor below starts
            # fresh threads that will see the updated value.
            self.delay = WAVE_DELAY
            try:
                with ThreadPoolExecutor(max_workers=WAVE_WORKERS) as executor:
                    future_to_url = {
                        executor.submit(self._get, url, binary): url for url in failed_urls
                    }
                    for future in tqdm(
                        as_completed(future_to_url),
                        total=len(future_to_url),
                        desc=f"Wave {wave}",
                        unit="page",
                    ):
                        url = future_to_url[future]
                        results[url] = future.result()
            finally:
                self.delay = self._normal_delay

        # Log final state
        final_transient = sum(
            1 for r in results.values() if not r.ok and r.error_type in _RETRIABLE
        )
        if final_transient:
            print(f"  {final_transient} transient failure(s) remain after {max_waves} wave(s)")

        return results

    # -- Step 1: Get all bill URLs ---------------------------------------------

    def get_bill_urls(self) -> list[str]:
        """Get all bill URLs from the listing pages.

        Delegates to ``tallgrass.bills.discover_bill_urls()`` which handles
        both HTML listing and JS data fallback for pre-2021 sessions.
        """
        print("=" * 60)
        print("Step 1: Fetching bill URLs from listing pages...")
        print("=" * 60)

        urls = discover_bill_urls(self.session, self._get)
        print(f"  Found {len(urls)} unique bill/resolution URLs")
        return urls

    @staticmethod
    def _bill_sort_key(url: str) -> tuple[int, int]:
        """Sort bills: SB before HB, then numerically."""
        return bill_sort_key(url)

    @staticmethod
    def _parse_js_array(js_content: str) -> list[dict]:
        """Extract the first JSON array from JS source, quoting bare keys."""
        return parse_js_array(js_content)

    @staticmethod
    def _parse_js_bill_data(js_content: str) -> list[str]:
        """Extract bill URLs from a ``measures_data = [...]`` JS assignment."""
        return parse_js_bill_data(js_content)

    # -- Step 1b: KLISS API pre-filter -----------------------------------------

    def _filter_bills_with_votes(self, bill_urls: list[str]) -> tuple[list[str], dict[str, dict]]:
        """Use the KLISS API to identify bills that have roll call votes.

        Fetches the bill_status API endpoint and checks each bill's history
        for "Yea:" in the status field, which indicates a recorded vote.
        Returns (filtered_urls, bill_metadata) where bill_metadata maps
        normalized codes like "sb1" to {"short_title": ..., "sponsor": ...}.
        Also stores the metadata on self.bill_metadata for downstream use
        by get_vote_links() (sponsor backfill).

        Falls back to (full list, {}) if the API call fails.
        """
        api_url = f"{BASE_URL}{self.session.api_path}/bill_status/"
        print(f"\n  Pre-filtering via KLISS API: {api_url}")

        result = self._get(api_url)
        if not result.ok:
            print(
                "  API pre-filter failed"
                f" ({result.error_type}: {result.error_message}),"
                " falling back to full scan"
            )
            return bill_urls, {}

        try:
            if result.html is None:
                print("  API pre-filter returned empty response, falling back to full scan")
                return bill_urls, {}
            data = json.loads(result.html)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  API pre-filter returned invalid JSON ({e}), falling back to full scan")
            return bill_urls, {}

        # Build a set of bill codes that have votes (e.g., "sb1", "hb2124")
        # and capture metadata (short_title, sponsor) for each
        bills_with_votes = set()
        bill_metadata: dict[str, dict] = {}
        if isinstance(data, list):
            content = data
        elif isinstance(data, dict):
            content = data.get("content", [])
        else:
            content = []
        for bill in content:
            bill_no = bill.get("BILLNO", "")
            history = bill.get("HISTORY", [])
            code = _normalize_bill_code(bill_no)

            # Capture lifecycle actions for ALL bills (Sankey funnel)
            for entry in history:
                self.bill_actions.append(
                    BillAction(
                        session=self.session.output_name,
                        bill_number=code,
                        action_code=entry.get("action_code", ""),
                        chamber=entry.get("chamber", ""),
                        committee_names=tuple(entry.get("committee_names", [])),
                        occurred_datetime=entry.get("occurred_datetime", ""),
                        session_date=entry.get("session_date", ""),
                        status=entry.get("status", ""),
                        journal_page_number=str(entry.get("journal_page_number", "")),
                    )
                )

            has_vote = any("Yea:" in entry.get("status", "") for entry in history)
            if has_vote:
                bills_with_votes.add(code)
                sponsors = bill.get("ORIGINAL_SPONSOR", [])
                bill_metadata[code] = {
                    "short_title": bill.get("SHORTTITLE", ""),
                    "sponsor": "; ".join(sponsors) if isinstance(sponsors, list) else str(sponsors),
                }

        if not bills_with_votes:
            print("  API returned no bills with votes, falling back to full scan")
            return bill_urls, {}

        # Filter bill_urls to only those matching a bill code with votes
        filtered = []
        for url in bill_urls:
            match = _BILL_URL_RE.search(url)
            if match:
                code = f"{match.group(1).lower()}{match.group(2)}"
                if code in bills_with_votes:
                    filtered.append(url)

        print(f"  API filter: {len(filtered)} of {len(bill_urls)} bills have votes")
        return filtered, bill_metadata

    # -- Step 2: Find vote links on each bill page ----------------------------

    def get_vote_links(self, bill_urls: list[str]) -> list[VoteLink]:
        """Visit each bill page and extract vote_view links."""
        print("\n" + "=" * 60)
        print("Step 2: Scanning bill pages for roll call vote links...")
        print("=" * 60)

        # Fetch phase (concurrent)
        full_urls = [BASE_URL + path for path in bill_urls]
        fetched = self._fetch_many(full_urls, desc="Scanning bills")

        # Parse phase (sequential)
        vote_links: list[VoteLink] = []
        bills_with_votes = 0
        sponsors_backfilled = 0

        for bill_path in bill_urls:
            url = BASE_URL + bill_path
            result = fetched.get(url)
            if not result or not result.ok:
                continue

            soup = BeautifulSoup(result.html, "lxml")
            bill_number = self._extract_bill_number(soup, bill_path)

            found_votes = False
            for link in soup.find_all("a", href=re.compile(r"(?:vote_view|odt_view)")):
                href = link["href"]
                vote_links.append(
                    VoteLink(
                        bill_number=bill_number,
                        bill_path=bill_path,
                        vote_url=href if href.startswith("http") else BASE_URL + href,
                        vote_text=link.get_text(strip=True),
                        is_odt="odt_view" in href,
                    )
                )
                found_votes = True

            if found_votes:
                bills_with_votes += 1

            # Extract sponsor text and slugs from bill page HTML
            bill_code = _normalize_bill_code(bill_number)
            meta = self.bill_metadata.get(bill_code, {})
            sponsor_text, sponsor_slugs = self._extract_sponsor(soup)
            if meta:
                if not meta.get("sponsor") and sponsor_text:
                    meta["sponsor"] = sponsor_text
                    sponsors_backfilled += 1
                if sponsor_slugs:
                    meta["sponsor_slugs"] = sponsor_slugs

        if sponsors_backfilled:
            print(f"  Backfilled {sponsors_backfilled} sponsors from bill page HTML")
        print(f"  Found {len(vote_links)} roll call votes across {bills_with_votes} bills")
        return vote_links

    @staticmethod
    def _extract_bill_number(soup: BeautifulSoup, bill_path: str) -> str:
        """Extract bill number like 'SB 1' or 'HB 2124' from the page."""
        h2 = soup.find("h2")
        if h2:
            text = h2.get_text(strip=True)
            match = re.match(r"((?:SB|HB|SCR|HCR|SR|HR)\s*\d+)", text, re.I)
            if match:
                return match.group(1).upper()

        match = _BILL_URL_RE.search(bill_path)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"
        return bill_path

    @staticmethod
    def _extract_sponsor(soup: BeautifulSoup) -> tuple[str, str]:
        """Extract original sponsor text and slugs from the bill page HTML.

        The sponsor is in a portlet structure:
          <div class="portlet-header">Original Sponsor</div>
          <div class="portlet-content">
            <ul><li><a href="/li/.../members/sen_steffen_joe_1/">Senator Steffen</a></li></ul>
          </div>

        Returns (sponsor_text, sponsor_slugs) where:
        - sponsor_text: "Senator Steffen; Senator Bowers" (semicolon-joined display names)
        - sponsor_slugs: "sen_steffen_joe_1; sen_bowers_larry_1" (semicolon-joined slugs)

        Committee links (/committees/) produce no slug. Falls back to ("", "") if not found.
        """
        header = soup.find(
            lambda tag: (
                tag.name == "div"
                and "portlet-header" in (tag.get("class") or [])
                and "Original Sponsor" in tag.get_text()
            )
        )
        if not header:
            return "", ""

        content = header.find_next_sibling("div", class_="portlet-content")
        if not content:
            return "", ""

        sponsors = []
        slugs = []
        for li in content.find_all("li"):
            text = li.get_text(strip=True)
            if text:
                sponsors.append(text)
            # Extract slug from <a href="/li/.../members/{slug}/">
            a_tag = li.find("a", href=True)
            if a_tag:
                href = a_tag["href"]
                if "/members/" in href and "/committees/" not in href:
                    # Extract slug: last non-empty path segment
                    parts = [p for p in href.split("/") if p]
                    if parts:
                        slugs.append(parts[-1])

        return "; ".join(sponsors), "; ".join(slugs)

    @staticmethod
    def _extract_bill_title(soup: BeautifulSoup) -> str:
        """Extract bill title from vote page HTML using 3-tier h4 fallback.

        Pitfall #1: <h4> = bill title (not <h2> which is bill number).
        """
        # Tier 1: regex match on h4 for standard bill title prefixes
        title_heading = soup.find(
            "h4", string=re.compile(r"AN ACT|A CONCURRENT|A RESOLUTION|A JOINT", re.I)
        )
        if title_heading:
            return _clean_text(title_heading)

        # Tier 2: scan h4 for text starting with "AN ACT" or length > 50
        for h4 in soup.find_all("h4"):
            text = _clean_text(h4)
            if text.startswith("AN ACT") or len(text) > 50:
                return text

        # Tier 3: scan h4 for text > 30 chars that isn't a known non-title heading
        for h4 in soup.find_all("h4"):
            text = _clean_text(h4)
            if len(text) > 30 and not text.startswith(
                ("SB", "HB", "On roll", "Yea", "Nay", "Senate", "House")
            ):
                return text

        return ""

    @staticmethod
    def _extract_chamber_motion_date(soup: BeautifulSoup) -> tuple[str, str, str]:
        """Extract chamber, motion text, and vote date from h3 headers.

        Pitfall #1: <h3> contains chamber/date/motion AND vote category headings.
        Uses _clean_text() to preserve spaces around inline <a> tags (Pitfall #5).

        Returns: (chamber, motion, vote_date) — date in MM/DD/YYYY format.
        """
        # Tier 1: strict regex with known delimiter pattern
        for h3 in soup.find_all("h3"):
            text = _clean_text(h3)
            match = re.match(
                r"(Senate|House)\s*-\s*(.+?)\s*-\s*(\d{2}/\d{2}/\d{4})$",
                text,
            )
            if match:
                motion = match.group(2).strip().rstrip(" -;")
                return match.group(1), motion, match.group(3)

        # Tier 2: looser parse for non-standard formatting
        for h3 in soup.find_all("h3"):
            text = _clean_text(h3)
            if text.startswith("Senate") or text.startswith("House"):
                chamber = "Senate" if text.startswith("Senate") else "House"
                date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
                vote_date = date_match.group(1) if date_match else ""
                motion = text.replace(chamber, "", 1).strip(" -")
                if vote_date:
                    motion = motion.replace(vote_date, "").strip(" -")
                return chamber, motion, vote_date

        return "", "", ""

    @staticmethod
    def _parse_vote_categories(
        soup: BeautifulSoup,
    ) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
        """Parse vote categories and member lists from vote page HTML.

        Pitfall #3: Must scan BOTH <h2> and <h3> for category headings.

        Returns:
            (vote_categories, new_legislators) where:
            - vote_categories maps category name to list of {"name": str, "slug": str}
            - new_legislators maps slug to {"name", "slug", "chamber", "member_url"}
        """
        vote_categories: dict[str, list[dict[str, str]]] = {cat: [] for cat in VOTE_CATEGORIES}
        new_legislators: dict[str, dict[str, str]] = {}

        current_category = None
        for element in soup.find_all(["h2", "h3", "a"]):
            if element.name in ("h2", "h3"):
                text = element.get_text(strip=True)
                for cat in vote_categories:
                    if text.lower().startswith(cat.lower()):
                        current_category = cat
                        break
            elif element.name == "a" and current_category is not None:
                href = element.get("href", "")
                if "/members/" in href:
                    name = element.get_text(strip=True).rstrip(",").strip()
                    slug_match = re.search(r"/members/([^/]+)/", href)
                    slug = slug_match.group(1) if slug_match else ""

                    if name:
                        vote_categories[current_category].append({"name": name, "slug": slug})

                        if slug:
                            leg_chamber = ""
                            if slug.startswith("sen_"):
                                leg_chamber = "Senate"
                            elif slug.startswith("rep_"):
                                leg_chamber = "House"
                            new_legislators[slug] = {
                                "name": name,
                                "legislator_slug": slug,
                                "chamber": leg_chamber,
                                "member_url": (
                                    f"{BASE_URL}{href}" if not href.startswith("http") else href
                                ),
                            }

        return vote_categories, new_legislators

    @staticmethod
    def _extract_party_and_district(soup: BeautifulSoup) -> dict[str, str]:
        """Extract full name, party, and district from a legislator page.

        Two fallback patterns:
        - Post-2015: <h2>District N - Republican</h2>
        - Pre-2015: <h3>Party: Republican</h3>

        Pitfall #2: Must parse the specific <h2> containing "District \\d+",
        NOT full page text (which always matches "Republican" via dropdown).
        Pitfall #2b: First <h1> is nav heading, not member name.

        Returns: {"full_name": str, "party": str, "district": str}
        """
        result: dict[str, str] = {"full_name": "", "party": "", "district": ""}

        # Full name from <h1> containing "Senator" or "Representative"
        name_h1 = soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))
        if name_h1:
            full_name = _clean_text(name_h1)
            full_name = re.sub(r"^(Senator|Representative)\s+", "", full_name)
            full_name = re.sub(r"\s+-\s+.*$", "", full_name)
            result["full_name"] = full_name

        # Post-2015: party and district from <h2> containing "District \d+"
        dist_h2 = soup.find("h2", string=re.compile(r"District\s+\d+"))
        if dist_h2:
            h2_text = dist_h2.get_text(strip=True)
            dist_match = re.search(r"District\s+(\d+)", h2_text)
            if dist_match:
                result["district"] = dist_match.group(1)
            if "Republican" in h2_text:
                result["party"] = "Republican"
            elif "Democrat" in h2_text:
                result["party"] = "Democrat"

        # Pre-2015 fallback: party from <h3> "Party: ..."
        if not result["party"]:
            for h3 in soup.find_all("h3"):
                h3_text = h3.get_text(strip=True)
                if "Party:" in h3_text:
                    if "Republican" in h3_text:
                        result["party"] = "Republican"
                    elif "Democrat" in h3_text:
                        result["party"] = "Democrat"
                    break

        return result

    # -- Member directory (for ODT sessions) ------------------------------------

    def _load_member_directory(self) -> None:
        """Build a member directory from the session's /members/ listing page.

        Creates a mapping of ``(chamber_lower, last_name_lower)`` to member info.
        Same-chamber last-name collisions are marked as ambiguous. When an initial
        is available (e.g., "C. Holmes"), the initial-qualified key
        ``(chamber, "c. holmes")`` is also added for disambiguation.

        Pre-2021 sessions render member lists via JavaScript, so we fall back to
        parsing the JS data files (``senate_members_list_li_{end_year}.js`` and
        ``house_members_list_li_{end_year}.js``).
        """
        members_path = f"{self.session.li_prefix}/members/"
        print(f"  Loading member directory from {members_path}...")
        result = self._get(BASE_URL + members_path)
        if not result.ok or not result.html:
            print("  WARNING: Could not load member directory — ODT name resolution disabled")
            self._member_directory = {}
            return

        soup = BeautifulSoup(result.html, "lxml")
        members: list[tuple[str, str]] = []  # (slug, last_name)

        for link in soup.find_all("a", href=re.compile(r"/members/(sen_|rep_)")):
            href = link["href"]
            slug_match = re.search(r"/members/([^/]+)/", href)
            if not slug_match:
                continue
            slug = slug_match.group(1)
            name = link.get_text(strip=True).rstrip(",").strip()
            if not name:
                continue
            # Extract last name
            if "," in name:
                last_name = name.split(",")[0].strip()
            else:
                last_name = name.split()[-1] if name.split() else name
            members.append((slug, last_name))

        # JS fallback: pre-2021 sessions render member lists via JavaScript
        if not members:
            members = self._load_members_from_js(soup)

        directory: dict[tuple[str, str], dict] = {}
        for slug, last_name in members:
            if slug.startswith("sen_"):
                chamber_lower = "senate"
            elif slug.startswith("rep_"):
                chamber_lower = "house"
            else:
                continue

            key = (chamber_lower, last_name.lower())
            if key in directory:
                directory[key]["ambiguous"] = True
            else:
                directory[key] = {
                    "slug": slug,
                    "name": last_name,
                    "chamber": chamber_lower.capitalize(),
                }

        self._member_directory = directory
        print(f"  Member directory: {len(directory)} entries")

    def _load_members_from_js(self, members_soup: BeautifulSoup) -> list[tuple[str, str]]:
        """Load member data from JavaScript files (pre-2021 sessions).

        Parses ``<script src="...members_list...">`` tags from the members page,
        fetches each JS file, and extracts ``members_url`` and ``last_name`` fields.

        Returns:
            List of (slug, last_name) tuples.
        """
        members: list[tuple[str, str]] = []

        # Find JS data files referenced in script tags
        js_urls: list[str] = []
        for script in members_soup.find_all("script", src=True):
            src = script["src"]
            if "member" in src.lower() and src.endswith(".js"):
                js_urls.append(src)

        if not js_urls:
            print("  WARNING: No member data JS files found — ODT name resolution disabled")
            return members

        for js_url in js_urls:
            full_url = BASE_URL + js_url if js_url.startswith("/") else js_url
            print(f"  Loading member data from {js_url}...")
            result = self._get(full_url)
            if not result.ok or not result.html:
                continue

            parsed = self._parse_js_member_data(result.html)
            members.extend(parsed)

        if members:
            print(f"  JS fallback found {len(members)} members")

        return members

    @staticmethod
    def _parse_js_member_data(js_content: str) -> list[tuple[str, str]]:
        """Extract (slug, last_name) pairs from a JS member data file."""
        data = KSVoteScraper._parse_js_array(js_content)
        members: list[tuple[str, str]] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            url = entry.get("members_url", "")
            last_name = entry.get("last_name", "")
            slug_match = re.search(r"/members/([^/]+)/", url)
            if slug_match and last_name:
                members.append((slug_match.group(1), last_name))
        return members

    # -- Step 3: Parse each vote page -----------------------------------------

    def parse_vote_pages(self, vote_links: list[VoteLink]) -> None:
        """Visit each vote page and extract individual legislator votes.

        Splits links into HTML (vote_view) and ODT (odt_view) groups and
        processes each with the appropriate parser.
        """
        print("\n" + "=" * 60)
        print("Step 3: Parsing individual roll call vote pages...")
        print("=" * 60)

        html_links = [vl for vl in vote_links if not vl.is_odt]
        odt_links = [vl for vl in vote_links if vl.is_odt]

        if html_links:
            self._parse_html_vote_pages(html_links)

        if odt_links:
            self._parse_odt_vote_pages(odt_links)

        print(f"  Parsed {len(self.rollcalls)} roll calls")
        print(f"  Collected {len(self.individual_votes)} individual votes")
        print(f"  Found {len(self.legislators)} unique legislators")

    def _parse_html_vote_pages(self, vote_links: list[VoteLink]) -> None:
        """Parse HTML vote_view pages (2015+)."""
        # Fetch phase (concurrent)
        vote_urls = [vl.vote_url for vl in vote_links]
        fetched = self._fetch_many(vote_urls, desc="Fetching votes")

        # Parse phase (sequential — mutates self.rollcalls, self.individual_votes, etc.)
        for vl in tqdm(vote_links, desc="Parsing votes", unit="vote"):
            result = fetched.get(vl.vote_url)
            if not result or not result.ok:
                self._record_fetch_failure(vl, result)
                continue
            soup = BeautifulSoup(result.html, "lxml")
            self._parse_vote_page(soup, vl)

    def _parse_odt_vote_pages(self, vote_links: list[VoteLink]) -> None:
        """Parse ODT vote files (2011-2014)."""
        from tallgrass.odt_parser import parse_odt_votes

        # Fetch phase (concurrent, binary mode)
        vote_urls = [vl.vote_url for vl in vote_links]
        fetched = self._fetch_many(vote_urls, desc="Fetching ODT votes", binary=True)

        # Parse phase (sequential)
        for vl in tqdm(vote_links, desc="Parsing ODT votes", unit="vote"):
            result = fetched.get(vl.vote_url)
            if not result or not result.ok:
                self._record_fetch_failure(vl, result)
                continue

            if not result.content_bytes:
                self._record_fetch_failure(vl, result)
                continue

            try:
                rollcalls, votes, new_legs = parse_odt_votes(
                    odt_bytes=result.content_bytes,
                    bill_number=vl.bill_number,
                    bill_path=vl.bill_path,
                    vote_url=vl.vote_url,
                    session_label=self.session.label,
                    member_directory=self._member_directory,
                    bill_metadata=self.bill_metadata,
                )
            except Exception as e:
                print(f"  WARNING: ODT parse error for {vl.bill_number}: {e}")
                self.failures.append(
                    FetchFailure(
                        bill_number=vl.bill_number,
                        vote_text=vl.vote_text,
                        vote_url=vl.vote_url,
                        bill_path=vl.bill_path,
                        status_code=200,
                        error_type="parsing",
                        error_message=f"ODT parse error: {e}",
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                    )
                )
                continue

            self.rollcalls.extend(rollcalls)
            self.individual_votes.extend(votes)

            for leg in new_legs:
                slug = leg["legislator_slug"]
                if slug and slug not in self.legislators:
                    self.legislators[slug] = {
                        "name": leg["name"],
                        "legislator_slug": slug,
                        "chamber": leg["chamber"],
                        "member_url": (f"{BASE_URL}{self.session.li_prefix}/members/{slug}/"),
                    }

            if not rollcalls:
                self.failures.append(
                    FetchFailure(
                        bill_number=vl.bill_number,
                        vote_text=vl.vote_text,
                        vote_url=vl.vote_url,
                        bill_path=vl.bill_path,
                        status_code=200,
                        error_type="parsing",
                        error_message="0 votes parsed from ODT",
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                    )
                )

    def _record_fetch_failure(self, vl: VoteLink, result: FetchResult | None) -> None:
        """Record a fetch failure for a vote link."""
        if result and result.error_type:
            self.failures.append(
                FetchFailure(
                    bill_number=vl.bill_number,
                    vote_text=vl.vote_text,
                    vote_url=vl.vote_url,
                    bill_path=vl.bill_path,
                    status_code=result.status_code,
                    error_type=result.error_type,
                    error_message=result.error_message or "",
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                )
            )
            print(
                f"  FAILED: {vl.bill_number} - {vl.vote_text}"
                f" ({result.error_type}: {result.error_message})"
            )

    def _parse_vote_page(self, soup: BeautifulSoup, vote_link: VoteLink) -> None:
        """Parse a single vote_view page."""
        bill_number = vote_link.bill_number
        vote_url = vote_link.vote_url

        # Extract vote ID from URL
        vote_id_match = re.search(r"vote_view/([^/]+)/", vote_url)
        vote_id = vote_id_match.group(1) if vote_id_match else vote_url

        # Parse precise datetime from vote_id
        vote_datetime = self._parse_vote_datetime(vote_id)

        # Extract structured fields via static helpers
        bill_title = self._extract_bill_title(soup)
        chamber, motion, vote_date = self._extract_chamber_motion_date(soup)

        # Parse vote_type and result from motion text
        vote_type, result = self._parse_vote_type_and_result(motion)

        # Derive passed from result
        passed = self._derive_passed(result)

        # Look up short_title and sponsor from KLISS API metadata
        bill_code = _normalize_bill_code(bill_number)
        meta = self.bill_metadata.get(bill_code, {})
        short_title = meta.get("short_title", "")
        sponsor = meta.get("sponsor", "")
        sponsor_slugs = meta.get("sponsor_slugs", "")

        # Parse vote categories and discover new legislators
        vote_categories, new_legislators = self._parse_vote_categories(soup)

        # Merge new legislators into registry
        for slug, info in new_legislators.items():
            if slug not in self.legislators:
                self.legislators[slug] = info

        # Compute total votes (all categories)
        total_votes = sum(len(members) for members in vote_categories.values())

        if total_votes == 0:
            print(f"  WARNING: 0 votes parsed for {bill_number} — skipping")
            self.failures.append(
                FetchFailure(
                    bill_number=bill_number,
                    vote_text=vote_link.vote_text,
                    vote_url=vote_link.vote_url,
                    bill_path=vote_link.bill_path,
                    status_code=200,
                    error_type="parsing",
                    error_message="0 votes parsed from page",
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                )
            )
            return

        # Truncate long bill titles with a warning
        if len(bill_title) > BILL_TITLE_MAX_LENGTH:
            print(
                f"  WARNING: {bill_number} title truncated"
                f" ({len(bill_title)} -> {BILL_TITLE_MAX_LENGTH} chars)"
            )
            bill_title = bill_title[:BILL_TITLE_MAX_LENGTH]

        # Create RollCall summary
        rollcall = RollCall(
            session=self.session.label,
            bill_number=bill_number,
            bill_title=bill_title,
            vote_id=vote_id,
            vote_url=vote_url,
            vote_datetime=vote_datetime,
            vote_date=vote_date,
            chamber=chamber,
            motion=motion,
            vote_type=vote_type,
            result=result,
            short_title=short_title,
            sponsor=sponsor,
            sponsor_slugs=sponsor_slugs,
            yea_count=len(vote_categories["Yea"]),
            nay_count=len(vote_categories["Nay"]),
            present_passing_count=len(vote_categories["Present and Passing"]),
            absent_not_voting_count=len(vote_categories["Absent and Not Voting"]),
            not_voting_count=len(vote_categories["Not Voting"]),
            total_votes=total_votes,
            passed=passed,
        )
        self.rollcalls.append(rollcall)

        # Create individual votes
        for category, members in vote_categories.items():
            for member in members:
                iv = IndividualVote(
                    session=self.session.label,
                    bill_number=bill_number,
                    bill_title=bill_title,
                    vote_id=vote_id,
                    vote_datetime=vote_datetime,
                    vote_date=vote_date,
                    chamber=chamber,
                    motion=motion,
                    legislator_name=member["name"],
                    legislator_slug=member["slug"],
                    vote=category,
                )
                self.individual_votes.append(iv)

    # -- Vote parsing helpers --------------------------------------------------

    @staticmethod
    def _parse_vote_datetime(vote_id: str) -> str:
        """Extract ISO 8601 datetime from vote_id like 'je_20250320203513_*'."""
        match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", vote_id)
        if match:
            y, mo, d, h, mi, s = match.groups()
            return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
        return ""

    @staticmethod
    def _parse_vote_type_and_result(motion: str) -> tuple[str, str]:
        """Classify motion text into structured vote_type and result."""
        if not motion:
            return "", ""

        motion_lower = motion.lower()

        # Check for specific multi-word types first (order matters)
        type_prefixes = [
            ("Emergency Final Action", "emergency final action"),
            ("Final Action", "final action"),
            ("Committee of the Whole", "committee of the whole"),
            ("Consent Calendar", "consent calendar"),
        ]
        for vote_type, prefix in type_prefixes:
            if motion_lower.startswith(prefix):
                remainder = motion[len(prefix) :].strip(" -;")
                return vote_type, remainder if remainder else motion

        # Keyword-based classification
        if "override" in motion_lower and "veto" in motion_lower:
            return "Veto Override", motion
        if "conference committee" in motion_lower:
            return "Conference Committee", motion
        if "concur" in motion_lower:
            return "Concurrence", motion
        if motion_lower.startswith(("motion", "citing rule")):
            return "Procedural Motion", motion

        return "", motion

    @staticmethod
    def _derive_passed(result: str) -> bool | None:
        """Derive passed boolean from result text."""
        if not result:
            return None
        result_lower = result.lower()
        # Check failure patterns FIRST — "not passed" contains "passed", so the
        # positive regex would incorrectly match if checked first.
        if re.search(r"\b(not\s+passed|failed|rejected)\b", result_lower):
            return False
        if "sustained" in result_lower:
            # "Veto sustained" means the bill failed
            return False
        if re.search(r"\b(passed|adopted|prevailed|concurred)\b", result_lower):
            return True
        return None

    # -- Step 4: Enrich legislator data ----------------------------------------

    def enrich_legislators(self) -> None:
        """Fetch each legislator's page to get full name, party, and district."""
        print("\n" + "=" * 60)
        print("Step 4: Enriching legislator data (full name, party, district)...")
        print("=" * 60)

        slugs_to_fetch = [slug for slug, info in self.legislators.items() if "party" not in info]
        urls_to_fetch = [self.legislators[slug].get("member_url", "") for slug in slugs_to_fetch]
        urls_to_fetch = [u for u in urls_to_fetch if u]

        # Fetch phase (concurrent)
        fetched = self._fetch_many(urls_to_fetch, desc="Legislators")

        # Parse phase (sequential)
        for slug in slugs_to_fetch:
            info = self.legislators[slug]
            url = info.get("member_url", "")
            if not url:
                continue

            result = fetched.get(url)
            if not result or not result.ok:
                continue

            soup = BeautifulSoup(result.html, "lxml")
            parsed = self._extract_party_and_district(soup)

            # Use parsed full_name if found, else fall back to existing name
            info["full_name"] = parsed["full_name"] or info.get("name", "")
            info["party"] = parsed["party"]
            info["district"] = parsed["district"]

        print(f"  Enriched {len(slugs_to_fetch)} legislators")

        # Attach OpenStates OCD person IDs for stable cross-biennium identity
        from tallgrass.roster import load_slug_lookup

        slug_to_ocd = load_slug_lookup()
        matched_count = 0
        for slug, info in self.legislators.items():
            ocd_id = slug_to_ocd.get(slug, "")
            info["ocd_id"] = ocd_id
            if ocd_id:
                matched_count += 1
        if slug_to_ocd:
            print(f"  OCD IDs: {matched_count}/{len(self.legislators)} legislators matched")
        else:
            print("  OCD IDs: roster not synced (run `just roster-sync` to populate)")

    # -- Failure reporting -----------------------------------------------------

    def _save_failure_manifest(self, total_vote_pages: int) -> Path:
        """Write a JSON manifest of all failed vote page fetches."""
        manifest = {
            "session": self.session.label,
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "total_vote_pages": total_vote_pages,
            "successful": total_vote_pages - len(self.failures),
            "failed_count": len(self.failures),
            "failures": [
                {
                    "bill_number": f.bill_number,
                    "vote_text": f.vote_text,
                    "vote_url": f.vote_url,
                    "bill_path": f.bill_path,
                    "status_code": f.status_code,
                    "error_type": f.error_type,
                    "error_message": f.error_message,
                    "timestamp": f.timestamp,
                }
                for f in self.failures
            ],
        }
        manifest_path = self.output_dir / "failure_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    @staticmethod
    def _parse_vote_tally(vote_text: str) -> tuple[int, int, int] | None:
        """Parse 'Yea: 63 Nay: 59' into (yea, nay, margin) or None if unparseable."""
        m = re.search(r"Yea:\s*(\d+)\s*Nay:\s*(\d+)", vote_text)
        if not m:
            return None
        yea, nay = int(m.group(1)), int(m.group(2))
        return yea, nay, abs(yea - nay)

    def _save_missing_votes_doc(self, total_vote_pages: int) -> Path:
        """Write a standalone missing_votes.md documenting failed vote page fetches."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        successful = total_vote_pages - len(self.failures)

        # Parse tallies and sort by margin (closest first), unparseable last
        rows: list[tuple[int | None, FetchFailure, tuple[int, int, int] | None]] = []
        for f in self.failures:
            tally = self._parse_vote_tally(f.vote_text)
            rows.append((tally[2] if tally else None, f, tally))
        rows.sort(key=lambda r: (r[0] is None, r[0] or 0))

        lines = [
            f"# Missing Votes — {self.session.label}",
            "",
            f"Scraped: {now}",
            f"Success rate: {successful}/{total_vote_pages}"
            f" ({successful / total_vote_pages * 100:.1f}%)"
            if total_vote_pages
            else "",
            "",
            "| Bill | Tally | Margin | Error | URL |",
            "|------|-------|--------|-------|-----|",
        ]

        for margin, f, tally in rows:
            if tally:
                yea, nay, mg = tally
                tally_str = f"{yea}-{nay}"
                margin_str = f"**{mg}**" if mg <= 10 else str(mg)
            else:
                tally_str = f.vote_text
                margin_str = "?"
            status = f"{f.status_code}" if f.status_code else f.error_type
            lines.append(
                f"| {f.bill_number} | {tally_str} | {margin_str} | {status} | {f.vote_url} |"
            )

        lines.extend(
            [
                "",
                "Re-running the scraper retries failed pages (they are never cached).",
                "For persistent 404s, check legislative journals"
                " or the Kansas State Library archives.",
                "",
            ]
        )

        doc_path = self.output_dir / "missing_votes.md"
        doc_path.write_text("\n".join(lines), encoding="utf-8")
        return doc_path

    def _print_failure_summary(self) -> None:
        """Print a grouped summary of all failed vote page fetches."""
        if not self.failures:
            return

        print("\n" + "!" * 60)
        print(f"  WARNING: {len(self.failures)} vote page(s) failed to fetch")
        print("!" * 60)

        # Group by error_type
        by_type: dict[str, list[FetchFailure]] = {}
        for f in self.failures:
            by_type.setdefault(f.error_type, []).append(f)

        for error_type, failures in sorted(by_type.items()):
            print(f"\n  {error_type} ({len(failures)}):")
            for f in failures:
                status = f"[HTTP {f.status_code}]" if f.status_code else f"[{f.error_type}]"
                print(f"    {f.bill_number:10s} {f.vote_text:40s} {status}")

        manifest_path = self.output_dir / "failure_manifest.json"
        print(f"\n  Failure manifest: {manifest_path}")
        print("  Failed pages are not cached — re-run to retry automatically.")

    # -- Main runner -----------------------------------------------------------

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        """Format elapsed seconds as 'Xm Ys' or 'X.Xs'."""
        if seconds >= 60:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        return f"{seconds:.1f}s"

    def run(self, enrich: bool = True) -> None:
        """Run the full scraping pipeline."""
        start = time.time()
        step_times: list[tuple[str, float]] = []
        print("=" * 60)
        print(f"  Kansas Legislature {self.session.label} Vote Scraper")
        print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Target: {BASE_URL}{self.session.bills_path}")
        print("=" * 60)

        t = time.time()
        bill_urls = self.get_bill_urls()
        step_times.append(("Bill URLs + API filter", 0.0))

        # Pre-filter using KLISS API to avoid fetching bills without votes
        filtered_urls, self.bill_metadata = self._filter_bills_with_votes(bill_urls)
        step_times[-1] = ("Bill URLs + API filter", time.time() - t)

        t = time.time()
        vote_links = self.get_vote_links(filtered_urls)
        step_times.append(("Scan bill pages", time.time() - t))

        # Load member directory if any vote links are ODT (needed for name resolution)
        if any(vl.is_odt for vl in vote_links):
            self._load_member_directory()

        t = time.time()
        self.parse_vote_pages(vote_links)
        step_times.append(("Parse vote pages", time.time() - t))

        if enrich and self.legislators:
            t = time.time()
            self.enrich_legislators()
            step_times.append(("Enrich legislators", time.time() - t))

        t = time.time()
        save_csvs(
            output_dir=self.output_dir,
            output_name=self.session.output_name,
            individual_votes=self.individual_votes,
            rollcalls=self.rollcalls,
            legislators=self.legislators,
            bill_actions=self.bill_actions,
        )
        step_times.append(("Save CSVs", time.time() - t))

        if self.failures:
            self._save_failure_manifest(len(vote_links))
            self._save_missing_votes_doc(len(vote_links))

        elapsed = time.time() - start

        print("\n" + "=" * 60)
        print(f"  Complete! Total elapsed: {self._fmt_elapsed(elapsed)}")
        print(f"  Output directory: {self.output_dir.absolute()}")
        print("=" * 60)
        print("\nStep timing:")
        for label, secs in step_times:
            print(f"  {label:30s} {self._fmt_elapsed(secs):>8s}")
        print(f"  {'':30s} {'--------':>8s}")
        print(f"  {'Total':30s} {self._fmt_elapsed(elapsed):>8s}")
        print("\nFiles created:")
        print(
            f"  {self.session.output_name}_votes.csv"
            f"         - {len(self.individual_votes)} individual votes"
        )
        print(
            f"  {self.session.output_name}_rollcalls.csv"
            f"     - {len(self.rollcalls)} roll call summaries"
        )
        print(
            f"  {self.session.output_name}_legislators.csv   - {len(self.legislators)} legislators"
        )
        if self.bill_actions:
            print(
                f"  {self.session.output_name}_bill_actions.csv"
                f"  - {len(self.bill_actions)} bill actions"
            )

        self._print_failure_summary()

    def clear_cache(self) -> None:
        """Remove cached HTML files to force fresh fetches."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache cleared.")
