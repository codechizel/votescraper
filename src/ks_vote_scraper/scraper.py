"""Core scraper class for Kansas Legislature roll call votes."""

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

from ks_vote_scraper.config import (
    BASE_URL,
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
from ks_vote_scraper.models import IndividualVote, RollCall
from ks_vote_scraper.output import save_csvs
from ks_vote_scraper.session import KSSession

# Compiled regex for extracting bill type and number from URLs
_BILL_URL_RE = re.compile(r"/(sb|hb|scr|hcr|sr|hr)(\d+)/", re.I)

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

    @property
    def ok(self) -> bool:
        return self.html is not None


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


class KSVoteScraper:
    """Scrapes Kansas Legislature roll call votes from kslegislature.gov."""

    def __init__(
        self,
        session: KSSession,
        output_dir: Path | None = None,
        delay: float = REQUEST_DELAY,
    ):
        self.session = session
        self.output_dir = output_dir or Path("data") / session.output_name
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
        self.failures: list[FetchFailure] = []

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

    def _get(self, url: str) -> FetchResult:
        """Fetch a URL with retries, caching, and rate limiting.

        Retry strategy varies by error type:
        - 404: max 2 attempts (one retry), no backoff
        - 5xx: exponential backoff (5s, 10s, 20s)
        - Timeout: exponential backoff
        - Connection error: fixed 5s delay
        """
        # Check cache first (no rate limiting needed)
        cache_key = url.replace("/", "_").replace(":", "_").replace("?", "_")
        cache_file = self.cache_dir / f"{cache_key[:200]}.html"
        if cache_file.exists():
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
    ) -> dict[str, FetchResult]:
        """Fetch multiple URLs concurrently using a thread pool.

        After the initial pass, transient failures (5xx, timeout, connection) are
        retried in up to ``max_waves`` additional passes with reduced concurrency
        and a cooldown between waves.  This lets the server recover from sustained
        load without the thundering-herd effect of all workers retrying at once.

        Returns a dict mapping each URL to its FetchResult.
        """
        # --- initial pass (full concurrency) ---
        results: dict[str, FetchResult] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(self._get, url): url for url in urls}
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

            # Reduce concurrency and slow rate limit for the retry wave
            self.delay = WAVE_DELAY
            try:
                with ThreadPoolExecutor(max_workers=WAVE_WORKERS) as executor:
                    future_to_url = {executor.submit(self._get, url): url for url in failed_urls}
                    for future in tqdm(
                        as_completed(future_to_url),
                        total=len(future_to_url),
                        desc=f"Wave {wave}",
                        unit="page",
                    ):
                        url = future_to_url[future]
                        result = future.result()
                        if result.ok or result.error_type not in _RETRIABLE:
                            results[url] = result
                        else:
                            # Keep the latest failure info
                            results[url] = result
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

        The KS legislature loads all bills client-side on one page,
        so we grab all links matching the bill URL pattern.
        """
        print("=" * 60)
        print("Step 1: Fetching bill URLs from listing pages...")
        print("=" * 60)

        bill_urls = set()
        pattern = self.session.bill_url_pattern

        for label, path in [
            ("All Bills", self.session.bills_path),
            ("Senate Bills", self.session.senate_bills_path),
            ("House Bills", self.session.house_bills_path),
        ]:
            print(f"  Fetching {label}...")
            result = self._get(BASE_URL + path)
            if not result.ok:
                continue
            soup = BeautifulSoup(result.html, "lxml")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                if pattern.match(href):
                    bill_urls.add(href)

        bill_urls = sorted(bill_urls, key=self._bill_sort_key)
        print(f"  Found {len(bill_urls)} unique bill/resolution URLs")
        return bill_urls

    @staticmethod
    def _bill_sort_key(url: str):
        """Sort bills: SB before HB, then numerically."""
        match = _BILL_URL_RE.search(url)
        if match:
            prefix = match.group(1).lower()
            number = int(match.group(2))
            order = {"sb": 0, "sr": 1, "scr": 2, "hb": 3, "hr": 4, "hcr": 5}
            return (order.get(prefix, 9), number)
        return (99, 0)

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
                f"  API pre-filter failed"
                f" ({result.error_type}: {result.error_message}),"
                f" falling back to full scan"
            )
            return bill_urls, {}

        try:
            data = json.loads(result.html)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  API pre-filter returned invalid JSON ({e}), falling back to full scan")
            return bill_urls, {}

        # Build a set of bill codes that have votes (e.g., "sb1", "hb2124")
        # and capture metadata (short_title, sponsor) for each
        bills_with_votes = set()
        bill_metadata: dict[str, dict] = {}
        content = data if isinstance(data, list) else data.get("content", [])
        for bill in content:
            bill_no = bill.get("BILLNO", "")
            history = bill.get("HISTORY", [])
            has_vote = any("Yea:" in entry.get("status", "") for entry in history)
            if has_vote:
                code = _normalize_bill_code(bill_no)
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
            for link in soup.find_all("a", href=re.compile(r"vote_view")):
                href = link["href"]
                vote_links.append(
                    VoteLink(
                        bill_number=bill_number,
                        bill_path=bill_path,
                        vote_url=href if href.startswith("http") else BASE_URL + href,
                        vote_text=link.get_text(strip=True),
                    )
                )
                found_votes = True

            if found_votes:
                bills_with_votes += 1

            # Backfill sponsor from HTML when the API returned it empty
            bill_code = _normalize_bill_code(bill_number)
            meta = self.bill_metadata.get(bill_code, {})
            if meta and not meta.get("sponsor"):
                sponsor = self._extract_sponsor(soup)
                if sponsor:
                    meta["sponsor"] = sponsor
                    sponsors_backfilled += 1

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
    def _extract_sponsor(soup: BeautifulSoup) -> str:
        """Extract original sponsor from the bill page HTML.

        The sponsor is in a portlet structure:
          <div class="portlet-header">Original Sponsor</div>
          <div class="portlet-content">
            <ul><li><a>Senator Steffen</a></li></ul>
          </div>

        Falls back to empty string if not found.
        """
        header = soup.find(
            lambda tag: (
                tag.name == "div"
                and "portlet-header" in (tag.get("class") or [])
                and "Original Sponsor" in tag.get_text()
            )
        )
        if not header:
            return ""

        content = header.find_next_sibling("div", class_="portlet-content")
        if not content:
            return ""

        sponsors = []
        for li in content.find_all("li"):
            text = li.get_text(strip=True)
            if text:
                sponsors.append(text)

        return "; ".join(sponsors)

    # -- Step 3: Parse each vote page -----------------------------------------

    def parse_vote_pages(self, vote_links: list[VoteLink]) -> None:
        """Visit each vote_view page and extract individual legislator votes."""
        print("\n" + "=" * 60)
        print("Step 3: Parsing individual roll call vote pages...")
        print("=" * 60)

        # Fetch phase (concurrent)
        vote_urls = [vl.vote_url for vl in vote_links]
        fetched = self._fetch_many(vote_urls, desc="Fetching votes")

        # Parse phase (sequential — mutates self.rollcalls, self.individual_votes, etc.)
        for vl in tqdm(vote_links, desc="Parsing votes", unit="vote"):
            result = fetched.get(vl.vote_url)
            if not result or not result.ok:
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
                continue
            soup = BeautifulSoup(result.html, "lxml")
            self._parse_vote_page(soup, vl)

        print(f"  Parsed {len(self.rollcalls)} roll calls")
        print(f"  Collected {len(self.individual_votes)} individual votes")
        print(f"  Found {len(self.legislators)} unique legislators")

    def _parse_vote_page(self, soup: BeautifulSoup, vote_link: VoteLink):
        """Parse a single vote_view page."""
        bill_number = vote_link.bill_number
        vote_url = vote_link.vote_url

        # Extract vote ID from URL
        vote_id_match = re.search(r"vote_view/([^/]+)/", vote_url)
        vote_id = vote_id_match.group(1) if vote_id_match else vote_url

        # Parse precise datetime from vote_id
        vote_datetime = self._parse_vote_datetime(vote_id)

        # Extract bill title from <h4> (not h2)
        bill_title = ""
        title_heading = soup.find(
            "h4", string=re.compile(r"AN ACT|A CONCURRENT|A RESOLUTION|A JOINT", re.I)
        )
        if not title_heading:
            for h4 in soup.find_all("h4"):
                text = _clean_text(h4)
                if text.startswith("AN ACT") or len(text) > 50:
                    bill_title = text
                    break
        else:
            bill_title = _clean_text(title_heading)

        if not bill_title:
            for h4 in soup.find_all("h4"):
                text = _clean_text(h4)
                if len(text) > 30 and not text.startswith(
                    ("SB", "HB", "On roll", "Yea", "Nay", "Senate", "House")
                ):
                    bill_title = text
                    break

        # Extract vote description from <h3> (not h2)
        # Must use _clean_text (separator=" ") because h3 tags can contain
        # inline <a> elements for legislator names in Committee of the Whole
        # motions.  get_text(strip=True) would drop spaces around the <a>,
        # producing mangled text like "Amendment bySenator Franciscowas rejected".
        chamber = ""
        vote_date = ""
        motion = ""

        for h3 in soup.find_all("h3"):
            text = _clean_text(h3)
            match = re.match(
                r"(Senate|House)\s*-\s*(.+?)\s*-\s*(\d{2}/\d{2}/\d{4})$",
                text,
            )
            if match:
                chamber = match.group(1)
                motion = match.group(2).strip().rstrip(" -;")
                vote_date = match.group(3)
                break

        # Fallback: looser parse
        if not chamber:
            for h3 in soup.find_all("h3"):
                text = _clean_text(h3)
                if text.startswith("Senate") or text.startswith("House"):
                    chamber = "Senate" if text.startswith("Senate") else "House"
                    date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
                    if date_match:
                        vote_date = date_match.group(1)
                    motion = text.replace(chamber, "", 1).strip(" -")
                    if vote_date:
                        motion = motion.replace(vote_date, "").strip(" -")
                    break

        # Parse vote_type and result from motion text
        vote_type, result = self._parse_vote_type_and_result(motion)

        # Derive passed from result
        passed = self._derive_passed(result)

        # Look up short_title and sponsor from KLISS API metadata
        bill_code = _normalize_bill_code(bill_number)
        meta = self.bill_metadata.get(bill_code, {})
        short_title = meta.get("short_title", "")
        sponsor = meta.get("sponsor", "")

        # Parse vote categories
        vote_categories: dict[str, list[dict]] = {cat: [] for cat in VOTE_CATEGORIES}

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
                    slug = re.search(r"/members/([^/]+)/", href)
                    slug = slug.group(1) if slug else ""

                    if name:
                        vote_categories[current_category].append(
                            {
                                "name": name,
                                "slug": slug,
                            }
                        )

                        if slug and slug not in self.legislators:
                            leg_chamber = ""
                            if slug.startswith("sen_"):
                                leg_chamber = "Senate"
                            elif slug.startswith("rep_"):
                                leg_chamber = "House"
                            self.legislators[slug] = {
                                "name": name,
                                "slug": slug,
                                "chamber": leg_chamber,
                                "member_url": (
                                    f"{BASE_URL}{href}" if not href.startswith("http") else href
                                ),
                            }

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

        # Create RollCall summary
        rollcall = RollCall(
            session=self.session.label,
            bill_number=bill_number,
            bill_title=bill_title[:500],
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
                    bill_title=bill_title[:500],
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

            # Full name from <h1> containing "Senator" or "Representative"
            name_h1 = soup.find("h1", string=re.compile(r"^(Senator|Representative)\s+"))
            if name_h1:
                full_name = _clean_text(name_h1)
                # Strip title prefix and leadership suffix
                full_name = re.sub(r"^(Senator|Representative)\s+", "", full_name)
                full_name = re.sub(r"\s+-\s+.*$", "", full_name)
                info["full_name"] = full_name
            else:
                info["full_name"] = info.get("name", "")

            # Party and district from <h2> containing "District \d+"
            info["party"] = ""
            info["district"] = ""
            dist_h2 = soup.find("h2", string=re.compile(r"District\s+\d+"))
            if dist_h2:
                h2_text = dist_h2.get_text(strip=True)
                # e.g. "District 27 - Republican"
                dist_match = re.search(r"District\s+(\d+)", h2_text)
                if dist_match:
                    info["district"] = dist_match.group(1)
                if "Republican" in h2_text:
                    info["party"] = "Republican"
                elif "Democrat" in h2_text:
                    info["party"] = "Democrat"

        print(f"  Enriched {len(slugs_to_fetch)} legislators")

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

    def _print_failure_summary(self):
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
        )
        step_times.append(("Save CSVs", time.time() - t))

        if self.failures:
            self._save_failure_manifest(len(vote_links))

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

        self._print_failure_summary()

    def clear_cache(self) -> None:
        """Remove cached HTML files to force fresh fetches."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache cleared.")
