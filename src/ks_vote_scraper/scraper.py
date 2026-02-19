"""Core scraper class for Kansas Legislature roll call votes."""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    USER_AGENT,
)
from ks_vote_scraper.models import IndividualVote, RollCall
from ks_vote_scraper.output import save_csvs
from ks_vote_scraper.session import KSSession


class KSVoteScraper:
    """Scrapes Kansas Legislature roll call votes from kslegislature.gov."""

    def __init__(
        self,
        session: KSSession,
        output_dir: Optional[Path] = None,
        delay: float = REQUEST_DELAY,
    ):
        self.session = session
        self.output_dir = output_dir or Path("data") / f"ks_{session.output_name}"
        self.cache_dir = self.output_dir / ".cache"
        self.delay = delay
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

        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- HTTP helpers ----------------------------------------------------------

    def _get(self, url: str) -> Optional[str]:
        """Fetch a URL with retries, caching, and rate limiting."""
        # Check cache first (no rate limiting needed)
        cache_key = url.replace("/", "_").replace(":", "_").replace("?", "_")
        cache_file = self.cache_dir / f"{cache_key[:200]}.html"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

        for attempt in range(MAX_RETRIES):
            try:
                # Rate limit only actual network requests
                with self._rate_lock:
                    elapsed = time.monotonic() - self._last_request_time
                    if elapsed < self.delay:
                        time.sleep(self.delay - elapsed)
                    self._last_request_time = time.monotonic()

                resp = self.http.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                html = resp.text
                try:
                    cache_file.write_text(html, encoding="utf-8")
                except OSError:
                    pass  # cache write failure is non-fatal
                return html
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  Retry {attempt + 1}/{MAX_RETRIES} for {url}: {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  Failed after {MAX_RETRIES} attempts: {url}: {e}")
                    return None

    def _fetch_many(self, urls: list[str], desc: str = "Fetching") -> dict[str, Optional[str]]:
        """Fetch multiple URLs concurrently using a thread pool.

        Returns a dict mapping each URL to its HTML content (or None on failure).
        """
        results: dict[str, Optional[str]] = {}
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
            html = self._get(BASE_URL + path)
            if not html:
                continue
            soup = BeautifulSoup(html, "lxml")

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
        match = re.search(r"/(sb|hb|scr|hcr|sr|hr)(\d+)/", url, re.I)
        if match:
            prefix = match.group(1).lower()
            number = int(match.group(2))
            order = {"sb": 0, "sr": 1, "scr": 2, "hb": 3, "hr": 4, "hcr": 5}
            return (order.get(prefix, 9), number)
        return (99, 0)

    # -- Step 1b: KLISS API pre-filter -----------------------------------------

    def _filter_bills_with_votes(
        self, bill_urls: list[str]
    ) -> tuple[list[str], dict[str, dict]]:
        """Use the KLISS API to identify bills that have roll call votes.

        Fetches the bill_status API endpoint and checks each bill's history
        for "Yea:" in the status field, which indicates a recorded vote.
        Returns (filtered_urls, bill_metadata) where bill_metadata maps
        normalized codes like "sb1" to {"short_title": ..., "sponsor": ...}.

        Falls back to (full list, {}) if the API call fails.
        """
        api_url = f"{BASE_URL}{self.session.api_path}/bill_status/"
        print(f"\n  Pre-filtering via KLISS API: {api_url}")

        try:
            with self._rate_lock:
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < self.delay:
                    time.sleep(self.delay - elapsed)
                self._last_request_time = time.monotonic()

            resp = self.http.get(api_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  API pre-filter failed ({e}), falling back to full scan")
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
                # Normalize: "SB 1" -> "sb1", "HB 2124" -> "hb2124"
                code = re.sub(r"\s+", "", bill_no).lower()
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
            match = re.search(r"/(sb|hb|scr|hcr|sr|hr)(\d+)/", url, re.I)
            if match:
                code = f"{match.group(1).lower()}{match.group(2)}"
                if code in bills_with_votes:
                    filtered.append(url)

        print(f"  API filter: {len(filtered)} of {len(bill_urls)} bills have votes")
        return filtered, bill_metadata

    # -- Step 2: Find vote links on each bill page ----------------------------

    def get_vote_links(self, bill_urls: list[str]) -> list[dict]:
        """Visit each bill page and extract vote_view links."""
        print("\n" + "=" * 60)
        print("Step 2: Scanning bill pages for roll call vote links...")
        print("=" * 60)

        # Fetch phase (concurrent)
        full_urls = [BASE_URL + path for path in bill_urls]
        url_to_html = self._fetch_many(full_urls, desc="Scanning bills")

        # Parse phase (sequential)
        vote_links = []
        bills_with_votes = 0

        for bill_path in bill_urls:
            url = BASE_URL + bill_path
            html = url_to_html.get(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")
            bill_number = self._extract_bill_number(soup, bill_path)

            found_votes = False
            for link in soup.find_all("a", href=re.compile(r"vote_view")):
                href = link["href"]
                text = link.get_text(strip=True)
                vote_links.append({
                    "bill_number": bill_number,
                    "bill_path": bill_path,
                    "vote_url": href if href.startswith("http") else BASE_URL + href,
                    "vote_text": text,
                })
                found_votes = True

            if found_votes:
                bills_with_votes += 1

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

        match = re.search(r"/(sb|hb|scr|hcr|sr|hr)(\d+)/", bill_path, re.I)
        if match:
            return f"{match.group(1).upper()} {match.group(2)}"
        return bill_path

    # -- Step 3: Parse each vote page -----------------------------------------

    def parse_vote_pages(self, vote_links: list[dict]):
        """Visit each vote_view page and extract individual legislator votes."""
        print("\n" + "=" * 60)
        print("Step 3: Parsing individual roll call vote pages...")
        print("=" * 60)

        # Fetch phase (concurrent)
        vote_urls = [vl["vote_url"] for vl in vote_links]
        url_to_html = self._fetch_many(vote_urls, desc="Fetching votes")

        # Parse phase (sequential — mutates self.rollcalls, self.individual_votes, etc.)
        for vl in tqdm(vote_links, desc="Parsing votes", unit="vote"):
            html = url_to_html.get(vl["vote_url"])
            if not html:
                continue
            soup = BeautifulSoup(html, "lxml")
            self._parse_vote_page(soup, vl)

        print(f"  Parsed {len(self.rollcalls)} roll calls")
        print(f"  Collected {len(self.individual_votes)} individual votes")
        print(f"  Found {len(self.legislators)} unique legislators")

    def _parse_vote_page(self, soup: BeautifulSoup, vote_link: dict):
        """Parse a single vote_view page."""
        bill_number = vote_link["bill_number"]
        vote_url = vote_link["vote_url"]

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
                text = h4.get_text(strip=True)
                if text.startswith("AN ACT") or len(text) > 50:
                    bill_title = text
                    break
        else:
            bill_title = title_heading.get_text(strip=True)

        if not bill_title:
            for h4 in soup.find_all("h4"):
                text = h4.get_text(strip=True)
                if len(text) > 30 and not text.startswith(
                    ("SB", "HB", "On roll", "Yea", "Nay", "Senate", "House")
                ):
                    bill_title = text
                    break

        # Extract vote description from <h3> (not h2)
        chamber = ""
        vote_date = ""
        motion = ""

        for h3 in soup.find_all("h3"):
            text = h3.get_text(strip=True)
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
                text = h3.get_text(strip=True)
                if text.startswith("Senate") or text.startswith("House"):
                    chamber = "Senate" if text.startswith("Senate") else "House"
                    date_match = re.search(r"(\d{2}/\d{2}/\d{4})", text)
                    if date_match:
                        vote_date = date_match.group(1)
                    motion = text.replace(chamber, "").strip(" -")
                    if vote_date:
                        motion = motion.replace(vote_date, "").strip(" -")
                    break

        # Parse vote_type and result from motion text
        vote_type, result = self._parse_vote_type_and_result(motion)

        # Derive passed from result
        passed = self._derive_passed(result)

        # Look up short_title and sponsor from KLISS API metadata
        bill_code = re.sub(r"\s+", "", bill_number).lower()
        meta = self.bill_metadata.get(bill_code, {})
        short_title = meta.get("short_title", "")
        sponsor = meta.get("sponsor", "")

        # Parse vote categories
        vote_categories = {
            "Yea": [],
            "Nay": [],
            "Present and Passing": [],
            "Absent and Not Voting": [],
            "Not Voting": [],
        }

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
                        vote_categories[current_category].append({
                            "name": name,
                            "slug": slug,
                        })

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
                remainder = motion[len(prefix):].strip(" -;")
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
    def _derive_passed(result: str) -> Optional[bool]:
        """Derive passed boolean from result text."""
        if not result:
            return None
        result_lower = result.lower()
        if re.search(r"\b(passed|adopted|prevailed|concurred)\b", result_lower):
            return True
        if re.search(r"\b(failed|rejected)\b", result_lower):
            return False
        if "sustained" in result_lower:
            # "Veto sustained" means the bill failed
            return False
        return None

    # -- Step 4: Enrich legislator data ----------------------------------------

    def enrich_legislators(self):
        """Fetch each legislator's page to get full name, party, and district."""
        print("\n" + "=" * 60)
        print("Step 4: Enriching legislator data (full name, party, district)...")
        print("=" * 60)

        slugs_to_fetch = [
            slug for slug, info in self.legislators.items() if "party" not in info
        ]
        urls_to_fetch = [
            self.legislators[slug].get("member_url", "")
            for slug in slugs_to_fetch
        ]
        urls_to_fetch = [u for u in urls_to_fetch if u]

        # Fetch phase (concurrent)
        url_to_html = self._fetch_many(urls_to_fetch, desc="Legislators")

        # Parse phase (sequential)
        for slug in slugs_to_fetch:
            info = self.legislators[slug]
            url = info.get("member_url", "")
            if not url:
                continue

            html = url_to_html.get(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")

            # Full name from <h1>
            h1 = soup.find("h1")
            if h1:
                full_name = h1.get_text(strip=True)
                # Strip title prefix
                full_name = re.sub(
                    r"^(Senator|Representative)\s+", "", full_name
                )
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

    # -- Main runner -----------------------------------------------------------

    def run(self, enrich: bool = True):
        """Run the full scraping pipeline."""
        start = time.time()
        print("=" * 60)
        print(f"  Kansas Legislature {self.session.label} Vote Scraper")
        print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Target: {BASE_URL}{self.session.bills_path}")
        print("=" * 60)

        bill_urls = self.get_bill_urls()

        # Pre-filter using KLISS API to avoid fetching bills without votes
        filtered_urls, self.bill_metadata = self._filter_bills_with_votes(bill_urls)

        vote_links = self.get_vote_links(filtered_urls)
        self.parse_vote_pages(vote_links)

        if enrich and self.legislators:
            self.enrich_legislators()

        save_csvs(
            output_dir=self.output_dir,
            output_name=self.session.output_name,
            individual_votes=self.individual_votes,
            rollcalls=self.rollcalls,
            legislators=self.legislators,
        )

        elapsed = time.time() - start
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print("\n" + "=" * 60)
        print(f"  Complete! Elapsed: {minutes}m {seconds}s")
        print(f"  Output directory: {self.output_dir.absolute()}")
        print("=" * 60)
        print("\nFiles created:")
        print(f"  ks_{self.session.output_name}_votes.csv"
              f"         - {len(self.individual_votes)} individual votes")
        print(f"  ks_{self.session.output_name}_rollcalls.csv"
              f"     - {len(self.rollcalls)} roll call summaries")
        print(f"  ks_{self.session.output_name}_legislators.csv"
              f"   - {len(self.legislators)} legislators")

    def clear_cache(self):
        """Remove cached HTML files to force fresh fetches."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cache cleared.")
