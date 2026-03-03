"""ALEC model legislation scraper.

Two-phase scraper: paginate listing pages to discover bill URLs,
then fetch individual bill pages for full text.  Caches raw HTML
by content hash.  Rate-limited for polite scraping.
"""

import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from tallgrass.alec.models import ALECModelBill
from tallgrass.config import USER_AGENT

# ── Constants ────────────────────────────────────────────────────────────────

ALEC_BASE_URL = "https://alec.org"
LISTING_URL = f"{ALEC_BASE_URL}/model-policy/page/{{page}}/"
FIRST_PAGE_URL = f"{ALEC_BASE_URL}/model-policy/"

REQUEST_DELAY = 1.5  # seconds between requests (polite scraping)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
MAX_WORKERS = 3  # conservative concurrency for external site

# Known max pages — discover dynamically but cap as safety valve
MAX_PAGES = 100


# ── Scraper Functions ────────────────────────────────────────────────────────


def _create_session() -> requests.Session:
    """Create an HTTP session with appropriate headers."""
    http = requests.Session()
    http.headers.update({"User-Agent": USER_AGENT})
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=MAX_WORKERS)
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http


def _fetch_with_cache(
    http: requests.Session,
    url: str,
    cache_dir: Path,
    rate_lock: threading.Lock,
    last_request: list[float],
) -> str | None:
    """Fetch a URL with caching and rate limiting.

    Returns HTML text on success, None on failure.
    Cache key: SHA-256 of URL (first 16 chars).
    """
    cache_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{cache_hash}.html"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            with rate_lock:
                elapsed = time.monotonic() - last_request[0]
                if elapsed < REQUEST_DELAY:
                    time.sleep(REQUEST_DELAY - elapsed)
                last_request[0] = time.monotonic()

            resp = http.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            html = resp.text
            try:
                cache_file.write_text(html, encoding="utf-8")
            except OSError:
                pass
            return html

        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
            continue

    return None


def _detect_max_page(soup: BeautifulSoup) -> int:
    """Detect the last page number from pagination links."""
    # Look for pagination links like /model-policy/page/54/
    page_nums = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"/model-policy/page/(\d+)/", a["href"])
        if m:
            page_nums.append(int(m.group(1)))
    return max(page_nums) if page_nums else 1


def _parse_listing_entry(article: BeautifulSoup) -> dict | None:
    """Extract bill metadata from a listing page article element.

    Returns dict with keys: url, title, category, bill_type, date, task_force.
    """
    # Title + URL from the heading link
    heading = article.find(["h2", "h3", "h4"])
    if not heading:
        return None
    link = heading.find("a", href=True)
    if not link:
        return None

    url = link["href"]
    title = link.get_text(strip=True)
    if not url or not title:
        return None

    # Metadata from spans/divs below the title
    meta = {
        "url": url,
        "title": title,
        "category": "",
        "bill_type": "",
        "date": "",
        "task_force": "",
    }

    # Look for category, type, date in entry-meta or similar containers
    for span in article.find_all(["span", "div", "p"]):
        text = span.get_text(strip=True)
        if not text:
            continue
        lower = text.lower()
        # Model Policy / Model Resolution / Model Act
        if lower.startswith("model ") and not meta["bill_type"]:
            meta["bill_type"] = text
        # Task force names
        if "task force" in lower and not meta["task_force"]:
            meta["task_force"] = text

    # Category from breadcrumb, tag, or parent category section
    for a in article.find_all("a", href=True):
        href = a["href"]
        if "/model-policy-category/" in href or "/issue/" in href:
            meta["category"] = a.get_text(strip=True)
            break

    return meta


def enumerate_bills(
    http: requests.Session,
    cache_dir: Path,
    rate_lock: threading.Lock,
    last_request: list[float],
) -> list[dict]:
    """Paginate through ALEC listing pages, extract bill URLs + metadata.

    Returns list of dicts with keys: url, title, category, bill_type, date, task_force.
    """
    # Fetch first page to detect pagination
    first_html = _fetch_with_cache(http, FIRST_PAGE_URL, cache_dir, rate_lock, last_request)
    if not first_html:
        print("  Failed to fetch first listing page")
        return []

    soup = BeautifulSoup(first_html, "lxml")
    max_page = _detect_max_page(soup)
    max_page = min(max_page, MAX_PAGES)
    print(f"  Detected {max_page} listing pages")

    # Parse first page
    bills: list[dict] = []
    seen_urls: set[str] = set()

    def _parse_page(html: str) -> list[dict]:
        page_soup = BeautifulSoup(html, "lxml")
        entries = []
        for article in page_soup.find_all("article"):
            entry = _parse_listing_entry(article)
            if entry and entry["url"] not in seen_urls:
                seen_urls.add(entry["url"])
                entries.append(entry)
        return entries

    bills.extend(_parse_page(first_html))

    # Fetch remaining pages
    for page in tqdm(range(2, max_page + 1), desc="Listing pages"):
        url = LISTING_URL.format(page=page)
        html = _fetch_with_cache(http, url, cache_dir, rate_lock, last_request)
        if html:
            bills.extend(_parse_page(html))

    print(f"  Found {len(bills)} unique model bill listings")
    return bills


def _extract_bill_text(html: str) -> str:
    """Extract the model bill text from an individual bill page.

    Looks for the main content area, strips navigation and boilerplate.
    """
    soup = BeautifulSoup(html, "lxml")

    # Try content sections in order of specificity
    # 1. Entry content (WordPress standard)
    content = soup.find("div", class_=re.compile(r"entry-content|post-content|article-content"))
    if not content:
        # 2. Main content area
        content = soup.find("main") or soup.find("article")
    if not content:
        # 3. Fallback to body
        content = soup.find("body")

    if not content:
        return ""

    # Remove navigation, sidebars, footers
    for tag in content.find_all(["nav", "aside", "footer", "header", "script", "style"]):
        tag.decompose()

    # Remove share buttons, related posts
    for div in content.find_all("div", class_=re.compile(r"share|social|related|sidebar")):
        div.decompose()

    # Extract text
    text = content.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _extract_bill_metadata(html: str, listing_meta: dict) -> dict:
    """Extract additional metadata from individual bill page.

    Supplements listing metadata with date, task force, and bill type
    if not already populated.
    """
    soup = BeautifulSoup(html, "lxml")
    meta = dict(listing_meta)

    # Look for date in meta tags or structured data
    date_el = soup.find("time")
    if date_el and not meta.get("date"):
        dt = date_el.get("datetime", "") or date_el.get_text(strip=True)
        # Normalize to YYYY-MM-DD if possible
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", dt)
        if m:
            meta["date"] = m.group(0)

    # Look for bill type in page content if not from listing
    if not meta.get("bill_type"):
        for el in soup.find_all(["span", "div", "p"]):
            text = el.get_text(strip=True).lower()
            if text.startswith("model ") and len(text) < 30:
                meta["bill_type"] = el.get_text(strip=True)
                break

    # Task force from page content
    if not meta.get("task_force"):
        for el in soup.find_all(["span", "div", "p", "a"]):
            text = el.get_text(strip=True)
            if "task force" in text.lower() and len(text) < 100:
                meta["task_force"] = text
                break

    # Category from page taxonomy
    if not meta.get("category"):
        for a in soup.find_all("a", href=True):
            if "/model-policy-category/" in a["href"] or "/issue/" in a["href"]:
                meta["category"] = a.get_text(strip=True)
                break

    return meta


def fetch_bill_texts(
    http: requests.Session,
    bill_listings: list[dict],
    cache_dir: Path,
    rate_lock: threading.Lock,
    last_request: list[float],
) -> list[ALECModelBill]:
    """Fetch individual bill pages and extract text.

    Returns list of ALECModelBill with full text populated.
    """
    results: list[ALECModelBill] = []
    failed = 0

    def _process_one(listing: dict) -> ALECModelBill | None:
        url = listing["url"]
        html = _fetch_with_cache(http, url, cache_dir, rate_lock, last_request)
        if not html:
            return None

        text = _extract_bill_text(html)
        if not text or len(text) < 50:
            return None

        meta = _extract_bill_metadata(html, listing)

        return ALECModelBill(
            title=meta.get("title", ""),
            text=text,
            category=meta.get("category", ""),
            bill_type=meta.get("bill_type", ""),
            date=meta.get("date", ""),
            url=url,
            task_force=meta.get("task_force", ""),
        )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_listing = {
            executor.submit(_process_one, listing): listing for listing in bill_listings
        }
        for future in tqdm(
            as_completed(future_to_listing),
            total=len(bill_listings),
            desc="Fetching bill texts",
        ):
            try:
                bill = future.result()
                if bill is not None:
                    results.append(bill)
                else:
                    failed += 1
            except Exception:
                failed += 1

    results.sort(key=lambda b: b.title)
    print(f"  Extracted {len(results)} model bills ({failed} failed/empty)")
    return results


def scrape_alec_corpus(cache_dir: Path) -> list[ALECModelBill]:
    """Full ALEC scraping pipeline: enumerate listings -> fetch texts.

    Cache directory stores raw HTML for resume/incremental updates.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    http = _create_session()
    rate_lock = threading.Lock()
    last_request: list[float] = [0.0]

    # Step 1: Enumerate
    print("\nStep 1: Enumerating ALEC model bill listings...")
    listings = enumerate_bills(http, cache_dir, rate_lock, last_request)

    if not listings:
        print("\nNo listings found. Exiting.")
        return []

    # Step 2: Fetch texts
    print("\nStep 2: Fetching individual bill texts...")
    bills = fetch_bill_texts(http, listings, cache_dir, rate_lock, last_request)

    return bills
