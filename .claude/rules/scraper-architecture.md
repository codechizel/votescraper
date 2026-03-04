---
paths:
  - "src/**/*.py"
---

# Scraper Architecture

## Session Coverage (2011-2026)

| Biennium | Bill Discovery | Vote Format | Party Detection | Member Dir |
|----------|---------------|-------------|-----------------|------------|
| 2025-26 (91st) | HTML links | vote_view | `<h2>District N - Party` | HTML links |
| 2023-24 (90th) | HTML links | vote_view | `<h2>` | HTML links |
| 2021-22 (89th) | HTML links | vote_view | `<h2>` | HTML links |
| 2019-20 (88th) | JS (quoted keys) | vote_view | `<h2>` | JS fallback |
| 2017-18 (87th) | JS (unquoted keys) | vote_view | `<h2>` | JS fallback |
| 2015-16 (86th) | JS (unquoted keys) | vote_view | `<h3>Party:` | JS fallback |
| 2013-14 (85th) | JS (unquoted keys) | odt_view | `<h3>Party:` | JS fallback |
| 2011-12 (84th) | JS (unquoted keys) | odt_view* | `<h3>Party:` | JS fallback |

\* 84th session ODTs: ~30% contain only tally metadata (no individual legislator names).

## Concurrency Pattern

Two-phase: concurrent fetch via ThreadPoolExecutor (MAX_WORKERS=5), then sequential parse. Rate limiting is thread-safe via `threading.Lock()`. Never mutate shared state during fetch.

## Key Data Structures

- `FetchResult` — typed HTTP result; `content_bytes` for binary downloads (ODT); cached as `.bin`
- `FetchFailure` — recorded with bill context (number, motion, path); written to `failure_manifest.json`
- `VoteLink` — frozen dataclass for vote page links; `is_odt=True` routes to ODT parser
- `BillInfo` — frozen dataclass in `bills.py` (bill_number, url, raw_entry)
- `BillDocumentRef` — frozen dataclass in `text/models.py` (bill_number, document_type, url, version, session)
- `BillText` — frozen dataclass in `text/models.py` (extracted text + metadata)
- Module constants: `_BILL_URL_RE` (compiled regex), `VOTE_CATEGORIES` (5-tuple), `_normalize_bill_code()`

## Shared Bill Discovery (`bills.py`)

Bill URL discovery logic (HTML listing + JS fallback) is shared between the vote scraper and the bill text adapter. Extracted from `scraper.py` into `src/tallgrass/bills.py`:

- `discover_bill_urls(session, get_fn, base_url)` — returns `list[str]` of bill page URLs
- `discover_bills(session, get_fn, base_url)` — returns `list[BillInfo]` with parsed bill numbers
- `parse_js_array()`, `parse_js_bill_data()` — JS data file parsing (unquoted keys, quoted keys)
- `bill_sort_key()` — natural sort: chamber prefix then numeric part
- `url_to_bill_number()` — extracts "SB 55" from URL path

The `get_fn: Callable[[str], object] | None` parameter decouples from scraper instance state. `KSVoteScraper.get_bill_urls()` delegates here. `KansasAdapter.discover_bills()` calls the same functions via `BillTextFetcher.get_html()`.

## Bill Text Subpackage (`text/`)

Separate subpackage for bill text retrieval, independent of the vote scraper:

```
src/tallgrass/text/
  models.py        — BillDocumentRef, BillText frozen dataclasses
  protocol.py      — StateAdapter Protocol (multi-state contract)
  kansas.py        — KansasAdapter: deterministic PDF URL construction
  fetcher.py       — BillTextFetcher: concurrent download + text extraction
  extractors.py    — PDF extraction via pdfplumber, text cleaning (pure functions)
  output.py        — CSV export (bill_texts.csv)
  cli.py           — tallgrass-text entry point
```

Multi-state-ready: `StateAdapter` Protocol defines the contract; adding a state requires one new file. `BillTextFetcher` is state-agnostic — takes `list[BillDocumentRef]` from any adapter. See ADR-0083.

## KanFocus Subpackage (`kanfocus/`)

Separate subpackage for scraping KanFocus (kanfocus.com) vote tally pages:

```
src/tallgrass/kanfocus/
  models.py      — KanFocusVoteRecord, KanFocusLegislator frozen dataclasses
  session.py     — session ID mapping (106-119), URL construction, vote_id generation
  parser.py      — parse_vote_page() pure function, HTML-to-text + legislator parsing
  slugs.py       — slug generation from "Name, R-32nd" format + cross-reference
  fetcher.py     — KanFocusFetcher: HTTP + Chrome cookie auth + cache + rate limiting
  output.py      — convert to standard format + gap-fill merge
  crossval.py    — cross-validate KF cache vs JE CSVs (read-only diagnostic, ADR-0097)
  cli.py         — tallgrass-kanfocus entry point + data archiving
```

Coverage: 78th-91st (1999-2026). Uses `kf_` prefix for vote_ids. Conservative rate limiting (7s default, 12s recommended during business hours). See ADR-0088.

**Cross-validation** (`--mode crossval`): re-parses KanFocus cache and compares overlapping rollcalls against kslegislature.gov CSVs. Matches on `(normalize_bill_number(bill_number), chamber, vote_date)` with tally-based sub-matching `(yea, nay, nv_total)` for multi-motion disambiguation. Individual vote comparison uses slug matching with name-based fallback (full normalized name, then last-name-only match) for remaining slug mismatches. Handles "Sub for" prefix stripping and ANV/NV category ambiguity. No network access, no data mutation. Writes `crossval_report.md` to data dir. See ADR-0097.

**Authentication**: Extracts session cookies from Chrome's encrypted cookie database on macOS (Keychain AES key, PBKDF2, 32-byte app-bound prefix skip). Requires active KanFocus login in Chrome.

**Parser**: Auto-detects HTML input and converts to text via BeautifulSoup before regex extraction. Handles newline-separated metadata format from `<table>` layout and strips `document.write()` JS from legislator sections.

**Data archiving**: After each successful biennium, raw HTML cache is copied to `data/kanfocus_archive/{output_name}/`. `--clear-cache` blocked unless archive exists. Cache is restart-safe (hash-keyed HTML files).

## Static Parsing Helpers

All `@staticmethod` on `KSVoteScraper` — callable without a scraper instance, tested directly:

| Method | Extracts | Pitfalls |
|--------|----------|----------|
| `_extract_bill_number(soup, bill_path)` | Bill number from `<h2>` or URL | — |
| `_extract_sponsor(soup)` | Sponsor from portlet structure | — |
| `_extract_bill_title(soup)` | Title from `<h4>` (3-tier fallback) | #1 |
| `_extract_chamber_motion_date(soup)` | Chamber, motion, date from `<h3>` (2-tier) | #1, #5 |
| `_parse_vote_categories(soup)` | Vote categories + new legislators | #3 |
| `_extract_party_and_district(soup)` | Name, party, district from legislator page | #2, #2b, #5 |

`_parse_vote_page()` is a thin coordinator that calls these helpers and builds `RollCall`/`IndividualVote` records. `enrich_legislators()` calls `_extract_party_and_district()` in its parse loop.

## Retry Strategy

`_get()` returns `FetchResult` and classifies errors:

- **404** -> `"permanent"`, max 2 attempts
- **5xx** -> `"transient"`, exponential backoff with jitter
- **Timeout** -> `"timeout"`, exponential backoff with jitter
- **Connection error** -> `"connection"`, fixed delay
- **Other 4xx** -> `"permanent"`, no retry
- **HTTP 200 error page** -> `"permanent"`, detected by HTML heuristics

All HTTP requests go through `_get()` (including KLISS API) for consistent retries, rate limiting, caching, and error-page detection.

## Retry Waves

When sustained server failures cause thundering herd at the `_fetch_many()` level:

1. Collect transient failures (5xx, timeout, connection)
2. Wait `WAVE_COOLDOWN` (90s)
3. Re-dispatch with reduced load: `WAVE_WORKERS=2`, `WAVE_DELAY=0.5s`
4. Repeat up to `RETRY_WAVES=3` times

Jitter on per-URL backoff prevents thundering herd within each wave. See ADR-0009.

## ODT Parser (2011-2014)

`odt_parser.py` — pure functions, no I/O, stdlib only (zipfile + xml.etree):
- `parse_odt_votes()`: bytes -> (RollCall, IndividualVote, legislators)
- Extracts `content.xml` from ZIP, parses `<text:user-field-decl>` metadata
- Maps House/Senate vote category variants to canonical names
- Last-name-only resolution via member directory (handles initials, ambiguity)

## Vote Deduplication

ODT sessions (2011-2014) can link the same vote page from multiple bills. `save_csvs()` deduplicates by `(legislator_slug, vote_id)`, keeping first occurrence.

## Session URL Logic

- Current (2025-26): `/li/b2025_26/...`
- Historical (2023-24): `/li_2024/b2023_24/...`
- Special (2024): `/li_2024s/...`
- API: current uses `/li/api/v13/rev-1`, historical uses `/li_{end_year}/api/v13/rev-1`
- `CURRENT_BIENNIUM_START` in session.py must be updated when a new biennium becomes current (next: 2027).
