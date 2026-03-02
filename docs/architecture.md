# Architecture

## Overview

The Kansas Legislature Vote Scraper is a Python CLI tool that extracts roll call vote data from kslegislature.gov and exports it to structured CSV files. It's designed for statistical and Bayesian analysis of legislative voting patterns.

## Pipeline

The scraper runs a 5-step pipeline:

```
Step 1: get_bill_urls()
  ↓  Fetches listing pages, extracts bill URLs via regex
  ↓  Falls back to JS data files for pre-2021 sessions
Step 1b: _filter_bills_with_votes()
  ↓  KLISS API pre-filter — reduces ~1000 bills to ~200 with votes
  ↓  Also captures short_title + sponsor metadata (no extra requests)
Step 2: get_vote_links()
  ↓  Fetches each bill page, extracts vote_view/odt_view links
  ↓  Sets VoteLink.is_odt=True for ODT links
Step 2b: _load_member_directory() (if ODT links present)
  ↓  Builds name→slug mapping for ODT last-name resolution
Step 3: parse_vote_pages()
  ↓  Routes HTML links → _parse_html_vote_pages()
  ↓  Routes ODT links → _parse_odt_vote_pages() (binary fetch + odt_parser)
Step 4: enrich_legislators()
  ↓  Fetches member pages → full name, party, district
  ↓  Pre-2015 fallback: h3 "Party:" when h2 has no party
Step 5: save_csvs()
     Writes 3 CSV files
```

## Module Responsibilities

### config.py
Constants only. No logic. `BASE_URL`, rate limiting params, `USER_AGENT`.

### session.py
`KSSession` frozen dataclass. Encapsulates the complex URL prefix logic that varies by biennium (current vs historical vs special session). All URL construction flows through properties on this class.

Key constant: `CURRENT_BIENNIUM_START` must be manually updated when the legislature starts a new biennium (every 2 years, odd years).

Properties for historical session support:
- `uses_odt`: True for pre-2015 sessions (ODT vote files instead of HTML)
- `js_data_paths`: Candidate JS data file URLs for pre-2021 sessions (bill discovery fallback)

### models.py
Two frozen dataclasses:
- `IndividualVote` — one legislator's vote on one roll call (the "long" table)
- `RollCall` — summary of one roll call with counts and metadata

### scraper.py
`KSVoteScraper` class. The bulk of the codebase. Contains:
- HTTP helpers with caching, retries, rate limiting
- Concurrent fetch via ThreadPoolExecutor
- All HTML parsing logic
- Vote type/result classification
- Legislator enrichment

Module-level constants and helpers:
- `_BILL_URL_RE` — compiled regex for extracting bill type/number from URLs
- `VOTE_CATEGORIES` — the 5 vote categories as a tuple
- `_normalize_bill_code()` — normalize "SB 1" → "sb1" for lookups

Dataclasses: `FetchResult` (with `content_bytes` for binary downloads), `FetchFailure`, `VoteLink` (with `is_odt` flag) — all frozen.

Historical session extensions:
- JS bill discovery fallback (`_get_bill_urls_from_js`, `_parse_js_bill_data`) for pre-2021 sessions
- Member directory (`_load_member_directory`) for ODT name resolution
- ODT vote parsing integration (`_parse_odt_vote_pages`) routes ODT links to the dedicated parser
- Pre-2015 party detection (h3 "Party:" fallback in `_extract_party_and_district()`)

Static parsing helpers (all `@staticmethod`, callable without scraper instance):
- `_extract_bill_number()` — bill number from `<h2>` or URL path
- `_extract_sponsor()` — sponsor from portlet structure
- `_extract_bill_title()` — 3-tier `<h4>` fallback (Pitfall #1)
- `_extract_chamber_motion_date()` — 2-tier `<h3>` fallback (Pitfalls #1, #5)
- `_parse_vote_categories()` — categories + new legislators from `<h2>`/`<h3>`/`<a>` scan (Pitfall #3)
- `_extract_party_and_district()` — name/party/district from legislator page (Pitfalls #2, #2b, #5)

### odt_parser.py
Pure-function module for parsing ODT (OpenDocument Text) vote files from 2011-2014 sessions. No I/O, no HTTP — takes bytes and context, returns `RollCall`/`IndividualVote` instances.

Key functions:
- `parse_odt_votes()` — main entry point, accepts ODT bytes and returns structured vote data
- `_extract_content_xml()` — unzips ODT archive, extracts content.xml
- `_parse_odt_metadata()` — parses user-field-decl XML elements (chamber, bill number, timestamp, tally)
- `_parse_odt_body_votes()` — parses vote categories from paragraph text (handles House/Senate naming variants)
- `_resolve_last_name()` — resolves last-name-only legislators via member directory with initial disambiguation

### output.py
CSV export. Uses `dataclasses.asdict()` for RollCall/IndividualVote. Legislators use explicit field list since they're stored as plain dicts.

### cli.py
argparse entry point. Minimal — constructs KSSession and KSVoteScraper, calls `run()`.

## Concurrency Model

```
┌────────────────────┐
│  ThreadPoolExecutor │  MAX_WORKERS = 5
│  (fetch phase)      │  Rate-limited via threading.Lock()
└────────┬───────────┘
         │ dict[url, FetchResult]
         ▼
┌────────────────────┐
│  Retry waves        │  Up to 3 waves for transient failures
│  (reduced load)     │  WAVE_WORKERS=2, WAVE_DELAY=0.5s, 90s cooldown
└────────┬───────────┘
         │ dict[url, FetchResult]  (failures overwritten by successes)
         ▼
┌────────────────────┐
│  Sequential parse   │  Mutates shared state safely
│  (parse phase)      │  (rollcalls, individual_votes, legislators)
└────────────────────┘
```

Every multi-URL operation follows this two-phase pattern. The fetch phase only reads URLs and returns HTML. If transient failures (5xx, timeout, connection) occur, `_fetch_many()` automatically retries them in up to 3 additional waves with reduced concurrency and a 90-second cooldown between waves — letting the server recover instead of hammering it. The parse phase runs single-threaded and is the only code that mutates `self.rollcalls`, `self.individual_votes`, or `self.legislators`.

## Caching

All responses fetched via `_get()` are cached to disk at `data/{output_name}/.cache/`, including KLISS API JSON. Cache key is the URL with `/`, `:`, `?` replaced by `_`, truncated to 200 chars. Cache hits skip rate limiting entirely. Use `--clear-cache` to force fresh fetches.

## Data Flow: Vote Page → Data Objects

A single vote page produces:
- 1 `RollCall` (summary with counts, vote_type, result, passed, short_title, sponsor)
- N `IndividualVote` records (one per legislator who voted/was absent)
- Updates to `self.legislators` dict (new slugs discovered)

The vote page HTML structure:
```html
<h2> <a href="...">SB 1</a> </h2>                    ← bill number
<h4>AN ACT exempting the state of Kansas from...</h4> ← bill title
<h3>Senate - Emergency Final Action - Passed as amended; - 03/20/2025</h3>  ← metadata
<h3>Yea - (33):</h3>                                  ← vote category
<a href="/members/sen_claeys/">Claeys</a>,             ← legislator vote
```

## KLISS API

The Kansas Legislative Information Systems and Services (KLISS) API provides structured bill data at `/li/api/v13/rev-1/bill_status/`. The scraper uses it for:

1. **Pre-filtering**: Only fetch bill pages for bills that have roll call votes (checks `HISTORY` entries for "Yea:" text)
2. **Metadata enrichment**: Captures `SHORTTITLE` and `ORIGINAL_SPONSOR` from the same API call — no additional HTTP requests needed. Stored on `self.bill_metadata` for downstream sponsor backfill in `get_vote_links()`.

The API call routes through `_get()` for retries, rate limiting, and caching. The response format varies: sometimes a raw JSON array, sometimes `{"content": [...]}`. The error-page guard in `_get()` only applies to HTML responses (detected by leading `<`), so JSON passes through safely.

## Vote Classification

Motion text like `"Emergency Final Action - Passed as amended"` is split into:
- `vote_type`: "Emergency Final Action"
- `result`: "Passed as amended"
- `passed`: True (derived from result keywords)

Classification priority:
1. Prefix match: Emergency Final Action > Final Action > Committee of the Whole > Consent Calendar
2. Keyword match: override+veto → Veto Override, conference committee → Conference Committee, concur → Concurrence
3. Start match: motion/citing rule → Procedural Motion
4. Fallback: empty vote_type, full motion as result
