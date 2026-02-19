# Architecture

## Overview

The Kansas Legislature Vote Scraper is a Python CLI tool that extracts roll call vote data from kslegislature.gov and exports it to structured CSV files. It's designed for statistical and Bayesian analysis of legislative voting patterns.

## Pipeline

The scraper runs a 5-step pipeline:

```
Step 1: get_bill_urls()
  ↓  Fetches listing pages, extracts bill URLs via regex
Step 1b: _filter_bills_with_votes()
  ↓  KLISS API pre-filter — reduces ~1000 bills to ~200 with votes
  ↓  Also captures short_title + sponsor metadata (no extra requests)
Step 2: get_vote_links()
  ↓  Fetches each bill page, extracts vote_view links
Step 3: parse_vote_pages()
  ↓  Fetches + parses each vote page → rollcalls + individual votes
Step 4: enrich_legislators()
  ↓  Fetches member pages → full name, party, district
Step 5: save_csvs()
     Writes 3 CSV files
```

## Module Responsibilities

### config.py
Constants only. No logic. `BASE_URL`, rate limiting params, `USER_AGENT`.

### session.py
`KSSession` frozen dataclass. Encapsulates the complex URL prefix logic that varies by biennium (current vs historical vs special session). All URL construction flows through properties on this class.

Key constant: `CURRENT_BIENNIUM_START` must be manually updated when the legislature starts a new biennium (every 2 years, odd years).

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
         │ dict[url, html]
         ▼
┌────────────────────┐
│  Sequential parse   │  Mutates shared state safely
│  (parse phase)      │  (rollcalls, individual_votes, legislators)
└────────────────────┘
```

Every multi-URL operation follows this two-phase pattern. The fetch phase only reads URLs and returns HTML. The parse phase runs single-threaded and is the only code that mutates `self.rollcalls`, `self.individual_votes`, or `self.legislators`.

## Caching

HTML responses are cached to disk at `data/ks_{session}/.cache/`. Cache key is the URL with `/`, `:`, `?` replaced by `_`, truncated to 200 chars. Cache hits skip rate limiting entirely. Use `--clear-cache` to force fresh fetches.

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
2. **Metadata enrichment**: Captures `SHORTTITLE` and `ORIGINAL_SPONSOR` from the same API call — no additional HTTP requests needed

The API response format varies: sometimes a raw JSON array, sometimes `{"content": [...]}`.

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
