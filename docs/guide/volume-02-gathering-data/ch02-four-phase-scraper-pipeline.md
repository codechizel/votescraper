# Chapter 2: The Four-Phase Scraper Pipeline

> *How do you turn a government website into a structured dataset? The same way you'd eat an elephant: one bite at a time.*

---

## The Big Picture

The Tallgrass scraper is a four-phase pipeline. Each phase does one job, produces a specific output, and hands it to the next phase. Running all four phases on a single biennium transforms a website into five CSV files containing every recorded vote.

Here's the pipeline at a glance:

```
Phase 1: DISCOVER        Phase 2: FILTER         Phase 3: PARSE          Phase 4: ENRICH
Find all bills    →    Keep only bills     →    Read every vote    →    Add legislator
on the website         with recorded votes       page and extract        party, district,
                                                  who voted what          and full name

~1,200 bill URLs       ~600 bill URLs            ~1,100 roll calls       181 legislators
                                                  ~95,000 votes           with metadata
```

**Think of it as a factory assembly line.** Raw materials (bill URLs) enter at one end. Each station refines them. Finished products (structured vote records) come out the other end. If any station breaks, the line stops — but you can restart from the last successful station without redoing earlier work, because every intermediate result is cached.

Let's walk through each phase using the 91st Legislature (2025–2026) as our running example.

## Phase 1: Discover — Find All Bills

The scraper's first job is to find every bill introduced in the session. This is like walking into the library and getting the catalog — you need to know what exists before you can start reading.

**How it works:**

1. The scraper fetches the session's bill listing pages — one for Senate bills, one for House bills, one for all bills combined.
2. It reads the HTML and extracts every link that matches the pattern for a bill page (e.g., `/li/b2025_26/measures/sb55/`).
3. It sorts them in a natural order: Senate bills first (SB, SR, SCR), then House bills (HB, HR, HCR), each group sorted numerically.

**What comes out:** A list of about 1,200 bill URLs for a typical session.

**What can go wrong:** For sessions before 2021, the bill listing pages don't exist as regular HTML. Instead, the website generates them with JavaScript — which means the raw HTML the scraper downloads is an empty shell. The solution is a fallback: the scraper downloads the JavaScript data files directly and parses the bill information from there. This sounds straightforward, but there's a catch — the JavaScript files from 2017 and earlier use an informal syntax where object keys aren't quoted (e.g., `{title: "SB 55"}` instead of `{"title": "SB 55"}`). The parser has to "fix" the JavaScript into valid JSON before it can read it. We'll cover this in detail in Chapter 3.

**Codebase:** `src/tallgrass/bills.py` (shared bill discovery module)

## Phase 2: Filter — Keep Only Bills with Votes

Not every bill gets a recorded vote. Many die in committee. Others pass by voice vote (where the presiding officer judges which side was louder, and no individual positions are recorded). Of the roughly 1,200 bills introduced each session, only about half have at least one recorded roll call vote.

Visiting all 1,200 bill pages to check whether each one has votes would be wasteful. Instead, the scraper takes a shortcut: it asks the **KLISS API**.

**How it works:**

1. The scraper calls the KLISS API's bill status endpoint, which returns metadata for every bill in the session as a JSON response.
2. For each bill, the API provides its full legislative history — every committee hearing, every floor action, every vote.
3. The scraper scans each bill's history for the telltale sign of a recorded vote: the string `"Yea:"` in an action description.
4. Bills that never had a recorded vote are filtered out.

**The bonus:** While querying the API, the scraper also collects valuable metadata that would otherwise require extra page visits: the bill's short title, its primary sponsor, and its complete action history (which later feeds the bill lifecycle analysis in the pipeline). This is like asking the librarian "which volumes have vote tallies?" and also getting a summary card for each volume at no extra cost.

**What comes out:** About 600 bill URLs (the ones with recorded votes) plus metadata for all 1,200+ bills.

**What can go wrong:** The API isn't always available. Older sessions may not have full API coverage, or the server may be temporarily down. If the API call fails, the scraper falls back gracefully: it keeps the full list of 1,200 bill URLs and checks each one individually in Phase 3. Slower, but reliable.

**Codebase:** `src/tallgrass/scraper.py` (the `_filter_bills_with_votes()` method)

## Phase 3: Parse — Read Every Vote Page

This is the heart of the scraper. For each bill that passed the filter, the scraper visits the bill's page, finds links to all its vote pages, downloads each vote page, and extracts the data.

**How it works:**

**Step 1 — Find vote links.** For each of the ~600 filtered bills, the scraper visits the bill's page and looks for links to vote pages. A single bill can have many votes: a committee report, several amendments, a final passage vote, and sometimes an emergency final action or veto override. The scraper collects all of these.

**Step 2 — Download vote pages.** The scraper downloads all vote pages — typically 1,100 or more per session. This is where the concurrent-but-courteous model matters most. Instead of downloading pages one at a time (which would take over an hour), the scraper sends up to five requests simultaneously. But each request respects the rate limit, so the server never sees more than a gentle stream of traffic.

**Step 3 — Extract the data.** Each vote page is parsed to extract:

| Field | Example | Source |
|-------|---------|--------|
| Bill number | SB 55 | `<h2>` tag or URL |
| Bill title | AN ACT relating to taxation... | `<h4>` tag (not `<h2>` — a hard-won lesson) |
| Chamber | Senate | `<h3>` tag |
| Motion | Final Action | `<h3>` tag |
| Date and time | 2025-03-20 20:35:13 | URL timestamp |
| Result | Passed | Parsed from motion text |
| Each legislator's vote | Sen. Smith: Yea | Vote category sections |

The output of this step is two lists:

- **Roll calls** — one record per vote, summarizing the totals (66 Yea, 53 Nay, etc.) and the motion, result, bill number, and date.
- **Individual votes** — one record per legislator per vote. If 165 legislators voted on 1,105 roll calls, that's up to 182,325 individual records (though absences and chamber splits reduce the actual number to about 95,000).

**The concurrency pattern:** This is worth understanding because it's a recurring design decision in Tallgrass. The scraper uses a "fetch in parallel, parse sequentially" pattern:

```
FETCH (5 threads)              PARSE (1 thread)
┌──────────────┐
│ Thread 1: GET vote_page_1    │
│ Thread 2: GET vote_page_2    │ → All results stored in memory →  Parse page 1
│ Thread 3: GET vote_page_3    │                                    Parse page 2
│ Thread 4: GET vote_page_4    │                                    Parse page 3
│ Thread 5: GET vote_page_5    │                                    ... (sequential)
└──────────────┘
```

Why not parse in parallel too? Because the parsing step updates shared data structures — the list of legislators, the list of votes, the roll call summaries. Updating shared state from multiple threads simultaneously is a recipe for subtle, hard-to-reproduce bugs. The network is the bottleneck anyway (downloading pages is slow; parsing HTML is fast), so parallel fetching with sequential parsing gives us almost all the speed benefit with none of the complexity risk.

**What can go wrong:** HTML parsing is where most scraper bugs live. The Kansas Legislature's website has evolved over 15 years, and the HTML structure is not always what you'd expect. Here are a few of the lessons learned the hard way:

- **The title is in `<h4>`, not `<h2>`.** On vote pages, `<h2>` contains the bill number, and `<h4>` contains the title. This is unusual (most websites put the most important heading in `<h2>`), and early versions of the scraper got it wrong.

- **Vote categories appear in both `<h2>` and `<h3>` tags.** The parser can't just look for one tag type — it has to scan both.

- **Inline HTML tags eat spaces.** The text "Amendment by Senator Francisco was rejected" can be rendered as `Amendment by<a>Senator Francisco</a>was rejected` — with no spaces around the link. The parser uses a special text-cleaning function to prevent "bySenator" from becoming one word.

- **The server returns error pages with HTTP 200 status codes.** Most servers return a 404 status when a page doesn't exist. This server sometimes returns a 200 (success) status with an error page. The scraper has to detect these by inspecting the HTML content.

These aren't theoretical concerns — each one was a real bug that produced incorrect data before it was caught and fixed. The scraper's test suite includes over 640 tests specifically for parsing, many of them derived from actual pages that triggered these edge cases.

**Codebase:** `src/tallgrass/scraper.py` (the `parse_vote_pages()` method and all `_extract_*` static methods)

## Phase 4: Enrich — Add Legislator Metadata

At this point, the scraper knows how each legislator voted, but it doesn't know much *about* them. Vote pages list legislators by name and URL slug (a machine-friendly identifier like `sen_smith_john_1`), but they don't include party affiliation, district number, or full legal name.

**How it works:**

1. The scraper builds a list of every unique legislator slug encountered during parsing.
2. For each legislator, it visits their profile page on the legislature's website.
3. It extracts their full name, party, and district number from the page's HTML.

**Another parsing pitfall:** Party detection has two eras. Post-2015 pages display district and party in an `<h2>` tag like "District 32 - Republican." Pre-2015 pages use a separate `<h3>` tag like "Party: Republican." The scraper tries the modern format first and falls back to the legacy format if needed.

There's also a trap: every legislator page contains a "filter by party" dropdown menu that includes the word "Republican" regardless of which legislator you're looking at. If the parser searched the entire page for the word "Republican," every legislator would be classified as a Republican. The parser avoids this by looking only in the specific HTML tag that holds the party information.

**The OpenStates connection:** In addition to party and district, the scraper assigns each legislator an **OCD ID** (Open Civic Data Identifier) — a globally unique identifier maintained by the OpenStates project. OpenStates is an open-source initiative that tracks all 50 state legislatures, and its identifier system ensures that the same person gets the same ID across different sessions, name changes, and even different data sources.

The mapping from Kansas Legislature slugs to OCD IDs is pre-computed and stored locally, so this step requires no additional network access. It's just a dictionary lookup.

**What comes out:** A complete legislator roster with full name, party, district, member page URL, and OCD ID for each of the ~181 legislators active in the session.

**Codebase:** `src/tallgrass/scraper.py` (the `enrich_legislators()` method), `src/tallgrass/roster.py` (OCD ID mapping)

## The Retry Wave System

Network requests fail. Servers go down, connections time out, packet loss happens. A scraper that gives up on the first failure will miss data. A scraper that retries too aggressively will make things worse — hammering a struggling server is like honking at a traffic jam.

Tallgrass uses a three-tier error recovery system:

### Tier 1: Per-Request Retries

Each individual request gets up to three attempts. The behavior depends on the type of error:

| Error | Classification | Retries | Wait Strategy |
|-------|---------------|---------|--------------|
| Page not found (404) | Permanent | 1 retry | None — the page genuinely doesn't exist |
| Server error (500-599) | Transient | 3 retries | Exponential backoff: 5s, 10s, 20s + random jitter |
| Timeout | Transient | 3 retries | Exponential backoff + jitter |
| Connection refused | Transient | 3 retries | Fixed 5s delay |
| Other client errors (400-499) | Permanent | No retry | The request itself is wrong |

**Exponential backoff** means each retry waits twice as long as the previous one. The idea is simple: if the server is struggling, give it progressively more time to recover. The random jitter (a small random addition to each wait time) prevents multiple threads from retrying at exactly the same moment, which would create a traffic spike.

### Tier 2: Retry Waves

After the scraper has attempted every request with per-request retries, it looks at what failed. If any failures were *transient* (server errors, timeouts, connection issues), the scraper launches a **retry wave**:

1. Wait 90 seconds (the wave cooldown) — give the server time to fully recover
2. Reduce concurrency from 5 threads to 2
3. Slow the request rate from 0.15s to 0.5s between requests
4. Re-attempt all transient failures

If failures persist, up to two more waves are attempted, each with the same cooldown and gentleness. The logic is: if the server was overwhelmed, send less traffic. If it was temporarily down, wait and try again.

### Tier 3: Failure Manifest

After all retry waves, any requests that still failed are recorded in a **failure manifest** — a JSON file listing each failed URL, the bill it belonged to, the type of error, and how many attempts were made. This is the scraper's honest accounting of what it couldn't get.

In practice, the failure manifest is almost always empty. The Kansas Legislature's server is reliable, and the retry system handles the occasional hiccup. But when it isn't empty, the manifest makes it easy to diagnose what went wrong and decide whether a re-run is needed.

**Codebase:** `src/tallgrass/config.py` (retry constants), `src/tallgrass/scraper.py` (the `_get()` and `_fetch_many()` methods)

## The Output: Five CSV Files

When the pipeline finishes, it writes five CSV files to the data directory:

| File | Contents | Rows (91st) | Key Fields |
|------|----------|-------------|------------|
| `*_votes.csv` | One row per legislator per vote | 95,199 | legislator_slug, vote_id, vote, bill_number |
| `*_rollcalls.csv` | One row per roll call | 1,105 | vote_id, bill_number, motion, result, yea/nay counts |
| `*_legislators.csv` | One row per legislator | 181 | legislator_slug, full_name, party, district, ocd_id |
| `*_bill_actions.csv` | One row per bill action | 10,442 | bill_number, action_code, status, date |
| `*_bill_texts.csv` | Full text of each bill | (Chapter 5) | bill_number, document_type, text |

The first three files are the core vote data that feeds the entire 28-phase analysis pipeline. The last two are supplemental — bill actions support lifecycle analysis, and bill text supports natural language processing.

### Deduplication

The scraper includes safety nets against duplicate records. In the 2011–2014 sessions, the same vote page can be linked from multiple bills (because a single committee-of-the-whole vote might affect several bills). Without deduplication, the same vote would appear multiple times in the output.

The solution is simple: each vote record is identified by the combination of `(legislator_slug, vote_id)`. If the scraper encounters the same combination twice, it keeps the first occurrence and discards the duplicate. Roll calls are similarly deduplicated by `vote_id`.

### The Data Models

Behind the CSV files are three frozen dataclasses — Python objects that hold structured data and can't be accidentally modified after creation:

**IndividualVote** — one legislator's vote on one roll call:
- Session, bill number, bill title, vote ID, date, chamber, motion
- Legislator name, legislator slug
- Vote: one of "Yea", "Nay", "Present and Passing", "Not Voting", or "Absent and Not Voting"

**RollCall** — summary of one vote:
- Session, bill number, bill title, vote ID, vote URL, date, chamber, motion
- Vote type, result, short title, sponsor
- Yea count, nay count, present/passing count, absent count, not voting count
- Total votes, whether the measure passed

**BillAction** — one step in a bill's legislative history:
- Session, bill number, action code, chamber
- Committee names, date, status, journal page number

These dataclasses are the contract between the scraper and the rest of the pipeline. Downstream code never touches HTML or network requests — it works entirely with these structured objects (via their CSV representation).

**Codebase:** `src/tallgrass/models.py` (data models), `src/tallgrass/output.py` (CSV export)

## Running the Scraper

For the hands-on reader, here's how to run the full pipeline:

```bash
# Scrape the current session (2025-2026)
just scrape 2025

# Scrape and load into PostgreSQL
just scrape 2025 --auto-load

# Scrape with a fresh cache (re-downloads everything)
just scrape-fresh 2025

# Scrape the 2024 special session
just scrape 2024 --special

# See all available sessions
just scrape --list-sessions
```

The `just` command is a thin wrapper around `uv run tallgrass <year>`. The Justfile also sets environment variables to cap thread pools on Apple Silicon Macs, preventing the operating system from spawning too many threads and creating contention.

A typical run looks like this:

```
$ just scrape 2025

Tallgrass vote scraper for KS Legislature
Session: 91st (2025-2026)

Phase 1: Discovering bills...
  Found 1,247 bill URLs

Phase 2: Filtering bills with votes (KLISS API)...
  631 bills have recorded votes (616 filtered out)
  Collected 10,442 bill actions

Phase 3: Finding vote links...
  Found 1,105 vote links across 631 bills

Phase 4: Parsing vote pages...
  ████████████████████████ 1,105/1,105 [00:12]
  Parsed 95,199 individual votes in 1,105 roll calls

Enriching legislator data...
  ████████████████████████ 181/181 [00:03]

Saved:
  data/kansas/91st_2025-2026/91st_2025-2026_votes.csv (95,199 rows)
  data/kansas/91st_2025-2026/91st_2025-2026_rollcalls.csv (1,105 rows)
  data/kansas/91st_2025-2026/91st_2025-2026_legislators.csv (181 rows)
  data/kansas/91st_2025-2026/91st_2025-2026_bill_actions.csv (10,442 rows)

Done in 14m 32s
```

---

## Key Takeaway

The Tallgrass scraper is a four-phase pipeline: discover all bills, filter to those with votes, parse every vote page, and enrich with legislator metadata. It uses concurrent fetching with sequential parsing to balance speed and correctness, a three-tier retry system to handle network failures gracefully, and produces five CSV files that form the foundation for everything that follows.

---

*Terms introduced: four-phase pipeline, KLISS API, concurrent fetch / sequential parse, exponential backoff, jitter, retry wave, failure manifest, frozen dataclass, deduplication*

*Next: [Historical Sessions and the ODT Challenge](ch03-historical-sessions-odt.md)*
