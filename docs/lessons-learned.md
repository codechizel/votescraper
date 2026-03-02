# Lessons Learned

Hard-won debugging insights from building this scraper. These are real bugs that cost time to diagnose — don't repeat them.

---

## Bug 1: The h2/h3/h4 Tag Mismatch

**Symptom**: `vote_date`, `chamber`, `motion`, and `bill_title` were ALL empty strings in every row of every output CSV. Vote counts and legislator data worked fine.

**Root cause**: The parser searched `<h2>` tags for vote metadata, but the actual HTML uses a non-obvious tag hierarchy:

```
<h2>  → bill number (e.g., "SB 1")
<h4>  → bill title (e.g., "AN ACT exempting the state of Kansas...")
<h3>  → chamber/motion/date (e.g., "Senate - Emergency Final Action - Passed as amended; - 03/20/2025")
<h3>  → vote category headings (e.g., "Yea - (33):")
```

You'd expect the title to be in `<h2>` or `<h3>`, not `<h4>`. And you'd expect chamber/date/motion to be in a higher-level heading than the vote categories, but they're both `<h3>`.

**Fix**: Search `<h4>` for bill title, `<h3>` for chamber/date/motion.

**Lesson**: Always inspect the actual HTML. Don't assume heading hierarchy follows semantic conventions. The KS Legislature website was likely built with a CMS that doesn't enforce consistent heading levels.

---

## Bug 2: Every Legislator Tagged as Republican

**Symptom**: All 172 legislators in the output had `party = "Republican"`, including known Democrats.

**Root cause**: The code did `re.search(r"\bRepublican\b", page.get_text())` on each legislator's member page. But every member page contains a party filter dropdown:

```html
<select>
  <option value="republican">Republican</option>
  <option value="democrat">Democrat</option>
</select>
```

Since `page.get_text()` flattens all text including form elements, "Republican" appears on every page before the actual party info. The regex always matched Republican first.

**Fix**: Instead of searching full page text, target the specific `<h2>` that contains the district info: `<h2>District 27 - Republican</h2>`. Use `soup.find("h2", string=re.compile(r"District\s+\d+"))`.

**Lesson**: Never search full page text for structured data. Form elements, navigation, filters, and footers all contain text that can create false positives. Always target the most specific containing element.

---

## Bug 3: KLISS API Response Format Inconsistency

**Symptom**: API pre-filtering occasionally returned zero bills with votes, falling back to the full scan (slower but functional).

**Root cause**: The API sometimes returns a raw JSON array `[{...}, {...}]` and sometimes wraps it in `{"content": [{...}, {...}]}`. Code that only handles one format silently gets an empty list.

**Fix**: `content = data if isinstance(data, list) else data.get("content", [])`.

**Lesson**: Government APIs are not versioned or documented consistently. Always handle multiple response shapes and fail gracefully.

---

## Design Decision: Why bill_metadata Exists

The KLISS API is already called for pre-filtering (to avoid fetching ~800 bill pages that have no votes). This same API response contains `SHORTTITLE` and `ORIGINAL_SPONSOR` — valuable enrichment data that would otherwise require parsing unreliable HTML.

Rather than discard this data and fetch it again later, `_filter_bills_with_votes()` returns both the filtered URLs and a metadata dict. This adds zero HTTP requests and provides cleaner data than HTML parsing would.

---

## Design Decision: Two-Phase Fetch/Parse

The scraper never mutates shared state during concurrent fetches. The pattern is always:

1. **Fetch phase** (concurrent): `ThreadPoolExecutor` fetches URLs → returns `dict[url, html]`
2. **Parse phase** (sequential): Single-threaded loop processes HTML, mutates `self.rollcalls`, `self.individual_votes`, `self.legislators`

This avoids race conditions without locks on the data structures. The rate-limiting lock only protects the HTTP request timing, not the parsed data.

---

## Design Decision: vote_datetime from vote_id

The vote_id string (e.g., `je_20250320203513_2584`) encodes a precise timestamp. Extracting this gives sub-second vote ordering and a robust deduplication key (`vote_datetime` + `legislator_slug`), which is more reliable than the human-readable `vote_date` (MM/DD/YYYY, no time component).

---

## Gotcha: Cache Key Collisions

Cache filenames are constructed by replacing `/`, `:`, `?` in URLs with `_`, then truncating to 200 chars. For very long URLs, truncation could theoretically cause collisions. In practice this hasn't been an issue because KS Legislature URLs are well under 200 chars, but it's worth knowing about.

---

## Gotcha: session.py CURRENT_BIENNIUM_START

This constant must be manually updated when the KS Legislature starts a new biennium (odd years, e.g., 2027). If it's wrong, the "current" session will use historical URL prefixes, which may or may not work depending on when the legislature updates their site.

---

## Gotcha: Special Session URL Patterns

Special sessions use a completely different URL scheme (`/li_2024s/` instead of `/li_2024/b2023_24/`). They also don't have a biennium code. New special sessions must be manually added to `SPECIAL_SESSION_YEARS` in session.py.

---

## Insight 1: Legislator Count ≠ Seat Count (Mid-Session Replacements)

**Discovered during**: EDA Phase 1 (2026-02)

**Symptom**: The 2025-26 session data contains 130 House legislators and 42 Senate legislators, but the Kansas House has exactly 125 seats and the Senate has exactly 40 seats.

**Root cause**: Mid-session resignations and replacements. Legislators who resign or are appointed to another office are replaced by new members. Both the outgoing and incoming members appear in the data with non-overlapping vote records. In the 2025-26 session:

- **5 House replacements** (Districts 5, 33, 70, 85, 86): Original members served 2025-01-30 → 2025-04-11 (~333 votes), replacements started 2026-01-15 (~151 votes)
- **2 Senate replacements** (Districts 24, 25): Original members served through 2025-04-11, replacements started 2026-01-27

Two legislators (Scott Hill and Silas Miller) appear in both chambers — they left the House and were elected/appointed to the Senate.

**Validation**: Service windows were confirmed non-overlapping. No duplicate votes (same legislator + same rollcall) were found. The data is correct.

**Lesson**: Always validate legislator counts against known chamber sizes. The Kansas House has **125** seats and the Senate has **40** seats — these are constitutional constants. Any count above those numbers signals mid-session turnover and requires verifying that service windows don't overlap (which would indicate a scraping bug). Downstream analysis must account for partial-session legislators: their participation rates will be low by design, and ideal point estimates will be less stable.

---

## Bug 4: "Not passed" Classified as Passed

**Discovered during**: EDA report review (2026-02-19)

**Symptom**: The EDA report showed 100% passage rate for Final Action votes. Manual inspection of the motion text revealed 4 rollcalls with "Not passed" or "Not  passed" (double space) in the motion, all incorrectly marked `passed=True`.

**Root cause**: `_derive_passed()` checked the positive regex `\b(passed)\b` before the negative patterns. Since "Not passed" contains the word "passed", the positive match fired first and returned `True`.

**Fix**: Reorder the checks — test failure patterns (`\b(not\s+passed|failed|rejected)\b`) first, then positive patterns.

**Impact**: 4 rollcalls (1 Final Action, 3 Emergency Final Action) had `passed=True` when they should have been `False`. This corrupted passage rate statistics. All individual vote records for these rollcalls were unaffected (votes were correctly counted), only the rollcall-level `passed` boolean was wrong.

**Lesson**: When regex-matching result text, always check negations first. String containment is not semantic — "not passed" *contains* "passed". This is a textbook order-of-evaluation bug.

---

## Bug 5: get_text(strip=True) Drops Spaces Around Inline Tags

**Discovered during**: Scraper code review (2026-02-19)

**Symptom**: 53 motions in the rollcalls CSV had mangled text like `"Amendment bySenator Franciscowas rejected"` instead of `"Amendment by Senator Francisco was rejected"`.

**Root cause**: BeautifulSoup's `get_text(strip=True)` strips whitespace from each text node individually, then concatenates them **without any separator**. When an `<h3>` tag contains inline `<a>` elements (as Committee of the Whole amendment motions do), the spaces between text nodes and the `<a>` tag are lost:

```html
<h3>... Amendment by <a href="..."> Senator Francisco</a> was rejected ...</h3>
```

Text nodes: `"Amendment by "`, `" Senator Francisco"`, `" was rejected"` → after strip each → `"Amendment by"`, `"Senator Francisco"`, `"was rejected"` → concatenated → `"Amendment bySenator Franciscowas rejected"`.

**Fix**: Use `get_text(separator=" ", strip=True)` which inserts a space between text nodes, then collapse multiple spaces with `" ".join(text.split())`. Extracted as a `_clean_text()` helper.

**Impact**: All 53 affected motions were Committee of the Whole amendment votes. The mangling was cosmetic — vote_type and passed classification still worked because the prefix "Committee of the Whole" and keyword "rejected" were in the non-mangled parts. But the data was ugly and would cause issues for text-based searches.

**Lesson**: `get_text(strip=True)` is almost never what you want for elements with inline children. Always use `separator=" "` when the element might contain `<a>`, `<span>`, `<em>`, or other inline tags.

---

## Gotcha: Legislative Day vs Wall Clock Time

**Discovered during**: Scraper code review (2026-02-19)

**Symptom**: 3 rollcalls have `vote_datetime` (from the vote_id timestamp) on March 18 but `vote_date` (from the h3 on the page) on March 17.

**Root cause**: These are late-night House votes where the legislative session extended past midnight. The legislature considers it still the "March 17 session" even though the clock says March 18. The h3 date reflects the legislative day; the vote_id timestamp reflects the wall clock.

**Not a bug** — both values are correct for their respective meanings. `vote_date` is the authoritative legislative day. `vote_datetime` is the precise wall clock time for ordering.

**Lesson**: Legislative bodies operate on "legislative days" that don't always align with calendar days. When votes extend past midnight, the session date stays the same. Our data correctly preserves both values.

---

## Gotcha: Null bill_title on Amendment Vote Pages

**Discovered during**: Scraper code review (2026-02-19)

15 rollcalls (all Senate Committee of the Whole amendment votes) have null `bill_title` because their vote pages have no `<h4>` tag and the `<h2>` contains only `[`. These pages simply don't include the bill title — it's a quirk of how the KS Legislature website renders amendment-specific vote pages.

`bill_number` is always present and can be used to look up the title from other rollcalls for the same bill. This is a cosmetic gap, not a data integrity issue.

---

## Lesson 6: Positive-Constrained β Prior Silences Half the Ideological Signal

**Discovered during:** IRT audit (2026-02-20)

**Symptom:** The IRT model's beta (discrimination) distribution was bimodal: a cluster at β ≈ 7 (party-line R-Yea bills) and a cluster at β ≈ 0.17 (all other contested bills). All 37 bills where Democrats voted Yea had β < 0.39 — the model treated them as uninformative noise.

**Root cause:** The LogNormal(0.5, 0.5) prior constrains β > 0. With `P(Yea) = logit⁻¹(β·ξ - α)` and β > 0, the probability of voting Yea always increases with ξ (conservatism). Bills where Democrats vote Yea need β < 0 to be modeled correctly. The positive constraint makes this impossible, so the model assigns near-zero discrimination.

The design doc incorrectly claimed "alpha handles directionality." Alpha shifts the probability curve up/down (threshold) but cannot flip its slope (direction). Only the sign of β controls direction.

**Why the standard recommendation was wrong for us:** The LogNormal prior is a standard recommendation for IRT models using *soft identification* (priors only). Our model uses *hard anchors* (two legislators fixed at ξ = ±1), which already solve the sign-switching problem. The positive constraint was redundant and actively harmful — it solved a problem the anchors had already solved, at the cost of discarding 12.5% of bills.

**Fix:** Switch to `pm.Normal("beta", mu=0, sigma=1)`. Experiment results (500/300 draws):

| Metric | LogNormal(0.5,0.5) | Normal(0,1) |
|---|---|---|
| Holdout accuracy | 90.8% | **94.3%** (+3.5%) |
| Holdout AUC-ROC | 0.954 | **0.979** |
| D-Yea \|β\| mean | 0.186 | **2.384** |
| ξ ESS min | 21 | **203** (10× better) |
| Divergences | 0 | 0 |
| PCA Pearson r | 0.950 | **0.972** |

Zero sign-switching. Zero divergences. Better on every metric.

**Lesson:** When using hard anchors in Bayesian IRT, an unconstrained Normal prior on discrimination is both simpler and better than a positive-constrained LogNormal. The anchors provide identification; the prior should provide regularization, not constraints. Don't blindly follow "standard" priors without checking whether their assumptions (soft identification) match your model (hard identification).

**See also:** `analysis/design/beta_prior_investigation.md` for the full investigation, experiment protocol, and plots.

---

## Design Decision: Failure Manifest for Vote Page Fetches

**Added during**: Scraper failure handling improvement (2026-02-20)

**Problem**: When scraping the 2023-24 session, 5 vote pages returned persistent 404s. The scraper silently skipped them — no bill context logged, no failure summary, no way to know which bills were affected without manually checking URLs.

**Solution**: `_get()` now returns a `FetchResult` dataclass instead of `str | None`, carrying error classification (permanent/transient/timeout/connection), status code, and error message. Failed vote page fetches are recorded as `FetchFailure` objects with full bill context (bill number, motion text, bill path).

Key design choices:
- **No `--retry-failed` flag**: Failed fetches are never cached, so re-running the scraper automatically retries them.
- **No logging framework**: Stays with `print()` per project convention.
- **FetchResult/FetchFailure live in scraper.py**, not models.py — they're internal to the scraper, not data export models.
- **Differentiated retries**: 404s get one retry (for transient routing guards), 5xx gets exponential backoff, other 4xx don't retry at all.
- **Failure manifest**: JSON file alongside CSVs, matching the `filtering_manifest.json` pattern from analysis scripts.

**2023-24 results**: 4 permanent 404s (SB 17, SB 379, HB 2273, HB 2390) and 1 transient 502 (HB 2313). The 404s are likely dead links on the legislature's website — the vote pages existed when the links were created but were later removed or relocated.

---

## Design Decision: Retry Waves for Server Resilience

**Added during**: Scraper resilience improvement (2026-02-21)

**Problem**: A fresh scrape of 2025-26 failed on 55 of 882 vote pages — all transient 5xx errors clustered in a 3-second window. The per-URL retry logic (3 attempts, exponential backoff) works for isolated hiccups but can't handle a server that needs minutes to recover. All 5 workers fail simultaneously, retry simultaneously (thundering herd), exhaust their 3 attempts within ~35 seconds, and give up forever.

**Solution**: `_fetch_many()` now runs up to 3 retry waves after the initial fetch pass. Each wave:
1. Collects transient failures (error_type in `transient`, `timeout`, `connection`)
2. Waits 90 seconds (`WAVE_COOLDOWN`) for the server to recover
3. Re-dispatches only failed URLs with reduced load: 2 workers (`WAVE_WORKERS`) at 0.5s rate (`WAVE_DELAY`)
4. Merges results — successes overwrite old failures
5. Exits early if no transient failures remain

Additionally, per-URL backoff in `_get()` now includes jitter (`* (1 + random.uniform(0, 0.5))`) for 5xx and timeout errors, spreading retry timing across workers to prevent thundering herd within a wave.

Key design choices:
- **Waves live inside `_fetch_many()`**: Callers don't change. The pipeline orchestration in `run()` is unaffected.
- **`self.delay` is temporarily increased during waves**: Restored in a `finally` block. Safe because waves run sequentially and `_fetch_many()` is the only concurrent caller of `_get()`.
- **Per-URL `MAX_RETRIES` unchanged at 3**: The waves provide longer-term resilience; per-URL retries handle within-wave hiccups.
- **Constants in `config.py`**: `RETRY_WAVES=3`, `WAVE_COOLDOWN=90`, `WAVE_WORKERS=2`, `WAVE_DELAY=0.5`.

**See also**: ADR-0009 for the full decision record.

---

## Bug 8: `.split()[-1]` Name Extraction Fails on Leadership Suffixes

**Discovered during:** Clustering visualization improvement (2026-02-22)

**Symptom:** The voting blocs and polar dendrogram plots showed "Senate" as a senator's name instead of "Shallenburger". Two senators both appeared as just "Claeys" with no disambiguation.

**Root cause:** Throughout the analysis codebase, legislator display names were extracted from `full_name` using `full_name.split()[-1]`. This pattern fails on legislators with leadership suffixes stored in the scraper data:

| full_name | `.split()[-1]` | Correct |
|-----------|----------------|---------|
| Tim Shallenburger - Vice President of the Senate | **Senate** | Shallenburger |
| Ty Masterson - President of the Senate | **Senate** | Masterson |
| Daniel Hawkins - Speaker of the House | **House** | Hawkins |
| Blake Carpenter - Speaker Pro Tem | **Tem** | Carpenter |

Additionally, duplicate last names (Joseph Claeys / J.R. Claeys in the Senate; two each of Carpenter, Williams, Smith, Ruiz in the House) were not disambiguated.

**Fix:** Added `_build_display_labels()` helper in `clustering.py` that:
1. Strips leadership suffixes by splitting on `" - "` and taking the first part
2. Extracts last name from the cleaned name
3. Detects duplicate last names via `Counter`
4. Disambiguates with abbreviated first-name prefix (e.g., "Jo. Claeys" vs "J.R. Claeys")

**Impact:** 30+ occurrences of `.split()[-1]` exist across the analysis codebase (prediction, profiles, synthesis, etc.). The clustering fix is localized; other modules still use the raw pattern but are protected by joining on `legislator_slug` (not display names). A shared utility would prevent this bug class entirely.

**Lesson:** Never assume `full_name.split()[-1]` produces a last name. Leadership suffixes, hyphenated names, and name suffixes (Jr., III) all break this pattern. Always strip known suffix patterns before extracting names, and always check for duplicate last names when building display labels.

---

## Bug 7: Empty DataFrame Indexing in Party Loyalty Summary

**Discovered during:** Clustering code review and test writing (2026-02-20)

**Symptom:** `compute_party_loyalty()` in `analysis/clustering.py` crashes with `IndexError: index 0 is out of bounds for sequence of length 0` when there are no contested votes.

**Root cause:** After computing party loyalty, the function prints the lowest and highest loyalty legislators by indexing into a sorted DataFrame: `sorted_loyalty['full_name'][0]`. When all votes are unanimous (no contested votes), the loyalty DataFrame is empty (0 rows), and indexing position 0 raises `IndexError`.

**Why this matters:** The bug never triggered in production because real Kansas Legislature data always has contested votes. But it's a latent crash path that would surface with synthetic data, subset analyses, or future sessions with different voting patterns.

**Fix:** Guard the summary print statements with `if loyalty.height > 0`.

**Lesson:** Any function that indexes into a DataFrame or Series must either (a) guarantee the collection is non-empty via an upstream check, or (b) guard the indexing operation. Print/logging statements are particularly easy to forget — they feel "safe" because they don't affect the return value, but they can still crash the function.

---

## Gotcha: Pre-2021 Sessions Load Bills via JavaScript

**Discovered during:** Historical session support (2026-02-22)

**Symptom:** `get_bill_urls()` returns 0 bills for sessions before 2021 (e.g., 2019-20, 2017-18). The HTML listing pages exist but contain zero `<a>` tags matching the bill URL pattern.

**Root cause:** The KS Legislature website transitioned from server-side rendered bill lists to JavaScript-loaded bill lists sometime between 2020 and 2021. Pre-2021 listing pages load their bill data client-side from JavaScript data files at paths like `/li_2020/s/js/data/bills_li_2020.js`. These files assign a `measures_data` variable containing a JSON array of bill objects with `measures_url` fields.

**Fix:** Added JS data file fallback in `get_bill_urls()`. When HTML scanning finds zero bills and `KSSession.js_data_paths` is non-empty, the scraper fetches the JS data file, extracts the JSON array between `[` and `]`, and parses `measures_url` values.

**Lesson:** Government websites silently change rendering strategies across sessions. Always verify that the scraper works on historical sessions, not just the current one. The absence of data (0 results) is harder to notice than broken data.

---

## Gotcha: Pre-2015 Sessions Use ODT Vote Files

**Discovered during:** Historical session support (2026-02-22)

**Symptom:** Vote pages for 2011-2014 sessions return binary data (ZIP archives) instead of HTML, causing parse failures.

**Root cause:** Sessions before 2015 link to `.odt` (OpenDocument Text) files via `odt_view` URLs instead of HTML `vote_view` pages. The ODT files are ZIP archives containing `content.xml` with structured metadata in `<text:user-field-decl>` elements and vote data in paragraph text.

**Key format differences from HTML:**
- Legislator names are last-name-only (e.g., "Smith" not "Smith, John"), requiring a member directory for slug resolution
- House and Senate use different vote category names ("Present but not voting" vs "Present and Passing")
- Ambiguous names (same last name, same chamber) cannot be resolved and get empty slugs
- The XML namespace handling requires `xml.etree.ElementTree` with explicit namespace URIs

**Fix:** Added `odt_parser.py` module (pure functions, no I/O) and `_parse_odt_vote_pages()` integration in the scraper. Binary fetch mode added to `FetchResult` and `_get()`.

**Lesson:** Format transitions in government data are typically undocumented. When extending a scraper backwards in time, expect that every 2-4 years will bring a different page structure, data format, or rendering strategy. Build format-specific parsers behind a common interface rather than trying to make one parser handle everything.

---

## Gotcha: Pre-2015 Party Detection Uses Different HTML

**Discovered during:** Historical session support (2026-02-22)

**Symptom:** All legislators from pre-2015 sessions have empty `party` fields even though their member pages exist and contain party information.

**Root cause:** Pre-2015 legislator pages display party as `<h3>Party: Republican</h3>` instead of encoding it in the `<h2>District N - Republican</h2>` format used by 2015+ pages. The existing parser only checks `<h2>`.

**Fix:** Added fallback in `_extract_party_and_district()` (static method called by `enrich_legislators()`): when `<h2>` yields no party, scan `<h3>` tags for `"Party: Republican"` or `"Party: Democrat"`.

**Lesson:** When a parser works for recent data but fails on historical data, check whether the HTML structure changed between sessions. Government website redesigns happen regularly and usually aren't documented.
