# Scraper Deep Dive

Code audit, ecosystem comparison, data quality review, and recommendations for the Kansas Legislature vote scraper.

## Scope

This review covers the full scraper codebase (2,332 lines across 7 files), all 8 bienniums of scraped data (84th-91st, 2011-2026), the test suite (202 scraper-specific tests), and a survey of comparable open-source legislative scraping projects.

## Ecosystem Survey

### OpenStates Kansas Scraper

[OpenStates](https://github.com/openstates/openstates-scrapers) is the gold standard for US state legislature scraping, covering all 50 states. Their Kansas scraper (`scrapers/ks/votes.py`) is 152 lines and reveals how differently the same problem can be approached:

| Dimension | OpenStates Kansas | Tallgrass |
|---|---|---|
| Session coverage | ~2015-present (HTML only) | 2011-2026 (HTML + ODT + JS) |
| Error handling | String matching in response body | Typed `FetchResult` with error classification |
| Retry strategy | None (relies on scrapelib defaults) | Per-URL exponential backoff + retry waves |
| Caching | None visible | `.cache/` directory with raw HTML/binary |
| Failure tracking | Logs to stderr | `failure_manifest.json` + `missing_votes.md` |
| Vote categories | Positional inference (yeas first, then nays) | Explicit 5-category canonical mapping |
| ODT parsing | Not supported | Full stdlib implementation |
| JS page handling | Not supported | Direct `.js` data file fetching |
| Deduplication | `dedupe_key = link` | `(legislator_slug, vote_id)` tuple |

The OpenStates scraper has a notable fragility: it counts `<a>` links positionally, assuming yeas come first, then nays. Our scraper reads the explicit category headers (`<h2>`/`<h3>`) and groups members under them, which is structurally correct regardless of page layout.

OpenStates uses `lxml.html` with XPath selectors; we use BeautifulSoup with `lxml` backend. The KS Legislature's malformed HTML (documented in CLAUDE.md pitfalls #1-#5) makes BS4's lenient parsing the right choice. The performance difference is irrelevant when rate-limiting is the bottleneck.

### Other Projects

- **[unitedstates/congress](https://github.com/unitedstates/congress)** — US Congress scraper. Cache-first design, hierarchical output structure, `--force`/`--fast` flags. Our cache approach is similar.
- **[spatula](https://github.com/jamesturk/spatula)** — Page-oriented scraping framework by the OpenStates creator. Enforces clean separation via Page classes. Our 4-step pipeline achieves similar separation without a framework dependency.
- **[freelawproject/juriscraper](https://github.com/freelawproject/juriscraper)** — Court opinion scraper. Two-part design (library + caller). Test fixtures (`*_example*` files) run scrapers against known HTML.
- **[LegiScan](https://legiscan.com/legiscan)** — Structured API covering all 50 states. Not a scraper but a key alternative for researchers who don't need ODT-era data.

No Kansas-specific open-source scrapers exist beyond OpenStates. The KS Legislature's use of ODT files, JavaScript-rendered pages, and non-standard HTML patterns makes it a uniquely difficult target.

### Technology Choices

**HTTP library**: `requests` with `ThreadPoolExecutor` is the standard choice for moderate concurrency (5-20 connections). `httpx` with `AsyncClient` would be the modern alternative for high concurrency, but is overkill when we deliberately limit to 5 workers for politeness. No change recommended.

**HTML parser**: BeautifulSoup with `lxml` backend is correct for this site. lxml alone is ~15x faster but less tolerant of malformed HTML. The performance difference is invisible at our request rate (0.15s delay dwarfs any parsing difference).

**ODT parser**: Our stdlib approach (`zipfile` + `xml.etree.ElementTree`) is the recommended pattern for read-only extraction. odfpy and odfdo are designed for round-trip editing and would add a dependency without value.

### Normalization Philosophy

The scraping community has converged on a hybrid approach:
1. **Scraper layer**: extract data faithfully, minimal normalization (whitespace, encoding)
2. **Validation boundary**: frozen dataclasses enforce structure
3. **Normalization layer**: separate code maps source-specific values to canonical forms

Our project follows this pattern correctly:
- `_clean_text()` handles mechanical whitespace cleanup
- `_normalize_bill_code()` standardizes identifiers
- `VOTE_CATEGORIES` and `_ODT_CATEGORY_MAP` define canonical mappings
- Frozen dataclasses enforce structural integrity
- Party normalization (`""` -> `"Independent"`) is deferred to the analysis layer (ADR-0021)

The quote from the profiles design doc — "Normalize upstream in the scraper. This was rejected because it would change the scraper's output format" — referred specifically to lowercasing the `chamber` column. The scraper outputs `"House"`/`"Senate"` (matching the source), and each analysis phase normalizes as needed. This is the correct separation. The scraper is not "sub-optimal" here; it's following the hybrid pattern that the ecosystem has converged on.

## Code Audit

### Architecture Assessment

The 4-step pipeline is clean and well-separated:

```
get_bill_urls()          → list[str]           # Step 1: discover bills
_filter_bills_with_votes() → (filtered, metadata)  # Step 1b: KLISS API pre-filter
get_vote_links()         → list[VoteLink]      # Step 2: scan for vote links
parse_vote_pages()       → mutates self.*      # Step 3: parse HTML/ODT
enrich_legislators()     → mutates self.*      # Step 4: fetch party/district
save_csvs()              → writes 3 CSV files  # Output
```

The two-phase concurrency pattern (concurrent fetch via `ThreadPoolExecutor`, sequential parse) is sound. All shared state mutation happens in the sequential phase. Rate limiting is thread-safe via `threading.Lock()`. No data races observed.

### Issues Found

**Issue 1: `assert` in production code** (`scraper.py:496`)

```python
assert result.html is not None  # guaranteed by result.ok above
```

This is a development guard that would raise `AssertionError` in production if `FetchResult.ok` were ever modified to allow `None` html. Should be a defensive check instead:

```python
if result.html is None:
    return bill_urls, {}
```

Severity: Low (the invariant currently holds). Risk: maintenance fragility.

**Issue 2: Magic numbers not in config** (`scraper.py:168, 1055, 1082; odt_parser.py:121`)

- `500` — bill title truncation length

Extracted as `BILL_TITLE_MAX_LENGTH` in `config.py`. Cache filenames now use SHA-256 hashing (ADR-0080), eliminating the truncation approach entirely.

**Issue 3: Silent bill title truncation**

`bill_title[:500]` truncates without logging. If a title exceeds 500 characters, the CSV will have incomplete data with no indication. Should log a warning when truncation occurs.

**Issue 4: Cache filename collision risk** (`scraper.py:168`) — **RESOLVED (ADR-0080)**

Cache filenames now use SHA-256 hashing (`hashlib.sha256(url.encode()).hexdigest()[:16]`), eliminating the truncation-based collision risk entirely.

**Issue 5: `self.delay` mutation during retry waves** (`scraper.py:331`)

```python
self.delay = WAVE_DELAY
try:
    ...
finally:
    self.delay = self._normal_delay
```

`self.delay` is read by `_rate_limit()` in concurrent threads. The wave loop runs after the initial `ThreadPoolExecutor` completes (all futures resolved), so there's no actual race. But the pattern is fragile — if someone added concurrent waves, it would break. Worth a comment documenting why it's safe.

**Issue 6: Empty slugs in ODT-era data** (84th: 900 rows, 85th: 1,140 rows)

The 84th and 85th bienniums have votes with empty `legislator_slug` fields. Investigation shows these are exactly 1 per `vote_id` — always a single legislator per rollcall who couldn't be resolved via the member directory. The top names (`Owens`, `Gordon`, `Hildabrand`, `C. Holmes`, `A. Schmidt`) suggest these are legislators whose last names are ambiguous in the member directory or who appear with initials that don't match any directory entry.

This is a data limitation, not a bug. The ODT parser correctly returns an empty slug when resolution fails. The analysis pipeline handles this by excluding rows with empty slugs from slug-keyed operations.

### Dead Code

None found. All functions and branches are used or are documented fallbacks (e.g., JS discovery is only used when HTML listing pages yield zero bills for pre-2021 sessions).

### RunContext Fixes (ADR-0037)

The pipeline review found two issues in `run_context.py` (shared infrastructure, not scraper-specific):

1. **`_append_missing_votes` format crash:** `f"{total:,}"` crashed with `ValueError` when `total_vote_pages` was missing from the failure manifest (defaults to `"?"` string). Fixed with `isinstance(total, int)` guard.
2. **`latest` symlink on failed runs:** `finalize()` updated the `latest` symlink unconditionally, so downstream phases could follow it to partial results. Fixed by skipping symlink update when the run failed.

### Refactoring Opportunities

**Completed (M2).** Four static methods were extracted from the two largest scraper functions without behavior changes:

- `_extract_bill_title(soup)` — 3-tier `<h4>` fallback (from `_parse_vote_page()`)
- `_extract_chamber_motion_date(soup)` — 2-tier `<h3>` fallback (from `_parse_vote_page()`)
- `_parse_vote_categories(soup)` — returns `(categories, new_legislators)` instead of mutating `self.legislators` (from `_parse_vote_page()`)
- `_extract_party_and_district(soup)` — post-2015/pre-2015 party detection (from `enrich_legislators()`)

All 10 HTML parsing pitfalls are preserved — each static method's docstring explicitly references the pitfalls it handles. Tests were updated to call the static methods directly (265 scraper tests pass). No further structural refactoring is recommended — the remaining coordinator logic in `_parse_vote_page()` and `enrich_legislators()` is straightforward glue code.

## Data Quality Review

### Output Integrity

All 8 bienniums produce three well-formed CSVs. Cross-validation results:

| Check | 86th-91st | 84th-85th |
|---|---|---|
| Vote categories match expected set | Pass | Pass (minus "Not Voting" — unused by KS Legislature) |
| No empty legislator_name | Pass | Pass |
| No empty bill_number | Pass | Pass |
| No whitespace in key fields | Pass | Pass |
| No duplicate `(slug, vote_id)` | Pass | Pass |
| Slug/chamber prefix consistency | Pass | Pass |
| Vote counts match individual records | Pass | Mismatch (ODT-era) |
| Party values valid | Pass | Pass |

### ODT-Era Vote Count Mismatches

The 84th biennium has 900/900 rollcalls with mismatches between `total_votes` in the rollcall CSV and the count of individual vote records. The 85th has 899/1159.

Investigation reveals a consistent pattern:

- **House**: gap of ~11-13 per rollcall (125 expected, ~112 actual individual votes)
- **Senate**: gap of ~2 per rollcall (40 expected, ~38 actual individual votes)

The gaps are in the "Absent and Not Voting" category. ODT files from this era don't always list all absent legislators by name — some are included only in the tally count but not enumerated individually. The `total_votes` field comes from the ODT tally metadata, while individual vote records come from the name-by-name parsing. This is a data limitation of the source format, not a parser bug.

The 86th biennium (2015-2016) and all subsequent HTML-era sessions have zero mismatches.

### "Not Voting" Category

The 5th vote category, "Not Voting," is defined in `VOTE_CATEGORIES` but has zero occurrences across all 8 bienniums. The Kansas Legislature doesn't use this category — their vote pages only have Yea, Nay, Present and Passing, and Absent and Not Voting. The category is kept in the constant tuple for completeness (it exists in the data model) and to prevent the parser from ignoring it if the Legislature ever starts using it.

### `short_title` and `sponsor` Availability

These fields come from the KLISS API (`_filter_bills_with_votes()`):

| Biennium | `short_title` | `sponsor` |
|---|---|---|
| 84th-89th (2011-2022) | 0% | 0% |
| 90th-91st (2023-2026) | 100% | 100% |

The KLISS API is only available for recent sessions. For older sessions, these fields are empty strings. This is expected behavior.

### Failure Rates

| Biennium | Failed | Total | Rate | Primary Cause |
|---|---|---|---|---|
| 84th | 374 | 1,274 | 29.4% | ODT parsing (committee-of-the-whole, tally-only) |
| 85th | 1 | 1,160 | 0.1% | — |
| 86th | 3 | 1,007 | 0.3% | — |
| 87th | 9 | 1,043 | 0.9% | — |
| 88th | 2 | 550 | 0.4% | — |
| 89th | 3 | 1,042 | 0.3% | — |
| 90th | 5 | 1,124 | 0.4% | — |
| 91st | 55 | 882 | 6.2% | Transient 500/502 server errors |

The 84th's 29% failure rate is documented in CLAUDE.md: "~30% are committee-of-the-whole (tally-only)." These ODT files contain vote tallies but no individual legislator names — the parser correctly identifies them as having 0 parseable individual votes and records the failure. This is a data limitation, not a scraper limitation.

The 91st's 55 transient failures are all 500/502 server errors from kslegislature.gov. Re-running the scraper would retry these (failed pages are never cached). The retry wave mechanism (`WAVE_COOLDOWN=90s`, `WAVE_WORKERS=2`, up to 3 waves) already attempted recovery.

## Test Coverage Analysis

### Well-Tested Areas (~85% of parsing logic)

| Component | Tests | Quality |
|---|---|---|
| Session/biennium URL logic | 40 | Excellent — all URL patterns, JS paths, ODT detection |
| Data models | 8 | Excellent — construction, immutability |
| Pure functions (normalize, parse, derive) | 45 | Excellent — edge cases, regression tests |
| HTML parsing (vote categories, party, name) | 35 | Excellent — inline BS4 fixtures, bug regression coverage |
| ODT parser | 47 | Excellent — pure functions, member directory, metadata |
| CSV export | 10 | Good — roundtrip write/read, deduplication |
| CLI | 17 | Good — arg parsing, monkeypatched scraper |
| Data integrity (against real CSVs) | 26 | Excellent — validates scraped output |

### Untested Areas

The HTTP and orchestration layers have no direct test coverage:

| Component | Lines | Tests | Risk |
|---|---|---|---|
| `_get()` — HTTP fetch with retries | ~130 | 0 | High — retry logic, error classification, cache behavior |
| `_fetch_many()` — concurrent fetch + retry waves | ~80 | 0 | High — concurrency, wave mechanics |
| `_rate_limit()` — thread-safe rate limiting | ~10 | 0 | Medium — lock correctness |
| `_filter_bills_with_votes()` — KLISS API | ~65 | 0 | Medium — API format handling |
| `get_bill_urls()` — bill discovery | ~40 | 0 | Medium — HTML + JS fallback flow |
| `get_vote_links()` — vote link scanning | ~55 | 0 | Low — tested via HTML parsing tests |
| `_load_member_directory()` — member lookup | ~65 | 0 | Medium — JS fallback, ambiguity handling |
| `enrich_legislators()` — party/district fetch | ~35 | 0 | Low — parsing tested directly via `_extract_party_and_district()` static method (M2) |
| `run()` — pipeline orchestration | ~95 | 0 | Low — glue code |
| Failure reporting (`failure_manifest.json`, `missing_votes.md`) | ~85 | 0 | Low — output formatting |

### Recommended Test Additions

In priority order by risk of silent data loss:

1. **HTTP layer** — monkeypatch `requests.Session` to simulate 404, 5xx, timeout, connection errors. Verify retry counts, backoff timing, error classification, and cache hits/misses. ~15-20 tests.

2. **KLISS API pre-filter** — mock `_get()` to return JSON in both formats (raw list and `{"content": [...]}`). Verify filtering, metadata extraction, and fallback behavior when API fails. ~8-10 tests.

3. **Member directory loading** — mock HTML and JS member pages. Verify directory construction, ambiguity detection, and initial-qualified key generation. ~8-10 tests.

4. **Cache behavior** — verify cache hits skip network, cache misses write files, binary cache uses `.bin`, `clear_cache()` removes directory. ~5-8 tests.

5. **Full pipeline smoke test** — mock all HTTP calls, run `run()`, verify output CSVs contain expected data. 1 integration test worth more than many unit tests here.

## Recommendations

### Do

1. **Replace `assert` with defensive check** in `_filter_bills_with_votes()` (line 496). Low effort, removes production fragility.

2. **Extract magic numbers** (200, 500) to named constants in `config.py`. Improves discoverability.

3. **Add warning on title truncation.** `if len(bill_title) > BILL_TITLE_MAX_LENGTH: logger.warning(...)`. Prevents silent data loss.

4. **Add HTTP layer tests** (priority 1 above). The retry strategy is the scraper's most critical reliability feature and has zero coverage.

5. **Add a comment** to the `self.delay` mutation in `_fetch_many()` explaining why it's safe (wave runs after all Phase 1 futures complete).

### Don't

1. **Don't switch to httpx/asyncio.** The bottleneck is rate-limiting (0.15s delay), not I/O throughput. Async adds complexity without benefit at 5 workers.

2. **Don't switch to lxml.** BeautifulSoup's lenient parsing is a feature for the KS Legislature's malformed HTML.

3. ~~**Don't refactor `_parse_vote_page()` into smaller functions.**~~ **Done (M2).** Extracted 4 static methods successfully — each method's docstring references the pitfalls it handles, preserving context. All 265 scraper tests pass. No further splitting recommended.

4. **Don't normalize chamber/party upstream in the scraper.** The current hybrid approach (mechanical cleanup at scrape time, semantic normalization at analysis time) matches ecosystem best practice and ADR-0021.

5. **Don't add a third-party ODT library.** stdlib `zipfile` + `xml.etree` is the recommended approach for read-only extraction of specific content.

## Sources

### Open-Source Projects
- [openstates/openstates-scrapers](https://github.com/openstates/openstates-scrapers) — Open States legislative scrapers (50 states)
- [jamesturk/spatula](https://github.com/jamesturk/spatula) — Page-oriented scraping framework
- [unitedstates/congress](https://github.com/unitedstates/congress) — US Congress data collectors
- [freelawproject/juriscraper](https://github.com/freelawproject/juriscraper) — Court opinion and PACER scraper
- [opencivicdata/python-legistar-scraper](https://github.com/opencivicdata/python-legistar-scraper) — Municipal Legistar scraper
- [poliquin/pylegiscan](https://github.com/poliquin/pylegiscan) — Python LegiScan API client
- [ka-chang/StateLegiscraper](https://github.com/ka-chang/StateLegiscraper) — State legislature text corpus scraper

### Libraries and Frameworks
- [jd/tenacity](https://github.com/jd/tenacity) — Python retry library with exponential backoff
- [eea/odfpy](https://github.com/eea/odfpy) — ODF Python API (not recommended for read-only extraction)
- [requests-cache](https://pypi.org/project/requests-cache/) — Persistent HTTP caching for requests

### Best Practices
- [HTTPX vs Requests vs AIOHTTP comparison](https://oxylabs.io/blog/httpx-vs-requests-vs-aiohttp) — HTTP client performance benchmarks
- [Parser performance comparison](https://medium.com/@yahyamrafe202/in-depth-comparison-of-web-scraping-parsers-lxml-beautifulsoup-and-selectolax-4f268ddea8df) — BS4 vs lxml vs selectolax benchmarks
- [OpenStates contributing guide](https://docs.openstates.org/contributing/scrapers/) — Scraper architecture patterns
- [Spatula anatomy of a scrape](https://jamesturk.github.io/spatula/anatomy-of-a-scrape/) — Pipeline stage design
