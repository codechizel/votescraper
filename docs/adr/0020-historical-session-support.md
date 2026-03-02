# ADR-0020: Historical Session Support (2011-2026)

**Date:** 2026-02-22
**Status:** Accepted

## Context

The scraper originally supported only the three most recent bienniums (2021-22, 2023-24, 2025-26). Extending back to 2011 unlocks 6 additional bienniums (15 years total) for longitudinal analysis, but two format changes block backwards compatibility:

1. **Bill discovery:** Sessions before 2021 load bill listings via JavaScript (`measures_data` arrays in `.js` files) instead of server-rendered HTML with `<a>` tags.
2. **Vote pages:** Sessions 2011-12 and 2013-14 use downloadable `.odt` (OpenDocument Text) files instead of HTML `vote_view` pages. The ODT files contain structured metadata in XML user-field declarations and comma-separated legislator names in paragraph text.
3. **Party detection:** Sessions before 2015 use `<h3>Party: Republican</h3>` on legislator pages instead of `<h2>District N - Republican</h2>`.

## Decision

Add three backwards-compatible extensions to the scraper:

### JS Bill Discovery Fallback
- `KSSession.js_data_paths` provides candidate JS file URLs for pre-2021 sessions
- `_get_bill_urls_from_js()` fetches the JS file, extracts the JSON array between `[` and `]`, and returns `measures_url` values
- `get_bill_urls()` falls back to JS discovery when HTML scanning finds zero bills

### ODT Vote Parser
- New module `odt_parser.py`: pure-function parser (no I/O) that takes ODT bytes and returns `RollCall`/`IndividualVote` instances
- Extracts `content.xml` from the ZIP archive, parses user-field declarations for metadata (chamber, bill number, timestamp, tally, action code)
- Parses comma-separated legislator names from paragraph text, mapping to both House ("Yeas"/"Nays") and Senate ("Present and Passing") category variants
- Resolves last-name-only legislators via a member directory built from the session's `/members/` listing page
- `VoteLink.is_odt` flag routes links to the appropriate parser in `parse_vote_pages()`

### Pre-2015 Party Detection
- Fallback in `_extract_party_and_district()` (called by `enrich_legislators()`): when `<h2>District N - Party</h2>` yields no party, scan `<h3>` tags for `"Party: Republican"` / `"Party: Democrat"`

### Binary Support in FetchResult
- `FetchResult.content_bytes` field for binary downloads (ODT files)
- `_get(binary=True)` stores `response.content` (bytes), skips HTML error-page detection, caches as `.bin`
- `_fetch_many(binary=True)` passes through to `_get()`

## Consequences

### Benefits
- Unlocks 15 years of Kansas Legislature voting data (2011-2026) for longitudinal analysis
- All changes are backwards-compatible: defaults preserve existing behavior for 2021+ sessions
- No new dependencies: uses stdlib `zipfile` and `xml.etree.ElementTree`
- ODT parser is pure logic with no I/O, making it straightforward to test

### Trade-offs
- Last-name-only identification in ODT files is inherently ambiguous for common surnames in the same chamber. Ambiguous names get empty slugs and appear in the failure manifest.
- ODT format may have minor variations between sessions that aren't captured in the inline test fixtures. Real-world scraping will reveal edge cases.
- The member directory is an additional HTTP request per session (only for ODT sessions).

### Session Coverage

| Biennium | Bill Discovery | Vote Format | Party Detection |
|----------|---------------|-------------|-----------------|
| 2025-26 (91st) | HTML links | vote_view | `<h2>District N - Party` |
| 2023-24 (90th) | HTML links | vote_view | `<h2>` |
| 2021-22 (89th) | HTML links | vote_view | `<h2>` |
| 2019-20 (88th) | JS fallback | vote_view | `<h2>` |
| 2017-18 (87th) | JS fallback | vote_view | `<h2>` |
| 2015-16 (86th) | JS fallback | vote_view | `<h3>Party:` |
| 2013-14 (85th) | JS fallback | odt_view | `<h3>Party:` |
| 2011-12 (84th) | JS fallback | odt_view | `<h3>Party:` |
