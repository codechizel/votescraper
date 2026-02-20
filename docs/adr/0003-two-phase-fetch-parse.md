# ADR-0003: Two-phase fetch/parse pattern

**Date:** 2026-02-19
**Status:** Accepted

## Context

The scraper makes hundreds of HTTP requests to kslegislature.gov (bill pages, vote pages, legislator pages). We need concurrency for acceptable performance, but parsing HTML and building data structures involves shared mutable state (lists of results, name-matching lookups).

## Decision

All HTTP fetching uses a **two-phase pattern**:

1. **Phase 1 (concurrent)**: Fetch all pages via `ThreadPoolExecutor` (MAX_WORKERS=5). Each worker returns raw HTML (or parsed partial data). No shared state is mutated during this phase.
2. **Phase 2 (sequential)**: Parse all fetched HTML and build data structures in a single thread.

Rate limiting is thread-safe via `threading.Lock()` — the only shared state accessed during Phase 1.

This pattern applies to every fetch operation in the scraper:
- `get_bill_urls()` — fetches bill listing pages
- `_filter_bills_with_votes()` — fetches KLISS API endpoints
- `get_vote_links()` — fetches vote page links per bill
- `parse_vote_pages()` — fetches and parses individual vote pages
- `enrich_legislators()` — fetches legislator profile pages

## Consequences

- **Good**: No data races — shared state (result lists, legislator dicts) is only written to in Phase 2
- **Good**: Simple mental model — "fetch everything, then process everything"
- **Good**: Rate limiting is the only synchronization point, and it's a simple lock
- **Trade-off**: Peak memory is higher because all raw HTML is held in memory before parsing. At our scale (~500 pages, each ~50KB) this is ~25MB — negligible.
- **Trade-off**: Cannot stream results or show incremental progress during the fetch phase
