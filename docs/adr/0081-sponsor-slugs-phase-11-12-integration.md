# ADR-0081: Sponsor Slugs Integration into Synthesis and Profiles

**Date:** 2026-03-02
**Status:** Accepted

## Context

M8 (ADR-0072, roadmap R26) added `sponsor_slugs` to the scraper output and `sponsor_party_R` to Phase 08 (Prediction). The `sponsor_slugs` field is a semicolon-joined list of legislator slugs extracted from bill page `<a>` hrefs (e.g. `"sen_tyson_caryn_1; sen_alley_larry_1"`). Available for sessions with structured HTML bill pages (89th+, 2021-2026); empty for committee sponsors, pre-89th sessions, and data scraped before this feature.

Phase 08 was the only consumer. The question was: which downstream phases would benefit from sponsorship data?

## Decision

After reviewing all 19 analysis phases, integrate `sponsor_slugs` into **Phase 11 (Synthesis)** and **Phase 12 (Profiles)** — the two phases that build per-legislator summaries and are natural consumers of bill metadata. Other phases either don't touch bill metadata or would require forced additions.

### Phase 12 (Profiles)

New `compute_sponsorship_stats()` function in `profiles_data.py`:
- Filters rollcalls where the target legislator's slug appears in the `sponsor_slugs` semicolon-split list
- Marks `is_primary` = True when the slug is the first in the list
- Returns a DataFrame (bill_number, short_title, motion, passed, is_primary) or `None` if column missing

New `_add_sponsorship_section()` in `profiles_report.py`:
- Summary text: "Sponsored N bills (M as primary sponsor). Passage rate: X%."
- Table of sponsored bills with Bill, Title, Motion, Outcome, Role columns
- Inserted between position figure and defections table in the per-legislator report

Enriched `find_defection_bills()`:
- Adds `sponsor` column (from rollcalls) to defection output when present
- Displayed as "Sponsor" column in the defections table

### Phase 11 (Synthesis)

New `_compute_sponsor_summary()` inline helper in `synthesis.py`:
- Splits `sponsor_slugs` by `"; "`, explodes, groups by legislator slug
- Computes `n_bills_sponsored` (count) and `sponsor_passage_rate` (mean of `passed`)
- LEFT JOINs onto each chamber's `leg_df`

New column in full scorecard (`synthesis_report.py`):
- `n_bills_sponsored` → "Bills Sponsored" (integer format) in the unified scorecard table

### Graceful Degradation

All new code degrades silently when `sponsor_slugs` is absent:
- `compute_sponsorship_stats()` returns `None` → report section skipped
- `_compute_sponsor_summary()` returns `None` → `leg_dfs` unchanged
- Defection `sponsor` column only added when present in rollcalls → backward compatible

This handles pre-89th data (84th-88th), pre-rescrape data, and committee-sponsored bills.

## Consequences

### Positive
- Profiles gain a sponsorship section showing each legislator's sponsored bills, primary vs co-sponsor role, and passage rate
- Synthesis scorecard includes bill-sponsored counts for cross-legislator comparison
- Defection table shows the sponsor context for defection votes
- Zero breakage on old data — all features are additive and guarded

### Negative
- `_compute_sponsor_summary()` loads rollcalls CSV at synthesis time (one additional file read per session)
- Phase 12 `compute_sponsorship_stats()` operates on already-loaded rollcalls, no additional I/O

### Data Flow

```
Scraper                 Phase 08            Phase 11             Phase 12
─────────              ──────────          ──────────           ──────────
sponsor_slugs  ──────► sponsor_party_R     n_bills_sponsored    sponsorship stats
(rollcalls.csv)        (prediction)        sponsor_passage_rate (per-legislator)
                                           (scorecard)          defection sponsor
```

All three consumers use the same `sponsor_slugs` column from `rollcalls.csv`. Phase 08 builds a binary feature for ML. Phase 11 aggregates per-legislator counts. Phase 12 does per-bill lookups for a specific legislator.

## Tests

11 new tests (1952 total):
- `TestSponsorshipStats` (4 tests): primary vs co-sponsor, None without column, empty when no matches
- `TestDefectionSponsorEnrichment` (2 tests): sponsor included when present, backward compat without
- `TestComputeSponsorSummary` (4 tests): counts, passage rate, graceful without column, graceful with empty slugs
