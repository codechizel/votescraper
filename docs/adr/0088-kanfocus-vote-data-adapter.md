# ADR-0088: KanFocus Vote Data Adapter

## Status

Accepted (2026-03-03)

## Context

The tallgrass scraper covers 2011-2026 (84th-91st legislatures) by scraping kslegislature.gov. KanFocus (kanfocus.com) is a paid Kansas Legislature tracking service with individual roll call vote data from 1999-2026 (78th-91st legislatures). KanFocus data can:

1. **Add 6 new bienniums** (78th-83rd, 1999-2010) — zero coverage today
2. **Fill 374 missing 84th votes** — committee-of-the-whole ODTs with tally-only data
3. **Cross-validate** existing data quality

## Decision

Build a `tallgrass-kanfocus` CLI tool as a new subpackage at `src/tallgrass/kanfocus/`, following the `tallgrass-text` adapter pattern. Produces the same 3 CSV files as the main scraper (votes, rollcalls, legislators) so the analysis pipeline works unchanged.

### Key Design Decisions

**vote_id scheme**: `kf_{vote_num}_{year}_{chamber}` (e.g., `kf_33_2011_S`). The `kf_` prefix distinguishes from existing `je_` timestamp IDs. For `vote_datetime`, uses `YYYY-MM-DDT00:00:00` (KanFocus provides date but not time).

**Vote category mapping**: KanFocus has 4 categories; tallgrass has 5.

| KanFocus | Tallgrass |
|----------|-----------|
| Yea | Yea |
| Nay | Nay |
| Present | Present and Passing |
| Not Voting | Not Voting |

The analysis pipeline treats "Not Voting" and "Absent and Not Voting" identically (both excluded from vote matrices). KanFocus does not distinguish between the two.

**Vote enumeration**: No API listing all votes. Iterates vote numbers from 1 upward for each of 4 streams per biennium (H/S x odd_year/even_year). Stops after 20 consecutive empty pages.

**Rate limiting**: 7-second delay between requests, single-threaded. KanFocus is a shared paid service — conservative defaults prevent degrading performance for other users. A biennium with ~2000 votes takes ~4 hours.

**Slug generation**: Generates `{sen_|rep_}_{lastname}_{firstname}_{1}` from the KanFocus "Name, R-32nd" format. For overlapping sessions (84th+), cross-references against existing legislator CSVs to reuse established slugs.

**CLI modes**: `--mode full` (default) for complete scrape; `--mode gap-fill` for appending missing votes to existing CSVs. Gap-fill filters by `kf_` vote_id prefix for idempotent re-runs.

**KSSession reuse**: `KSSession` produces correct directory names for any start_year. The KanFocus adapter uses KSSession only for directory naming, never for kslegislature.gov URL properties.

## URL Structure

```
https://kanfocus.com/Tally_House_Alpha_{session_id}.shtml?&Unique_VoteID={vote_num}{year}{chamber}
```

Session IDs: 106 = 78th (1999-2000), 107 = 79th (2001-2002), ..., 119 = 91st (2025-2026). Formula: `(start_year - 1999) // 2 + 106`.

## Consequences

- Coverage extends from 84th-91st to 78th-91st (1999-2026)
- 374 previously tally-only 84th votes gain individual legislator data
- KanFocus-sourced data identifiable by `kf_` vote_id prefix
- Analysis pipeline requires no changes (same CSV format)
- Cache in `data/kansas/{output_name}/.cache/kanfocus/` prevents redundant fetches
- 141 new tests covering session mapping, HTML parsing, slug generation, output conversion, and caching
