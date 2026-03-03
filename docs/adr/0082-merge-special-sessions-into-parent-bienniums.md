# ADR-0082: Merge Special Sessions into Parent Bienniums

**Date:** 2026-03-02
**Status:** Accepted

## Context

Kansas special sessions are short (1-2 days, 3-8 roll calls) addressing 1-2 issues. The analysis pipeline's MIN_VOTES=20 filter eliminates all legislators, making IRT/PCA/clustering impossible as standalone runs. Previously, special sessions had their own pipeline pathway that would skip 14 of 16 phases — producing only EDA and a few basic counts.

The five known special sessions (2013s, 2016s, 2020s, 2021s, 2024s) contain 3-8 roll calls each. Their votes are substantively interesting (COVID response, school finance, redistricting) but statistically useless in isolation.

## Decision

Instead of running special sessions through a separate pipeline, merge their scraped data into the parent biennium CSVs. The scraper is untouched — specials still scrape into `data/kansas/2020s/`. A post-scrape `merge_special.py` utility reads special session CSVs and concatenates them into the parent biennium's CSVs.

### Parent biennium derivation

`parent_start = year if year % 2 == 1 else year - 1`

| Special | Parent |
|---------|--------|
| 2013s | 85th (2013-2014) |
| 2016s | 86th (2015-2016) |
| 2020s | 88th (2019-2020) |
| 2021s | 89th (2021-2022) |
| 2024s | 90th (2023-2024) |

### Merge rules

- **votes.csv**: Filter out previous special rows by `session` column, concat. `vote_id` is globally unique.
- **rollcalls.csv**: Same filter-then-concat. Column alignment adds `sponsor_slugs=""` to parent if missing.
- **legislators.csv**: Concat + `unique(subset=["slug"], keep="first")`. Parent rows preferred.
- **bill_actions.csv**: No-op. Specials don't produce bill_actions (KLISS API 404/500).

### Idempotency

Filtering by the `session` column before concat means running merge twice produces identical output.

### Interface

```bash
just merge-special 2020     # merge one
just merge-special all       # merge all five
```

## Consequences

**Positive:**
- Special session votes become analyzable via the standard pipeline (`just pipeline 2019-20` now includes 2020 Special data)
- No scraper changes required — separation of concerns preserved
- Idempotent — safe to re-run after re-scraping
- `KSSession.parent_session` property reusable for future cross-session work

**Negative:**
- Parent biennium CSVs are modified in-place (original data can be restored by re-scraping)
- Special session data directories remain as-is (not cleaned up) — serves as source of truth

**Neutral:**
- The `session` column in merged CSVs distinguishes special vs regular rows, so analysis phases can filter if needed
