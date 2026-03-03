# ADR-0075: Name Matcher District Tiebreaker + Shrinkage Null Investigation

**Date:** 2026-03-02
**Status:** Accepted

## Context

Pipeline audit (ADR-0072) identified two backlog items requiring investigation:

1. **Backlog #1 — Name matcher incorrect matches.** Phase 14 (Shor-McCarty) and Phase 18 (DIME) used last-name-only matching in `_phase2_last_name_match()` without district disambiguation. Three confirmed incorrect matches: Bethell (84th, same-district successor), Dannebohm (86th, same-district successor), and Weber (86th, different district). The report text and docstrings falsely claimed a district tiebreaker was already implemented.

2. **Backlog #2 / A13 — Null `hier_shrinkage_pct` in synthesis.** 24.8% of legislators across all 8 bienniums had null `hier_shrinkage_pct` in the synthesis DataFrame. The `SHRINKAGE_MIN_DISTANCE=0.5` threshold in `hierarchical.py` sets shrinkage to null when a legislator's flat IRT estimate is within 0.5 of their party mean on the hierarchical scale. Required investigation to determine if this was a bug or working-as-designed.

## Decision

### District Tiebreaker (Backlog #1)

Implemented district-based disambiguation in both Phase 14 and Phase 18:

- **SM (Phase 14):** New `_extract_sm_district()` extracts year-specific `hdistrict{YYYY}`/`sdistrict{YYYY}` columns from the SM dataset, coalesces across biennium years, casts to Int64. `_phase2_last_name_match()` now accepts `start_year` and `chamber` parameters.

- **DIME (Phase 18):** New `_parse_dime_district()` parses DIME's variable-format district strings (`"KS-113"`, `"KS01"`, `"KS-7"`, `"27"`) to Int64.

- **Shared logic** (`_deduplicate_with_district()`): When multiple external candidates share a last name with one of our legislators:
  - Single candidate → kept as-is (no change from previous behavior)
  - Multiple candidates, one matches on district → use that match
  - Multiple candidates, none match on district → reject entirely (no match is better than a wrong match)

### Shrinkage Null Investigation (Backlog #2 / A13)

**Accepted as working-as-designed.** Deep dive across all 8 bienniums found:

| Metric | Value |
|--------|-------|
| Overall null rate | 24.8% (13.5%-35.7% per chamber-session) |
| Bienniums affected | All 8 (not 84th-specific) |
| Threshold relative to scale | 2.3-4.4% of total range, 4.0-9.0% of party gap |
| Without threshold | Values swing from -2222% to +86% due to near-zero denominators |
| Downstream consumers | 1 interactive table in hierarchical report (blank cells, no crash) |

The `toward_party_mean` boolean (always non-null for non-anchors) already captures the actionable direction of shrinkage. The percentage is informational and is not used in synthesis detection, profiles, cross-session, or any scoring logic. The A13 audit finding ("report gaps") overstates the actual impact.

## Consequences

### Positive

- 3 incorrect name matches eliminated across all bienniums
- Report text about district tiebreaker is now truthful (was aspirational)
- Shrinkage investigation closed with documented rationale — no further work needed
- 6 new tests (3 per external validation phase)

### Negative

- Some same-district successor cases (Bethell, Dannebohm) remain unmatched — district alone cannot distinguish predecessors from successors. Negligible impact (~2 matches across all bienniums).
- `match_legislators()` gains a new `start_year` parameter (default=0 for backward compatibility).

### Neutral

- Phase 14 and Phase 18 callers updated to pass `start_year` — minimal code change.
- 84th biennium pipeline re-run recommended to pick up the name matcher fix plus 6 other code improvements since the last run.

## Files Changed

| File | Change |
|------|--------|
| `analysis/17_external_validation/external_validation_data.py` | `_extract_sm_district()`, `_deduplicate_with_district()`, updated `_phase2_last_name_match()` and `match_legislators()` |
| `analysis/18_dime/external_validation_dime_data.py` | `_parse_dime_district()`, `_deduplicate_with_district()`, updated `_phase2_last_name_match()` |
| `analysis/17_external_validation/external_validation.py` | Pass `start_year` to `match_legislators()` |
| `tests/test_external_validation.py` | `TestDistrictTiebreaker` (3 tests) |
| `tests/test_external_validation_dime.py` | `TestDimeDistrictTiebreaker` (3 tests) |
| `docs/roadmap.md` | Backlog #1 done, #2 accepted, #3 done, A13 accepted |
