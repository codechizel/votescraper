# ADR-0122: Cross-Chamber Person Identity Resolution

**Date:** 2026-03-25
**Status:** Accepted
**Deciders:** Joseph Claeys

## Context

The common-space phase (Phase 28) uses OCD person IDs from OpenStates as the primary identity key for matching legislators across bienniums. However, the `ks_slug_to_ocd.json` mapping only contains slugs that OpenStates has indexed. When a legislator switches chambers (e.g., Caryn Tyson: `rep_tyson_caryn_1` in 84th, `sen_tyson_caryn_1` in 85th+), often only one chamber's slug appears in the OCD mapping. The unmapped slug falls through to slug-based identity resolution, producing a different person_key (e.g., `tyson_caryn_1` vs `ocd-person/cc84252c-...`).

This creates duplicate entries in unified career scores — the same legislator appears twice with different scores, splitting their voting history into two incomplete records. Investigation revealed 30 chamber-switch orphans across the 14-biennium dataset, plus 2 cases (Dan Goddard, Ronald Ryckman Sr.) where OpenStates itself assigns different OCD IDs to the same person across chambers.

## Decision

Three-layer fix in `common_space_data.py`:

### 1. Cross-chamber variant expansion

In `build_person_key_lookup()`, after building the initial slug→OCD mapping, derive the other-chamber variant of each slug (`rep_` ↔ `sen_`) and add it to the lookup — but only if that variant doesn't already have its own mapping to a different OCD ID (which would indicate a genuinely different person, like the two Mike Thompsons).

This automatically resolves 28 of 30 orphans without manual overrides.

### 2. OCD overrides for split OCD IDs

Added `_OCD_OVERRIDES` entries for:
- **Dan Goddard**: Senate (87th-88th) → House (90th-91st), two different OCD IDs
- **Ronald Ryckman Sr.**: House (84th-86th) → Senate (89th-91st), two different OCD IDs

Same pattern as the existing J.R. Claeys override (3 OCD IDs → 1 canonical).

### 3. Duplicate detection quality gate

`detect_potential_duplicates()` runs after roster construction and checks for different person_keys that share the same slug root. If collisions are found that aren't in the `_SAME_NAME_DIFFERENT_PERSON` allowlist (currently just Mike Thompson), the phase raises `ValueError` with actionable guidance.

This ensures future chamber-switchers or OpenStates data quality issues are caught immediately rather than silently producing duplicate rows.

## Consequences

**Positive:**
- Caryn Tyson (and 29 other chamber-switchers) correctly unified into single career scores
- Quality gate catches future regressions automatically — no manual checking needed
- Allowlist pattern handles genuinely different same-name legislators (two Mike Thompsons)
- 15 new tests cover the identity resolution and duplicate detection

**Negative:**
- `_SAME_NAME_DIFFERENT_PERSON` allowlist must be maintained if new same-name-different-person cases appear (rare)
- Cross-chamber expansion assumes `rep_`/`sen_` prefix convention; non-standard slugs won't benefit (not an issue for Kansas data)

**Trade-offs:**
- Quality gate is strict (raises `ValueError`) rather than soft (warning only). This is intentional: silent duplicates are worse than a stopped pipeline. The fix is always adding an override or allowlist entry.
