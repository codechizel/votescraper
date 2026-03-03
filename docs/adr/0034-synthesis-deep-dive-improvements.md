# ADR-0034: Synthesis Deep Dive Improvements

**Date:** 2026-02-25
**Status:** Accepted

## Context

A comprehensive audit of the synthesis phase (Phase 11) identified seven issues, nine test gaps, and two refactoring opportunities. The audit compared the implementation against the academic literature for multi-method legislative analysis synthesis, reviewed detection algorithm correctness, and surveyed the open-source landscape. Full findings documented in `docs/synthesis-deep-dive.md`.

Key findings:
- No open-source Python project combines IRT + network + clustering + indices + prediction into a unified synthesis report — our implementation fills a genuine gap.
- The `plot_pipeline_summary` function had dead code (a `pass`-body loop) and hardcoded AUC/vote-count values from the 91st Legislature that would silently display wrong numbers on other sessions.
- Only majority-party mavericks were detected, missing an analytically interesting signal: minority-party members who cross the aisle.
- Data loading functions (~200 lines) lived in the heavyweight `synthesis.py` orchestrator but were imported by two downstream phases (`profiles.py`, `cross_session.py`), creating an unnecessarily heavy dependency chain.
- `synthesis_data.py` had zero tests; `detect_all()` had no integration test.

## Decision

Implement 9 of 12 recommendations from the deep dive. Defer 3 lower-priority items (cross-method correlation table, 3+ party bridge midpoint, multiple paradoxes per session).

### Changes made:

1. **Dynamic AUC extraction:** New `_extract_best_auc()` helper reads XGBoost AUC from holdout_results parquets. Replaced dead loop + hardcoded `0.98`. Manifest-derived fallbacks changed from session-specific numbers to `"?"`.

2. **Minority-party maverick detection:** `detect_all()` now calls `detect_chamber_maverick()` for each minority party. New `_minority_parties()` helper. Report builder adds narrative paragraphs about "crossing the aisle". Return dict includes `minority_mavericks` key.

3. **`synthesis_data.py` extraction:** Moved `UPSTREAM_PHASES`, `_read_parquet_safe`, `_read_manifest`, `load_all_upstream`, and `build_legislator_df` into a new pure-I/O module. Updated imports in `synthesis.py`, `profiles.py`, `cross_session.py`, and `analysis/__init__.py` module map.

4. **Code quality fixes:** `report: object` → `report: ReportBuilder` with TYPE_CHECKING guard; `slug.split("_")[1]` → `slug.split("_", 1)[1]` in 3 locations; removed duplicated annotation logic from `plot_dashboard_scatter`; updated all stale docstrings (8→10 phases, 27-30→29-32 sections).

5. **47 new tests** in `test_synthesis.py` covering data loading, joins, AUC extraction, `detect_all` integration, minority mavericks, Democrat-majority paradox, bridge-builder fallback, and edge cases.

## Consequences

**Positive:**
- Pipeline infographic now shows correct (or explicit "?") values on any session, not silently wrong 91st-specific numbers.
- Minority-party mavericks surface an analytically interesting signal previously invisible in the report.
- `synthesis_data.py` gives downstream phases (`profiles.py`, `cross_session.py`) a lightweight import path that doesn't pull in matplotlib or plotting code.
- Test count: 975 → 1022. The synthesis data layer and detection orchestrator now have meaningful coverage.

**Negative:**
- One more module in the `11_synthesis/` directory (4 files instead of 3). The module map in `analysis/__init__.py` grew by one entry.
- Minority mavericks add report length. In sessions where both parties have very high unity (> 0.95), the section is skipped gracefully.

**Files changed:**
- `analysis/24_synthesis/synthesis_data.py` — New, ~236 lines.
- `analysis/24_synthesis/synthesis.py` — Removed data loading functions, added `_extract_best_auc`, updated imports and `main()`.
- `analysis/24_synthesis/synthesis_detect.py` — Added `_minority_parties()`, updated `detect_all()`.
- `analysis/24_synthesis/synthesis_report.py` — Added minority maverick narrative, fixed type annotation, fixed slug splitting.
- `analysis/25_profiles/profiles.py` — Import path updated (synthesis → synthesis_data).
- `analysis/26_cross_session/cross_session.py` — Import path updated.
- `analysis/__init__.py` — Added `synthesis_data` to module map.
- `tests/test_synthesis.py` — New, 47 tests.
- `docs/synthesis-deep-dive.md` — New deep dive article.
- `analysis/design/synthesis.md` — Updated assumptions, architecture, maverick description.
