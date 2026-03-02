# ADR-0078: Code Audit — phase_utils Extraction, Bug Fixes, and Efficiency Improvements

**Date:** 2026-03-02
**Status:** Accepted

## Context

A full-codebase audit identified 22 items across the scraper, analysis infrastructure, all 17 phases, and tests. Key patterns:

1. **Cross-file duplication.** `print_header()` and `save_fig()` were copy-pasted across 21 phase scripts. `load_metadata()` (rollcalls + legislators with slug rename, leadership suffix strip, party fill) was duplicated in 8 phases. `normalize_name()` was duplicated in 2 phases.

2. **Bugs.** Dead if/else in scraper retry logic (both branches identical), division-by-zero potential in IRT linking denominators, duplicate `_parse_vote_tally()` call in sort key.

3. **Dead code.** `_itables_init_html()` in report.py always returned empty string (vestigial from ITables init investigation). Empty `NICKNAME_MAP` in external validation never populated.

4. **Efficiency.** O(n²) bill polarization loop in bipartite analysis (nested Python for-loop over legislators × bills). Repeated `.to_list()` calls on same Polars column in clustering. Per-expression DataFrame creation in report number formatting.

5. **Error handling gaps.** Missing KeyError guard in IRT anchor extraction, unguarded `os.open()` in experiment monitor, KLISS API response assumed to be list or dict but could be other types.

## Decision

### Created `analysis/phase_utils.py` (R1-R3)

Central module for utilities duplicated across phase scripts:

- `print_header(title)` — 80-char banner
- `save_fig(fig, path, dpi=150)` — save + close + print
- `load_metadata(data_dir)` → `(rollcalls, legislators)` — loads both CSVs, applies `_clean_legislators()`
- `load_legislators(data_dir)` → `legislators` — loads legislators CSV only
- `normalize_name(name)` — lowercase + strip leadership suffixes

`_clean_legislators()` handles the slug→legislator_slug rename (ADR-0066), `strip_leadership_suffix()`, and Independent party fill (ADR-0021) — the same logic previously duplicated in every phase.

Replaced local definitions in 21 phase scripts (~400 lines removed).

### Bug Fixes (B1-B3)

- **B1:** Collapsed dead if/else in `scraper.py` retry wave (`ok` was always True at that point).
- **B2:** Added `ValueError` guards in `irt_linking.py` `link_mean_mean()` and `link_mean_sigma()` when `mean(a_target)` or `std(a_target)` is zero (degenerate anchor items).
- **B3:** Stored tally result in sort tuple `(margin, failure, tally)` to avoid redundant re-parse of `_parse_vote_tally()`.

### Dead Code Removal (D1-D2)

- **D1:** Removed `_itables_init_html()` and its call site from `report.py`.
- **D2:** Removed empty `NICKNAME_MAP` and its conditional usage from `external_validation_data.py`.

### Efficiency (E1-E3)

- **E1:** Vectorized bipartite bill polarization with numpy integer masks and dot products. Party membership vectors (`is_r`, `is_d`) as `int8` arrays, matrix multiply against voted/yea masks. Eliminates per-bill Python loop entirely.
- **E2:** Cached `legislator_slug` column `.to_list()` once in clustering, reused for 3 dict constructions.
- **E3:** Batched all number format expressions into single `with_columns()` call in `report.py`.

### Error Handling (H1-H3)

- **H1:** Added membership check in `irt_linking.py` `extract_anchor_params()` — unmatched vote IDs skipped with drop counter.
- **H2:** Wrapped `os.open()` in `experiment_monitor.py` with `try/except OSError` for better error context.
- **H3:** Added `isinstance(data, dict)` guard in scraper KLISS API response handling — non-list/non-dict falls through to empty list.

### Scraper Refactoring (R4)

Extracted `_parse_js_array()` static method shared by `_parse_js_bill_data()` and `_parse_js_member_data()` — both had identical unquoted-key normalization logic.

### Test Improvements (T2)

- `test_clustering.py`: Added `0.0 <= avg_loyalty <= 1.0` range check.
- `test_dynamic_irt.py`: Added `hasattr(model, "named_vars")` structure check.

### Deferred Items

- **R6:** Test helper consolidation (`_make_legislators` in 5 files, `_make_votes` in 3 files) — deferred to separate pass.
- **R8:** Split `_parse_vote_page()` (191 lines) — deferred, function is well-structured internally.
- **R9:** Extract `_extract_party_and_district()` from `enrich_legislators()` — deferred.
- **T1:** Mark long-running tests with `@pytest.mark.slow` — deferred.

## Consequences

- **~400 lines removed** from phase scripts via `phase_utils.py` extraction.
- **Single source of truth** for legislator loading conventions (slug rename, suffix strip, party fill).
- **Bipartite polarization** is now O(n) numpy vectorized instead of O(n²) Python loop.
- **3 latent bugs fixed** (dead branch, division-by-zero, duplicate parse) — none were causing failures in practice but could in edge cases.
- **Import pattern:** All phases use `from analysis.phase_utils import print_header, save_fig` (or `load_metadata`, etc.) with the standard `try/except ImportError` fallback for the PEP 302 meta-path finder.
- **1822 tests passing**, lint clean, typecheck clean after all changes.
