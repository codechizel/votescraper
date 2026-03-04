# ADR-0080: Code Audit 2026-03-02 — Resolution

**Date:** 2026-03-02
**Status:** Accepted

## Context

A static code audit (`docs/code-audit-2026-03-02.md`) identified 6 findings at commit `7811177`. This ADR documents the review, triage, and resolution of each finding.

## Findings and Disposition

### Fixed

**A. Synthesis manifest key mismatch (high priority — BUG)**

`synthesis_data.py` stores manifests under full phase IDs (`"01_eda"`, `"13_indices"`, `"09_clustering"`) but 7 access sites in `synthesis.py` and `synthesis_report.py` used short keys (`"eda"`, `"indices"`, `"clustering"`). Every `.get()` silently returned `{}`, causing the pipeline summary infographic and report intro to display `"?"` or fall back to hardcoded defaults instead of actual counts from the EDA manifest.

Root cause: `UPSTREAM_PHASES` was refactored from short to full IDs, but consumer code was never updated.

**Fix:** Changed all 7 short-key accesses to full phase IDs:
- `synthesis.py:483-485` — `"eda"` → `"01_eda"`, `"indices"` → `"13_indices"`, `"clustering"` → `"09_clustering"`
- `synthesis_report.py:157,253,691,1225,1280` — same substitutions

The one correct access (`manifests.get("11_network")` at line 638) was already using the full ID.

### Also Fixed

**B. Monitoring callback non-functional with nutpie (low priority — FIXED)**

`create_monitoring_callback()` wrote status JSON for `pm.sample(callback=...)`, which nutpie doesn't support. All sampling functions accepted `callback=` for API compatibility but printed "ignored". `just monitor` always showed "No experiment running."

**Fix:** Removed the entire dead callback chain:
- Deleted `create_monitoring_callback()` from `experiment_monitor.py` (~60 lines)
- Removed `callback=` parameter from `build_per_chamber_model()` and `build_joint_model()`
- Removed callback creation/passing from `experiment_runner.py`
- Removed 5 callback tests from `test_experiment_monitor.py`
- Updated `just monitor` to check PID file instead of status JSON (nutpie shows its own terminal progress bar via Rust's indicatif crate)

**C. Dead helpers in `irt.py` (low priority — FIXED)**

Two functions removed (~155 lines):
- `plot_joint_vs_chamber` — joint MCMC visualization orphaned when joint model was abandoned for mean/sigma equating
- `run_joint_pca_for_anchors` — joint anchor selection never integrated

All other functions the audit flagged (`plot_paradox_spotlight`, `compare_with_pca`) are called from `main()` in production. `unmerge_bridging_legislators` has dedicated tests and was retained.

**D. Special session analysis support (medium priority — FIXED)**

`KSSession.from_session_string("2024s")` raised `ValueError: invalid literal for int()`. All 20+ analysis phases call this method. The infrastructure below was already ready (`RunContext`, directory layout, `from_year(..., special=True)`), but the entry point crashed.

**Fix:** `from_session_string()` now detects the `s` suffix and routes to `from_year(year, special=True)`. `data_dir_for_session()` also auto-detects the `"2024s"` format. 4 new tests added.

**E. Cache key collision risk (low priority — FIXED)**

Cache filenames were derived from URLs via character replacement and truncated to 200 chars. While no real collision was plausible with current URL patterns (max ~92 chars), the truncation approach was needlessly fragile.

**Fix:** Replaced with SHA-256 hash (first 16 hex chars = 64 bits of entropy). Fixed-length filenames, zero collision risk, no truncation. Removed `CACHE_FILENAME_MAX_LENGTH` constant. Existing caches are invalidated (re-fetched on next run).

## Consequences

- Synthesis reports now display actual EDA/indices/clustering manifest data instead of fallback defaults
- All 20+ analysis phases now support `--session 2024s` for special sessions
- Cache key collisions eliminated via SHA-256 hashing (existing caches invalidated)
- ~270 lines of dead code removed (callback infrastructure + joint model helpers)
- `just monitor` now checks PID file; nutpie's built-in progress bar handles real-time MCMC monitoring
- All 5 audit findings resolved
