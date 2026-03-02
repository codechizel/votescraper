# ADR-0080: Code Audit 2026-03-02 — Resolution

**Date:** 2026-03-02
**Status:** Accepted

## Context

A static code audit (`docs/code-audit-2026-03-02.md`) identified 6 findings at commit `7811177`. This ADR documents the review, triage, and resolution of each finding.

## Findings and Disposition

### Fixed

**A. Synthesis manifest key mismatch (high priority — BUG)**

`synthesis_data.py` stores manifests under full phase IDs (`"01_eda"`, `"07_indices"`, `"05_clustering"`) but 7 access sites in `synthesis.py` and `synthesis_report.py` used short keys (`"eda"`, `"indices"`, `"clustering"`). Every `.get()` silently returned `{}`, causing the pipeline summary infographic and report intro to display `"?"` or fall back to hardcoded defaults instead of actual counts from the EDA manifest.

Root cause: `UPSTREAM_PHASES` was refactored from short to full IDs, but consumer code was never updated.

**Fix:** Changed all 7 short-key accesses to full phase IDs:
- `synthesis.py:483-485` — `"eda"` → `"01_eda"`, `"indices"` → `"07_indices"`, `"clustering"` → `"05_clustering"`
- `synthesis_report.py:157,253,691,1225,1280` — same substitutions

The one correct access (`manifests.get("06_network")` at line 638) was already using the full ID.

### Remaining Items (Roadmap)

**B. Monitoring callback non-functional with nutpie (low priority — DOCUMENT)**

The `create_monitoring_callback()` in `experiment_monitor.py` writes status JSON for `pm.sample(callback=...)`, which nutpie doesn't support. All three sampling functions accept `callback=` for API compatibility but explicitly print "ignored". `just monitor` always shows "No experiment running."

Mitigating factor: nutpie provides its own terminal progress bar via `progress_bar=True` (step size, divergences, gradients/draw per chain). Real-time monitoring works — just not through `just monitor`.

Action: Either strip the dead callback infrastructure or replace with a nutpie-aware heartbeat. Not urgent — the operational need (watching MCMC progress) is met by the terminal progress bar.

**C. Dead helpers in `irt.py` (low priority — CLEANUP)**

Two functions are genuinely dead:
- `plot_joint_vs_chamber` (line 601, ~100 lines) — joint MCMC visualization orphaned when joint model was abandoned for mean/sigma equating
- `run_joint_pca_for_anchors` (line 489, ~50 lines) — joint anchor selection never integrated

All other functions the audit flagged (`plot_paradox_spotlight`, `compare_with_pca`) are called from `main()` in production.

Action: Remove ~150 lines of dead code.

**D. Special session analysis support (medium priority — FIXED)**

`KSSession.from_session_string("2024s")` raised `ValueError: invalid literal for int()`. All 20+ analysis phases call this method. The infrastructure below was already ready (`RunContext`, directory layout, `from_year(..., special=True)`), but the entry point crashed.

**Fix:** `from_session_string()` now detects the `s` suffix and routes to `from_year(year, special=True)`. `data_dir_for_session()` also auto-detects the `"2024s"` format. 4 new tests added.

**E. Cache key collision risk (low priority — DEFENSE-IN-DEPTH)**

Cache filenames are derived from URLs via character replacement and truncated to 200 chars. In practice, all kslegislature.gov URLs produce cache keys of 55-92 chars — no truncation ever occurs. The risk is theoretical.

Action: Switch to a hash-based cache key (2-line change) to eliminate the bug class entirely. Not urgent — no known or plausible scenario triggers a collision.

## Consequences

- Synthesis reports now display actual EDA/indices/clustering manifest data instead of fallback defaults
- Remaining items are tracked here for future cleanup passes
- No architectural changes needed — all remaining items are localized fixes
