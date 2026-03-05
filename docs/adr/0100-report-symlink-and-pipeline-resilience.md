# ADR-0100: Report Symlink Stability and Pipeline Resilience

## Status

Accepted (2026-03-05)

## Context

Three issues surfaced during the first KanFocus-only pipeline run (78th biennium, 1999-2000):

1. **Report symlinks broke during re-runs.** In run-directory mode, report convenience symlinks (e.g. `01_eda_report.html`) chained through the session-level `latest` symlink: `latest/<phase>/<report>.html`. When a new pipeline run started, Phase 01's `finalize()` moved `latest` to the new run_id. All report symlinks for phases not yet completed in the new run became dangling.

2. **DB connection failed silently.** `analysis/db.py` defaulted to `postgresql://localhost:5432/tallgrass` (no credentials). The Django settings use `tallgrass:tallgrass@` credentials. Every phase fell back to CSV without the user realizing the DB was available.

3. **Phases 20 (Bill Text) and 21 (TBIP) crashed on missing bill texts.** KanFocus data lacks bill texts. Phases 22 (Issue IRT) and 23 (Model Legislation) already handled `FileNotFoundError` gracefully, but Phases 20 and 21 did not.

## Decision

### Report symlinks point directly at concrete paths

Changed `run_context.py` so report convenience symlinks point to `<run_id>/<phase>/<report>.html` instead of `latest/<phase>/<report>.html`. Each phase updates only its own report symlink on successful completion. The session-level `latest` symlink still updates (for `resolve_upstream_dir()` precedence 4), but report links no longer depend on it.

**Before:** `01_eda_report.html → latest/01_eda/01_eda_report.html`
**After:** `01_eda_report.html → 78-260305.2/01_eda/01_eda_report.html`

### Default DATABASE_URL includes credentials

Changed `_DEFAULT_DATABASE_URL` in `analysis/db.py` from `postgresql://localhost:5432/tallgrass` to `postgresql://tallgrass:tallgrass@localhost:5432/tallgrass`. These are the standard local dev credentials matching the Docker Compose config and Django settings.

### Phases 20 and 21 skip gracefully on missing bill texts

Added `try/except FileNotFoundError` around `load_bill_texts()` in both `analysis/20_bill_text/bill_text.py` and `analysis/21_tbip/tbip.py`. On missing data, the phase prints a message and returns cleanly — matching the pattern already used by Phases 22 and 23.

## Consequences

- Report symlinks survive pipeline re-runs — each always points to its most recent successful output
- The session-level `latest` symlink continues to serve `resolve_upstream_dir()` for inter-phase data resolution
- DB connection works out of the box for local development without setting `DATABASE_URL`
- All four bill-text-dependent phases (20-23) now skip gracefully when bill texts are unavailable
- KanFocus-only bienniums (no bill texts, no bill actions) run the full pipeline without errors
