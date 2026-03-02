# ADR-0071: Tier 3 Report Enhancements

**Date:** 2026-03-01
**Status:** Accepted

## Context

ADR-0069 delivered Tier 1 (R1-R6) and Tier 2 (R7-R13) report enhancements. Tier 3 covers seven substantial new features spanning downloadable data, full voting records, geographic visualization, cross-session cohort analysis, a unified dashboard, and scrollytelling narrative presentation.

## Decision

Seven enhancements implemented in dependency order:

### R17 — Downloadable CSV Alongside Reports
- `DownloadSection` in `analysis/report.py` renders `<a download>` links
- `RunContext.export_csv()` writes Polars DataFrames to CSV and auto-appends a `DownloadSection`
- 28 export calls across 14 phase scripts (IRT ideal points, indices, scorecards, etc.)

### R15 — Full Voting Record Per Legislator
- `build_full_voting_record()` in `profiles_data.py` joins votes, rollcalls, and party context
- Rendered via `make_interactive_table()` (ITables, searchable/sortable)
- `--full-record` flag; auto-enabled when `--names` is used

### R18 — Freshmen Cohort Analysis
- `FreshmenAnalysis` frozen dataclass with KS test (ideology), t-test (unity), maverick comparison
- `analyze_freshmen_cohort()` in `cross_session_data.py` tags new vs returning legislators
- KDE density overlay and comparison table in cross-session report

### R19 — Voting Bloc Stability Tracking
- `compute_bloc_stability()` using sklearn `adjusted_rand_score`
- Plotly Sankey diagram for cluster transitions between sessions
- Transition matrix (make_gt) and switcher list (ITables) in cross-session report

### R14 — Folium District Choropleth Maps
- New `analysis/01_eda/geographic.py` module
- Downloads TIGER/Line GeoJSON from Census Bureau (Kansas House/Senate districts)
- Folium maps with party + ideology layers, LayerControl toggle, hover tooltips
- Graceful fallback if `folium`/`geopandas` not installed or GeoJSON unavailable

### R16 — Pipeline Dashboard Index
- `analysis/dashboard.py` generates `index.html` with sidebar nav + iframe embedding
- Scans run directory for phase reports, reads `run_info.json` for elapsed times
- `just dashboard` recipe; auto-called at end of `just pipeline`
- Lightweight iframe approach (no Quarto dependency)

### R20 — Scrollytelling in Synthesis
- `ScrollyStep` and `ScrollySection` frozen dataclasses in `report.py`
- IntersectionObserver-based scroll transitions (inline JS, no external deps)
- `build_scrolly_synthesis_report()` converts narrative into 6 chapters
- `--scrolly` flag (default: linear layout preserved)

## Consequences

- 57 new tests across 4 test files (report sections, profiles, cross-session, dashboard)
- New dependencies: `folium>=0.18`, `geopandas>=1.0` (dev only, for R14)
- Dashboard replaces manual navigation between phase reports
- Scrollytelling is opt-in (`--scrolly`), preserving the default linear report
- All existing reports unchanged; new features are additive
