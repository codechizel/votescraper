# 79th Legislature Report Audit

**Date:** 2026-03-12
**Scope:** All 20 HTML reports for the 79th Kansas Legislature (2001-2002)
**Context:** The 79th is the canonical horseshoe biennium (30R/10D Senate) with KanFocus-only data (no bill titles, no bill_actions, no bill_texts).

## Summary

A full review of all 20 HTML reports uncovered 27 issues spanning data bugs, missing horseshoe warnings, hardcoded template text, KanFocus data limitations, and analytical improvements. No report mentioned the horseshoe effect despite it distorting all Senate IRT-based interpretations.

## Issue Inventory

### Tranche 1: Critical Bugs (5 issues)

| # | File | Issue | Root Cause | Fix |
|---|------|-------|-----------|-----|
| 1 | `wnominate_report.py` | W-NOMINATE "8460.4%" classification rate | W-NOMINATE CC is 0-100 scale, OC CC is 0-1 scale. `:.1%` multiplies by 100 again. | Normalize W-NOMINATE CC/APRE to 0-1 when value > 1.0. Replace NaN GMP for OC with "N/A". |
| 2 | `prediction_report.py` | Hardcoded "82% Yea base rate" | `_add_how_to_read()` line 191 has hardcoded "82%". Second instance "~73%" at line 782. | Parameterize — pass actual yea rates from results dict. |
| 3 | `pca_report.py` | Scree caption always says "one-dimensional" | `_add_scree_figure()` line 252 has hardcoded caption. | Use eigenvalue ratio to generate data-driven caption. |
| 4 | `irt_2d_report.py` | References "Senator Caryn Tyson" | `_add_interpretation_guide()` line 507 names a legislator from the 91st (2025-26). | Replace with generic language. |
| 5 | `indices_report.py` | Rice "< 0.50" may HTML-escape to nothing | `<` in HTML string. | Use `&lt;` explicitly. |

### Tranche 2: Horseshoe Awareness (6 issues)

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| 6a | `phase_utils.py` | No horseshoe utility function | Add `load_horseshoe_status()` and `horseshoe_warning_html()`. |
| 6b | 8 report builders | No horseshoe warnings anywhere | Each report checks horseshoe status and adds warning banner. |
| 7 | `synthesis_report.py` | "most bipartisan" label inaccurate | Change to "lowest party loyalty in {chamber}". |
| 8 | `synthesis_report.py` | Veto override Rice interpretation hardcoded | Make narrative data-driven based on Rice thresholds. |
| 22 | `irt_report.py` | IRT-PCA correlation caption hardcoded | Use actual r value to generate data-driven caption. |
| 23 | `irt_report.py` | No warning when R mean < D mean | Add key finding warning for party mean inversion. |

### Tranche 3: Template Parameterization (4 issues)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 11 | `clustering.py` | Cluster labels assume no horseshoe | Use party-composition labels when horseshoe detected. |
| 12 | `beta_binomial_report.py` | Key findings only show one chamber | Remove `break` statement at line 446. |
| 13 | `prediction_report.py` | Feature guards missing | Guard "sponsor party" on feature presence. Name bare `vt_` features. |
| 14 | `eda_report.py` | Absenteeism 0% display | Change `:.1%` to `:.2%` for absence rates. Guard strategic ratio near-zero. |

### Tranche 4: KanFocus Graceful Degradation (3 issues)

| # | File(s) | Issue | Fix |
|---|---------|-------|-----|
| 15 | `pca_report.py`, `lca_report.py`, `bipartite_report.py`, `profiles_data.py` | Empty Title columns | Add `drop_empty_optional_columns()` utility. Drop `short_title` when all-null. |
| 16 | `issue_irt.py` | Stub report with methodology-only content | Generate minimal "Phase Skipped" report when no topic data. |
| 17 | District maps | "Trust Notebook" text visible | Jupyter artifact, not visible in standalone browser. Document only. |

### Tranche 5: Analytical Improvements (10 issues)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 18 | `tsa.py` | Drift sign-flip artifacts | Correlate per-window PC1 with full-session PC1 when horseshoe detected. |
| 19 | `mca_report.py` | Absence-dominated dimensions | Add warning when top-5 contributions are all `*__Absent`. |
| 20 | `umap_viz.py` | Senate n_neighbors too high | Cap at `n_members // 3`. Warn when ratio > 25%. |
| 21 | `pca_report.py` | No warning when sensitivity r < 0.95 | Add warning section. |
| 24 | `hierarchical_report.py` | No Stocking-Lord linking disagreement flag | When slope CV > 0.20, warn about cross-chamber reliability. |
| 25 | `mca_report.py` | Cumulative inertia > 100% | Add footnote explaining Greenacre correction artifact. |
| 26 | `profiles_data.py` | Duplicate defection table rows | Deduplicate by `vote_id` not `bill_number`. |
| 27 | `tsa_report.py` | No PELT vs Bai-Perron comparison | Add table showing confirmed vs unconfirmed breaks. |
| 9/10 | — | Horseshoe profile framing | Addressed by Tranche 2 horseshoe banners. |

## Verification Plan

1. `just lint` — must pass
2. `just test` — all ~2962 tests must pass
3. Re-run 79th pipeline: `just pipeline 2001-02`
4. Spot-check 91st pipeline for no regression

## Related Documents

- ADR-0114: Horseshoe-Aware Report System (to be created)
- `docs/79th-horseshoe-robustness-analysis.md` (existing horseshoe analysis)
- `docs/canonical-ideal-points.md` (canonical routing documentation)
