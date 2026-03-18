# ADR-0119: Pipeline Tuning Centralization

**Date:** 2026-03-18
**Status:** Accepted

## Context

A comprehensive codebase audit revealed that key analysis parameters — vote filtering thresholds, legislator inclusion cutoffs, party colors, discrimination thresholds, and external validation correlation thresholds — were defined independently in up to 26 files. This created three problems:

1. **Tuning friction.** Changing a threshold (e.g., contested vote filter from 2.5% to 10%) required editing 6+ phase files and their corresponding reports. Easy to miss one.
2. **Silent divergence risk.** Independently-defined constants could drift out of sync across phases without detection.
3. **Dashboard blocking.** The planned Django tuning dashboard (post-DB5) needs a single source of truth to read/write parameters.

The audit also found bugs (wrong phase number in Phase 20, incorrect UMAP stability calculation), dead code (`died_count` in bill lifecycle), and duplicated constants in report files (Phases 13, 19).

## Decision

### 1. Create `analysis/tuning.py` as single source of truth

All pipeline-wide tuning parameters live in one file:

| Parameter | Default | Phases Affected | Purpose |
|---|---|---|---|
| `CONTESTED_THRESHOLD` | 0.025 | 01, 02, 03, 05, 06, 09, 10 | Min dissent fraction to include a vote |
| `SENSITIVITY_THRESHOLD` | 0.10 | 02, 03, 05, 09 | Alternative threshold for sensitivity analysis |
| `MIN_VOTES` | 20 | 01, 02, 03, 05, 09, 10, 16 | Min votes per legislator |
| `PARTY_COLORS` | R/D/I hex dict | 28 files | Standard plot color scheme |
| `SUPERMAJORITY_THRESHOLD` | 0.70 | 05, 06, 07b | Adaptive MCMC tuning trigger |
| `HIGH_DISC_THRESHOLD` | 1.5 | 05, 11, 25 | High bill discrimination cutoff |
| `LOW_DISC_THRESHOLD` | 0.5 | 05, 25 | Low bill discrimination cutoff |
| `STRONG_CORRELATION` | 0.90 | 16, 17, 18 | External validation: strong agreement |
| `GOOD_CORRELATION` | 0.85 | 16, 17, 18 | External validation: good agreement |
| `CONCERN_CORRELATION` | 0.70 | 16, 17, 18 | External validation: concern flag |

### 2. Rename `MINORITY_THRESHOLD` → `CONTESTED_THRESHOLD`

The old name was ambiguous — "minority" could refer to the minority party or the minority voting fraction. `CONTESTED_THRESHOLD` is clearer: it's the minimum dissent level for a vote to be considered contested enough for analysis.

### 3. Phase-specific parameters stay in phase files

MCMC parameters (N_TUNE, N_CHAINS), convergence thresholds (R-hat, ESS), and text-based correlation thresholds (Phases 21-22 use intentionally lower values) remain in their phase modules. Only parameters that are shared identically across phases belong in `tuning.py`.

### 4. Fix bugs and dead code discovered during audit

- Phase 20: `print_header("Phase 18 Complete")` → `"Phase 20 Complete"`
- UMAP report: broken `len(pairs)*2 - len(pairs)` → `len(STABILITY_SEEDS)`
- Bill lifecycle: removed dead `died_count` variable
- Phase 13: `indices_report.py` now imports constants from `indices.py`
- Phase 19: `tsa_report.py` now imports constants from `tsa.py`

### 5. Add comments to complex code sections

Six areas identified as non-obvious received explanatory comments: rolling PCA window sizing, R-to-Python index conversion, 2D IRT dimension swap detection, joint sign correction algorithm, bill text source preference ordering, and synthesis paradox detection ranking.

## Consequences

**Positive:**
- Single-line changes in `tuning.py` retune the entire pipeline
- Django dashboard integration is straightforward (read/write one module)
- Zero risk of phase-to-phase constant drift
- Two bugs fixed, dead code removed, 16 duplicate constant definitions eliminated

**Negative:**
- Re-export pattern needed in 4 data modules (wnominate_data, tbip_data, issue_irt_data, viz_helpers) where downstream code imports constants from them
- Phase-specific thresholds still live in their phase files (intentional — centralizing them would obscure the design rationale)

**Migration:**
- All `MINORITY_THRESHOLD` references updated (code, tests, reports, design docs)
- Test suite updated: 2970 tests pass
- No behavioral change — all default values preserved exactly

**Files changed:** ~55 files across analysis phases, reports, tests, docs, and CLAUDE.md.
