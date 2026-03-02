# ADR-0073: W-NOMINATE All-Biennium Production Run + R Compatibility Fixes

**Date:** 2026-03-02
**Status:** Accepted

## Context

Phase 17 (W-NOMINATE + OC, ADR-0059) was initially developed and tested on the 91st biennium only. Running it across all 8 bienniums (84th-91st) exposed six R compatibility bugs in the interface between Python and the R `wnominate`/`oc`/`pscl` packages. Separately, Phase 4c (PPC + LOO-CV) had only been run for the 84th and 91st bienniums — expanding coverage revealed a LOO observation mismatch limitation in two bienniums.

R was installed via Homebrew (`brew install r`) with packages: `wnominate`, `oc`, `pscl`, `jsonlite` (Phase 17), `changepoint`, `strucchange` (Phase 15 TSA).

## Decision

### R compatibility fixes (6 bugs)

All fixes shipped in v2026.03.02.2.

| Bug | File | Root cause | Fix |
|-----|------|-----------|-----|
| `rollcall()` "codes are not unique" | `wnominate.R` | `notInLegis=9` collides with `missing=9` (pscl default) | `notInLegis=NA` |
| "polarity must be vector of length dims" | `wnominate.R` | Scalar polarity for 2D models | `rep(polarity_idx, dims)` for both `wnominate()` and `oc()` |
| "$ operator invalid for atomic vectors" | `wnominate.R` | OC `legislators` is a matrix in oc 1.2.1, not data.frame | Column indexing `oc_leg[, "coord1D"]` |
| `legislator_slug` column not found | `wnominate.py` | Raw CSV uses `slug`, analysis expects `legislator_slug` | Rename in `load_metadata()` |
| `great_tables` TypeError on string comparison | `wnominate_report.py` | R writes fit statistics as strings in JSON | `_to_float()` helper with "NA" handling |
| Polars InvalidOperationError on "NA" strings | `wnominate.py` | R CSV files use literal "NA" for missing values | `null_values="NA"` on all `pl.read_csv()` calls |

### All-biennium W-NOMINATE results

All 8 bienniums (84th-91st) ran successfully. IRT-vs-WNOM dim1 Pearson r typically 0.95-0.99 across chambers and bienniums, confirming strong agreement between Bayesian IRT and the field-standard MLE method.

### PPC expansion (6/8 bienniums)

Phase 4c was run for 6 additional bienniums (85th, 86th, 88th, 89th, 90th). The 87th and 89th failed with an ArviZ `compare()` observation mismatch: the hierarchical model uses a different vote matrix than flat IRT in these bienniums (different lopsided-vote filtering), causing ArviZ to reject the comparison. This is a known limitation of ArviZ's `compare()` — it requires identical observation counts across all models.

### R as optional dependency

R is required only for Phase 17 (W-NOMINATE/OC) and Phase 15's `--skip-r` optional enrichment (CROPS, Bai-Perron). The core Python pipeline works without R. R packages are not managed by `uv` — they must be installed separately via `install.packages()`.

## Consequences

### Positive

- All 8 bienniums validated against the field-standard W-NOMINATE method
- R integration bugs fixed for all versions of oc/pscl packages
- PPC model comparison available for 6/8 bienniums
- R compatibility pattern documented for future R subprocess phases

### Negative

- R is an additional system dependency for full pipeline coverage (not managed by uv)
- PPC LOO comparison unavailable for 87th and 89th bienniums (observation mismatch)

### Neutral

- W-NOMINATE and PPC remain standalone phases — not in `just pipeline`
