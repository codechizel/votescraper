# ADR-0066: 84th Pipeline Stress Test Fixes

**Date:** 2026-03-01
**Status:** Accepted

## Context

Running the full 17-phase pipeline plus standalone phases (PPC, External Validation, DIME) against the 84th biennium (2011-12) as a stress test after adding Phases 5b, 6b, 4c, 14b, and 17 surfaced several bugs. Most were column-name mismatches between the scraper's CSV schema and assumptions in newer analysis phases.

## Decision

Fix all blocking bugs in a single pass. Eight fixes total:

### Column-Name Mismatches (3 fixes)

The scraper's `legislators.csv` uses `slug` as its column name, but several newer phases assumed `legislator_slug`:

1. **Phase 5b LCA** (`lca.py`): Added `.rename({"slug": "legislator_slug"})` in `load_metadata()`.
2. **Phase 15 TSA** (`tsa.py`): Added conditional rename `if "slug" in legislators.columns`.
3. **Phase 15 TSA** (`tsa.py`): Votes CSV uses `vote` (not `vote_category`). Replaced all 4 occurrences.

### IRT Sensitivity Sign Flip (1 fix)

**Phase 4 IRT** (`irt.py`): `run_sensitivity()` computed raw Pearson r between default and sensitivity IRT runs. Independent IRT runs on different vote subsets can flip sign convention (different anchor selection), yielding r=-0.96 and a false "SENSITIVE" result. Fix: use `abs(raw_correlation)` and align sensitivity scores when negative. Also stores `raw_pearson_r` in findings for transparency.

### arviz API Deprecation (1 fix)

**Phase 4 IRT** (`irt.py`): `idata.posterior.dims.get("chain", 1)` uses the deprecated `Dataset.dims` property (returns a `Frozen` dict). Changed to `idata.posterior.sizes.get("chain", 1)` (the non-deprecated equivalent).

### matplotlib Deprecation (1 fix)

**Phase 5b LCA** (`lca.py`): `boxplot(labels=...)` renamed to `tick_labels=` in matplotlib 3.9.

### DIME CSV Party Column Type (1 fix)

**Phase 18 DIME** (`external_validation_dime_data.py`): The DIME CSV (144 MB) has `party` and `party.orig` columns with mixed types — numeric party codes (100, 200) in most rows, string values ("UNK") in rare rows. Polars `infer_schema_length=10000` inferred `i64`, then failed on "UNK". Fix: `schema_overrides={"party": pl.Utf8, "party.orig": pl.Utf8}`.

### LCA Report Enhancement (1 enhancement)

Added class membership tables to the LCA report — every legislator's class assignment, party, IRT ideal point, and classification certainty (Max P). This makes the LCA report useful for general audiences who want to know *who* is in each group, not just how many.

### Dependency (1 fix)

Added `fastcluster>=1.3` to dev dependencies to eliminate seaborn's runtime warning about slow hierarchical clustering fallback.

## Consequences

- All 17 pipeline phases + PPC + External Validation + DIME complete without errors on the 84th biennium.
- W-NOMINATE (Phase 17) requires R installation — not a code bug, documented as an environment dependency.
- The `slug` vs `legislator_slug` mismatch is a systemic issue: the scraper uses `slug` and many analysis phases assume `legislator_slug`. Each phase handles the rename at load time rather than changing the scraper output (separation of concerns per ADR-0021).
- LCA report now produces 2 additional sections per chamber (class membership tables).
