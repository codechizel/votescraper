# ADR-0030: Analysis Directory Restructuring

**Date:** 2026-02-25
**Status:** Accepted

## Context

The `analysis/` directory contained 34 Python files at the root level. For an open-source audience, the pipeline ordering (EDA → PCA → UMAP → IRT → ... → External Validation) was implicit — you had to read CLAUDE.md or the design docs to understand which files ran in what order.

Numbered subdirectories (e.g., `01_eda/`, `13_indices/`) make the pipeline ordering self-documenting in a directory listing.

**Problem:** Directory names like `01_eda` are not valid Python identifiers, so `from analysis.01_eda.eda import X` is a syntax error. All existing imports use `from analysis.eda import X`.

## Decision

1. **Numbered subdirectories** under `analysis/` for each pipeline phase, plus an `experimental/` directory.
2. **PEP 302 meta-path finder** in `analysis/__init__.py` transparently redirects `from analysis.eda import X` to `analysis.01_eda.eda`. Zero import changes needed in any Python file.
3. **Shared infrastructure stays at the root:** `run_context.py`, `report.py`, and `design/`.
4. **Files moved via `git mv`** to preserve history.

### Directory layout

```
analysis/
  __init__.py          ← PEP 302 redirect finder
  run_context.py       ← shared infrastructure
  report.py            ← shared infrastructure
  design/              ← statistical design docs
  01_eda/              ← Exploratory Data Analysis
  02_pca/              ← Principal Component Analysis
  04_umap/             ← UMAP dimensionality reduction
  05_irt/              ← Bayesian IRT ideal points
  09_clustering/       ← Voting bloc detection
  11_network/          ← Legislator network analysis
  13_indices/          ← Classical political science indices
  15_prediction/       ← Vote prediction (XGBoost)
  14_beta_binomial/    ← Bayesian party loyalty
  07_hierarchical/     ← Hierarchical IRT
  24_synthesis/        ← Narrative synthesis
  25_profiles/         ← Legislator deep dives
  26_cross_session/    ← Cross-biennium validation
  17_external_validation/ ← Shor-McCarty comparison
  experimental/        ← Experimental scripts
```

### Import redirect mechanism

The `_AnalysisRedirectFinder` in `analysis/__init__.py` intercepts imports of the form `analysis.<name>` where `<name>` is a known module (e.g., `eda`, `indices_report`). It loads the real module from its numbered subdirectory and registers it under the alias name.

## Consequences

**Pros:**
- Pipeline ordering is self-documenting from a directory listing
- All existing imports continue to work unchanged (zero migration cost)
- `git mv` preserves file history
- Each phase is a self-contained package with `__init__.py`

**Cons:**
- The meta-path finder adds a layer of indirection that could confuse IDE navigation (though most IDEs handle this fine after a cache refresh)
- Debugging import errors requires understanding the redirect mechanism
- Adding a new phase requires updating `_MODULE_MAP` in `analysis/__init__.py`

**Migration checklist for new phases:**
1. Create `analysis/NN_name/` with `__init__.py`
2. Add module-to-subdirectory entries in `_MODULE_MAP`
3. Update Justfile recipe path
4. Update `pyproject.toml` `allowed-unresolved-imports` — add **both** bare names (for try/except fallbacks) **and** `analysis.*` qualified names (for the meta-path finder imports that ty cannot follow at static analysis time)
