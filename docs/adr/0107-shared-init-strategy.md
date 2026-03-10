# ADR-0107: Shared MCMC Initialization Strategy

**Date:** 2026-03-10
**Status:** Accepted

## Context

Multiple IRT phases initialize MCMC chains from upstream scores to improve convergence:

- Phase 06 (2D IRT): PCA PC1 → Dim 1, PCA PC2 → Dim 2
- Phase 07 (Hierarchical): PCA PC1 → xi_offset
- Phase 27 (Dynamic): Static IRT → xi_init prior mean

Each phase implemented initialization independently with inline code. The source was always PCA, despite the 1D IRT (Phase 05) producing converged posterior means that are a more direct measure of ideology. For sessions where PCA doesn't separate parties cleanly (e.g., the 79th), this led to poor 2D convergence (Dim 1 vs PCA PC1: r = -0.19).

As the pipeline moves toward Django, initialization strategy needs to be a configurable, storable choice — not hardcoded.

## Decision

Implement `analysis/init_strategy.py` — a shared module following the `IdentificationStrategy` pattern (ADR-0103):

### Strategies

| Strategy | CLI value | Source | When to use |
|----------|-----------|--------|-------------|
| IRT-informed | `irt-informed` | 1D IRT xi_mean (Phase 05) | Best for ideology dimension — converged posterior means |
| PCA-informed | `pca-informed` | PCA PC1/PC2 (Phase 02) | Fallback when IRT unavailable, or for Dim 2 (PC2) |
| Auto | `auto` | Prefer IRT, fall back to PCA | Default — robust across sessions |

### API

```python
from analysis.init_strategy import InitStrategy, resolve_init_source

vals, strategy, source = resolve_init_source(
    strategy="auto",           # or "irt-informed", "pca-informed"
    slugs=data["leg_slugs"],   # model's legislator ordering
    irt_scores=irt_df,         # 1D IRT ideal points (or None)
    pca_scores=pca_df,         # PCA scores (or None)
    pca_column="PC1",          # "PC1" for ideology, "PC2" for secondary
)
```

Returns `(standardized_values, resolved_strategy, source_label)`. Values are zero-mean, unit-variance (matching Normal(0,1) priors).

### CLI

Both Phase 06 and Phase 07 accept `--init-strategy {auto,irt-informed,pca-informed}` (default: `auto`).

### Django-ready

`InitStrategy.CHOICES` provides a `(db_value, display_label)` list ready for `CharField(choices=...)`.

## Consequences

**Benefits:**
- Single source of truth for initialization logic — no duplicated code
- IRT-informed init gives the 2D model a strong ideology starting point, improving convergence for difficult sessions
- Configurable per-run, logged in params, storable as a Django field
- Full rationale logging (which strategy, why, match counts) for reproducibility
- Graceful degradation: auto falls back through IRT → PCA → zeros

**Trade-offs:**
- Adds a dependency ordering: Phase 05 must run before Phase 06 for IRT-informed init (auto handles this gracefully by falling back to PCA)
- Phase 27 (Dynamic IRT) uses a different pattern (informative prior, not initial points) — not yet migrated

**Testing:** 27 tests covering constants, resolution, auto-detection, error cases, rationale generation, and file loading.
