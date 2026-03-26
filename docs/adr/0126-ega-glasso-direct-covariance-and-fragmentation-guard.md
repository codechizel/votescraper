# ADR-0126: EGA GLASSO Direct Covariance and Fragmentation Guard

**Date:** 2026-03-25
**Status:** Accepted

## Context

Phase 02b EGA (ADR-0124) produced pathological results on the 78th Senate (1999-2000): K=196 communities from 226 bills. Two compounding issues:

### 1. Synthetic data inflated effective sample size

The GLASSO module (`analysis/ega/glasso.py`) used sklearn's `GraphicalLasso` class, which requires raw data. Since EGA starts from a tetrachoric correlation matrix (no raw data), the code generated synthetic data via Cholesky decomposition:

```python
z = rng.standard_normal((max(n_obs, p + 1), p))
synth_data = z @ sqrt_cov.T
model.fit(synth_data)
```

For the 78th Senate (40 legislators, 226 bills), `max(40, 227) = 227` synthetic observations — nearly 6x the real sample size. This gave GLASSO a much cleaner covariance estimate than the real 40-observation data supported, producing a denser partial correlation network than warranted.

### 2. Walktrap fragmented sparse, disconnected networks

The resulting GLASSO network had 377 edges across 226 nodes (avg degree ~3.3). Many nodes were in disconnected components. Walktrap assigns each disconnected component to its own community at minimum, producing K=196 (most bills in singletons or tiny isolated cliques).

The unidimensional check (Louvain on the full zero-order correlation matrix) did not override because it found K > 1 on the dense correlation matrix. So K=196 was reported as the dimensionality estimate — a meaningless result.

## Decision

### Fix 1: Use `graphical_lasso()` function directly

Replace `GraphicalLasso` (class, requires raw data) with `graphical_lasso()` (function, accepts empirical covariance matrix directly):

```python
from sklearn.covariance import graphical_lasso

_, precision = graphical_lasso(emp_cov, alpha=float(lam), max_iter=200, tol=1e-4)
```

This eliminates the synthetic data generation entirely. The GLASSO optimization operates on the actual tetrachoric correlation matrix, and EBIC model selection uses the real `n_obs` for its penalty term. No sample size inflation.

### Fix 2: Fragmentation guard in community detection

When Walktrap or Leiden produces K > p/4 (or K > 10, whichever is larger), the network is too fragmented for meaningful community detection. The guard:

1. Detects fragmentation: K > max(p/4, 10)
2. Finds connected components in the GLASSO network
3. Re-runs community detection on only the largest connected component
4. Assigns all other nodes to a catch-all community
5. If even the largest component is fragmented, falls back to K=1 (unidimensional)

The `CommunityResult` dataclass gains a `fragmented: bool` field so downstream phases can see when the guard activated.

## Consequences

**Fixes:**
- 78th Senate (and other small-N chambers) will get meaningful K estimates instead of K ≈ p.
- GLASSO regularization strength now properly reflects the actual sample size.
- Fragmentation guard provides graceful degradation for any chamber where the GLASSO network is too sparse.

**Performance:**
- GLASSO fitting is ~2-3x faster (no synthetic data generation, no eigendecomposition per lambda).

**Behavioral changes:**
- GLASSO results will differ from pre-fix runs because the precision matrix is estimated from the actual covariance rather than from synthetic data. Denser-n chambers (House, n=125) will see minimal change; sparse-n chambers (Senate, n=40) will see sparser networks.
- Community detection may produce different K values. This is expected and correct — the old K values for small chambers were unreliable.

**No downstream impact:** EGA is advisory (not used in canonical routing). All downstream phases continue to use canonical IRT routing (ADR-0109/0110/0123).
