# ADR-0053: nutpie Rust NUTS for all MCMC models

**Date:** 2026-02-28
**Status:** Accepted

## Context

ADR-0051 migrated per-chamber hierarchical IRT to nutpie's Rust NUTS sampler. The joint cross-chamber model and flat IRT remained on PyMC's `pm.sample()`. Both model families were already validated with nutpie:

- **Flat IRT (Experiment 1):** |r|=0.994 vs PyMC, zero divergences, identical ideal points
- **Per-chamber hierarchical (Experiment 2b):** Both chambers pass (House R-hat 1.003/ESS 1204, Senate R-hat 1.001/ESS 1658)

Both remaining models use the same PyTensor ops (Normal, HalfNormal, Bernoulli, set_subtensor, sort) already proven with nutpie's Numba backend. The joint model additionally uses HalfNormal(sigma_chamber), Normal(chamber_offset), and concatenate — all standard ops with Numba support.

The joint model consistently fails convergence on PyMC (R-hat 1.53, ESS=7 across 8 bienniums). nutpie's normalizing flow adaptation was specifically designed for hierarchical models with ~1000 parameters — almost exactly our joint model's parameter count.

## Decision

Migrate all remaining MCMC models to nutpie:

1. **Flat IRT (`analysis/05_irt/irt.py`):** Extract `build_irt_graph()` (model graph only), rewrite `build_and_sample()` to compile+sample with nutpie. PCA init via `initial_points={"xi_free": ...}`, jitter all other RVs.

2. **Joint hierarchical IRT (`analysis/07_hierarchical/hierarchical.py`):** Extract `build_joint_graph()` returning `(pm.Model, combined_data)`, rewrite `build_joint_model()` to compile+sample with nutpie. Adds optional `xi_offset_initvals` parameter for future PCA init.

Implementation mirrors ADR-0051's pattern:
- `callback`, `target_accept`, `cores` parameters accepted but ignored (API compatibility)
- `jitter_rvs` excludes the PCA-initialized variable (avoids HalfNormal log(0)=-inf)
- `store_divergences=True` for diagnostic parity

## Consequences

**Gains:**
- Unified sampler across all models (simpler mental model, one code path)
- Joint model may benefit from nutpie's normalizing flow adaptation (designed for hierarchical models)
- `build_irt_graph()` and `build_joint_graph()` are now importable by experiments
- Single-process Rust threads instead of Python multiprocessing for all MCMC

**Loses:**
- PyMC `callback` parameter no longer functional for any model
- `target_accept` no longer directly controllable (nutpie adaptive dual averaging)
- `cores` parameter no longer meaningful (nutpie manages its own threads)

**Risks:**
- Joint model convergence may still fail (nutpie doesn't guarantee convergence for poorly identified models) — but can't be worse than PyMC's R-hat 1.53
- **Update (ADR-0055):** Joint model now uses `JOINT_BETA` (`lognormal_reparam`) with `alpha_sigma=2.0` and PCA initialization. R-hat(xi) improved to 1.010 on 84th but 828 divergences remain. Stocking-Lord IRT linking added as production alternative for cross-chamber scaling.
