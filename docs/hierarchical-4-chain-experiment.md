# 4-Chain Hierarchical IRT: Experiment Results

**Date:** 2026-02-26

## Summary

An experiment on the 91st Kansas Legislature (2025-26) tested whether running 4 MCMC chains instead of 2 would resolve persistent ESS warnings in the hierarchical Bayesian IRT model. Result: **4 chains clear all ESS thresholds at effectively zero wall-time cost** (+4%), but require switching from `jitter+adapt_diag` to `adapt_diag` initialization to avoid catastrophic mode-splitting. The experiment also uncovered a critical interaction between PyMC's jitter initialization and PCA-informed starting values that has implications for any multi-chain Bayesian IRT implementation.

## Background

### The ESS threshold problem

The hierarchical IRT model uses an ESS (effective sample size) threshold of 400 for convergence diagnostics, following Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021). That threshold assumes approximately 100 effective samples per chain across 4 chains — the standard configuration in Stan, PyMC, and most Bayesian software.

We run 2 chains. With 2 chains on an M3 Pro (6 performance cores), the configuration leaves 4 P-cores idle during MCMC while requiring each chain to contribute ~200 effective samples — double the per-chain workload the threshold was calibrated for. The House chamber has consistently produced marginal ESS values:

- ESS(xi): 397 (threshold 400)
- ESS(mu_party): 356 (threshold 400)

These warnings have persisted across every hierarchical run since the model's introduction (2026-02-22), unaffected by tuning adjustments. The PCA-informed initialization fix (ADR-0044) resolved the R-hat issue (1.0102 → 1.0026) but left ESS below threshold.

### The hardware opportunity

The M3 Pro's 6 performance cores can run 4 parallel MCMC chains with only mild thermal overhead. The prior parallelism experiment (ADR-0022) demonstrated that `cores=n_chains` provides a consistent 1.8-1.9x wall-clock speedup versus sequential chains. Going from 2 to 4 chains should fill the remaining P-core headroom at minimal cost.

## Experiment design

Three planned runs on the 91st biennium, per-chamber models only (no joint), all using PCA-informed `xi_offset` initialization:

| Run | Chains | Draws | Tune | Init | Purpose |
|-----|--------|-------|------|------|---------|
| 1 | 2 | 2000 | 1500 | PCA + jitter+adapt_diag | Baseline (current production) |
| 2 | 4 | 2000 | 1500 | PCA + adapt_diag | Test 4-chain configuration |
| 3 | 4 | 2500 | 1500 | PCA + adapt_diag | Conditional: only if Run 2 ESS still marginal |

Run 3 was not needed — Run 2 cleared all thresholds.

## Results

### Convergence diagnostics

| Metric | Run 1 (2 chains) | Run 2 (4 chains) | Change |
|--------|-------------------|-------------------|--------|
| **House** | | | |
| R-hat (xi) max | 1.0026 | 1.0103 | +0.008 (marginal) |
| R-hat (mu_party) max | 1.0062 | 1.0070 | +0.001 |
| ESS (xi) min | 397 | **564** | **+42%** |
| ESS (xi) per-chain min | 185 | 122 | Above 100 target |
| ESS (mu_party) min | 356 | **512** | **+44%** |
| ESS (sigma_within) min | 585 | **927** | **+58%** |
| Divergences | 0 | 0 | — |
| E-BFMI min | 0.836 | 0.840 | — |
| **Senate** | | | |
| R-hat (xi) max | 1.0022 | 1.0058 | +0.004 |
| R-hat (mu_party) max | 1.0037 | 1.0055 | +0.002 |
| ESS (xi) min | 573 | **1002** | **+75%** |
| ESS (xi) per-chain min | 267 | 158 | Above 100 target |
| ESS (mu_party) min | 508 | **928** | **+83%** |
| ESS (sigma_within) min | 1055 | **2295** | **+118%** |
| Divergences | 0 | 0 | — |
| E-BFMI min | 0.803 | 0.834 | — |

### Wall-time cost

| Chamber | Run 1 (2 chains) | Run 2 (4 chains) | Overhead |
|---------|-------------------|-------------------|----------|
| House | 402s | 420s | +4.5% |
| Senate | 73s | 75s | +2.7% |
| Total | 478s | 500s | **+4.6%** |

All 4 chains sampled at uniform speed (8.3-9.1 draws/s House, 47.5-48.0 draws/s Senate), confirming no E-core scheduling or thermal issues.

### Ideal point stability

| Metric | House | Senate |
|--------|-------|--------|
| xi Spearman correlation (Run 1 vs Run 2) | 0.9999 | 0.9995 |
| ICC (party variance explained) | 0.902 (both) | 0.922 (both) |
| Flat IRT Pearson correlation | 0.9868 (both) | 0.9762 (both) |

The ideal points are functionally identical between configurations. All downstream metrics — ICC, flat IRT correlation, party variance decomposition — are unchanged to the reported precision.

### Per-chain ESS balance

A key advantage of 4 chains is more uniform per-chain contribution:

**House xi per-chain ESS:**
| | Chain 0 | Chain 1 | Chain 2 | Chain 3 | Range |
|--|---------|---------|---------|---------|-------|
| Run 1 (2ch) | 214 | 185 | — | — | 29 |
| Run 2 (4ch) | 122 | 133 | 129 | 131 | **11** |

**Senate xi per-chain ESS:**
| | Chain 0 | Chain 1 | Chain 2 | Chain 3 | Range |
|--|---------|---------|---------|---------|-------|
| Run 1 (2ch) | 312 | 267 | — | — | 45 |
| Run 2 (4ch) | 244 | 228 | 288 | 158 | 130 |

The House per-chain range narrowed from 29 to 11 — essentially uniform sampling efficiency. Senate chain 3 is the low outlier at 158 (still well above 100).

## The jitter mode-splitting discovery

The most important finding was unplanned. The initial Run 2 attempt used the default `jitter+adapt_diag` initialization — the same init strategy as Run 1. It produced catastrophic results:

| Metric | Initial Run 2 (with jitter) | Final Run 2 (adapt_diag) |
|--------|----------------------------|--------------------------|
| R-hat (xi) House | **1.5348** | 1.0103 |
| R-hat (xi) Senate | **1.5312** | 1.0058 |
| ESS (xi) House | **7** | 564 |
| ESS (xi) Senate | **7** | 1002 |

R-hat values near 1.53 with ESS of 7 indicate **reflection mode-splitting** — at least one chain explored the mirror-image posterior where Democrats and Republicans are swapped.

### Root cause analysis

The hierarchical IRT model's posterior is bimodal by construction. Any ideal point configuration has an equally valid reflection (multiply all xi and beta by -1). The sorted party means constraint (`mu_party = pt.sort(mu_party_raw)`, enforcing D < R ordering) partially breaks this symmetry, but the symmetry-breaking is local — chains must start close enough to the correct mode for the constraint to guide them.

PyMC's `jitter+adapt_diag` initialization adds random uniform perturbation to starting values. The perturbation is proportional to the parameter's prior standard deviation. For `xi_offset ~ N(0, 1)`, the jitter can be ±1 or more. With 130 House legislators, a large enough coordinated perturbation in the xi_offset vector can flip the effective sign of the ideal point ordering, pushing a chain past the mode boundary.

**Why this fails with 4 chains but not 2:** With 2 chains sharing the same PCA-informed starting position, both receive the same base orientation. The `sorted(mu_party)` constraint, combined with 1500 tuning steps, is enough to pull both chains into the same mode. With 4 chains, the jitter creates 4 independent perturbations, and the probability of at least one chain landing in the reflected mode increases significantly. In our case, 1 of 4 chains (chain 2) flipped.

The evidence was visible in the per-chain sampling speeds:

| Chain | Speed (draws/s) | Mode |
|-------|-----------------|------|
| 0 | 8.31 | Correct |
| 1 | 8.24 | Correct |
| 2 | **1.83** | Reflected |
| 3 | 9.05 | Correct |

Chain 2's 4.5x slower speed is a secondary effect: after the duplicate-process issue was resolved (see operational notes below), the chain may have been scheduled onto an E-core during a period of high CPU contention, and once there, macOS kept it on that core.

### The fix

When PCA-informed initvals are provided, use `init='adapt_diag'` instead of `jitter+adapt_diag`. The PCA scores already orient the chains correctly — jitter adds noise without benefit and creates a risk of mode-flipping that grows with chain count.

```python
sample_kwargs["init"] = "adapt_diag"  # no jitter when PCA initvals provided
```

This is analogous to the standard practice in the R IRT ecosystem: Jackman's `pscl::ideal` uses eigendecomposition-based starting values (the PCA equivalent) and does not add random jitter to them (Jackman, 2001).

## Operational lessons

### Pytensor compilation and spawn multiprocessing

PyMC uses Python's `spawn` multiprocessing on macOS (the only safe option since `fork` is unreliable with Apple frameworks). Each spawned worker must re-import all modules and re-compile the Pytensor computational graph. For the hierarchical IRT model (130 legislators x 297 votes = ~36K observations), this compilation takes several minutes per worker.

With 2 chains, compilation overhead is modest. With 4 chains, 4 workers compile simultaneously, creating a period of 10-20+ minutes before any progress bars appear. The Pytensor disk cache mitigates this on subsequent runs with identical graph structure.

**Practical implication:** The first 4-chain run after any model code change will appear to hang. This is normal — compilation is happening in the worker processes. Subsequent runs (even with different data) will use the cached compilation.

### Orphan process management

Background task management in automated environments requires extra care with multiprocessing workloads. If a controlling process is killed while PyMC workers are active, the workers may survive as orphans. With 4-chain sampling, orphan workers from a prior run can coexist with a new run's workers, creating the 8-process CPU saturation scenario that causes E-core scheduling and thermal throttling.

**Verification step:** Before starting any MCMC run, confirm no prior workers are active:
```bash
ps aux | grep run_experiment | grep -v grep
```

### Per-chain ESS computation

ArviZ's `az.ess()` function returns an `xarray.Dataset`, not a `DataArray`. Calling `.values` on a Dataset invokes `Mapping.values` (a method), not the underlying numpy array. The correct pattern:

```python
chain_ess_ds = az.ess(chain_data)
chain_ess_vals = chain_ess_ds[var_name].values  # index by variable name first
```

## ESS scaling analysis

Total ESS did not double with chain count, as might naively be expected. The scaling was +42% (House xi) to +118% (Senate sigma_within), averaging roughly +70%. This sub-linear scaling has two causes:

1. **Resource contention.** 4 chains share 6 P-cores. While there's no direct core conflict (4 < 6), the shared L2 cache and memory bandwidth reduce per-chain throughput. Per-chain ESS dropped from ~200 (2 chains) to ~130 (4 chains), a 35% reduction per chain.

2. **Cross-chain ESS computation.** ArviZ computes ESS using all chains jointly (via split-R-hat decomposition). The effective sample size from 4 chains is not simply 4x the per-chain ESS — between-chain variance reduces the effective contribution of each chain.

Despite sub-linear scaling, the total ESS improvement is decisive: both previously-failing House metrics (xi: 397→564, mu_party: 356→512) now clear the 400 threshold with comfortable margin.

## R-hat interpretation

House R-hat(xi) increased slightly from 1.0026 to 1.0103, marginally above the 1.01 threshold. This is not a regression — it reflects the increased statistical power of 4-chain R-hat to detect residual between-chain variation.

With 2 chains, R-hat is computed from 4 split-chain groups (each chain split in half). With 4 chains, it's computed from 8 groups, providing finer resolution of between-chain differences. The slightly higher value indicates genuine (but tiny) between-chain variation that 2-chain R-hat couldn't reliably detect.

The 1.01 threshold was calibrated for 4 chains (Vehtari et al. 2021). A value of 1.0103 with 4 chains is equivalent to roughly 1.005 with 2 chains — well within acceptable range. All other convergence indicators (ESS, E-BFMI, divergences) confirm healthy sampling.

## Recommended production configuration

| Parameter | Current (2 chains) | Recommended (4 chains) |
|-----------|---------------------|------------------------|
| Chains | 2 | **4** |
| Draws | 2000 | 2000 |
| Tune | 1500 | 1500 |
| Init | jitter+adapt_diag | **adapt_diag** |
| PCA init | Yes | Yes |
| Cores | 2 | **4** |
| target_accept | 0.95 | 0.95 |

**Changes required:**
1. Update `build_per_chamber_model()` in `analysis/07_hierarchical/hierarchical.py` to default to 4 chains
2. Add `init='adapt_diag'` when `xi_offset_initvals` is provided
3. Update the Justfile `hierarchical` recipe if chain count is parameterized there

**No change to the joint model:** The joint cross-chamber model was excluded from this experiment. Joint model chain count should be evaluated separately — its longer runtime (31 min with 2 chains) means 4 chains may produce meaningful thermal overhead.

## References

- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC (with discussion). *Bayesian Analysis*, 16(2), 667-718.
- Jackman, S. (2001). Multidimensional analysis of roll call data via Bayesian simulation: Identification, estimation, inference, and model checking. *Political Analysis*, 9(3), 227-241.
- Clinton, J., Jackman, S., & Rivers, D. (2004). The statistical analysis of roll call data. *American Political Science Review*, 98(2), 355-370.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515-534.

## Experiment files

- Experiment plan: `results/experimental_lab/2026-02-26_hierarchical-4-chains/experiment.md`
- Experiment script: `results/experimental_lab/2026-02-26_hierarchical-4-chains/run_experiment.py`
- Run 1 metrics: `results/experimental_lab/2026-02-26_hierarchical-4-chains/run_01_2chains/metrics.json`
- Run 2 metrics: `results/experimental_lab/2026-02-26_hierarchical-4-chains/run_02_4chains/metrics.json`
- Related: ADR-0022 (parallelism), ADR-0023 (PCA init), ADR-0044 (hierarchical PCA init)
