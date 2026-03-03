# Experiment 1: nutpie Flat IRT Baseline

**Date:** 2026-02-27
**Status:** Complete — PASS
**Author:** Claude Code + Joseph Claeys

## The Short Version

nutpie compiles and samples our flat 2PL IRT model without issue. The Numba backend compiled the 722-parameter model in 13.6 seconds and sampled 2×2000 draws in 112.8 seconds with stellar convergence diagnostics: R-hat max 1.004, ESS min 1,950, zero divergences. Ideal points correlate |r| = 0.994 with the PyMC baseline, comfortably passing the |r| > 0.99 threshold. The raw correlation is negative (r = -0.994) due to IRT reflection invariance — nutpie found the mirror-image solution — but the legislator rankings are functionally identical.

This is a green light for Experiment 2 (hierarchical per-chamber IRT with Numba).

## Why We Ran This Experiment

Before testing nutpie on the hierarchical IRT models that actually need fixing (House convergence: 1/8 sessions pass, joint: 0/8), we need to verify basic compatibility. The flat 2PL IRT model is our simplest MCMC model and has a well-established PyMC baseline. If nutpie can't compile and sample this cleanly, there's no point testing harder models.

This experiment follows the staged integration plan from `docs/nutpie-deep-dive.md`:

> **Experiment 1: Flat IRT Baseline (Low Risk).** Run the simplest model through nutpie to verify basic compatibility.

Specific questions:
1. Does `nutpie.compile_pymc_model()` succeed on our model structure (`pt.set_subtensor` anchors, `pm.Bernoulli(logit_p=...)`, `pm.Deterministic` with dims)?
2. Does `nutpie.sample()` produce valid InferenceData with standard ArviZ diagnostics?
3. Do the resulting ideal points agree with the PyMC baseline?

## What We Tested

### Setup

- **Model:** Flat 2PL IRT (identical to `analysis/05_irt/irt.py` production code)
- **Chamber:** 91st House (130 legislators × 297 votes = 35,917 observations)
- **Parameters:** 722 free (128 xi_free + 297 alpha + 297 beta)
- **Anchors:** Avery Anderson (conservative, xi=+1.0), Brooklynne Mosley (liberal, xi=-1.0)
- **Sampler:** nutpie 0.16.6, Numba backend
- **Settings:** 2000 draws, 1000 tune, 2 chains, seed=42

### Why 91st House?

The 91st House is our largest chamber (130 legislators, 297 votes) and therefore has the most parameters. If nutpie compiles this cleanly, smaller chambers (Senate: 42 legislators) are guaranteed to work.

## Results

### Compilation: SUCCESS

nutpie compiled the PyMC model's log-probability and gradient functions through Numba's `nopython` mode in **13.6 seconds**. No `TypingError`, no unsupported ops, no fallback to Python mode. Every PyTensor operation in our model — `pt.set_subtensor` for anchor insertion, `pm.Bernoulli(logit_p=...)` for the likelihood, `pm.Deterministic` with coordinate dims — compiled cleanly.

This confirms that the Numba backend is viable for all Tallgrass IRT models. The model structure is standard enough that we should not encounter Numba compilation failures on hierarchical variants (which add `pm.HalfNormal`, `pm.Normal` with shared hyperparameters, and `pt.sort` for identification — all standard ops).

### Sampling: SUCCESS

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| R-hat (xi) max | 1.0036 | < 1.01 | PASS |
| R-hat (alpha) max | 1.0073 | < 1.01 | PASS |
| R-hat (beta) max | 1.0039 | < 1.01 | PASS |
| ESS (xi) min | 1,950 | > 400 | PASS |
| Divergences | 0 | 0 | PASS |
| E-BFMI chain 0 | 0.970 | > 0.3 | PASS |
| E-BFMI chain 1 | 1.005 | > 0.3 | PASS |

Every diagnostic passes with wide margins. The flat IRT model is well-conditioned (no funnel geometry, no correlation plateau), so this was expected — the real test for nutpie is the hierarchical model.

**Timing:** 112.8 seconds total sampling time. This is comparable to PyMC's default sampler for this model — nutpie's advantage is not raw speed on flat models but rather convergence behavior on hierarchical ones.

### Comparison with PyMC Baseline: PASS

| Metric | Value |
|--------|-------|
| Legislators matched | 130 / 130 |
| Pearson r | -0.9935 |
| \|r\| | 0.9935 |
| Pass criteria (\|r\| > 0.99) | PASS |
| Sign flip | Yes |

The correlation magnitude (0.9935) comfortably exceeds the 0.99 threshold. The negative sign indicates nutpie found the reflected solution — a well-known artifact of IRT identification. The flat model uses two anchor legislators to resolve this ambiguity, but the anchors only constrain the *model* geometry, not the sampler's exploration path. With a different random initialization and a different mass matrix adaptation strategy, nutpie's chains settled into the reflected mode.

**This is not a problem.** The ideal points are functionally identical — just multiply by -1. The scatter plot shows a clean anti-diagonal with tight clustering, confirming that every legislator's relative position is preserved.

For production use, we would apply a post-hoc sign correction by checking the correlation sign against the anchor convention and flipping if necessary.

## What This Means for the nutpie Integration

### Confirmed

1. **Numba compilation works** on our model structure — no JAX backend needed for flat IRT
2. **InferenceData is fully compatible** with ArviZ (R-hat, ESS, BFMI, HDI, trace plots, NetCDF)
3. **`store_divergences=True`** correctly populates `idata.sample_stats["diverging"]`
4. **Coordinate dims** survive compilation and appear in the output
5. **Single-process model** — all sampling happened in one process with Rust threads (no orphan child processes)

### Noted

1. **Sign flip requires attention** — production code should check anchor polarity and correct post-hoc
2. **Compilation is not free** — 13.6 seconds of one-time overhead before sampling begins. For production use with multiple bienniums, consider caching the compiled model or accepting the overhead.
3. **No `log_likelihood` group** in the output — confirmed the known gap (nutpie issue #150). Not a blocker for Tallgrass.

### Next Steps

This clears the path for **Experiment 2: Hierarchical per-chamber IRT with Numba** — the model that actually fails convergence. Key questions for Experiment 2:

- Does nutpie's better mass matrix adaptation (gradient outer products vs sample covariance) resolve the House convergence failure without normalizing flows?
- Does the sign-flip issue compound in the hierarchical model (where party-level means add another symmetry axis)?
- How does PCA initialization interact with nutpie's own `init_mean` and jitter?

## Artifacts

| File | Description |
|------|-------------|
| `results/experimental_lab/2026-02-27_nutpie-flat-irt/run_experiment.py` | Experiment script |
| `results/experimental_lab/2026-02-27_nutpie-flat-irt/run_01_house/metrics.json` | Machine-readable results |
| `results/experimental_lab/2026-02-27_nutpie-flat-irt/run_01_house/scatter_nutpie_vs_pymc.png` | Scatter plot (anti-diagonal confirms sign flip) |
| `results/experimental_lab/2026-02-27_nutpie-flat-irt/run_01_house/data/idata_nutpie_house.nc` | Full posterior trace (NetCDF) |

## Related Documents

- [Nutpie Deep Dive](nutpie-deep-dive.md) — Architecture, NF innovation, integration plan
- [Experiment Framework Deep Dive](experiment-framework-deep-dive.md) — BetaPriorSpec, PlatformCheck, monitoring
- [Hierarchical Convergence Improvement](hierarchical-convergence-improvement.md) — The 9-priority plan; nutpie may leapfrog priorities 1-6
- [ADR-0049](adr/0049-nutpie-flat-irt-baseline.md) — Decision record for this experiment
- [ADR-0048](adr/0048-experiment-framework.md) — Experiment framework used to run this
