# ADR-0124: Exploratory Graph Analysis (EGA) — Network Psychometrics for Dimensionality Assessment

**Date:** 2026-03-25
**Status:** Accepted

## Context

The pipeline's dimensionality assessment relied on a cascade of model-comparison methods: fit 1D IRT, fit 2D IRT, compare convergence diagnostics, cross-validate against W-NOMINATE (ADR-0110, ADR-0123). This cascade is effective but reactive — it determines dimensionality by fitting multiple models and comparing, rather than estimating dimensionality directly from the data.

Hudson Golino's Exploratory Graph Analysis (EGA) framework provides a principled, pre-IRT dimensionality estimate using network psychometrics: build a sparse partial correlation network via GLASSO, detect communities, and count dimensions. EGA outperformed parallel analysis, MAP, and BIC in simulation studies (Golino et al., 2020, Psychological Methods), particularly under high inter-factor correlations — exactly the condition in legislative data.

Several specific gaps motivated adoption:

1. **No conditional dependency network.** Phase 11's Kappa network uses marginal associations; GLASSO isolates conditional dependencies (what IRT's local independence assumption addresses).
2. **No per-item stability assessment.** The 7-gate quality system (ADR-0118) detects instability at the session level but not per bill.
3. **No information-theoretic model comparison.** TEFI (Von Neumann entropy) properly penalizes over-extraction, unlike RMSEA/CFI.
4. **No principled bill redundancy detection.** `--contested-only` uses a blunt dissent threshold; UVA uses topological overlap for structural redundancy.

## Decision

Implement EGA as a Python-native library (`analysis/ega/`) and a new pipeline phase (Phase 02b) between PCA and MCA.

### Library: `analysis/ega/`

Seven modules implementing Golino's framework:

| Module | Purpose |
|--------|---------|
| `tetrachoric.py` | Tetrachoric correlations for binary vote data (MLE on bivariate normal) |
| `glasso.py` | GLASSO + EBIC model selection (100-point lambda sweep, gamma=0.5) |
| `community.py` | Walktrap/Leiden community detection + unidimensional check |
| `ega.py` | Core orchestrator: tetrachoric → GLASSO → community → loadings |
| `boot_ega.py` | Bootstrap stability (parametric/nonparametric, 500 replicates) |
| `tefi.py` | Total Entropy Fit Index (Von Neumann entropy) |
| `uva.py` | Unique Variable Analysis (weighted topological overlap) |

### Phase 02b: `analysis/02b_ega/`

Pipeline position: EDA (01) → PCA (02) → **EGA (02b)** → MCA (03) → ... → IRT (05)

Outputs per chamber:
- Dimensionality estimate (K) with bootstrap confidence
- Per-bill community assignments and item stability
- TEFI comparison across K=1..5
- UVA redundant bill pairs
- GLASSO partial correlation network

### Integration into existing phases

- **Phase 02 (PCA):** TEFI computed from PCA loadings as supplementary metric.
- **Phase 08 (PPC):** Q3 per-bill-pair heatmap and top violation pairs table.
- **Phase 11 (Network):** Residual network (observed Kappa − IRT-predicted Kappa).

### Tuning parameters (in `analysis/tuning.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `EGA_GLASSO_GAMMA` | 0.50 | EBIC sparsity hyperparameter |
| `EGA_BOOT_N` | 500 | Bootstrap replicates |
| `EGA_STABILITY_THRESHOLD` | 0.70 | Item stability cutoff |
| `UVA_WTO_THRESHOLD` | 0.25 | Redundancy detection cutoff |

## Consequences

**Benefits:**
- Pre-IRT dimensionality estimate (seconds, not MCMC hours).
- Per-bill stability identifies which bills cause dimensional instability.
- TEFI provides objective model comparison without convergence dependence.
- Residual network bridges the IRT/network analysis gap.
- All Python-native — no R dependency.

**Costs:**
- ~2-5 minutes per chamber for full EGA + bootEGA.
- Tetrachoric estimation is O(p²) — scales quadratically with bill count.
- Senate chambers (N~40) may produce very sparse GLASSO networks. Fixed in ADR-0126 (direct covariance + fragmentation guard).

**Advisory, not authoritative.** EGA does not replace canonical routing (ADR-0109/0110). It provides an independent dimensionality signal. Canonical routing remains the final arbiter for 1D vs 2D selection.

**No new dependencies.** All building blocks (scikit-learn GLASSO, igraph, leidenalg, numpy, scipy) were already in `pyproject.toml`.

**References:**
- Golino, H., & Epskamp, S. (2017). Exploratory graph analysis. PLoS ONE, 12(6).
- Golino, H., et al. (2020). Investigating the performance of EGA. Psychological Methods, 25(3).
- Christensen, A. P., & Golino, H. (2021). Estimating stability via bootEGA. Psych, 3(3).
- Golino, H., et al. (2021). Entropy fit indices. Multivariate Behavioral Research, 56(6).
- Christensen, A. P., et al. (2023). Unique variable analysis. Multivariate Behavioral Research, 58(6).
