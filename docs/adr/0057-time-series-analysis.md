# ADR-0057: Time Series Analysis (Rolling PCA Drift + PELT Changepoints)

**Date:** 2026-02-28
**Status:** Accepted

## Context

The pipeline through Phase 14 treats each legislative session as a static snapshot — one set of scores per legislator for the full biennium. But legislators can shift positions during a session, and party cohesion can change abruptly around events like veto overrides or leadership changes. We needed a temporal analysis layer to detect:

1. **Ideological drift** — did any legislator's voting position change between the first and second half of the session?
2. **Structural breaks in party cohesion** — were there moments when the Rice Index (party unity) shifted abruptly?

Several approaches were considered for each component.

## Decision

### Drift: Rolling-window PCA over dynamic IRT

We chose rolling-window PCA (sliding a window of 75 roll calls with step size 15) rather than dynamic IRT (e.g., Martin-Quinn scores with state-space models). Rationale:

- **Speed**: Rolling PCA runs in seconds; dynamic IRT takes hours and requires careful MCMC diagnostics.
- **Correlation**: PC1 correlates r > 0.95 with IRT ideal points in our data. For temporal tracking (relative change, not absolute position), PCA is sufficient.
- **Simplicity**: No convergence concerns, no priors to tune, no identification constraints.
- **Existing validation**: Our PCA implementation is already validated in Phase 02 with sign convention, imputation, and holdout checks.

### Changepoints: PELT over BOCPD

We chose PELT (Pruned Exact Linear Time) from the `ruptures` library rather than Bayesian Online Changepoint Detection (BOCPD). Rationale:

- **Offline data**: The full session is already scraped; we don't need online detection. PELT is optimal for offline segmentation.
- **Proven library**: `ruptures` is well-tested, actively maintained, and provides multiple kernels and penalty methods.
- **RBF kernel**: Detects changes in both mean and variance of the Rice Index, appropriate for bounded [0, 1] data.
- **Sensitivity analysis**: PELT's penalty parameter is easily swept to assess robustness. Flat regions in the penalty sensitivity plot indicate robust changepoints.

### Weekly Rice aggregation over daily

The Kansas Legislature averages ~2 roll calls per day, making daily Rice Index extremely noisy. Weekly aggregation produces ~14 observations per data point and stable estimates. Self-contained computation (recomputes Rice from raw votes rather than depending on Phase 07) ensures reproducibility.

### Joint multivariate detection

In addition to per-party changepoint detection, we run a 2D joint analysis stacking both parties' weekly Rice series. Joint changepoints — breaks affecting both parties simultaneously — often correspond to session-wide events rather than party-specific dynamics.

### Language policy update

Updated the project's technology preference from "Python-only, no rpy2" to "Python-first, best tool wins." The project already uses Rust (nutpie) for MCMC sampling; this formalizes the pragmatic approach. Historical deep dive docs that discuss the original policy are left as-is (they describe decisions made at the time).

## Consequences

**Positive:**
- Temporal dynamics are now visible — the pipeline can detect legislators who changed position and moments of cohesion breakdown.
- Veto override cross-referencing provides automatic contextual annotation for detected changepoints.
- Rolling PCA's speed (seconds, not hours) makes it practical for iterative exploration and sensitivity analysis.
- 64 tests cover all core functions with synthetic data.

**Negative:**
- PCA captures only the dominant dimension; issue-space changes across windows can shift what PC1 means. Sign convention (Republicans = positive) is enforced per window to maintain consistency.
- PELT's penalty parameter requires judgment. The sensitivity analysis mitigates this by showing which changepoints are robust.
- Weekly aggregation smooths over within-week variation. Alternative windows (3-day, 10-day) could be explored.

**Files created:**
- `analysis/19_tsa/tsa.py` — Main script (~1464 lines)
- `analysis/19_tsa/tsa_report.py` — HTML report builder
- `analysis/design/tsa.md` — Design document
- `tests/test_tsa.py` — 64 tests

**Files modified:**
- `pyproject.toml` — Added `ruptures>=1.1`, ty config for unresolved imports
- `analysis/__init__.py` — Registered `tsa` and `tsa_report` in module map
- `Justfile` — Added `tsa` recipe and pipeline integration
- `.claude/rules/analysis-framework.md` — Updated pipeline, design doc index, language policy
- `CLAUDE.md` — Updated recipe listing, phase count, test count
- `docs/roadmap.md` — Moved TSA to completed phases
- `docs/analysis-primer.md` — Added Step 15
- `.claude/rules/testing.md` — Added test_tsa.py to inventory
