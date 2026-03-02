# ADR-0072: Full Pipeline Audit — 8-Biennium Review and Fixes

**Date:** 2026-03-02
**Status:** Accepted

## Context

After completing a full pipeline run across all 8 bienniums (84th-91st, 2011-2026), a systematic audit reviewed all 17 phases per biennium plus cross-session and dynamic IRT results. The audit examined convergence diagnostics, statistical validity, data leakage, exception handling, and report completeness. 18 findings were catalogued (A1-A18), prioritized by severity.

## Decision

### 6 fixes implemented immediately

| ID | Category | Fix | Files |
|----|----------|-----|-------|
| A1 | Critical | **Python 2 `except` syntax**: ruff formatter stripped parens from multi-exception `except` clauses. Originally fixed with `# fmt: skip`; now resolved permanently by PEP 758 (Python 3.14) which makes bracketless `except A, B:` valid syntax meaning "catch both." All `# fmt: skip` workarounds removed. | `network.py` (3), `geographic.py` (3), `tsa_r_data.py` (1), `experiment_monitor.py` (1), `dynamic_irt_report.py` (1) |
| A5 | High | **Prediction in-sample leakage**: `find_surprising_bills()` evaluated bill passage surprises on the full dataset instead of holdout test set. Fixed by threading `test_indices` through `train_passage_models()` return value. | `prediction.py` |
| A7 | High | **Prediction minimum sample threshold**: per-legislator accuracy reported legislators with 1-2 votes as "hardest to predict." Added `MIN_VOTES_RELIABLE=10` constant and `reliable` boolean column. | `prediction.py` |
| A12 | Medium | **Beta-binomial clamping warning**: `estimate_beta_params()` silently clamped alpha/beta to 0.5 when method-of-moments produced sub-0.5 values. Now emits `warnings.warn()`. | `beta_binomial.py` |
| A14 | Medium | **TSA imputation sensitivity logging**: `compute_imputation_sensitivity()` silently returned None when insufficient complete cases. Now prints explicit skip message. | `tsa.py` |
| A15 | Medium | **TSA penalty sweep summary**: PELT penalty sensitivity sweep results were only in the HTML report, not the run log. Now prints max changepoints and penalty value. | `tsa.py` |

### 12 findings catalogued for future work

**High — Systematic convergence failures:**
- **A2**: Joint hierarchical model fails all 8 bienniums (256-4,281 divergences). Root cause: ADR-0042 `vote_id` deduplication prevents bill matching. Per-chamber + Stocking-Lord linking works; joint model wastes ~4 min/biennium.
- **A3**: 2D IRT (Phase 04b) Senate ESS catastrophically low (6-52 vs threshold 200) across all bienniums. House marginal. Dim 2 captures noise.
- **A4**: Dynamic IRT Senate still fails (R-hat 1.84, ESS 3) despite ADR-0070 fixes. 87th/88th have sign-flip artifacts.

**Medium — Methodology observations:**
- **A6**: Prediction false-positive asymmetry (90% of surprising votes are FP due to 73% Yea base rate).
- **A8**: LCA degenerate class probabilities in 91st (all max_probability=1.0).
- **A9**: Clustering trivial party split (k=2 = ARI 1.0 across all bienniums).
- **A10**: Network betweenness sparsity (66-73% zeros).
- **A11**: House IRT sensitivity to minority vote threshold.
- **A13**: Hierarchical shrinkage_pct nulls (35% Senate, 23% House) propagate to synthesis gaps.

**Low — Known limitations:**
- **A16**: Small Senate Democrat groups (8-11 members).
- **A17**: BiCM backbone extremely sparse (95% edge reduction in Senate).
- **A18**: Bill communities mirror party split.

## Consequences

- 9 silent exception-handling bugs fixed that could have caused incorrect error handling in production
- Prediction phase now produces valid holdout-only surprise evaluation (eliminates data leakage)
- Per-legislator accuracy reporting is now statistically meaningful (minimum sample filter)
- Beta-binomial and TSA phases produce better diagnostic output for pipeline operators
- 12 remaining items tracked in `docs/roadmap.md` for prioritization
- All 1779 tests pass after fixes
