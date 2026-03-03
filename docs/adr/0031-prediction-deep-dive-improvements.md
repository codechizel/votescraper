# ADR-0031: Prediction Deep Dive Improvements

**Date:** 2026-02-25
**Status:** Accepted

## Context

A literature survey and code audit of the prediction phase (Phase 8) identified six improvements ranging from a methodological bug to missing metrics. The deep dive article (`docs/prediction-deep-dive.md`) documents the full analysis.

Key findings:
1. Per-legislator accuracy and surprising votes were evaluated on the full dataset (including training data), inflating accuracy and missing genuinely surprising out-of-sample predictions.
2. `find_surprising_votes()` returned an untyped empty DataFrame when the model made no errors, inconsistent with `find_surprising_bills()` which correctly used a typed schema.
3. Brier score and log-loss (proper scoring rules) were absent from all evaluations, despite being standard in the literature.
4. The IRT circularity concern (ideal points estimated from the same votes being predicted) was not documented in the report or design doc.

## Decision

1. **Holdout-only evaluation:** Per-legislator accuracy and surprising votes are now computed exclusively on the 20% holdout test set. `train_vote_models()` returns `test_indices` so `main()` can slice the features DataFrame.

2. **Empty schema fix:** `find_surprising_votes()` returns a typed empty DataFrame (matching `find_surprising_bills()`).

3. **Proper scoring rules:** Brier score (`brier_score_loss`) and log-loss (`log_loss`) added to:
   - Vote model CV (per fold)
   - Holdout evaluation
   - Passage model CV (per fold)
   - Temporal split evaluation
   - Report tables (vote + passage)

4. **IRT circularity caveat:** Added to the HTML report's vote interpretation section. Explains that the high AUC reflects explanatory power, not true out-of-sample prediction.

5. **Phase number fix:** Docstring updated from "Phase 7" to "Phase 8" to match directory `15_prediction`.

6. **16 new tests** covering: `evaluate_holdout()`, `_compute_day_of_session()`, baselines, proper scoring rules, test indices, surprising votes empty schema, and `_temporal_split_eval()`.

## Consequences

### Output Changes

- **Per-legislator accuracy:** Values may decrease slightly (holdout-only removes in-sample inflation). The number of observations per legislator is ~20% of the previous count.
- **Surprising votes:** May differ from previous runs — now drawn from holdout only (~12K observations instead of ~60K).
- **CV/holdout tables:** Two additional columns (BRIER, LOGLOSS) in the report.
- **Report:** New methodological caveat paragraph in the vote interpretation section.

### Trade-offs

- Holdout-only evaluation reduces the sample size for per-legislator analysis from ~60K to ~12K. Some legislators with few holdout votes will have noisier accuracy estimates. This is the methodologically correct trade-off — in-sample accuracy was optimistically biased.
- Brier score and log-loss add no runtime cost (two function calls per evaluation).
- The circularity caveat is a documentation clarification, not a code change. The fundamental design (IRT features as inputs) remains unchanged — this is standard practice in political science.

### Test Count

958 total tests (previously 942). 54 prediction tests (previously 38).
