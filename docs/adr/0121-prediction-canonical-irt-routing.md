# ADR-0121: Prediction phase canonical IRT routing

**Date:** 2026-03-25
**Status:** Accepted

## Context

Phase 15 (prediction) loaded raw Phase 05 (1D IRT) ideal points directly via `_load_parquet_pair(irt_dir, "ideal_points")`, bypassing the canonical ideal point routing system established in ADR-0109. The canonical routing system selects 1D IRT or 2D Dim 1 per chamber based on horseshoe detection — correcting 7/14 Senate sessions (78th-83rd, 88th) and 2 House sessions (84th, 85th) where 1D IRT captures intra-Republican factionalism rather than the party divide.

Because Phase 15 bypassed this routing, `vote_features_{chamber}.parquet` contained confounded `xi_mean` values for horseshoe-affected chambers. When cross-session prediction transfer (Phase 26) trained XGBoost on these features and tested across sessions, the confounded ideology scores caused AUC < 0.5 (anti-prediction) in affected pairs.

The most dramatic case: 90th-vs-91st Senate had AUC = 0.217 (severely worse than random) because the 90th Senate's 1D IRT had party separation d = 1.27 while canonical routing via 2D Dim 1 achieves d = 8.39.

Cross-session phase 26 also had a secondary bug: `_load_vote_features()` and `_load_within_session_auc()` hardcoded the path `results_dir / "15_prediction" / "latest" / "data"`, which didn't match the run-grouped directory layout. All 13 pairs were silently skipping prediction transfer. Fixed to use `resolve_upstream_dir()`.

## Decision

1. **Phase 15 `load_ideal_points()` now prefers canonical ideal points.** A new `_load_canonical_irt()` helper checks Phase 06's `canonical_irt/canonical_ideal_points_{chamber}.parquet`. Falls back to raw Phase 05 if canonical routing hasn't run. The function signature adds optional `results_root` and `run_id` parameters (backward-compatible).

2. **Phase 26 path resolution fixed.** `_load_vote_features()` and `_load_within_session_auc()` use `resolve_upstream_dir("15_prediction", ...)` instead of hardcoded paths. Both accept an optional `run_id` parameter, passed through from `_run_cross_prediction()`.

## Consequences

**Before fix (raw 1D IRT for all chambers):**

| Pair | Senate AUC (Fwd/Bwd) |
|------|-----------------------|
| 87-vs-88 | 0.588 / 0.414 |
| 88-vs-89 | 0.931 / 0.900 |
| 89-vs-90 | 0.952 / 0.970 |
| 90-vs-91 | 0.217 / 0.246 |

**After fix (canonical routing):**

| Pair | Senate AUC (Fwd/Bwd) |
|------|-----------------------|
| 87-vs-88 | 0.417 / 0.379 |
| 88-vs-89 | 0.963 / 0.951 |
| 89-vs-90 | 0.955 / 0.970 |
| 90-vs-91 | 0.967 / 0.960 |

The 90-vs-91 Senate fix is the headline result: AUC 0.23 → 0.97.

The 88-vs-89 Senate pair also improved (0.92 → 0.96) because both sessions now use consistent 2D Dim 1 scores.

The 87-vs-88 Senate pair remains weak (0.42/0.38) — the 87th uses 1D IRT (no horseshoe) while the 88th uses 2D Dim 1, creating a genuine scale mismatch. The 83-vs-84 pair (both chambers, AUC ~0.3) reflects the 2010 Tea Party wave — genuine legislative discontinuity.

**Trade-off:** Phases 11 (network) and 12 (bipartite) also load raw Phase 05 IRT for annotation. Not changed because they use IRT for plot coloring, not quantitative features. The impact is cosmetic (wrong axis labels in horseshoe chambers) not analytical.
