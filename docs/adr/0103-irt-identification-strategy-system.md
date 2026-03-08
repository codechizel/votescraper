# ADR-0103: IRT Identification Strategy System

**Date:** 2026-03-07
**Status:** Accepted

## Context

The flat IRT model (Phase 05) previously used a single identification strategy: PCA-based hard anchors (ADR-0006, Decision 2). Post-hoc `validate_sign()` (ADR-0101) catches sign flips in supermajority chambers, but the fundamental problem remained — PCA anchors are distorted by the horseshoe effect when one party holds ≥70% of seats.

A literature review catalogued 24 identification strategies in the IRT literature. Seven are applicable to 1D flat IRT on roll call data. Different strategies are optimal for different chamber compositions (balanced vs. supermajority) and data availability (external scores vs. internal-only).

Rather than choosing a single "best" strategy, we implement all seven and auto-select based on chamber properties.

## Decision

### Seven Strategies

Implemented in `IdentificationStrategy` class with `build_irt_graph()` accepting a `strategy` parameter:

1. **anchor-pca** — Hard anchors via PCA PC1 party-aware extremes (Clinton-Jackman-Rivers 2004). Default for balanced chambers.
2. **anchor-agreement** — Hard anchors via cross-party contested vote agreement. Picks genuine ideological extremes rather than PCA artifacts. Default for supermajority chambers.
3. **sort-constraint** — `pm.Potential` ordering: D mean < R mean. No individual anchors. Fallback for supermajority chambers with insufficient contested votes.
4. **positive-beta** — `beta ~ HalfNormal(1)`. Eliminates reflection invariance but silences D-Yea bills (~12.5%). Manual override only.
5. **hierarchical-prior** — Party-informed priors: `xi ~ Normal(±0.5, 1)`. Soft identification via priors (Bafumi-Gelman-Park-Kaplan 2005). Manual override only.
6. **unconstrained** — No identification during MCMC; relies entirely on post-hoc sign correction. Diagnostic use only.
7. **external-prior** — Informative prior from Shor-McCarty scores: `xi ~ Normal(sm_score, 0.5)`. Auto-selected when Phase 17 data is available.

### Auto-Detection (`--identification auto`)

`select_identification_strategy()` auto-selects based on:

```
1. External scores available → external-prior
2. Supermajority (≥70%) + ≥10 contested votes → anchor-agreement
3. Supermajority + insufficient contested votes + both parties → sort-constraint
4. Balanced chamber → anchor-pca
```

### Strategy Rationale Table

Every IRT run prints a rationale table showing all seven strategies, which was selected (and why), and why each alternative was not selected. This appears in both console output and the HTML report.

### CLI

```bash
just irt --identification auto             # default
just irt --identification sort-constraint  # manual override
```

### Helper Functions

- `detect_supermajority(legislators, chamber)` — Returns `(is_super, fraction)` at 70% threshold.
- `select_identification_strategy(requested, legislators, matrix, chamber, external_scores_available)` — Returns `(strategy, rationale_dict)`.
- `compute_cross_party_agreement(matrix, legislators)` — Shared helper for both anchor selection and sign validation.
- `select_anchors()` — Updated: returns 5-tuple `(cons_idx, cons_slug, lib_idx, lib_slug, agreement_rates_or_None)`. Primary: agreement-based. Fallback: PCA.

## Consequences

**Benefits:**
- Correct identification for all chamber compositions (balanced, supermajority, single-party)
- Auto-detection removes manual tuning — the pipeline selects the best strategy per chamber
- Full transparency — every run documents why a strategy was chosen and what alternatives exist
- Sort-constraint strategy avoids anchoring on any individual legislator (no horseshoe risk)
- External-prior strategy provides national comparability when Shor-McCarty scores are available
- 28 new tests cover all strategies, auto-detection, and model construction

**Trade-offs:**
- Increased code complexity (7 strategy paths in `build_irt_graph()`)
- `positive-beta` and `unconstrained` are available but rarely appropriate — included for completeness
- Auto-detection heuristics may need tuning as more bienniums are analyzed

**Files changed:**
- `analysis/05_irt/irt.py` — `IdentificationStrategy`, `detect_supermajority()`, `select_identification_strategy()`, `compute_cross_party_agreement()`, updated `build_irt_graph()`, updated `select_anchors()`, `--identification` CLI flag
- `tests/test_irt.py` — 28 new tests (120 total, up from 92)
- `analysis/22_issue_irt/issue_irt.py` — Updated `select_anchors` unpacking (5-tuple)
- `analysis/experimental/irt_2d_experiment.py` — Updated `select_anchors` unpacking
- `analysis/experimental/irt_beta_experiment.py` — Updated `select_anchors` unpacking
- `docs/irt-identification-strategies.md` — Comprehensive article covering all strategies, literature, and CLI usage

**Related ADRs:** ADR-0006 (IRT implementation, Decision 2 expanded), ADR-0101 (party-aware anchors, now one of seven strategies), ADR-0047 (positive-beta trade-off, now a named strategy), ADR-0104 (robustness flags — runtime diagnostics for horseshoe detection, contested-only refit, 2D cross-reference).
