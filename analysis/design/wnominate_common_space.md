# W-NOMINATE Common Space Design

**Phase:** 30
**Script:** `analysis/30_wnominate_common_space/wnominate_common.py`
**Deep dive:** `docs/wnominate-common-space.md`
**ADR:** ADR-0120 (shared methodology with Phase 28)

## Assumptions

1. **W-NOMINATE Dim 1 is the input.** Per-session W-NOMINATE scores from Phase 16 (`wnominate_coords_{chamber}.parquet`). Dim 2 is diagnostic only — not linked cross-temporally.

2. **Same linking algorithm as Phase 28.** Pairwise chain affine alignment via bridge legislators, bootstrap uncertainty, delta-method propagation, quality gates. The code reuses Phase 28's data module functions directly — no reimplementation.

3. **91st Legislature is the reference.** Same as Phase 28. All other sessions are mapped onto the 91st's W-NOMINATE scale.

4. **W-NOMINATE SEs may be zero.** The R `wnominate` package's bootstrap SEs are often reported as 0 in our parquet exports. Use a minimum SE floor of 0.01 (matching Phase 28's `1e-6` clamp but larger due to W-NOMINATE's [-1, +1] bounded scale).

5. **Identity resolution uses OCD person IDs.** Same as Phase 28 — OpenStates OCD IDs as primary key, slug-based fallback for pre-2011 sessions, override tables for known errors.

6. **This phase is supplementary to Phase 28.** IRT common space remains the primary pipeline output. W-NOMINATE common space provides field-standard validation and cross-method robustness.

## Parameters & Constants

| Constant | Value | Justification |
|----------|-------|---------------|
| `REFERENCE_SESSION` | `91st_2025-2026` | Most recent, best data quality (shared with Phase 28) |
| `TRIM_PCT` | 10 | 10% trimmed regression for bridge pairs (shared with Phase 28) |
| `MIN_BRIDGES` | 5 | Minimum bridge legislators per session pair (shared with Phase 28) |
| `PARTY_D_MIN` | 1.5 | Minimum party separation (Cohen's d) for quality gate |
| `MIN_SE` | 0.01 | Floor for W-NOMINATE SEs (avoids zero-variance in meta-analysis) |
| `N_BOOTSTRAP` | 1000 | Bootstrap iterations for alignment uncertainty |

## Algorithm

### Step 1: Load W-NOMINATE Scores

For each biennium × chamber, load `wnominate_coords_{chamber}.parquet` from Phase 16 output. Extract `legislator_slug`, `wnom_dim1`, `wnom_se1`, `full_name`, `party`. Rename to match Phase 28 interface: `wnom_dim1` → `xi_mean`, `wnom_se1` → `xi_sd`.

Apply SE floor: `xi_sd = max(xi_sd, MIN_SE)`.

### Step 2: Build Global Roster

Reuse `build_global_roster()` from `common_space_data.py`. Identity resolution via `resolve_person_key()` (OCD IDs → slug fallback).

### Step 3: Pairwise Chain Linking

Reuse `solve_simultaneous_alignment()` — despite the function name, it implements pairwise chain linking (ADR-0120 revision).

For each adjacent session pair:
- Find bridge legislators (same `person_key` in both sessions)
- Estimate affine (A, B) via trimmed OLS: `xi_ref = A * xi_target + B`
- Chain transforms toward the 91st reference

### Step 4: Bootstrap Uncertainty

Reuse `bootstrap_alignment_direct()`. Resample bridge legislators with replacement, re-estimate pairwise links, propagate through chain. Produces Var(A_total), Var(B_total), Cov(A_total, B_total) per session.

### Step 5: Transform Scores

Reuse `transform_scores()`. Apply chained (A_total, B_total) to all W-NOMINATE scores. Combined SE via delta method: `Var(xi_common) = A²·Var(xi) + xi²·Var(A) + Var(B) + 2·xi·Cov(A,B)`.

### Step 6: Quality Gates

Reuse `compute_quality_gates()`. Check party separation (d > 1.5) and sign consistency (R > D) on the common W-NOMINATE scale.

### Step 7: Cross-Chamber Unification

Reuse `link_chambers()`. Affine transform mapping Senate common-space W-NOMINATE onto House scale via chamber-switcher bridges.

### Step 8: Career Scores

Reuse `compute_career_scores()` (per-chamber) and `compute_unified_career_scores()` (cross-chamber). DerSimonian-Laird random-effects meta-analysis.

### Step 9: Cross-Method Validation

**New to Phase 30.** Load Phase 28 IRT common-space career scores and correlate with Phase 30 W-NOMINATE career scores:
- Pearson r and Spearman ρ (overall and within-party)
- Rank agreement (do the same legislators appear as extreme/moderate?)
- Scatter plot: IRT career score vs W-NOMINATE career score
- Highlight divergent legislators (large rank difference between methods)

## Output

| File | Content |
|------|---------|
| `wnom_common_space_house.parquet` | House legislator × biennium on common W-NOMINATE scale |
| `wnom_common_space_senate.parquet` | Senate |
| `wnom_common_space_unified.parquet` | Cross-chamber unified scale |
| `wnom_career_scores_house.parquet` | Per-chamber career scores (RE meta-analysis) |
| `wnom_career_scores_senate.parquet` | Senate |
| `wnom_career_scores_unified.parquet` | One number per legislator across both chambers |
| `wnom_linking_coefficients.parquet` | A_t, B_t per session with bootstrap CIs |
| `wnom_bridge_coverage.parquet` | Pairwise bridge counts (should match Phase 28) |
| `wnom_validation.json` | Quality gates, cross-method correlations |
| `wnom_vs_irt_comparison.parquet` | Career score comparison (W-NOMINATE vs IRT) |

## Reports

Split into focused reports (same pattern as Phase 28):

| Report | Content |
|--------|---------|
| `wnominate_common_overview_report.html` | Key findings, bridge coverage, quality gates |
| `wnominate_common_house_report.html` | House scores, linking coefficients, career scores |
| `wnominate_common_senate_report.html` | Senate |
| `wnominate_common_unified_report.html` | Unified career scores |
| `wnominate_common_validation_report.html` | W-NOMINATE vs IRT: top 25 divergent legislators + full searchable comparison table |

## Implementation Plan

### Task 1: Directory and Data Loader

Create `analysis/30_wnominate_common_space/` with `__init__.py`. Write `wnominate_common.py` with:
- `load_all_wnominate_scores()` — loads Phase 16 W-NOMINATE parquets, renames columns to Phase 28 interface (`xi_mean`, `xi_sd`), applies SE floor
- `parse_args()` — CLI (same pattern as Phase 28)
- Justfile recipe: `wnominate-common-space`

### Task 2: Alignment Pipeline

Import and call Phase 28's `common_space_data` functions directly — no reimplementation:
- `build_global_roster()`, `compute_bridge_matrix()`, `solve_simultaneous_alignment()`, `bootstrap_alignment_direct()`, `transform_scores()`, `compute_quality_gates()`, `compute_polarization_trajectory()`
- `link_chambers()`, `compute_career_scores()`, `compute_unified_career_scores()`

### Task 3: Cross-Method Validation

Write `wnominate_common_data.py` with:
- `load_irt_career_scores()` — reads Phase 28 unified career scores
- `compute_cross_method_comparison()` — correlations, rank differences, divergent legislator detection
- Returns a DataFrame and summary dict for the report

### Task 4: Report Builder

Write `wnominate_common_report.py`:
- Reuse Phase 28's report section builders (adapted for W-NOMINATE column names)
- Add validation report section (IRT vs W-NOMINATE scatter, correlation table, divergent legislators)
- Split reports (overview, house, senate, unified, validation)

### Task 5: Integration

- Add to `just cross-pipeline` recipe
- Add Phase 30 entry to `analysis/design/README.md`
- Update CLAUDE.md architecture section
- Write ADR if design decisions diverge from Phase 28

### Task 6: Tests

- Test W-NOMINATE score loading (column rename, SE floor)
- Test that bridge counts match Phase 28 (same roster, same identity resolution)
- Test quality gates pass for all 28 chamber-sessions
- Test cross-method correlation is r > 0.80 (sanity check)

## Risks

1. **W-NOMINATE SE = 0 for all legislators in some sessions.** The SE floor handles this, but uncertainty estimates will be dominated by alignment uncertainty rather than per-legislator estimation uncertainty. Document this caveat.

2. **84th Legislature divergence.** IRT↔W-NOMINATE correlation is only r = 0.62 for the 84th House. The linking coefficient for this session will have wide CIs and may trigger a quality gate warning.

3. **Bounded scale compression.** W-NOMINATE scores near [-1, +1] boundary lose discrimination. After affine transformation to a common scale, this compression propagates — legislators who were "maximally extreme" in their per-session estimation remain compressed relative to IRT's unbounded scores.
